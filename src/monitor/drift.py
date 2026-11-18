"""Detect whether an incoming batch looks different from the training data.

The question this module answers is narrow: *has the shape of the input
changed?* It does not ask whether the model is still accurate - that needs
true labels and belongs to a later phase.

Two different failures are deliberately kept apart:

* **Data-quality failure** - the batch is malformed. A column is missing, a
  number arrived as text, a value is impossible. Phase 2's validator finds
  these, and a `DriftReport` carries its result separately. These are bugs in
  the pipeline feeding the model.
* **Distribution drift** - the batch is perfectly well-formed, but the values
  are distributed differently than the model was trained on. Nothing is
  broken; the world moved. That is what this module measures.

Conflating them would be a mistake, because the fixes are completely
different: one is an engineering bug, the other is a reason to retrain.

Why not Evidently: it is not installed, and adding it would pull in a large
dependency whose API has changed substantially between releases. The methods
needed here - a two-sample KS test, a chi-square test, and PSI - are a few
dozen lines on top of scipy, fully testable, and transparent about every
threshold. A library would hide exactly the parts a monitoring project should
be explicit about.

Nothing here imports or depends on Streamlit; the service returns plain
dataclasses and DataFrames, so any front end can use it.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from . import schema
from .validation import ValidationResult, validate_batch

# Feature-level and batch-level status values.
STABLE = "stable"
MODERATE = "moderate_drift"
DRIFT = "drift"
NOT_COMPUTED = "not_computed"

NUMERIC = "numeric"
CATEGORICAL = "categorical"


@dataclass(frozen=True)
class DriftThresholds:
    """Tunable decision boundaries, with the reasoning for each default.

    **None of these numbers is a law of nature.** They are conventions, and
    the right values depend on how much change a particular application can
    absorb. They are grouped here so they can be changed in one place and so
    that a report always records which values produced it.

    `psi_moderate` / `psi_major` (0.10 / 0.25)
        Population Stability Index bands that come from credit-scoring
        practice, where PSI has been used for decades: below 0.10 is normally
        treated as no meaningful change, 0.10-0.25 as worth watching, above
        0.25 as a material shift. Widely used, but a rule of thumb from one
        industry - not a universal constant.

    `p_value` (0.05)
        The conventional significance cut-off. Reported for completeness, but
        deliberately **not** what decides drift status - see
        `_describe_significance` for why.

    `n_bins` (10)
        How many bins PSI uses. Ten is the usual choice: enough resolution to
        see a shift, few enough that each bin holds a usable count.

    `min_batch_rows` (400)
        Below this, results are still computed but flagged as unreliable.
        This number was **measured, not guessed**. Unmodified holdout batches
        were scored against the reference, so any flag raised is by
        definition a false alarm; ten batches at each size gave:

            100 rows -> 10/10 falsely flagged, mean max PSI 0.36
            150 rows -> 10/10                            0.20
            200 rows ->  8/10                            0.13
            300 rows ->  2/10                            0.09
            400 rows ->  0/10                            0.06
            750 rows ->  0/10                            0.03

        The cause is simple: PSI uses ten bins, so a 100-row batch puts about
        ten rows in each, and ordinary sampling noise then moves the shares
        enough to cross the 0.10 band. 400 is where that stops happening for
        *this* dataset with *these* settings - a different number of bins,
        features, or a different dataset would move it. Treat it as a figure
        to re-measure, not a constant to copy.

    `min_expected_count` (5.0)
        Chi-square p-values assume every cell has a reasonable expected
        count. The usual guidance is at least 5. Rare categories in a small
        batch break this, so the result is flagged rather than quietly
        trusted.

    `missing_rate_tolerance` (0.05)
        A change in how often a field is missing is reported once it exceeds
        5 percentage points. Missingness is a data-quality signal as much as
        a distribution one, so it is called out explicitly.

    `min_category_share` (0.01)
        Reference categories holding less than 1% of rows are pooled into a
        single `__other__` group before PSI is computed. This is not
        cosmetic: `native_country` has 41 categories, most of them rare, and
        in a 100-row batch many receive zero or one observation. PSI takes a
        logarithm of the ratio of shares, so those near-empty groups dominate
        the sum and produce large values on data that has not changed at all.
        Measured on unmodified holdout batches, every one of eight 100-row
        batches was flagged before pooling. Pooling is the conventional fix
        and keeps PSI measuring the bulk of the distribution. A surge in a
        rare or brand-new category is still caught, because such rows land in
        `__other__` whose reference share is small, and because unseen
        categories are additionally reported by the chi-square test, the
        `unseen_categories` field, and validation.
    """

    psi_moderate: float = 0.10
    psi_major: float = 0.25
    p_value: float = 0.05
    n_bins: int = 10
    min_batch_rows: int = 400
    min_expected_count: float = 5.0
    missing_rate_tolerance: float = 0.05
    min_category_share: float = 0.01
    # Proportions of zero are replaced by this before taking a logarithm,
    # because ln(0) is undefined. Small enough not to distort a real result,
    # large enough to keep an empty bin from dominating the sum.
    psi_epsilon: float = 1e-4

    def __post_init__(self):
        if not 0 < self.psi_moderate < self.psi_major:
            raise ValueError(
                "Thresholds must satisfy 0 < psi_moderate < psi_major, got "
                f"{self.psi_moderate} and {self.psi_major}"
            )
        if not 0 < self.p_value < 1:
            raise ValueError(f"p_value must be between 0 and 1, got {self.p_value}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {self.n_bins}")


@dataclass
class FeatureDriftResult:
    """What was found for one feature."""

    feature: str
    feature_type: str
    method: str
    status: str
    psi: float | None = None
    statistic: float | None = None
    statistic_name: str | None = None
    p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None
    n_reference: int = 0
    n_batch: int = 0
    missing_rate_reference: float = 0.0
    missing_rate_batch: float = 0.0
    n_bins: int | None = None
    unseen_categories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def drifted(self) -> bool:
        return self.status in (DRIFT, MODERATE)

    @property
    def computed(self) -> bool:
        return self.status != NOT_COMPUTED

    def __str__(self) -> str:
        psi = f"{self.psi:.4f}" if self.psi is not None else "   n/a"
        p = f"{self.p_value:.2e}" if self.p_value is not None else "     n/a"
        return (
            f"{self.feature:16s} {self.status:14s} psi={psi} p={p} "
            f"n_ref={self.n_reference} n_batch={self.n_batch}"
        )


def population_stability_index(
    reference: pd.Series,
    batch: pd.Series,
    bin_edges: np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """Measure how far a batch's distribution has moved from the reference.

    PSI compares the share of rows falling in each bin:

        PSI = sum over bins of (ref_share - batch_share) * ln(ref_share / batch_share)

    It is zero when the two match and grows as they diverge. Unlike a
    p-value, it does not get larger merely because the sample is larger,
    which is what makes it the better basis for a decision.
    """
    reference_counts, _ = np.histogram(reference.dropna(), bins=bin_edges)
    batch_counts, _ = np.histogram(batch.dropna(), bins=bin_edges)

    reference_total = reference_counts.sum()
    batch_total = batch_counts.sum()
    if reference_total == 0 or batch_total == 0:
        raise ValueError("Cannot compute PSI: one of the samples is empty.")

    reference_share = np.clip(reference_counts / reference_total, epsilon, None)
    batch_share = np.clip(batch_counts / batch_total, epsilon, None)
    return float(np.sum((reference_share - batch_share) * np.log(reference_share / batch_share)))


OTHER_CATEGORY = "__other__"


def frequent_categories(reference: pd.Series, min_share: float) -> set:
    """Reference categories common enough for PSI to measure reliably.

    Everything rarer is pooled into `__other__`. See
    `DriftThresholds.min_category_share` for why this matters so much.
    """
    shares = reference.value_counts(normalize=True)
    return set(shares[shares >= min_share].index)


def _pool_rare(values: pd.Series, keep: set) -> pd.Series:
    """Replace any value outside `keep` with the `__other__` label."""
    return values.where(values.isin(keep), OTHER_CATEGORY)


def categorical_psi(
    reference: pd.Series,
    batch: pd.Series,
    epsilon: float = 1e-4,
    keep: set | None = None,
) -> float:
    """PSI over category shares, using the union of categories seen in either.

    When `keep` is given, categories outside it - rare reference categories
    and anything unseen - are pooled into `__other__` first.
    """
    if keep is not None:
        reference = _pool_rare(reference, keep)
        batch = _pool_rare(batch, keep)

    reference_share = reference.value_counts(normalize=True)
    batch_share = batch.value_counts(normalize=True)
    categories = reference_share.index.union(batch_share.index)

    reference_values = np.clip(reference_share.reindex(categories, fill_value=0.0).to_numpy(), epsilon, None)
    batch_values = np.clip(batch_share.reindex(categories, fill_value=0.0).to_numpy(), epsilon, None)
    return float(np.sum((reference_values - batch_values) * np.log(reference_values / batch_values)))


def quantile_bin_edges(
    reference: pd.Series, n_bins: int = 10
) -> np.ndarray | None:
    """Bin edges taken from the reference distribution, or None if impossible.

    Two details matter and are easy to get wrong.

    **The outer edges are infinite.** A drifted batch will contain values
    beyond anything in the reference. With finite outer edges those rows fall
    outside every bin and are dropped, which would hide the very shift being
    looked for.

    **A dominant single value gets its own boundary.** `capital_gain` is zero
    for about 92% of rows, so ten equal-frequency bins collapse into one -
    and a single bin always yields PSI = 0, reporting "no drift" whatever
    happens. When that collapse is detected, the dominant value is separated
    out and the remaining values are binned among themselves.
    """
    values = reference.dropna()
    if values.nunique() < 2:
        return None  # constant: there is no distribution to compare

    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))

    if len(edges) < 3:
        dominant = values.mode().iloc[0]
        remainder = values[values != dominant]
        if remainder.empty:
            return None
        pieces = [np.asarray([dominant], dtype=float)]
        if remainder.nunique() >= 2:
            pieces.append(np.quantile(remainder, np.linspace(0, 1, n_bins + 1)))
        else:
            pieces.append(np.asarray([float(remainder.iloc[0])], dtype=float))
        edges = np.unique(np.concatenate(pieces))

    if len(edges) < 2:
        return None
    interior = edges[1:-1] if len(edges) > 2 else edges[:1]
    return np.concatenate([[-np.inf], interior, [np.inf]])


def cramers_v(chi2_statistic: float, n_observations: int, table_shape: tuple) -> float:
    """Effect size for a chi-square test, scaled to roughly 0-1.

    The chi-square statistic itself grows with sample size, so it cannot say
    *how big* a difference is. Cramer's V divides that dependence out.
    """
    smaller_dimension = min(table_shape[0] - 1, table_shape[1] - 1)
    if smaller_dimension <= 0 or n_observations <= 0:
        return 0.0
    return float(np.sqrt(chi2_statistic / (n_observations * smaller_dimension)))


def _status_from_psi(psi: float, thresholds: DriftThresholds) -> str:
    if psi >= thresholds.psi_major:
        return DRIFT
    if psi >= thresholds.psi_moderate:
        return MODERATE
    return STABLE


def _describe_significance(
    p_value: float | None, psi: float, thresholds: DriftThresholds, n_reference: int
) -> list[str]:
    """Explain any disagreement between the p-value and the effect size.

    This is the heart of why status is decided by PSI rather than by p. A
    hypothesis test asks "could this difference be chance?" - and with tens
    of thousands of reference rows, the answer is almost always no, even for
    a difference far too small to matter. Reporting such a result as drift
    would bury the real signals under false alarms.
    """
    notes = []
    if p_value is None:
        return notes
    significant = p_value < thresholds.p_value
    material = psi >= thresholds.psi_moderate

    if significant and not material:
        notes.append(
            f"p={p_value:.2e} is statistically significant but PSI={psi:.4f} is "
            f"below {thresholds.psi_moderate}; with {n_reference:,} reference rows "
            "even negligible differences reach significance, so this is not "
            "treated as drift."
        )
    elif material and not significant:
        notes.append(
            f"PSI={psi:.4f} indicates a material shift although p={p_value:.2e} "
            "is not significant; this usually means the batch is small, so the "
            "effect size is the more trustworthy signal."
        )
    return notes


class DriftDetector:
    """Compares batches against a fixed reference distribution.

    The reference is captured once, at construction, and never updated from
    incoming batches. That is deliberate: a reference that absorbed each new
    batch would slowly follow the drift and stop detecting it - the
    statistical equivalent of refitting a scaler at inference time.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        thresholds: DriftThresholds | None = None,
    ):
        missing = [c for c in schema.FEATURE_COLUMNS if c not in reference.columns]
        if missing:
            raise ValueError(f"Reference data is missing feature columns: {missing}")
        if reference.empty:
            raise ValueError("Reference data is empty; cannot detect drift against it.")

        self.reference = reference[schema.FEATURE_COLUMNS].copy()
        self.thresholds = thresholds or DriftThresholds()
        # Bin edges depend only on the reference, so they are computed once
        # and reused for every batch. This also guarantees that two batches
        # are binned identically and are therefore comparable to each other.
        self._bin_edges = {
            column: quantile_bin_edges(self.reference[column], self.thresholds.n_bins)
            for column in schema.NUMERIC_COLUMNS
        }
        # Likewise fixed from the reference, so rare categories are pooled the
        # same way for every batch and two batches stay comparable.
        self._frequent_categories = {
            column: frequent_categories(
                self.reference[column].dropna(), self.thresholds.min_category_share
            )
            for column in schema.CATEGORICAL_COLUMNS
        }

    # --- per-feature checks -------------------------------------------------

    def _numeric_drift(self, column: str, batch: pd.DataFrame) -> FeatureDriftResult:
        reference_values = self.reference[column]
        batch_values = batch[column]
        result = FeatureDriftResult(
            feature=column,
            feature_type=NUMERIC,
            method="kolmogorov_smirnov + psi",
            status=NOT_COMPUTED,
            n_reference=int(reference_values.notna().sum()),
            n_batch=int(batch_values.notna().sum()),
            missing_rate_reference=round(float(reference_values.isna().mean()), 4),
            missing_rate_batch=round(float(batch_values.isna().mean()), 4),
            statistic_name="ks_statistic",
            effect_size_name="ks_statistic",
        )
        self._note_missing_change(result)

        reference_clean = reference_values.dropna()
        batch_clean = batch_values.dropna()

        if batch_clean.empty:
            result.notes.append(
                "Every value in this batch is missing, so no distribution can "
                "be compared. Reported as not computed rather than as stable."
            )
            return result
        if reference_clean.nunique() < 2:
            result.notes.append(
                "The reference values are constant, so there is no distribution "
                "to drift from. A change here would show up as a value-range "
                "validation issue instead."
            )
            return result

        edges = self._bin_edges[column]
        if edges is None:
            result.notes.append(
                "Bin edges could not be derived from the reference distribution, "
                "so PSI is unavailable."
            )
            return result

        ks = ks_2samp(reference_clean, batch_clean)
        result.statistic = float(ks.statistic)
        result.p_value = float(ks.pvalue)
        result.effect_size = float(ks.statistic)
        result.psi = population_stability_index(
            reference_clean, batch_clean, edges, self.thresholds.psi_epsilon
        )
        result.n_bins = len(edges) - 1
        result.status = _status_from_psi(result.psi, self.thresholds)
        result.notes.extend(
            _describe_significance(
                result.p_value, result.psi, self.thresholds, result.n_reference
            )
        )
        if batch_clean.nunique() == 1:
            result.notes.append(
                f"Every row in this batch has the same value "
                f"({batch_clean.iloc[0]}), which is itself worth investigating."
            )
        return result

    def _categorical_drift(self, column: str, batch: pd.DataFrame) -> FeatureDriftResult:
        reference_values = self.reference[column]
        batch_values = batch[column]
        result = FeatureDriftResult(
            feature=column,
            feature_type=CATEGORICAL,
            method="chi_square + psi",
            status=NOT_COMPUTED,
            n_reference=int(reference_values.notna().sum()),
            n_batch=int(batch_values.notna().sum()),
            missing_rate_reference=round(float(reference_values.isna().mean()), 4),
            missing_rate_batch=round(float(batch_values.isna().mean()), 4),
            statistic_name="chi2_statistic",
            effect_size_name="cramers_v",
        )
        self._note_missing_change(result)

        reference_clean = reference_values.dropna()
        batch_clean = batch_values.dropna()

        if batch_clean.empty:
            result.notes.append(
                "Every value in this batch is missing, so no distribution can "
                "be compared. Reported as not computed rather than as stable."
            )
            return result
        if reference_clean.nunique() < 1:
            result.notes.append("The reference has no observed categories.")
            return result

        # Category labels are compared as text. A batch that delivered numbers
        # where labels belong would otherwise make the category sets
        # unorderable and crash; validation reports the wrong type separately,
        # and drift is still worth computing on whatever arrived.
        if not pd.api.types.is_string_dtype(batch_clean) and not pd.api.types.is_object_dtype(
            batch_clean
        ):
            result.notes.append(
                f"This batch delivered dtype '{batch_values.dtype}' where text "
                "labels were expected. Values are compared as text so a result "
                "can still be produced; the type itself is reported as a "
                "data-quality error by validation."
            )
        reference_clean = reference_clean.astype(str)
        batch_clean = batch_clean.astype(str)

        known = set(reference_clean.unique())
        unseen = sorted(set(batch_clean.unique()) - known)
        result.unseen_categories = unseen
        if unseen:
            affected = int(batch_clean.isin(unseen).sum())
            result.notes.append(
                f"{affected} row(s) use categories absent from the reference: "
                f"{unseen}. The model encodes these as all-zeros, so they are "
                "also reported by validation as a data-quality finding."
            )

        keep = self._frequent_categories[column]
        n_pooled = reference_clean.nunique() - len(keep)
        if n_pooled > 0:
            result.notes.append(
                f"{n_pooled} reference category/categories below "
                f"{self.thresholds.min_category_share:.0%} of rows were pooled into "
                f"'{OTHER_CATEGORY}' for PSI, which would otherwise be dominated by "
                "near-empty groups. The chi-square test below uses all categories."
            )
        result.psi = categorical_psi(
            reference_clean, batch_clean, self.thresholds.psi_epsilon, keep=keep
        )
        result.n_bins = len(keep) + 1 if n_pooled > 0 else len(keep)
        result.status = _status_from_psi(result.psi, self.thresholds)

        # Chi-square on a 2 x categories table of counts.
        categories = sorted(known | set(batch_clean.unique()))
        table = np.vstack(
            [
                reference_clean.value_counts().reindex(categories, fill_value=0).to_numpy(),
                batch_clean.value_counts().reindex(categories, fill_value=0).to_numpy(),
            ]
        )
        # A category with zero count on both rows carries no information and
        # makes the test undefined, so such columns are dropped. This removes
        # empty categories, never a feature.
        table = table[:, table.sum(axis=0) > 0]

        if table.shape[1] < 2:
            result.notes.append(
                "Only one category is present across reference and batch, so a "
                "chi-square test is undefined. PSI is still reported."
            )
            return result

        try:
            chi2, p_value, _, expected = chi2_contingency(table)
        except ValueError as error:
            result.notes.append(f"Chi-square test could not be computed: {error}")
            return result

        result.statistic = float(chi2)
        result.p_value = float(p_value)
        result.effect_size = cramers_v(chi2, int(table.sum()), table.shape)

        smallest_expected = float(expected.min())
        if smallest_expected < self.thresholds.min_expected_count:
            result.notes.append(
                f"Smallest expected count is {smallest_expected:.2f}, below "
                f"{self.thresholds.min_expected_count}. Rare categories make the "
                "chi-square p-value unreliable here; PSI and Cramer's V do not "
                "depend on that assumption."
            )
        result.notes.extend(
            _describe_significance(
                result.p_value, result.psi, self.thresholds, result.n_reference
            )
        )
        return result

    def _note_missing_change(self, result: FeatureDriftResult) -> None:
        change = abs(result.missing_rate_batch - result.missing_rate_reference)
        if change > self.thresholds.missing_rate_tolerance:
            result.notes.append(
                f"Missing-value rate moved from "
                f"{result.missing_rate_reference:.2%} in the reference to "
                f"{result.missing_rate_batch:.2%} in this batch. A change in how "
                "often a field arrives empty is a data-quality signal, not only "
                "a distribution one."
            )

    # --- batch level --------------------------------------------------------

    def detect(
        self, batch: pd.DataFrame, batch_id: str | None = None
    ) -> "DriftReport":
        """Check every feature in a batch and summarise the result.

        Validation runs too, so the report can separate "this data is broken"
        from "this data is different". A batch that fails validation outright
        still gets whatever drift results can be computed, because refusing
        everything would hide information at the moment it is most needed.
        """
        missing = [c for c in schema.FEATURE_COLUMNS if c not in batch.columns]
        if missing:
            raise ValueError(
                f"Batch is missing feature columns {missing}, so it cannot be "
                "compared against the reference."
            )
        if batch.empty:
            raise ValueError("Cannot detect drift on an empty batch.")

        results = []
        for column in schema.NUMERIC_COLUMNS:
            results.append(self._numeric_drift(column, batch))
        for column in schema.CATEGORICAL_COLUMNS:
            results.append(self._categorical_drift(column, batch))

        report = DriftReport(
            batch_id=batch_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            feature_results=results,
            validation=validate_batch(batch),
            thresholds=self.thresholds,
            n_reference=len(self.reference),
            n_batch=len(batch),
        )
        if len(batch) < self.thresholds.min_batch_rows:
            report.notes.append(
                f"SMALL SAMPLE: this batch has {len(batch)} rows, below the "
                f"{self.thresholds.min_batch_rows} needed for PSI to be reliable "
                f"with {self.thresholds.n_bins} bins. On unmodified data, batches "
                "of 100 rows were falsely flagged every time. Any drift verdict "
                "here may be sampling noise - prefer a larger batch, or compare "
                "several consecutive batches before concluding anything."
            )
        return report


@dataclass
class DriftReport:
    """Feature-level results plus a batch-level verdict."""

    batch_id: str | None
    evaluated_at: str
    feature_results: list[FeatureDriftResult]
    validation: ValidationResult
    thresholds: DriftThresholds
    n_reference: int
    n_batch: int
    notes: list[str] = field(default_factory=list)

    @property
    def drifted_features(self) -> list[FeatureDriftResult]:
        return [r for r in self.feature_results if r.status == DRIFT]

    @property
    def moderate_features(self) -> list[FeatureDriftResult]:
        return [r for r in self.feature_results if r.status == MODERATE]

    @property
    def not_computed_features(self) -> list[FeatureDriftResult]:
        return [r for r in self.feature_results if r.status == NOT_COMPUTED]

    @property
    def computed_features(self) -> list[FeatureDriftResult]:
        return [r for r in self.feature_results if r.computed]

    @property
    def share_drifted(self) -> float:
        """Fraction of the features that could be checked which show drift."""
        if not self.computed_features:
            return 0.0
        flagged = len(self.drifted_features) + len(self.moderate_features)
        return round(flagged / len(self.computed_features), 4)

    @property
    def max_psi(self) -> float | None:
        values = [r.psi for r in self.feature_results if r.psi is not None]
        return round(max(values), 4) if values else None

    @property
    def status(self) -> str:
        """Batch verdict.

        A single feature above `psi_major` is enough to call the batch
        drifted: one badly shifted input can be sufficient to change the
        model's behaviour, so the loudest feature decides rather than an
        average that would dilute it.
        """
        if self.drifted_features:
            return DRIFT
        if self.moderate_features:
            return MODERATE
        if not self.computed_features:
            return NOT_COMPUTED
        return STABLE

    @property
    def has_data_quality_errors(self) -> bool:
        """True when the batch is malformed, as opposed to merely different."""
        return not self.validation.is_valid

    def to_frame(self) -> pd.DataFrame:
        """One row per feature, sorted with the largest PSI first."""
        rows = [
            {
                "feature": r.feature,
                "type": r.feature_type,
                "method": r.method,
                "status": r.status,
                "psi": r.psi,
                "statistic": r.statistic,
                "statistic_name": r.statistic_name,
                "p_value": r.p_value,
                "effect_size": r.effect_size,
                "effect_size_name": r.effect_size_name,
                "n_reference": r.n_reference,
                "n_batch": r.n_batch,
                "missing_rate_batch": r.missing_rate_batch,
                "n_bins": r.n_bins,
                "unseen_categories": ", ".join(r.unseen_categories),
                "n_notes": len(r.notes),
            }
            for r in self.feature_results
        ]
        frame = pd.DataFrame(rows)
        return frame.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)

    def to_dict(self) -> dict:
        """Plain dictionary, ready for JSON. No front-end framework involved."""
        return {
            "batch_id": self.batch_id,
            "evaluated_at": self.evaluated_at,
            "status": self.status,
            "n_reference": self.n_reference,
            "n_batch": self.n_batch,
            "share_drifted": self.share_drifted,
            "max_psi": self.max_psi,
            "thresholds": asdict(self.thresholds),
            "data_quality": {
                "is_valid": self.validation.is_valid,
                "n_errors": len(self.validation.errors),
                "n_warnings": len(self.validation.warnings),
                "issues": [str(issue) for issue in self.validation.issues],
            },
            "features": [asdict(r) for r in self.feature_results],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [
            f"Drift report for batch {self.batch_id or '(unnamed)'}",
            f"  evaluated at   {self.evaluated_at}",
            f"  reference      {self.n_reference:,} rows",
            f"  batch          {self.n_batch:,} rows",
            f"  verdict        {self.status.upper()}",
            f"  drifted        {len(self.drifted_features)} features above "
            f"PSI {self.thresholds.psi_major}",
            f"  moderate       {len(self.moderate_features)} features above "
            f"PSI {self.thresholds.psi_moderate}",
            f"  not computed   {len(self.not_computed_features)} features",
            f"  max PSI        {self.max_psi}",
            "",
            "  Data quality (separate from drift): "
            f"{'passed' if self.validation.is_valid else 'FAILED'} "
            f"({len(self.validation.errors)} errors, "
            f"{len(self.validation.warnings)} warnings)",
        ]
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)

    def explain(self, only_flagged: bool = True) -> str:
        """Readable per-feature detail, including why a result was not computed."""
        lines = []
        for result in sorted(
            self.feature_results,
            key=lambda r: (-1 if r.psi is None else -r.psi),
        ):
            if only_flagged and result.status == STABLE and not result.notes:
                continue
            lines.append(f"  {result}")
            for note in result.notes:
                lines.append(f"      - {note}")
        return "\n".join(lines) if lines else "  (nothing to report)"
