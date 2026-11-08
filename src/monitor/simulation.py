"""Produce batches of "incoming" records to feed the prediction service.

The records come from the held-out 20% of the dataset - the same split the
model was never trained on, rebuilt with the same seed so it is guaranteed to
contain no training rows. Each batch is therefore genuinely unseen data,
which is what makes the simulation meaningful.

Three deliberate choices:

**Sampling is random, not stratified.** A random draw keeps the natural
76/24 class split in expectation while letting individual batches fluctuate,
exactly as real arrivals would. Forcing every batch to an exact 76/24 would
erase that variation, and a monitor needs to see normal fluctuation to tell it
apart from a real change. Nothing here balances the classes.

**Batches are disjoint.** The pool is shuffled once, then handed out in
consecutive slices, so a record arrives at most once. Re-sampling the same
rows would create duplicates that later look like a distribution change.

**Labels travel beside the features, never inside them.** A `SimulatedBatch`
exposes `features` (no income column) and `labels` separately. The prediction
service is only ever given `features`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from . import config, schema
from .prediction import make_batch_id, utc_now
from .preprocessing import split_train_test

# A modifier takes a feature frame and returns a changed copy. This is the
# hook later phases use to introduce controlled drift; nothing in this module
# detects or measures it.
Modifier = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class SimulatedBatch:
    """One batch of incoming records, with its labels held separately."""

    batch_id: str
    created_at: str
    features: pd.DataFrame
    labels: pd.Series
    seed: int
    batch_index: int
    modifications: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.features)

    def actual_class_distribution(self) -> dict:
        """The true label mix. Available for later scoring, never for predicting."""
        counts = self.labels.value_counts()
        total = int(counts.sum())
        return {
            label: {
                "count": int(count),
                "percent": round(float(count) / total * 100, 2),
            }
            for label, count in counts.items()
        }

    def summary(self) -> str:
        parts = ", ".join(
            f"{label} {stats['count']} ({stats['percent']}%)"
            for label, stats in self.actual_class_distribution().items()
        )
        lines = [
            f"Batch {self.batch_id} (index {self.batch_index}, seed {self.seed})",
            f"  rows          {self.n_rows}",
            f"  actual labels {parts}",
        ]
        if self.modifications:
            for change in self.modifications:
                lines.append(f"  modified      {change}")
        return "\n".join(lines)


def split_reference_and_holdout(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the exact train/test split used to fit the model.

    Returns `(reference, holdout)`:

    * **reference** - the 80% the model was trained on. Later phases compare
      incoming batches against this to decide whether the input has shifted,
      because this is the distribution the model actually learned.
    * **holdout** - the untouched 20%, used here as the source of incoming
      records.

    The split is recreated with the same seed and `test_size` as training, so
    no row can appear in both. Reusing `split_train_test` keeps a single
    definition rather than a second, possibly diverging, copy.
    """
    return split_train_test(df, test_size=config.TEST_SIZE)


def get_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature columns of the training split, for later drift comparison."""
    reference, _ = split_reference_and_holdout(df)
    return reference[schema.FEATURE_COLUMNS]


class BatchSimulator:
    """Hands out successive batches of unseen records.

    The pool is shuffled once using `seed`, so the whole sequence of batches
    is reproducible: the same seed and batch size always produce the same
    batches in the same order.
    """

    def __init__(
        self,
        holdout: pd.DataFrame,
        batch_size: int = 500,
        seed: int = config.RANDOM_SEED,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if len(holdout) == 0:
            raise ValueError("Cannot simulate batches from an empty holdout set.")
        if batch_size > len(holdout):
            raise ValueError(
                f"batch_size {batch_size} exceeds the {len(holdout)} available "
                "holdout rows."
            )
        missing = [c for c in schema.FEATURE_COLUMNS if c not in holdout.columns]
        if missing:
            raise ValueError(f"Holdout data is missing feature columns: {missing}")
        if config.TARGET_COLUMN not in holdout.columns:
            raise ValueError(
                f"Holdout data must include the label column "
                f"'{config.TARGET_COLUMN}' so that labels can be kept aside."
            )

        self.batch_size = batch_size
        self.seed = seed
        # frac=1 shuffles every row; the fixed seed makes the order repeatable.
        self._pool = holdout.sample(frac=1, random_state=seed)
        self._cursor = 0
        self._batches_drawn = 0

    @property
    def rows_remaining(self) -> int:
        return len(self._pool) - self._cursor

    @property
    def batches_remaining(self) -> int:
        return self.rows_remaining // self.batch_size

    def reset(self) -> None:
        """Rewind to the start, so the same sequence can be replayed."""
        self._cursor = 0
        self._batches_drawn = 0

    def next_batch(
        self,
        size: int | None = None,
        modifier: Modifier | None = None,
        modification_label: str | None = None,
        batch_id: str | None = None,
    ) -> SimulatedBatch:
        """Take the next records from the pool as one batch.

        `modifier` optionally alters the feature values before they are
        returned - the extension point for introducing controlled drift in a
        later phase. It is applied to the features only, so the labels and
        the class balance are never touched.
        """
        size = size or self.batch_size
        if size < 1:
            raise ValueError(f"Batch size must be at least 1, got {size}")
        if size > self.rows_remaining:
            raise ValueError(
                f"Only {self.rows_remaining} rows remain in the holdout pool, "
                f"but {size} were requested. Call reset() to start over, or "
                "use a smaller batch size."
            )

        rows = self._pool.iloc[self._cursor : self._cursor + size]
        self._cursor += size
        self._batches_drawn += 1

        features = rows[schema.FEATURE_COLUMNS].copy()
        labels = rows[config.TARGET_COLUMN].copy()

        modifications = []
        if modifier is not None:
            features = modifier(features)
            modifications.append(modification_label or "custom modifier applied")
            missing = [c for c in schema.FEATURE_COLUMNS if c not in features.columns]
            if missing:
                raise ValueError(
                    f"Modifier dropped required feature columns: {missing}"
                )
            if len(features) != len(labels):
                raise ValueError(
                    "Modifier changed the number of rows, so features and "
                    f"labels no longer match ({len(features)} vs {len(labels)})."
                )

        return SimulatedBatch(
            batch_id=batch_id or make_batch_id(),
            created_at=utc_now(),
            features=features,
            labels=labels,
            seed=self.seed,
            batch_index=self._batches_drawn,
            modifications=modifications,
        )

    def generate_batches(self, n_batches: int, **kwargs) -> list[SimulatedBatch]:
        """Draw several batches in a row."""
        if n_batches > self.batches_remaining:
            raise ValueError(
                f"Requested {n_batches} batches of {self.batch_size} rows, but "
                f"only {self.batches_remaining} fit in the remaining "
                f"{self.rows_remaining} rows."
            )
        return [self.next_batch(**kwargs) for _ in range(n_batches)]


# --- modifiers, for later drift experiments ---------------------------------
# These build the `modifier` callables above. They exist so a later phase can
# create a batch with a known, deliberate change and check that drift
# detection notices it. Nothing here measures anything.


def shift_numeric(column: str, shift: float) -> Modifier:
    """Add a constant to a numeric column, e.g. make everyone 10 years older."""
    if column not in schema.NUMERIC_COLUMNS:
        raise ValueError(
            f"'{column}' is not a numeric feature. Choose from: "
            f"{schema.NUMERIC_COLUMNS}"
        )

    def apply(features: pd.DataFrame) -> pd.DataFrame:
        changed = features.copy()
        changed[column] = changed[column] + shift
        return changed

    return apply


def scale_numeric(column: str, factor: float) -> Modifier:
    """Multiply a numeric column, for a proportional rather than flat change."""
    if column not in schema.NUMERIC_COLUMNS:
        raise ValueError(
            f"'{column}' is not a numeric feature. Choose from: "
            f"{schema.NUMERIC_COLUMNS}"
        )

    def apply(features: pd.DataFrame) -> pd.DataFrame:
        changed = features.copy()
        changed[column] = changed[column] * factor
        return changed

    return apply


def overwrite_category(column: str, value: str, fraction: float = 1.0) -> Modifier:
    """Set a categorical column to `value` for a fraction of the rows.

    Passing a value the model has never seen is how to simulate an unknown
    category arriving in production.
    """
    if column not in schema.CATEGORICAL_COLUMNS:
        raise ValueError(
            f"'{column}' is not a categorical feature. Choose from: "
            f"{schema.CATEGORICAL_COLUMNS}"
        )
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    def apply(features: pd.DataFrame) -> pd.DataFrame:
        changed = features.copy()
        n_rows = max(1, int(round(len(changed) * fraction)))
        # Takes the first n rows; the pool was already shuffled, so these are
        # an arbitrary subset rather than a biased one.
        changed.iloc[:n_rows, changed.columns.get_loc(column)] = value
        return changed

    return apply


def combine(*modifiers: Modifier) -> Modifier:
    """Apply several modifiers in order, for multi-feature drift."""

    def apply(features: pd.DataFrame) -> pd.DataFrame:
        for modifier in modifiers:
            features = modifier(features)
        return features

    return apply
