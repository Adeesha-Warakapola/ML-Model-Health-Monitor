# Data Drift Detection

How the project decides whether an incoming batch looks different from the
data the model was trained on — and, just as importantly, how it avoids
crying wolf.

All numbers below were measured by running the code, not estimated.

## The question being asked

Drift detection answers one narrow question: **has the shape of the input
changed?** It does not ask whether the model is still accurate — that needs
true labels and belongs to a later phase.

### Two failures that must not be confused

| | Data-quality failure | Distribution drift |
| --- | --- | --- |
| What happened | The batch is malformed | The batch is well-formed but different |
| Example | `age` arrived as text, or is `-5` | `age` is genuinely 15 years higher |
| Found by | Phase 2 validation | This module |
| What it means | A bug in the pipeline | The world moved |
| What to do | Fix the code | Consider retraining |

A `DriftReport` carries both, kept separate. Conflating them would be a
serious mistake, because the fixes have nothing in common.

**Demonstrated** (`detect_drift.py`, scenario 5) — a batch with 50 impossible
ages (`-5`) and 30 extra missing `workclass` values:

```
verdict        STABLE
max PSI        0.0763

Data-quality issues found (reported separately from drift):
  WARNING [workclass] 12.00% of values are missing and will be imputed (60 rows)
  WARNING [age] Values below the minimum of 17 (50 rows)
```

The distribution barely moved, so drift says *stable* — correctly. Validation
catches the real problem. Had only one of the two checks existed, this batch
would have looked fine.

## The reference distribution

Every batch is compared against a **fixed** reference: the feature columns of
the 80% training split, rebuilt with the same seed (`get_reference_features`
from Phase 4). It is captured once when the detector is constructed and is
**never updated from incoming batches**.

That is deliberate. A reference that absorbed each new batch would slowly
follow the drift and stop detecting it — the statistical equivalent of
refitting a scaler at inference time. `test_detector_does_not_learn_from_batches`
asserts that scoring a drifted batch does not change later verdicts.

Bin edges and category groupings are also derived once from the reference, so
any two batches are binned identically and remain comparable to each other.

## The methods

### Numerical features: KS test + PSI

**Kolmogorov–Smirnov test.** Line up both samples as cumulative
distributions and measure the largest vertical gap between them. That gap is
the KS statistic (0 to 1) — it is itself an effect size. The test also gives a
p-value.

**Population Stability Index (PSI).** Split the reference into bins, then
compare what share of rows falls in each bin:

```
PSI = Σ (ref_share − batch_share) × ln(ref_share / batch_share)
```

Zero when the two match, growing as they diverge. **PSI decides the status**
in this project; the p-value is reported alongside as supporting evidence.

### Categorical features: Chi-square + PSI + Cramér's V

**Chi-square test** on a 2 × categories table of counts, asking whether the
two rows could plausibly come from the same distribution.

**Cramér's V** is the effect size. The chi-square statistic grows with sample
size, so it cannot say *how big* a difference is; Cramér's V divides that out.

**PSI** over category shares, same formula as above.

## Why p-values are not trusted alone

This is the single most important idea in the module.

A p-value answers "could this difference be chance?" With **39,073 reference
rows**, the answer is almost always no — even for a difference far too small
to matter. Deciding drift on p alone would bury real signals under false
alarms.

**Demonstrated.** Take the entire reference and add exactly 1 year to every
age — a trivially small shift:

```
p-value = 1.52e-13   (overwhelmingly "significant")
PSI     = 0.0081     (far below the 0.10 band)
status  = stable
note    = "p=1.52e-13 is statistically significant but PSI=0.0081 is below
           0.1; with 39,073 reference rows even negligible differences reach
           significance, so this is not treated as drift."
```

A p-value of 1.5 × 10⁻¹³ sounds like a five-alarm fire. The effect size says
the distribution moved by almost nothing. The effect size is right.

The report explains the disagreement rather than hiding it. The reverse case
is handled too: when PSI is material but p is not significant (usually a small
batch), the note says the effect size is the more trustworthy signal.

### PSI does not inflate with sample size

That is precisely why it is the decision basis. Verified in
`test_psi_does_not_grow_merely_because_the_batch_is_larger`: comparing a
reference against fresh samples from the *same* distribution gives PSI < 0.05
whether the batch has 200 rows or 20,000. A p-value does the opposite.

## Thresholds, and where they came from

Configurable via `DriftThresholds`, recorded in every report, and **none of
them is a law of nature**.

| Setting | Default | Where it comes from |
| --- | --- | --- |
| `psi_moderate` | 0.10 | Credit-scoring convention: below 0.10 is normally treated as no meaningful change |
| `psi_major` | 0.25 | Same convention: above 0.25 is a material shift |
| `p_value` | 0.05 | The usual significance cut-off. Reported, not decisive |
| `n_bins` | 10 | Enough resolution to see a shift, few enough for usable counts per bin |
| `min_batch_rows` | 400 | **Measured** — see below |
| `min_expected_count` | 5.0 | Standard guidance for chi-square validity |
| `missing_rate_tolerance` | 0.05 | A missingness change above 5 percentage points is called out |
| `min_category_share` | 0.01 | **Measured** — see below |

The PSI bands are a rule of thumb from one industry, widely used but not a
universal constant. The right value depends on how much change a particular
application can absorb.

## Two problems found by measurement

Both of these were discovered by running the code against **unmodified**
holdout data, where any flag raised is by definition a false alarm.

### Problem 1: zero-inflated features silently reported "no drift"

`capital_gain` is zero for about 92% of rows. Ten equal-frequency bins
therefore collapse into **one**:

```
age              10 quantile edges -> 11 unique
capital_gain     10 quantile edges ->  2 unique   (one bin)
capital_loss     10 quantile edges ->  2 unique   (one bin)
```

A single bin always yields PSI = 0 — it would have reported "no drift" no
matter what happened to the feature. A silent false negative, the worst kind.

**Fix:** when that collapse is detected, the dominant value is given its own
boundary and the remaining values are binned among themselves. `capital_gain`
now gets 10 bins, and a ×3 change to its non-zero values produces PSI 0.195
where the naive version produced 0.000.

A related detail: **outer bin edges are infinite** (`-inf`, `+inf`). With
finite edges, a drifted batch's out-of-range rows would fall outside every bin
and be dropped — hiding the very shift being looked for.

### Problem 2: rare categories caused constant false alarms

`native_country` has 41 categories, ~91% of them `United-States`. In a
500-row batch most categories receive zero or one observation, and PSI takes a
logarithm of the ratio of shares, so those near-empty groups dominated the
sum.

Reference categories below `min_category_share` (1% of rows) are now pooled
into a single `__other__` group before PSI is computed — the conventional fix.
Effect on an unmodified batch:

| Feature | PSI before pooling | PSI after |
| --- | --- | --- |
| `native_country` | 0.0966 (nearly flagged) | **0.0009** |

Nothing is lost: a surge in a rare or brand-new category still lands in
`__other__`, whose reference share is small, and unseen categories are
*additionally* reported by the chi-square test, the `unseen_categories` field,
and validation. The chi-square test itself uses all categories.

### Why `min_batch_rows` is 400, not 50

My first guess was 50. Measuring showed that was badly wrong. Ten unmodified
holdout batches at each size:

| Batch size | Mean max PSI | Falsely flagged |
| --- | --- | --- |
| 100 | 0.3565 | **10 / 10** |
| 200 | 0.1330 | 8 / 10 |
| 300 | 0.0891 | 2 / 10 |
| 400 | 0.0628 | 0 / 10 |
| 750 | 0.0306 | 0 / 10 |

PSI uses ten bins, so a 100-row batch puts ~10 rows in each, and ordinary
sampling noise moves the shares enough to cross the 0.10 band. **At 100 rows
the detector flagged every single unmodified batch as drifted.**

400 is where that stops for *this* dataset with *these* settings. A different
number of bins, features, or dataset would move it — re-measure rather than
copy the number. Below the floor, results are still computed and returned, but
the report carries a prominent `SMALL SAMPLE` note. Nothing is hidden.

Reproduce with `python detect_drift.py --small-sample-demo`.

## Edge cases, and the no-silent-removal rule

**A feature is never dropped from a report.** When a statistic cannot be
computed, the feature appears with status `not_computed` and a note saying
why. `test_every_feature_appears_in_the_report` and
`test_features_that_cannot_be_computed_are_reported_with_a_reason` enforce it.

| Situation | Behaviour |
| --- | --- |
| Reference feature is constant | `not_computed` — there is no distribution to drift from |
| Every batch value missing | `not_computed`, with the reason stated |
| Some values missing | NaNs excluded from the statistics but **counted and reported**; a missingness change above tolerance is noted |
| Batch feature is constant | Computed, plus a note that every row shares one value |
| Category never seen in training | Listed in `unseen_categories`, noted, and flagged by validation |
| Only one category present overall | PSI reported; chi-square is undefined and said to be so |
| Chi-square expected count < 5 | Computed, with a note that the p-value is unreliable while PSI and Cramér's V are not |
| Wrong dtype (numbers where labels belong) | Compared as text so a result still appears; the type error is reported by validation |
| Batch smaller than `min_batch_rows` | Computed, with a prominent small-sample warning |
| Empty batch / missing column / empty reference | Raises `ValueError` with a clear message |
| Empty PSI bin | `ln(0)` avoided by clipping shares to `psi_epsilon` (1e-4) |

## Results: feature level and batch level

Each feature yields a `FeatureDriftResult` with the feature name, type,
method, status, PSI, test statistic, p-value, effect size, both sample sizes,
missing rates, bin count, unseen categories, and any notes.

Batch level: `status`, `share_drifted`, `max_psi`, counts of drifted /
moderate / not-computed features, and the separate validation result.

**A single feature above `psi_major` makes the batch drifted.** One badly
shifted input can be enough to change the model's behaviour, so the loudest
feature decides rather than an average that would dilute it.

## Measured results

### Stable batch (500 unmodified holdout rows)

```
verdict        STABLE
drifted        0 features above PSI 0.25
moderate       0 features above PSI 0.1
not computed   0 features
max PSI        0.0613        (education)
```

### Deliberately drifted batches

| Scenario | Verdict | Loudest feature | PSI |
| --- | --- | --- | --- |
| `age + 15` | DRIFT | `age` | 2.6811 |
| `hours_per_week × 1.5` | DRIFT | `hours_per_week` | 2.5850 |
| 40% of `occupation` → unseen value | DRIFT | `occupation` | 2.0960 |
| 30% of `sex` → one value | MODERATE | `sex` | 0.1960 |
| `age +10`, `hours ×1.4`, 60% `education` | DRIFT | multiple | — |

In every case **only the deliberately modified features were flagged** —
`test_only_the_changed_feature_is_flagged` asserts this.

### Label balance is never touched

Drift scenarios modify features only. The natural 76/24 class split is
preserved in every batch, verified by
`test_simulator_batches_keep_their_natural_class_balance`: a plain batch and a
drifted batch have byte-identical label distributions.

## Usable without any UI

The service returns plain dataclasses and DataFrames and imports no front-end
library (`test_drift_module_does_not_depend_on_streamlit` asserts Streamlit is
never imported).

```python
from monitor.data import load_adult_data
from monitor.drift import DriftDetector, DriftThresholds
from monitor.simulation import (
    BatchSimulator, get_reference_features, split_reference_and_holdout,
)

df = load_adult_data()
reference = get_reference_features(df)          # fixed training distribution
_, holdout = split_reference_and_holdout(df)

detector = DriftDetector(reference)             # or pass custom DriftThresholds
batch = BatchSimulator(holdout, batch_size=500, seed=42).next_batch()

report = detector.detect(batch.features, batch_id=batch.batch_id)

print(report.status)            # "stable" | "moderate_drift" | "drift"
print(report.summary())         # human-readable
report.to_frame()               # one row per feature, sorted by PSI
report.to_dict()                # JSON-ready, for storage or an API
report.validation               # data quality, kept separate
```

## Why not Evidently

Evidently was considered and rejected, which the phase brief allows.

- It is **not installed**, so adding it means a large new dependency.
- Its API has **changed substantially** between releases, making it a
  maintenance risk for a project meant to stay runnable.
- The three methods needed — KS, chi-square, PSI — are a few dozen lines on
  top of scipy, fully testable, and transparent about every threshold.
- A library would have **hidden** the two problems documented above. The
  zero-inflated binning collapse and the rare-category false alarms were only
  found because the binning was visible and measurable.

For a project whose subject *is* monitoring, being explicit about the
thresholds beats delegating them.

## Limitations

- **Univariate only.** Each feature is checked on its own, so a change in the
  *relationship* between two features — each still individually normal — would
  not be detected.
- **PSI bands are conventions**, inherited from credit scoring. They are not
  validated for this dataset or any particular application.
- **`min_batch_rows` is dataset-specific.** 400 was measured here; it is not
  transferable.
- **Rare-category pooling trades sensitivity for stability.** A small shift
  confined to a rare category will be muted in PSI. The chi-square test and
  the unseen-category report are the compensating signals.
- **No performance monitoring.** Drift in the inputs does not necessarily mean
  the model got worse, and a model can degrade with no input drift at all.
  Measuring that needs labels and is not implemented yet.
- **No alert persistence or history.** Each report is independent; nothing is
  stored.
