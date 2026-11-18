# Prediction Distribution and Model Performance Monitoring

Three different questions, three different modules, and a demonstration that
they genuinely disagree on the same batch. All numbers below were produced by
running `monitor_performance.py`, not estimated.

## Three questions that are not the same question

| | Question | Needs true labels? | Module |
| --- | --- | --- | --- |
| **Data drift** | Has the input changed? | No | `drift.py` (Phase 5) |
| **Prediction drift** | Has what the model *says* changed? | No | `prediction_monitoring.py` (this phase) |
| **Performance degradation** | Is the model still *right*? | **Yes** | `performance_monitoring.py` (this phase) |

Data drift and prediction drift are both available the instant a batch
arrives. Performance is not - it requires ground truth that, in a real
system, shows up later, for some rows, or never.

### Measured: the three numbers move independently

Take 600 unmodified holdout rows, then add 25 years to every `age` while
leaving the true labels completely untouched:

| Measurement | Stable batch | Age +25 batch |
| --- | --- | --- |
| **Data drift** - age PSI | 0.0373 (stable) | **5.1124** (drift) |
| **Prediction drift** - label PSI | 0.0001 (stable) | 0.0324 (still stable) |
| **Prediction drift** - confidence PSI | 0.0112 (stable) | **0.1181 (moderate)** |
| **Performance** - accuracy | 0.8317 | **0.7683** (degraded) |
| **Performance** - precision | 0.5691 | **0.4752** (degraded) |
| **Performance** - ROC-AUC | 0.9094 | 0.9094 (barely moved) |

Four things worth noticing:

1. **Data drift is severe** (PSI > 5) - the input genuinely changed a lot.
2. **The predicted-*label* mix barely moved** (PSI 0.03, still "stable"):
   the hard 0.29 threshold absorbed much of the shift.
3. **The predicted *probabilities* moved more** (PSI 0.12, "moderate"): the
   underlying scores shifted before the thresholded labels caught up. This is
   exactly why confidence is tracked separately from the label mix - it is an
   earlier warning.
4. **Performance genuinely degraded** - accuracy down 5.7 points, precision
   down 12.8 points - while **ROC-AUC barely moved** (+0.0036). The model's
   ability to *rank* people correctly is intact; the fixed decision threshold
   is now miscalibrated for this shifted population. Data drift explains
   *why*; only the labelled performance numbers prove *that* it happened.

None of the three can substitute for the others. A dashboard that only
watched data drift would have seen "severe," correctly suspected trouble, but
could not have confirmed real accuracy loss without labels. A dashboard that
only watched predicted-label proportions would have under-reacted here,
because the label mix moved far less than the truth did.

## Part A: Prediction monitoring (no labels needed)

### The reference prediction distribution

"The original reference prediction distribution" is built by running the
saved model, through the ordinary `PredictionService`, on the Phase 4
reference features (the training split). This is not invented - it is
measured from the model's own behaviour on data already in scope, using
exactly the same code path as every other batch. See
`PredictionMonitor.from_reference_features`.

### Reusing `drift.py` rather than re-implementing it

A predicted label is just another categorical column; a predicted probability
is just another numeric column. `prediction_monitoring.py` imports
`categorical_psi`, `population_stability_index`, `quantile_bin_edges`,
`frequent_categories`, `cramers_v`, `_status_from_psi` and
`_describe_significance` directly from `drift.py`. That means every edge case
Phase 5 found and fixed - the zero-inflation-safe binning, the infinite outer
bin edges, rare-category pooling, the p-value-vs-effect-size explanation - is
inherited for free, rather than re-discovered.

### Label drift vs. confidence drift

Two separate checks, because they can disagree (as measured above):

- **Label drift**: chi-square + PSI on the predicted `<=50K` / `>50K` mix.
- **Confidence drift**: KS test + PSI on the predicted probability of
  `>50K`, plus a `confidence_summary` (mean, std, median, quartiles, and a
  **borderline rate** - the share of predictions within 0.10 of the decision
  threshold, i.e. cases a small nudge would flip).

The batch verdict is the worse of the two, same rule as `DriftReport`: the
loudest signal decides.

### Not assumed to mean failure

Every `PredictionDriftReport` carries a standing note:

> A shift in predicted-class proportions does not by itself mean the model
> has gotten worse. It can also reflect a genuine change in the incoming
> population - compare against the data-drift report for the same batch - or
> simply be sampling noise in a small batch. Confirming actual performance
> change needs true labels; see performance_monitoring.py.

This is not boilerplate - the stable-batch / drifted-batch comparison above is
the literal reason it is stated as a fact rather than a hope: the same model,
the same code, produced moderate prediction drift that was *not* yet visible
in the raw label mix, and separately produced performance loss that
prediction monitoring alone cannot see at all.

## Part B: Performance monitoring (requires true labels)

### Stable record identifiers

Every prediction is keyed by `f"{batch_id}:{original_row_index}"`
(`make_record_id`). This is stable because both halves are fixed at
prediction time - the key never depends on whether, when, or how a label
later arrives. Labels are submitted indexed the same way the batch's features
were, so a caller never constructs the key by hand; `submit_labels` does it
internally.

### Three rules, enforced structurally

1. **Predictions are recorded before any label exists.** `register_batch`
   never writes to `actual_label`. There is no code path that could use a
   prediction as a stand-in for ground truth, because the field is simply
   absent until `submit_labels` is called with labels from elsewhere.
2. **Zero labels never produces a number.**
   `compute_performance` with no labels returns `metrics=None`, not zeros or
   an empty confusion matrix. Verified by
   `test_zero_labels_never_fabricates_metrics`.
3. **Labels can arrive in any number of waves without duplicating anything.**
   Records are keyed by `record_id`, so a second `register_batch` call on the
   same batch creates nothing new, and `submit_labels` updates a record's
   `actual_label` in place. `test_labels_arriving_in_two_waves_do_not_duplicate_records`
   submits labels in two separate calls and checks the row count stays fixed.

### Measured: how many labels are enough

`min_labeled_sample` defaults to **300**, and it was measured the same way
Phase 5 measured `min_batch_rows` - by resampling, not guessing.

Repeated random samples (300 trials each) from the real holdout set, scored
with the real saved model, using the real labels:

| Sample size | std(accuracy) | std(recall) | mean(recall) |
| --- | --- | --- | --- |
| 10 | 0.1141 | 0.2958 | 0.80 |
| 30 | 0.0713 | 0.1753 | 0.80 |
| 100 | 0.0360 | 0.0824 | 0.80 |
| 200 | 0.0272 | 0.0580 | 0.78 |
| **300** | 0.0221 | **0.0489** | 0.79 |
| 500 | — | 0.0346 | — |

Recall's sampling noise is far larger than accuracy's at every size, because
recall is computed only over the ~24% of rows that are genuinely `>50K` - the
effective sample size behind a recall estimate is roughly a quarter of the
batch size. At `n=300`, recall's standard deviation settles to about 0.049 -
close to the 0.05 `DEFAULT_DEGRADATION_MARGIN` used to flag a drop against the
baseline. Below 300, a "degraded" flag on recall is uncomfortably likely to be
noise rather than signal; `DEFAULT_DEGRADATION_MARGIN` was chosen with that
in mind, not picked first and left unchecked.

### Insufficient samples are flagged, not hidden

Consistent with how Phase 5 handled small drift batches: a labelled sample
below the measured floor still gets its metrics **computed and returned**,
with a `SMALL LABELLED SAMPLE` note, rather than being suppressed. Hiding the
number would be worse than showing it with an honest caveat - see
`test_small_labeled_sample_is_flagged_but_still_computed`.

Zero labels is a different case entirely and is never computed - see rule 2
above.

### Comparing against the Phase 3 baseline

`load_baseline_test_metrics()` reads the `"test"` section of
`reports/baseline_metrics.json` - the held-out test-set metrics Phase 3
actually measured (accuracy 0.8251, precision 0.6028, recall 0.7887, F1
0.6833, ROC-AUC 0.9059). `compare_to_baseline` reports the delta per metric
and flags `degraded` when the drop exceeds the margin.

This is **a plain threshold, not a statistical test.** It does not know
whether the current sample is large enough to trust - that is exactly why
`PerformanceReport` always states `n_labeled` next to the verdict, and why the
report adds a note that a degraded flag "does not by itself prove the model
needs retraining" when one fires.

### ROC-AUC is optional, not assumed

`evaluate_predictions` (refactored out of Phase 3's `evaluate_classifier` so
the exact same metric logic serves both) returns `roc_auc=None` rather than
raising when either no probability was recorded for a row, or the labelled
subset happens to contain only one class - both real possibilities for a
small or partially labelled monitoring batch, neither of which could ever
happen on Phase 3's full stratified test set.

## Usage

```python
from monitor.data import load_adult_data
from monitor.drift import DriftDetector
from monitor.prediction import PredictionService
from monitor.prediction_monitoring import PredictionMonitor
from monitor.performance_monitoring import PerformanceMonitor, load_baseline_test_metrics
from monitor.simulation import BatchSimulator, get_reference_features, split_reference_and_holdout

df = load_adult_data()
reference_features = get_reference_features(df)
_, holdout = split_reference_and_holdout(df)

service = PredictionService()
prediction_monitor = PredictionMonitor.from_reference_features(service, reference_features)
performance_monitor = PerformanceMonitor(min_labeled_sample=300)
baseline = load_baseline_test_metrics()

batch = BatchSimulator(holdout, batch_size=500, seed=42).next_batch()
prediction = service.predict_simulated(batch)             # no labels involved

prediction_report = prediction_monitor.evaluate(prediction)
print(prediction_report.summary())                        # available immediately

performance_monitor.register_batch(prediction)             # still no labels
# ... time passes; labels arrive, maybe in several waves, maybe partially ...
performance_monitor.submit_labels(batch.batch_id, batch.labels)

performance_report = performance_monitor.compute_performance(batch.batch_id, baseline=baseline)
print(performance_report.summary())                        # only now is "accuracy" meaningful
```

## Limitations

- **The degradation margin (0.05) is a single uniform cut**, not tuned per
  metric. It was sized against recall's noise at the minimum sample, which
  makes it conservative for the less noisy metrics - this is intentional
  (fewer false "degraded" flags) but worth knowing.
- **`min_labeled_sample=300` is dataset-specific**, like Phase 5's
  `min_batch_rows=400`. Both were measured against this dataset's ~24%
  minority class rate; a dataset with a rarer positive class would need a
  larger floor, and this project's own tests show exactly how to re-measure
  one.
- **Confidence drift and label drift can disagree**, as demonstrated above.
  Neither is wrong; they measure different things, and that disagreement is
  itself useful information rather than a bug to resolve.
- **No alerting or persistence yet.** Every report here is a one-off
  calculation from in-memory records; nothing is stored between runs, and no
  threshold crossing triggers an action. That is the next phase.
