# Prediction Engine and Batch Simulation

How the saved model scores incoming data, and what the different kinds of
data in this project actually are.

## The five kinds of data

These terms get used loosely in a lot of writing about monitoring, which
causes real confusion. In this project they mean exactly this:

| Term | What it is | Size | Does the model see it? |
| --- | --- | --- | --- |
| **Training data** | The 80% split the model learned from | 39,073 rows | Yes — it was fitted on these |
| **Reference data** | The training features, kept as the "normal" baseline | 39,073 rows | Already has |
| **Incoming batch** | A group of new records arriving to be scored | configurable, e.g. 500 | Only the features, only to predict |
| **Predictions** | What the model says about each row | one per row | It produces them |
| **Actual labels** | The true answer for each row | one per row | **Never** |

### Training data

The 80% of the dataset used to fit the model in Phase 3. Created by
`split_train_test(df, test_size=0.2)` with `random_state=42`, stratified so
it keeps the natural 76/24 class split.

### Reference data

The *same rows* as the training data, but thought about differently. Training
data is an input to fitting; reference data is a **yardstick**. When a batch
arrives later and we ask "does this look normal?", normal means "like the
reference data", because that is the distribution the model learned.

`get_reference_features(df)` returns it. Nothing in this phase compares
against it yet — that is drift detection, which comes later. It is defined
now so that the comparison has a fixed, documented baseline when it arrives.

### Incoming batches

Groups of records that simulate production traffic. They are drawn from the
**held-out 20%** — the same test split from Phase 3, rebuilt with the same
seed. This matters: every batch is data the model has genuinely never seen,
so the predictions are honest rather than inflated by memorisation.

Verified directly, not assumed:

```
holdout IS the Phase 3 test split:        True
zero training rows in holdout:            True
19 batches x 500 rows, all disjoint:      True
none of those rows were trained on:       True
```

### Predictions

What the model outputs for each row: a label (`<=50K` or `>50K`) and, where
the model supports it, the probability of `>50K`.

A prediction is **not** a label. It is a claim. The difference between the two
is exactly what performance monitoring measures.

### Actual labels

The true income class. In this simulation they already exist, because the
dataset has them — but the system treats them the way a real one must: as
information that arrives **later, separately, and sometimes never**.

## Label separation, enforced rather than promised

This is the single most important design rule in this phase. If labels leaked
into prediction, every metric afterwards would be meaningless.

Two mechanisms:

1. **Structural.** A `SimulatedBatch` holds `features` (no income column) and
   `labels` (a separate Series) as different attributes. The service is handed
   `batch.features` only.
2. **An active guard.** `PredictionService.predict_batch()` raises if the frame
   still contains the income column:

   ```
   ValueError: Batch contains the label column 'income'. The prediction
   service must never receive true labels - pass only feature columns and
   keep labels separate.
   ```

The guard exists because convention is not enough — a future change could
easily pass the wrong frame, and a loud failure is far better than silently
good-looking results.

Labels are not thrown away. They stay on the batch so that performance can be
scored later, once they would realistically have become available. They just
never influence a prediction.

## The prediction flow

```
SimulatedBatch
  ├── features  ──────►  PredictionService.predict_batch()
  │                          │
  │                          ├── 1. refuse if the label column is present
  │                          ├── 2. validate against the schema (Phase 2)
  │                          ├── 3. model.predict()   — preprocessing included
  │                          └── 4. model.predict_proba()  (if supported)
  │                          │
  │                          ▼
  │                      BatchPrediction
  │                        batch_id, predicted_at, model_type, threshold,
  │                        features + predictions + probabilities, validation
  │
  └── labels  ───────►  held back for later performance scoring
```

### One model file does everything

The saved artifact is a single `Pipeline` containing the preprocessor, the
classifier and the decision threshold. So `model.predict(raw_rows)` applies
the imputing, scaling and one-hot encoding **that were fitted on the training
data**, automatically.

There is no separate preprocessing step at inference time. That is deliberate:
a second, independently fitted preprocessor is the classic way for training
and serving to silently diverge.

### Batch identity

Every batch gets an ID and a timestamp, so a prediction can always be traced
back to when it happened and which batch it belonged to:

```
batch_20261108T162133Z_8bcff2d5
└┬──┘ └──────┬──────┘ └───┬───┘
 │           │            └── random suffix, keeps same-second batches distinct
 │           └── UTC timestamp, so IDs sort chronologically
 └── prefix
```

Timestamps are ISO-8601 and timezone-aware, so they survive being written to
JSON and read back without ambiguity. An ID can also be supplied explicitly
when a fixed one is wanted.

### Features are preserved with the predictions

`BatchPrediction.to_frame()` returns the original feature values next to the
predictions, plus the batch ID and timestamp. Monitoring needs both halves:
the features to notice the *input* changing, and the predictions to notice the
*output* changing. Keeping only predictions would make it impossible to tell
which of the two had happened.

## How batches are sampled

Three decisions, each with a reason:

**Random, not stratified.** A random draw keeps the natural 76/24 split *on
average* while letting individual batches vary. Real arrivals fluctuate, and
a monitor has to see ordinary fluctuation to distinguish it from a genuine
change. Forcing every batch to exactly 23.93% would hide that.

Measured across three batches of 500:

| Batch | Actual `>50K` | Predicted `>50K` |
| --- | --- | --- |
| 1 | 20.80% | 30.00% |
| 2 | 22.20% | 33.60% |
| 3 | 25.60% | 32.80% |
| *whole holdout* | *23.93%* | |

The actual rates scatter around 23.93% — that scatter is the normal variation
later phases must not mistake for drift. **No balancing is applied anywhere.**

(The predicted rate sits above the actual rate because the model runs at
threshold 0.29, which deliberately over-predicts `>50K` to gain recall. That
is expected behaviour from Phase 3, not an error.)

**Batches are disjoint.** The pool is shuffled once with the seed, then handed
out in consecutive slices, so each record arrives at most once. Re-sampling
the same rows would create duplicates that later look like a distribution
change when nothing had actually changed.

**Reproducible.** The same seed and batch size always produce the same
batches, in the same order. `reset()` replays the sequence from the start.

## The hook for later drift experiments

Phase 5 will need batches containing a *known* change, to check that drift
detection actually notices it. The simulator therefore accepts a `modifier`:
any function that takes a feature frame and returns a changed copy.

Four ready-made builders:

| Builder | Effect |
| --- | --- |
| `shift_numeric(column, amount)` | Add a constant, e.g. everyone 15 years older |
| `scale_numeric(column, factor)` | Multiply, for a proportional change |
| `overwrite_category(column, value, fraction)` | Set a category on part of the batch, including values never seen in training |
| `combine(*modifiers)` | Apply several in order |

**No drift detection is implemented here.** This only injects changes.

Demonstrated with `python simulate_batches.py --demo-modifier`, applying
`age + 15` and setting half of `occupation` to an unseen `"Prompt-Engineer"`:

| Metric | Original | Modified |
| --- | --- | --- |
| mean age | 39.81 | 54.81 |
| predicted `>50K` % | 28.33 | 35.00 |
| mean P(`>50K`) | 0.2218 | 0.2598 |
| **actual `>50K` %** | **18.67** | **18.67** |

Three things this shows:

- The change propagates into the predictions, so there will be a signal for
  Phase 5 to find.
- The unknown category did not break anything. Validation reported it
  (`Unknown categories not in the schema: ['Prompt-Engineer'] (150 rows)`)
  and the encoder represented it as all-zeros, so scoring continued.
- **The labels are identical.** Modifiers touch features only, so ground
  truth and class balance are never altered. The simulator enforces this and
  also rejects a modifier that drops a column or changes the row count.

## Error handling

| Situation | Behaviour |
| --- | --- |
| Model file missing | `FileNotFoundError` naming `train_baseline.py` |
| Batch contains the label column | `ValueError` explaining label separation |
| Empty batch | `ValueError` |
| Required column missing / wrong type | `ValueError` from Phase 2 validation, listing what is wrong |
| Unknown category, missing values, out-of-range values | Scored normally, recorded as warnings |
| Extra unexpected column | Scored normally, recorded as a warning |
| `predict` or `predict_proba` raises | `RuntimeError` naming the batch size and original error |
| Model has no `predict_proba` | Labels returned, `probabilities` is `None`, a warning is recorded |
| Holdout pool exhausted | `ValueError` saying how many rows remain |
| Modifier drops a column or changes row count | `ValueError` |

Validation **reports** rather than repairs, consistent with Phase 2: a batch
with a few odd rows is still worth scoring, and silently dropping those rows
would destroy the signal the monitor exists to surface.

## Usage

```python
from monitor.data import load_adult_data
from monitor.prediction import PredictionService
from monitor.simulation import BatchSimulator, split_reference_and_holdout

df = load_adult_data()
reference, holdout = split_reference_and_holdout(df)

service = PredictionService()                 # loads artifacts/baseline_model.joblib
simulator = BatchSimulator(holdout, batch_size=500, seed=42)

batch = simulator.next_batch()                # features + labels, kept apart
result = service.predict_simulated(batch)     # only features are passed in

print(result.summary())
record = result.to_frame()                    # features + predictions, for monitoring
truth = batch.labels                          # available later, never used above
```

Nothing in this phase retrains the model, writes to `artifacts/`, or modifies
the dataset.
