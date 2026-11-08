# Baseline Model

The baseline classifier, what it actually scored, and what each metric means.

All numbers come from `reports/baseline_metrics.json`, produced by
`python train_baseline.py --search`. Nothing here is estimated.

## What was trained

| Field | Value |
| --- | --- |
| Model | Logistic Regression (`C=1.0`, `max_iter=1000`) |
| Pipeline | Phase 2 preprocessor + classifier, bundled together |
| Input features | 14 raw columns, expanded to 108 after one-hot encoding |
| Target | `income`, positive class `>50K` |
| Split | 80% train / 20% test, stratified, `random_state=42` |
| Decision threshold | **0.29**, selected by cross-validation on training data |
| Class balance handling | **none** — no `class_weight`, no resampling |

Logistic Regression was chosen because it is fast, stable, and its
coefficients are interpretable. A baseline exists to be a reference point.

## Class distribution

| Split | Rows | `<=50K` | `>50K` |
| --- | --- | --- | --- |
| Train | 39,073 | 29,724 (76.07%) | 9,349 (23.93%) |
| Test | 9,769 | 7,431 (76.07%) | 2,338 (23.93%) |

Both splits carry exactly the 76.07 / 23.93 split of the full dataset, because
the split is stratified. The imbalance is **deliberately preserved** — a
monitor has to see the distribution real incoming data actually has, so no
rebalancing, reweighting or resampling is applied anywhere.

## Cross-validated results

Metrics from a single split carry sampling noise, so 5-fold stratified
cross-validation on the **training half** measures how much they move. At the
default 0.5 threshold:

| Metric | Mean | Std dev |
| --- | --- | --- |
| Accuracy | 0.8527 | ±0.0014 |
| Precision | 0.7352 | ±0.0042 |
| Recall | 0.6006 | ±0.0073 |
| F1 | 0.6611 | ±0.0044 |
| ROC-AUC | 0.9067 | ±0.0013 |

The spreads are very small — ROC-AUC varies by about ±0.001 across folds. So
the scores are stable, and a future change worth believing in has to move a
metric by more than roughly 0.005.

## The decision threshold

A classifier produces a probability; a threshold turns it into a label. The
default 0.5 is an arbitrary inheritance, and on imbalanced data it is usually
too reluctant to predict the minority class.

The threshold was chosen by sweeping candidate values against **out-of-fold
predictions on the training half** and taking the best F1. The test set was
not involved — choosing a threshold by looking at test scores would make the
test metrics optimistic and worthless.

Measured on out-of-fold training predictions:

| Threshold | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| 0.50 (default) | 0.8527 | 0.7351 | 0.6006 | 0.6611 |
| **0.29 (selected)** | 0.8277 | 0.6057 | **0.8020** | **0.6901** |

This is a **trade-off, not a free improvement**. Recall rises sharply, while
precision and accuracy fall. It is the right trade for this project because
the minority `>50K` class is the one worth detecting, and because accuracy is
an unreliable measure here anyway.

The threshold is baked into the saved model via
`FixedThresholdClassifier`, so `model.predict(...)` uses 0.29 and can never
silently disagree with the metrics reported for it.

## Measured results on the held-out test set

At the selected threshold of 0.29:

| Metric | Score | At 0.50 for comparison |
| --- | --- | --- |
| Accuracy | **0.8251** | 0.8519 |
| Precision (`>50K`) | **0.6028** | 0.7322 |
| Recall (`>50K`) | **0.7887** | 0.6009 |
| F1 (`>50K`) | **0.6833** | 0.6601 |
| ROC-AUC | **0.9059** | 0.9059 |

Confusion matrix at 0.29:

|  | predicted `<=50K` | predicted `>50K` |
| --- | --- | --- |
| **actual `<=50K`** | 6,216 | 1,215 |
| **actual `>50K`** | 494 | 1,844 |

Per-class report:

```
              precision    recall  f1-score   support

       <=50K       0.93      0.84      0.88      7431
        >50K       0.60      0.79      0.68      2338

    accuracy                           0.83      9769
   macro avg       0.76      0.81      0.78      9769
weighted avg       0.85      0.83      0.83      9769
```

Note **ROC-AUC is identical** at both thresholds (0.9059). That is the proof
that threshold tuning did not change the model at all — only which side of
the line each prediction falls on. The underlying ranking ability is the same.

### What changed, concretely

Of the 2,338 people in the test set who really earn `>50K`, the model at 0.5
missed 933. At 0.29 it misses **494** — it now finds 1,844 instead of 1,405.
The cost is 1,215 false alarms instead of 514.

### Is this good?

Honestly: it is a reasonable baseline with a known cost, not a strong model.

- **Always guessing `<=50K`** scores 0.7607 accuracy. At 0.29 the model scores
  0.8251 — higher, but accuracy is the wrong lens. Its real advantage is that
  it identifies 79% of high earners, which the trivial model does 0% of.
- **Train 0.8280 vs test 0.8251** (gap +0.0029). No meaningful overfitting.
- **Precision of 0.60** means two in five `>50K` predictions are wrong. In any
  real application that cost would have to be weighed deliberately.

## What each metric means

Four outcomes are possible for each prediction, with `>50K` as positive:

| | Meaning | Count at 0.29 |
| --- | --- | --- |
| **True positive** | Really `>50K`, predicted `>50K` | 1,844 |
| **False positive** | Really `<=50K`, predicted `>50K` — false alarm | 1,215 |
| **False negative** | Really `>50K`, predicted `<=50K` — missed | 494 |
| **True negative** | Really `<=50K`, predicted `<=50K` | 6,216 |

**Accuracy** — fraction of all predictions that were right:
`(1844 + 6216) / 9769 = 0.8251`. Intuitive but *misleading on imbalanced
data*, since 0.7607 comes free from always guessing the majority. Never quote
it here without that reference number.

**Precision** — when the model says `>50K`, how often is it right?
`1844 / (1844 + 1215) = 0.6028`. Matters when acting on a positive prediction
is expensive.

**Recall** — of all the people who really are `>50K`, how many were found?
`1844 / (1844 + 494) = 0.7887`. Matters when *missing* a positive case is
expensive.

Precision and recall pull against each other, and the threshold is the dial
between them. Which one matters more is a judgement about consequences, not a
property of the model.

**F1-score** — harmonic mean of precision and recall, `0.6833`. One number
for "is this good at the positive class?", and it stays low unless both are
decent.

**ROC-AUC** — `0.9059`. Ignores the threshold entirely: take a random `>50K`
person and a random `<=50K` person; how often does the model score the former
higher? About 91% of the time. 0.5 is coin-flipping, 1.0 is perfect.

### Why these metrics matter for monitoring

Each degrades for a different reason, so tracking them together tells you
*what kind* of problem appeared:

- Accuracy drops while ROC-AUC holds → the class mix or threshold shifted, not
  the model's ability to rank.
- ROC-AUC drops → the relationship between features and income genuinely
  changed. This is the serious one.
- Precision and recall moving in opposite directions → the incoming mix of
  classes changed, not the model.

## Hyperparameter check

`--search` cross-validates Logistic Regression's regularisation strength `C`,
scored on ROC-AUC because that is threshold-independent:

| C | ROC-AUC |
| --- | --- |
| 0.01 | 0.9049 ±0.0014 |
| 0.1 | 0.9067 ±0.0013 |
| **1.0 (default)** | **0.9067 ±0.0013** |
| 10.0 | 0.9066 ±0.0013 |

**The default is already optimal.** The best value (`C=0.1`) ties with the
default to four decimal places — a gain of +0.0000. Regularisation strength
simply is not a lever on this problem, so the default was kept. This is worth
recording precisely because it is a negative result: it rules out a tuning
avenue rather than leaving it as an open question.

## Comparison model

`--compare` also trains a Random Forest, using the same threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8251 | 0.6028 | 0.7887 | 0.6833 | 0.9059 |
| Random Forest | 0.8265 | 0.5996 | 0.8276 | 0.6954 | 0.9184 |

The forest is slightly ahead on recall, F1 and ROC-AUC, and essentially tied
on accuracy. Notably the gap is *narrower* than at the default threshold,
where the forest led on accuracy 0.8671 to 0.8519 — much of its apparent
advantage was really just a better-placed implicit threshold.

Logistic Regression is kept as the baseline: simpler, faster, interpretable,
and the remaining gap does not justify the extra complexity in a project whose
subject is monitoring rather than maximising accuracy.

## Per-group behaviour (measured, not corrected)

A single overall score can hide very different treatment of different groups,
so the same metrics are broken down by `sex` and `race` on the test set.

By `sex`:

| group | n | actual `>50K` rate | selection rate | accuracy | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| Female | 3,289 | 0.1116 | 0.1198 | 0.9115 | 0.5964 | 0.6403 |
| Male | 6,480 | 0.3042 | 0.4113 | 0.7812 | 0.6038 | 0.8163 |

By `race`:

| group | n | actual `>50K` rate | selection rate | accuracy | precision | recall |
| --- | --- | --- | --- | --- | --- | --- |
| Amer-Indian-Eskimo | 104 | 0.0865 | 0.1731 | 0.8558 | 0.3333 | 0.6667 |
| Asian-Pac-Islander | 304 | 0.2566 | 0.3684 | 0.7961 | 0.5714 | 0.8205 |
| Black | 944 | 0.1282 | 0.1282 | 0.8962 | 0.5950 | 0.5950 |
| Other | 69 | 0.1594 | 0.1594 | 0.8841 | 0.6364 | 0.6364 |
| White | 8,348 | 0.2538 | 0.3351 | 0.8172 | 0.6060 | 0.7999 |

Two disparity ratios (lowest group divided by highest, computed only over
groups with at least 100 rows so tiny samples do not dominate):

| Column | Selection rate ratio | Recall ratio |
| --- | --- | --- |
| `sex` | 0.2913 | 0.7844 |
| `race` | 0.3480 | 0.7252 |

**How to read this.** The model predicts `>50K` for 41% of men and 12% of
women — a selection-rate ratio of 0.29. But the underlying data also differs:
30.4% of men in the test set genuinely earn `>50K` versus 11.2% of women. So
most of the selection gap reflects the 1994 census population the data was
drawn from, not something the model invented.

The recall ratio is the more pointed number. The model finds 81.6% of genuine
high-earning men but only 64.0% of genuine high-earning women, and 59.5% of
genuine high earners who are Black versus 80.0% of those who are White. These
are cases where the model is **worse at the same task** depending on the
group, which is not explained away by base rates.

This is a measurement, not a fix. No mitigation has been applied, and this
model must not be used to make decisions about real people. The purpose here
is to show that the disparity is quantified and visible rather than unknown.

## The saved artifacts

| File | Contents |
| --- | --- |
| `artifacts/baseline_model.joblib` | Complete fitted pipeline — preprocessing, classifier **and** the 0.29 threshold |
| `artifacts/baseline_model_metadata.json` | Training date, seed, model type, feature names, dataset info, threshold selection, CV results, hyperparameter search, fairness, measured metrics, library versions |
| `reports/baseline_metrics.json` | All of the measured numbers above |

One load is enough to score raw rows:

```python
from monitor.model import load_model, get_threshold

model = load_model()
get_threshold(model)                  # 0.29
model.predict(raw_dataframe)          # preprocessing + threshold applied
```

`artifacts/` is git-ignored because the model is a regenerable binary.
`reports/` **is** committed: small, readable, and it records what the baseline
actually achieved.

## How leakage is prevented

Three separate places:

1. **Preprocessing.** `Pipeline.fit(X_train, y_train)` fits the preprocessor
   on training features only, inside the same call, so imputation medians,
   scaler statistics and category lists never see the test set.
   `test_preprocessor_is_fitted_on_training_rows_only` asserts the scaler's
   learned mean for `age` equals the training split's mean and **not** the
   mean of train and test combined.
2. **Cross-validation.** A fresh pipeline is built per fold, so no fold reuses
   parameters learned on another.
3. **Threshold selection.** Chosen from out-of-fold predictions within the
   training half. The test set is scored once, at the end, with the threshold
   already fixed.

## Remaining limitations

These are genuine and not fixable by better engineering:

- **Precision of 0.6028.** Two in five `>50K` predictions are wrong. This is
  the deliberate cost of the recall gain; a different application might want
  the threshold moved back.
- **One train/test split for the final numbers.** Cross-validation quantifies
  the uncertainty (±0.0014 on accuracy) but the headline test figures still
  come from one 20% sample.
- **Measured disparity across sex and race**, documented above. Quantified,
  not mitigated.
- **1994 data.** A census extract that does not describe present-day incomes.
  No amount of modelling fixes this; it is a property of the dataset.
- **Educational prototype.** Not validated for any real use, and not to be
  used for decisions about people.
