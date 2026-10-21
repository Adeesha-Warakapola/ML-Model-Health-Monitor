# Validation and Preprocessing

How incoming data is checked and converted into model-ready features, and
how the same pipeline is reused for monitoring batches in later phases.

## The flow

```
raw batch
   |
   v
validate_batch(df)  -->  ValidationResult (errors + warnings)
   |                     caller decides whether to continue
   v
transform_batch(preprocessor, df)  -->  108 named numeric features
```

Validation and transformation are separate steps on purpose. The validator
reports; it never modifies data and never drops rows. If it silently cleaned
up what it found, the monitor would hide exactly the problems it exists to
detect.

## The schema

`src/monitor/schema.py` describes every column once: whether it is numeric
or categorical, which categories are allowed, and what range of values is
physically possible. Validation and preprocessing both read from it, so they
cannot disagree.

The category lists were read off the real dataset with
`schema.derive_categories()` and cross-checked against the attribute listing
in the UCI documentation. Both agree. The test
`test_schema_matches_the_real_dataset` re-derives them on every test run, so
the schema cannot quietly drift away from the data it claims to describe.

### Plausibility bounds are not the observed range

This distinction matters and is easy to get wrong.

| Idea | Example | Who handles it |
| --- | --- | --- |
| **Plausibility bound** | `hours_per_week` of 200, when a week has 168 hours | Validation (Phase 2) |
| **Observed range** | `hours_per_week` of 120, higher than anything in the reference data | Drift detection (later phase) |

If the bounds were set to the observed minimum and maximum, the validator
would fire on data that is merely *new* rather than wrong, and the alerts
would become noise. So `capital_loss` has a lower bound of 0 and **no upper
bound**: unlike `capital_gain`, which tops out at the dataset's 99999
top-code, it has no documented ceiling, and inventing one would be guesswork.

## What validation checks

| Check | Severity | Why |
| --- | --- | --- |
| Required column missing | **error** | Nothing can be transformed without it |
| Numeric column holding text (or vice versa) | **error** | The column means something different than expected |
| Empty batch | **error** | Nothing to process |
| Unexpected extra column | warning | Harmless; `ColumnTransformer` drops it |
| Missing values | warning | The imputer has a defined behaviour |
| Unknown category | warning | The encoder has a defined behaviour |
| Value outside plausibility bounds | warning | Worth recording, but the row can still be scored |

Errors block processing; warnings do not. `ValidationResult.is_valid` reflects
errors only, and `raise_if_invalid()` is there for callers that want a hard
stop rather than a decision.

A batch with a handful of odd rows is usually still worth scoring. Refusing
the whole batch would mean losing visibility at precisely the moment
something is going wrong.

## What preprocessing does

| Column type | Steps | Reasoning |
| --- | --- | --- |
| Numeric | median imputation, then standard scaling | The median is used because `capital_gain` and `capital_loss` are over 90% zeros, which drags a mean around |
| Categorical | fill missing with the string `"Missing"`, then one-hot encode | Keeps missingness as a visible, countable signal instead of hiding it inside the most common category |

Output: **108 named features**, returned as a DataFrame rather than a bare
array so later phases can refer to individual encoded features by name and
trace rows back through the preserved index.

### Two safety behaviours worth knowing

**Unknown categories encode as all-zeros.** `OneHotEncoder` is configured
with `handle_unknown="ignore"`, so a `native_country` of `"Atlantis"` produces
zeros across every country column instead of raising. The output shape never
changes, which means a monitoring batch can always be scored. The validator
reports the unknown value separately, so it is never lost silently.

**Missing values in a never-missing column degrade gracefully.** Only
`workclass`, `occupation` and `native_country` contain missing values in the
training data, so only those three learn a `Missing` category. If a batch
arrives with a missing `sex`, it is filled with `"Missing"`, which the encoder
has never seen for that column, so it falls through to the all-zeros path
above. Safe, and still reported by the validator.

## Preventing data leakage

The preprocessor is **fitted once, on training data only**. Every later batch
is only ever transformed.

Fitting on a monitoring batch would be wrong twice over:

1. The scaler would re-centre on the new data, so a shift in the input would
   be scaled away and become invisible — destroying the signal the monitor
   exists to find.
2. The encoder would learn new categories, changing the feature columns and
   making comparisons against the reference period meaningless.

The module makes this hard to get wrong by accident: `fit_preprocessor` and
`transform_batch` are separate functions, and **there is no `fit_transform`
helper for batches**. The test
`test_transforming_a_batch_does_not_refit_the_preprocessor` asserts that the
scaler's learned mean is byte-identical before and after an extreme batch
passes through.

## Reuse for monitoring batches

This is the point of saving the fitted preprocessor to
`artifacts/preprocessor.joblib`. Later phases load that one file rather than
fitting anything:

```python
from monitor.preprocessing import load_preprocessor, transform_batch
from monitor.validation import validate_batch

preprocessor = load_preprocessor()        # the one fitted on training data

result = validate_batch(incoming_batch)   # always validate first
print(result.summary())                   # record the issues for the dashboard

if result.is_valid:
    features = transform_batch(preprocessor, incoming_batch)
    # features is ready for prediction and for drift comparison
```

Three properties make this reusable across every future batch:

- **Stable output.** The same 108 columns in the same order every time,
  whatever the batch contains, so batches are directly comparable to each
  other and to the reference period.
- **Stable parameters.** The imputation values, scaler statistics and
  category lists are frozen at fit time, so a change in the output genuinely
  reflects a change in the input rather than a change in the preprocessing.
- **A validation record per batch.** `ValidationResult` is a structured
  object, not printed text, so the monitoring history and the dashboard can
  store and chart issue counts over time.

The `artifacts/` directory is git-ignored. Regenerate it with:

```bash
python validate_and_preprocess.py
```

## Note on the train/test split

`split_train_test` uses `stratify`, so both halves carry the same 23.93%
`>50K` rate as the full dataset. This **preserves** the natural imbalance
rather than correcting it, which is what the project requires: rebalancing
would misrepresent what real incoming data looks like, and the monitor needs
to see the true distribution.
