# Dataset Exploration Report

Generated on 2026-10-21 by `explore_dataset.py`. All figures are computed from the dataset at run time.

## Source

| Field | Value |
| --- | --- |
| Dataset | Adult (also called Census Income) |
| UCI repository | https://archive.ics.uci.edu/dataset/2/adult |
| UCI dataset id | 2 |
| DOI | 10.24432/C5XW20 |
| Creators | Barry Becker, Ronny Kohavi |
| Donated | 1996 (extracted from the 1994 US Census database) |
| Accessed via | `ucimlrepo` Python package |

### Usage terms

The UCI Machine Learning Repository publishes this dataset for public research and educational use. The `ucimlrepo` metadata response does not include a machine-readable licence field, so the licence shown on the dataset page (https://archive.ics.uci.edu/dataset/2/adult) is the authoritative statement of terms. This project uses the data only for non-commercial educational purposes and redistributes no part of it.

The records describe real people and include sensitive attributes (`race`, `sex`, `native_country`, `marital_status`). They are used here solely to demonstrate monitoring techniques, never to make or suggest decisions about individuals.

## Dimensions

- Rows: **48,842**
- Columns: **15** (6 numeric features, 8 categorical features, 1 target)
- Target column: **`income`**

This is the combined dataset. UCI distributes it as a 32,561-row training split and a 16,281-row test split; `ucimlrepo` returns both concatenated into a single 48,842-row table.

## Schema

| column         | dtype   |   non_null |   missing |   missing_pct |   n_unique |
|:---------------|:--------|-----------:|----------:|--------------:|-----------:|
| age            | int64   |      48842 |         0 |          0    |         74 |
| fnlwgt         | int64   |      48842 |         0 |          0    |      28523 |
| education_num  | int64   |      48842 |         0 |          0    |         16 |
| capital_gain   | int64   |      48842 |         0 |          0    |        123 |
| capital_loss   | int64   |      48842 |         0 |          0    |         99 |
| hours_per_week | int64   |      48842 |         0 |          0    |         96 |
| workclass      | str     |      46043 |      2799 |          5.73 |          8 |
| education      | str     |      48842 |         0 |          0    |         16 |
| marital_status | str     |      48842 |         0 |          0    |          7 |
| occupation     | str     |      46033 |      2809 |          5.75 |         14 |
| relationship   | str     |      48842 |         0 |          0    |          6 |
| race           | str     |      48842 |         0 |          0    |          5 |
| sex            | str     |      48842 |         0 |          0    |          2 |
| native_country | str     |      47985 |       857 |          1.75 |         41 |
| income         | str     |      48842 |         0 |          0    |          2 |

## Missing values

- Rows with at least one missing value: **3,620** (7.41% of the dataset)
- Only three columns contain missing values; all three are categorical.

| column         | dtype   |   non_null |   missing |   missing_pct |   n_unique |
|:---------------|:--------|-----------:|----------:|--------------:|-----------:|
| occupation     | str     |      46033 |      2809 |          5.75 |         14 |
| workclass      | str     |      46043 |      2799 |          5.73 |          8 |
| native_country | str     |      47985 |       857 |          1.75 |         41 |

### Why the raw missing counts are misleading

Missing values are encoded **inconsistently** between the two original splits. Rows from the training split use the literal string `"?"`, while rows from the test split use a real empty value that pandas reads as `NaN`. Loading the data without handling this hides roughly two thirds of the missing values and creates phantom categories such as `"?"` appearing as a legitimate `workclass`.

`data.clean_data()` converts every `"?"` to `NaN`. After this fix the category counts match the UCI documentation exactly (workclass 8, occupation 14, native_country 41), which confirms the cleaning is correct.

## Duplicate rows

- Fully identical rows: **52** (0.1065% of the dataset)
- Distinct rows: **48,790**

These are not necessarily errors. The records are anonymous census responses with no identifier column, so two different people can legitimately share every recorded value. At roughly 0.1% they are left in place.

## Target class distribution

| label   |   count |   percent |
|:--------|--------:|----------:|
| <=50K   |   37155 |     76.07 |
| >50K    |   11687 |     23.93 |

The classes are imbalanced at roughly **3.18:1** in favour of `<=50K`. This matters for two reasons: a model that always predicts `<=50K` would already score about 76% accuracy, so accuracy alone is a poor measure; and this natural distribution is deliberately preserved throughout the project, because rebalancing it would misrepresent what real incoming data looks like.

## Numeric features

| column         | count     | mean       | std        | min       | 25%        | 50%        | 75%        | max          |
|:---------------|:----------|:-----------|:-----------|:----------|:-----------|:-----------|:-----------|:-------------|
| age            | 48,842.00 | 38.64      | 13.71      | 17.00     | 28.00      | 37.00      | 48.00      | 90.00        |
| fnlwgt         | 48,842.00 | 189,664.13 | 105,604.03 | 12,285.00 | 117,550.50 | 178,144.50 | 237,642.00 | 1,490,400.00 |
| education_num  | 48,842.00 | 10.08      | 2.57       | 1.00      | 9.00       | 10.00      | 12.00      | 16.00        |
| capital_gain   | 48,842.00 | 1,079.07   | 7,452.02   | 0.00      | 0.00       | 0.00       | 0.00       | 99,999.00    |
| capital_loss   | 48,842.00 | 87.50      | 403.00     | 0.00      | 0.00       | 0.00       | 0.00       | 4,356.00     |
| hours_per_week | 48,842.00 | 40.42      | 12.39      | 1.00      | 40.00      | 40.00      | 45.00      | 99.00        |

Points worth noting for later drift monitoring:

- `capital_gain` is zero for **91.74%** of rows and `capital_loss` is zero for **95.33%**. Both are heavily zero-inflated, so ordinary mean-based drift checks will be dominated by the few non-zero values.
- `capital_gain` has **244** rows at exactly 99999, which is a top-coded sentinel value rather than a real amount.
- `education` and `education_num` are a perfect one-to-one pair: `education_num` is simply the numeric rank of `education`. They carry identical information in two forms.
- `fnlwgt` is a census sampling weight describing how many people a row represents. It is not a property of the individual and is not meaningful as a predictive feature.

## Categorical features

| column         |   n_categories | most_common        |   most_common_pct |
|:---------------|---------------:|:-------------------|------------------:|
| workclass      |              8 | Private            |             73.64 |
| education      |             16 | HS-grad            |             32.32 |
| marital_status |              7 | Married-civ-spouse |             45.82 |
| occupation     |             14 | Prof-specialty     |             13.41 |
| relationship   |              6 | Husband            |             40.37 |
| race           |              5 | White              |             85.5  |
| sex            |              2 | Male               |             66.85 |
| native_country |             41 | United-States      |             91.35 |

Several columns are strongly dominated by one category (`native_country` is 91.35% United-States, `race` is 85.50% White). Rare categories will produce noisy drift signals on small batches, which is something the monitoring phases will need to account for.

## Fairness note

The positive-class rate differs sharply across groups in this data: 30.38% of male records are labelled `>50K` versus 10.93% of female records. This reflects documented inequities in the 1994 US census population plus the dataset's own sampling, not a ground truth about earning potential. Any model trained on it will reproduce these patterns, which is one reason this project is explicitly an educational monitoring prototype and not a decision-making system.
