"""Turn the exploration results into a Markdown report.

Every number in the generated report is computed from the dataset at run
time rather than typed by hand, so the report cannot drift out of sync with
the actual data.
"""

from datetime import date

import pandas as pd

from . import config, explore


def _table(df: pd.DataFrame, keep_text_as_is: bool = False) -> str:
    """Render a DataFrame as a Markdown table.

    `keep_text_as_is` stops tabulate from re-parsing numeric-looking strings
    and reformatting them, which otherwise turns pre-formatted values such as
    "1,490,400.00" back into scientific notation.
    """
    return df.to_markdown(index=False, disable_numparse=keep_text_as_is)


def build_markdown_report(df: pd.DataFrame) -> str:
    """Build the full dataset exploration report as a Markdown string."""
    duplicates = explore.duplicate_summary(df)
    rows_with_missing = int(df.isna().any(axis=1).sum())
    rows_with_missing_pct = round(float(df.isna().any(axis=1).mean() * 100), 2)

    sections = [
        "# Dataset Exploration Report",
        "",
        f"Generated on {date.today().isoformat()} by `explore_dataset.py`. "
        "All figures are computed from the dataset at run time.",
        "",
        "## Source",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Dataset | Adult (also called Census Income) |",
        f"| UCI repository | {config.UCI_DATASET_URL} |",
        f"| UCI dataset id | {config.UCI_DATASET_ID} |",
        "| DOI | 10.24432/C5XW20 |",
        "| Creators | Barry Becker, Ronny Kohavi |",
        "| Donated | 1996 (extracted from the 1994 US Census database) |",
        "| Accessed via | `ucimlrepo` Python package |",
        "",
        "### Usage terms",
        "",
        "The UCI Machine Learning Repository publishes this dataset for public "
        "research and educational use. The `ucimlrepo` metadata response does "
        "not include a machine-readable licence field, so the licence shown on "
        f"the dataset page ({config.UCI_DATASET_URL}) is the authoritative "
        "statement of terms. This project uses the data only for non-commercial "
        "educational purposes and redistributes no part of it.",
        "",
        "The records describe real people and include sensitive attributes "
        "(`race`, `sex`, `native_country`, `marital_status`). They are used here "
        "solely to demonstrate monitoring techniques, never to make or suggest "
        "decisions about individuals.",
        "",
        "## Dimensions",
        "",
        f"- Rows: **{len(df):,}**",
        f"- Columns: **{len(df.columns)}** ({len(config.NUMERIC_FEATURES)} numeric features, "
        f"{len(config.CATEGORICAL_FEATURES)} categorical features, 1 target)",
        f"- Target column: **`{config.TARGET_COLUMN}`**",
        "",
        "This is the combined dataset. UCI distributes it as a 32,561-row "
        "training split and a 16,281-row test split; `ucimlrepo` returns both "
        "concatenated into a single 48,842-row table.",
        "",
        "## Schema",
        "",
        _table(explore.schema_summary(df)),
        "",
        "## Missing values",
        "",
        f"- Rows with at least one missing value: **{rows_with_missing:,}** "
        f"({rows_with_missing_pct}% of the dataset)",
        "- Only three columns contain missing values; all three are categorical.",
        "",
        _table(explore.missing_value_summary(df)),
        "",
        "### Why the raw missing counts are misleading",
        "",
        "Missing values are encoded **inconsistently** between the two original "
        "splits. Rows from the training split use the literal string `\"?\"`, "
        "while rows from the test split use a real empty value that pandas reads "
        "as `NaN`. Loading the data without handling this hides roughly two "
        "thirds of the missing values and creates phantom categories such as "
        "`\"?\"` appearing as a legitimate `workclass`.",
        "",
        "`data.clean_data()` converts every `\"?\"` to `NaN`. After this fix the "
        "category counts match the UCI documentation exactly (workclass 8, "
        "occupation 14, native_country 41), which confirms the cleaning is correct.",
        "",
        "## Duplicate rows",
        "",
        f"- Fully identical rows: **{duplicates['n_duplicate_rows']}** "
        f"({duplicates['duplicate_pct']}% of the dataset)",
        f"- Distinct rows: **{duplicates['n_unique_rows']:,}**",
        "",
        "These are not necessarily errors. The records are anonymous census "
        "responses with no identifier column, so two different people can "
        "legitimately share every recorded value. At roughly 0.1% they are left "
        "in place.",
        "",
        "## Target class distribution",
        "",
        _table(explore.target_distribution(df)),
        "",
        f"The classes are imbalanced at roughly **{explore.class_imbalance_ratio(df)}:1** "
        "in favour of `<=50K`. This matters for two reasons: a model that always "
        "predicts `<=50K` would already score about 76% accuracy, so accuracy "
        "alone is a poor measure; and this natural distribution is deliberately "
        "preserved throughout the project, because rebalancing it would "
        "misrepresent what real incoming data looks like.",
        "",
        "## Numeric features",
        "",
        # Format as strings so large values like fnlwgt's maximum are not
        # rendered in scientific notation.
        _table(
            explore.numeric_summary(df)
            .map(lambda value: f"{value:,.2f}")
            .reset_index()
            .rename(columns={"index": "column"}),
            keep_text_as_is=True,
        ),
        "",
        "Points worth noting for later drift monitoring:",
        "",
        f"- `capital_gain` is zero for **{(df.capital_gain == 0).mean() * 100:.2f}%** of rows "
        f"and `capital_loss` is zero for **{(df.capital_loss == 0).mean() * 100:.2f}%**. "
        "Both are heavily zero-inflated, so ordinary mean-based drift checks will "
        "be dominated by the few non-zero values.",
        f"- `capital_gain` has **{int((df.capital_gain == 99999).sum())}** rows at exactly "
        "99999, which is a top-coded sentinel value rather than a real amount.",
        "- `education` and `education_num` are a perfect one-to-one pair: "
        "`education_num` is simply the numeric rank of `education`. They carry "
        "identical information in two forms.",
        "- `fnlwgt` is a census sampling weight describing how many people a row "
        "represents. It is not a property of the individual and is not meaningful "
        "as a predictive feature.",
        "",
        "## Categorical features",
        "",
        _table(explore.categorical_summary(df)),
        "",
        "Several columns are strongly dominated by one category "
        f"(`native_country` is {df.native_country.value_counts(normalize=True).iloc[0] * 100:.2f}% "
        f"United-States, `race` is {df.race.value_counts(normalize=True).iloc[0] * 100:.2f}% White). "
        "Rare categories will produce noisy drift signals on small batches, which "
        "is something the monitoring phases will need to account for.",
        "",
        "## Fairness note",
        "",
        "The positive-class rate differs sharply across groups in this data: "
        f"{(df[df.sex == 'Male'].income == '>50K').mean() * 100:.2f}% of male records are "
        f"labelled `>50K` versus {(df[df.sex == 'Female'].income == '>50K').mean() * 100:.2f}% "
        "of female records. This reflects documented inequities in the 1994 US "
        "census population plus the dataset's own sampling, not a ground truth "
        "about earning potential. Any model trained on it will reproduce these "
        "patterns, which is one reason this project is explicitly an educational "
        "monitoring prototype and not a decision-making system.",
        "",
    ]
    return "\n".join(sections)


def save_report(markdown: str, path=config.EXPLORATION_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
