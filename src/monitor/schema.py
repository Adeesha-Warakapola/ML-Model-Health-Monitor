"""The expected shape of an Adult dataset batch.

This is the contract that incoming data is checked against. Every column is
described once here, and both validation and preprocessing read from it, so
the two can never disagree about what the data should look like.

Two different ideas are deliberately kept apart:

* **Plausibility bounds** (`min_value` / `max_value`) describe values that are
  physically or logically impossible, such as a negative age or more than 168
  hours in a week. Breaking one of these means the data is wrong.
* **The observed range** in the reference data is a different thing entirely.
  `hours_per_week` happens to top out at 99 in the Adult data, but a batch
  containing 120 is unusual rather than invalid. Detecting that kind of shift
  is drift monitoring, which belongs to a later phase - not validation.

Mixing the two would make the validator fire constantly on data that is
merely new, which is exactly the noise a monitoring tool must avoid.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Everything the project knows about one column."""

    name: str
    kind: str  # "numeric" or "categorical"
    required: bool = True
    allowed_values: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    rationale: str = ""

    @property
    def is_numeric(self) -> bool:
        return self.kind == "numeric"

    @property
    def is_categorical(self) -> bool:
        return self.kind == "categorical"


# Category lists below were read off the real dataset (see
# docs/dataset_exploration.md) and cross-checked against the attribute listing
# in the UCI documentation. Both agree: 8 workclass, 16 education, 7
# marital_status, 14 occupation, 6 relationship, 5 race, 2 sex, 41
# native_country. `test_schema_matches_the_real_dataset` re-checks this.

_NUMERIC_SPECS = [
    ColumnSpec(
        name="age",
        kind="numeric",
        min_value=17,
        max_value=120,
        rationale="UCI extracted only records with age > 16; 120 is a human lifespan limit.",
    ),
    ColumnSpec(
        name="fnlwgt",
        kind="numeric",
        min_value=1,
        rationale=(
            "Census sampling weight; UCI extracted only records with fnlwgt > 1. "
            "No upper bound, because a weight has no documented ceiling."
        ),
    ),
    ColumnSpec(
        name="education_num",
        kind="numeric",
        min_value=1,
        max_value=16,
        rationale="Ordinal rank of the 16 education categories, so it cannot leave 1-16.",
    ),
    ColumnSpec(
        name="capital_gain",
        kind="numeric",
        min_value=0,
        max_value=99999,
        rationale=(
            "Cannot be negative. 99999 is the top-coded maximum used by the dataset "
            "(244 reference rows sit exactly on it), so larger values are not meaningful."
        ),
    ),
    ColumnSpec(
        name="capital_loss",
        kind="numeric",
        min_value=0,
        rationale=(
            "Cannot be negative. No upper bound: unlike capital_gain there is no "
            "documented top-code, and the observed maximum is only a sample property."
        ),
    ),
    ColumnSpec(
        name="hours_per_week",
        kind="numeric",
        min_value=1,
        max_value=168,
        rationale="UCI extracted only records with hours > 0; a week contains 168 hours.",
    ),
]

_CATEGORICAL_SPECS = [
    ColumnSpec(
        name="workclass",
        kind="categorical",
        allowed_values=(
            "Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc",
            "Self-emp-not-inc", "State-gov", "Without-pay",
        ),
    ),
    ColumnSpec(
        name="education",
        kind="categorical",
        allowed_values=(
            "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th",
            "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad",
            "Masters", "Preschool", "Prof-school", "Some-college",
        ),
    ),
    ColumnSpec(
        name="marital_status",
        kind="categorical",
        allowed_values=(
            "Divorced", "Married-AF-spouse", "Married-civ-spouse",
            "Married-spouse-absent", "Never-married", "Separated", "Widowed",
        ),
    ),
    ColumnSpec(
        name="occupation",
        kind="categorical",
        allowed_values=(
            "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
            "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
            "Other-service", "Priv-house-serv", "Prof-specialty",
            "Protective-serv", "Sales", "Tech-support", "Transport-moving",
        ),
    ),
    ColumnSpec(
        name="relationship",
        kind="categorical",
        allowed_values=(
            "Husband", "Not-in-family", "Other-relative", "Own-child",
            "Unmarried", "Wife",
        ),
    ),
    ColumnSpec(
        name="race",
        kind="categorical",
        allowed_values=(
            "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White",
        ),
    ),
    ColumnSpec(
        name="sex",
        kind="categorical",
        allowed_values=("Female", "Male"),
    ),
    ColumnSpec(
        name="native_country",
        kind="categorical",
        allowed_values=(
            "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
            "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
            "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong",
            "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan",
            "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru",
            "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland",
            "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States",
            "Vietnam", "Yugoslavia",
        ),
    ),
]

FEATURE_SPECS: dict[str, ColumnSpec] = {
    spec.name: spec for spec in _NUMERIC_SPECS + _CATEGORICAL_SPECS
}

NUMERIC_COLUMNS = [spec.name for spec in _NUMERIC_SPECS]
CATEGORICAL_COLUMNS = [spec.name for spec in _CATEGORICAL_SPECS]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def derive_categories(reference_df: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Read the categories actually present in a reference DataFrame.

    This is how the hard-coded `allowed_values` above were produced. Keeping
    it as a function means the schema can be re-checked against the data
    rather than taken on trust.
    """
    return {
        column: tuple(sorted(reference_df[column].dropna().unique().tolist()))
        for column in CATEGORICAL_COLUMNS
        if column in reference_df.columns
    }
