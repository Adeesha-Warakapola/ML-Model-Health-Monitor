"""Check an incoming batch against the schema before it is used.

The validator never changes the data and never raises on its own. It returns
a `ValidationResult` describing everything it found, so the caller decides
what to do. This matters for monitoring: a batch with a few odd rows is
usually still worth scoring, and silently dropping those rows would destroy
the very signal the monitor exists to surface.

Issues come in two severities:

* **error** - the batch cannot be preprocessed reliably. A required column is
  absent, or a column holds the wrong kind of data entirely.
* **warning** - the batch is usable, but something is worth recording.
  Missing values, unknown categories and out-of-range numbers all land here,
  because the pipeline has a defined, safe behaviour for each of them.
"""

from dataclasses import dataclass, field

import pandas as pd

from . import schema

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """One specific problem found in the data."""

    severity: str
    check: str
    message: str
    column: str | None = None
    affected_rows: int | None = None

    def __str__(self) -> str:
        location = f"[{self.column}] " if self.column else ""
        rows = f" ({self.affected_rows} rows)" if self.affected_rows is not None else ""
        return f"{self.severity.upper():7s} {location}{self.message}{rows}"


@dataclass
class ValidationResult:
    """Everything the validator found, plus convenience accessors."""

    issues: list[ValidationIssue] = field(default_factory=list)
    n_rows: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == WARNING]

    @property
    def is_valid(self) -> bool:
        """True when nothing blocks preprocessing. Warnings do not block."""
        return not self.errors

    def summary(self) -> str:
        """A readable report suitable for printing or logging."""
        header = (
            f"Validation of {self.n_rows} rows: "
            f"{'PASSED' if self.is_valid else 'FAILED'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )
        if not self.issues:
            return header + "\n  No issues found."
        lines = [header] + [f"  {issue}" for issue in self.issues]
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        """Turn errors into an exception, for callers that want a hard stop."""
        if not self.is_valid:
            details = "\n".join(f"  {issue}" for issue in self.errors)
            raise ValueError(f"Batch failed validation:\n{details}")


def _check_columns(df: pd.DataFrame) -> list[ValidationIssue]:
    """Required columns must be present; extra columns are noted but harmless."""
    issues = []
    present = set(df.columns)

    missing = [c for c in schema.FEATURE_COLUMNS if c not in present]
    if missing:
        issues.append(
            ValidationIssue(
                severity=ERROR,
                check="required_columns",
                message=f"Required columns are missing: {missing}",
            )
        )

    unexpected = sorted(present - set(schema.FEATURE_COLUMNS))
    if unexpected:
        issues.append(
            ValidationIssue(
                severity=WARNING,
                check="unexpected_columns",
                message=(
                    f"Unexpected columns will be ignored by the pipeline: {unexpected}"
                ),
            )
        )
    return issues


def _check_dtypes(df: pd.DataFrame) -> list[ValidationIssue]:
    """Numeric columns must hold numbers; categorical columns must hold text.

    The check is deliberately loose about *which* numeric type. A column read
    back from CSV may be float rather than int simply because it contains a
    missing value, and that is not a problem.
    """
    issues = []
    for name, spec in schema.FEATURE_SPECS.items():
        if name not in df.columns:
            continue
        column = df[name]
        if spec.is_numeric and not pd.api.types.is_numeric_dtype(column):
            issues.append(
                ValidationIssue(
                    severity=ERROR,
                    check="dtype",
                    column=name,
                    message=f"Expected a numeric column but found dtype '{column.dtype}'",
                )
            )
        elif spec.is_categorical and pd.api.types.is_numeric_dtype(column):
            issues.append(
                ValidationIssue(
                    severity=ERROR,
                    check="dtype",
                    column=name,
                    message=f"Expected a text column but found dtype '{column.dtype}'",
                )
            )
    return issues


def _check_missing_values(df: pd.DataFrame) -> list[ValidationIssue]:
    """Report missing values per column. The imputer handles them, so: warning."""
    issues = []
    for name in schema.FEATURE_COLUMNS:
        if name not in df.columns:
            continue
        n_missing = int(df[name].isna().sum())
        if n_missing:
            pct = n_missing / len(df) * 100 if len(df) else 0.0
            issues.append(
                ValidationIssue(
                    severity=WARNING,
                    check="missing_values",
                    column=name,
                    message=f"{pct:.2f}% of values are missing and will be imputed",
                    affected_rows=n_missing,
                )
            )
    return issues


def _check_unknown_categories(df: pd.DataFrame) -> list[ValidationIssue]:
    """Flag category values the schema has never seen.

    These are safe - the encoder is configured to ignore them - but they are
    one of the clearest early signs that incoming data has shifted, so they
    are always reported.
    """
    issues = []
    for name in schema.CATEGORICAL_COLUMNS:
        if name not in df.columns:
            continue
        spec = schema.FEATURE_SPECS[name]
        if spec.allowed_values is None:
            continue
        seen = set(df[name].dropna().unique())
        unknown = sorted(seen - set(spec.allowed_values))
        if unknown:
            affected = int(df[name].isin(unknown).sum())
            issues.append(
                ValidationIssue(
                    severity=WARNING,
                    check="unknown_categories",
                    column=name,
                    message=(
                        f"Unknown categories not in the schema: {unknown}. "
                        "The encoder will represent them as all-zero."
                    ),
                    affected_rows=affected,
                )
            )
    return issues


def _check_value_ranges(df: pd.DataFrame) -> list[ValidationIssue]:
    """Flag numbers outside the schema's plausibility bounds."""
    issues = []
    for name in schema.NUMERIC_COLUMNS:
        if name not in df.columns:
            continue
        spec = schema.FEATURE_SPECS[name]
        column = df[name]
        if not pd.api.types.is_numeric_dtype(column):
            continue  # already reported by the dtype check

        if spec.min_value is not None:
            below = int((column < spec.min_value).sum())
            if below:
                issues.append(
                    ValidationIssue(
                        severity=WARNING,
                        check="value_range",
                        column=name,
                        message=f"Values below the minimum of {spec.min_value}",
                        affected_rows=below,
                    )
                )
        if spec.max_value is not None:
            above = int((column > spec.max_value).sum())
            if above:
                issues.append(
                    ValidationIssue(
                        severity=WARNING,
                        check="value_range",
                        column=name,
                        message=f"Values above the maximum of {spec.max_value}",
                        affected_rows=above,
                    )
                )
    return issues


def validate_batch(df: pd.DataFrame) -> ValidationResult:
    """Run every check against a batch and return the combined result.

    The column check runs first: if required columns are missing there is no
    point inspecting their contents, so the remaining checks simply skip any
    column that is not there.
    """
    if df.empty:
        return ValidationResult(
            issues=[
                ValidationIssue(
                    severity=ERROR,
                    check="empty_batch",
                    message="Batch contains no rows",
                )
            ],
            n_rows=0,
        )

    issues = (
        _check_columns(df)
        + _check_dtypes(df)
        + _check_missing_values(df)
        + _check_unknown_categories(df)
        + _check_value_ranges(df)
    )
    return ValidationResult(issues=issues, n_rows=len(df))
