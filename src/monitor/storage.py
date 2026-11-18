"""Persist monitoring history to SQLite, independent of any dashboard.

Nothing in this module computes a metric or a drift statistic - it only
saves and retrieves the results that `drift.py`, `prediction_monitoring.py`,
`performance_monitoring.py` and `alerts.py` already produced. That split
matters for two reasons: this module can be fully tested with a throwaway
database file and no model or dataset at all, and the monitoring logic stays
usable by something other than this storage layer (a notebook, a different
database, a dashboard) without being tangled up with SQL.

Six tables, one SQLite file:

    monitoring_runs           one row per time the monitoring pipeline ran
    batch_summaries           one row per batch: size, label coverage, rates
    drift_results             one row per feature per run (from DriftReport)
    prediction_drift_results  one row per run (from PredictionDriftReport)
    performance_metrics       one row per run (from PerformanceReport)
    alerts                    one row per *distinct* problem (see below)

Every table carries its own stable identifier and a timestamp, set once at
creation. The `alerts` table is the one table where identity is deliberately
blurred across time: a problem that keeps recurring updates one row (its
`occurrence_count` goes up) rather than ever creating a second row for "the
same thing happening again" - see `MonitoringRepository.save_alert`.

A connection is opened and closed for each call rather than held open for
the repository's lifetime. For a project at this scale (one writer, batch-at
-a-time, no concurrency) that trades a small amount of overhead for a much
simpler story: every operation is its own transaction, and there is no
connection state to go stale between calls.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .alerts import Alert, escalate
from .drift import DriftReport
from .performance_monitoring import PerformanceReport
from .prediction import BatchPrediction
from .prediction_monitoring import PredictionDriftReport


class MonitoringStorageError(RuntimeError):
    """A database operation failed for a reason the caller should see clearly.

    Wraps `sqlite3.Error` so callers of this module never need to import
    `sqlite3` themselves just to catch a storage failure.
    """


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    """A unique, sortable identifier, in the same style as `make_batch_id`."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{uuid.uuid4().hex[:8]}"


@dataclass
class AlertSaveOutcome:
    """What happened when one alert was saved: a new row, or an update to one."""

    alert_id: str
    dedup_key: str
    action: str  # "created" or "deduplicated"
    occurrence_count: int
    severity: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_runs (
    run_id       TEXT PRIMARY KEY,
    batch_id     TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    status       TEXT NOT NULL,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS batch_summaries (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   TEXT NOT NULL REFERENCES monitoring_runs(run_id),
    batch_id                 TEXT NOT NULL,
    n_rows                   INTEGER NOT NULL,
    n_labeled                INTEGER,
    predicted_positive_rate  REAL,
    actual_positive_rate     REAL,
    recorded_at              TEXT NOT NULL,
    UNIQUE(run_id, batch_id)
);

CREATE TABLE IF NOT EXISTS drift_results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES monitoring_runs(run_id),
    batch_id      TEXT NOT NULL,
    feature       TEXT NOT NULL,
    feature_type  TEXT,
    method        TEXT,
    status        TEXT NOT NULL,
    psi           REAL,
    statistic     REAL,
    p_value       REAL,
    effect_size   REAL,
    n_reference   INTEGER,
    n_batch       INTEGER,
    notes_json    TEXT,
    recorded_at   TEXT NOT NULL,
    UNIQUE(run_id, feature)
);

CREATE TABLE IF NOT EXISTS prediction_drift_results (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                        TEXT NOT NULL REFERENCES monitoring_runs(run_id),
    batch_id                      TEXT NOT NULL,
    label_status                  TEXT NOT NULL,
    label_psi                     REAL,
    confidence_status             TEXT,
    confidence_psi                REAL,
    batch_label_distribution_json TEXT,
    batch_confidence_json         TEXT,
    notes_json                    TEXT,
    recorded_at                   TEXT NOT NULL,
    UNIQUE(run_id)
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES monitoring_runs(run_id),
    batch_id         TEXT NOT NULL,
    status           TEXT NOT NULL,
    n_total          INTEGER NOT NULL,
    n_labeled        INTEGER NOT NULL,
    accuracy         REAL,
    precision_score  REAL,
    recall           REAL,
    f1_score         REAL,
    roc_auc          REAL,
    comparison_json  TEXT,
    metrics_json     TEXT,
    recorded_at      TEXT NOT NULL,
    UNIQUE(run_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id          TEXT PRIMARY KEY,
    dedup_key         TEXT NOT NULL UNIQUE,
    category          TEXT NOT NULL,
    severity          TEXT NOT NULL,
    title             TEXT NOT NULL,
    description       TEXT NOT NULL,
    reason            TEXT,
    feature           TEXT,
    metric_name       TEXT,
    metric_value      REAL,
    threshold_value   REAL,
    batch_id          TEXT NOT NULL,
    run_id            TEXT NOT NULL REFERENCES monitoring_runs(run_id),
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    occurrence_count  INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_category ON alerts(category);
CREATE INDEX IF NOT EXISTS idx_alerts_batch ON alerts(batch_id);
CREATE INDEX IF NOT EXISTS idx_drift_batch ON drift_results(batch_id);
CREATE INDEX IF NOT EXISTS idx_perf_batch ON performance_metrics(batch_id);
"""

# Columns added after the tables above were first created. A fresh database
# gets them directly from _SCHEMA; this list exists only so a monitoring.db
# file created by an earlier version of this module still works, instead of
# failing with "no such column" the first time a dashboard page queries it.
_ADDED_COLUMNS = {
    "drift_results": [("notes_json", "TEXT")],
    "prediction_drift_results": [
        ("batch_label_distribution_json", "TEXT"),
        ("batch_confidence_json", "TEXT"),
        ("notes_json", "TEXT"),
    ],
    "performance_metrics": [("metrics_json", "TEXT")],
}


def _ensure_added_columns(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        present = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, column_type in columns:
            if name not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")


class MonitoringRepository:
    """Saves and retrieves monitoring history for one SQLite database file."""

    def __init__(self, db_path: Path = config.MONITORING_DB_PATH):
        self.db_path = Path(db_path)
        # A missing database file is not an error here: the parent directory
        # and an empty file are created, and the schema below brings it to a
        # usable state. This is what "handle a missing database file safely"
        # means in practice - there is nothing to recover, because there was
        # never anything to lose.
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
                _ensure_added_columns(conn)
        except sqlite3.Error as error:
            raise MonitoringStorageError(
                f"Could not initialise the monitoring database at "
                f"{self.db_path}: {error}"
            ) from error

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # --- runs ----------------------------------------------------------

    def save_run(
        self, run_id: str, batch_id: str, started_at: str, status: str = "running"
    ) -> None:
        """Record that a monitoring run started.

        A duplicate `run_id` (the same run saved twice, for instance after a
        retry) is not an error: the row is simply left as it was, because
        `run_id` is generated once per run and saving it again describes
        nothing new.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO monitoring_runs "
                    "(run_id, batch_id, started_at, status) VALUES (?, ?, ?, ?)",
                    (run_id, batch_id, started_at, status),
                )
        except sqlite3.Error as error:
            raise MonitoringStorageError(f"Could not save run {run_id}: {error}") from error

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE monitoring_runs SET completed_at = ?, status = ? "
                    "WHERE run_id = ?",
                    (_utc_now(), status, run_id),
                )
        except sqlite3.Error as error:
            raise MonitoringStorageError(f"Could not complete run {run_id}: {error}") from error

    # --- batch summaries -------------------------------------------------

    def save_batch_summary(
        self,
        run_id: str,
        prediction: BatchPrediction,
        actual_positive_rate: float | None = None,
        n_labeled: int | None = None,
    ) -> None:
        distribution = prediction.prediction_distribution()
        predicted_positive_rate = (
            distribution[config.POSITIVE_LABEL]["percent"] / 100
            if config.POSITIVE_LABEL in distribution
            else None
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO batch_summaries "
                    "(run_id, batch_id, n_rows, n_labeled, predicted_positive_rate, "
                    " actual_positive_rate, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, prediction.batch_id, prediction.n_rows, n_labeled,
                        predicted_positive_rate, actual_positive_rate, _utc_now(),
                    ),
                )
        except sqlite3.Error as error:
            raise MonitoringStorageError(
                f"Could not save batch summary for {prediction.batch_id}: {error}"
            ) from error

    # --- drift results ----------------------------------------------------

    def save_drift_report(self, run_id: str, report: DriftReport) -> None:
        """One row per feature. Saving the same run twice replaces, not duplicates."""
        now = _utc_now()
        rows = [
            (
                run_id, report.batch_id, r.feature, r.feature_type, r.method,
                r.status, r.psi, r.statistic, r.p_value, r.effect_size,
                r.n_reference, r.n_batch, json.dumps(r.notes) if r.notes else None, now,
            )
            for r in report.feature_results
        ]
        try:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO drift_results "
                    "(run_id, batch_id, feature, feature_type, method, status, psi, "
                    " statistic, p_value, effect_size, n_reference, n_batch, notes_json, "
                    " recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
        except sqlite3.Error as error:
            raise MonitoringStorageError(
                f"Could not save drift results for run {run_id}: {error}"
            ) from error

    def save_prediction_drift_report(
        self, run_id: str, report: PredictionDriftReport
    ) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO prediction_drift_results "
                    "(run_id, batch_id, label_status, label_psi, confidence_status, "
                    " confidence_psi, batch_label_distribution_json, "
                    " batch_confidence_json, notes_json, recorded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, report.batch_id, report.label_status, report.label_psi,
                        report.confidence_status, report.confidence_psi,
                        json.dumps(report.batch_distribution),
                        json.dumps(report.batch_confidence),
                        json.dumps(report.notes) if report.notes else None,
                        _utc_now(),
                    ),
                )
        except sqlite3.Error as error:
            raise MonitoringStorageError(
                f"Could not save prediction-drift results for run {run_id}: {error}"
            ) from error

    # --- performance --------------------------------------------------------

    def save_performance_report(self, run_id: str, report: PerformanceReport) -> None:
        metrics = report.metrics or {}
        comparison_json = json.dumps(report.comparison) if report.comparison else None
        metrics_json = json.dumps(metrics) if metrics else None
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO performance_metrics "
                    "(run_id, batch_id, status, n_total, n_labeled, accuracy, "
                    " precision_score, recall, f1_score, roc_auc, comparison_json, "
                    " metrics_json, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, report.batch_id, report.status, report.n_total,
                        report.n_labeled, metrics.get("accuracy"), metrics.get("precision"),
                        metrics.get("recall"), metrics.get("f1_score"), metrics.get("roc_auc"),
                        comparison_json, metrics_json, _utc_now(),
                    ),
                )
        except sqlite3.Error as error:
            raise MonitoringStorageError(
                f"Could not save performance metrics for run {run_id}: {error}"
            ) from error

    # --- alerts: the deduplicating save -------------------------------------

    def save_alert(self, alert: Alert, run_id: str) -> AlertSaveOutcome:
        """Insert a new alert, or fold it into an existing one with the same dedup_key.

        This is the one method responsible for "prevent repeated identical
        alerts from flooding the system." If an alert with this `dedup_key`
        already exists, its `occurrence_count` is incremented, its
        `last_seen_at` and `batch_id`/`run_id` move forward to this
        occurrence, and its severity is escalated (never silently
        downgraded) - no second row is created. Otherwise a new row is
        created with `occurrence_count = 1`.
        """
        now = _utc_now()
        try:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT alert_id, severity, occurrence_count FROM alerts "
                    "WHERE dedup_key = ?",
                    (alert.dedup_key,),
                ).fetchone()

                if existing is None:
                    conn.execute(
                        "INSERT INTO alerts (alert_id, dedup_key, category, severity, "
                        "title, description, reason, feature, metric_name, metric_value, "
                        "threshold_value, batch_id, run_id, first_seen_at, last_seen_at, "
                        "occurrence_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            str(uuid.uuid4()), alert.dedup_key, alert.category,
                            alert.severity, alert.title, alert.description, alert.reason,
                            alert.feature, alert.metric_name, alert.metric_value,
                            alert.threshold_value, alert.batch_id, run_id, now, now,
                        ),
                    )
                    row = conn.execute(
                        "SELECT alert_id FROM alerts WHERE dedup_key = ?",
                        (alert.dedup_key,),
                    ).fetchone()
                    return AlertSaveOutcome(
                        alert_id=row["alert_id"], dedup_key=alert.dedup_key,
                        action="created", occurrence_count=1, severity=alert.severity,
                    )

                new_severity = escalate(existing["severity"], alert.severity)
                new_count = existing["occurrence_count"] + 1
                conn.execute(
                    "UPDATE alerts SET severity = ?, occurrence_count = ?, "
                    "last_seen_at = ?, batch_id = ?, run_id = ?, metric_value = ? "
                    "WHERE dedup_key = ?",
                    (
                        new_severity, new_count, now, alert.batch_id, run_id,
                        alert.metric_value, alert.dedup_key,
                    ),
                )
                return AlertSaveOutcome(
                    alert_id=existing["alert_id"], dedup_key=alert.dedup_key,
                    action="deduplicated", occurrence_count=new_count,
                    severity=new_severity,
                )
        except sqlite3.Error as error:
            raise MonitoringStorageError(
                f"Could not save alert '{alert.title}': {error}"
            ) from error

    def save_alerts(self, alerts: list[Alert], run_id: str) -> list[AlertSaveOutcome]:
        """Save several alerts, each independently deduplicated."""
        return [self.save_alert(alert, run_id) for alert in alerts]

    # --- retrieval -----------------------------------------------------------

    def get_runs(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM monitoring_runs WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND started_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND started_at <= ?"
            params.append(end_date)
        query += " ORDER BY started_at DESC"
        return self._query(query, params)

    def get_drift_results(
        self, batch_id: str | None = None, feature: str | None = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM drift_results WHERE 1=1"
        params: list = []
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        if feature:
            query += " AND feature = ?"
            params.append(feature)
        query += " ORDER BY recorded_at DESC"
        return self._query(query, params)

    def get_prediction_drift_results(self, batch_id: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM prediction_drift_results WHERE 1=1"
        params: list = []
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        query += " ORDER BY recorded_at DESC"
        return self._query(query, params)

    def get_performance_metrics(self, batch_id: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM performance_metrics WHERE 1=1"
        params: list = []
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        query += " ORDER BY recorded_at DESC"
        return self._query(query, params)

    def get_batch_summaries(self, batch_id: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM batch_summaries WHERE 1=1"
        params: list = []
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        query += " ORDER BY recorded_at DESC"
        return self._query(query, params)

    def get_alerts(
        self,
        batch_id: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve alerts, optionally filtered by batch, type, severity, or date.

        Dates are compared against `last_seen_at`, so a recurring alert is
        found by *when it was last seen*, which is the question "is this
        still an active problem in this period" actually asks.
        """
        query = "SELECT * FROM alerts WHERE 1=1"
        params: list = []
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        if category:
            query += " AND category = ?"
            params.append(category)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if start_date:
            query += " AND last_seen_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND last_seen_at <= ?"
            params.append(end_date)
        query += " ORDER BY last_seen_at DESC"
        return self._query(query, params)

    def _query(self, query: str, params: list) -> pd.DataFrame:
        try:
            with self._connect() as conn:
                return pd.read_sql_query(query, conn, params=params)
        except sqlite3.Error as error:
            raise MonitoringStorageError(f"Query failed: {error}") from error
