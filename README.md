# ML Model Health Monitor

An educational prototype that trains a classifier on the UCI Adult Census
Income dataset and then monitors it for data-quality problems, feature
drift, and performance degradation over time.

Built as an undergraduate portfolio project. It is **not** a production
system and must not be used to make decisions about real people.

---

## Status: Phase 1 complete — project setup and dataset exploration

Phase 1 establishes the project skeleton and documents what is actually in
the dataset. **No model is trained yet.**

| Phase | Scope | State |
| --- | --- | --- |
| 1 | Project setup, dataset loading and exploration | Done |
| 2+ | Validation, drift detection, monitoring, dashboard | Not started |

---

## Setup

Create and activate a virtual environment, then install the dependencies.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (Git Bash)**

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To leave the virtual environment later, run `deactivate`.

---

## Verify the setup

Run these three commands in order. Expected results are listed beside each.

```bash
python -c "import pandas, numpy, ucimlrepo, tabulate; print('dependencies ok')"
python explore_dataset.py
python -m pytest -q
```

| Command | What it proves | Expected output |
| --- | --- | --- |
| `python -c "import ..."` | Dependencies installed | `dependencies ok` |
| `python explore_dataset.py` | Dataset downloads and is inspectable | A summary table, then `Report written to .../docs/dataset_exploration.md` |
| `python -m pytest -q` | Loading and inspection code works | `16 passed` |

Run `explore_dataset.py` **before** `pytest` the first time. It downloads the
dataset from UCI (this can take a minute) and caches it in `data/`, so later
runs — including the tests — are fast and work offline. If you run the tests
first they will simply do the download themselves. Use
`python explore_dataset.py --refresh` to force a re-download.

---

## Project layout

| Path | Purpose |
| --- | --- |
| `src/monitor/config.py` | Single source of truth for file paths, column names, the target column, and the random seed |
| `src/monitor/data.py` | Downloads the dataset from UCI, caches it locally, and cleans it |
| `src/monitor/explore.py` | Inspection helpers — schema, missing values, duplicates, class distribution, summary statistics |
| `src/monitor/report.py` | Formats the inspection results into the Markdown report |
| `explore_dataset.py` | Entry point: run the inspection and write the report |
| `tests/test_data.py` | Tests for loading and cleaning |
| `tests/test_explore.py` | Tests for the inspection helpers |
| `docs/dataset_exploration.md` | **The Phase 1 deliverable** — findings, generated from real data |
| `data/` | Local dataset cache (git-ignored, regenerated on demand) |
| `pyproject.toml` | pytest configuration (puts `src/` on the import path) |
| `requirements.txt` | Python dependencies |

The two root-level scripts exist so the project can be run with a plain
`python explore_dataset.py`, without needing to install it as a package or
set `PYTHONPATH` by hand.

---

## Dataset

| Field | Value |
| --- | --- |
| Name | Adult (also called Census Income) |
| Source | https://archive.ics.uci.edu/dataset/2/adult |
| UCI id | 2 |
| DOI | 10.24432/C5XW20 |
| Creators | Barry Becker, Ronny Kohavi |
| Accessed via | the `ucimlrepo` package |
| Size | 48,842 rows x 15 columns |

The dataset is **not committed to this repository**. It is downloaded on
first run and cached in `data/`, which keeps the repo small and makes the
data acquisition step reproducible and visible in code.

See [`docs/dataset_exploration.md`](docs/dataset_exploration.md) for the full
findings. The headline points:

- **Class imbalance of about 3.18:1** (76.07% `<=50K`, 23.93% `>50K`). This
  natural distribution is kept everywhere in the project; rebalancing it
  would misrepresent real incoming data.
- **Missing values are encoded inconsistently.** Rows from the original
  training split use the string `"?"`; rows from the test split use real
  empty values. Loading the data naively hides about two thirds of the
  missing values and invents a phantom `"?"` category.
- **Three columns have missing values**: `occupation` (5.75%),
  `workclass` (5.73%), `native_country` (1.75%).
- **52 fully duplicated rows** (0.11%), left in place because anonymous
  census records can legitimately coincide.

### Usage terms

The UCI ML Repository publishes this dataset for public research and
educational use. The `ucimlrepo` metadata response carries no
machine-readable licence field, so the licence shown on the
[dataset page](https://archive.ics.uci.edu/dataset/2/adult) is the
authoritative statement of terms. This project uses the data for
non-commercial educational purposes only and redistributes no part of it.

---

## Limitations

- The data is a 1994 US census extract and does not describe the present day.
- It contains sensitive attributes (`race`, `sex`, `native_country`) and the
  label rate differs sharply across groups. Any model trained on it will
  reproduce those patterns.
- This project demonstrates monitoring techniques. It has not been assessed
  for fairness, and no part of it should inform real decisions about people.

---

## Note on `.deferred/`

An earlier working session produced model training and evaluation code before
the phase plan was set. Rather than delete it, it was moved to `.deferred/`
(git-ignored) and will be restored when the model-training phase begins.
