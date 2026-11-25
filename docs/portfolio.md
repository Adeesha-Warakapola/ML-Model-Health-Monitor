# Portfolio Materials

Reference text for publishing this project - a GitHub description and
topics, a LinkedIn announcement, and a short technical-interview briefing.
All of it describes only what is actually built and tested in this
repository; nothing here claims production-readiness or invents a result.

## GitHub repository description

Use as the one-line description field (GitHub repo settings → "About"):

> Educational ML monitoring system: trains a classifier on the UCI Adult
> dataset, then detects data drift, prediction drift, and performance
> degradation with a Streamlit + Plotly dashboard and SQLite-backed alert
> history. 446 tests.

(156 characters - fits GitHub's "About" field without truncation on most
screens.)

## Suggested topics / tags

Lowercase, hyphenated, matching GitHub's topic conventions:

```
machine-learning
mlops
model-monitoring
data-drift
concept-drift
model-performance
streamlit
plotly
scikit-learn
python
sqlite
pandas
dashboard
data-science
portfolio-project
educational-project
pytest
```

Pick the dozen or so most relevant if GitHub's UI limits how many display
well; `model-monitoring`, `data-drift`, `mlops`, `streamlit`, and
`scikit-learn` are the ones most likely to surface the repo to the right
audience.

## LinkedIn announcement

> **Built an end-to-end ML monitoring system — from a trained model to a live dashboard tracking its health.**
>
> Most ML tutorials stop at "train a model, report accuracy." The harder
> and more realistic problem is what happens *after* deployment — does the
> incoming data still look like what the model was trained on? Is the model
> still saying the same kinds of things? And once real outcomes are known,
> is it still actually right?
>
> I built a project that answers all three, end to end, on the UCI Adult
> Census Income dataset:
>
> 🔹 **A real baseline model** — logistic regression, 5-fold cross-validated,
> a decision threshold chosen deliberately (not the default 0.5) to trade
> precision for recall on an imbalanced 76/24 class split.
>
> 🔹 **Statistical drift detection** — Kolmogorov-Smirnov and chi-square
> tests plus the Population Stability Index (PSI) for every feature,
> separately from *prediction* drift (has what the model is predicting
> changed?) and from *performance* degradation (is it still accurate, once
> true labels arrive?). These three measured completely differently on the
> same deliberately-shifted batch in my testing — which is exactly why a
> real monitoring system needs all three, not just one.
>
> 🔹 **An alerting layer with real deduplication** — INFO/WARNING/CRITICAL
> severities, plain-language descriptions, and a dedup rule that collapses
> a recurring problem into one updating row instead of flooding the system
> — caught and fixed a real bug in that logic along the way, which taught
> me more about testing alerting systems than getting it right the first
> time would have.
>
> 🔹 **A Streamlit + Plotly dashboard** — seven pages reading from the same
> SQLite history the command-line tool writes to, so the dashboard can
> never show something the underlying pipeline didn't actually compute.
>
> 🔹 **446 automated tests**, including end-to-end integration tests that
> run the real pipeline against the real model and found (and fixed) four
> genuine bugs that only showed up once every piece ran together — a
> useful reminder that unit tests passing is not the same as the system
> working.
>
> This is an educational/portfolio project, not a production system — the
> dataset is a 1994 census extract and nothing here has been assessed for
> fairness or safety for real decisions about people. But the monitoring
> *techniques* — drift detection, delayed-label performance tracking,
> alert deduplication, restart-safe persistence — are the same ones a
> production MLOps system needs.
>
> Code: [repository link]
>
> #MachineLearning #MLOps #DataScience #Python #ModelMonitoring

## Technical interview briefing

Short talking points for each area an interviewer is likely to probe.

### Architecture

"It's split into two lifecycles that never share a code path: training
runs once, offline, and saves a single artifact bundling preprocessing,
the classifier, and the decision threshold together. Monitoring runs
repeatedly, loads that artifact, and never changes it. Each monitoring
cycle goes through validation, inference, data-drift detection,
prediction-drift detection, and — once labels exist — performance scoring,
then alert generation, then persistence to SQLite. The CLI tool and the
Streamlit dashboard both call the exact same orchestration function, so
they can't drift apart in behavior."

### The model

"A logistic regression baseline, chosen deliberately over a more complex
model first — the project also trains a Random Forest for comparison, and
the two score close enough that the simpler, more interpretable model was
the right call. Five-fold cross-validation gave ROC-AUC 0.9067 ± 0.0013, so
the result is stable, not a lucky split. The decision threshold was tuned
to 0.29 — lower than the default 0.5 — chosen from out-of-fold training
predictions only, to deliberately trade precision for recall, since this
is an imbalanced problem (about 76% negative class) where the minority
class is the one worth finding."

### Preprocessing

"Bundled inside the same scikit-learn `Pipeline` as the classifier, fitted
once on the training split only. That matters for monitoring specifically:
a batch arriving later is only ever *transformed*, never re-fit — refitting
on new data would scale away the exact distribution shift the monitor
exists to catch. Unknown categories encode as all-zero rather than
crashing, so any batch, however unusual, can still be scored and then
separately flagged."

### Drift detection

"Two different statistical tests depending on the feature type — KS for
numeric, chi-square for categorical — but the actual verdict comes from
the Population Stability Index, not the p-value. With tens of thousands of
reference rows, a p-value gets vanishingly small for even a trivial shift;
PSI doesn't inflate with sample size the same way, so it's the more honest
signal for 'does this actually matter.' I also measured — not assumed —
the batch-size floor below which PSI gives false alarms on *unmodified*
data: below about 400 rows with 10 bins, ordinary sampling noise crosses
the drift threshold most of the time."

### Metrics

"Accuracy alone is misleading on this dataset — always predicting the
majority class already scores 76%. So the project tracks precision,
recall, F1, and ROC-AUC together, and keeps a fixed Phase-3 baseline to
compare every later batch against, rather than comparing against whatever
the previous batch happened to show, which would let the comparison point
silently drift over time."

### Alert system

"Each alert is generated by reading already-computed results — nothing
recomputes a statistic at alert time. The one rule I'd highlight: a
recurring problem (like a column with a stable 5% missing rate) updates
one existing row's occurrence count instead of creating a new alert every
cycle. I actually shipped that wrong the first time — the dedup key
originally included severity, so a problem escalating from WARNING to
CRITICAL created a second, unrelated-looking row instead of updating the
first. A test caught it, and fixing it taught me to think harder about
what 'the same problem' should mean independent of how urgent it looks
*this* time."

### Limitations (be upfront about these without being asked)

"It's an educational prototype on a 1994 census dataset — not current,
contains sensitive attributes, and has a label rate that differs sharply
by group, which the model reproduces. It hasn't been assessed for fairness
and shouldn't inform real decisions. The drift thresholds are measured for
this specific dataset and feature set, not universal constants. And
critically — drift alone doesn't mean the model is wrong; it means the
input changed, which is a reason to look, not a verdict. The project keeps
those three questions (did the input change / did the output change / is
it still accurate) deliberately separate for exactly that reason."
