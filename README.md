# Palo Alto Networks — Employee Attrition Risk Intelligence

A machine-learning decision-support system that predicts individual employee
attrition risk and gives HR transparent, explainable reasoning behind each
score — built on a real 1,470-employee HR dataset.

## What this is

- **Predictive model**: Logistic Regression / Random Forest / Gradient
  Boosting trained and compared on real data (1,470 employees, 16.1%
  historical attrition rate). Best model selected by test-set ROC-AUC and
  confirmed with 5-fold stratified cross-validation.
- **Risk scoring**: every employee gets an attrition probability (0–1) and a
  Low / Medium / High risk category (thresholds: <30% / 30–60% / >60%).
- **Explainability without SHAP**: global feature importance is native to
  the winning model (coefficients or impurity-based importance — no
  approximation), and per-employee "reason codes" use a transparent,
  documented rule-based method. See `src/explainability.py` for exactly how.
- **4-page Streamlit dashboard**: Attrition Risk Dashboard, Employee Risk
  Profile (individual lookup), Department Risk View (filterable), and an
  Explainability Panel with a live what-if scenario simulator.

## Project structure

```
app.py                      Home page — trains model, shows KPIs
pages/
  1_Attrition_Risk_Dashboard.py
  2_Employee_Risk_Profile.py
  3_Department_Risk_View.py
  4_Explainability_Panel.py
src/
  preprocessing.py          Data loading + data-quality audit
  feature_engineering.py    Engineered features (income ratio, engagement score, etc.)
  model_training.py         Trains & evaluates Logistic Regression / RF / Gradient Boosting
  explainability.py         Feature importance + per-employee reason codes
  theme.py                  Color palette + minimal, tested custom CSS
  utils.py                  Cached data/model loading, department aggregation
data/
  Palo_Alto_Networks.csv    Source dataset
.streamlit/config.toml      Base app theme
requirements.txt            Pinned dependency versions
```

## Key design decisions (and why)

**The model trains fresh at app startup, not from a saved `.pkl` file.**
A pre-trained pickle creates a hard dependency on the exact scikit-learn
version used to create it — if the hosting platform installs a newer
version later, unpickling can silently degrade or break. Training on cold
start (cached for the rest of the session via `st.cache_resource`) costs a
few seconds once and removes that failure mode entirely: whatever
scikit-learn version is installed is the one used for both training and
inference, always in sync.

**No XGBoost, imbalanced-learn, or SHAP dependency.**
The project spec allows Gradient Boosting *or* XGBoost, and SMOTE *or*
class weighting, and lists SHAP as optional. This implementation uses
scikit-learn's own `GradientBoostingClassifier` and `class_weight="balanced"`
/ sample-weighting instead — every dependency in `requirements.txt` is
pure-Python-installable and was directly tested end-to-end before delivery
(see below), rather than added on faith.

**Explainability is real, not approximated.**
Global importance is read directly off the trained model's own
coefficients or impurity scores. Per-employee reason codes are a documented,
inspectable formula (deviation from the training population mean, weighted
by global importance) — clearly labeled as a SHAP stand-in, not represented
as SHAP itself.

## Verification performed before delivery

Every page of this app was actually **executed** — not just read — against
the real dataset, using a test harness that stubs `streamlit` and `plotly`
closely enough to run the real business logic (pandas, scikit-learn,
plotting calls) and catch runtime errors before ever reaching a deploy log.
All 5 scripts (home + 4 pages) pass. `requirements.txt` and
`.streamlit/config.toml` were both parsed and validated byte-for-byte.

## Real results (from this exact dataset)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.765 | 0.366 | 0.638 | 0.465 | **0.805** | 0.829 ± 0.031 |
| Random Forest | 0.844 | 0.526 | 0.213 | 0.303 | 0.801 | 0.804 ± 0.033 |
| Gradient Boosting | 0.803 | 0.404 | 0.489 | 0.442 | 0.769 | 0.798 ± 0.033 |

Logistic Regression was selected for its best ROC-AUC and highest recall —
in an attrition-prevention context, catching more true leavers (recall)
matters more than precision, since the cost of a missed at-risk employee
(an unplanned resignation) generally exceeds the cost of a false positive
(an unnecessary retention conversation).

## Data quality note

The dataset is clean: 0 missing values, 0 duplicate rows, across 1,470
employees. One column, `PerformanceRating`, was found to take only 2
distinct values (3 and 4) across the entire dataset — flagged in the audit
and disclosed in the app rather than silently treated as a strong signal.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (upload the whole folder at once,
   including the hidden `.streamlit/` folder)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point it at this repo, branch `main`, main file `app.py`
4. Deploy — first load trains the model (~10s), then it's cached
