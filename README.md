# Customer Churn Prediction & Business Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811.svg)](https://powerbi.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end churn analytics project on a 10,000-customer retail banking dataset —
covering data cleaning, exploratory analysis, a Random Forest classification pipeline,
and a five-page interactive Power BI dashboard. The project's core finding is a
**data leakage investigation**: identifying that a feature initially assumed to be a
strong predictor is in fact a near-duplicate of the target variable, and re-deriving
the *actionable* drivers of churn once that leakage is accounted for.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Architecture](#project-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Modeling Approach — Theory](#modeling-approach--theory)
7. [The Data Leakage Finding](#the-data-leakage-finding)
8. [Model Evaluation](#model-evaluation)
9. [Power BI Dashboard](#power-bi-dashboard)
10. [Key Business Insights](#key-business-insights)
11. [Tech Stack](#tech-stack)
12. [How to Run](#how-to-run)
13. [Repository Structure](#repository-structure)
14. [Future Work](#future-work)

---

## Problem Statement

Customer churn — the loss of existing customers — directly erodes recurring revenue and is
significantly more expensive to reverse than to prevent, since acquiring a new customer
typically costs more than retaining an existing one. This project builds a supervised
classification pipeline to predict churn risk at the individual customer level, and pairs
it with a BI layer that translates model output into segment-level, decision-ready insight
for a retention team.

---

## Dataset

**Source:** Customer-Churn-Records.csv — 10,000 rows, 18 columns, retail banking domain.

| Category | Columns |
|---|---|
| Identifiers (dropped) | RowNumber, CustomerId, Surname |
| Numerical continuous | CreditScore, Age, Balance, EstimatedSalary, Point Earned |
| Categorical | Geography, Gender, Card Type |
| Discrete / binary flags | Tenure, NumOfProducts, HasCrCard, IsActiveMember, Complain, Satisfaction Score |
| Target | Exited (1 = churned, 0 = retained) |

Full field-level definitions are in [`docs/data_dictionary.md`](docs/data_dictionary.md).

**Class balance:** 20.38% churned (2,038) vs. 79.62% retained (7,962) — a moderate class
imbalance, handled via stratified train/test splitting rather than resampling, to preserve
the true population distribution for evaluation.

---

## Project Architecture

```mermaid
flowchart LR
    A[Raw CSV<br/>10,000 rows x 18 cols] --> B[Data Cleaning<br/>Pandas/NumPy]
    B --> C[Outlier Detection<br/>Z-score, |Z| > 3]
    C --> D[Feature Engineering<br/>Age Binning]
    D --> E[EDA & Visualization<br/>Plotly, Seaborn]
    D --> F[Preprocessing<br/>StandardScaler + OneHotEncoder]
    F --> G[Random Forest Classifier<br/>300 trees, stratified split]
    G --> H[Evaluation<br/>Confusion Matrix, ROC-AUC, F1]
    G --> I[Feature Importance<br/>Extraction]
    E --> J[cleaned_churn_data.csv]
    I --> K[feature_importance.csv]
    J --> L[Power BI Data Model<br/>DAX Measures]
    K --> L
    L --> M[5-Page Interactive Dashboard]
```

---

## Data Pipeline

The pipeline runs end-to-end in [`notebooks/CustomerChurn.ipynb`](notebooks/CustomerChurn.ipynb):

1. **Structural cleaning** — drop identifier columns (`RowNumber`, `CustomerId`, `Surname`)
   that carry no predictive signal and would only introduce noise or, worse, leak row-order
   artifacts into the model.
2. **Integrity checks** — verify zero duplicate rows and zero null values across all 18 columns.
3. **Text normalization** — standardize categorical string fields (`.str.strip().str.upper()`)
   to prevent silent encoding bugs from trailing whitespace or inconsistent casing.
4. **Outlier detection** — compute Z-scores on `Balance`, `EstimatedSalary`, and `CreditScore`;
   flag (not drop) rows where `|Z| > 3`, since extreme values in a churn context are frequently
   *meaningful* signal rather than noise.
5. **Feature engineering** — bin `Age` into six generational cohorts (18-25 through 65+) for
   interpretable segmentation in the BI layer.
6. **Export** — write the cleaned dataset to `cleaned_churn_data.csv` for both modeling and
   direct Power BI ingestion.

---

## Exploratory Data Analysis

Key visualizations produced in the notebook (see `images/`):

- **Correlation heatmap** across all numeric features
- **Age distribution histogram**, overlaid by churn status
- **Categorical composition charts** — Card Type and Geography distributions
- **Satisfaction Score vs. churn** stacked breakdown

These visuals informed which features to prioritize in both the model and the dashboard —
notably, they surfaced that **Satisfaction Score shows almost no variation in churn rate**
(19.6%–21.8% across all five score levels), an early hint that self-reported satisfaction is
a weaker signal than assumed.

---

## Modeling Approach — Theory

### Preprocessing

- **`StandardScaler`** on numeric features: transforms each feature to zero mean, unit
  variance via `z = (x - μ) / σ`. Tree-based models like Random Forest don't strictly
  require feature scaling (splits are based on ordering, not magnitude), but scaling is
  retained here for pipeline consistency and to keep the preprocessing generalizable if the
  model is later swapped for a distance- or gradient-based algorithm.
- **`OneHotEncoder(drop='first')`** on categorical features (`Geography`, `Gender`,
  `Card Type`): converts each category into binary indicator columns, dropping one level
  per feature to avoid the **dummy variable trap** (perfect multicollinearity between
  encoded columns).

### Train/Test Split

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

Stratifying on `y` (the target) ensures the ~20.4% churn ratio is preserved identically in
both the training and test sets — without this, a random split risks producing a test set
with a meaningfully different class balance, which would distort evaluation metrics.

### Algorithm: Random Forest Classifier

**Why Random Forest:**
- An **ensemble of decision trees**, each trained on a bootstrap-sampled subset of the
  training data (bagging), with each split considering only a random subset of features.
- This dual randomness (row sampling + feature sampling) decorrelates individual trees,
  reducing variance compared to a single decision tree while retaining low bias.
- Naturally handles a **mix of continuous and categorical (encoded) features** without
  assuming linear relationships between predictors and the target.
- Robust to outliers and monotonic feature transformations, since splits are based on
  relative ordering, not absolute scale.
- Provides a built-in, interpretable **feature importance** ranking
  (`.feature_importances_`), computed as the mean decrease in node impurity (Gini
  importance) attributable to each feature, averaged across all trees.

**Configuration used:** `n_estimators=300`, `max_depth=None` (trees grown to full depth,
regularized implicitly via the bagging ensemble rather than pruning), `random_state=42`
for reproducibility.

---

## The Data Leakage Finding

This is the central analytical contribution of the project.

Initial feature importance results showed `Complain` accounting for **82.6%** of total
model importance — vastly exceeding every other feature combined. Rather than reporting
this at face value, it was investigated further:

| Complain = 0 (no complaint) | Complain = 1 (filed complaint) |
|---|---|
| Churn rate: **0.05%** (4 of 7,956) | Churn rate: **99.51%** (2,034 of 2,044) |

This near-total separation (a crosstab that is almost perfectly diagonal) indicates that
`Complain` is not an independent predictive signal — it is **functionally a duplicate
encoding of the target variable**, most plausibly because the complaint flag is recorded
at or after the point of churn, rather than as a leading indicator.

**Conclusion:** any model or dashboard reporting `Complain` as the "top churn driver"
without this context is presenting a data artifact as a business insight. The corrected,
leakage-aware view is documented in [Key Business Insights](#key-business-insights) below.

---

## Model Evaluation

> **Note:** the metrics below reflect the model *including* `Complain` as a feature, and
> are consequently inflated by the leakage described above — they are reported here for
> completeness and reproducibility, not as a validated production benchmark.

| Metric | Value |
|---|---|
| Accuracy | 99.85% |
| ROC-AUC | 0.9995 |
| Precision (Churned class) | 1.00 |
| Recall (Churned class) | 1.00 |
| F1-score (Churned class) | 1.00 |
| False Negatives | 2 (out of 2,000 test rows) |
| False Positives | 1 (out of 2,000 test rows) |

**Confusion Matrix** (test set, n=2,000):

| | Predicted Retained | Predicted Churned |
|---|---|---|
| **Actual Retained** | 1,591 | 1 |
| **Actual Churned** | 2 | 406 |

See `images/confusion_matrix.png` and `images/roc_curve.png` for the full diagnostic plots.

**Honest interpretation:** near-perfect accuracy on a real-world churn problem is a red
flag, not a success — it is a direct consequence of `Complain` leaking the target. A
model retrained without `Complain` would be expected to show substantially lower but far
more realistic and *actionable* performance, since it would be forced to rely on genuine
leading indicators (age, product count, activity status) instead of a proxy for the
outcome itself.

---

## Power BI Dashboard

The `.pbix` file (`dashboard/CustomerChurn.pbix`) contains a five-page interactive report:

1. **Executive Overview** — KPI cards (churn rate, balance at risk, complaint rate), a
   geographic churn map, and an age-cohort churn breakdown.
2. **Behavioral & Product Drivers** — churn rate by product count, tenure, activity status,
   and satisfaction score.
3. **Complaint & Model Leakage Analysis** — a dedicated page isolating and explaining the
   `Complain`–`Exited` overlap, including a leakage-adjusted churn rate measure.
4. **Model Insights** — imported feature importance charts (with and without `Complain`),
   plus the confusion matrix and ROC curve from the notebook.
5. **Financial Risk Segmentation** — balance-at-risk breakdowns by geography, age, and card
   tier.

Key DAX measures include `Churn Rate`, `Balance at Risk`, `Complaint Rate`, and a custom
leakage-isolating measure, `Churn Rate (Non-Complainers Only)`, that recomputes churn rate
after excluding the leaky signal.

Screenshots of each page are in `images/dashboard_page1_overview.png` through
`images/dashboard_page5_financial_risk.png`.

---

## Key Business Insights

- **Germany churns at 32.44%**, roughly double France (16.17%) and Spain (16.67%),
  despite a comparable customer base — the single largest geographic risk concentration.
- **Product overexposure is a strong genuine driver:** churn rate is 7.60% at 2 products
  but climbs to 82.71% at 3 products and 100% at 4 products (n=326 combined — a small but
  near-certain-attrition segment).
- **Inactive members churn at 26.87%** vs. 14.27% for active members — engagement status
  is a legitimate, actionable lever.
- **The 46–55 and 56–65 age cohorts churn at 50.57% and 48.32%** respectively, far above
  the 7.47%–8.50% churn rate seen in customers under 35 — the highest-value retention
  targeting segment by both rate and average balance.
- **Satisfaction Score and Card Type show negligible predictive value** (churn rate varies
  by less than 2 percentage points across all levels of each) — resources aimed at these
  dimensions are unlikely to move retention outcomes.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | Python, Pandas, NumPy |
| Statistical computing | SciPy |
| Visualization (EDA) | Matplotlib, Seaborn, Plotly Express |
| Machine learning | scikit-learn (RandomForestClassifier, ColumnTransformer, Pipeline) |
| BI / Dashboarding | Power BI Desktop, DAX |
| Environment | Google Colab |

---

## How to Run

### Notebook
```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction-bi.git
cd customer-churn-prediction-bi
pip install -r requirements.txt
jupyter notebook notebooks/CustomerChurn.ipynb
```
Or open directly in Google Colab via **File → Upload notebook** and upload
`notebooks/CustomerChurn.ipynb`, then upload `data/raw/Customer-Churn-Records.csv` when
prompted in the first cell.

### Dashboard
Open `dashboard/CustomerChurn.pbix` in Power BI Desktop (free download from
[powerbi.microsoft.com](https://powerbi.microsoft.com/)).

---

## Repository Structure

```
customer-churn-prediction-bi/
├── data/
│   ├── raw/Customer-Churn-Records.csv
│   └── processed/{cleaned_churn_data.csv, feature_importance.csv}
├── notebooks/CustomerChurn.ipynb
├── dashboard/CustomerChurn.pbix
├── images/               # exported diagnostic plots + dashboard screenshots
├── docs/data_dictionary.md
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Future Work

- Retrain the model **excluding `Complain`** as a dedicated leakage-free baseline, and
  report its standalone metrics rather than only a filtered feature-importance chart.
- Add **SHAP value analysis** for individual-level, model-agnostic explanation of churn
  risk, extending beyond global feature importance.
- Experiment with **class-balancing techniques** (SMOTE, class weighting) and compare
  against the current stratified-split baseline.
- Deploy the leakage-free model behind a lightweight **FastAPI** endpoint for real-time
  scoring, consistent with the architecture pattern used in this author's other projects
  (see [SENTINAL](#) and [VERITAS](#) repositories).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
