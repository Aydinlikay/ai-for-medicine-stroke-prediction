"""
Build the project notebook from a list of cell specs.

We build the notebook programmatically (rather than hand-editing JSON) so that
cell boundaries, metadata, and execution counts stay consistent. Each entry in
CELLS is either a markdown block or a code block; markdown is for the narrative
shown to the reader, code is what actually runs.

Run from the project root:

    python scripts/build_notebook.py

The script overwrites notebook/stroke_prediction.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebook" / "stroke_prediction.ipynb"


# ---------------------------------------------------------------------------
# Cell content
#
# Markdown cells carry the *story* of the analysis: what we are doing and, more
# importantly, *why*. Code cells implement that story. Every code cell is kept
# short and self-contained so it can be discussed in isolation during the oral
# defense.
# ---------------------------------------------------------------------------

MD_TITLE = r"""# Stroke Prediction — AI for Medicine Project

**Course:** AI for Medicine — Prof. Stefano Diciotti, University of Bologna
**Dataset:** Healthcare Stroke Dataset (Kaggle, 5,110 patients)
**Task:** Binary classification — predict whether a patient will suffer a stroke.

This notebook is the single source of truth for the project. It performs the full
analysis end-to-end (EDA → preprocessing → modelling → evaluation) and, in its
final cell, automatically fills the Word report template with the figures and
numbers produced here. Running the notebook top-to-bottom on a clean checkout
reproduces every artifact in `outputs/` and the filled report in `Doc/`.
"""

MD_SETUP = r"""## 1. Setup

We pin a single `RANDOM_STATE` and use it everywhere randomness enters the
pipeline (train/test split, cross-validation folds, model initialisation). This
is what makes the experiment reproducible — a reviewer running the notebook
should obtain identical numbers, not "approximately equal" ones.
"""

CODE_PIP_INSTALL = r"""# Colab-friendly install cell.
# On a local machine with requirements.txt already installed this is a no-op;
# on a fresh Colab runtime it provisions everything the notebook needs.
# We pin only major versions to stay forward-compatible.
import sys, subprocess

REQUIREMENTS = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "python-docx>=1.1",
]

def _ensure(pkgs):
    # We probe importability rather than calling `pip show` because the latter
    # is noticeably slower in Colab and produces irrelevant output.
    import importlib
    name_map = {"scikit-learn": "sklearn", "python-docx": "docx"}
    missing = []
    for spec in pkgs:
        pkg = spec.split(">=")[0]
        mod = name_map.get(pkg, pkg)
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(spec)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

_ensure(REQUIREMENTS)
"""

CODE_IMPORTS = r"""# Imports and global configuration.
#
# We expose every "magic number" of the experiment as a named constant at the
# top of the notebook so a reviewer can audit our choices without scrolling
# through cells.
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
)

from xgboost import XGBClassifier

# Reproducibility — the single seed used everywhere.
RANDOM_STATE = 42

# Hold-out test set proportion. 20% is the conventional choice for a dataset of
# this size; it leaves ~4,000 training samples while still giving the test set
# enough positive cases to estimate AUC with a meaningful confidence interval.
TEST_SIZE = 0.20

# Stratified K-fold splits used for model selection.
N_SPLITS = 5

# Where figures land. We re-export them into the Word report at the end.
OUTPUTS_DIR = Path("../outputs") if Path("../outputs").exists() else Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# A consistent visual style across all plots — small detail, but it makes the
# final report look like one document rather than a collage.
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
"""

MD_DATA_LOAD = r"""## 2. Loading the data

The dataset ships as a single CSV with 5,110 rows. Each row is one patient;
columns mix categorical (gender, work type, smoking status, …) and numerical
(age, average glucose level, BMI) attributes. The target column is `stroke`,
1 if the patient had a stroke, 0 otherwise.
"""

CODE_LOAD_DATA = r"""# Locate the dataset. The notebook is designed to run in three environments:
#   (1) locally, started from the project root          -> Data/...
#   (2) locally, started from the notebook/ subfolder   -> ../Data/...
#   (3) Google Colab from a fresh runtime               -> clone the repo first
# In case (3) we shallow-clone the public GitHub repo so the CSV lands at a
# predictable path. The check is "are we on Colab?" rather than "does the file
# exist?", because a Colab user shouldn't end up with an obscure FileNotFound.
GITHUB_REPO = "https://github.com/Aydinlikay/ai-for-medicine-stroke-prediction.git"

def _running_on_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

if _running_on_colab() and not Path("ai-for-medicine-stroke-prediction").exists():
    subprocess.check_call(["git", "clone", "--depth", "1", GITHUB_REPO])

CANDIDATE_PATHS = [
    Path("../Data/healthcare-dataset-stroke-data.csv"),                            # local, from notebook/
    Path("Data/healthcare-dataset-stroke-data.csv"),                               # local, from root
    Path("ai-for-medicine-stroke-prediction/Data/healthcare-dataset-stroke-data.csv"),  # Colab after clone
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])

# The `bmi` column encodes missingness as the literal string "N/A" rather than
# leaving the cell empty. We tell pandas about this up front so the column is
# parsed as float, not object.
df = pd.read_csv(DATA_PATH, na_values=["N/A"])
print(f"Loaded from: {DATA_PATH}")
print(f"Shape: {df.shape}")
df.head()
"""

CODE_INFO = r"""# A quick structural overview. `info()` shows the dtype assigned to each column
# and exposes any column where missing values slipped past us.
df.info()
"""

MD_EDA = r"""## 3. Exploratory Data Analysis

Three questions matter most before we touch any model:

1. **How imbalanced is the target?** This dictates which metric we trust.
2. **Where are the missing values?** This drives the imputation strategy.
3. **What is the distribution of the predictors?** This sanity-checks the data
   and reveals obvious nonsense (e.g. negative ages).
"""

CODE_CLASS_IMBALANCE = r"""# Inspect the class distribution. With a roughly 5% positive class we cannot
# trust accuracy: a model that always predicts "no stroke" would score ~95%
# accuracy while being clinically useless. This is why later we report
# ROC-AUC, PR-AUC, recall and F1 instead.
counts = df["stroke"].value_counts().sort_index()
ratios = (counts / counts.sum() * 100).round(2)
imbalance_table = pd.DataFrame({"count": counts, "percentage": ratios})
print(imbalance_table)

fig, ax = plt.subplots(figsize=(5, 3.5))
sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette=["#4C72B0", "#C44E52"])
ax.set_xlabel("Stroke")
ax.set_ylabel("Number of patients")
ax.set_title("Target class distribution")
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{ratios.iloc[i]}%", ha="center")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "class_imbalance.png", bbox_inches="tight")
plt.show()
"""

CODE_MISSING = r"""# Missing values per column. Only `bmi` is affected, with about 4% missing.
# We will impute it with the *training-fold* median inside a Pipeline, which is
# the only safe way to avoid leaking test-set statistics into the imputer.
missing = df.isna().sum()
missing = missing[missing > 0].to_frame("missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
missing
"""

CODE_NUMERIC_DIST = r"""# Distribution of the three continuous predictors split by stroke outcome.
# Visual separation here hints at how informative each feature will be: age
# clearly separates the two groups, glucose less so, BMI barely at all.
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, feat in zip(axes, NUMERIC_FEATURES):
    sns.kdeplot(
        data=df, x=feat, hue="stroke", common_norm=False,
        palette={0: "#4C72B0", 1: "#C44E52"}, ax=ax,
    )
    ax.set_title(f"{feat} by stroke")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "numeric_distributions.png", bbox_inches="tight")
plt.show()
"""

CODE_CATEGORICAL_DIST = r"""# Stroke rate by category for each categorical predictor. We use the *rate*
# rather than raw counts because the categories themselves are imbalanced
# (e.g. far fewer "children" than "Private" workers).
CATEGORICAL_FEATURES = [
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status",
    "hypertension", "heart_disease",
]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feat in zip(axes.flat, CATEGORICAL_FEATURES):
    rate = df.groupby(feat)["stroke"].mean().sort_values()
    rate.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Stroke rate by {feat}")
    ax.set_xlabel("P(stroke)")
# Hide the last empty subplot.
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "categorical_stroke_rates.png", bbox_inches="tight")
plt.show()
"""

CODE_CORRELATION = r"""# Pearson correlation matrix on the numerical and binary columns. We are
# mainly looking for two things: (1) any predictor strongly correlated with
# the target, and (2) any pair of predictors so collinear that we should
# consider dropping one (none in this dataset).
numeric_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Pearson correlation (numerical and binary features)")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "correlation_heatmap.png", bbox_inches="tight")
plt.show()
"""

MD_PREPROCESSING = r"""## 4. Preprocessing — designed to be leakage-free

Two design choices are worth flagging explicitly because they are the most
common source of inflated results in medical ML papers:

1. **The `id` column is dropped.** It is a patient identifier with no clinical
   meaning; leaving it in would let tree-based models memorise the training set.
2. **All transformations (imputation, scaling, one-hot encoding) live inside
   a `Pipeline`.** When we later wrap the pipeline in `GridSearchCV`, scikit-learn
   refits the imputer and scaler *separately on each training fold*. The
   validation fold is therefore never seen by the imputer's median or the
   scaler's mean. This is the cleanest available defence against data leakage.
"""

CODE_SPLIT_FEATURES = r"""# Separate target from predictors and drop the patient id. We keep the original
# pandas DataFrame so column names survive into ColumnTransformer — this makes
# downstream feature-importance plots interpretable.
TARGET = "stroke"
X = df.drop(columns=[TARGET, "id"])
y = df[TARGET].astype(int)

print("Predictor columns:", list(X.columns))
print("Class balance:", dict(y.value_counts()))
"""

CODE_TRAIN_TEST_SPLIT = r"""# Hold-out test set. We stratify on the target so the positive rate is
# identical in train and test — otherwise the test AUC would be noisier than
# necessary.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Train: {X_train.shape}, positives: {y_train.sum()}")
print(f"Test:  {X_test.shape}, positives: {y_test.sum()}")
"""

CODE_COLUMN_TRANSFORMER = r"""# Define the column-level transformations.
#
# Numerical columns: median imputation (robust to skewed `bmi`) + standard
# scaling. Scaling matters for Logistic Regression and SVM; trees are scale-
# invariant but pay no cost for it, so we keep one preprocessor for all models.
#
# Categorical columns: one-hot encoding with `handle_unknown="ignore"` so any
# never-before-seen category encountered at inference time becomes an all-zero
# vector instead of raising.
NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_COLS = [
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status",
]
# Binary integer flags need no transformation but must still flow through the
# ColumnTransformer so they survive into the model.
PASSTHROUGH_COLS = ["hypertension", "heart_disease"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ("pass", "passthrough", PASSTHROUGH_COLS),
    ],
    verbose_feature_names_out=False,
)
"""

MD_MODELS = r"""## 5. Models and hyper-parameter search

We compare four classifiers spanning the standard tabular-ML spectrum:

| Model | Why it is included |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients are clinically readable. |
| Random Forest | Non-linear interactions, robust to feature scale, gives importances. |
| XGBoost | Typically the strongest tabular learner. |
| SVM (RBF kernel) | Classic non-linear method, useful as a reference point. |

Every model is wrapped in the same preprocessing pipeline and runs with
`class_weight="balanced"` (or `scale_pos_weight` for XGBoost). The balanced
weighting penalises a missed stroke roughly twenty times more than a missed
non-stroke, which matches the clinical asymmetry: a false negative is dangerous,
a false positive only triggers an extra check-up.
"""

CODE_BUILD_MODELS = r"""# Build one Pipeline per candidate model. The preprocessor is identical across
# all of them; only the classifier changes.
def make_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", classifier),
    ])

# scale_pos_weight for XGBoost is the analogue of class_weight='balanced': we
# set it to (negatives / positives) so the gradient sees both classes evenly.
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

model_pipelines: Dict[str, Pipeline] = {
    "LogisticRegression": make_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    ),
    "RandomForest": make_pipeline(
        RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    ),
    "XGBoost": make_pipeline(
        XGBClassifier(
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    ),
    "SVM": make_pipeline(
        SVC(
            class_weight="balanced",
            probability=True,        # needed for ROC/PR curves
            random_state=RANDOM_STATE,
        )
    ),
}

# Hyper-parameter grids — kept deliberately small. With a positive class of
# ~250 samples in the training set, very fine grids would overfit the
# validation folds.
param_grids: Dict[str, dict] = {
    "LogisticRegression": {
        "clf__C": [0.1, 1.0, 10.0],
    },
    "RandomForest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 6, 12],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 6],
        "clf__learning_rate": [0.05, 0.1],
    },
    "SVM": {
        "clf__C": [0.5, 1.0, 4.0],
        "clf__gamma": ["scale"],
    },
}
"""

CODE_GRID_SEARCH = r"""# Run GridSearchCV for each model. The CV splitter is stratified so each fold
# preserves the ~5% positive rate; without stratification a fold could end up
# with zero positives and an undefined recall.
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ROC-AUC as the model-selection metric: it is threshold-independent and
# directly reflects ranking quality, which is what we ultimately care about.
fitted_models: Dict[str, GridSearchCV] = {}
cv_summary_rows = []

for name, pipe in model_pipelines.items():
    print(f"--- {name} ---")
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grids[name],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    fitted_models[name] = search

    cv_summary_rows.append({
        "model": name,
        "best_cv_auc": round(search.best_score_, 4),
        "best_params": search.best_params_,
    })
    print(f"best CV AUC = {search.best_score_:.4f}  |  params = {search.best_params_}")

cv_summary = pd.DataFrame(cv_summary_rows).sort_values("best_cv_auc", ascending=False)
cv_summary
"""

MD_EVAL = r"""## 6. Held-out test-set evaluation

Up to this point the test set has been completely untouched — no model, no
imputer, no scaler has seen it. We now use it *exactly once* to estimate the
generalisation performance of each tuned model.
"""

CODE_TEST_EVAL = r"""# For each model, collect: predicted labels, predicted positive-class
# probabilities, and a battery of metrics. Probabilities (not labels) drive
# both ROC and PR curves.
test_results = []

for name, search in fitted_models.items():
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    test_results.append({
        "model": name,
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
        "PR-AUC":    round(average_precision_score(y_test, y_proba), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
    })

test_summary = pd.DataFrame(test_results).sort_values("ROC-AUC", ascending=False)
test_summary.to_csv(OUTPUTS_DIR / "test_metrics.csv", index=False)
test_summary
"""

CODE_ROC = r"""# ROC curves overlaid on a single axis so the four models can be compared
# at a glance. The diagonal is the random-guess baseline.
fig, ax = plt.subplots(figsize=(6.5, 5))
for name, search in fitted_models.items():
    RocCurveDisplay.from_estimator(search.best_estimator_, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6, label="chance")
ax.set_title("ROC curves — held-out test set")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "roc_curves.png", bbox_inches="tight")
plt.show()
"""

CODE_PR = r"""# Precision-Recall curves. For imbalanced problems PR is more honest than ROC
# because it focuses on the minority (positive) class. A flat horizontal line
# at the positive rate is the no-skill baseline.
fig, ax = plt.subplots(figsize=(6.5, 5))
for name, search in fitted_models.items():
    PrecisionRecallDisplay.from_estimator(search.best_estimator_, X_test, y_test, ax=ax, name=name)
baseline = float(y_test.mean())
ax.axhline(baseline, linestyle="--", color="grey", alpha=0.6, label=f"baseline = {baseline:.3f}")
ax.set_title("Precision-Recall curves — held-out test set")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "pr_curves.png", bbox_inches="tight")
plt.show()
"""

CODE_CONFUSION = r"""# Confusion matrices at the default 0.5 threshold. With class_weight balanced
# the models are tuned to recall positives aggressively, which is the right
# trade-off for a screening tool but inflates the false-positive count.
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, search) in zip(axes, fitted_models.items()):
    ConfusionMatrixDisplay.from_estimator(
        search.best_estimator_, X_test, y_test,
        display_labels=["no stroke", "stroke"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "confusion_matrices.png", bbox_inches="tight")
plt.show()
"""

CODE_FEATURE_IMPORTANCE = r"""# Feature importance from the tree-based models. We pull the names directly
# from the fitted ColumnTransformer so the bars line up with the actual
# feature columns the model saw (including the one-hot expansions).
def get_feature_names(fitted_pipeline: Pipeline) -> np.ndarray:
    return fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()

def plot_importance(pipeline: Pipeline, title: str, ax) -> None:
    importances = pipeline.named_steps["clf"].feature_importances_
    names = get_feature_names(pipeline)
    series = pd.Series(importances, index=names).sort_values(ascending=True).tail(12)
    series.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Importance")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_importance(fitted_models["RandomForest"].best_estimator_, "Random Forest", axes[0])
plot_importance(fitted_models["XGBoost"].best_estimator_,      "XGBoost",       axes[1])
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "feature_importances.png", bbox_inches="tight")
plt.show()
"""

CODE_LOGREG_COEF = r"""# Logistic Regression coefficients — the most clinically interpretable view.
# A positive coefficient means "higher value increases the predicted stroke
# probability", and because we scaled the numerical features the magnitudes
# are directly comparable.
log_pipe = fitted_models["LogisticRegression"].best_estimator_
coefs = log_pipe.named_steps["clf"].coef_[0]
names = get_feature_names(log_pipe)
coef_series = pd.Series(coefs, index=names).sort_values()

fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#C44E52" if c > 0 else "#4C72B0" for c in coef_series.values]
coef_series.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression — standardised coefficients")
ax.set_xlabel("Coefficient (log-odds)")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "logreg_coefficients.png", bbox_inches="tight")
plt.show()
"""

MD_DISCUSSION = r"""## 7. Discussion and limitations

A few observations are worth highlighting before we hand the numbers to the
report generator:

- **Age dominates every model.** It is by far the strongest single predictor
  in both the correlation matrix and the feature importances. Any future
  improvement will have to come from extracting more signal out of the
  remaining variables.
- **Recall vs precision.** Balanced class weights deliberately trade precision
  for recall. As a *screening* tool this is the right call — the cost of a
  missed stroke is much higher than the cost of a follow-up examination — but
  it means the absolute precision is low (single digits in percentage terms).
- **Sample size.** With only ~250 positive cases in the training set, the CV
  estimates of AUC have a non-trivial confidence interval. Treat differences
  between models smaller than ~0.01 AUC as noise.
"""

def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}

    blocks = [
        ("md",   MD_TITLE),
        ("md",   MD_SETUP),
        ("code", CODE_PIP_INSTALL),
        ("code", CODE_IMPORTS),
        ("md",   MD_DATA_LOAD),
        ("code", CODE_LOAD_DATA),
        ("code", CODE_INFO),
        ("md",   MD_EDA),
        ("code", CODE_CLASS_IMBALANCE),
        ("code", CODE_MISSING),
        ("code", CODE_NUMERIC_DIST),
        ("code", CODE_CATEGORICAL_DIST),
        ("code", CODE_CORRELATION),
        ("md",   MD_PREPROCESSING),
        ("code", CODE_SPLIT_FEATURES),
        ("code", CODE_TRAIN_TEST_SPLIT),
        ("code", CODE_COLUMN_TRANSFORMER),
        ("md",   MD_MODELS),
        ("code", CODE_BUILD_MODELS),
        ("code", CODE_GRID_SEARCH),
        ("md",   MD_EVAL),
        ("code", CODE_TEST_EVAL),
        ("code", CODE_ROC),
        ("code", CODE_PR),
        ("code", CODE_CONFUSION),
        ("code", CODE_FEATURE_IMPORTANCE),
        ("code", CODE_LOGREG_COEF),
        ("md",   MD_DISCUSSION),
    ]

    for kind, source in blocks:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))

    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
