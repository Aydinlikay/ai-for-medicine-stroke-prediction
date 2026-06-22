# AI for Medicine — Stroke Prediction

University of Bologna, *AI for Medicine* course project (Prof. Stefano Diciotti).
Binary classification of stroke risk from tabular health records, with a
deliberate emphasis on leakage-free preprocessing and clinically honest metrics.

## Motivation

Stroke is the second leading cause of death worldwide and a major driver of
long-term disability. A tool that can flag high-risk patients from routinely
collected variables (age, glucose, BMI, hypertension, lifestyle factors) is
clinically useful as a *triage* step ahead of more expensive workups. The
challenge in this dataset is severe class imbalance — only **4.87% of the
5,110 patients are positive** — which makes accuracy a misleading metric and
forces every modelling choice to defend itself against label noise.

## Methodology at a glance

| Stage | Choice | Why |
|---|---|---|
| Hold-out | 20% stratified, touched **once** | Honest generalisation estimate |
| Preprocessing | `ColumnTransformer` inside a `Pipeline` | Imputer / scaler refit per CV fold → no leakage |
| Imputation | Median (only `bmi`, ~4% missing) | Robust to skew |
| Encoding | One-hot, `handle_unknown="ignore"` | Graceful on unseen categories |
| Class imbalance | `class_weight="balanced"` / `scale_pos_weight` | Penalises missed strokes |
| Model selection | 5-fold stratified CV, ROC-AUC | Threshold-independent |
| Models | LogReg, Random Forest, XGBoost, SVM (RBF) | Linear baseline + non-linear tree + boosted + kernel |

## Results

Held-out test-set metrics (from `outputs/test_metrics.csv`):

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | FPR | FNR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LogisticRegression** | **0.8423** | 0.2648 | **0.80** | 0.744 | 0.256 | 0.20 | 0.138 | 0.236 |
| XGBoost | 0.8422 | **0.2689** | 0.78 | **0.784** | **0.216** | 0.22 | **0.157** | **0.261** |
| RandomForest | 0.8226 | 0.2359 | 0.76 | 0.750 | 0.250 | 0.24 | 0.135 | 0.230 |
| SVM | 0.8095 | 0.1749 | 0.78 | 0.739 | 0.261 | 0.22 | 0.133 | 0.227 |

LogisticRegression and XGBoost are tied on ROC-AUC (within the ≈0.01 noise
band documented in the report). LogReg has the higher sensitivity (0.80 vs
0.78), XGBoost has the better specificity, PR-AUC and F1 — i.e. it catches
slightly fewer strokes but raises noticeably fewer false alarms. Either choice
is defensible depending on the desired operating point.

Raw confusion-matrix counts per model are also exported to
`outputs/test_confusion_counts.csv` for full transparency.

Age dominates every classifier; average glucose level, hypertension and
heart-disease flags follow. The low absolute precision is the intended
trade-off for a screening tool — see the report's discussion for the clinical
reasoning.

All figures referenced in the report (`Doc/AI_for_Medicine_Project_Filled.docx`)
are produced by the notebook and live in `outputs/`:

```
outputs/
├── class_imbalance.png         # 95 / 5 target split
├── numeric_distributions.png   # age / glucose / bmi by stroke
├── categorical_stroke_rates.png
├── correlation_heatmap.png
├── roc_curves.png              # all four models, one axis
├── pr_curves.png               # honest view under imbalance
├── confusion_matrices.png      # row-normalised: (Specificity, Sensitivity) on the diagonal
├── feature_importances.png     # RF + XGBoost
├── logreg_coefficients.png     # standardised, clinically readable
├── test_metrics.csv            # ROC-AUC, PR-AUC, Sensitivity, Specificity, FPR, FNR, Precision, F1
└── test_confusion_counts.csv   # raw TN / FP / FN / TP per model
```

## Layout

```
.
├── Data/healthcare-dataset-stroke-data.csv   Raw Kaggle dataset
├── Doc/AI_for_Medicine_Project_Filled.docx   Project report (filled template)
├── notebook/stroke_prediction.ipynb          Main analysis
├── outputs/                                  Figures and metrics
├── requirements.txt
└── README.md
```

## How to run

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook notebook/stroke_prediction.ipynb
```

Run all cells in order. A clean run regenerates every artifact in `outputs/`.

**On Google Colab:** upload `notebook/stroke_prediction.ipynb`. The first cell
installs any missing dependencies; the second cell clones the repository if it
detects a Colab runtime so the CSV is found at the expected path.

## Reproducibility

A single `RANDOM_STATE = 42` constant controls every source of randomness in the
pipeline — train/test split, K-fold shuffles, model initialisation. A reviewer
running the notebook on a fresh checkout should see *identical* numbers, not
"approximately equal" ones.

## Data and license

Healthcare Stroke Dataset by *fedesoriano* (Kaggle, 2021). Publicly available
for educational and research use. The dataset contains no identifying
information; the only identifier column (`id`) is dropped at load time.
