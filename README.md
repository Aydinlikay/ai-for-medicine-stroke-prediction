# AI for Medicine — Stroke Prediction

University of Bologna, *AI for Medicine* (Prof. Stefano Diciotti).
Course project: train a machine-learning model on the Kaggle Stroke Prediction Dataset.

## Layout

```
.
├── Data/                                       Raw dataset (CSV)
├── Doc/                                        Project report (Word)
├── notebook/stroke_prediction.ipynb            Main analysis (Colab-compatible)
├── outputs/                                    Figures and metrics produced by the notebook
└── requirements.txt
```

## How to run

Locally:

```bash
pip install -r requirements.txt
jupyter notebook notebook/stroke_prediction.ipynb
```

On Google Colab: upload the notebook and the CSV (or mount Drive), then run all cells.
The first cell installs any missing dependencies.

## Reproducibility

All randomness is controlled by `RANDOM_STATE = 42`, defined as a constant at the
top of the notebook. The notebook re-creates every figure in `outputs/` on each
clean run.
