# ML-based Network Intrusion Detection System

> One-line pitch: a real-time network IDS that scores live traffic flows with
> XGBoost, flags unseen attacks with an Isolation Forest, and explains every
> alert with SHAP. *(Fill in real numbers on day 9.)*

**Status:** day 2 of 10. Data cleaned, features selected, splits ready.

## Architecture

*(Diagram on day 9.)* Packets → CICFlowMeter flows → FastAPI `/score`
(XGBoost + Isolation Forest + SHAP) → SQLite alerts → Streamlit dashboard.

## Quick start

```bash
make setup                      # venv + requirements + pip install -e .
source .venv/bin/activate
# download CICIDS2017 first, see data/README.md
make data                       # clean CSVs -> data/processed/cicids2017_clean.parquet
make features                   # families, top-25 features, train/val/test parquet
make test
```

The full stack (`docker compose up`) arrives on day 8.

## Dataset

CICIDS2017 (Canadian Institute for Cybersecurity), MachineLearningCVE CSVs.
Cleaning steps and row counts: `src/ids/data/clean.py` and
`data/processed/cleaning_summary.json`.

## Data preparation

- 15 labels grouped into 7 families: Benign, DoS, DDoS, PortScan, Brute force,
  Web attack, Bot. Infiltration (36 rows) and Heartbleed (11 rows) are dropped as
  too small to learn from.
- `Destination Port` is dropped because it identifies the service, not the behaviour.
- Near-duplicate features (|r| > 0.98) are pruned, then the top 25 by Random Forest
  importance are kept (`configs/features.yaml`).
- Stratified 70/15/15 split on family, made before feature selection. A time-based
  split would be stricter; see Limitations.

## Results

| Model | Macro-F1 | Benign FPR | ms / flow |
|---|---|---|---|
| Random Forest | | | |
| XGBoost | | | |
| Isolation Forest (held-out family) | | | |

## Repo layout

```
configs/      feature list, model params, thresholds
data/         raw/ and processed/ are gitignored; README.md has download steps
notebooks/    01_eda ... 06_shap
src/ids/      data, features, models, explain, capture, api, storage
dashboard/    Streamlit app
lab/          isolated attack lab notes and scripts
reports/      figures and metrics JSON
tests/        pytest
```

## Limitations

*(Day 9: known CICIDS2017 labelling issues, stratified vs time-based split,
lab-only live testing.)*

## Ethics

All live attacks run only against containers I own on an isolated network.
