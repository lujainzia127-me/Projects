# AI Threat Intelligence Platform

Machine-learning cybersecurity tool. Upload a network-traffic log, classify each
flow as benign or a specific attack type (DoS, DDoS, PortScan, Brute Force, Bot,
Web Attack), then view a dashboard, get a risk score, generate a PDF report, and
send an email alert if the risk is high.

Dataset: CICIDS2017 (cleaned & preprocessed version, `cicids2017_cleaned.csv`).

## The two phases

**Phase 1 — Training pipeline (offline, run once)**
`inspect_data.py` -> `src/preprocess.py` -> `src/train.py`
Produces `models/model.pkl` and `models/scaler.pkl`.

**Phase 2 — Streamlit app (runtime, every upload)**
`app.py` loads the saved model + scaler and runs the full analysis on uploads.
It calls the SAME `src/preprocess.py` so training and inference stay consistent.

## Setup

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Put `cicids2017_cleaned.csv` inside the `data/` folder.

## Build order

1. `python inspect_data.py`        # look at columns + class balance FIRST
2. Fill in FEATURE_COLUMNS in `src/preprocess.py` using the printed column list
3. `python src/train.py`           # trains, evaluates, saves model + scaler
4. `streamlit run app.py`          # the app

## Golden rules (do not break these)

- Split train/test BEFORE scaling or balancing.
- Fit the scaler on the TRAINING set only, then save it.
- The app must use the SAME preprocess function as training.
- Never retrain inside the app — it only loads the saved model.

## Folders

```
data/        cicids2017_cleaned.csv  (you add this; not committed)
notebooks/   exploration
src/         reusable code (preprocess, train, risk, report, alerts)
models/      saved model.pkl + scaler.pkl (created by train.py)
app.py       Streamlit app
inspect_data.py   first thing to run
```
