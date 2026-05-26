# Gold & Silver Price Predictor

An end-to-end machine learning project for predicting next-day gold and silver price direction from technical indicators, with a Streamlit dashboard for inspection and reporting.

## What This Project Does

- Downloads gold and silver futures data from Yahoo Finance
- Engineers technical features from daily prices
- Trains separate XGBoost classifiers for gold and silver
- Selects the best model family per asset using validation performance
- Evaluates the models with a time-based holdout split
- Saves model metadata and evaluation reports for reproducibility

## What This Project Does Not Do

- It does **not** forecast exact future prices
- It does **not** guarantee trading alpha
- It should be treated as an educational ML workflow, not financial advice

## Project Structure

```text
app/                Streamlit dashboard
data/               Downloaded CSV files
models/             Trained models and training metadata
reports/            Evaluation reports
src/                Core pipeline code
tests/              Automated tests
requirements.txt    Pip dependencies
environment.yml     Conda environment
run_tests.py        Test runner
```

## Quick Start

### Option 1: pip

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate gold-silver-price-predictor
```

## Run The Pipeline

```bash
python src/download_data.py
python src/train.py
python src/evaluate.py
streamlit run app/app.py
```

## Evaluation Design

- Feature generation uses only information available up to each prediction date
- Training uses a fixed time-based split:
  - `75%` train
  - `15%` validation
  - `10%` test
- Candidate models are compared on the validation window and the best classification threshold is saved per asset
- Evaluation now uses the same held-out test window as training metadata
- Reports include model accuracy, baseline accuracy, confusion matrix, returns, and split date ranges

## Current Performance Notes

The current models are educational baselines. Performance can vary as new market data arrives, and the latest runs may not beat a naive baseline or buy-and-hold benchmark. Check the saved JSON files in `models/` and `reports/` for the latest metrics instead of relying on a fixed README claim.

## Testing

```bash
python -m pytest
```

or

```bash
python run_tests.py --quick
```

## Notes

- `requirements.txt` is the canonical pip dependency file
- `requirement.txt` is kept for backward compatibility with older commands
- The Streamlit UI is intentionally separate from the core ML pipeline

## Disclaimer

This repository is for educational purposes only. It should not be used as the sole basis for investment decisions.
