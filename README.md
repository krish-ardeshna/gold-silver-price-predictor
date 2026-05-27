# 📈 Gold & Silver Price Predictor

**An end-to-end ML pipeline for predicting next-day gold and silver price direction,  
with a Streamlit dashboard for live signals, technical analysis, and model inspection.**

[Live Demo](#) · [Report Bug](../../issues) · [Request Feature](../../issues)

---

## 🧠 What This Project Does

- Downloads **10 years** of gold (`GC=F`) and silver (`SI=F`) futures data via Yahoo Finance
- Engineers **23 technical features** from daily OHLCV data (RSI, MACD, Bollinger Bands, momentum, volatility, lag returns, z-score, ADX proxy, and more)
- Trains **3 candidate models** per asset (Logistic Regression, XGBoost Medium Depth, XGBoost Current) and auto-selects the best performer on validation
- Uses **time-based train/val/test splits** to prevent data leakage
- Evaluates with classification metrics **and** a simulated directional trading strategy (Sharpe, drawdown, win rate, alpha)
- Serves results via an **interactive Streamlit dashboard** with live signals, charts, and auto-retraining

## ❌ What This Project Does NOT Do

- Does **not** forecast exact future prices
- Does **not** guarantee trading alpha or positive returns
- Is **not** financial advice — treat as an educational ML workflow

---

## ⚡ Live Demo

> Deploy link here after Streamlit Cloud deployment.  
> `https://your-app-name.streamlit.app`

---

## 🏗️ Architecture

```
Yahoo Finance (yfinance)
        │
        ▼
  download_data.py          ← fetches gold/silver CSVs
        │
        ▼
  preprocess.py             ← 23 technical features, .shift(1) anti-leakage
        │
        ▼
  train.py                  ← 3 candidates → best model selected on val set
        │
        ▼
  evaluate.py               ← classification + trading strategy metrics
        │
        ▼
  app/app.py                ← Streamlit dashboard (signals, charts, retrain)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data | yfinance, pandas, numpy |
| ML | XGBoost, scikit-learn, joblib |
| Visualization | Plotly, Streamlit |
| Testing | pytest, pytest-cov, pytest-mock |
| Environment | Conda / pip |

---

## 📊 Features Engineered (23 total)

| Category | Features |
|---|---|
| Moving Averages | `MA7`, `MA30`, `MA_Ratio` |
| Lag Returns | `Lag1_Return`, `Lag2_Return`, `Lag3_Return` |
| Price Ratios | `Lag1_PriceRatio`, `Lag2_PriceRatio`, `Lag3_PriceRatio` |
| Volatility | `Volatility7`, `Volatility14`, `Vol_Ratio` |
| Momentum | `Momentum_3`, `Momentum_7`, `Trend_Strength` |
| Oscillators | `RSI`, `MACD`, `MACD_Signal`, `MACD_Hist` |
| Statistical | `Zscore_7`, `Breakout_7`, `BB_Position`, `ADX` |

All features use `.shift(1)` to ensure **zero lookahead bias** — only prior-day information is used to predict the next day's direction.

---

## 📈 Model Performance

> Example results from a recent test run. Actual metrics may change after retraining or dataset updates.

### Gold

| Metric | Value |
|---|---|
| Model Selected | Logistic Regression |
| Test Accuracy | 49.8% |
| Baseline Accuracy | 53.8% |
| Sharpe Ratio | 0.62 |
| Strategy Return | +6.0% |
| Buy & Hold Return | +34.3% |
| Win Rate | 44.4% |
| Max Drawdown | -5.4% |

### Silver

| Metric | Value |
|---|---|
| Model Selected | XGBoost Medium Depth |
| Test Accuracy | 56.2% |
| Baseline Accuracy | 55.4% |
| Sharpe Ratio | 1.61 |
| Strategy Return | +3.5% |
| Buy & Hold Return | +118.9% |
| Win Rate | 85.7% |
| Max Drawdown | -2.1% |

> **Note:** These are educational baselines. Financial direction classification is inherently noisy — 55–57% accuracy on unseen market data is typical and honest. Check `models/` and `reports/` JSON files for always-current metrics.

---

## 📁 Project Structure

```
gold-silver-price-predictor/
│
├── app/
│   └── app.py                  # Streamlit dashboard
│
├── src/
│   ├── __init__.py
│   ├── download_data.py        # yfinance data fetcher
│   ├── preprocess.py           # Feature engineering pipeline
│   ├── train.py                # Model training + candidate selection
│   └── evaluate.py             # Classification + strategy evaluation
│
├── models/
│   ├── gold_model.pkl          # Trained gold model
│   ├── silver_model.pkl        # Trained silver model
│   ├── gold_info.json          # Training metadata
│   └── silver_info.json
│
├── reports/
│   ├── gold_evaluation.json    # Full evaluation report
│   └── silver_evaluation.json
│
├── data/                       # Downloaded CSVs (gitignored)
│
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_download_data.py
│   ├── test_model_selection.py
│   └── test_app.py
│
├── requirements.txt
├── environment.yml
├── pytest.ini
├── run_tests.py
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — pip

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Option 2 — conda

```bash
conda env create -f environment.yml
conda activate gold-silver-price-predictor
```

---

## ▶️ Run the Full Pipeline

```bash
# 1. Download data
python src/download_data.py

# 2. Train models
python src/train.py

# 3. Evaluate models
python src/evaluate.py

# 4. Launch dashboard
streamlit run app/app.py
```

Or just launch the dashboard — it auto-retrains if the model is stale (>7 days old).

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Quick run (unit + integration only)
python run_tests.py --quick

# With coverage report
python -m pytest --cov=src --cov-report=html

# Specific test file
python -m pytest tests/test_preprocess.py -v
```

Test suite covers: preprocessing correctness, anti-lookahead-bias checks, model selection logic, evaluation metrics consistency, edge cases (empty files, corrupted data, single-class targets), and performance benchmarks.

---

## 📐 Evaluation Design

- **Time-based split** — no shuffling, no future leakage
  - Train: 75% · Validation: 15% · Test: 10%
- **Threshold tuning** on validation set per asset (grid search 0.35 → 0.65)
- **Model selection** ranks candidates by balanced accuracy → accuracy → F1
- **Strategy simulation** uses continuous positioning (not binary), includes transaction costs (0.05% per trade), and reports Sharpe, Calmar-style drawdown, and alpha vs buy-and-hold

---

## 🔄 Auto-Retraining

The dashboard checks model age on startup:

- Model **>7 days old** → auto-downloads fresh data and retrains
- Data **>1 day old but model fresh** → shows warning, no auto-retrain
- Everything **fresh** → shows model age in sidebar

Manual retrain is also available via the **🔄 Retrain Models** button in the sidebar.

---

## 🌐 Deployment

Deployed on **Streamlit Community Cloud**.

To deploy your own fork:

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → branch `main` → main file: `app/app.py`
4. Deploy

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
It is not financial advice and should not be used as the sole basis for investment decisions.  
Past model performance does not guarantee future results.

---

## 🙋 Author

**Krish Ardeshna**  
B.Tech CSE · Amity University Rajasthan  
[LinkedIn](https://linkedin.com/in/ardeshnakrish) · [GitHub](https://github.com/krish-ardeshna)