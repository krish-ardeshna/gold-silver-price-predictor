import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime, timedelta

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import load_and_prepare, FEATURE_COLS
from src.download_data import main as download_main
from src.train import main as train_main
from src.evaluate import main as evaluate_main

# Theme configuration
THEMES = {
    "light": {
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
    },
    "dark": {
        "primaryColor": "#FF6B6B",
        "backgroundColor": "#0E1117",
        "secondaryBackgroundColor": "#262730",
        "textColor": "#FAFAFA",
    }
}

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Apply theme
theme = st.session_state.theme
st.set_page_config(
    page_title="Gold & Silver Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈",
)

# Custom CSS for theme
st.markdown(f"""
<style>
    body, html, .stApp, .main, .block-container, .reportview-container, .stSidebar, .sidebar-content, .streamlit-expanderHeader, .stTabs, .streamlit-expanderContent, .streamlit-expanderContent > div {{
        background-color: {THEMES[theme]['backgroundColor']} !important;
        color: {THEMES[theme]['textColor']} !important;
    }}

    .stSidebar, .sidebar-content, .css-1avcm0n, .css-1k1zq4w, .css-1v3fvcr, .css-1d391kg, .css-1m8bv3o, .css-1w3pq9k {{
        background-color: {THEMES[theme]['secondaryBackgroundColor']} !important;
        color: {THEMES[theme]['textColor']} !important;
    }}

    /* Fix Top Right Menu (Deploy & 3 dots) and Sidebar Toggle Arrow Visibility */
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    [data-testid="collapsedControl"] svg, [data-testid="stSidebar"] button svg, [data-testid="stHeader"] svg, [data-testid="stHeader"] span {{
        fill: {THEMES[theme]['textColor']} !important;
        color: {THEMES[theme]['textColor']} !important;
    }}

    .stMetric, .stMetric .metric-label, .stMetric .metric-value, .stMetric .metricDelta, .metric-container {{
        color: {THEMES[theme]['textColor']} !important;
        background-color: {THEMES[theme]['secondaryBackgroundColor']} !important;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}

    .stMarkdown, .stText, .stButton, .stSelectbox, .stNumberInput, .stTextInput, .css-1xkdx3g, .css-18e3th9, .css-q8zr2y {{
        color: {THEMES[theme]['textColor']} !important;
    }}

    .stButton button {{
        color: {THEMES[theme]['textColor']} !important;
        background-color: {THEMES[theme]['primaryColor']} !important;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }}

    .stButton button:hover {{
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}

    /* Fix Market Settings Selectbox and Inputs */
    [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
        color: {THEMES[theme]['textColor']} !important;
        background-color: {THEMES[theme]['secondaryBackgroundColor']} !important;
        border-radius: 5px;
        border: 1px solid {THEMES[theme]['primaryColor']};
    }}

    .css-1avcm0n *, .stSidebar * {{
        color: {THEMES[theme]['textColor']} !important;
    }}

    h1, h2, h3, h4, h5, h6, p, label, a, strong {{
        color: {THEMES[theme]['textColor']} !important;
    }}

    .signal-card {{
        background: linear-gradient(135deg, {THEMES[theme]['secondaryBackgroundColor']} 0%, {THEMES[theme]['backgroundColor']} 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.title("🎯 Controls")
    
    # Theme toggle
    st.markdown("### 🌙 Theme")
    theme_toggle = st.toggle("Dark Mode", value=st.session_state.theme == "dark")
    if theme_toggle != (st.session_state.theme == "dark"):
        st.session_state.theme = "dark" if theme_toggle else "light"
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📊 Market Settings")
    asset = st.selectbox("Choose asset", ["gold", "silver"], help="Select the commodity to analyze")
    chart_days = st.slider("Chart history (days)", 30, 365, 180, help="Number of days to display in charts")
    
    st.divider()
    
    # Prediction settings
    st.markdown("### 🔮 Prediction Settings")
    prediction_days = st.slider("Future predictions (days)", 1, 30, 7, help="Days ahead to predict")
    confidence_threshold = st.slider("Signal threshold", 0.4, 0.6, 0.5, 0.01, help="Confidence level for buy/sell signals")
    
    st.divider()
    
    if st.button("🔄 Retrain Models", help="Download fresh data and rebuild models", use_container_width=True):
        with st.spinner("Refreshing data and rebuilding models... This may take a moment."):
            if download_main() != 0:
                st.error("Failed to download data. Check terminal logs.")
            elif train_main() != 0:
                st.error("Failed to train models. Check terminal logs.")
            elif evaluate_main() != 0:
                st.error("Failed to evaluate models. Check terminal logs.")
            else:
                # Clear memory caches so the newly trained files are loaded
                st.cache_data.clear()
                st.cache_resource.clear()
                
                st.success("Retrain complete! Reloading app...")
                time.sleep(1) # Brief pause for user to read success message
                st.rerun()
    
    st.divider()
    
    st.markdown(
        "💡 **Tips:**\n"
        "- Use the theme toggle for better visibility\n"
        "- Adjust prediction days for future outlook\n"
        "- Automatic model refresh runs in the background for professional accuracy"
    )

# File paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_path, "data", f"{asset}.csv")
model_path = os.path.join(base_path, "models", f"{asset}_model.pkl")
info_path = os.path.join(base_path, "models", f"{asset}_info.json")
eval_path = os.path.join(base_path, "reports", f"{asset}_evaluation.json")

AUTO_RETRAIN_DAYS = 7
DATA_REFRESH_DAYS = 1


def get_file_age_days(path):
    if not os.path.exists(path):
        return None
    return (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days


def needs_auto_retrain():
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return True

    model_age = get_file_age_days(model_path)
    data_age = get_file_age_days(data_path)
    return (model_age is not None and model_age >= AUTO_RETRAIN_DAYS) or (
        data_age is not None and data_age >= DATA_REFRESH_DAYS
    )


def get_model_status():
    if not os.path.exists(model_path):
        return "No trained model found. Automatic rebuild is enabled."
    model_age = get_file_age_days(model_path)
    data_age = get_file_age_days(data_path)
    return f"Model age: {model_age} day(s), Data age: {data_age} day(s)"

auto_retrain_needed = needs_auto_retrain()
if auto_retrain_needed:
    st.sidebar.warning("Automatic refresh required: model or data is stale. The app will refresh automatically.")
else:
    st.sidebar.success(get_model_status())

if auto_retrain_needed and not st.session_state.get("auto_retrain_attempted", False):
    st.session_state.auto_retrain_attempted = True
    with st.spinner("Refreshing stale market data and rebuilding models for professional reliability..."):
        retrain_failed = False
        if download_main() != 0:
            st.sidebar.error("Failed to download fresh data. Check terminal logs.")
            retrain_failed = True
        elif train_main() != 0:
            st.sidebar.error("Failed to train the model. Check terminal logs.")
            retrain_failed = True
        elif evaluate_main() != 0:
            st.sidebar.error("Failed to evaluate the new model. Check terminal logs.")
            retrain_failed = True
        else:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Automatic rebuild complete. Reloading dashboard...")
            st.rerun()

        if retrain_failed:
            st.error("Automatic refresh could not complete. Please use the retrain button or check logs.")
            st.stop()

# Page header
st.title("📈 Gold & Silver Price Predictor")
st.markdown(
    "**Advanced ML-powered dashboard for commodity price prediction and market analysis.**"
)

# Current market status
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.subheader(f"📊 {asset.upper()} Analysis Dashboard")
with col2:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.metric("Last Updated", current_time)
with col3:
    st.metric("Theme", st.session_state.theme.title())

st.divider()

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.metric("Model Status", "Stale refresh pending" if auto_retrain_needed else "Model up to date")
with status_col2:
    st.metric("Data Age", f"{get_file_age_days(data_path)} day(s)")
with status_col3:
    st.metric("Model Age", f"{get_file_age_days(model_path)} day(s)")

st.divider()

# Validation checks
if not os.path.exists(data_path):
    st.error(f"Missing data file: {data_path}. Run retrain to fetch data.")
    st.stop()

if not os.path.exists(model_path):
    st.error(f"Missing model file: {model_path}. Run retrain to build the model.")
    st.stop()

@st.cache_data
def get_data(path):
    return load_and_prepare(path)

@st.cache_resource
def get_model(path):
    return joblib.load(path)

try:
    df = get_data(data_path)
    if len(df) < 2:
        st.error("Not enough data to display. Need at least 2 days of valid data.")
        st.stop()
except Exception as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

try:
    model = get_model(model_path)
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

try:
    latest = df.iloc[-1]
    features = latest[FEATURE_COLS].to_frame().T
    # Ensure features are numeric
    features = features.astype(float)
    proba = model.predict_proba(features)[0]
    confidence = float(proba[1])
except Exception as exc:
    st.error(f"Error generating predictions: {exc}. Try retraining the models.")
    st.stop()

# Signal summary
st.subheader("🎯 Current Signal")

if confidence > confidence_threshold + 0.08:
    signal = "BUY"
    signal_color = "green"
    signal_icon = "🟢"
    signal_desc = "Strong bullish signal"
elif confidence > confidence_threshold:
    signal = "BUY"
    signal_color = "lightgreen"
    signal_icon = "🟡"
    signal_desc = "Moderate bullish signal"
elif confidence < 1 - confidence_threshold - 0.08:
    signal = "SELL"
    signal_color = "red"
    signal_icon = "🔴"
    signal_desc = "Strong bearish signal"
elif confidence < 1 - confidence_threshold:
    signal = "SELL"
    signal_color = "lightcoral"
    signal_icon = "🟡"
    signal_desc = "Moderate bearish signal"
else:
    signal = "HOLD"
    signal_color = "orange"
    signal_icon = "🟡"
    signal_desc = "Neutral signal"

# Signal display with enhanced styling
st.markdown(f"""
<div class="signal-card">
    <h1 style='color: {signal_color}; margin: 0;'>{signal_icon} {signal}</h1>
    <h3 style='margin: 5px 0;'>Confidence: {confidence*100:.1f}%</h3>
    <p style='margin: 0; font-style: italic;'>{signal_desc}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# KPI cards
st.subheader("📊 Key Metrics")

try:
    # Calculate additional metrics
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest['Close']
    price_change = latest['Close'] - prev_close
    price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
    
    vol_7d = df['Close'].tail(7).std()
    vol_30d = df['Close'].tail(30).std()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Close", f"${latest['Close']:.2f}", 
                 f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
                 delta_color="normal")
    with col2:
        st.metric("7-day MA", f"${latest['MA7']:.2f}")
    with col3:
        st.metric("30-day MA", f"${latest['MA30']:.2f}")
    with col4:
        st.metric("RSI", f"{latest['RSI']:.1f}", 
                 "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("7-day Volatility", f"{vol_7d:.2f}")
    with col6:
        st.metric("30-day Volatility", f"{vol_30d:.2f}")
    with col7:
        momentum = latest['Momentum_3']
        st.metric("3-day Momentum", f"{momentum:.3f}", 
                 "Bullish" if momentum > 1 else "Bearish")
    with col8:
        trend = latest['Trend_Strength']
        st.metric("Trend Strength", f"{trend:+.2f}", 
                 "Uptrend" if trend > 0 else "Downtrend")

except Exception as exc:
    st.warning("Could not calculate all metrics. The data might be incomplete.")

st.divider()

# Main view tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "🎯 Model Summary", "📊 Feature Importance", "🔮 Predictions"])

with tab1:
    st.subheader("Price and Technical Analysis")
    
    # Interactive price chart with Plotly
    chart_df = df.tail(chart_days)[["Close", "MA7", "MA30", "BB_Upper", "BB_Lower"]].copy()
    chart_df['Date'] = pd.date_range(end=datetime.now(), periods=len(chart_df), freq='D')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Close'], mode='lines', name='Close Price',
                            line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['MA7'], mode='lines', name='7-day MA',
                            line=dict(color='orange', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['MA30'], mode='lines', name='30-day MA',
                            line=dict(color='red', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['BB_Upper'], mode='lines', name='BB Upper',
                            line=dict(color='gray', width=1, dash='dot'), showlegend=False))
    fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['BB_Lower'], mode='lines', name='BB Lower',
                            line=dict(color='gray', width=1, dash='dot'), fill='tonexty', showlegend=False))
    
    fig.update_layout(title=f"{asset.upper()} Price Chart", xaxis_title="Date", yaxis_title="Price ($)",
                     height=400, template="plotly_white" if theme == "light" else "plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RSI & Momentum")
        tech_df = df.tail(chart_days)[["RSI", "Momentum_3"]].copy()
        tech_df['Date'] = pd.date_range(end=datetime.now(), periods=len(tech_df), freq='D')
        
        fig_rsi = px.line(tech_df, x='Date', y='RSI', title="RSI Indicator")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(height=300, template="plotly_white" if theme == "light" else "plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        st.subheader("MACD")
        macd_df = df.tail(chart_days)[["MACD", "MACD_Signal", "MACD_Hist"]].copy()
        macd_df['Date'] = pd.date_range(end=datetime.now(), periods=len(macd_df), freq='D')
        
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=macd_df['Date'], y=macd_df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=macd_df['Date'], y=macd_df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')))
        fig_macd.add_trace(go.Bar(x=macd_df['Date'], y=macd_df['MACD_Hist'], name='Histogram', marker_color='gray'))
        fig_macd.update_layout(title="MACD Indicator", height=300, template="plotly_white" if theme == "light" else "plotly_dark")
        st.plotly_chart(fig_macd, use_container_width=True)

with tab2:
    st.subheader("🤖 Model Performance & Evaluation")
    st.markdown("Use this section to review the latest model status, accuracy, and strategy performance.")

    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                info = json.load(f)

            st.markdown("#### Model Summary")
            model_cols = st.columns(3)
            with model_cols[0]:
                st.metric("Accuracy", f"{info['accuracy']*100:.1f}%")
                st.metric("Best Iteration", f"{info['best_iteration']}")
            with model_cols[1]:
                st.metric("Training Rows", f"{info['train_rows']:,}")
                st.metric("Validation Rows", f"{info['val_rows']:,}")
            with model_cols[2]:
                st.metric("Test Rows", f"{info['test_rows']:,}")
                st.metric("Features", f"{info['n_features']}")

            with st.expander("View training configuration"):
                st.json({
                    "scale_pos_weight": info.get("scale_pos_weight", "N/A"),
                    "features": info.get("features", []),
                })

        except json.JSONDecodeError:
            st.warning("Model info file is corrupted. Retrain to generate it.")
    else:
        st.info("Model info not found. Retrain to generate it.")

    if os.path.exists(eval_path):
        try:
            with open(eval_path) as f:
                results = json.load(f)

            st.markdown("#### Strategy Performance")
            perf_cols = st.columns(4)
            with perf_cols[0]:
                st.metric("Strategy Return", f"{results['strategy_return']*100:+.2f}%",
                         help="Total return from following model signals")
            with perf_cols[1]:
                st.metric("Buy & Hold", f"{results['buy_hold_return']*100:+.2f}%",
                         help="Return from simply holding the asset")
            with perf_cols[2]:
                st.metric("Alpha", f"{results['alpha']*100:+.2f}%",
                         help="Excess return over buy & hold")
            with perf_cols[3]:
                st.metric("Sharpe Ratio", f"{results['sharpe']:.2f}",
                         help="Risk-adjusted return measure")

            risk_cols = st.columns(3)
            with risk_cols[0]:
                st.metric("Win Rate", f"{results['win_rate']*100:.1f}%")
            with risk_cols[1]:
                st.metric("Max Drawdown", f"{results['max_drawdown']*100:.1f}%")
            with risk_cols[2]:
                st.metric("Trades Taken", f"{results['trades_taken']}")

        except json.JSONDecodeError:
            st.warning("Evaluation report is corrupted. Retrain to generate it.")
    else:
        st.info("Evaluation report not found. Retrain to generate it.")

with tab3:
    st.subheader("📊 Feature Importance Analysis")
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        
        # Top 10 features bar chart
        fig_importance = px.bar(importance.head(10), x=importance.head(10).values, y=importance.head(10).index,
                               orientation='h', title="Top 10 Most Important Features")
        fig_importance.update_layout(xaxis_title="Importance Score", yaxis_title="Feature",
                                   height=400, template="plotly_white" if theme == "light" else "plotly_dark")
        st.plotly_chart(fig_importance, use_container_width=True)
        
        with st.expander("📋 All Features Ranking"):
            st.dataframe(importance.to_frame(name="Importance Score").style.format({"Importance Score": "{:.4f}"}))
            
        # Feature categories
        st.markdown("### Feature Categories")
        categories = {
            "Price & Moving Averages": ["MA7", "MA30", "MA_Ratio"],
            "Returns & Momentum": ["Lag1_Return", "Lag2_Return", "Lag3_Return", "Momentum_3", "Momentum_7"],
            "Technical Indicators": ["RSI", "MACD", "MACD_Signal", "MACD_Hist", "ADX"],
            "Volatility": ["Volatility7", "Volatility14", "Vol_Ratio"],
            "Statistical": ["Zscore_7", "Breakout_7", "BB_Position"]
        }
        
        for category, features in categories.items():
            if any(f in importance.index for f in features):
                with st.expander(f"🔍 {category}"):
                    cat_importance = importance[importance.index.isin(features)]
                    st.dataframe(cat_importance.to_frame(name="Importance").style.format({"Importance": "{:.4f}"}))
    else:
        st.info("Feature importance is unavailable for this model.")

with tab4:
    st.subheader("🔮 Future Price Predictions")
    
    # Generate future predictions
    try:
        future_dates = pd.date_range(start=datetime.now(), periods=prediction_days + 1, freq='D')[1:]
        future_predictions = []
        
        # Use current features and simulate forward
        current_features = latest[FEATURE_COLS].copy()
        
        for i in range(prediction_days):
            # Predict for current day
            pred_proba = model.predict_proba(current_features.to_frame().T.astype(float))[0]
            pred_confidence = float(pred_proba[1])
            
            # Simulate price movement based on prediction
            expected_return = (pred_confidence - 0.5) * 0.02  # 2% max daily move
            predicted_price = latest['Close'] * (1 + expected_return * (i + 1))
            
            future_predictions.append({
                'date': future_dates[i],
                'predicted_price': predicted_price,
                'confidence': pred_confidence,
                'signal': 'BUY' if pred_confidence > confidence_threshold else 'SELL' if pred_confidence < 1 - confidence_threshold else 'HOLD'
            })
            
            # Update features for next prediction (simplified)
            current_features = current_features * 0.99  # Slight decay
        
        pred_df = pd.DataFrame(future_predictions)
        
        # Prediction chart
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=[datetime.now()], y=[latest['Close']], mode='markers', 
                                    name='Current Price', marker=dict(size=10, color='red')))
        fig_pred.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['predicted_price'], mode='lines+markers',
                                    name='Predicted Price', line=dict(color='blue', dash='dash')))
        
        colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        for signal in pred_df['signal'].unique():
            mask = pred_df['signal'] == signal
            fig_pred.add_trace(go.Scatter(x=pred_df[mask]['date'], y=pred_df[mask]['predicted_price'],
                                        mode='markers', name=f'{signal} Signal',
                                        marker=dict(size=8, color=colors[signal]), showlegend=False))
        
        fig_pred.update_layout(title=f"{prediction_days}-Day Price Prediction for {asset.upper()}",
                              xaxis_title="Date", yaxis_title="Price ($)", height=400,
                              template="plotly_white" if theme == "light" else "plotly_dark")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Prediction table
        st.markdown("### Prediction Details")
        display_df = pred_df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['predicted_price'] = display_df['predicted_price'].map('${:.2f}'.format)
        display_df['confidence'] = display_df['confidence'].map('{:.1%}'.format)
        display_df = display_df.rename(columns={
            'date': 'Date',
            'predicted_price': 'Predicted Price',
            'confidence': 'Confidence',
            'signal': 'Signal'
        })
        st.dataframe(display_df, use_container_width=True)
        
        st.warning("⚠️ **Disclaimer:** These predictions are for educational purposes only and should not be used for actual trading decisions. Past performance does not guarantee future results.")
        
    except Exception as e:
        st.error(f"Could not generate predictions: {str(e)}")

st.divider()
st.write(
    "📊 **Dashboard Information:** The predicted signal is generated from the latest feature set. Use this app to monitor trends and update the model after new data becomes available. "
    "For a fresh model, click the Retrain Models button in the sidebar."
)
