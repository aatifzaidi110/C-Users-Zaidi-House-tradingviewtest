# app.py - Version 1.18
# app.py
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Needed for plt.close() in display_components
import yfinance as yf # Keep this import here for direct yf usage if any, though it's also in utils

print("Current working directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("=== DEBUG INFO ===")
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("Python path:", sys.path)
print("=================")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from modules
from utils import (
    get_finviz_data, get_data, get_options_chain,
    calculate_indicators, calculate_pivot_points,
    generate_signals_for_row, backtest_strategy,
    generate_option_trade_plan, convert_compound_to_100_scale, EXPERT_RATING_MAP
)
try:
    from display_components import (
        display_main_analysis_tab, display_trade_plan_options_tab,
        display_backtest_tab, display_news_info_tab, display_trade_log_tab,
        display_interactive_payoff_calculator # Import the new interactive payoff calculator
    )
except ImportError as e:
    print("Import error details:", str(e))
    raise

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# Initialize session state for analysis control and ticker input
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'current_ticker_input' not in st.session_state:
    st.session_state.current_ticker_input = "NVDA" # Default value

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
# Use st.session_state to manage the ticker input value
new_ticker_input = st.sidebar.text_input("Enter a Ticker Symbol", value=st.session_state.current_ticker_input).upper()

# Update session state only if the input value has changed
if new_ticker_input != st.session_state.current_ticker_input:
    st.session_state.current_ticker_input = new_ticker_input
    # No rerun here, let the button click handle it

timeframe = st.sidebar.radio("Choose Trading Style:", ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"], index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50, 200)", value=True),
        "Ichimoku Cloud": st.checkbox("Ichimoku Cloud", value=False, disabled=True), # Disabled Ichimoku
        "Parabolic SAR": st.checkbox("Parabolic SAR", value=True),
        "ADX": st.checkbox("ADX", value=True),
    }
with st.sidebar.expander("Momentum & Volume Indicators"):
    indicator_selection.update({
        "RSI Momentum": st.checkbox("RSI Momentum", value=True),
        "Stochastic": st.checkbox("Stochastic Oscillator", value=True),
        "CCI": st.checkbox("Commodity Channel Index (CCI)", value=True),
        "ROC": st.checkbox("Rate of Change (ROC)", value=True),
        "Volume Spike": st.checkbox("Volume Spike", value=True),
        "OBV": st.checkbox("On-Balance Volume (OBV)", value=True),
        "VWAP": st.checkbox("VWAP (Intraday only)", value=True, disabled=(timeframe not in ["Scalp Trading", "Day Trading"])),
    })
with st.sidebar.expander("Display-Only Indicators"):
    indicator_selection.update({
        "Bollinger Bands": st.checkbox("Bollinger Bands Display", value=True),
        "Pivot Points": st.checkbox("Pivot Points Display (Daily only)", value=True, disabled=(timeframe not in ["Swing Trading", "Position Trading"])),
    })

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only the Technical Score will count.")
auto_sentiment_score_placeholder = st.sidebar.empty()
auto_expert_score_placeholder = st.sidebar.empty()

# === Buttons ===
col_buttons1, col_buttons2 = st.columns([0.2, 0.8])

with col_buttons1:
    # When "Analyze Ticker" is clicked, set analysis_started to True and rerun
    if st.button("▶️ Analyze Ticker", help="Click to analyze the entered ticker and display results."):
        st.session_state.analysis_started = True
        st.rerun()

with col_buttons2:
    # When "Clear Cache & Refresh Data" is clicked, clear cache, reset analysis_started, and rerun
    if st.button("🔄 Clear Cache & Refresh Data", help="Click to clear all cached data and re-run analysis from scratch."):
        st.cache_data.clear() # Clear all cached data
        st.session_state.analysis_started = False # Reset analysis state
        st.rerun()

# === Constants and Configuration ===
LOG_FILE = "trade_log.csv" # Define here or pass from a config module

# === Main Script Execution ===
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
selected_params_main = TIMEFRAME_MAP[timeframe]

# Only run analysis if the button has been clicked
if st.session_state.analysis_started:
    # Use the ticker from session state, which is updated by text_input
    ticker_to_analyze = st.session_state.current_ticker_input

    if ticker_to_analyze:
        st.write(f"Analyzing ticker: {ticker_to_analyze}") # Debug print to confirm the ticker being processed
        try:
            # --- Dynamic Qualitative Score Calculation ---
            sentiment_score = 50
            expert_score = 50
            finviz_data = {"headlines": ["Automation is disabled."]}

            if use_automation:
                # Ensure finviz_data is fetched for the current ticker_to_analyze
                finviz_data = get_finviz_data(ticker_to_analyze)
                
                if 'error' in finviz_data and "429 Client Error" in finviz_data['error']:
                    st.warning("Finviz data could not be fetched due to rate limiting (Too Many Requests). Using default/manual scores for Sentiment and Expert Rating.", icon="⚠️")
                    auto_sentiment_score = 50 # Default to neutral
                    auto_expert_score = 50 # Default to hold
                    finviz_recom_display = "N/A (Rate Limited)"
                else:
                    auto_sentiment_score = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
                    auto_expert_score = EXPERT_RATING_MAP.get(finviz_data['recom'], 50)
                    finviz_recom_display = finviz_data['recom']

                auto_sentiment_score_placeholder.markdown(f"**Automated Sentiment:** `{auto_sentiment_score}`")
                auto_expert_score_placeholder.markdown(f"**Automated Expert Rating:** `{auto_expert_score}` ({finviz_recom_display})")
                
                sentiment_score = st.sidebar.slider("Adjust Final Sentiment Score", 1, 100, auto_sentiment_score)
                expert_score = st.sidebar.slider("Adjust Final Expert Score", 1, 100, auto_expert_score)
            else:
                st.sidebar.info("Automation OFF. Manual scores are used for display, but only technical score contributes to confidence.")
                sentiment_score = st.sidebar.slider("Manual Sentiment Score", 1, 100, 50)
                expert_score = st.sidebar.slider("Manual Expert Score", 1, 100, 50)
            # --- End Dynamic Qualitative Score Calculation ---

            hist_data, info_data = get_data(ticker_to_analyze, selected_params_main['period'], selected_params_main['interval'])
            if hist_data is None or info_data is None:
                st.error(f"Could not fetch data for {ticker_to_analyze} on a {selected_params_main['interval']} interval. Please check the ticker symbol or try again later.")
            else:
                # Calculate indicators once for the main display
                is_intraday_data = selected_params_main['interval'] in ['5m', '60m']
                df_calculated = calculate_indicators(hist_data.copy(), is_intraday_data)
                
                # Calculate pivot points separately for display
                df_pivots = calculate_pivot_points(hist_data.copy()) # Use original hist_data for pivots

                if df_calculated.empty:
                    st.warning("No data available after indicator calculations and cleaning. Please check ticker or time period.", icon="⚠️")
                    st.stop()

                # Calculate scores for display
                last_row_for_signals = df_calculated.iloc[-1]
                signals_for_score = generate_signals_for_row(last_row_for_signals, indicator_selection, df_calculated, is_intraday_data)
                
                technical_score = (sum(1 for f in signals_for_score.values() if f) / len(signals_for_score)) * 100 if signals_for_score else 0
                
                scores = {"technical": technical_score, "sentiment": sentiment_score, "expert": expert_score}
                
                # Apply weights for overall confidence
                final_weights = selected_params_main['weights'].copy()
                if not use_automation:
                    final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0} # Only technical counts if automation is off
                
                overall_confidence = min(round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)

                # Display tabs
                tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log"]
                main_tab, trade_tab, backtest_tab, news_tab, log_tab = st.tabs(tab_list)

                with main_tab:
                    # Pass df_pivots to main analysis tab for display
                    display_main_analysis_tab(ticker_to_analyze, df_calculated, info_data, selected_params_main, indicator_selection, overall_confidence, scores, final_weights, sentiment_score, expert_score, df_pivots, show_finviz_link=use_automation)
                
                with trade_tab:
                    display_trade_plan_options_tab(ticker_to_analyze, df_calculated, overall_confidence)
                
                with backtest_tab:
                    # Pass is_intraday=False to display_backtest_tab because backtest is always daily data
                    display_backtest_tab(ticker_to_analyze, indicator_selection)
                
                with news_tab:
                    display_news_info_tab(ticker_to_analyze, info_data, finviz_data)
                
                with log_tab:
                    display_trade_log_tab(LOG_FILE, ticker_to_analyze, timeframe, overall_confidence)

        except Exception as e:
            st.error(f"An unexpected error occurred during data processing for {ticker_to_analyze}: {e}", icon="🚫")
            st.exception(e)
    else:
        st.info("Please enter a stock ticker in the sidebar and click 'Analyze Ticker' to begin analysis.")
else:
    st.info("Enter a stock ticker in the sidebar and click 'Analyze Ticker' to begin analysis.")
