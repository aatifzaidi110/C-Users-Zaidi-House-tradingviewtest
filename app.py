# app.py - Version 1.25
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Needed for plt.close() in display_components
import yfinance as yf # Keep this import here for direct yf usage if any, though it's also in utils
import time # Import time for delays

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
    generate_option_trade_plan, convert_compound_to_100_scale, EXPERT_RATING_MAP,
    get_moneyness, analyze_options_chain, convert_finviz_recom_to_score # Import the new function
)
try:
    from display_components import (
        display_main_analysis_tab, display_trade_plan_options_tab,
        display_backtest_tab, display_news_info_tab, display_trade_log_tab,
        display_option_calculator_tab # Ensure this is imported for the calculator tab
    )
    from glossary_components import display_glossary_tab
except ImportError as e:
    print("Import error details:", str(e))
    raise

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === Refresh Button (Global) ===
if st.button("🔄 Clear Cache & Refresh Data", help="Click to clear all cached data and re-run analysis for the current ticker."):
    st.cache_data.clear() # Clear all cached data
    st.rerun() # Rerun the app

# === Constants and Configuration ===
LOG_FILE = "trade_log.csv" # Define here or pass from a config module

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter a Ticker Symbol", value="NVDA").upper()

timeframe = st.sidebar.radio("Choose Trading Style:", ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"], index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50, 200)", value=True),
        "Ichimoku Cloud": st.checkbox("Ichimoku Cloud", value=False, disabled=True),
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
        "Pivot Points": st.checkbox("Pivot Points Display (Daily only)", value=True, disabled=(timeframe not in ["Scalp Trading", "Day Trading"])), # Pivot points are generally not used for intraday
    })

st.sidebar.header("🧠 Qualitative Scores")
use_automation = st.sidebar.toggle("Enable Automated Scoring", value=True, help="ON: AI scores are used. OFF: Use manual sliders and only the Technical Score will count.")

include_finviz_sentiment = st.sidebar.checkbox("Include Finviz Sentiment", value=True, disabled=not use_automation)
include_finviz_expert = st.sidebar.checkbox("Include Finviz Expert Rating", value=True, disabled=not use_automation)


if not use_automation:
    sentiment_score_manual = st.sidebar.slider("Manual Sentiment Score", 0, 100, 50, help="Adjust if automated sentiment is off.")
    expert_score_manual = st.sidebar.slider("Manual Expert Score", 0, 100, 50, help="Adjust if automated expert rating is off.")
else:
    sentiment_score_manual = 50
    expert_score_manual = 50

# === Main Analysis Trigger ===
if st.button("🚀 Analyze Ticker"):
    if not ticker:
        st.warning("Please enter a ticker symbol to begin analysis.", icon="⚠️")
    else:
        TIMEFRAME_MAP = {
            "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
            "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
            "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
            "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
        }
        selected_params_main = TIMEFRAME_MAP[timeframe]

        st.subheader(f"📈 Analysis for {ticker}")

        sentiment_score_current = sentiment_score_manual
        expert_score_current = expert_score_manual
        finviz_data = {"headlines": ["Automated scoring is disabled."], "recom": "N/A", "sentiment_compound": 0}

        if use_automation and (include_finviz_sentiment or include_finviz_expert):
            finviz_data = get_finviz_data(ticker)
            if finviz_data.get('error'): # Check for error message from get_finviz_data
                st.warning(f"Could not fetch live Finviz data for sentiment and expert rating due to: {finviz_data['error']}. Using default/manual scores.", icon="⚠️")
                # Keep sentiment_score_current and expert_score_current as their initial values (manual or 50)
            else:
                if include_finviz_sentiment:
                    sentiment_score_current = convert_compound_to_100_scale(finviz_data['sentiment_compound'])
                else:
                    sentiment_score_current = 0
                
                if include_finviz_expert:
                    # Use the new conversion function for expert score
                    expert_score_current = convert_finviz_recom_to_score(finviz_data['recom'])
                else:
                    expert_score_current = 0
        
        # Debugging print statements
        print(f"DEBUG: use_automation: {use_automation}")
        print(f"DEBUG: include_finviz_sentiment: {include_finviz_sentiment}")
        print(f"DEBUG: include_finviz_expert: {include_finviz_expert}")
        print(f"DEBUG: finviz_data (raw): {finviz_data}")
        print(f"DEBUG: sentiment_score_current (after Finviz): {sentiment_score_current}")
        print(f"DEBUG: expert_score_current (after Finviz): {expert_score_current}")


        final_weights = selected_params_main['weights'].copy()
        if not use_automation:
            final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0}
        else:
            if not include_finviz_sentiment:
                final_weights['sentiment'] = 0.0
            if not include_finviz_expert:
                final_weights['expert'] = 0.0
            total_active_weight = final_weights['technical'] + final_weights['sentiment'] + final_weights['expert']
            if total_active_weight > 0:
                final_weights['technical'] /= total_active_weight
                final_weights['sentiment'] /= total_active_weight
                final_weights['expert'] /= total_active_weight
        
        print(f"DEBUG: Final Weights: {final_weights}")


        try:
            hist_data, info_data = get_data(ticker, selected_params_main['period'], selected_params_main['interval'])
            if hist_data is None or info_data is None:
                st.error(f"Could not fetch data for {ticker} on a {selected_params_main['interval']} interval. Please check the ticker symbol or try again later.")
            else:
                is_intraday_data = selected_params_main['interval'] in ['5m', '60m']
                df_calculated = calculate_indicators(hist_data.copy(), is_intraday_data)
                
                df_pivots = calculate_pivot_points(hist_data.copy())

                if df_calculated.empty:
                    st.warning(f"No data available for {ticker} after indicator calculations and cleaning. Please check ticker or time period.", icon="⚠️")
                    time.sleep(1)
                    st.stop()
                
                last_row_for_signals = df_calculated.iloc[-1]
                current_price = last_row_for_signals['Close']
                prev_close = hist_data['Close'].iloc[-2] if len(hist_data) >= 2 else current_price # Get previous close
                
                signals_for_score = generate_signals_for_row(last_row_for_signals, indicator_selection, df_calculated, is_intraday_data)
                
                technical_score = (sum(1 for f in signals_for_score.values() if f) / len(signals_for_score)) * 100 if signals_for_score else 0
                
                scores = {"technical": technical_score, "sentiment": sentiment_score_current, "expert": expert_score_current}
                
                overall_confidence = min(round((final_weights["technical"]*scores["technical"] + final_weights["sentiment"]*scores["sentiment"] + final_weights["expert"]*scores["expert"]), 2), 100)

                print(f"DEBUG: Technical Score: {technical_score}")
                print(f"DEBUG: Scores (before display): {scores}")
                print(f"DEBUG: Overall Confidence: {overall_confidence}")


                tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧮 Option Calculator", "🧪 Backtest", "📰 News & Info", "📝 Trade Log", "📚 Glossary"] # Added Option Calculator tab
                main_tab, trade_tab, option_calculator_tab, backtest_tab, news_tab, log_tab, glossary_tab = st.tabs(tab_list)

                with main_tab:
                    display_main_analysis_tab(ticker, df_calculated, info_data, selected_params_main, indicator_selection, overall_confidence, scores, final_weights, sentiment_score_current, expert_score_current, df_pivots, use_automation and (include_finviz_sentiment or include_finviz_expert))
                
                with trade_tab:
                    display_trade_plan_options_tab(ticker, df_calculated, overall_confidence)
                
                with option_calculator_tab: # New tab for Option Calculator
                    # Pass the required arguments including current_price, prev_close, overall_confidence
                    display_option_calculator_tab(ticker, current_price, info_data.get('options', []), prev_close, overall_confidence)
                
                with backtest_tab:
                    display_backtest_tab(ticker, indicator_selection, current_price, prev_close, overall_confidence)
                
                with news_tab:
                    display_news_info_tab(ticker, info_data, finviz_data, current_price, prev_close, overall_confidence)
                
                with log_tab:
                    display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence, current_price, prev_close)

                with glossary_tab:
                    display_glossary_tab(current_stock_price=current_price)
            
        except Exception as e:
            st.error(f"An unexpected error occurred during data processing for {ticker}: {e}", icon="🚫")
            st.exception(e)
            time.sleep(1)
