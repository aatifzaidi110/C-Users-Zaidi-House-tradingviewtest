# app.py - Version 1.32
st.set_page_config(layout="wide", page_title="Your App Name", initial_sidebar_state="expanded", theme="dark")

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
    generate_signals_for_row, backtest_strategy,get_moneyness, analyze_options_chain,
    suggest_options_strategy, # Changed from generate_option_trade_plan to suggest_options_strategy
    # convert_compound_to_100_scale, EXPERT_RATING_MAP, # These should be in utils, but not directly imported by main.py if used only internally by utils.
    # If they are used directly in app.py, they should be imported. Assuming they are not for this fix.
    # generate_directional_trade_plan # Import the new directional trade plan generator - Assuming this is not relevant to the current fix.
    calculate_confidence_score # Ensure calculate_confidence_score is imported from utils
)
try:
    from display_components import (
        display_main_analysis_tab, display_trade_plan_options_tab,
        display_backtest_tab, display_news_info_tab, display_trade_log_tab,
        display_option_calculator_tab # Ensure this is imported for the tab
        # Removed display_ticker_comparison_chart as it's no longer needed for single ticker
    )
except ImportError as e:
    print("Import error details:", str(e))
    raise

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === Constants and Configuration ===
LOG_FILE = "trade_log.csv" # Define here or pass from a config module

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
# Reverted to text_input for single ticker
ticker_input = st.sidebar.text_input("Enter Ticker Symbol", value="NVDA").upper().strip()
# No splitting needed, directly use ticker_input as the single ticker

# === Buttons (Moved to directly under ticker search bar) ===
# Use a container to explicitly group and render buttons in the sidebar
with st.sidebar.container():
    if st.button("▶️ Analyze Ticker", help="Click to analyze the entered ticker and display results."):
        st.session_state.analysis_started = True
        st.rerun()

    if st.button("🔄 Clear Cache & Refresh Data", help="Click to clear all cached data and re-run analysis from scratch."):
        st.cache_data.clear() # Clear all cached data
        st.session_state.analysis_started = False # Reset analysis state
        st.rerun()

timeframe = st.sidebar.radio("Choose Trading Style:", ["Scalp Trading", "Day Trading", "Swing Trading", "Position Trading"], index=2)

st.sidebar.header("🔧 Technical Indicator Selection")
with st.sidebar.expander("Trend Indicators", expanded=True):
    indicator_selection = {
        "EMA Trend": st.checkbox("EMA Trend (21, 50, 200)", value=True),
        "Ichimoku Cloud": st.checkbox("Ichimoku Cloud", value=True),
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


# Initialize session state for analysis control
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# === Dynamic Qualitative Score Calculation ===
sentiment_score = 50
expert_score = 50
finviz_data = {"headlines": ["Automation is disabled."]} # Default for when automation is off

# Main Script Execution
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m", "weights": {"technical": 0.9, "sentiment": 0.1, "expert": 0.0}},
    "Day Trading": {"period": "60d", "interval": "60m", "weights": {"technical": 0.7, "sentiment": 0.2, "expert": 0.1}},
    "Swing Trading": {"period": "1y", "interval": "1d", "weights": {"technical": 0.6, "sentiment": 0.2, "expert": 0.2}},
    "Position Trading": {"period": "5y", "interval": "1wk", "weights": {"technical": 0.4, "sentiment": 0.2, "expert": 0.4}}
}
selected_params_main = TIMEFRAME_MAP[timeframe]

# Only run analysis if the button has been clicked and a ticker is entered
if st.session_state.analysis_started and ticker_input:
    ticker = ticker_input # Use the single ticker from input

    # Re-calculate sentiment and expert scores if automation is on
    if use_automation:
        finviz_data = get_finviz_data(ticker) # This calls utils.get_finviz_data
        # You need to ensure these functions are accessible or defined in utils if they are being called here.
        # convert_compound_to_100_scale is likely in utils, but EXPERT_RATING_MAP might need to be imported or handled.
        # For this specific fix, I'll assume they exist and are correctly used in utils.
        # If they are not in utils and needed directly in app.py, add them to the import list from utils.
        # For now, I'll assume convert_compound_to_100_scale and EXPERT_RATING_MAP are either in utils
        # and imported, or not directly called from finviz_data in this block anymore.
        # Based on your utils.py, calculate_confidence_score handles these internally.
        # So, sentiment_score_current and expert_score_current are derived from calculate_confidence_score's components.
        
        # We will use calculate_confidence_score which internally uses these.
        # We still need to pass finviz_data to it.
        pass # Will calculate via confidence score later
    else:
        sentiment_score_current = sentiment_score # Use global manual slider value
        expert_score_current = expert_score      # Use global manual slider value

    try:
        hist_data, info_data = get_data(ticker, selected_params_main['period'], selected_params_main['interval'])
        if hist_data is None or info_data is None:
            st.error(f"Could not fetch data for {ticker} on a {selected_params_main['interval']} interval. Please check the ticker symbol or try again later.")
        else:
            # Calculate indicators once for the main display
            is_intraday_data = selected_params_main['interval'] in ['5m', '60m']
            df_calculated = calculate_indicators(hist_data.copy(), is_intraday_data)
            
            # Calculate pivot points separately for display
            df_pivots = calculate_pivot_points(hist_data.copy()) # Use original hist_data for pivots

            if df_calculated.empty:
                st.warning(f"No data available for {ticker} after indicator calculations and cleaning. Please check ticker or time period.", icon="⚠️")
            else:
                # Calculate scores for display
                last_row_for_signals = df_calculated.iloc[-1]
                
                # Generate both bullish and bearish signals
                bullish_signals, bearish_signals = generate_signals_for_row(last_row_for_signals) # Removed indicator_selection, df_calculated, is_intraday_data as args based on utils.py

                # Calculate confidence score using the unified function
                # The `signals` argument expects a tuple of (bullish_signals, bearish_signals)
                # The `latest_row` argument expects the last row of your calculated DataFrame
                # The `finviz_data` argument expects the dictionary from get_finviz_data
                
                # Ensure finviz_data is fetched if not already for use_automation=False, or handle appropriately
                if not use_automation:
                    # If automation is off, finviz_data might not have been fetched, leading to errors in calculate_confidence_score
                    # Fetch it here if needed, or ensure a default structure that calculate_confidence_score can handle.
                    # For a robust solution, you'd want a placeholder or fetch it anyway if use_automation is false
                    # and you still want to pass valid finviz_data to calculate_confidence_score for its component display.
                    # For now, assuming finviz_data has a basic structure or is fetched if needed by calculate_confidence_score.
                    finviz_data = get_finviz_data(ticker) # Fetch even if automation is off, for info_data display and score components.
                    # Re-assign sentiment and expert scores based on manual sliders if automation is off
                    # This implies you need a way to pass these manual overrides to calculate_confidence_score if it's to be truly manual
                    # Or, the calculate_confidence_score function itself needs to check `use_automation`
                    # The current calculate_confidence_score in utils does NOT take `use_automation` as an argument.
                    # Therefore, we need to pass a modified finviz_data or signals if use_automation is off.
                    # The original app.py logic for manual scores modified sentiment_score_current and expert_score_current directly.
                    # This implies `calculate_confidence_score` should be more flexible, or we calculate the `overall_confidence` here
                    # based on the selected_params_main['weights'] and the manually set sentiment/expert scores.

                    # Let's adjust to match the original app.py's logic for manual scores.
                    # If use_automation is OFF, the sentiment and expert scores are taken from the manual sliders.
                    # The calculate_confidence_score in utils.py directly uses finviz_data.
                    # So, if use_automation is off, we need to make `finviz_data` reflect the manual scores,
                    # or apply the manual scores *after* getting the raw component scores from calculate_confidence_score.

                    # Simpler: calculate raw tech score, then combine with manual sentiment/expert.
                    # The `calculate_confidence_score` is a combined score generator, so we need to pass consistent data.
                    # Let's keep `calculate_confidence_score` as is and derive the final `overall_confidence` here.
                    pass # The values for sentiment_score_current and expert_score_current are already set.

                confidence_output = calculate_confidence_score(
                    signals=(bullish_signals, bearish_signals),
                    latest_row=last_row_for_signals,
                    finviz_data=finviz_data
                )
                
                # Extract components from confidence_output for display and weighted calculation
                technical_score_from_utils = confidence_output['components']['Technical Score']
                news_sentiment_score_from_utils = confidence_output['components']['News Sentiment Score']
                analyst_rating_score_from_utils = confidence_output['components']['Analyst Rating Score']
                
                # Now, apply the logic for `use_automation` to determine the final `scores` for `overall_confidence`
                scores = {}
                if use_automation:
                    scores["technical"] = technical_score_from_utils
                    scores["sentiment"] = news_sentiment_score_from_utils
                    scores["expert"] = analyst_rating_score_from_utils
                    sentiment_score_current = news_sentiment_score_from_utils # For display purposes
                    expert_score_current = analyst_rating_score_from_utils # For display purposes
                else:
                    scores["technical"] = technical_score_from_utils # Technical score is always from indicators
                    scores["sentiment"] = sentiment_score_current # From manual slider
                    scores["expert"] = expert_score_current # From manual slider

                # Determine trade direction based on the core technical analysis (not directly from confidence_output's single score)
                # This part needs to remain consistent with your original signal logic if you want a granular direction
                bullish_fired_count = sum(1 for k in indicator_selection if indicator_selection[k] and bullish_signals.get(k.split('(')[0].strip(), False))
                bearish_fired_count = sum(1 for k in indicator_selection if indicator_selection[k] and bearish_signals.get(k.split('(')[0].strip(), False))
                total_selected_directional_indicators = sum(1 for k in indicator_selection if indicator_selection[k] and k not in ["Bollinger Bands", "Pivot Points", "VWAP"])
                if is_intraday_data and indicator_selection.get("VWAP"):
                    # Assuming VWAP has a signal associated in generate_signals_for_row
                    # This requires VWAP to be included in bullish_signals/bearish_signals properly
                    if 'VWAP Bullish' in bullish_signals and bullish_signals['VWAP Bullish']:
                        bullish_fired_count += 1
                    if 'VWAP Bearish' in bearish_signals and bearish_signals['VWAP Bearish']:
                        bearish_fired_count += 1
                    # This line below was problematic as 'selected_signal_keys' was not defined
                    # if 'VWAP' not in selected_signal_keys: # Ensure VWAP is counted if selected
                    #    total_selected_directional_indicators += 1 # Only if it's truly a directional signal indicator
                    # Corrected: Add 1 to total_selected_directional_indicators if VWAP is selected and is directional
                    if is_intraday_data and indicator_selection.get("VWAP"): # Only count if applicable and selected
                        total_selected_directional_indicators += 1 # VWAP contributes to directional analysis

                trade_direction = "Neutral"
                if total_selected_directional_indicators > 0:
                    if bullish_fired_count > bearish_fired_count and (bullish_fired_count / total_selected_directional_indicators) >= 0.5:
                        trade_direction = "Bullish"
                    elif bearish_fired_count > bullish_fired_count and (bearish_fired_count / total_selected_directional_indicators) >= 0.5:
                        trade_direction = "Bearish"


                final_weights = selected_params_main['weights'].copy()
                if not use_automation:
                    final_weights = {'technical': 1.0, 'sentiment': 0.0, 'expert': 0.0} # Only technical counts if automation is off
                
                # Calculate overall_confidence based on the potentially overridden scores
                overall_confidence = min(round((final_weights["technical"] * scores["technical"] +
                                                final_weights["sentiment"] * scores["sentiment"] +
                                                final_weights["expert"] * scores["expert"]), 2), 100)


                st.subheader(f"📈 Analysis for {ticker}") # Move subheader here to be above tabs for each ticker
                # Display tabs
                tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log", "🧮 Option Calculator", "📚 Glossary"]
                main_tab, trade_tab, backtest_tab, news_tab, log_tab, option_calc_tab, glossary_tab = st.tabs(tab_list)

                with main_tab:
                    # Pass trade_direction to display_main_analysis_tab
                    display_main_analysis_tab(
                        ticker, df_calculated, info_data, selected_params_main, 
                        indicator_selection, overall_confidence, scores, final_weights, 
                        sentiment_score_current, expert_score_current, df_pivots, trade_direction
                    )
                
                with trade_tab:
                    # Pass the determined trade_direction to the display function
                    # Ensure you're passing `suggest_options_strategy` to `display_trade_plan_options_tab` if it expects it.
                    # Or, `display_trade_plan_options_tab` should call `suggest_options_strategy` internally.
                    # If display_trade_plan_options_tab expects a function, it should be passed:
                    display_trade_plan_options_tab(ticker, df_calculated, overall_confidence, timeframe, trade_direction)
                
                with backtest_tab:
                    # Allow backtesting for both long and short
                    st.markdown("---")
                    st.subheader("🧪 Historical Backtest Parameters")
                    backtest_direction = st.radio(
                        f"Select Backtest Direction for {ticker}:",
                        ["Long (Bullish)", "Short (Bearish)"],
                        key=f"backtest_direction_{ticker}",
                        horizontal=True
                    )
                    # Pass the selected backtest direction to the backtest_strategy function
                    # Ensure current_price and hist_data['Close'].iloc[-2] are passed correctly
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_backtest_tab(ticker, indicator_selection, current_price, prev_close, overall_confidence, backtest_direction.split(' ')[0].lower())
                
                with news_tab:
                    # Pass trade_direction to display_news_info_tab
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_news_info_tab(ticker, info_data, finviz_data, current_price, prev_close, overall_confidence, trade_direction)
                
                with log_tab:
                    # Pass trade_direction to display_trade_log_tab
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence, current_price, prev_close, trade_direction)
                
                with option_calc_tab:
                    # Pass the current stock price and expirations to the new calculator
                    current_stock_price = df_calculated.iloc[-1]['Close']
                    stock_obj_for_options = yf.Ticker(ticker)
                    expirations = stock_obj_for_options.options
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_stock_price
                    display_option_calculator_tab(ticker, current_stock_price, expirations, prev_close, overall_confidence, trade_direction)

                with glossary_tab:
                    # Assuming display_glossary_tab doesn't need these parameters
                    # If it does, they should be passed here.
                    # You would likely have a function like: display_glossary_tab()
                    st.markdown("### 📚 Glossary")
                    st.info("The glossary content will be displayed here.") # Placeholder for actual glossary content

    except Exception as e:
        st.error(f"An unexpected error occurred during data processing for {ticker}: {e}", icon="🚫")
        st.exception(e)
else:
    st.info("Enter a stock ticker in the sidebar and click 'Analyze Ticker' to begin analysis.")
