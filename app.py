# app.py - Final Version
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Needed for plt.close() in display_components
import yfinance as yf # Keep this import here for direct yf usage if any, though it's also in utils
import datetime # Import datetime for date calculations

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
    generate_signals_for_row, backtest_strategy, get_moneyness, analyze_options_chain,
    suggest_options_strategy,
    calculate_confidence_score, # Ensure calculate_confidence_score is imported from utils
    calculate_economic_score, # NEW
    calculate_sentiment_score, # NEW
    run_stock_scanner, # NEW
    generate_directional_trade_plan # Ensure this is imported for the scanner
)
try:
    from display_components import (
        display_main_analysis_tab, display_trade_plan_options_tab,
        display_backtest_tab, display_news_info_tab, display_trade_log_tab,
        display_option_calculator_tab,
        display_economic_data_tab, # NEW
        display_investor_sentiment_tab, # NEW
        display_scanner_results_tab # NEW
    )
except ImportError as e:
    print("Import error details:", str(e))
    raise

# === Page Setup ===
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide", theme="dark") # Set dark theme
st.title("🚀 Aatif's AI-Powered Trading Hub")

# === Constants and Configuration ===
LOG_FILE = "trade_log.csv" # Define here or pass from a config module

# === SIDEBAR: Controls & Selections ===
st.sidebar.header("⚙️ Controls")
ticker_input = st.sidebar.text_input("Enter Ticker Symbol", value="NVDA").upper().strip()

# === Buttons (Moved to directly under ticker search bar) ===
with st.sidebar.container():
    if st.button("▶️ Analyze Ticker", help="Click to analyze the entered ticker and display results."):
        st.session_state.analysis_started = True
        st.session_state.scan_started = False # Stop scan if analysis is started
        st.rerun()

    if st.button("🔄 Clear Cache & Refresh Data", help="Click to clear all cached data and re-run analysis from scratch."):
        st.cache_data.clear() # Clear all cached data
        st.session_state.analysis_started = False # Reset analysis state
        st.session_state.scan_started = False # Reset scan state
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

# NEW: Confidence Score Weights (Moved here for better organization)
st.sidebar.subheader("Confidence Score Weights")
weight_technical = st.sidebar.slider("Technical Analysis Weight", 0.0, 1.0, 0.4, 0.05, key="w_tech")
weight_sentiment = st.sidebar.slider("News Sentiment Weight", 0.0, 1.0, 0.2, 0.05, key="w_sent")
weight_expert = st.sidebar.slider("Expert Rating Weight", 0.0, 1.0, 0.2, 0.05, key="w_expert")
weight_economic = st.sidebar.slider("Economic Data Weight", 0.0, 1.0, 0.1, 0.05, key="w_eco") # NEW
weight_investor_sentiment = st.sidebar.slider("Investor Sentiment Weight", 0.0, 1.0, 0.1, 0.05, key="w_inv_sent") # NEW

total_weights = weight_technical + weight_sentiment + weight_expert + weight_economic + weight_investor_sentiment
if total_weights == 0:
    st.sidebar.warning("Total weights cannot be zero. Adjust sliders.")
    # Set a default if all are zero to prevent division by zero
    final_weights = {'technical': 0.25, 'sentiment': 0.25, 'expert': 0.25, 'economic': 0.125, 'investor_sentiment': 0.125}
else:
    final_weights = {
        'technical': weight_technical / total_weights,
        'sentiment': weight_sentiment / total_weights,
        'expert': weight_expert / total_weights,
        'economic': economic_weight / total_weights, # Corrected variable name
        'investor_sentiment': weight_investor_sentiment / total_weights
    }


# === NEW: Stock Scanner Controls ===
st.sidebar.markdown("---")
st.sidebar.header("🔍 Stock Scanner")
scanner_tickers_input = st.sidebar.text_area("Tickers to Scan (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN,NVDA").upper().strip()
scanner_trading_style = st.sidebar.selectbox(
    "Scanner Trading Style",
    ["Day Trading Long", "Day Trading Short", "Swing Trading Call", "Swing Trading Put"],
    key="scanner_style"
)
min_scanner_confidence = st.sidebar.slider("Minimum Scanner Confidence (%)", 0, 100, 70, 5, key="min_scan_conf")

if st.sidebar.button("🚀 Run Scanner", help="Scan multiple tickers for opportunities."):
    st.session_state.scan_started = True
    st.session_state.analysis_started = False # Stop single ticker analysis if scan is started
    st.session_state.scanner_results = pd.DataFrame() # Clear previous results
    st.rerun()


# Initialize session state for analysis and scan control
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'scan_started' not in st.session_state:
    st.session_state.scan_started = False
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = pd.DataFrame()


# Main Script Execution
TIMEFRAME_MAP = {
    "Scalp Trading": {"period": "5d", "interval": "5m"},
    "Day Trading": {"period": "60d", "interval": "60m"},
    "Swing Trading": {"period": "1y", "interval": "1d"},
    "Position Trading": {"period": "5y", "interval": "1wk"}
}
selected_params_main = TIMEFRAME_MAP[timeframe]

# --- Main Analysis Logic ---
if st.session_state.analysis_started and ticker_input:
    ticker = ticker_input

    # Fetch Finviz data once for the main analysis flow
    finviz_data = get_finviz_data(ticker)

    # Fetch Economic Data
    today = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)
    
    gdp_data = get_economic_data_fred('GDP', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    cpi_data = get_economic_data_fred('CPIAUCSL', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    unemployment_data = get_economic_data_fred('UNRATE', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))

    latest_gdp = gdp_data.iloc[-1] if gdp_data is not None and not gdp_data.empty else None
    latest_cpi = cpi_data.iloc[-1] if cpi_data is not None and not cpi_data.empty else None
    latest_unemployment = unemployment_data.iloc[-1] if unemployment_data is not None and not unemployment_data.empty else None

    economic_score_current = calculate_economic_score(latest_gdp, latest_cpi, latest_unemployment)

    # Fetch Investor Sentiment Data (VIX example)
    vix_data = get_vix_data(one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    latest_vix = vix_data['Close'].iloc[-1] if vix_data is not None and not vix_data.empty else None
    historical_vix_avg = vix_data['Close'].mean() if vix_data is not None and not vix_data.empty else None

    investor_sentiment_score_current = calculate_sentiment_score(latest_vix, historical_vix_avg)

    try:
        hist_data, info_data = get_data(ticker, selected_params_main['period'], selected_params_main['interval'])
        if hist_data.empty or not info_data:
            st.error(f"Could not fetch data for {ticker} on a {selected_params_main['interval']} interval. Please check the ticker symbol or try again later.")
        else:
            is_intraday_data = selected_params_main['interval'] in ['5m', '60m']
            df_calculated = calculate_indicators(hist_data.copy(), is_intraday_data)
            df_pivots = calculate_pivot_points(hist_data.copy())

            if df_calculated.empty:
                st.warning(f"No data available for {ticker} after indicator calculations and cleaning. Please check ticker or time period.", icon="⚠️")
            else:
                last_row_for_signals = df_calculated.iloc[-1]
                bullish_signals, bearish_signals = generate_signals_for_row(last_row_for_signals)

                # Calculate the technical score component
                tech_score_raw = 0
                total_possible_tech_points = 0
                all_indicator_names = [
                    "EMA Trend", "Ichimoku Cloud", "Parabolic SAR", "ADX",
                    "RSI Momentum", "Stochastic", "MACD", "Volume Spike",
                    "CCI", "ROC", "OBV", "VWAP"
                ]
                for ind_name in all_indicator_names:
                    is_selected_in_sidebar = indicator_selection.get(ind_name, False)
                    if is_selected_in_sidebar:
                        total_possible_tech_points += 1
                        if bullish_signals.get(ind_name, False):
                            tech_score_raw += 1
                        elif bearish_signals.get(ind_name, False):
                            tech_score_raw -= 1
                
                technical_score_current = 50 # Default neutral
                if total_possible_tech_points > 0:
                    technical_score_current = ((tech_score_raw + total_possible_tech_points) / (2 * total_possible_tech_points)) * 100

                # Use the fetched Finviz sentiment and expert scores directly if automation is on
                # Otherwise, use manual sliders (which are now handled by passing to calculate_confidence_score)
                sentiment_score_for_confidence = finviz_data.get("sentiment_compound", 0) * 100
                expert_score_for_confidence = (lambda recom_str: (100 - (float(recom_str) - 1) * 25) if recom_str != "N/A" else 50)(finviz_data.get("recom_str", "N/A"))


                # Calculate overall confidence using the unified function
                confidence_results = calculate_confidence_score(
                    technical_score_current,
                    sentiment_score_for_confidence if use_automation else st.session_state.get('manual_sentiment_score', 50),
                    expert_score_for_confidence if use_automation else st.session_state.get('manual_expert_score', 50),
                    economic_score_current,
                    investor_sentiment_score_current,
                    final_weights
                )
                overall_confidence = confidence_results['score']
                trade_direction = confidence_results['direction']
                scores = confidence_results['components'] # This now contains all 5 component scores

                st.subheader(f"📈 Analysis for {ticker}")
                tab_list = ["📊 Main Analysis", "📈 Trade Plan & Options", "🧪 Backtest", "📰 News & Info", "📝 Trade Log", "🧮 Option Calculator", "📊 Economic Data", "❤️ Investor Sentiment", "📚 Glossary"] # Added Glossary tab
                main_tab, trade_tab, backtest_tab, news_tab, log_tab, option_calc_tab, economic_tab, investor_sentiment_tab, glossary_tab = st.tabs(tab_list) # Added glossary_tab

                with main_tab:
                    display_main_analysis_tab(
                        ticker, df_calculated, info_data, selected_params_main, 
                        indicator_selection, overall_confidence, scores, final_weights, 
                        sentiment_score_for_confidence, expert_score_for_confidence, df_pivots, trade_direction
                    )
                
                with trade_tab:
                    display_trade_plan_options_tab(ticker, df_calculated, overall_confidence, timeframe, trade_direction)
                
                with backtest_tab:
                    st.markdown("---")
                    st.subheader("🧪 Historical Backtest Parameters")
                    backtest_direction_radio = st.radio(
                        f"Select Backtest Direction for {ticker}:",
                        ["Long (Bullish)", "Short (Bearish)"],
                        key=f"backtest_direction_{ticker}",
                        horizontal=True
                    )
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_backtest_tab(ticker, indicator_selection, current_price, prev_close, overall_confidence, backtest_direction_radio.split(' ')[0].lower())
                
                with news_tab:
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_news_info_tab(ticker, info_data, finviz_data, current_price, prev_close, overall_confidence, trade_direction)
                
                with log_tab:
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence, current_price, prev_close, trade_direction)
                
                with option_calc_tab:
                    current_stock_price = df_calculated.iloc[-1]['Close']
                    stock_obj_for_options = yf.Ticker(ticker)
                    expirations = stock_obj_for_options.options
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_stock_price
                    display_option_calculator_tab(ticker, current_stock_price, expirations, prev_close, overall_confidence, trade_direction)

                with economic_tab: # NEW Economic Data Tab
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_economic_data_tab(ticker, current_price, prev_close, overall_confidence, trade_direction,
                                              latest_gdp, latest_cpi, latest_unemployment)

                with investor_sentiment_tab: # NEW Investor Sentiment Tab
                    current_price = df_calculated.iloc[-1]['Close']
                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) > 1 else current_price
                    display_investor_sentiment_tab(ticker, current_price, prev_close, overall_confidence, trade_direction,
                                                   latest_vix, historical_vix_avg)
                with glossary_tab: # Added Glossary tab
                    st.markdown("### 📚 Glossary")
                    st.info("The glossary content will be displayed here.") # Placeholder for actual glossary content


    except Exception as e:
        st.error(f"An unexpected error occurred during data processing for {ticker}: {e}", icon="🚫")
        st.exception(e)

# --- Scanner Results Logic ---
elif st.session_state.scan_started and scanner_tickers_input:
    st.subheader(f"🔍 Scan Results for {scanner_trading_style}")
    tickers_to_scan = [t.strip() for t in scanner_tickers_input.split(',') if t.strip()]
    
    if not tickers_to_scan:
        st.warning("Please enter at least one ticker to scan.")
    else:
        with st.spinner(f"Running scanner for {scanner_trading_style} on {len(tickers_to_scan)} tickers... This may take a while."):
            # Pass the indicator_selection and final_weights to the scanner
            scanner_results_df = run_stock_scanner(
                tickers_to_scan,
                scanner_trading_style,
                min_scanner_confidence,
                indicator_selection, # Pass the current indicator selection
                final_weights # Pass the current confidence weights
            )
            st.session_state.scanner_results = scanner_results_df # Store results in session state

        if not st.session_state.scanner_results.empty:
            display_scanner_results_tab(st.session_state.scanner_results)
        else:
            st.info(f"No tickers found matching the criteria for '{scanner_trading_style}' with confidence >= {min_scanner_confidence}%.")
else:
    st.info("Enter a stock ticker in the sidebar and click 'Analyze Ticker' to begin analysis, or use the 'Stock Scanner' to find opportunities across multiple tickers.")


