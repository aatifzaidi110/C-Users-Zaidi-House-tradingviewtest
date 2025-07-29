# app.py - Final Version (v1.35 - Corrected for Syntax and Logic)
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Needed for plt.close() in display_components
import yfinance as yf # Keep this import here for direct yf usage if any, though it's also in utils
from datetime import datetime, date, timedelta # Import the datetime module

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
    suggest_options_strategy, generate_directional_trade_plan,
    calculate_confidence_score, convert_finviz_recom_to_score,
    get_economic_data_fred, get_vix_data, calculate_economic_score, calculate_sentiment_score,
    scan_for_trades # Renamed from run_stock_scanner for consistency with utils.py
)

from display_components import (
    display_technical_analysis_tab, display_options_analysis_tab,
    display_backtesting_tab, display_trade_log_tab,
    display_economic_data_tab, display_investor_sentiment_tab,
    display_scanner_tab # Ensure display_scanner_tab is imported
)

# --- Configuration ---
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Advanced Stock Analyzer")

# --- Session State Initialization ---
# Initialize session state for consistent UI across reruns
if 'ticker' not in st.session_state:
    st.session_state.ticker = ""
if 'data_interval' not in st.session_state:
    st.session_state.data_interval = "1d" # Default to daily
if 'start_date' not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=365) # Default to 1 year ago
if 'end_date' not in st.session_state:
    st.session_state.end_date = date.today()
if 'indicator_selection' not in st.session_state:
    st.session_state.indicator_selection = {
        "EMA Trend": True,
        "MACD": True,
        "RSI Momentum": True,
        "Bollinger Bands": False,
        "Stochastic": False,
        "Ichimoku Cloud": False,
        "Parabolic SAR": False,
        "ADX": False,
        "Volume Spike": False,
        "CCI": False,
        "ROC": False,
        "OBV": False,
        "VWAP": False, # VWAP is typically for intraday
        "Pivot Points": False # Pivot Points for daily/weekly
    }
if 'confidence_weights' not in st.session_state:
    st.session_state.confidence_weights = {
        "technical": 0.4,
        "sentiment": 0.2,
        "expert": 0.2, # This 'expert' weight is not directly used by current utils.py confidence score logic
        "economic": 0.1,
        "investor_sentiment": 0.1
    }
if 'trade_plan_result' not in st.session_state:
    st.session_state.trade_plan_result = None # To store the generated trade plan

# --- Main Application Logic Function ---
def main():
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Configuration")

    # Ticker Input
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", st.session_state.ticker).upper()
    st.session_state.ticker = ticker

    # Timeframe Selection
    timeframe_options = {
        "1d": "1 Day (Daily)", "1wk": "1 Week (Weekly)", "1mo": "1 Month (Monthly)",
        "1m": "1 Minute (Intraday)", "2m": "2 Minutes (Intraday)", "5m": "5 Minutes (Intraday)",
        "15m": "15 Minutes (Intraday)", "30m": "30 Minutes (Intraday)", "60m": "60 Minutes (Intraday)",
        "90m": "90 Minutes (Intraday)", "1h": "1 Hour (Intraday)"
    }
    selected_timeframe_key = st.sidebar.selectbox(
        "Select Timeframe",
        list(timeframe_options.keys()),
        format_func=lambda x: timeframe_options[x],
        index=list(timeframe_options.keys()).index(st.session_state.data_interval)
    )
    st.session_state.data_interval = selected_timeframe_key
    selected_timeframe = timeframe_options[selected_timeframe_key]

    # Date Range for Historical Data
    is_intraday = "Intraday" in selected_timeframe

    if not is_intraday:
        st.sidebar.subheader("Historical Data Range")
        start_date = st.sidebar.date_input("Start Date", st.session_state.start_date)
        end_date = st.sidebar.date_input("End Date", st.session_state.end_date)
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
    else:
        st.sidebar.info("For intraday data, the data range is typically limited by the provider (e.g., 7 days for Yahoo Finance).")
        start_date = date.today() - timedelta(days=7) # Yahoo Finance limits intraday to 7 days
        end_date = date.today()
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date


    # Indicator Selection
    st.sidebar.subheader("Technical Indicators")
    for indicator, selected in st.session_state.indicator_selection.items():
        # Disable VWAP and Pivot Points for non-intraday/intraday respectively
        if (indicator == "VWAP" and not is_intraday) or \
           (indicator == "Pivot Points" and is_intraday):
            st.session_state.indicator_selection[indicator] = st.sidebar.checkbox(indicator, value=selected, disabled=True, help="Automatically disabled for this timeframe.")
        else:
            st.session_state.indicator_selection[indicator] = st.sidebar.checkbox(indicator, value=selected)


    # Confidence Score Weights
    st.sidebar.subheader("Confidence Score Weights (%)")
    total_weight = 0
    for component in st.session_state.confidence_weights.keys():
        # Use a unique key for each slider
        # Clamp the initial value to be between 0 and 100 to prevent JSNumberBoundsException
        initial_slider_value = int(st.session_state.confidence_weights[component] * 100)
        clamped_slider_value = max(0, min(100, initial_slider_value))
        
        st.session_state.confidence_weights[component] = st.sidebar.slider(
            f"{component.replace('_', ' ').title()}",
            0, 100, clamped_slider_value, key=f"weight_{component}"
        )
        total_weight += st.session_state.confidence_weights[component]

    # Normalize weights if total is not 100
    normalized_weights = {}
    if total_weight > 0:
        for component, weight in st.session_state.confidence_weights.items():
            normalized_weights[component] = weight / total_weight
    else:
        st.sidebar.warning("Total weight is 0. Please adjust weights.")
        # Default to equal weights if total is 0 to avoid division by zero
        num_components = len(st.session_state.confidence_weights)
        if num_components > 0:
            for component in st.session_state.confidence_weights.keys():
                normalized_weights[component] = 1 / num_components
        else:
            normalized_weights = {component: 0 for component in st.session_state.confidence_weights.keys()} # Should not happen


    # Stock Scanner Configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Stock Scanner")
    scanner_ticker_list_raw = st.sidebar.text_area("Tickers for Scanner (comma-separated)", "AAPL,MSFT,GOOGL")
    scanner_ticker_list = [t.strip().upper() for t in scanner_ticker_list_raw.split(',') if t.strip()]
    selected_trading_style = st.sidebar.selectbox("Scanner Trading Style", ["Swing", "Day", "Long-Term"])
    min_scanner_confidence = st.sidebar.slider("Minimum Scanner Confidence (%)", 0, 100, 70)
    run_scanner_button = st.sidebar.button("Run Stock Scanner")


    # Analyze Ticker Button
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("Analyze Ticker")

    st.title("Advanced Stock Analysis Dashboard")

    if analyze_button and ticker:
        # Fetch data and generate comprehensive trade plan
        with st.spinner(f"Analyzing {ticker} ({selected_timeframe})..."):
            # generate_directional_trade_plan handles all data fetching and indicator calculation internally
            # It returns a dictionary with all the analysis results.
            # We pass the options_expiration_date as None for now, it can be added as a user input later.
            st.session_state.trade_plan_result = generate_directional_trade_plan(
                ticker=st.session_state.ticker,
                interval=st.session_state.data_interval,
                start_date=st.session_state.start_date,
                end_date=st.session_state.end_date,
                indicator_selection=st.session_state.indicator_selection,
                weights=st.session_state.confidence_weights, # Pass raw weights, generate_directional_trade_plan normalizes them
                options_expiration_date=None # Placeholder for user-selected options expiration
            )
        
        trade_plan_result = st.session_state.trade_plan_result

        if trade_plan_result and trade_plan_result.get("current_price") is not None:
            st.success(f"Analysis complete for {ticker}.")

            # Extract key variables from the trade_plan_result for easier access
            # Ensure current_price is a scalar float
            current_price_raw = trade_plan_result.get("current_price")
            current_price = float(current_price_raw.iloc[-1]) if isinstance(current_price_raw, pd.Series) else float(current_price_raw)

            overall_confidence = trade_plan_result.get("overall_confidence")
            trade_direction = trade_plan_result.get("trade_direction")
            
            # Fetch raw data again for plotting, as df_calculated might be filtered/modified
            df = get_data(ticker, st.session_state.data_interval, st.session_state.start_date, st.session_state.end_date)
            df_calculated = calculate_indicators(df.copy(), st.session_state.indicator_selection, is_intraday) # Recalculate for display

            # Get options chain expiration dates
            options_chain_dates = []
            try:
                options_chain_dates = yf.Ticker(ticker).options
            except Exception as e:
                st.warning(f"Could not fetch options chain dates for {ticker}: {e}")

            # Create tabs for different analyses
            tab_titles = [
                "📊 Technical Analysis", "🔮 Options Analysis", "💡 Trade Plan",
                "📜 Trade Log", "🤖 Backtesting", "🌍 Economic & Sentiment", "📚 Glossary"
            ]
            tabs = st.tabs(tab_titles)

            with tabs[0]: # 📊 Technical Analysis
                display_technical_analysis_tab(
                    ticker,
                    df_calculated,
                    is_intraday,
                    st.session_state.indicator_selection, # Pass the full selection dict
                    normalized_weights # Pass normalized weights
                )

            with tabs[1]: # 🔮 Options Analysis
                display_options_analysis_tab(
                    ticker,
                    current_price, # Pass the scalar current_price
                    options_chain_dates, # Pass expirations
                    trade_direction,
                    overall_confidence
                )

            with tabs[2]: # 💡 Trade Plan
                st.subheader("🗺️ Directional Trade Plan (Based on Current Data)")
                if trade_plan_result:
                    st.write(f"**Direction:** {trade_plan_result.get('trade_direction', 'N/A')}")
                    st.write(f"**Confidence Score:** {trade_plan_result.get('overall_confidence', 'N/A'):.2f}%")
                    st.write(f"**Entry Zone:** ${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} - ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}")
                    st.write(f"**Target Price:** ${trade_plan_result.get('target_price', 'N/A'):.2f}")
                    st.write(f"**Stop Loss:** ${trade_plan_result.get('stop_loss', 'N/A'):.2f}")
                    st.write(f"**Reward/Risk Ratio:** {trade_plan_result.get('reward_risk_ratio', 'N/A'):.1f}:1")
                    st.markdown("---")
                    st.write("**Key Rationale:**")
                    st.write(trade_plan_result.get('key_rationale', 'No specific rationale available.'))
                    # Pivot points are now part of trade_plan_result
                    if trade_plan_result.get('pivot_points'):
                        st.write("**Pivot Points (Latest):**")
                        for p_key, p_val in trade_plan_result['pivot_points'].items():
                            st.write(f"- {p_key}: ${p_val:.2f}")
                    
                    st.write("**Technical Signals & Entry Criteria:**")
                    for criteria in trade_plan_result.get('technical_signals', []): # Using 'technical_signals' from trade_plan_result
                        st.markdown(f"- {criteria}")

                    # Exit criteria are part of the main rationale or target/stop
                    st.write("**Exit Criteria:**")
                    st.markdown(f"- Target Price: ${trade_plan_result.get('target_price', 'N/A'):.2f}")
                    st.markdown(f"- Stop Loss: ${trade_plan_result.get('stop_loss', 'N/A'):.2f}")

                else:
                    st.info("Could not generate a directional trade plan. Ensure sufficient data and valid indicator selections.")

            with tabs[3]: # 📜 Trade Log
                # Need prev_close for trade log, calculate it here if df_calculated is not empty
                prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) >= 2 else current_price

                display_trade_log_tab(
                    "trade_log.csv", # Placeholder, actual file name constructed inside function
                    ticker,
                    selected_timeframe,
                    overall_confidence,
                    current_price, # Pass the scalar current_price
                    prev_close,
                    trade_direction
                )

            with tabs[4]: # 🤖 Backtesting
                if not df_calculated.empty:
                    display_backtesting_tab(
                        df_calculated.copy(), # Pass a copy of df_calculated
                        st.session_state.indicator_selection, # Pass the full selection dict
                        normalized_weights # Pass normalized weights
                    )
                else:
                    st.info("No historical data available for backtesting.")

            with tabs[5]: # 🌍 Economic & Sentiment
                # Display Economic Data
                display_economic_data_tab(
                    ticker,
                    current_price, # Pass the scalar current_price
                    trade_plan_result.get('current_price'), # This is still a Series, but display_economic_data_tab uses it as prev_close
                                                            # Let's adjust this to pass a proper prev_close or ensure it's handled in display_economic_data_tab
                    overall_confidence,
                    trade_direction,
                    trade_plan_result['economic_context'].get('latest_gdp'),
                    trade_plan_result['economic_context'].get('latest_cpi'),
                    trade_plan_result['economic_context'].get('latest_unemployment_rate')
                )

                st.markdown("---")
                # Display Investor Sentiment Data
                display_investor_sentiment_tab(
                    ticker,
                    current_price, # Pass the scalar current_price
                    trade_plan_result.get('current_price'), # This is still a Series, same as above
                    overall_confidence,
                    trade_direction,
                    trade_plan_result['sentiment_analysis'].get('latest_vix'),
                    None # Removed historical_vix_avg as it's not in trade_plan_result['sentiment_analysis']
                )

            with tabs[6]: # 📚 Glossary
                st.markdown("### 📚 Glossary")
                st.info("The glossary content will be displayed here.") # Placeholder for actual glossary content

        else:
            st.warning("No data fetched or sufficient data for analysis for the given ticker and timeframe. Please check the ticker symbol and try again.")
            st.info("Ensure the ticker is valid and data is available for the selected period. Intraday data typically has a limited history (e.g., 7 days).")

    elif run_scanner_button: # This block is for the scanner
        st.header("⚡ Stock Scanner Results")
        if scanner_ticker_list:
            with st.spinner(f"Running scanner for {len(scanner_ticker_list)} tickers with '{selected_trading_style}' style..."):
                # Pass all necessary parameters to the scanner function
                scanner_results_df = scan_for_trades( # Changed from run_stock_scanner
                    scanner_ticker_list,
                    st.session_state.data_interval,
                    st.session_state.start_date,
                    st.session_state.end_date,
                    st.session_state.indicator_selection, # Pass the full selection dict
                    normalized_weights, # Pass the normalized weights
                    min_confidence=min_scanner_confidence
                )

                if not scanner_results_df.empty:
                    display_scanner_tab(scanner_results_df)
                else:
                    st.info("No qualifying stocks found based on your criteria.")
        else:
            st.info("Please enter tickers in the 'Tickers for Scanner' box in the sidebar to run the scanner.")
            
    else:
        st.info("Enter a stock ticker in the sidebar and click 'Analyze Ticker' to begin analysis, or configure and run the 'Stock Scanner'.")


if __name__ == "__main__":
    main() # Call the main function to run the app
