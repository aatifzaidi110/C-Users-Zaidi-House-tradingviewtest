# app.py - Final Version (v1.34)
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Needed for plt.close() in display_components
import yfinance as yf # Keep this import here for direct yf usage if any, though it's also in utils
from datetime import datetime, date, timedelta # ADDED THIS LINE: Import the datetime module

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
    run_stock_scanner
)

from display_components import (
    display_technical_analysis_tab, display_options_analysis_tab,
    display_backtesting_tab, display_trade_log_tab,
    display_scanner_tab,
    display_economic_data_tab, # <--- New function name
    display_investor_sentiment_tab, # <--- New function name
    # ... other imports
)

# === Page Setup ===
# Removed 'theme="dark"' as it's deprecated/removed in newer Streamlit versions
st.set_page_config(page_title="Aatif's AI Trading Hub", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Aatif's AI-Powered Trading Hub")

# --- Main Analysis Button ---
analyze_button = st.sidebar.button("Analyze Ticker")
clear_cache_button = st.sidebar.button("Clear Cache & Refresh Data")

if clear_cache_button:
    st.cache_data.clear()
    st.rerun()
# === Constants and Configuration ===
LOG_FILE = "trade_log.csv"

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL").upper()

# Timeframe selection
timeframe_options = {
    "1 Day": {"period": "1y", "interval": "1d"},
    "5 Day": {"period": "5d", "interval": "1m"}, # Intraday
    "1 Month": {"period": "1mo", "interval": "1m"}, # Intraday
    "3 Month": {"period": "3mo", "interval": "1h"}, # Intraday
    "6 Month": {"period": "6mo", "interval": "1d"},
    "1 Year": {"period": "1y", "interval": "1d"},
    "2 Year": {"period": "2y", "interval": "1d"},
    "5 Year": {"period": "5y", "interval": "1wk"},
    "10 Year": {"period": "10y", "interval": "1wk"},
    "YTD": {"period": "ytd", "interval": "1d"},
    "Max": {"period": "max", "interval": "1mo"}
}
selected_timeframe = st.sidebar.selectbox("Select Timeframe", list(timeframe_options.keys()))
period = timeframe_options[selected_timeframe]["period"]
interval = timeframe_options[selected_timeframe]["interval"]
is_intraday = interval in ["1m", "1h"]

# Confidence Weight Sliders
st.sidebar.subheader("Confidence Score Weights (%)")
# Initialize session state for weights if not already present
if 'weights' not in st.session_state:
    st.session_state.weights = {
        'technical': 40,
        'sentiment': 20,
        'expert': 20,
        'economic': 10,
        'investor_sentiment': 10
    }

# Sliders for weights
tech_weight = st.sidebar.slider("Technical Analysis", 0, 100, st.session_state.weights['technical'], key="tech_weight_slider")
sentiment_weight = st.sidebar.slider("News Sentiment", 0, 100, st.session_state.weights['sentiment'], key="sentiment_weight_slider")
expert_weight = st.sidebar.slider("Expert Ratings", 0, 100, st.session_state.weights['expert'], key="expert_weight_slider")
economic_weight = st.sidebar.slider("Economic Data", 0, 100, st.session_state.weights['economic'], key="economic_weight_slider")
investor_sentiment_weight = st.sidebar.slider("Investor Sentiment", 0, 100, st.session_state.weights['investor_sentiment'], key="investor_sentiment_weight_slider")

# Update session state weights
st.session_state.weights['technical'] = tech_weight
st.session_state.weights['sentiment'] = sentiment_weight
st.session_state.weights['expert'] = expert_weight
st.session_state.weights['economic'] = economic_weight
st.session_state.weights['investor_sentiment'] = investor_sentiment_weight

# Normalize weights to sum to 1.0 for the calculation function
total_weights = (tech_weight + sentiment_weight + expert_weight + economic_weight + investor_sentiment_weight)
if total_weights == 0: # Avoid division by zero
    st.sidebar.warning("Total weights sum to 0. Please adjust weights.")
    normalized_weights = {k: 0 for k in st.session_state.weights}
else:
    normalized_weights = {
        'technical': tech_weight / total_weights,
        'sentiment': sentiment_weight / total_weights,
        'expert': expert_weight / total_weights,
        'economic': economic_weight / total_weights,
        'investor_sentiment': investor_sentiment_weight / total_weights
    }

st.sidebar.markdown(f"**Total Weight: {total_weights}%**")
if total_weights != 100:
    st.sidebar.warning("Weights do not sum to 100%. Please adjust for accurate scoring.")


# === Technical Indicator Selection for Signal Generation and Scanner ===
st.sidebar.subheader("Technical Indicators for Signals")
# Initialize session state for indicator selection
if 'indicator_selection' not in st.session_state:
    st.session_state.indicator_selection = {
        "EMA Trend": True,
        "Ichimoku Cloud": True,
        "Parabolic SAR": True,
        "ADX": True,
        "RSI Momentum": True,
        "Stochastic": True,
        "MACD": True,
        "Bollinger Bands": False, # Typically for volatility, not direct signal
        "Volume Spike": True,
        "CCI": True,
        "ROC": True,
        "OBV": True,
        "VWAP": True, # Intraday specific
        "Pivot Points": False # For support/resistance, not direct signal
    }

# Checkboxes for indicator selection
st.sidebar.markdown("**Trend Indicators**")
st.session_state.indicator_selection["EMA Trend"] = st.sidebar.checkbox("EMA Trend (21, 50, 200)", value=st.session_state.indicator_selection["EMA Trend"])
st.session_state.indicator_selection["Ichimoku Cloud"] = st.sidebar.checkbox("Ichimoku Cloud", value=st.session_state.indicator_selection["Ichimoku Cloud"])
st.session_state.indicator_selection["Parabolic SAR"] = st.sidebar.checkbox("Parabolic SAR", value=st.session_state.indicator_selection["Parabolic SAR"])
st.session_state.indicator_selection["ADX"] = st.sidebar.checkbox("ADX (Average Directional Index)", value=st.session_state.indicator_selection["ADX"])

st.sidebar.markdown("**Momentum & Volume Indicators**")
st.session_state.indicator_selection["RSI Momentum"] = st.sidebar.checkbox("RSI (Relative Strength Index)", value=st.session_state.indicator_selection["RSI Momentum"])
st.session_state.indicator_selection["Stochastic"] = st.sidebar.checkbox("Stochastic Oscillator", value=st.session_state.indicator_selection["Stochastic"])
st.session_state.indicator_selection["MACD"] = st.sidebar.checkbox("MACD (Moving Average Convergence Divergence)", value=st.session_state.indicator_selection["MACD"])
st.session_state.indicator_selection["Volume Spike"] = st.sidebar.checkbox("Volume Spike", value=st.session_state.indicator_selection["Volume Spike"])
st.session_state.indicator_selection["CCI"] = st.sidebar.checkbox("CCI (Commodity Channel Index)", value=st.session_state.indicator_selection["CCI"])
st.session_state.indicator_selection["ROC"] = st.sidebar.checkbox("ROC (Rate of Change)", value=st.session_state.indicator_selection["ROC"])
st.session_state.indicator_selection["OBV"] = st.sidebar.checkbox("OBV (On Balance Volume)", value=st.session_state.indicator_selection["OBV"])

st.sidebar.markdown("**Volatility & Other Indicators**")
st.session_state.indicator_selection["Bollinger Bands"] = st.sidebar.checkbox("Bollinger Bands", value=st.session_state.indicator_selection["Bollinger Bands"])
st.session_state.indicator_selection["VWAP"] = st.sidebar.checkbox("VWAP (Volume Weighted Average Price) - Intraday Only", value=st.session_state.indicator_selection["VWAP"])
st.session_state.indicator_selection["Pivot Points"] = st.sidebar.checkbox("Pivot Points (Support/Resistance)", value=st.session_state.indicator_selection["Pivot Points"])

# --- Stock Scanner Section ---
st.sidebar.markdown("---")
st.sidebar.header("Stock Scanner")
scanner_ticker_list_str = st.sidebar.text_area("Tickers for Scanner (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
scanner_ticker_list = [t.strip().upper() for t in scanner_ticker_list_str.split(',') if t.strip()]

trading_style_options = [
    "Day Trading Long", "Day Trading Short",
    "Swing Trading Call", "Swing Trading Put",
    "Long-Term Investment" # Added for completeness, though not actively used for signals yet
]
selected_trading_style = st.sidebar.selectbox("Select Trading Style", trading_style_options)
min_scanner_confidence = st.sidebar.slider("Minimum Scanner Confidence (%)", 0, 100, 70)

run_scanner_button = st.sidebar.button("Run Scanner")


# --- Main Content Area ---
if analyze_button and ticker:
    st.header(f"Analyzing {ticker}")
    
    # Fetch data and calculate indicators
    with st.spinner(f"Fetching and calculating data for {ticker} ({selected_timeframe})..."):
        df_hist, stock_info = get_data(ticker, period, interval)
        
        if df_hist.empty:
            st.error(f"Could not retrieve data for {ticker}. Please check the ticker symbol and timeframe.", icon="🚫")
            st.stop()

        df_calculated = calculate_indicators(df_hist.copy(), is_intraday)
        
        if df_calculated.empty:
            st.warning(f"Not enough data to calculate indicators for {ticker} with the selected timeframe. Displaying raw data if available.", icon="⚠️")
            df_calculated = df_hist.copy() # Fallback to raw data if indicators can't be calculated

        # Get Finviz data for sentiment and expert ratings
        finviz_data = get_finviz_data(ticker)
        
        # Get economic and investor sentiment data
        today = date.today() # Using 'date' directly as it's imported
        one_year_ago = today - timedelta(days=365) # Using 'timedelta' directly as it's imported

        latest_gdp = get_economic_data_fred('GDP', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
        latest_cpi = get_economic_data_fred('CPIAUCSL', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
        latest_unemployment = get_economic_data_fred('UNRATE', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))

        economic_score_val = calculate_economic_score(
            latest_gdp.iloc[-1] if latest_gdp is not None and not latest_gdp.empty else None,
            latest_cpi.iloc[-1] if latest_cpi is not None and not latest_cpi.empty else None,
            latest_unemployment.iloc[-1] if latest_unemployment is not None and not latest_unemployment.empty else None
        )

        vix_data = get_vix_data(one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
        latest_vix = vix_data['Close'].iloc[-1] if vix_data is not None and not vix_data.empty else None
        historical_vix_avg = vix_data['Close'].mean() if vix_data is not None and not vix_data.empty else None
        investor_sentiment_score_val = calculate_sentiment_score(latest_vix, historical_vix_avg)

    # Calculate overall confidence score
    if not df_calculated.empty:
        last_row = df_calculated.iloc[-1]
        
        # Generate technical signals for the latest row
        bullish_signals, bearish_signals = generate_signals_for_row(last_row)

        tech_score_raw = 0
        total_possible_tech_points = 0
        for ind_name, is_selected in st.session_state.indicator_selection.items():
            if is_selected and ind_name not in ["Bollinger Bands", "Pivot Points"]: # Exclude non-directional for scoring
                total_possible_tech_points += 1
                if bullish_signals.get(ind_name, False):
                    tech_score_raw += 1
                elif bearish_signals.get(ind_name, False):
                    tech_score_raw -= 1
        
        if total_possible_tech_points > 0:
            technical_score = ((tech_score_raw + total_possible_tech_points) / (2 * total_possible_tech_points)) * 100
        else:
            technical_score = 50 # Neutral if no selected indicators were directional

        # News sentiment score (Finviz)
        news_sentiment_score = finviz_data.get("sentiment_compound", 0) * 100

        # Expert rating score (Finviz)
        expert_recom_str = stock_info.get('recommendationMean', None)
        expert_score = convert_finviz_recom_to_score(str(expert_recom_str))

        # Calculate overall confidence
        confidence_results = calculate_confidence_score(
            technical_score,
            news_sentiment_score,
            expert_score,
            economic_score_val, # Use the calculated economic score
            investor_sentiment_score_val, # Use the calculated investor sentiment score
            normalized_weights # Pass the normalized weights here
        )
        overall_confidence = confidence_results['score']
        trade_direction = confidence_results['direction']
    else:
        overall_confidence = 50
        trade_direction = "Neutral"
        technical_score = 50
        news_sentiment_score = 50
        expert_score = 50
        economic_score_val = 50
        investor_sentiment_score_val = 50

    # Create tabs for different analysis views
    tab_titles = [
        "📊 Technical Analysis", "📈 Options Analysis", "🤖 Backtesting",
        "📝 Trade Log", "💡 Trade Plan", "🌍 Economic & Sentiment", "📚 Glossary"
    ]
    
    # Check if df_calculated is empty and adjust tabs accordingly
    if df_calculated.empty:
        st.warning("No data available for technical analysis. Some tabs may be empty.")
        # Filter out tabs that heavily rely on df_calculated if it's empty
        tab_titles = ["📝 Trade Log", "💡 Trade Plan", "🌍 Economic & Sentiment", "📚 Glossary"]


    tabs = st.tabs(tab_titles)

    with tabs[0]: # 📊 Technical Analysis
        if not df_calculated.empty:
            display_technical_analysis_tab(ticker, df_calculated, is_intraday, st.session_state.indicator_selection)
        else:
            st.info("No data to display technical analysis.")

    with tabs[1]: # 📈 Options Analysis
        current_stock_price = df_calculated.iloc[-1]['Close'] if not df_calculated.empty else None
        if current_stock_price:
            stock_obj_for_options = yf.Ticker(ticker)
            expirations = stock_obj_for_options.options
            display_options_analysis_tab(ticker, current_stock_price, expirations, trade_direction, overall_confidence)
        else:
            st.info("No stock price data to perform options analysis.")

    with tabs[2]: # 🤖 Backtesting
        if not df_hist.empty:
            display_backtesting_tab(df_hist, st.session_state.indicator_selection)
        else:
            st.info("No historical data available for backtesting.")

    with tabs[3]: # 📝 Trade Log
        display_trade_log_tab(LOG_FILE, ticker, selected_timeframe, overall_confidence,
                              df_calculated.iloc[-1]['Close'] if not df_calculated.empty else None,
                              df_calculated.iloc[-2]['Close'] if not df_calculated.empty and len(df_calculated) > 1 else (df_calculated.iloc[-1]['Close'] if not df_calculated.empty else None),
                              trade_direction)

    with tabs[4]: # 💡 Trade Plan
        if not df_calculated.empty:
            # Pass the full latest_row to generate_directional_trade_plan
            trade_plan = generate_directional_trade_plan(
                {'score': overall_confidence, 'band': trade_direction},
                df_calculated.iloc[-1]['Close'],
                df_calculated.iloc[-1], # Pass the latest row of df_calculated
                interval # Pass the interval
            )
            
            st.markdown("### 💡 AI-Generated Trade Plan")
            if trade_plan['status'] == 'success':
                st.success(trade_plan['message'])
                st.write(f"**Direction:** {trade_plan['direction']}")
                st.write(f"**Current Price:** ${df_calculated.iloc[-1]['Close']:.2f}")
                st.write(f"**Entry Zone:** ${trade_plan['entry_zone_start']:.2f} - ${trade_plan['entry_zone_end']:.2f}")
                st.write(f"**Stop Loss:** ${trade_plan['stop_loss']:.2f}")
                st.write(f"**Profit Target:** ${trade_plan['profit_target']:.2f}")
                st.write(f"**Reward/Risk Ratio:** {trade_plan['reward_risk_ratio']:.2f}:1")
                st.markdown(f"**Rationale:** {trade_plan['key_rationale']}")
            else:
                st.info(trade_plan['message'])
        else:
            st.info("No data to generate a trade plan.")

  with tabs[5]: # 🌍 Economic & Sentiment
        # Display Economic Data
        display_economic_data_tab(
            ticker, # Assuming ticker is available
            current_price, # Assuming current_price is available
            prev_close, # Assuming prev_close is available
            overall_confidence, # Assuming overall_confidence is available
            trade_direction, # Assuming trade_direction is available
            latest_gdp,
            latest_cpi,
            latest_unemployment
        )
        
        st.markdown("---") # Add a separator if you like
        
        # Display Investor Sentiment Data
        display_investor_sentiment_tab(
            ticker, # Assuming ticker is available
            current_price, # Assuming current_price is available
            prev_close, # Assuming prev_close is available
            overall_confidence, # Assuming overall_confidence is available
            trade_direction, # Assuming trade_direction is available
            vix_data['Close'].iloc[-1] if not vix_data.empty else None, # latest_vix
            vix_data['Close'].mean() if not vix_data.empty else None # historical_vix_avg
        )
        
        # You might need to decide where to display news_sentiment_score and finviz_headlines
        # You could add them to display_news_info_tab or create a separate small section here.
        # For example, if you want news sentiment here:
        st.markdown("---")
        st.subheader("News Sentiment (from Finviz)")
        st.metric("News Sentiment Score (Finviz)", f"{news_sentiment_score:.0f}%")
        if finviz_data and finviz_data.get('headlines'):
            st.markdown("##### Latest Headlines:")
            for headline in finviz_data['headlines']:
                st.markdown(f"- [{headline['title']}]({headline['link']}) ({headline['date']})")
        else:
            st.info("No recent news headlines found.")

    with tabs[6]: # 📚 Glossary
        st.markdown("### 📚 Glossary")
        st.info("The glossary content will be displayed here.") # Placeholder for actual glossary content
elif run_scanner_button:
st.header("⚡ Stock Scanner Results")
with st.spinner(f"Running scanner for {len(scanner_ticker_list)} tickers with '{selected_trading_style}' style..."):
        # Pass all necessary parameters to the scanner function
        scanner_results_df = run_stock_scanner(
            scanner_ticker_list,
            selected_trading_style,
            min_scanner_confidence,
            st.session_state.indicator_selection, # Pass the full selection dict
            normalized_weights # Pass the normalized weights
        )

        if not scanner_results_df.empty:
            display_scanner_tab(scanner_results_df)
        else:
            st.info("No qualifying stocks found based on your criteria.")
            
else:
st.info("Enter a stock ticker in the sidebar and click 'Analyze Ticker' to begin analysis, or configure and run the 'Stock Scanner'.")
