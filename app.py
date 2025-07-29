import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import pytz # Import pytz for timezone handling

# Import functions from utils.py and display_components.py
from utils import (
    get_data, calculate_indicators, generate_directional_trade_plan,
    get_finviz_data, get_options_chain, get_economic_data_fred, get_vix_data,
    calculate_economic_score, calculate_sentiment_score, scan_for_trades
)
from display_components import (
    display_technical_analysis_tab, display_options_analysis_tab,
    display_backtesting_tab, display_trade_log_tab,
    display_economic_data_tab, display_investor_sentiment_tab,
    display_scanner_tab
)

# --- Configuration ---
# Define the path for the trade log file
LOG_FILE = "trade_log.csv"

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Advanced Trading Dashboard")

# --- Session State Initialization ---
if 'trade_plan_result' not in st.session_state:
    st.session_state.trade_plan_result = None
if 'df_calculated' not in st.session_state:
    st.session_state.df_calculated = pd.DataFrame()
if 'options_chain_dates' not in st.session_state:
    st.session_state.options_chain_dates = []
if 'finviz_data' not in st.session_state:
    st.session_state.finviz_data = {"recom_score": None, "news_sentiment_score": None}
if 'vix_data' not in st.session_state:
    st.session_state.vix_data = pd.DataFrame()
if 'economic_data' not in st.session_state:
    st.session_state.economic_data = {
        "gdp": pd.Series(dtype=float),
        "cpi": pd.Series(dtype=float),
        "unemployment": pd.Series(dtype=float)
    }
if 'scanner_results_df' not in st.session_state:
    st.session_state.scanner_results_df = pd.DataFrame()

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Settings & Filters")

# Ticker Input
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

# Timeframe Selection
timeframe_options = {
    "1 Minute": "1m", "2 Minutes": "2m", "5 Minutes": "5m", "15 Minutes": "15m",
    "30 Minutes": "30m", "60 Minutes": "60m", "90 Minutes": "90m",
    "1 Day": "1d", "5 Days": "5d", "1 Week": "1wk", "1 Month": "1mo", "3 Months": "3mo" # Added 3 Months
}
selected_timeframe_display = st.sidebar.selectbox("Select Timeframe", list(timeframe_options.keys()))
interval = timeframe_options[selected_timeframe_display]

# Date Range Selection
today = date.today()
default_start_date = today - timedelta(days=365) # Default to 1 year of data
default_end_date = today

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start_date, default_end_date),
    max_value=today
)

start_date = date_range[0]
end_date = date_range[1] if len(date_range) > 1 else today # Ensure end_date is set even if only start is picked

# Indicator Selection
st.sidebar.subheader("📊 Technical Indicators")
indicator_options = [
    "EMA Trend", "MACD", "RSI Momentum", "Bollinger Bands", "Stochastic",
    "Ichimoku Cloud", "Parabolic SAR", "ADX", "Volume Spike", "CCI", "ROC", "OBV", "VWAP", "Pivot Points"
]
# Initialize indicator_selection in session state if not present
if 'indicator_selection' not in st.session_state:
    st.session_state.indicator_selection = {ind: True for ind in indicator_options} # All selected by default

# Allow user to toggle indicators
for ind in indicator_options:
    st.session_state.indicator_selection[ind] = st.sidebar.checkbox(ind, value=st.session_state.indicator_selection[ind])

# Weights for Confidence Score
st.sidebar.subheader("⚖️ Confidence Weights (Total: 100%)")
# Initialize weights in session state if not present
if 'weights' not in st.session_state:
    st.session_state.weights = {
        "EMA Trend": 10, "MACD": 10, "RSI Momentum": 10,
        "Bollinger Bands": 10, "Stochastic": 10, "Ichimoku Cloud": 10,
        "Parabolic SAR": 10, "ADX": 10, "Volume Spike": 5,
        "CCI": 5, "ROC": 5, "OBV": 5, "VWAP": 5, "Pivot Points": 5,
        "Sentiment": 10, "Economic": 10
    }

# Sliders for weights
total_weights_sum = 0
for component, default_weight in st.session_state.weights.items():
    st.session_state.weights[component] = st.sidebar.slider(
        f"Weight for {component}", 0, 100, default_weight, key=f"weight_{component}"
    )
    total_weights_sum += st.session_state.weights[component]

# Normalize weights to sum to 100
if total_weights_sum > 0:
    normalized_weights = {k: v / total_weights_sum for k, v in st.session_state.weights.items()}
else:
    normalized_weights = {k: 0 for k in st.session_state.weights.keys()} # Avoid division by zero


# --- Main Application Logic ---
def main():
    st.title("📈 AI-Powered Trading Dashboard")

    # Fetch data and calculate indicators only once per ticker/timeframe/date change
    # Use st.session_state to store these results
    data_key = f"{ticker}_{interval}_{start_date}_{end_date}"
    if 'last_data_key' not in st.session_state or st.session_state.last_data_key != data_key:
        with st.spinner(f"Fetching data for {ticker} ({interval})..."):
            df_raw = get_data(ticker, interval, start_date, end_date)
            is_intraday_data = (interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m'])
            
            # Calculate indicators
            df_calculated = calculate_indicators(df_raw.copy(), st.session_state.indicator_selection, is_intraday_data)
            
            st.session_state.df_calculated = df_calculated
            st.session_state.last_data_key = data_key
            st.session_state.is_intraday_data = is_intraday_data

            # Fetch options chain dates
            try:
                tk = yf.Ticker(ticker)
                st.session_state.options_chain_dates = tk.options
            except Exception as e:
                st.session_state.options_chain_dates = []
                st.warning(f"Could not fetch options expiration dates for {ticker}: {e}")
            
            # Fetch Finviz data
            st.session_state.finviz_data = get_finviz_data(ticker)

            # Fetch VIX data
            vix_start_date = (date.today() - timedelta(days=365)) # Last year for VIX average
            vix_end_date = date.today()
            st.session_state.vix_data = get_vix_data(vix_start_date, vix_end_date)

            # Fetch Economic Data
            econ_start_date = (date.today() - timedelta(days=730)) # Last 2 years for economic data
            econ_end_date = date.today()
            st.session_state.economic_data["gdp"] = get_economic_data_fred("GDP", econ_start_date, econ_end_date)
            st.session_state.economic_data["cpi"] = get_economic_data_fred("CPI", econ_start_date, econ_end_date)
            st.session_state.economic_data["unemployment"] = get_economic_data_fred("UNRATE", econ_start_date, econ_end_date)


    df_calculated = st.session_state.df_calculated
    is_intraday_data = st.session_state.is_intraday_data

    current_price = df_calculated['Close'].iloc[-1] if not df_calculated.empty else None
    prev_close = df_calculated['Close'].iloc[-2] if len(df_calculated) > 1 else None

    # Generate the comprehensive trade plan
    # This needs to be re-run if indicator selections or weights change
    trade_plan_key = f"trade_plan_{ticker}_{interval}_{start_date}_{end_date}_{st.session_state.indicator_selection}_{normalized_weights}"
    if 'last_trade_plan_key' not in st.session_state or st.session_state.last_trade_plan_key != trade_plan_key:
        with st.spinner("Generating trade plan..."):
            st.session_state.trade_plan_result = generate_directional_trade_plan(
                ticker,
                interval,
                start_date,
                end_date,
                st.session_state.indicator_selection,
                normalized_weights,
                options_expiration_date=st.session_state.options_chain_dates[0] if st.session_state.options_chain_dates else None
            )
            st.session_state.last_trade_plan_key = trade_plan_key
    
    trade_plan_result = st.session_state.trade_plan_result

    # Determine overall confidence and direction for header
    overall_confidence = trade_plan_result.get('overall_confidence', 0)
    trade_direction = trade_plan_result.get('trade_direction', 'Neutral')

    # Display common header for all tabs
    display_components._display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction)


    # --- Tabbed Interface ---
    tab_titles = [
        "📈 Technical Analysis", "🔮 Options Analysis", "💡 Trade Plan",
        "📜 Trade Log", "🌍 Economic Data", "❤️ Investor Sentiment", "🔍 Scanner"
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]: # 📈 Technical Analysis
        display_technical_analysis_tab(
            ticker,
            df_calculated,
            is_intraday_data,
            st.session_state.indicator_selection,
            normalized_weights
        )

    with tabs[1]: # 🔮 Options Analysis
        display_options_analysis_tab(
            ticker,
            current_price,
            st.session_state.options_chain_dates,
            trade_direction,
            overall_confidence,
            trade_plan_result.get('target_price'), # Pass target from trade plan
            trade_plan_result.get('stop_loss')    # Pass stop loss from trade plan
        )

    with tabs[2]: # 💡 Trade Plan
        st.subheader("🗺️ Directional Trade Plan (Based on Current Data)")
        if trade_plan_result:
            st.write(f"**Direction:** {trade_plan_result.get('trade_direction', 'N/A')}")
            st.write(f"**Confidence Score:** {trade_plan_result.get('overall_confidence', 'N/A'):.2f}%")
            
            # Safely display Entry Zone
            entry_start = trade_plan_result.get('entry_zone_start')
            entry_end = trade_plan_result.get('entry_zone_end')
            entry_start_str = f"${entry_start:.2f}" if entry_start is not None and pd.notna(entry_start) else "N/A"
            entry_end_str = f"${entry_end:.2f}" if entry_end is not None and pd.notna(entry_end) else "N/A"
            st.write(f"**Entry Zone:** {entry_start_str} - {entry_end_str}")

            # Safely display Target Price
            target_price_val = trade_plan_result.get('target_price')
            target_price_str = f"${target_price_val:.2f}" if target_price_val is not None and pd.notna(target_price_val) else "N/A"
            st.write(f"**Target Price:** {target_price_str}")

            # Safely display Stop Loss
            stop_loss_val = trade_plan_result.get('stop_loss')
            stop_loss_str = f"${stop_loss_val:.2f}" if stop_loss_val is not None and pd.notna(stop_loss_val) else "N/A"
            st.write(f"**Stop Loss:** {stop_loss_str}")

            # Safely display Reward/Risk Ratio
            reward_risk_val = trade_plan_result.get('reward_risk_ratio')
            reward_risk_str = f"{reward_risk_val:.1f}:1" if reward_risk_val is not None and pd.notna(reward_risk_val) else "N/A"
            st.write(f"**Reward/Risk Ratio:** {reward_risk_str}")

            st.markdown("---")
            st.write("**Key Rationale:**")
            st.write(trade_plan_result.get('key_rationale', 'No specific rationale available.'))
            
            # Pivot points are now part of trade_plan_result
            if trade_plan_result.get('pivot_points'):
                st.write("**Pivot Points (Latest):**")
                for p_key, p_val in trade_plan_result['pivot_points'].items():
                    # Safely display pivot point values
                    p_val_str = f"${p_val:.2f}" if p_val is not None and pd.notna(p_val) else "N/A"
                    st.write(f"- {p_key}: {p_val_str}")
            
            st.write("**Technical Signals & Entry Criteria:**")
            for criteria in trade_plan_result.get('technical_signals', []): 
                st.markdown(f"- {criteria}")

            # Exit criteria are part of the main rationale or target/stop
            st.write("**Exit Criteria:**")
            # Safely display Target Price in Exit Criteria
            st.markdown(f"- Target Price: {target_price_str}")
            # Safely display Stop Loss in Exit Criteria
            st.markdown(f"- Stop Loss: {stop_loss_str}")

        else:
            st.info("Could not generate a directional trade plan. Ensure sufficient data and valid indicator selections.")

    with tabs[3]: # 📜 Trade Log
        display_trade_log_tab(LOG_FILE, ticker, selected_timeframe_display, overall_confidence, current_price, prev_close, trade_direction)

    with tabs[4]: # 🌍 Economic Data
        display_economic_data_tab(
            ticker,
            current_price,
            prev_close,
            overall_confidence,
            trade_direction,
            st.session_state.economic_data["gdp"],
            st.session_state.economic_data["cpi"],
            st.session_state.economic_data["unemployment"]
        )

    with tabs[5]: # ❤️ Investor Sentiment
        latest_vix = st.session_state.vix_data['Close'].iloc[-1] if not st.session_state.vix_data.empty else None
        historical_vix_avg = st.session_state.vix_data['Close'].mean() if not st.session_state.vix_data.empty else None
        display_investor_sentiment_tab(
            ticker,
            current_price,
            prev_close,
            overall_confidence,
            trade_direction,
            latest_vix,
            historical_vix_avg
        )

    with tabs[6]: # 🔍 Scanner
        st.subheader("🔍 Scan for Opportunities")
        scan_tickers_input = st.text_area("Enter Tickers to Scan (comma-separated)", "MSFT,GOOGL,AMZN,TSLA")
        min_confidence_scanner = st.slider("Minimum Confidence for Scanner (%)", 0, 100, 70)
        run_scanner_button = st.button("Run Scanner")

        if run_scanner_button:
            tickers_to_scan = [t.strip().upper() for t in scan_tickers_input.split(',') if t.strip()]
            if not tickers_to_scan:
                st.warning("Please enter at least one ticker to scan.")
            else:
                with st.spinner("Scanning for trade opportunities... This may take a while for many tickers."):
                    st.session_state.scanner_results_df = scan_for_trades(
                        tickers_to_scan,
                        interval,
                        start_date,
                        end_date,
                        st.session_state.indicator_selection,
                        normalized_weights,
                        min_confidence=min_confidence_scanner,
                        options_expiration_date=st.session_state.options_chain_dates[0] if st.session_state.options_chain_dates else None
                    )
                if st.session_state.scanner_results_df.empty:
                    st.info("No trade opportunities found matching your criteria.")
                else:
                    display_scanner_tab(st.session_state.scanner_results_df)
        elif not st.session_state.scanner_results_df.empty:
            display_scanner_tab(st.session_state.scanner_results_df)
        else:
            st.info("Click 'Run Scanner' to find trade opportunities.")


if __name__ == "__main__":
    main()
