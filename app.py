# app.py - Final Version (v1.34) 0730
import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # Needed for plt.close() in display_components
import yfinance as yf # Keep this import here for direct yf usage if any, though it's also in utils
from datetime import datetime, date, timedelta # ADDED THIS LINE: Import the datetime module

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧭 sys.path:", sys.path)
print("📁 Current directory contents:", os.listdir())
print("Current working directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("=== DEBUG INFO ===")
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("Python path:", sys.path)
print("=================")

# --- Import functions from modules ---
try:
    # Attempt to import calculate_confidence_score first, for early debug check
    from utils import calculate_confidence_score
    print("✅ Successfully imported calculate_confidence_score (early check)")

    # ✅ Debug print after successful import (MOVED HERE)
    import inspect
    print("✅ Signature:", inspect.signature(calculate_confidence_score))
    print("✅ Imported from:", calculate_confidence_score.__code__.co_filename)

except ImportError as e:print(f"❌ Initial import of calculate_confidence_score failed: {e}")
# Consider raising the error or stopping the app if this is critical
# st.error("Application startup failed due to missing utility functions.")
# st.stop()


# Import all other functions from modules (including calculate_confidence_score again, which is fine)
from utils import (
get_finviz_data, get_data, get_options_chain,
calculate_indicators, calculate_pivot_points,
generate_signals_for_row, backtest_strategy, analyze_options_chain,
suggest_options_strategy, generate_directional_trade_plan,
calculate_confidence_score, 
get_economic_data_fred, get_vix_data, calculate_economic_score, calculate_sentiment_score,
scan_for_trades # Changed from run_stock_scanner to scan_for_trades
)
# You can remove `import inspect` here if it's already at the top level
# or if you only need it for the debug prints. If it's used elsewhere, keep it.
# import inspect # This line is redundant if already imported above and not used again.

from display_components import (
_display_common_header, # Ensure this is imported directly
display_technical_analysis_tab, display_options_analysis_tab,
display_backtesting_tab, display_trade_log_tab,
display_economic_data_tab, display_investor_sentiment_tab,
display_scanner_tab # Ensure display_scanner_tab is imported
)

# ... rest of your app.py code ...
# --- Configuration ---
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Advanced Stock Analyzer")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", key="ticker_input")

# Store the ticker in session_state safely
if 'ticker' not in st.session_state:
    st.session_state.ticker = ticker
else:
    st.session_state.ticker = ticker
st.session_state.ticker = ticker
analyze = st.sidebar.button("Analyze Ticker")
clear = st.sidebar.button("Clear Cache")

if clear:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("🔁 Cache cleared.")
if analyze:
    if not ticker:
        st.warning("⚠️ Please enter a stock ticker symbol.")
    else:
        st.markdown(f"## Analyzing {ticker.upper()}")
        try:
            df = yf.download(ticker, period="3mo", interval="1h")

            # --- CORRECTED CODE FOR MULTIINDEX HANDLING ---
            # Ensure columns are single-level (not MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                # This drops the second level of the MultiIndex (e.g., 'AAPL')
                # and keeps the first level (e.g., 'Close') as the column name.
                df.columns = df.columns.droplevel(1)
            # --- END OF CORRECTED CODE ---

            if 'Adj Close' in df.columns and 'Close' not in df.columns:
                df['Close'] = df['Adj Close']
            
            # Drop rows with any NaN values in critical columns before indicator calculation
            # This is important if some OHLCV data points are missing
            df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

            if df.empty:
                st.warning(f"No data available for {ticker} with the selected period/interval. Please try a different ticker or timeframe.")
                return

            # ... (rest of your app.py code that calls calculate_indicators) ...
            # For example:
            # indicators_df = calculate_indicators(df, selected_indicators, is_intraday_data)

        except Exception as e:
            st.error(f"Error analyzing ticker: {e}")
            st.warning("Please ensure the ticker is valid and try again.")

            
            df.columns = df.columns.droplevel(1)
            df.dropna(inplace=True)
            indicator_selection = {
                "EMA Trend": True,
                "MACD": True,
                "RSI Momentum": True,
                "Bollinger Bands": True
            }
            df_indicators = calculate_indicators(df, indicator_selection, is_intraday=True)

            # Simulated signal strengths for demo purposes
            signal_strengths = {
                "EMA Trend": 0.7,
                "MACD": 0.8,
                "RSI Momentum": 0.6,
                "Bollinger Bands": 0.9
            }

            bullish_signals = 2  # placeholder
            bearish_signals = 1  # placeholder
            normalized_weights = {key: 1 for key in signal_strengths}

            confidence_result = calculate_confidence_score(
                bullish_signals=bullish_signals,
                bearish_signals=bearish_signals,
                signal_strength=signal_strengths,
                normalized_weights=normalized_weights
            )

            st.markdown("### ✅ Confidence Score Result")
            st.write(confidence_result)

        except Exception as e:
            st.error(f"❌ Error analyzing ticker: {e}")

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
                        "expert": 0.2,
                        "economic": 0.1,
                        "investor_sentiment": 0.1
                        }
                        # Initialize trade_plan_result in session state for sentiment tab access
                        if 'trade_plan_result' not in st.session_state:
                            st.session_state.trade_plan_result = {}


                            # --- Sidebar for User Inputs ---
                            st.sidebar.header("Configuration")

                            # Ticker Input
                            ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", st.session_state.ticker).upper()
                            st.session_state.ticker = ticker

                            # Analyze Ticker Button
                            st.sidebar.markdown("---")
                            analyze_button = st.sidebar.button("Analyze Ticker")

                            # Clear Cache Button
                            if st.sidebar.button("Clear Cache"):
                                st.cache_data.clear()
                                st.rerun()
                                # --- Trading Style Selection ---
                                trading_style = st.selectbox("Trading Style", [
                                "Scalp Trading (1m)",
                                "Day Trading (5m)",
                                "Swing Trading (1h)",
                                "Position Trading (1d)"
                                ])

                                # Map trading style to interval and duration (days)
                                interval_map = {
                                "Scalp Trading (1m)": ("1m", 5),      # 5 days max for 1m data
                                "Day Trading (5m)": ("5m", 15),       # 15 days
                                "Swing Trading (1h)": ("1h", 90),     # 3 months
                                "Position Trading (1d)": ("1d", 365)  # 1 year
                                }

                                # Apply selected mapping
                                interval, days = interval_map[trading_style]
                                st.session_state.data_interval = interval  # Set interval to session state
                                start_date = datetime.today() - timedelta(days=days)
                                end_date = datetime.today()
                                st.session_state.start_date = start_date
                                st.session_state.end_date = end_date

                                # Infer if intraday
                                is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]

                                # Display timeframe summary for debug/info (optional)
                                st.sidebar.markdown(f"**Interval:** `{interval}` — **Days:** `{days}`")

                                # ✅ Set selected_timeframe safely based on interval
                                selected_timeframe = interval  # This replaces the incorrect `timeframe` reference

                                # ✅ Rest of your application logic can now use `selected_timeframe` safely
                                # Note: Already used in analyze_button condition block


                                # Timeframe Selection
                                #timeframe_options = {
                                #    "1d": "1 Day (Daily)", "1wk": "1 Week (Weekly)", "1mo": "1 Month (Monthly)",
                                #    "1m": "1 Minute (Intraday)", "2m": "2 Minutes (Intraday)", "5m": "5 Minutes (Intraday)",
                                #    "15m": "15 Minutes (Intraday)", "30m": "30 Minutes (Intraday)", "60m": "60 Minutes (Intraday)",
                                #    "90m": "90 Minutes (Intraday)", "1h": "1 Hour (Intraday)"
                                #}
                                #selected_timeframe_key = st.sidebar.selectbox(
                                #   "Select Timeframe",
                                #   list(timeframe_options.keys()),
                                #  format_func=lambda x: timeframe_options[x],
                                #  index=list(timeframe_options.keys()).index(st.session_state.data_interval)
                                #)
                                #st.session_state.data_interval = selected_timeframe_key
                                #selected_timeframe = timeframe_options[selected_timeframe_key]

                                # Date Range for Historical Data
                                #is_intraday = "Intraday" in selected_timeframe

                                #if not is_intraday:
                                    #   st.sidebar.subheader("Historical Data Range")
                                    #    start_date = st.sidebar.date_input("Start Date", st.session_state.start_date)
                                    #   end_date = st.sidebar.date_input("End Date", st.session_state.end_date)
                                    #   st.session_state.start_date = start_date
                                    #   st.session_state.end_date = end_date
                                    #else:
                                        #   st.sidebar.info("For intraday data, the data range is typically limited by the provider (e.g., 7 days for Yahoo Finance).")
                                        #   start_date = date.today() - timedelta(days=7) # Yahoo Finance limits intraday to 7 days
                                        #  end_date = date.today()
                                        #   st.session_state.start_date = start_date
                                        #   st.session_state.end_date = end_date


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


                                                        # --- Main Application Logic ---
                                                        st.title("Advanced Stock Analysis Dashboard")

                                                        if analyze_button and ticker:
                                                            # Fetch data
                                                            with st.spinner(f"Fetching data for {ticker} ({selected_timeframe})..."):
                                                                # Get data might return a tuple or None, so handle it to ensure `df` is a DataFrame
                                                                df_result = get_data(ticker, st.session_state.data_interval, st.session_state.start_date, st.session_state.end_date)

                                                                # Initialize df to an empty DataFrame
                                                                df = pd.DataFrame()
                                                                # Check if df_result is a DataFrame or a tuple containing a DataFrame
                                                                if isinstance(df_result, pd.DataFrame):
                                                                    df = df_result
                                                                elif isinstance(df_result, tuple) and len(df_result) > 0 and isinstance(df_result[0], pd.DataFrame):
                                                                 df = df_result[0]
                                                                # If df_result is None or an unexpected type, df remains an empty DataFrame


                                                                info = None
                                                                try:
                                                                    ticker_obj = yf.Ticker(ticker)
                                                                    info = ticker_obj.info # Fetch ticker info for company profile
                                                                except Exception as e:
                                                                 st.warning(f"Could not fetch ticker info for {ticker}: {e}")
                                                                print(f"Error fetching ticker info for {ticker}: {e}")


                                                                finviz_data = get_finviz_data(ticker) # Fetch Finviz data

                                                                # Get economic data, passing start_date and end_date
                                                                latest_gdp = get_economic_data_fred("GDP", st.session_state.start_date, st.session_state.end_date)
                                                                latest_cpi = get_economic_data_fred("CPI", st.session_state.start_date, st.session_state.end_date)
                                                                latest_unemployment = get_economic_data_fred("UNRATE", st.session_state.start_date, st.session_state.end_date)

                                                                # Get VIX data, passing start_date and end_date
                                                                vix_data_raw = get_vix_data(st.session_state.start_date, st.session_state.end_date)

                                                                # Initialize vix_data to None
                                                                vix_data = pd.DataFrame() # Initialize as empty DataFrame
                                                                # Check if vix_data_raw is a tuple and extract the DataFrame if it is
                                                                if isinstance(vix_data_raw, tuple) and len(vix_data_raw) > 0:
                                                                    # Assume the first element is the intended DataFrame
                                                                    if isinstance(vix_data_raw[0], pd.DataFrame):
                                                                        vix_data = vix_data_raw[0]
                                                                    elif isinstance(vix_data_raw, pd.DataFrame):
                                                                     vix_data = vix_data_raw # If it's already a DataFrame, use it directly

                                                                    latest_vix = None
                                                                    historical_vix_avg = None

                                                                    if not vix_data.empty and 'Close' in vix_data.columns:
                                                                        # Explicitly convert to scalar float values
                                                                        latest_vix_val_candidate = vix_data['Close'].iloc[-1]
                                                                        if isinstance(latest_vix_val_candidate, pd.Series):
                                                                            if not latest_vix_val_candidate.empty:
                                                                                latest_vix_val = latest_vix_val_candidate.item() # Get the scalar value
                                                                            else:
                                                                             latest_vix_val = None
                                                                        else:
                                                                            latest_vix_val = latest_vix_val_candidate

                                                                            if pd.notna(latest_vix_val):
                                                                                latest_vix = float(latest_vix_val)

                                                                                historical_vix_avg_val_candidate = vix_data['Close'].mean()
                                                                                if isinstance(historical_vix_avg_val_candidate, pd.Series):
                                                                                    if not historical_vix_avg_val_candidate.empty:
                                                                                        historical_vix_avg_val = historical_vix_avg_val_candidate.item() # Get the scalar value
                                                                                    else:
                                                                                     historical_vix_avg_val = None
                                                                                else:
                                                                                    historical_vix_avg_val = historical_vix_avg_val_candidate

                                                                                    if pd.notna(historical_vix_avg_val):
                                                                                        historical_vix_avg = float(historical_vix_avg_val)


                                                                                        if not df.empty:
                                                                                            st.success(f"Successfully fetched {len(df)} data points for {ticker}.")

                                                                                            # Calculate indicators
                                                                                            df_calculated = calculate_indicators(df.copy(), st.session_state.indicator_selection, is_intraday)

                                                                                            # ⛔ Prevent IndexError if df_calculated is empty
                                                                                            if df_calculated.empty:
                                                                                                st.warning("⚠️ No valid data available after indicator calculations.")
                                                                                                st.stop()

                                                                                                # Calculate Pivot Points separately as they need to be passed to display_technical_analysis_tab
                                                                                                # and are only calculated for non-intraday data.
                                                                                                df_pivots = pd.DataFrame()
                                                                                                if st.session_state.indicator_selection.get("Pivot Points") and not is_intraday:
                                                                                                    df_pivots = calculate_pivot_points(df.copy()) # Pass a copy

                                                                                                    # Get latest row for confidence score and trade plan
                                                                                                    last_row = df_calculated.iloc[-1]
                                                                                                    current_price = last_row['Close']
                                                                                                    prev_close = df_calculated.iloc[-2]['Close'] if len(df_calculated) >= 2 else current_price

                                                                                                    # === Generate Signal Strengths ===
                                                                                                    signal_strengths = {}

                                                                                                    if 'RSI' in df_calculated.columns:
                                                                                                        rsi_value = last_row['RSI']
                                                                                                        signal_strengths['RSI Momentum'] = 1 - abs(rsi_value - 50) / 50

                                                                                                        if 'MACD' in df_calculated.columns and 'MACD_Signal' in df_calculated.columns:
                                                                                                            macd_diff = last_row['MACD'] - last_row['MACD_Signal']
                                                                                                            if isinstance(macd_diff, pd.Series):
                                                                                                                macd_diff = macd_diff.item()
                                                                                                                signal_strengths['MACD'] = 1 if macd_diff > 0 else 0

                                                                                                                if 'ADX' in df_calculated.columns:
                                                                                                                    adx_value = last_row['ADX']
                                                                                                                    signal_strengths['ADX'] = min(adx_value / 40, 1)

                                                                                                                    # === Initialize Weights and Flags ===
                                                                                                                    user_sentiment_weights = {"sentiment": 1.0}
                                                                                                                    expert_sentiment_weights = {"expert": 1.0}
                                                                                                                    use_sentiment = True
                                                                                                                    use_expert = True


                                                                                                                    # Calculate Confidence Scores
                                                                                                                    # Pass the full indicator_selection and normalized_weights to calculate_confidence_score
                                                                                                                    def calculate_confidence_score(
                                                                                                                    last_row=last_row,
                                                                                                                    news_sentiment=finviz_data.get('news_sentiment_score'),
                                                                                                                    recom_score=finviz_data.get('recom_score'),
                                                                                                                    latest_gdp=latest_gdp.iloc[-1] if latest_gdp is not None and not latest_gdp.empty and len(latest_gdp) > 0 else None,
                                                                                                                    latest_cpi=latest_cpi.iloc[-1] if latest_cpi is not None and not latest_cpi.empty and len(latest_cpi) > 0 else None,
                                                                                                                    latest_unemployment=latest_unemployment.iloc[-1] if latest_unemployment is not None and not latest_unemployment.empty and len(latest_unemployment) > 0 else None,
                                                                                                                    latest_vix=latest_vix,
                                                                                                                    historical_vix_avg=historical_vix_avg,
                                                                                                                    normalized_weights=normalized_weights,
                                                                                                                    indicator_selection=st.session_state.indicator_selection,
                                                                                                                    signal_strengths=signal_strengths,
                                                                                                                    user_sentiment_weights=user_sentiment_weights,
                                                                                                                    expert_sentiment_weights=expert_sentiment_weights,
                                                                                                                    use_sentiment=use_sentiment,
                                                                                                                    use_expert=use_expert
                                                                                                                    ):
                                                                                                                        scores = confidence_result["confidence_score"]
                                                                                                                        overall_confidence = confidence_result["confidence_level"]
                                                                                                                        trade_direction = confidence_result["direction"]

                                                                                                                        # Debug Prints
                                                                                                                        print("✅ Confidence Score:", scores)
                                                                                                                        print("📊 Confidence Level:", overall_confidence)
                                                                                                                        print("📈 Direction:", trade_direction)


                                                                                                                        # Get options chain expiration dates
                                                                                                                        options_chain = yf.Ticker(ticker).options

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
                                                                                                                        current_price,
                                                                                                                        options_chain, # Pass expirations
                                                                                                                        trade_direction,
                                                                                                                        overall_confidence
                                                                                                                        )

                                                                                                                        with tabs[2]: # 💡 Trade Plan
                                                                                                                         st.subheader("🗺️ Directional Trade Plan (Based on Current Data)")
                                                                                                                        if not df_calculated.empty:
                                                                                                                            try:
                                                                                                                                # Ensure generate_directional_trade_plan receives the correct arguments
                                                                                                                                trade_plan_result = generate_directional_trade_plan(
                                                                                                                                ticker, # Pass ticker
                                                                                                                                st.session_state.data_interval, # Pass interval
                                                                                                                                st.session_state.start_date, # Pass start_date
                                                                                                                                st.session_state.end_date, # Pass end_date
                                                                                                                                st.session_state.indicator_selection, # Pass the full selection dict
                                                                                                                                normalized_weights, # Pass normalized weights
                                                                                                                                options_expiration_date=options_chain[0] if options_chain else None # Pass first options expiration date
                                                                                                                                )
                                                                                                                                st.session_state.trade_plan_result = trade_plan_result # Store trade plan in session state

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
                                                                                                                            except Exception as e:
                                                                                                                                        st.error(f"An error occurred while generating the trade plan: {e}")
                                                                                                                                        st.exception(e) # For debugging
                                                                                                                            else:
                                                                                                                                        st.info("No calculated data available to generate a trade plan.")


                                                                                                                                        with tabs[3]: # 📜 Trade Log
                                                                                                                                         display_trade_log_tab(
                                                                                                                                        "trade_log.csv", # Placeholder, actual file name constructed inside function
                                                                                                                                        ticker,
                                                                                                                                        selected_timeframe,
                                                                                                                                        overall_confidence,
                                                                                                                                        current_price,
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
                                                                                                                                        ticker, # Assuming ticker is available
                                                                                                                                        current_price, # Assuming current_price is available
                                                                                                                                        prev_close, # Assuming prev_close is available
                                                                                                                                        overall_confidence,
                                                                                                                                        trade_direction,
                                                                                                                                        latest_gdp,
                                                                                                                                        latest_cpi,
                                                                                                                                        latest_unemployment
                                                                                                                                        )

                                                                                                                                        st.markdown("---")
                                                                                                                                        # Display Investor Sentiment Data
                                                                                                                                        display_investor_sentiment_tab(
                                                                                                                                        ticker, # Assuming ticker is available
                                                                                                                                        current_price, # Assuming current_price is available
                                                                                                                                        # Assuming  is available
                                                                                                                                        overall_confidence,
                                                                                                                                        trade_direction,
                                                                                                                                        latest_vix, # Pass scalar VIX
                                                                                                                                        historical_vix_avg # Pass scalar VIX average
                                                                                                                                        )

                                                                                                                                        with tabs[6]: # 📚 Glossary
                                                                                                                                         st.markdown("### 📚 Glossary")
                                                                                                                                        st.info("The glossary content will be displayed here.") # Placeholder for actual glossary content

                                                                                                                        else:
                                                                                                                                        st.warning("No data fetched for the given ticker and timeframe. Please check the ticker symbol and try again.")
                                                                                                                                        st.info("Ensure the ticker is valid and data is available for the selected period. Intraday data typically has a limited history (e.g., 7 days).")

                                                                                                                elif run_scanner_button: # This block is for the scanner
                                                                                                                                        st.header("⚡ Stock Scanner Results")
                                                                                                                                        if scanner_ticker_list:
                                                                                                                                            with st.spinner(f"Running scanner for {len(scanner_ticker_list)} tickers with '{selected_trading_style}' style..."):
                                                                                                                                                # Pass all necessary parameters to the scanner function
                                                                                                                                                scanner_results_df = scan_for_trades( # Changed from run_stock_scanner to scan_for_trades
                                                                                                                                                scanner_ticker_list,
                                                                                                                                                st.session_state.data_interval, # Pass interval
                                                                                                                                                st.session_state.start_date, # Pass start_date
                                                                                                                                                st.session_state.end_date, # Pass end_date
                                                                                                                                                st.session_state.indicator_selection, # Pass the full selection dict
                                                                                                                                                normalized_weights, # Pass the normalized weights
                                                                                                                                                min_confidence=min_scanner_confidence,
                                                                                                                                                options_expiration_date=None # Scanner doesn't use options expiration for now
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
                                                                                                                                                    # Main function is implicitly run by Streamlit
                                                                                                                                                    pass
