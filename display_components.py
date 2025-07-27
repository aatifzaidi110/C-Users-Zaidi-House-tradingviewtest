# display_components.py - Final Version (v4.2)

import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf
import os
import numpy as np
from datetime import datetime

# Import functions from utils.py
from utils import (
    backtest_strategy, calculate_indicators, generate_signals_for_row,
    suggest_options_strategy, get_options_chain, get_data, get_finviz_data,
    calculate_pivot_points, get_moneyness, analyze_options_chain,
    generate_directional_trade_plan, get_indicator_summary_text
)

# Mapping for Finviz recommendation numbers to qualitative descriptions
FINVIZ_RECOM_QUALITATIVE_MAP = {
    "1.00": "Strong Buy",
    "1.50": "Strong Buy / Buy",
    "2.00": "Buy",
    "2.50": "Buy / Hold",
    "3.00": "Hold",
    "3.50": "Hold / Sell",
    "4.00": "Sell",
    "4.50": "Sell / Strong Sell",
    "5.00": "Strong Sell"
}

# Define EXPERT_RATING_MAP locally within display_components.py
EXPERT_RATING_MAP = {
    "Strong Buy": 100,
    "Buy": 80,
    "Hold": 50,
    "Sell": 20,
    "Strong Sell": 0,
    "N/A": 50 # Neutral default for missing recommendations
}

# --- Display Functions for Tabs ---

def display_technical_analysis_tab(ticker, df_calculated, is_intraday, indicator_selection):
    """
    Displays the technical analysis tab with a candlestick chart and indicator plots.
    """
    st.markdown(f"### 📊 Technical Analysis for {ticker}")

    if df_calculated.empty:
        st.info("No data available to display technical analysis.")
        return

    # Ensure the DataFrame index is a DatetimeIndex, coercing errors
    # This handles cases where the index might be 'object' dtype but contains strings
    # that look like dates, or where the conversion might fail for some entries.
    df_calculated.index = pd.to_datetime(df_calculated.index, errors='coerce')
    # Remove any rows where the date conversion failed (index became NaT)
    df_calculated = df_calculated[df_calculated.index.notna()]
    
    if df_calculated.empty: # Check if it became empty after cleaning
        st.info("No valid date data in the DataFrame index after cleaning. Cannot display chart.")
        return

    # Filter out future dates if any (e.g., from some data sources)
    df_calculated = df_calculated[df_calculated.index <= pd.Timestamp.now()]

    # Create subplots for the chart and indicators
    add_plots = []
    if indicator_selection.get("EMA Trend"):
        add_plots.append(mpf.make_addplot(df_calculated['EMA21'], color='blue', panel=0, width=0.7, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_calculated['EMA50'], color='orange', panel=0, width=0.7, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_calculated['EMA200'], color='purple', panel=0, width=0.7, secondary_y=False))
    
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku cloud fills
        add_plots.append(mpf.make_addplot(df_calculated['ichimoku_a'], color='green', panel=0, width=0.7, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_calculated['ichimoku_b'], color='red', panel=0, width=0.7, secondary_y=False))
        # Fill between ichimoku_a and ichimoku_b for the cloud
        # mplfinance does not directly support fill_between, so we might need a workaround or just plot lines.
        # For simplicity, we'll just plot the lines.
        add_plots.append(mpf.make_addplot(df_calculated['ichimoku_conversion_line'], color='cyan', panel=0, width=0.7, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_calculated['ichimoku_base_line'], color='magenta', panel=0, width=0.7, secondary_y=False))

    if indicator_selection.get("Parabolic SAR"):
        add_plots.append(mpf.make_addplot(df_calculated['psar'], type='scatter', marker='.', markersize=50, color='lime', panel=0, secondary_y=False))

    # Add Bollinger Bands to the main panel
    if indicator_selection.get("Bollinger Bands"):
        add_plots.append(mpf.make_addplot(df_calculated['BB_high'], color='gray', panel=0, width=0.7, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_calculated['BB_low'], color='gray', panel=0, width=0.7, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_calculated['BB_mid'], color='blue', panel=0, width=0.7, secondary_y=False))

    # Add VWAP to the main panel (if intraday)
    if is_intraday and indicator_selection.get("VWAP"):
        add_plots.append(mpf.make_addplot(df_calculated['VWAP'], color='darkred', panel=0, width=1.0, secondary_y=False))


    # Setup subplots for other indicators
    fig_panels = []
    if indicator_selection.get("RSI Momentum"):
        fig_panels.append(1) # RSI
        add_plots.append(mpf.make_addplot(df_calculated['RSI'], panel=1, color='purple', ylabel='RSI'))
        add_plots.append(mpf.make_addplot(pd.Series(70, index=df_calculated.index), panel=1, color='red', width=0.5, linestyle='--', secondary_y=False))
        add_plots.append(mpf.make_addplot(pd.Series(30, index=df_calculated.index), panel=1, color='green', width=0.5, linestyle='--', secondary_y=False))
    
    if indicator_selection.get("Stochastic"):
        fig_panels.append(len(fig_panels)) # Stochastic
        add_plots.append(mpf.make_addplot(df_calculated['stoch_k'], panel=len(fig_panels)-1, color='blue', ylabel='Stoch %K'))
        add_plots.append(mpf.make_addplot(df_calculated['stoch_d'], panel=len(fig_panels)-1, color='orange', ylabel='Stoch %D'))
        add_plots.append(mpf.make_addplot(pd.Series(80, index=df_calculated.index), panel=len(fig_panels)-1, color='red', width=0.5, linestyle='--', secondary_y=False))
        add_plots.append(mpf.make_addplot(pd.Series(20, index=df_calculated.index), panel=len(fig_panels)-1, color='green', width=0.5, linestyle='--', secondary_y=False))

    if indicator_selection.get("MACD"):
        fig_panels.append(len(fig_panels)) # MACD
        add_plots.append(mpf.make_addplot(df_calculated['macd'], panel=len(fig_panels)-1, color='blue', ylabel='MACD'))
        add_plots.append(mpf.make_addplot(df_calculated['macd_signal'], panel=len(fig_panels)-1, color='orange'))
        # MACD Histogram
        colors = ['red' if val < 0 else 'green' for val in df_calculated['macd_diff']]
        add_plots.append(mpf.make_addplot(df_calculated['macd_diff'], type='bar', panel=len(fig_panels)-1, color=colors, width=0.7, alpha=0.7, secondary_y=False))

    if indicator_selection.get("ADX"):
        fig_panels.append(len(fig_panels)) # ADX
        add_plots.append(mpf.make_addplot(df_calculated['adx'], panel=len(fig_panels)-1, color='blue', ylabel='ADX'))
        add_plots.append(mpf.make_addplot(df_calculated['plus_di'], panel=len(fig_panels)-1, color='green'))
        add_plots.append(mpf.make_addplot(df_calculated['minus_di'], panel=len(fig_panels)-1, color='red'))
        add_plots.append(mpf.make_addplot(pd.Series(25, index=df_calculated.index), panel=len(fig_panels)-1, color='gray', width=0.5, linestyle='--', secondary_y=False))

    if indicator_selection.get("CCI"):
        fig_panels.append(len(fig_panels)) # CCI
        add_plots.append(mpf.make_addplot(df_calculated['CCI'], panel=len(fig_panels)-1, color='teal', ylabel='CCI'))
        add_plots.append(mpf.make_addplot(pd.Series(100, index=df_calculated.index), panel=len(fig_panels)-1, color='red', width=0.5, linestyle='--', secondary_y=False))
        add_plots.append(mpf.make_addplot(pd.Series(-100, index=df_calculated.index), panel=len(fig_panels)-1, color='green', width=0.5, linestyle='--', secondary_y=False))

    if indicator_selection.get("ROC"):
        fig_panels.append(len(fig_panels)) # ROC
        add_plots.append(mpf.make_addplot(df_calculated['ROC'], panel=len(fig_panels)-1, color='brown', ylabel='ROC'))
        add_plots.append(mpf.make_addplot(pd.Series(0, index=df_calculated.index), panel=len(fig_panels)-1, color='gray', width=0.5, linestyle='--', secondary_y=False))

    if indicator_selection.get("OBV"):
        fig_panels.append(len(fig_panels)) # OBV
        add_plots.append(mpf.make_addplot(df_calculated['obv'], panel=len(fig_panels)-1, color='darkgreen', ylabel='OBV'))
        add_plots.append(mpf.make_addplot(df_calculated['obv_ema'], panel=len(fig_panels)-1, color='orange', width=0.7, secondary_y=False))


    # Plotting
    try:
        fig, axlist = mpf.plot(
            df_calculated,
            type='candle',
            style='yahoo', # You can choose other styles like 'binance', 'charles', 'yahoo'
            title=f"{ticker} Candlestick Chart",
            ylabel='Price',
            ylabel_lower='Volume',
            volume=True,
            figscale=1.5, # Adjust figure size
            addplot=add_plots,
            panel_ratios=[6] + [2] * len(fig_panels), # Main chart taller, other panels shorter
            returnfig=True
        )
        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent memory issues
    except Exception as e:
        st.warning(f"Could not render chart. Please check data and indicator selections. Error: {e}", icon="⚠️")
        st.exception(e)

    st.markdown("---")
    st.markdown("### Latest Indicator Values")
    if not df_calculated.empty:
        latest_row = df_calculated.iloc[-1]
        st.write(latest_row.tail(20)) # Display last 20 columns of the latest row

def display_options_analysis_tab(ticker, current_stock_price, expirations, trade_direction, overall_confidence):
    """
    Displays options analysis, including chain data and suggested strategies.
    """
    st.markdown(f"### 📈 Options Analysis for {ticker}")

    if not expirations:
        st.info("No options data available for this ticker.")
        return

    selected_expiry = st.selectbox("Select Expiration Date", expirations)

    if selected_expiry:
        calls_df, puts_df = get_options_chain(ticker, selected_expiry)

        if calls_df.empty and puts_df.empty:
            st.warning(f"No options chain data found for {ticker} on {selected_expiry}.", icon="⚠️")
            return

        st.markdown("#### Call Options")
        st.dataframe(calls_df)

        st.markdown("#### Put Options")
        st.dataframe(puts_df)

        analysis_results = analyze_options_chain(calls_df, puts_df, current_stock_price)

        st.markdown("#### Options Chain Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Calls Summary")
            st.write(f"Total Volume: {analysis_results['calls'].get('total_volume', 'N/A')}")
            st.write(f"Total Open Interest: {analysis_results['calls'].get('total_open_interest', 'N/A')}")
            st.write(f"Avg IV (ITM): {analysis_results['calls'].get('avg_iv_itm', 0):.2f}")
            st.write(f"Avg IV (OTM): {analysis_results['calls'].get('avg_iv_otm', 0):.2f}")
            st.write(f"Avg IV (ATM): {analysis_results['calls'].get('avg_iv_atm', 0):.2f}")
        with col2:
            st.subheader("Puts Summary")
            st.write(f"Total Volume: {analysis_results['puts'].get('total_volume', 'N/A')}")
            st.write(f"Total Open Interest: {analysis_results['puts'].get('total_open_interest', 'N/A')}")
            st.write(f"Avg IV (ITM): {analysis_results['puts'].get('avg_iv_itm', 0):.2f}")
            st.write(f"Avg IV (OTM): {analysis_results['puts'].get('avg_iv_otm', 0):.2f}")
            st.write(f"Avg IV (ATM): {analysis_results['puts'].get('avg_iv_atm', 0):.2f}")

        st.markdown("---")
        st.markdown("#### Suggested Options Strategy")
        
        # Pass the correct parameters to the suggestion function
        suggested_strategy = suggest_options_strategy(
            ticker,
            overall_confidence,
            current_stock_price,
            expirations, # Pass all expirations, function will pick one
            trade_direction
        )

        if suggested_strategy['status'] == 'success':
            st.success(suggested_strategy['message'])
            st.write(f"**Strategy:** {suggested_strategy['Strategy']}")
            st.write(f"**Direction:** {suggested_strategy['Direction']}")
            st.write(f"**Expiration:** {suggested_strategy['Expiration']}")
            
            if suggested_strategy['Strategy'] == "Bull Call Spread":
                st.write(f"**Buy Call:** Strike ${suggested_strategy['Buy Strike']:.2f}, Premium ${suggested_strategy['Contracts']['Buy']['lastPrice']:.2f}")
                st.write(f"**Sell Call:** Strike ${suggested_strategy['Sell Strike']:.2f}, Premium ${suggested_strategy['Contracts']['Sell']['lastPrice']:.2f}")
            elif suggested_strategy['Strategy'] == "Bear Put Spread":
                st.write(f"**Buy Put:** Strike ${suggested_strategy['Buy Strike']:.2f}, Premium ${suggested_strategy['Contracts']['Buy']['lastPrice']:.2f}")
                st.write(f"**Sell Put:** Strike ${suggested_strategy['Sell Strike']:.2f}, Premium ${suggested_strategy['Contracts']['Sell']['lastPrice']:.2f}")
            
            st.write(f"**Net Debit/Credit:** {suggested_strategy['Net Debit']}")
            st.write(f"**Max Profit:** {suggested_strategy['Max Profit']}")
            st.write(f"**Max Risk:** {suggested_strategy['Max Risk']}")
            st.write(f"**Reward / Risk:** {suggested_strategy['Reward / Risk']}")
            st.markdown(f"**Notes:** {suggested_strategy['Notes']}")

            # Optional: Plotting the payoff diagram (simplified)
            st.markdown("##### Simplified Payoff Diagram")
            plot_payoff_diagram(current_stock_price, suggested_strategy['option_legs_for_chart'])

        else:
            st.info(suggested_strategy['message'])

def plot_payoff_diagram(current_price, option_legs):
    """
    Plots a simplified payoff diagram for a given options strategy.
    This is a basic representation and doesn't account for time decay, volatility changes etc.
    """
    if not option_legs:
        st.info("No option legs to plot payoff diagram.")
        return

    # Define a range of stock prices for the x-axis
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 200)
    payoff = np.zeros_like(price_range)

    for leg in option_legs:
        strike = leg['strike']
        option_type = leg['type'] # 'call' or 'put'
        action = leg['action'] # 'buy' or 'sell'
        premium = leg['premium']

        if option_type == 'call':
            if action == 'buy':
                payoff_leg = np.maximum(0, price_range - strike) - premium
            else: # sell call
                payoff_leg = -(np.maximum(0, price_range - strike) - premium)
        else: # put
            if action == 'buy':
                payoff_leg = np.maximum(0, strike - price_range) - premium
            else: # sell put
                payoff_leg = -(np.maximum(0, strike - price_range) - premium)
        
        payoff += payoff_leg

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(price_range, payoff, label='Strategy Payoff')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(current_price, color='red', linestyle=':', label='Current Price')
    ax.set_title('Options Strategy Payoff Diagram')
    ax.set_xlabel('Stock Price at Expiration')
    ax.set_ylabel('Profit/Loss')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def display_backtesting_tab(df_historical, indicator_selection):
    """
    Displays the backtesting interface and results.
    """
    st.markdown("### 🤖 Backtesting Strategy")
    st.info("This section allows you to backtest a simple strategy based on selected indicators.")

    # Backtesting parameters
    st.subheader("Backtesting Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        atr_multiplier = st.number_input("ATR Multiplier for Stop Loss", value=1.5, min_value=0.1, step=0.1)
    with col2:
        reward_risk_ratio = st.number_input("Reward/Risk Ratio for Take Profit", value=2.0, min_value=0.5, step=0.1)
    with col3:
        signal_threshold_percentage = st.slider("Min % of Selected Signals to Trigger", 0.0, 1.0, 0.7, 0.05)

    trade_direction_bt = st.radio("Backtest Trade Direction", ["long", "short"])
    exit_strategy_bt = st.radio("Exit Strategy", ["fixed_rr", "trailing_psar"], help="fixed_rr: Fixed Reward/Risk. trailing_psar: Uses Parabolic SAR as a trailing stop.")

    if st.button("Run Backtest"):
        if df_historical.empty:
            st.warning("No historical data available for backtesting. Please analyze a ticker first.")
            return

        with st.spinner("Running backtest... This may take a moment."):
            # Ensure indicators are calculated before backtesting
            df_bt = calculate_indicators(df_historical.copy())
            if df_bt.empty:
                st.error("Not enough data to calculate indicators for backtesting.")
                return

            trades, performance = backtest_strategy(
                df_bt,
                indicator_selection, # Pass the full indicator selection
                atr_multiplier,
                reward_risk_ratio,
                signal_threshold_percentage,
                trade_direction_bt,
                exit_strategy_bt
            )

            if "error" in performance:
                st.error(f"Backtesting failed: {performance['error']}")
                return

            st.subheader("Backtest Results")
            st.json(performance) # Display performance metrics as JSON for clarity

            st.subheader("Trade Log (Sample)")
            if trades:
                df_trades = pd.DataFrame(trades)
                st.dataframe(df_trades.head(10)) # Show first 10 trades
                if len(df_trades) > 10:
                    st.info(f"Showing first 10 of {len(df_trades)} trades. Full log available in the 'Trade Log' tab.")
            else:
                st.info("No trades were executed based on the specified strategy and parameters.")


def display_trade_log_tab(log_file, ticker, timeframe, overall_confidence, current_price, prev_close, trade_direction):
    """
    Displays the trade log and allows adding new entries.
    """
    st.markdown("### 📝 Trade Log")

    # Load existing log
    if os.path.exists(log_file):
        trade_log_df = pd.read_csv(log_file)
    else:
        trade_log_df = pd.DataFrame(columns=["Timestamp", "Ticker", "Timeframe", "Confidence", "Direction", "Price", "PnL", "Notes"])

    st.dataframe(trade_log_df)

    st.markdown("#### Add New Trade Entry")
    with st.form("new_trade_form"):
        trade_type = st.selectbox("Trade Type", ["Long", "Short", "Exit Long", "Exit Short"])
        price = st.number_input("Price", value=float(current_price) if current_price else 0.0, format="%.2f")
        pnl = st.number_input("PnL (if exit)", value=0.0, format="%.2f")
        notes = st.text_area("Notes")

        submitted = st.form_submit_button("Add Trade to Log")
        if submitted:
            new_entry = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Ticker": ticker,
                "Timeframe": timeframe,
                "Confidence": f"{overall_confidence:.0f}%",
                "Direction": trade_direction,
                "Price": price,
                "PnL": pnl,
                "Notes": notes
            }
            new_entry_df = pd.DataFrame([new_entry])
            trade_log_df = pd.concat([trade_log_df, new_entry_df], ignore_index=True)
            trade_log_df.to_csv(log_file, index=False)
            st.success("Trade added to log!")
            st.rerun() # Rerun to update the displayed dataframe

def display_option_calculator_tab(ticker, current_stock_price, expirations, prev_close, overall_confidence, trade_direction):
    """
    Displays a simplified options calculator.
    """
    st.markdown(f"### 🧮 Options Calculator for {ticker}")
    st.write(f"**Current Stock Price:** ${current_stock_price:.2f}")

    if not expirations:
        st.info("No expiration dates available for options calculation.")
        return

    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ["Call", "Put"])
    with col2:
        selected_expiry = st.selectbox("Expiration Date", expirations)

    strike_price = st.number_input("Strike Price", value=float(current_stock_price), format="%.2f")
    implied_volatility = st.slider("Implied Volatility (%)", 10, 100, 30) / 100.0
    time_to_expiry_days = (datetime.strptime(selected_expiry, "%Y-%m-%d").date() - datetime.now().date()).days
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0) / 100.0

    if st.button("Calculate Option Price (Black-Scholes Approximation)"):
        # This is a highly simplified Black-Scholes approximation for demonstration.
        # A full implementation is complex and requires more robust libraries.
        
        # For simplicity, let's just use a basic approximation or mock value
        # In a real app, you'd use a proper Black-Scholes library or API
        
        # Mock calculation based on simplified inputs
        if option_type == "Call":
            premium = max(0, current_stock_price - strike_price) + (implied_volatility * current_stock_price * np.sqrt(time_to_expiry_days / 365)) * 0.5
        else: # Put
            premium = max(0, strike_price - current_stock_price) + (implied_volatility * current_stock_price * np.sqrt(time_to_expiry_days / 365)) * 0.5
        
        st.success(f"Estimated Option Premium: ${premium:.2f}")
        st.info("Note: This is a simplified approximation. Real option pricing involves complex models and real-time data.")


def display_scanner_tab(scanner_results_df):
    """
    Displays the results of the stock scanner.
    """
    st.markdown("### ⚡ Stock Scanner Results")

    if scanner_results_df.empty:
        st.info("No qualifying stocks found based on your criteria.")
        return

    st.dataframe(scanner_results_df)

    st.markdown("#### Detailed Trade Plans for Scanned Stocks")
    for index, row in scanner_results_df.iterrows():
        with st.expander(f"**{row['Ticker']}** | {row['Trading Style']} | Confidence: {row['Overall Confidence']}% | Direction: {row['Direction']}"):
            st.markdown(f"**Current Price:** {row['Current Price']}")
            st.markdown(f"**ATR:** {row['ATR']}")
            st.markdown(f"**Target Price:** {row['Target Price']}")
            st.markdown(f"**Stop Loss:** {row['Stop Loss']}")
            st.markdown(f"**Entry Zone:** {row['Entry Zone']}")
            st.markdown(f"**Reward/Risk:** {row['Reward/Risk']}")
            
            st.markdown("---")
            st.markdown("**Pivot Points:**")
            st.write(f"P: {row['Pivot (P)']}, R1: {row['Resistance 1 (R1)']}, R2: {row['Resistance 2 (R2)']}, S1: {row['Support 1 (S1)']}, S2: {row['Support 2 (S2)']}")

            st.markdown("---")
            st.markdown("**Rationale:**")
            st.write(row['Rationale'])

            st.markdown("---")
            st.markdown("**Detailed Entry Criteria:**")
            st.markdown(row['Entry Criteria Details'])

            st.markdown("---")
            st.markdown("**Detailed Exit Criteria:**")
            st.markdown(row['Exit Criteria Details'])

def display_economic_sentiment_tab(economic_score, investor_sentiment_score, news_sentiment_score, finviz_headlines, latest_gdp, latest_cpi, latest_unemployment, vix_data):
    """
    Displays economic and investor sentiment data and scores.
    """
    st.markdown("### 🌍 Economic & Investor Sentiment Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Economic Score", f"{economic_score:.0f}%")
    with col2:
        st.metric("Overall Investor Sentiment Score", f"{investor_sentiment_score:.0f}%")
    with col3:
        st.metric("News Sentiment Score (Finviz)", f"{news_sentiment_score:.0f}%")

    st.markdown("---")
    st.subheader("Key Economic Indicators")
    
    if latest_gdp is not None and not latest_gdp.empty:
        st.write(f"**Latest GDP Growth:** {latest_gdp.iloc[-1]:.2f}% (as of {latest_gdp.index[-1].strftime('%Y-%m-%d')})")
    else:
        st.info("GDP data not available.")

    if latest_cpi is not None and not latest_cpi.empty:
        st.write(f"**Latest CPI (Inflation):** {latest_cpi.iloc[-1]:.2f} (as of {latest_cpi.index[-1].strftime('%Y-%m-%d')})")
    else:
        st.info("CPI data not available.")

    if latest_unemployment is not None and not latest_unemployment.empty:
        st.write(f"**Latest Unemployment Rate:** {latest_unemployment.iloc[-1]:.2f}% (as of {latest_unemployment.index[-1].strftime('%Y-%m-%d')})")
    else:
        st.info("Unemployment data not available.")

    st.markdown("---")
    st.subheader("VIX (Volatility Index)")
    if vix_data is not None and not vix_data.empty:
        st.write(f"**Latest VIX Reading:** {vix_data['Close'].iloc[-1]:.2f}")
        st.write(f"**VIX 1-Year Average:** {vix_data['Close'].mean():.2f}")
        
        # Plot VIX
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(vix_data.index, vix_data['Close'], label='VIX Close')
        ax.axhline(vix_data['Close'].mean(), color='red', linestyle='--', label='1-Year Average')
        ax.set_title('VIX (CBOE Volatility Index)')
        ax.set_xlabel('Date')
        ax.set_ylabel('VIX Value')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("VIX data not available.")

    st.markdown("---")
    st.subheader("Latest News Headlines (from Finviz)")
    if finviz_headlines:
        for i, headline in enumerate(finviz_headlines):
            st.write(f"- {headline}")
    else:
        st.info("No recent news headlines found.")

# === Helper for Indicator Display ===
def format_indicator_display(signal_name_base, current_value, bullish_fired, bearish_fired, is_selected):
    """
    Formats and displays a single technical indicator's concise information,
    showing both bullish and bearish status, plus qualitative insights.
    """
    if not is_selected:
        return "" # Don't display if not selected

    bullish_icon = '🟢' if bullish_fired else '⚪'
    bearish_icon = '🔴' if bearish_fired else '⚪'
    
    value_str = ""
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            value_str = f"Current: `{current_value:.2f}`"
        else:
            value_str = "Current: N/A"
    else:
        value_str = "Current: N/A"

    # Use the new get_indicator_summary_text to generate details
    details_text = get_indicator_summary_text(signal_name_base, current_value, bullish_fired, bearish_fired)
    # Extract the part after the initial bolded name and current value for cleaner display here
    # This is a bit of a hack, ideally get_indicator_summary_text would return structured data
    # For now, let's just append the full text, it will be markdown formatted
    
    base_display = ""
    if "ADX" in signal_name_base:
        base_display = f"{bullish_icon} {bearish_icon} **{signal_name_base}** ({value_str})"
    else:
        base_display = f"{bullish_icon} **{signal_name_base} Bullish** | {bearish_icon} **{signal_name_base} Bearish** ({value_str})"
    
    # Append the detailed summary text, starting from the qualitative status
    # This might need refinement if the output is too verbose.
    # For now, we'll just append it directly for simplicity.
    return f"{base_display}\n    - {details_text.replace(f'**{signal_name_base}:** ', '')}"


# === Common Header for Tabs ===
def _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction):
    """
    Displays common header information (ticker, current price, overall sentiment)
    at the top of various tabs.
    """
    st.markdown(f"#### {ticker} Overview")
    
    price_delta = current_price - prev_close
    
    sentiment_status = trade_direction # Use the determined trade_direction
    sentiment_icon = "⚪"
    if trade_direction == "Bullish":
        sentiment_icon = "⬆️"
    elif trade_direction == "Bearish":
        sentiment_icon = "⬇️"
        
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${price_delta:.2f}")
    with col2:
        st.markdown(f"**Overall Sentiment:** {sentiment_icon} {sentiment_status}")
    st.markdown("---")


# === Option Payoff Chart Functions ===

def calculate_payoff_from_legs(stock_prices, legs):
    """
    Calculates the total payoff for a given set of option legs across a range of stock prices.
    Each leg is expected to be a dictionary: {'type': 'call'/'put', 'strike': float, 'premium': float, 'action': 'buy'/'sell'}
    """
    total_payoff = np.zeros_like(stock_prices, dtype=float)

    for leg in legs:
        option_type = leg['type']
        strike = leg['strike']
        premium = leg['premium']
        action = leg['action']

        if option_type == 'call':
            payoff_per_share = np.maximum(0, stock_prices - strike)
        elif option_type == 'put':
            payoff_per_share = np.maximum(0, strike - stock_prices)
        else:
            continue

        if action == 'buy':
            total_payoff += (payoff_per_share - premium)
        elif action == 'sell':
            total_payoff += (premium - payoff_per_share)
    return total_payoff

def plot_generic_payoff_chart(stock_prices, payoffs, legs, strategy_name, ticker, current_stock_price):
    """
    Generates and displays an option payoff chart for a generic strategy
    based on calculated payoffs and individual legs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Breakeven Line')

    ax.plot(stock_prices, payoffs, label=f'{strategy_name} Payoff', color='blue')

    for leg in legs:
        color = 'green' if leg['action'] == 'buy' else 'red'
        linestyle = ':'
        ax.axvline(leg['strike'], color=color, linestyle=linestyle, label=f"{leg['action'].capitalize()} {leg['type'].capitalize()} Strike: ${leg['strike']:.2f}")

    ax.axvline(current_stock_price, color='orange', linestyle='-', linewidth=1.5, label=f'Current Price: ${current_stock_price:.2f}')

    breakeven_points = []
    for i in range(1, len(payoffs)):
        if (payoffs[i-1] < 0 and payoffs[i] >= 0) or (payoffs[i-1] > 0 and payoffs[i] <= 0):
            x1, y1 = stock_prices[i-1], payoffs[i-1]
            x2, y2 = stock_prices[i], payoffs[i]
            if (y2 - y1) != 0:
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakeven_points.append(breakeven)
    
    unique_breakeven_points = sorted(list(set(round(bp, 2) for bp in breakeven_points)))
    for bp in unique_breakeven_points:
        ax.axvline(bp, color='purple', linestyle='--', label=f'Breakeven: ${bp:.2f}')

    max_payoff = np.max(payoffs)
    min_payoff = np.min(payoffs)
    
    if max_payoff > 0:
        ax.text(stock_prices[-1], max_payoff * 0.9, f'Max Profit: ${max_payoff:.2f}', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=9)
    if min_payoff < 0:
        ax.text(stock_prices[-1], min_payoff * 1.1, f'Max Loss: ${min_payoff:.2f}', verticalalignment='top', horizontalalignment='right', color='red', fontsize=9)


    ax.set_title(f'{ticker} {strategy_name} Payoff Chart')
    ax.set_xlabel('Stock Price at Expiration ($)')
    ax.set_ylabel('Profit/Loss ($)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig


def display_option_calculator_tab(ticker, current_stock_price, expirations, prev_close, overall_confidence, trade_direction):
    """
    Displays a comprehensive options calculator tab, allowing users to define and visualize
    custom options strategies, including stock legs.
    """
    _display_common_header(ticker, current_stock_price, prev_close, overall_confidence, trade_direction) # Display common header
    st.subheader(f"🧮 Option Strategy Calculator for {ticker}")
    st.info("Build and analyze complex options strategies, including stock components.")

    # --- Stock Leg Input ---
    st.markdown("---")
    st.markdown("#### 📈 Stock Leg")
    col1, col2, col3 = st.columns(3)
    with col1:
        stock_action = st.selectbox("Action", ["None", "Buy", "Sell"], key="stock_action")
    with col2:
        stock_purchase_price = st.number_input("Purchase/Sale Price ($)", min_value=0.01, value=current_stock_price, format="%.2f", key="stock_price_input")
    with col3:
        num_shares = st.number_input("Number of Shares", min_value=0, value=0, step=100, key="num_shares")

    # --- Option Legs Input (Allow multiple legs) ---
    st.markdown("---")
    st.markdown("#### 📊 Option Legs")

    if 'option_legs' not in st.session_state:
        st.session_state.option_legs = []

    # Button to add a new option leg
    if st.button("➕ Add Option Leg"):
        st.session_state.option_legs.append({
            "type": "call", "action": "buy", "strike": round(current_stock_price, 2),
            "premium": 1.00, "contracts": 1, "expiration": expirations[0] if expirations else ""
        })

    # Display and allow editing of existing option legs
    legs_to_calculate = []
    for i, leg in enumerate(st.session_state.option_legs):
        st.markdown(f"**Option Leg {i+1}**")
        leg_cols = st.columns(6)
        with leg_cols[0]:
            leg["type"] = st.selectbox(f"Type {i+1}", ["call", "put"], index=0 if leg["type"] == "call" else 1, key=f"leg_type_{i}")
        with leg_cols[1]:
            leg["action"] = st.selectbox(f"Action {i+1}", ["buy", "sell"], index=0 if leg["action"] == "buy" else 1, key=f"leg_action_{i}")
        with leg_cols[2]:
            # Convert expiration strings to datetime objects for sorting
            exp_options_dt = [datetime.strptime(e, '%Y-%m-%d') for e in expirations]
            # Sort them
            exp_options_dt.sort()
            # Convert back to string for display
            sorted_expirations = [e.strftime('%Y-%m-%d') for e in exp_options_dt]

            # Find the index of the current leg's expiration in the sorted list
            try:
                current_exp_index = sorted_expirations.index(leg["expiration"])
            except ValueError:
                current_exp_index = 0 # Default to first if not found

            leg["expiration"] = st.selectbox(f"Exp. {i+1}", sorted_expirations, index=current_exp_index, key=f"leg_exp_{i}")
        with leg_cols[3]:
            leg["strike"] = st.number_input(f"Strike {i+1} ($)", min_value=0.01, value=float(leg["strike"]), format="%.2f", key=f"leg_strike_{i}")
        with leg_cols[4]:
            leg["premium"] = st.number_input(f"Premium {i+1} ($)", min_value=0.01, value=float(leg["premium"]), format="%.2f", key=f"leg_premium_{i}")
        with leg_cols[5]:
            leg["contracts"] = st.number_input(f"Contracts {i+1}", min_value=1, value=int(leg["contracts"]), step=1, key=f"leg_contracts_{i}")
        
        # Add a remove button for each leg
        if st.button(f"Remove Leg {i+1}", key=f"remove_leg_{i}"):
            st.session_state.option_legs.pop(i)
            st.rerun() # Rerun to update the list of legs

        # Add the leg to the list for calculation (adjusted for contracts)
        legs_to_calculate.extend([leg] * leg["contracts"]) # Duplicate leg for each contract

    st.markdown("---")

    # --- Calculation and Display ---
    if st.button("Calculate Payoff"):
        if not stock_action == "None" and num_shares == 0:
            st.warning("Please enter the number of shares for the stock leg, or set action to 'None'.")
        elif not legs_to_calculate and stock_action == "None":
            st.warning("Please add at least one stock or option leg to calculate the payoff.")
        else:
            # Determine the range of stock prices for the chart
            min_strike = current_stock_price * 0.8
            max_strike = current_stock_price * 1.2
            if legs_to_calculate:
                strikes = [leg['strike'] for leg in legs_to_calculate]
                min_strike = min(min_strike, min(strikes) * 0.9)
                max_strike = max(max_strike, max(strikes) * 1.1)
            
            # Extend range for potential unlimited profit/loss
            if any(leg['type'] == 'call' and leg['action'] == 'buy' for leg in legs_to_calculate):
                max_strike += current_stock_price * 0.5
            if any(leg['type'] == 'put' and leg['action'] == 'buy' for leg in legs_to_calculate):
                min_strike -= current_stock_price * 0.5

            stock_prices_range = np.linspace(min_strike, max_strike, 200)

            # Calculate payoff from stock leg
            stock_payoff = np.zeros_like(stock_prices_range, dtype=float)
            if stock_action == "Buy":
                stock_payoff = (stock_prices_range - stock_purchase_price) * num_shares
            elif stock_action == "Sell":
                stock_payoff = (stock_purchase_price - stock_prices_range) * num_shares # Corrected variable name

            # Calculate payoff from option legs
            option_payoff = calculate_payoff_from_legs(stock_prices_range, legs_to_calculate)

            total_payoff = stock_payoff + option_payoff * 100 # Options are per contract (100 shares)

            # Plot the payoff chart
            payoff_fig = plot_generic_payoff_chart(stock_prices_range, total_payoff, legs_to_calculate, "Custom Strategy", ticker, current_stock_price)
            if payoff_fig:
                st.pyplot(payoff_fig, clear_figure=True)
                plt.close(payoff_fig)
            else:
                st.error("Could not generate payoff chart.")

            st.markdown("---")
            st.subheader("📊 Estimated Returns")

            # Calculate Max Profit/Loss and Breakeven
            max_profit = np.max(total_payoff)
            min_profit = np.min(total_payoff) # This is the max loss (most negative profit)

            # Find breakeven points
            breakeven_points = []
            for i in range(1, len(total_payoff)):
                if (total_payoff[i-1] < 0 and total_payoff[i] >= 0) or \
                   (total_payoff[i-1] > 0 and total_payoff[i] <= 0):
                    x1, y1 = stock_prices_range[i-1], total_payoff[i-1]
                    x2, y2 = stock_prices_range[i], total_payoff[i]
                    if (y2 - y1) != 0:
                        breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                        breakeven_points.append(breakeven)
            unique_breakeven_points = sorted(list(set(round(bp, 2) for bp in breakeven_points)))

            st.markdown(f"**Maximum Profit:** ${max_profit:.2f}" if max_profit != np.inf else "**Maximum Profit:** Unlimited")
            st.markdown(f"**Maximum Risk:** ${-min_profit:.2f}" if min_profit != -np.inf else "**Maximum Risk:** Unlimited")
            st.markdown(f"**Breakeven Point(s):** {', '.join([f'${bp:.2f}' for bp in unique_breakeven_points]) if unique_breakeven_points else 'None'}")

            st.markdown("---")
            st.subheader("📈 Profit/Loss Table at Expiration")

            # Create a table for profit/loss at different stock prices
            table_data = []
            # Generate a more granular range for the table
            table_stock_prices = np.linspace(min_strike, max_strike, 20).round(2) # 20 points for table
            
            for price in table_stock_prices:
                # Recalculate payoff for each specific price point
                current_stock_payoff = 0
                if stock_action == "Buy":
                    current_stock_payoff = (price - stock_purchase_price) * num_shares
                elif stock_action == "Sell":
                    current_stock_payoff = (stock_purchase_price - price) * num_shares
                
                current_option_payoff = calculate_payoff_from_legs(np.array([price]), legs_to_calculate)[0] * 100
                
                total_pl = current_stock_payoff + current_option_payoff
                table_data.append({"Stock Price ($)": price, "Profit/Loss ($)": total_pl})
            
            st.dataframe(pd.DataFrame(table_data).set_index("Stock Price ($)"))

    st.markdown("---")
    st.info("Note: This calculator assumes expiration and does not account for time decay or implied volatility changes before expiration.")


# === Dashboard Tab Display Functions ===
def display_main_analysis_tab(ticker, df, info, params, selection, overall_confidence, scores, final_weights, sentiment_score, expert_score, df_pivots, trade_direction):
    """Displays the main technical analysis and confidence score tab."""
    is_intraday = params['interval'] in ['5m', '60m']
    last = df.iloc[-1]
    
    # Generate both bullish and bearish signals for the last row
    bullish_signals, bearish_signals = generate_signals_for_row(last)

    col1, col2 = st.columns([1, 2])
    with col1:
        # --- Ticker Price and General Info (Moved to Top) ---
        st.subheader(f"📊 {info.get('longName', ticker)}")
        st.write(f"**Ticker:** {ticker}")

        current_price = last['Close']
        prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
        price_delta = current_price - prev_close
        
        # Determine bullish/bearish based on overall confidence
        sentiment_status = trade_direction # Use the determined trade_direction
        sentiment_icon = "⚪"
        if trade_direction == "Bullish":
            sentiment_icon = "⬆️"
        elif trade_direction == "Bearish":
            sentiment_icon = "⬇️"
        
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"${price_delta:.2f}")
        st.markdown(f"**Overall Sentiment:** {sentiment_icon} {sentiment_status}")

        st.markdown("---")

        st.subheader("💡 Confidence Score")
        st.metric("Overall Confidence", f"{overall_confidence:.0f}/100")
        st.progress(overall_confidence / 100)
        
        # Convert numerical sentiment score to descriptive text
        sentiment_text = "N/A (Excluded)"
        if sentiment_score is not None:
            if sentiment_score >= 75:
                sentiment_text = "High"
            elif sentiment_score <= 25:
                sentiment_text = "Low"
            else:
                sentiment_text = "Neutral"

        # Convert numerical expert score to descriptive text using EXPERT_RATING_MAP
        expert_text = "N/A (Excluded)"
        if expert_score is not None:
            for key, value in EXPERT_RATING_MAP.items():
                # Corrected logic: Compare expert_score (numerical) with value (numerical)
                if expert_score == value:
                    expert_text = key
                    break
            if expert_text == "N/A (Excluded)" and expert_score == 50: # Default for Hold if not explicitly mapped
                expert_text = "Hold"


        st.markdown(f"- **Technical Score:** `{scores['Technical']:.0f}` (Weight: `{final_weights['technical']*100:.0f}%`)\n"
                    f"- **Sentiment Score:** {sentiment_text} (Weight: `{final_weights['sentiment']*100:.0f}%`)\n"
                    f"- **Expert Rating:** {expert_text} (Weight: `{final_weights['expert']*100:.0f}%`)\n"
                    f"- **Economic Score:** `{scores['Economic']:.0f}` (Weight: `{final_weights['economic']*100:.0f}%`)\n"
                    f"- **Investor Sentiment Score:** `{scores['Investor Sentiment']:.0f}` (Weight: `{final_weights['investor_sentiment']*100:.0f}%`)")
        
        # Always show Finviz link if automation is enabled
        st.markdown(f"**Source for Sentiment & Expert Scores:** [Finviz.com]({f'https://finviz.com/quote.ashx?t={ticker}'})")

        st.markdown("---")

        st.subheader("✅ Technical Analysis Readout")
        with st.expander("📈 Trend Indicators", expanded=True):
            # Updated calls to format_indicator_display to show both bullish/bearish status
            if selection.get("EMA Trend"):
                st.markdown(format_indicator_display("EMA Trend", None, bullish_signals.get("EMA Trend", False), bearish_signals.get("EMA Trend", False), selection.get("EMA Trend")))
            
            if selection.get("Ichimoku Cloud"):
                st.markdown(format_indicator_display("Ichimoku Cloud", None, bullish_signals.get("Ichimoku Cloud", False), bearish_signals.get("Ichimoku Cloud", False), selection.get("Ichimoku Cloud")))

            if selection.get("Parabolic SAR"):
                st.markdown(format_indicator_display("Parabolic SAR", last.get('psar'), bullish_signals.get("Parabolic SAR", False), bearish_signals.get("Parabolic SAR", False), selection.get("Parabolic SAR")))

            if selection.get("ADX"):
                st.markdown(format_indicator_display("ADX", last.get("adx"), bullish_signals.get("ADX", False), bearish_signals.get("ADX", False), selection.get("ADX")))
        
        with st.expander("💨 Momentum & Volume Indicators", expanded=True):
            if selection.get("RSI Momentum"):
                st.markdown(format_indicator_display("RSI Momentum", last.get("RSI"), bullish_signals.get("RSI Momentum", False), bearish_signals.get("RSI Momentum", False), selection.get("RSI Momentum")))

            if selection.get("Stochastic"):
                st.markdown(format_indicator_display("Stochastic Oscillator", last.get("stoch_k"), bullish_signals.get("Stochastic", False), bearish_signals.get("Stochastic", False), selection.get("Stochastic")))

            if selection.get("CCI"):
                st.markdown(format_indicator_display("CCI", last.get("CCI"), bullish_signals.get("CCI", False), bearish_signals.get("CCI", False), selection.get("CCI")))

            if selection.get("ROC"):
                st.markdown(format_indicator_display("ROC", last.get("ROC"), bullish_signals.get("ROC", False), bearish_signals.get("ROC", False), selection.get("ROC")))

            if selection.get("Volume Spike"):
                st.markdown(format_indicator_display("Volume Spike", last.get("Volume"), bullish_signals.get("Volume Spike", False), bearish_signals.get("Volume Spike", False), selection.get("Volume Spike")))

            if selection.get("OBV"):
                st.markdown(format_indicator_display("OBV", last.get("obv"), bullish_signals.get("OBV", False), bearish_signals.get("OBV", False), selection.get("OBV")))
            
            if is_intraday and selection.get("VWAP"):
                st.markdown(format_indicator_display("VWAP", last.get("VWAP"), bullish_signals.get("VWAP", False), bearish_signals.get("VWAP", False), selection.get("VWAP")))
        
        with st.expander("📊 Display-Only Indicators Status"):
            # Bollinger Bands Status
            if selection.get("Bollinger Bands"):
                if 'BB_high' in last and 'BB_low' in last and not pd.isna(last['BB_high']) and not pd.isna(last['BB_low']):
                    if last['Close'] > last['BB_high']:
                        bb_status = '🔴 **Price Above Upper Band** (Overbought/Strong Uptrend)'
                    elif last['Close'] < last['BB_low']:
                        bb_status = '🟢 **Price Below Lower Band** (Oversold/Strong Downtrend)'
                    else:
                        bb_status = '🟡 **Price Within Bands** (Neutral/Consolidation)'
                    st.markdown(f"**Bollinger Bands:** {bb_status}")
                else:
                    st.info("Bollinger Bands data not available for display.")

            # Pivot Points Status
            if selection.get("Pivot Points") and not is_intraday: # Pivot Points are for daily/weekly
                if not df_pivots.empty and len(df_pivots) > 1:
                    last_pivot = df_pivots.iloc[-1] # This is the pivot for the current day (calculated from previous day's data)
                    if 'Pivot' in last_pivot and not pd.isna(last_pivot['Pivot']):
                        if last['Close'] > last_pivot['R1']:
                            pivot_status = '🟢 **Price Above R1** (Strong Bullish)'
                        elif last['Close'] > last_pivot['Pivot']:
                            pivot_status = '🟡 **Price Above Pivot** (Bullish)'
                        elif last['Close'] < last_pivot['S1']:
                            pivot_status = '🔴 **Price Below S1** (Strong Bearish)'
                        elif last['Close'] < last_pivot['Pivot']:
                            pivot_status = '🟡 **Price Below Pivot** (Bearish)'
                        else:
                            pivot_status = '⚪ **Price Near Pivot** (Neutral/Ranging)'
                        st.markdown(f"**Pivot Points:** {pivot_status}")
                    else:
                        st.info("Pivot Points data not fully available for display.")
                else:
                    st.info("Pivot Points data not available for display or not enough history.")
            elif selection.get("Pivot Points") and is_intraday:
                st.info("Pivot Points are typically used for daily/weekly timeframes, not intraday.")

    with col2:
        st.subheader("📈 Price Chart")
        mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
        
        ap = [] # Initialize addplot as an empty list
        
        # Add Bollinger Bands to addplot if selected and data is available
        if selection.get("Bollinger Bands"):
            # Check if BB columns exist and are not all NaN in the tail data
            if 'BB_high' in df.columns and 'BB_low' in df.columns and not df[['BB_high', 'BB_low']].tail(120).isnull().all().all():
                ap.append(mpf.make_addplot(df.tail(120)[['BB_high', 'BB_low']]))
            else:
                st.warning("Bollinger Bands data not available or all NaN for plotting.", icon="⚠️")

        # Add Pivot Points to addplot if selected and data is available (for daily/weekly)
        if selection.get("Pivot Points") and not is_intraday and not df_pivots.empty and len(df_pivots) > 1:
            last_pivot = df_pivots.iloc[-1]
            # Ensure pivot values are not NaN before attempting to plot
            if not pd.isna(last_pivot.get('Pivot')):
                # Create Series aligned with the chart's index (df.tail(120).index)
                # This ensures the horizontal lines span the visible chart
                chart_index = df.tail(120).index
                
                pivot_values = pd.Series(last_pivot['Pivot'], index=chart_index)
                r1_values = pd.Series(last_pivot['R1'], index=chart_index)
                s1_values = pd.Series(last_pivot['S1'], index=chart_index)
                r2_values = pd.Series(last_pivot['R2'], index=chart_index)
                s2_values = pd.Series(last_pivot['S2'], index=chart_index)

                ap.append(mpf.make_addplot(pivot_values, color='purple', linestyle='--', panel=0, width=0.7, secondary_y=False))
                ap.append(mpf.make_addplot(r1_values, color='red', linestyle=':', panel=0, width=0.7, secondary_y=False))
                ap.append(mpf.make_addplot(s1_values, color='green', linestyle=':', panel=0, width=0.7, secondary_y=False))
                ap.append(mpf.make_addplot(r2_values, color='darkred', linestyle='--', panel=0, width=0.7, secondary_y=False))
                ap.append(mpf.make_addplot(s2_values, color='darkgreen', linestyle='--', panel=0, width=0.7, secondary_y=False))
            else:
                st.info("Pivot Points data not fully available for plotting on chart.")


        if not df.empty:
            fig, axlist = mpf.plot(
                df.tail(120),
                type='candle',
                style='yahoo',
                mav=mav_tuple,
                volume=True,
                addplot=ap,
                title=f"{ticker} - {params['interval']} chart",
                returnfig=True
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.info("Not enough data to generate chart.")

# Removed duplicate display_backtest_tab here. The one below is kept.

def display_news_info_tab(ticker, info_data, finviz_data, current_price, prev_close, overall_confidence, trade_direction):
    """Displays general information and news headlines for the ticker."""
    _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction) # Display common header
    st.subheader(f"📰 News and Information for {ticker}")

    if info_data:
        st.markdown("---")
        st.subheader("Company Profile")
        st.write(f"**Sector:** {info_data.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info_data.get('industry', 'N/A')}")
        st.write(f"**Full Time Employees:** {info_data.get('fullTimeEmployees', 'N/A')}")
        st.write(f"**Website:** [{info_data.get('website', 'N/A')}]({info_data.get('website', '#')})")
        
        st.write("**Description:**")
        st.write(info_data.get('longBusinessSummary', 'No description available.'))
        
        st.markdown("---")
        st.subheader("Key Financials & Metrics (from Yahoo Finance)")
        col1, col2 = st.columns(2)
        with col1:
            market_cap = info_data.get('marketCap', 'N/A')
            st.write(f"**Market Cap:** {market_cap:,}" if isinstance(market_cap, (int, float)) else f"**Market Cap:** {market_cap}")
            
            shares_outstanding = info_data.get('sharesOutstanding', 'N/A')
            st.write(f"**Shares Outstanding:** {shares_outstanding:,}" if isinstance(shares_outstanding, (int, float)) else f"**Shares Outstanding:** {shares_outstanding}")
            
            beta = info_data.get('beta', 'N/A')
            st.write(f"**Beta:** {beta:.2f}" if isinstance(beta, (int, float)) else f"**Beta:** {beta}")
            
            peg_ratio = info_data.get('pegRatio', 'N/A')
            st.write(f"**PEG Ratio:** {peg_ratio:.2f}" if isinstance(peg_ratio, (int, float)) else f"**PEG Ratio:** {peg_ratio}")
            
            dividend_yield = info_data.get('dividendYield', 'N/A')
            st.write(f"**Dividend Yield:** {dividend_yield*100:.2f}%" if isinstance(dividend_yield, (int, float)) else f"**Dividend Yield:** {dividend_yield}")
        with col2:
            trailing_pe = info_data.get('trailingPE', 'N/A')
            st.write(f"**P/E Ratio (TTM):** {trailing_pe:.2f}" if isinstance(trailing_pe, (int, float)) else f"**P/E Ratio (TTM):** {trailing_pe}")
            
            forward_pe = info_data.get('forwardPE', 'N/A')
            st.write(f"**Forward P/E:** {forward_pe:.2f}" if isinstance(forward_pe, (int, float)) else f"**Forward P/E:** {forward_pe}")
            
            ebitda = info_data.get('ebitda', 'N/A')
            st.write(f"**EBITDA:** {ebitda:,}" if isinstance(ebitda, (int, float)) else f"**EBITDA:** {ebitda}")
            
            revenue_ttm = info_data.get('revenueTTM', 'N/A')
            st.write(f"**Revenue (TTM):** {revenue_ttm:,}" if isinstance(revenue_ttm, (int, float)) else f"**Revenue (TTM):** {revenue_ttm}")
            
            gross_profits = info_data.get('grossProfits', 'N/A')
            st.write(f"**Gross Profits (TTM):** {gross_profits:,}" if isinstance(gross_profits, (int, float)) else f"**Gross Profits (TTM):** {gross_profits}")


    else:
        st.warning("No comprehensive company information available from Yahoo Finance.")

    st.markdown("---")
    st.subheader("Latest News Headlines (from Finviz)")
    if finviz_data and finviz_data.get('headlines'):
        for headline in finviz_data['headlines']:
            st.markdown(f"- [{headline['title']}]({headline['link']}) ({headline['date']})")
    else:
        st.info("No recent news headlines available from Finviz.")

def display_trade_log_tab(log_file, ticker, timeframe, overall_confidence, current_price, prev_close, trade_direction):
    """
    Displays the trade log and provides functionality to add new trades.
    """
    _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction) # Display common header
    st.subheader(f"📝 Trade Log for {ticker}")

    # Ensure the log file exists
    if not os.path.exists(log_file):
        df_log = pd.DataFrame(columns=["Date", "Time", "Ticker", "Trade Type", "Entry Price", "Exit Price", "Quantity", "P/L", "Notes"])
        df_log.to_csv(log_file, index=False)
    else:
        df_log = pd.read_csv(log_file)

    st.markdown("---")
    st.subheader("Add New Trade")

    with st.form("trade_entry_form"):
        col1, col2 = st.columns(2)
        with col1:
            trade_type = st.selectbox("Trade Type", ["Long", "Short"], key="new_trade_type")
            entry_price = st.number_input("Entry Price", min_value=0.01, format="%.2f", key="new_entry_price")
        with col2:
            quantity = st.number_input("Quantity (Shares)", min_value=1, step=1, key="new_quantity")
            exit_price = st.number_input("Exit Price (Optional, for closed trades)", min_value=0.01, format="%.2f", value=None, key="new_exit_price")
        
        notes = st.text_area("Notes", key="new_notes")
        
        submitted = st.form_submit_button("Add Trade to Log")

        if submitted:
            if entry_price <= 0 or quantity <= 0:
                st.error("Entry Price and Quantity must be positive values.")
            else:
                current_datetime = datetime.now()
                date_str = current_datetime.strftime("%Y-%m-%d")
                time_str = current_datetime.strftime("%H:%M:%S")
                
                pl = None
                if exit_price is not None:
                    if trade_type == "Long":
                        pl = (exit_price - entry_price) * quantity
                    elif trade_type == "Short":
                        pl = (entry_price - exit_price) * quantity

                new_trade = pd.DataFrame([{
                    "Date": date_str,
                    "Time": time_str,
                    "Ticker": ticker,
                    "Trade Type": trade_type,
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Quantity": quantity,
                    "P/L": pl,
                    "Notes": notes
                }])
                
                df_log = pd.concat([df_log, new_trade], ignore_index=True)
                df_log.to_csv(log_file, index=False)
                st.success("Trade added successfully!")
                st.rerun() # Refresh to show updated log

    st.markdown("---")
    st.subheader("Your Trade History")
    if not df_log.empty:
        # Filter log to show only trades for the current ticker
        df_ticker_log = df_log[df_log['Ticker'].str.upper() == ticker.upper()].copy()

        if not df_ticker_log.empty:
            st.dataframe(df_ticker_log)
            # Option to download full log
            csv = df_log.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Trade Log (CSV)",
                data=csv,
                file_name="trade_log.csv",
                mime="text/csv",
                key="download_trade_log"
            )
        else:
            st.info(f"No trades logged for {ticker} yet.")
    else:
        st.info("No trades logged yet.")

# --- NEW: Display Economic Data Tab ---
def display_economic_data_tab(ticker, current_price, prev_close, overall_confidence, trade_direction,
                              latest_gdp, latest_cpi, latest_unemployment):
    """Displays key economic data."""
    _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction)
    st.subheader("🌍 Key Economic Indicators")
    st.info("Understanding the broader economic landscape is crucial for long-term trading decisions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latest GDP Growth (Annualized)", f"{latest_gdp:.2f}%" if latest_gdp is not None else "N/A")
        st.markdown("*(Source: FRED - Gross Domestic Product)*")
    with col2:
        st.metric("Latest CPI (Inflation)", f"{latest_cpi:.2f}" if latest_cpi is not None else "N/A")
        st.markdown("*(Source: FRED - Consumer Price Index)*")
    with col3:
        st.metric("Latest Unemployment Rate", f"{latest_unemployment:.2f}%" if latest_unemployment is not None else "N/A")
        st.markdown("*(Source: FRED - Unemployment Rate)*")
    
    st.markdown("---")
    st.subheader("Economic Outlook Summary")
    # You could add more detailed interpretation here based on the values
    st.markdown("""
    * **GDP Growth:** Indicates the overall health and expansion of the economy. Strong growth is generally bullish for stocks.
    * **CPI (Inflation):** Measures the rate of price increases. High inflation can lead to interest rate hikes, which may negatively impact markets.
    * **Unemployment Rate:** A low unemployment rate suggests a strong labor market and consumer spending, generally bullish.
    """)
    st.markdown("---")
    st.info("Note: Economic data is often released periodically (e.g., monthly, quarterly) and may not reflect real-time changes.")

# --- NEW: Display Investor Sentiment Tab ---
def display_investor_sentiment_tab(ticker, current_price, prev_close, overall_confidence, trade_direction,
                                   latest_vix, historical_vix_avg):
    """Displays key investor sentiment indicators."""
    _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction)
    st.subheader("❤️ Investor Sentiment Indicators")
    st.info("Gauging market fear and greed can provide insights into potential reversals or continuations.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest VIX", f"{latest_vix:.2f}" if latest_vix is not None else "N/A")
        st.markdown("*(Source: CBOE Volatility Index, via Yahoo Finance)*")
    with col2:
        if historical_vix_avg is not None:
            st.metric("Historical VIX Average (Past Year)", f"{historical_vix_avg:.2f}")
        else:
            st.metric("Historical VIX Average (Past Year)", "N/A")
    
    st.markdown("---")
    st.subheader("VIX Interpretation")
    if latest_vix is not None:
        if latest_vix < 15:
            st.success("Current VIX is **low (<15)**, indicating low market fear and complacency. This can sometimes precede market tops.")
        elif 15 <= latest_vix <= 20:
            st.info("Current VIX is **moderate (15-20)**, reflecting typical market volatility.")
        elif 20 < latest_vix <= 30:
            st.warning("Current VIX is **elevated (20-30)**, suggesting increased market fear and uncertainty. This can signal potential bottoms or heightened volatility.")
        else: # VIX > 30
            st.error("Current VIX is **high (>30)**, indicating extreme market fear and panic. Historically, high VIX levels have often coincided with market bottoms.")
    else:
        st.info("VIX data not available for interpretation.")
    
    st.markdown("---")
    st.info("Note: Sentiment indicators are often contrarian. Extreme fear can be a buying opportunity, and extreme complacency a selling opportunity.")

# --- NEW: Display Scanner Results Tab ---
def display_scanner_results_tab(scanner_results_df):
    """
    Displays the results of the stock scanner in a detailed, expandable table.
    """
    st.subheader("📈 Scanned Opportunities")
    if scanner_results_df.empty:
        st.info("No opportunities found matching your criteria.")
        return

    # Create a list of dictionaries for display, with expanders for details
    display_data = []
    for index, row in scanner_results_df.iterrows():
        # Ensure all expected keys exist, provide defaults if not
        entry_details_expander = f"**Entry Criteria Details for {row['Ticker']}**\n\n{row.get('Entry Criteria Details', 'N/A')}"
        exit_details_expander = f"**Exit Criteria Details for {row['Ticker']}**\n\n{row.get('Exit Criteria Details', 'N/A')}"

        display_data.append({
            "Ticker": row.get('Ticker', 'N/A'),
            "Style": row.get('Trading Style', 'N/A'),
            "Confidence": row.get('Overall Confidence', 'N/A'),
            "Direction": row.get('Direction', 'N/A'),
            "Price": row.get('Current Price', 'N/A'),
            "ATR": row.get('ATR', 'N/A'),
            "Target": row.get('Target Price', 'N/A'),
            "Stop Loss": row.get('Stop Loss', 'N/A'),
            "Entry Zone": row.get('Entry Zone', 'N/A'),
            "R/R": row.get('Reward/Risk', 'N/A'),
            "Pivot (P)": row.get('Pivot (P)', 'N/A'),
            "R1": row.get('Resistance 1 (R1)', 'N/A'),
            "S1": row.get('Support 1 (S1)', 'N/A'),
            "R2": row.get('Resistance 2 (R2)', 'N/A'),
            "S2": row.get('Support 2 (S2)', 'N/A'),
            "Entry Details": entry_details_expander, # This will be expanded
            "Exit Details": exit_details_expander,   # This will be expanded
            "Rationale": row.get('Rationale', 'N/A')
        })
    
    # Use st.expander for each row to show details
    for item in display_data:
        with st.expander(f"**{item['Ticker']}** | {item['Style']} | Confidence: {item['Confidence']}% | Direction: {item['Direction']}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", item['Price'])
            col2.metric("Target Price", item['Target'])
            col3.metric("Stop Loss", item['Stop Loss'])
            col4.metric("Reward/Risk", item['R/R'])
            
            st.markdown("---")
            st.markdown(f"**Entry Zone:** {item['Entry Zone']}")
            st.markdown(f"**ATR:** {item['ATR']}")

            st.markdown("---")
            st.markdown("**Pivot Points:**")
            st.write(f"P: {item['Pivot (P)']}, R1: {item['R1']}, S1: {item['S1']}, R2: {item['R2']}, S2: {item['S2']}")

            st.markdown("---")
            st.markdown("**Rationale:**")
            st.write(item['Rationale'])

            st.markdown("---")
            st.markdown("**Detailed Entry Criteria:**")
            st.markdown(item['Entry Details'])

            st.markdown("---")
            st.markdown("**Detailed Exit Criteria:**")
            st.markdown(item['Exit Details'])
            
            st.markdown("---") # Separator for next item

    st.markdown("---")
    st.info("Click on each ticker's header to expand/collapse detailed trade plan information.")

