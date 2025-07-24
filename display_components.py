# display_components.py - Version 1.47

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
    generate_option_trade_plan, get_options_chain, get_data, get_finviz_data,
    calculate_pivot_points, EXPERT_RATING_MAP, get_moneyness, analyze_options_chain,
    generate_directional_trade_plan # NEW: Import the directional trade plan generator
)

# === Helper for Indicator Display ===
def format_indicator_display(signal_name_base, current_value, bullish_fired, bearish_fired, is_selected):
    """
    Formats and displays a single technical indicator's concise information,
    showing both bullish and bearish status.
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

    # For ADX, it's non-directional, so we'll just show its value and overall trend strength
    if "ADX" in signal_name_base: # Check for "ADX" in the base name
        return f"{bullish_icon} {bearish_icon} **{signal_name_base}** ({value_str})"
    else:
        return f"{bullish_icon} **{signal_name_base} Bullish** | {bearish_icon} **{signal_name_base} Bearish** ({value_str})"


# === Common Header for Tabs ===
def _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction): # Added trade_direction
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


def display_option_calculator_tab(ticker, current_stock_price, expirations, prev_close, overall_confidence, trade_direction): # Added trade_direction
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
def display_main_analysis_tab(ticker, df, info, params, selection, overall_confidence, scores, final_weights, sentiment_score, expert_score, df_pivots, trade_direction): # Added trade_direction
    """Displays the main technical analysis and confidence score tab."""
    is_intraday = params['interval'] in ['5m', '60m']
    last = df.iloc[-1]
    
    # Generate both bullish and bearish signals for the last row
    bullish_signals, bearish_signals = generate_signals_for_row(last, selection, df, is_intraday)

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
                if expert_score == value:
                    expert_text = key
                    break
            if expert_text == "N/A (Excluded)" and expert_score == 50:
                expert_text = "Hold"


        st.markdown(f"- **Technical Score:** `{scores['technical']:.0f}` (Weight: `{final_weights['technical']*100:.0f}%`)\n"
                    f"- **Sentiment Score:** {sentiment_text} (Weight: `{final_weights['sentiment']*100:.0f}%`)\n"
                    f"- **Expert Rating:** {expert_text} (Weight: `{final_weights['expert']*100:.0f}%`)")
        
        # Always show Finviz link if automation is enabled
        st.markdown(f"**Source for Sentiment & Expert Scores:** [Finviz.com]({f'https://finviz.com/quote.ashx?t={ticker}'})")

        st.markdown("---")

        # --- 52-Week High/Low ---
        st.subheader("🗓️ 52-Week Range")
        week_52_high = info.get('fiftyTwoWeekHigh', 'N/A')
        week_52_low = info.get('fiftyTwoWeekLow', 'N/A')
        st.write(f"**High:** ${week_52_high:.2f}" if isinstance(week_52_high, (int, float)) else f"**High:** {week_52_high}")
        st.write(f"**Low:** ${week_52_low:.2f}" if isinstance(week_52_low, (int, float)) else f"**Low:** {week_52_low}")

        st.markdown("---")

        st.subheader("✅ Technical Analysis Readout")
        with st.expander("📈 Trend Indicators", expanded=True):
            # Updated calls to format_indicator_display to show both bullish/bearish status
            if selection.get("EMA Trend"):
                st.markdown(format_indicator_display("EMA Trend", None, bullish_signals.get("Uptrend (21>50>200 EMA)", False), bearish_signals.get("Downtrend (21<50<200 EMA)", False), selection.get("EMA Trend")))
            
            if selection.get("Ichimoku Cloud"):
                st.markdown(format_indicator_display("Ichimoku Cloud", None, bullish_signals.get("Bullish Ichimoku", False), bearish_signals.get("Bearish Ichimoku", False), selection.get("Ichimoku Cloud")))

            if selection.get("Parabolic SAR"):
                st.markdown(format_indicator_display("Parabolic SAR", last.get('psar'), bullish_signals.get("Bullish PSAR", False), bearish_signals.get("Bearish PSAR", False), selection.get("Parabolic SAR")))

            if selection.get("ADX"):
                st.markdown(format_indicator_display("ADX", last.get("adx"), bullish_signals.get("Strong Trend (ADX > 25)", False), bearish_signals.get("Strong Trend (ADX > 25)", False), selection.get("ADX")))
        
        with st.expander("💨 Momentum & Volume Indicators", expanded=True):
            if selection.get("RSI Momentum"):
                st.markdown(format_indicator_display("RSI Momentum", last.get("RSI"), bullish_signals.get("Bullish Momentum (RSI > 50)", False), bearish_signals.get("Bearish Momentum (RSI < 50)", False), selection.get("RSI Momentum")))

            if selection.get("Stochastic"):
                st.markdown(format_indicator_display("Stochastic Oscillator", last.get("stoch_k"), bullish_signals.get("Bullish Stoch Cross", False), bearish_signals.get("Bearish Stoch Cross", False), selection.get("Stochastic")))

            if selection.get("CCI"):
                st.markdown(format_indicator_display("CCI", last.get("cci"), bullish_signals.get("Bullish CCI (>0)", False), bearish_signals.get("Bearish CCI (<0)", False), selection.get("CCI")))

            if selection.get("ROC"):
                st.markdown(format_indicator_display("ROC", last.get("roc"), bullish_signals.get("Positive ROC (>0)", False), bearish_signals.get("Negative ROC (<0)", False), selection.get("ROC")))

            if selection.get("Volume Spike"):
                st.markdown(format_indicator_display("Volume Spike", last.get("Volume"), bullish_signals.get("Volume Spike (Up Move)", False), bearish_signals.get("Volume Spike (Down Move)", False), selection.get("Volume Spike")))

            if selection.get("OBV"):
                st.markdown(format_indicator_display("OBV", last.get("obv"), bullish_signals.get("OBV Rising", False), bearish_signals.get("OBV Falling", False), selection.get("OBV")))
            
            if is_intraday and selection.get("VWAP"):
                st.markdown(format_indicator_display("VWAP", last.get("vwap"), bullish_signals.get("Price > VWAP", False), bearish_signals.get("Price < VWAP", False), selection.get("VWAP")))
        
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

def display_trade_plan_options_tab(ticker, df, overall_confidence, timeframe, trade_direction): # Added timeframe, trade_direction
    """Displays the suggested trade plan and options strategy."""
    last = df.iloc[-1]
    current_stock_price = last['Close']
    prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_stock_price
    
    # This mapping should ideally come from app.py's TIMEFRAME_MAP for consistency
    interval_map_for_trade_plan = {
        "Scalp Trading": "5m",
        "Day Trading": "60m",
        "Swing Trading": "1d",
        "Position Trading": "1wk"
    }
    period_interval = interval_map_for_trade_plan.get(timeframe, "1d") # Default to 1d if not found

    _display_common_header(ticker, current_stock_price, prev_close, overall_confidence, trade_direction) # Display common header

    # Dynamic subheader based on determined trade direction
    if trade_direction != "Neutral":
        st.subheader(f"📋 Suggested Stock Trade Plan ({trade_direction} {timeframe})")
    else:
        st.subheader("📋 Stock Trade Plan")

    # Generate the trade plan using the new directional function
    trade_plan_result = generate_directional_trade_plan(
        current_stock_price,
        last.get('ATR'), # Ensure ATR is available
        trade_direction,
        timeframe,
        period_interval # Pass the interval
    )

    if trade_plan_result['status'] == 'success':
        st.info(f"**Based on {overall_confidence:.0f}% Overall Confidence ({trade_plan_result['direction']}):**\n\n"
                f"**Entry Zone:** Between **${trade_plan_result['entry_zone_start']:.2f}** and **${trade_plan_result['entry_zone_end']:.2f}**.\n"
                f"**Stop-Loss:** A close {'below' if trade_plan_result['direction'] == 'Bullish' else 'above'} **${trade_plan_result['stop_loss']:.2f}**.\n"
                f"**Profit Target:** Around **${trade_plan_result['profit_target']:.2f}** ({trade_plan_result['reward_risk_ratio']:.1f}:1 Reward/Risk).")
    else:
        st.warning(trade_plan_result['message'])
    
    st.markdown("---")
    
    st.subheader("🎭 Automated Options Strategy")
    stock_obj = yf.Ticker(ticker)
    expirations = stock_obj.options
    if not expirations:
        st.warning("No options data available for this ticker.")
    else:
        # Pass trade_direction to generate_option_trade_plan
        trade_plan = generate_option_trade_plan(ticker, overall_confidence, current_stock_price, expirations, trade_direction)
        
        # --- Start of new detailed options display ---
        if trade_plan['status'] == 'success':
            st.success(f"**Recommended Strategy: {trade_plan['Strategy']}** (Confidence: {overall_confidence:.0f}%)")
            st.info(trade_plan['Reason'])
            
            if trade_plan['Strategy'] == "Bull Call Spread":
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Buy Strike", trade_plan['Buy Strike'])
                col2.metric("Sell Strike", trade_plan['Sell Strike'])
                col3.metric("Expiration", trade_plan['Expiration'])
                col4.metric("Net Debit", trade_plan['Net Debit'])
                col5.metric("Max Profit / Max Risk", f"{trade_plan['Max Profit']} / {trade_plan['Max Risk']}")
                st.write(f"**Reward / Risk:** `{trade_plan['Reward / Risk']}`")
                st.markdown("---")
                st.subheader("🔬 Recommended Option Deep-Dive (Spread Legs)")
                
                if 'Buy' in trade_plan['Contracts']:
                    st.write("**Buy Leg:**")
                    rec_option_buy = trade_plan['Contracts']['Buy']
                    moneyness_buy = get_moneyness(rec_option_buy.get('strike'), current_stock_price, "call")

                    option_metrics_buy = [
                        {"Metric": "Strike", "Value": f"${rec_option_buy.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Buyers": "Lower for calls, higher for puts"},
                        {"Metric": "Moneyness", "Value": moneyness_buy, "Description": "In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).", "Ideal for Buyers": "Depends on strategy"},
                        {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Buyers": "Longer term (45-365 days)"},
                        {"Metric": f"Value ({ticker})", "Value": f"${rec_option_buy.get('lastPrice', None):.2f}" if rec_option_buy.get('lastPrice') is not None and not pd.isna(rec_option_buy.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Bid", "Value": f"${rec_option_buy.get('bid', None):.2f}" if rec_option_buy.get('bid') is not None and not pd.isna(rec_option_buy.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Ask", "Value": f"${rec_option_buy.get('ask', None):.2f}" if rec_option_buy.get('ask') is not None and not pd.isna(rec_option_buy.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_buy.get('impliedVolatility', None):.2%}" if rec_option_buy.get('impliedVolatility') is not None and not pd.isna(rec_option_buy.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Buyers": "Lower is better"},
                        {"Metric": "Delta", "Value": f"{rec_option_buy.get('delta', None):.2f}" if rec_option_buy.get('delta') is not None and not pd.isna(rec_option_buy.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Buyers": "0.60 to 0.80 (for ITM calls)"},
                        {"Metric": "Theta", "Value": f"{rec_option_buy.get('theta', None):.3f}" if rec_option_buy.get('theta') is not None and not pd.isna(rec_option_buy.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Buyers": "As low as possible"},
                        {"Metric": "Gamma", "Value": f"{rec_option_buy.get('gamma', None):.3f}" if rec_option_buy.get('gamma') is not None and not pd.isna(rec_option_buy.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Buyers": "Higher for directional plays"},
                        {"Metric": "Vega", "Value": f"{rec_option_buy.get('vega', None):.3f}" if rec_option_buy.get('vega') is not None and not pd.isna(rec_option_buy.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                        {"Metric": "Rho", "Value": f"{rec_option_buy.get('rho', None):.3f}" if rec_option_buy.get('rho') is not None and not pd.isna(rec_option_buy.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                        {"Metric": "Open Interest", "Value": f"{rec_option_buy.get('openInterest', None):,}" if rec_option_buy.get('openInterest') is not None and not pd.isna(rec_option_buy.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Buyers": "> 100s"},
                        {"Metric": "Volume (Today)", "Value": f"{rec_option_buy.get('volume', None):,}" if rec_option_buy.get('volume') is not None and not pd.isna(rec_option_buy.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Buyers": "Higher (>100)"},
                    ]
                    st.table(pd.DataFrame(option_metrics_buy).set_index("Metric"))
                else:
                    st.warning("Buy leg details not available for the Bull Call Spread.")

                if 'Sell' in trade_plan['Contracts']:
                    st.write("**Sell Leg:**")
                    rec_option_sell = trade_plan['Contracts']['Sell']
                    moneyness_sell = get_moneyness(rec_option_sell.get('strike'), current_stock_price, "call")

                    option_metrics_sell = [
                        {"Metric": "Strike", "Value": f"${rec_option_sell.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Sellers": "Higher for calls, lower for puts"},
                        {"Metric": "Moneyness", "Value": moneyness_sell, "Description": "In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).", "Ideal for Sellers": "Depends on strategy"},
                        {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Sellers": "Shorter term (to maximize time decay)"},
                        {"Metric": f"Value ({ticker})", "Value": f"${rec_option_sell.get('lastPrice', None):.2f}" if rec_option_sell.get('lastPrice') is not None and not pd.isna(rec_option_sell.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Sellers": "Higher to enter"},
                        {"Metric": "Bid", "Value": f"${rec_option_sell.get('bid', None):.2f}" if rec_option_sell.get('bid') is not None and not pd.isna(rec_option_sell.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Sellers": "Higher to enter"},
                        {"Metric": "Ask", "Value": f"${rec_option_sell.get('ask', None):.2f}" if rec_option_sell.get('ask') is not None and not pd.isna(rec_option_sell.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Sellers": "Higher to enter"},
                        {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_sell.get('impliedVolatility', None):.2%}" if rec_option_sell.get('impliedVolatility') is not None and not pd.isna(rec_option_sell.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Sellers": "Higher is better"},
                        {"Metric": "Delta", "Value": f"{rec_option_sell.get('delta', None):.2f}" if rec_option_sell.get('delta') is not None and not pd.isna(rec_option_sell.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Sellers": "Lower (0.20-0.40) for defined risk spreads"},
                        {"Metric": "Theta", "Value": f"{rec_option_sell.get('theta', None):.3f}" if rec_option_sell.get('theta') is not None and not pd.isna(rec_option_sell.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Sellers": "Higher (more decay)"},
                        {"Metric": "Gamma", "Value": f"{rec_option_sell.get('gamma', None):.3f}" if rec_option_sell.get('gamma') is not None and not pd.isna(rec_option_sell.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Sellers": "Lower for stability"},
                        {"Metric": "Vega", "Value": f"{rec_option_sell.get('vega', None):.3f}" if rec_option_sell.get('vega') is not None and not pd.isna(rec_option_sell.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Sellers": "Less significant for short-term"},
                        {"Metric": "Rho", "Value": f"{rec_option_sell.get('rho', None):.3f}" if rec_option_sell.get('rho') is not None and not pd.isna(rec_option_sell.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Sellers": "Less significant for short-term"},
                        {"Metric": "Open Interest", "Value": f"{rec_option_sell.get('openInterest', None):,}" if rec_option_sell.get('openInterest') is not None and not pd.isna(rec_option_sell.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Sellers": "> 100s"},
                        {"Metric": "Volume (Today)", "Value": f"{rec_option_sell.get('volume', None):,}" if rec_option_sell.get('volume') is not None and not pd.isna(rec_option_sell.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Sellers": "Higher (>100)"},
                    ]
                    st.table(pd.DataFrame(option_metrics_sell).set_index("Metric"))
                else:
                    st.warning("Sell leg details not available for the Bull Call Spread.")

            elif trade_plan['Strategy'] in ["Buy ITM Call", "Buy ATM Call", "Buy ITM Put", "Buy ATM Put"]: # Added put strategies
                rec_option = trade_plan['Contract']
                entry_premium = rec_option.get('ask', rec_option.get('lastPrice', 0))
                option_type_for_moneyness = "call" if trade_plan['Strategy'] in ["Buy ITM Call", "Buy ATM Call"] else "put"
                moneyness = get_moneyness(rec_option.get('strike'), current_stock_price, option_type_for_moneyness)
                
                option_metrics = [
                    {"Metric": "Strike", "Value": f"${rec_option.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Buyers": "Lower for calls, higher for puts"},
                    {"Metric": "Moneyness", "Value": moneyness, "Description": "In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).", "Ideal for Buyers": "Depends on strategy"},
                    {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Buyers": "Longer term (45-365 days)"},
                    {"Metric": "Entry Premium", "Value": f"${entry_premium:.2f}" if entry_premium is not None and not pd.isna(entry_premium) else "N/A", "Description": "The cost paid per share for the option.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "**Max Loss (per share)**", "Value": f"${entry_premium:.2f}" if entry_premium is not None and not pd.isna(entry_premium) else "N/A", "Description": "The maximum theoretical loss is the premium paid for the option.", "Ideal for Buyers": "Lower"},
                    {"Metric": "Profit Target (per share)", "Value": f"${trade_plan.get('Profit Target', 'N/A')}" if trade_plan.get('Profit Target') is not None else "N/A", "Description": "The target price for the option to reach for profit.", "Ideal for Buyers": "Higher"},
                    {"Metric": "Bid", "Value": f"${rec_option.get('bid', None):.2f}" if rec_option.get('bid') is not None and not pd.isna(rec_option.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Ask", "Value": f"${rec_option.get('ask', None):.2f}" if rec_option.get('ask') is not None and not pd.isna(rec_option.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Buyers": "Lower to enter"},
                    {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option.get('impliedVolatility', None):.2%}" if rec_option.get('impliedVolatility') is not None and not pd.isna(rec_option.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Buyers": "Lower is better"},
                    {"Metric": "Delta", "Value": f"{rec_option.get('delta', None):.2f}" if rec_option.get('delta') is not None and not pd.isna(rec_option.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Buyers": "0.60 to 0.80 (for ITM calls)"},
                    {"Metric": "Theta", "Value": f"{rec_option.get('theta', None):.3f}" if rec_option.get('theta') is not None and not pd.isna(rec_option.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Buyers": "As low as possible"},
                    {"Metric": "Gamma", "Value": f"{rec_option.get('gamma', None):.3f}" if rec_option.get('gamma') is not None and not pd.isna(rec_option.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Buyers": "Higher for directional plays"},
                    {"Metric": "Vega", "Value": f"{rec_option.get('vega', None):.3f}" if rec_option.get('vega') is not None and not pd.isna(rec_option.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                    {"Metric": "Rho", "Value": f"{rec_option.get('rho', None):.3f}" if rec_option.get('rho') is not None and not pd.isna(rec_option.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                    {"Metric": "Open Interest", "Value": f"{rec_option.get('openInterest', None):,}" if rec_option.get('openInterest') is not None and not pd.isna(rec_option.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Buyers": "> 100s"},
                    {"Metric": "Volume (Today)", "Value": f"{rec_option.get('volume', None):,}" if rec_option.get('volume') is not None and not pd.isna(rec_option.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Buyers": "Higher (>100)"},
                ]
                st.table(pd.DataFrame(option_metrics).set_index("Metric"))

            elif trade_plan['Strategy'] == "Bear Put Spread": # New: Bear Put Spread
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Buy Strike", trade_plan['Buy Strike'])
                col2.metric("Sell Strike", trade_plan['Sell Strike'])
                col3.metric("Expiration", trade_plan['Expiration'])
                col4.metric("Net Debit", trade_plan['Net Debit']) # Bear Put Spread is also a debit spread
                col5.metric("Max Profit / Max Risk", f"{trade_plan['Max Profit']} / {trade_plan['Max Risk']}")
                st.write(f"**Reward / Risk:** `{trade_plan['Reward / Risk']}`")
                st.markdown("---")
                st.subheader("🔬 Recommended Option Deep-Dive (Spread Legs)")
                
                if 'Buy' in trade_plan['Contracts']:
                    st.write("**Buy Leg:**")
                    rec_option_buy = trade_plan['Contracts']['Buy']
                    moneyness_buy = get_moneyness(rec_option_buy.get('strike'), current_stock_price, "put")

                    option_metrics_buy = [
                        {"Metric": "Strike", "Value": f"${rec_option_buy.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Buyers": "Lower for calls, higher for puts"},
                        {"Metric": "Moneyness", "Value": moneyness_buy, "Description": "In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).", "Ideal for Buyers": "Depends on strategy"},
                        {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Buyers": "Longer term (45-365 days)"},
                        {"Metric": f"Value ({ticker})", "Value": f"${rec_option_buy.get('lastPrice', None):.2f}" if rec_option_buy.get('lastPrice') is not None and not pd.isna(rec_option_buy.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Bid", "Value": f"${rec_option_buy.get('bid', None):.2f}" if rec_option_buy.get('bid') is not None and not pd.isna(rec_option_buy.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Ask", "Value": f"${rec_option_buy.get('ask', None):.2f}" if rec_option_buy.get('ask') is not None and not pd.isna(rec_option_buy.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_buy.get('impliedVolatility', None):.2%}" if rec_option_buy.get('impliedVolatility') is not None and not pd.isna(rec_option_buy.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Buyers": "Lower is better"},
                        {"Metric": "Delta", "Value": f"{rec_option_buy.get('delta', None):.2f}" if rec_option_buy.get('delta') is not None and not pd.isna(rec_option_buy.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Buyers": "0.60 to 0.80 (for ITM calls)"},
                        {"Metric": "Theta", "Value": f"{rec_option_buy.get('theta', None):.3f}" if rec_option_buy.get('theta') is not None and not pd.isna(rec_option_buy.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Buyers": "As low as possible"},
                        {"Metric": "Gamma", "Value": f"{rec_option_buy.get('gamma', None):.3f}" if rec_option_buy.get('gamma') is not None and not pd.isna(rec_option_buy.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Buyers": "Higher for directional plays"},
                        {"Metric": "Vega", "Value": f"{rec_option_buy.get('vega', None):.3f}" if rec_option_buy.get('vega') is not None and not pd.isna(rec_option_buy.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                        {"Metric": "Rho", "Value": f"{rec_option_buy.get('rho', None):.3f}" if rec_option_buy.get('rho') is not None and not pd.isna(rec_option_buy.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Buyers": "Less significant for short-term"},
                        {"Metric": "Open Interest", "Value": f"{rec_option_buy.get('openInterest', None):,}" if rec_option_buy.get('openInterest') is not None and not pd.isna(rec_option_buy.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Buyers": "> 100s"},
                        {"Metric": "Volume (Today)", "Value": f"{rec_option_buy.get('volume', None):,}" if rec_option_buy.get('volume') is not None and not pd.isna(rec_option_buy.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Buyers": "Higher (>100)"},
                    ]
                    st.table(pd.DataFrame(option_metrics_buy).set_index("Metric"))
                else:
                    st.warning("Buy leg details not available for the Bear Put Spread.")

                if 'Sell' in trade_plan['Contracts']:
                    st.write("**Sell Leg:**")
                    rec_option_sell = trade_plan['Contracts']['Sell']
                    moneyness_sell = get_moneyness(rec_option_sell.get('strike'), current_stock_price, "put")

                    option_metrics_sell = [
                        {"Metric": "Strike", "Value": f"${rec_option_sell.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Sellers": "Higher for calls, lower for puts"},
                        {"Metric": "Moneyness", "Value": moneyness_sell, "Description": "In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).", "Ideal for Sellers": "Depends on strategy"},
                        {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Sellers": "Shorter term (to maximize time decay)"},
                        {"Metric": f"Value ({ticker})", "Value": f"${rec_option_sell.get('lastPrice', None):.2f}" if rec_option_sell.get('lastPrice') is not None and not pd.isna(rec_option_sell.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Sellers": "Higher to enter"},
                        {"Metric": "Bid", "Value": f"${rec_option_sell.get('bid', None):.2f}" if rec_option_sell.get('bid') is not None and not pd.isna(rec_option_sell.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Sellers": "Higher to enter"},
                        {"Metric": "Ask", "Value": f"${rec_option_sell.get('ask', None):.2f}" if rec_option_sell.get('ask') is not None and not pd.isna(rec_option_sell.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Sellers": "Higher to enter"},
                        {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_sell.get('impliedVolatility', None):.2%}" if rec_option_sell.get('impliedVolatility') is not None and not pd.isna(rec_option_sell.get('impliedVolatility')) else "N/A", "Description": "Market's forecast of volatility. High IV = expensive premium.", "Ideal for Sellers": "Higher is better"},
                        {"Metric": "Delta", "Value": f"{rec_option_sell.get('delta', None):.2f}" if rec_option_sell.get('delta') is not None and not pd.isna(rec_option_sell.get('delta')) else "N/A", "Description": "Option's price change per $1 stock change.", "Ideal for Sellers": "Lower (0.20-0.40) for defined risk spreads"},
                        {"Metric": "Theta", "Value": f"{rec_option_sell.get('theta', None):.3f}" if rec_option_sell.get('theta') is not None and not pd.isna(rec_option_sell.get('theta')) else "N/A", "Description": "Time decay. Daily value lost from the premium.", "Ideal for Sellers": "Higher (more decay)"},
                        {"Metric": "Gamma", "Value": f"{rec_option_sell.get('gamma', None):.3f}" if rec_option_sell.get('gamma') is not None and not pd.isna(rec_option_sell.get('gamma')) else "N/A", "Description": "Rate of change of Delta. High Gamma = faster delta changes.", "Ideal for Sellers": "Lower for stability"},
                        {"Metric": "Vega", "Value": f"{rec_option_sell.get('vega', None):.3f}" if rec_option_sell.get('vega') is not None and not pd.isna(rec_option_sell.get('vega')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Sellers": "Less significant for short-term"},
                        {"Metric": "Rho", "Value": f"{rec_option_sell.get('rho', None):.3f}" if rec_option_sell.get('rho') is not None and not pd.isna(rec_option_sell.get('rho')) else "N/A", "Description": "Option's price change per 1% change in interest rates.", "Ideal for Sellers": "Less significant for short-term"},
                        {"Metric": "Open Interest", "Value": f"{rec_option_sell.get('openInterest', None):,}" if rec_option_sell.get('openInterest') is not None and not pd.isna(rec_option_sell.get('openInterest')) else "N/A", "Description": "Total open contracts. High OI indicates good liquidity.", "Ideal for Sellers": "> 100s"},
                        {"Metric": "Volume (Today)", "Value": f"{rec_option_sell.get('volume', None):,}" if rec_option_sell.get('volume') is not None and not pd.isna(rec_option_sell.get('volume')) else "N/A", "Description": "Number of contracts traded today.", "Ideal for Sellers": "Higher (>100)"},
                    ]
                    st.table(pd.DataFrame(option_metrics_sell).set_index("Metric"))
                else:
                    st.warning("Sell leg details not available for the Bear Put Spread.")

            else: # If status is success but not a recognized strategy, something is off
                st.error("An options strategy was recommended, but its details could not be fully displayed.")
        
        else: # This 'else' correctly belongs to the 'if trade_plan['status'] == 'success':
            st.warning(trade_plan['message'])
        
        st.markdown("---")
        st.subheader("⛓️ Full Option Chain")
        option_type = st.radio("Select Option Type", ["Calls", "Puts"], horizontal=True, key=f"option_type_{ticker}")
        exp_date_str = st.selectbox("Select Expiration Date to View", expirations, key=f"exp_date_select_{ticker}")
        if exp_date_str:
            calls, puts = get_options_chain(ticker, exp_date_str)

            st.markdown("##### Options Chain Highlights & Suggestions")
            chain_analysis_results = analyze_options_chain(calls, puts, current_stock_price, exp_date_str) 
            
            if chain_analysis_results:
                has_content = False
                for category, options_list in chain_analysis_results.items():
                    if options_list:
                        has_content = True
                        st.markdown(f"**{category}:**")
                        for opt_summary in options_list:
                            st.markdown(f"- **{opt_summary['Type']}** (Strike: ${opt_summary['Strike']:.2f}, Exp: {opt_summary['Expiration']}): {opt_summary['Reason']}")
                if not has_content:
                    st.info("No specific options chain highlights found for this expiration based on current criteria.")
            else:
                st.info("No detailed options chain analysis available for this expiration.")
            st.markdown("---")

            st.markdown(f"[**🔗 Analyze this chain on OptionCharts.io**](https://optioncharts.io/options/{ticker}/chain/{exp_date_str})")
            
            chain_to_display = calls if option_type == "Calls" else puts
            chain_to_display_copy = chain_to_display.copy()
            if 'strike' in chain_to_display_copy.columns:
                chain_to_display_copy['Moneyness'] = chain_to_display_copy.apply(
                    lambda row: get_moneyness(row['strike'], current_stock_price, "call" if option_type == "Calls" else "put"), axis=1
                )
            
            # Define column descriptions for tooltips
            column_descriptions = {
                'strike': 'The predetermined price at which the underlying asset can be bought (for a call) or sold (for a put) when the option is exercised.',
                'Moneyness': 'Describes an option\'s relationship between its strike price and the underlying asset\'s current price (In-The-Money, At-The-Money, or Out-of-The-Money).',
                'lastPrice': 'The last traded price of that specific option contract.',
                'bid': 'The highest price a buyer is willing to pay for the option.',
                'ask': 'The lowest price a seller is willing to accept for the option.',
                'volume': 'The number of option contracts traded for that specific strike and expiration today. High volume indicates high trading activity and liquidity.',
                'openInterest': 'The total number of outstanding option contracts that have not yet been closed or exercised. High open interest suggests strong market interest and good liquidity for that particular contract.',
                'impliedVolatility': 'The market\'s expectation of how much the underlying stock\'s price will move in the future. Higher implied volatility generally means higher option premiums (prices), as there\'s a greater chance of the option moving in-the-money.',
                'delta': 'Measures how much an option\'s price is expected to move for every $1 change in the underlying stock\'s price. Also represents the approximate probability of an option expiring in-the-money.',
                'theta': 'Measures the rate at which an option\'s price decays over time (time decay). Theta is typically negative, meaning the option loses value as it gets closer to expiration, all else being equal.',
                'gamma': 'Measures the rate of change of Delta. High gamma = faster delta changes.',
                'vega': 'Measures how much an option\'s price is expected to change for every 1% change in implied volatility. Options with higher vega are more sensitive to changes in IV.',
                'rho': 'Measures how much an option\'s price is expected to change for every 1% change in interest rates. This is typically less significant for short-term options.'
            }

            # Define the desired order of columns (moved this definition up)
            desired_cols_to_display = ['strike', 'Moneyness', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'delta', 'theta', 'gamma', 'vega', 'rho']

            # Prepare columns for display with tooltips
            cols_to_display_with_tooltips = {}
            for col in desired_cols_to_display:
                if col in chain_to_display_copy.columns:
                    cols_to_display_with_tooltips[col] = st.column_config.Column(
                        col,
                        help=column_descriptions.get(col, "No description available.")
                    )

            # Filter chain_to_display_copy to only include available and desired columns
            final_cols_to_show = [col for col in desired_cols_to_display if col in chain_to_display_copy.columns]

            if final_cols_to_show:
                st.dataframe(
                    chain_to_display_copy[final_cols_to_show].set_index('strike'),
                    column_config=cols_to_display_with_tooltips
                )
            else:
                st.info("No relevant columns found in the options chain to display.")
    
    st.markdown("---")
    st.info("Note: This calculator assumes expiration and does not account for time decay or implied volatility changes before expiration.")


def display_backtest_tab(ticker, selection, current_price, prev_close, overall_confidence, backtest_direction): # Added backtest_direction
    """Displays the historical backtest results."""
    _display_common_header(ticker, current_price, prev_close, overall_confidence, backtest_direction) # Pass backtest_direction as sentiment_status for header

    st.subheader(f"🧪 Historical Backtest for {ticker}")
    st.info(f"Simulating **{backtest_direction.capitalize()}** trades based on your **currently selected indicators**. Entry is triggered if a sufficient percentage of selected signals are positive for the chosen direction.")
    
    daily_hist, _ = get_data(ticker, "2y", "1d")
    if daily_hist is not None and not daily_hist.empty:
        daily_df_calculated = calculate_indicators(daily_hist.copy(), is_intraday=False)

        # Pass the selected backtest_direction to backtest_strategy
        trades, wins, losses = backtest_strategy(daily_df_calculated, selection, trade_direction=backtest_direction)
        
        total_trades = wins + losses
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Trades Simulated", total_trades)
        col2.metric("Wins", wins)
        col3.metric("Win Rate", f"{win_rate:.1f}%")
        
        if trades: st.dataframe(pd.DataFrame(trades).tail(20))
        else: st.info("No trades were executed based on the current strategy and historical data. Please review the warnings above for potential reasons (e.g., insufficient data, indicator calculation issues, or strict entry conditions).")
    else:
        st.warning("Could not fetch daily data for backtesting or data is empty. Ensure the ticker is valid and enough historical data is available for the selected period (e.g., 2 years).")

def display_news_info_tab(ticker, info, finviz_data, current_price, prev_close, overall_confidence, trade_direction): # Added trade_direction
    """Displays news and company information."""
    _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction) # Pass trade_direction
    st.subheader(f"📰 News & Information for {ticker}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ℹ️ Company Info")
        st.write(f"**Name:** {info.get('longName', ticker)}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown("#### 🔗 External Research Links")
        st.markdown(f"- [Yahoo Finance]({info.get('website', 'https://finance.yahoo.com')}) | [Finviz](https://finviz.com/quote.ashx?t={ticker})")
    with col2:
        st.markdown("#### 📅 Company Calendar")
        stock_obj_for_cal = yf.Ticker(ticker)
        if stock_obj_for_cal and hasattr(stock_obj_for_cal, 'calendar') and isinstance(stock_obj_for_cal.calendar, pd.DataFrame) and not stock_obj_for_cal.empty:
            st.dataframe(stock_obj_for_cal.calendar.T)
        else:
            st.info("No upcoming calendar events found.")
    st.markdown("#### 🗞️ Latest Headlines")
    if finviz_data and finviz_data.get('headlines'):
        for h in finviz_data['headlines']:
            st.markdown(f"_{h}_")
    else:
        st.info("No recent headlines found or automated scoring is disabled.")

def display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence, current_price, prev_close, trade_direction): # Added trade_direction
    """Displays and manages the trade log."""
    _display_common_header(ticker, current_price, prev_close, overall_confidence, trade_direction) # Pass trade_direction
    st.subheader("📝 Log Your Trade Analysis")
    user_notes = st.text_area("Add your personal notes or trade thesis here:", key=f"trade_notes_{ticker}")
    
    st.info("Trade log functionality is pending implementation.")

