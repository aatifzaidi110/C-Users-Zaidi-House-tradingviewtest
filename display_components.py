# display_components.py - 4.0

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
    generate_directional_trade_plan, get_indicator_summary_text # NEW: Import get_indicator_summary_text
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
    "Buy": 75,
    "Hold": 50,
    "Sell": 25,
    "Strong Sell": 0
}

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
                if expert_text == value: # Corrected from expert_text == value to expert_score == value
                    expert_text = key
                    break
            if expert_text == "N/A (Excluded)" and expert_score == 50: # Default for Hold if not explicitly mapped
                expert_text = "Hold"


        st.markdown(f"- **Technical Score:** `{scores['Technical']:.0f}` (Weight: `{final_weights['technical']*100:.0f}%`)\n" # Changed key from 'technical' to 'Technical'
                    f"- **Sentiment Score:** {sentiment_text} (Weight: `{final_weights['sentiment']*100:.0f}%`)\n"
                    f"- **Expert Rating:** {expert_text} (Weight: `{final_weights['expert']*100:.0f}%`)\n"
                    f"- **Economic Score:** `{scores['Economic']:.0f}` (Weight: `{final_weights['economic']*100:.0f}%`)\n" # NEW
                    f"- **Investor Sentiment Score:** `{scores['Investor Sentiment']:.0f}` (Weight: `{final_weights['investor_sentiment']*100:.0f}%`)") # NEW
        
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

def display_trade_plan_options_tab(ticker, df, overall_confidence, timeframe, trade_direction):
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

    # Prepare the confidence_score dictionary as expected by generate_directional_trade_plan
    # 'overall_confidence' is a scalar (e.g., 85.0), and 'trade_direction' is a string (e.g., "Bullish")
    confidence_for_plan = {
        'score': overall_confidence,
        'band': trade_direction # Using trade_direction as the 'band' for the trade plan
    }

    # Generate the trade plan using the new directional function
    # Pass the correctly structured confidence_for_plan dictionary as the first argument
    trade_plan_result = generate_directional_trade_plan(
        confidence_for_plan,    # First argument: the confidence_score dictionary
        current_stock_price,    # Second argument: current_price
        last,                   # Third argument: latest_row (the full 'last' Series)
        period_interval         # Fourth argument: period_interval
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
        # Pass overall_confidence as the second argument (confidence_score_value)
        trade_plan = suggest_options_strategy(ticker, overall_confidence, current_stock_price, expirations, trade_direction)
        
        # --- Start of new detailed options display ---
        if trade_plan['status'] == 'success':
            st.success(f"**Recommended Strategy: {trade_plan['Strategy']}** (Confidence: {overall_confidence:.0f}%)")
            st.info(trade_plan['message']) # Use 'message' from the plan
            
            if trade_plan['Strategy'] == "Bull Call Spread" or trade_plan['Strategy'] == "Bear Put Spread": # Handle both spreads
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
                    moneyness_buy = get_moneyness(current_stock_price, rec_option_buy.get('strike'), "call" if trade_plan['Strategy'] == "Bull Call Spread" else "put")

                    option_metrics_buy = [
                        {"Metric": "Strike", "Value": f"${rec_option_buy.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Buyers": "Lower for calls, higher for puts"},
                        {"Metric": "Moneyness", "Value": moneyness_buy, "Description": "In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).", "Ideal for Buyers": "Depends on strategy"},
                        {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Buyers": "Longer term (45-365 days)"},
                        {"Metric": f"Value ({ticker})", "Value": f"${rec_option_buy.get('lastPrice', None):.2f}" if rec_option_buy.get('lastPrice') is not None and not pd.isna(rec_option_buy.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Bid", "Value": f"${rec_option_buy.get('bid', None):.2f}" if rec_option_buy.get('bid') is not None and not pd.isna(rec_option_buy.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Ask", "Value": f"${rec_option_buy.get('ask', None):.2f}" if rec_option_buy.get('ask') is not None and not pd.isna(rec_option_buy.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Buyers": "Lower to enter"},
                        {"Metric": "Volume", "Value": f"{rec_option_buy.get('volume', 0)}", "Description": "Number of contracts traded today.", "Ideal for Buyers": "Higher (indicates liquidity)"},
                        {"Metric": "Open Interest", "Value": f"{rec_option_buy.get('openInterest', 0)}", "Description": "Total outstanding contracts.", "Ideal for Buyers": "Higher (indicates liquidity)"},
                        {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_buy.get('impliedVolatility', 0)*100:.2f}%", "Description": "Market's expectation of future price swings.", "Ideal for Buyers": "Lower (cheaper options)"},
                        {"Metric": "Delta", "Value": f"{rec_option_buy.get('delta', 0):.2f}" if rec_option_buy.get('delta') is not None and not pd.isna(rec_option_buy.get('delta')) else "N/A", "Description": "Sensitivity to price changes.", "Ideal for Buyers": "Higher for ITM, lower for OTM"},
                        {"Metric": "Theta", "Value": f"{rec_option_buy.get('theta', 0):.4f}" if rec_option_buy.get('theta') is not None and not pd.isna(rec_option_buy.get('theta')) else "N/A", "Description": "Time decay.", "Ideal for Buyers": "Less negative (slower decay)"},
                        {"Metric": "Gamma", "Value": f"{rec_option_buy.get('gamma', 0):.4f}" if rec_option_buy.get('gamma') is not None and not pd.isna(rec_option_buy.get('gamma')) else "N/A", "Description": "Rate of change of Delta.", "Ideal for Buyers": "Higher (for speculative moves)"},
                    ]
                    st.dataframe(pd.DataFrame(option_metrics_buy).set_index("Metric"))

                if 'Sell' in trade_plan['Contracts']:
                    st.write("**Sell Leg:**")
                    rec_option_sell = trade_plan['Contracts']['Sell']
                    moneyness_sell = get_moneyness(current_stock_price, rec_option_sell.get('strike'), "call" if trade_plan['Strategy'] == "Bull Call Spread" else "put")

                    option_metrics_sell = [
                        {"Metric": "Strike", "Value": f"${rec_option_sell.get('strike', 0):.2f}", "Description": "The price at which the option can be exercised.", "Ideal for Sellers": "Higher for calls, lower for puts"},
                        {"Metric": "Moneyness", "Value": moneyness_sell, "Description": "In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).", "Ideal for Sellers": "Depends on strategy"},
                        {"Metric": "Expiration", "Value": trade_plan['Expiration'], "Description": "Date the option expires.", "Ideal for Sellers": "Shorter term (to maximize time decay)"},
                        {"Metric": f"Value ({ticker})", "Value": f"${rec_option_sell.get('lastPrice', None):.2f}" if rec_option_sell.get('lastPrice') is not None and not pd.isna(rec_option_sell.get('lastPrice')) else "N/A", "Description": "The last traded price of the option.", "Ideal for Sellers": "Higher to receive more premium"},
                        {"Metric": "Bid", "Value": f"${rec_option_sell.get('bid', None):.2f}" if rec_option_sell.get('bid') is not None and not pd.isna(rec_option_sell.get('bid')) else "N/A", "Description": "Highest price a buyer is willing to pay.", "Ideal for Sellers": "Higher to receive more premium"},
                        {"Metric": "Ask", "Value": f"${rec_option_sell.get('ask', None):.2f}" if rec_option_sell.get('ask') is not None and not pd.isna(rec_option_sell.get('ask')) else "N/A", "Description": "Lowest price a seller is willing to accept.", "Ideal for Sellers": "Higher to receive more premium"},
                        {"Metric": "Volume", "Value": f"{rec_option_sell.get('volume', 0)}", "Description": "Number of contracts traded today.", "Ideal for Sellers": "Higher (indicates liquidity)"},
                        {"Metric": "Open Interest", "Value": f"{rec_option_sell.get('openInterest', 0)}", "Description": "Total outstanding contracts.", "Ideal for Sellers": "Higher (indicates liquidity)"},
                        {"Metric": "Implied Volatility (IV)", "Value": f"{rec_option_sell.get('impliedVolatility', 0)*100:.2f}%", "Description": "Market's expectation of future price swings.", "Ideal for Sellers": "Higher (to receive more premium)"},
                        {"Metric": "Delta", "Value": f"{rec_option_sell.get('delta', 0):.2f}" if rec_option_sell.get('delta') is not None and not pd.isna(rec_option_sell.get('delta')) else "N/A", "Description": "Sensitivity to price changes.", "Ideal for Sellers": "Lower for ITM, higher for OTM"},
                        {"Metric": "Theta", "Value": f"{rec_option_sell.get('theta', 0):.4f}" if rec_option_sell.get('theta') is not None and not pd.isna(rec_option_sell.get('theta')) else "N/A", "Description": "Time decay.", "Ideal for Sellers": "More negative (faster decay)"},
                        {"Metric": "Gamma", "Value": f"{rec_option_sell.get('gamma', 0):.4f}" if rec_option_sell.get('gamma') is not None and not pd.isna(rec_option_sell.get('gamma')) else "N/A", "Description": "Rate of change of Delta.", "Ideal for Sellers": "Lower (less sensitive to price changes)"},
                    ]
                    st.dataframe(pd.DataFrame(option_metrics_sell).set_index("Metric"))

                # Plot payoff for the recommended strategy
                if 'option_legs_for_chart' in trade_plan:
                    st.markdown("---")
                    st.subheader(f"📈 {trade_plan['Strategy']} Payoff Chart")
                    chart_legs = trade_plan['option_legs_for_chart']
                    
                    # Determine chart range
                    min_strike_chart = min(leg['strike'] for leg in chart_legs) * 0.9
                    max_strike_chart = max(leg['strike'] for leg in chart_legs) * 1.1
                    
                    # Extend range for potential unlimited profit/loss based on the strategy type
                    # For bull call spread, it's typically capped, so the 1.1 multiplier is usually sufficient.
                    stock_prices_chart = np.linspace(min_strike_chart, max_strike_chart, 200)
                    payoffs_chart = calculate_payoff_from_legs(stock_prices_chart, chart_legs) * 100 # Multiply by 100 for contracts

                    payoff_fig_strategy = plot_generic_payoff_chart(stock_prices_chart, payoffs_chart, chart_legs, trade_plan['Strategy'], ticker, current_stock_price)
                    if payoff_fig_strategy:
                        st.pyplot(payoff_fig_strategy, clear_figure=True)
                        plt.close(payoff_fig_strategy)
            else:
                st.warning(f"No specific detailed display logic for '{trade_plan['Strategy']}' yet. Showing basic info.")
                st.json(trade_plan) # Display raw trade plan for unhandled strategies

        else:
            st.warning(trade_plan['message']) # Display error message from suggest_options_strategy

def display_backtest_tab(ticker, indicator_selection, current_price, prev_close, overall_confidence, backtest_direction):
    """
    Displays the backtesting tab, allowing users to run backtests on historical data.
    """
    _display_common_header(ticker, current_price, prev_close, overall_confidence, backtest_direction.capitalize()) # Pass backtest_direction to header
    st.subheader(f"🧪 Backtesting Results for {ticker}")
    st.info("Run a backtest on historical data to evaluate strategy performance.")

    st.markdown("---")
    st.markdown("#### Backtest Parameters")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"), key=f"backtest_start_date_{ticker}")
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("today"), key=f"backtest_end_date_{ticker}")

    # Backtest interval should ideally be tied to the selected trading style's interval
    backtest_interval_map = {
        "Long (Bullish)": "1d",
        "Short (Bearish)": "1d" # Assuming daily for now, can be made configurable
    }
    selected_backtest_interval = backtest_interval_map.get(f"{backtest_direction.capitalize()} ({backtest_direction.capitalize()})", "1d")

    if st.button(f"Run {backtest_direction.capitalize()} Backtest", key=f"run_backtest_{ticker}_{backtest_direction}"):
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            with st.spinner(f"Running {backtest_direction.lower()} backtest... This may take a moment."):
                # Fetch data for the selected period for backtesting
                try:
                    hist_data_backtest, _ = get_data(ticker, f"{end_date - start_date}", selected_backtest_interval, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                    if hist_data_backtest.empty:
                        st.warning("Could not fetch historical data for the selected period. Please adjust dates or ticker.")
                    else:
                        trades, performance = backtest_strategy(
                            hist_data_backtest.copy(), 
                            indicator_selection, 
                            trade_direction=backtest_direction # Pass the direction to the backtest function
                        )

                        if trades and performance:
                            st.subheader("Backtest Performance Summary")
                            col_p1, col_p2, col_p3 = st.columns(3)
                            col_p1.metric("Total Trades", performance.get('Total Trades', 0))
                            col_p2.metric("Winning Trades", performance.get('Winning Trades', 0))
                            col_p3.metric("Losing Trades", performance.get('Losing Trades', 0))

                            col_p4, col_p5, col_p6 = st.columns(3)
                            col_p4.metric("Win Rate", performance.get('Win Rate', "0.00%"))
                            col_p5.metric("Gross Profit", f"${performance.get('Gross Profit', 0):.2f}")
                            col_p6.metric("Gross Loss", f"${performance.get('Gross Loss', 0):.2f}")
                            
                            col_p7, col_p8 = st.columns(2)
                            col_p7.metric("Profit Factor", performance.get('Profit Factor', "0.00"))
                            col_p8.metric("Net PnL", f"${performance.get('Net PnL', 0):.2f}")


                            if trades:
                                st.subheader("Detailed Trade Log")
                                # Convert list of dicts to DataFrame for display
                                trade_log_df = pd.DataFrame(trades)
                                st.dataframe(trade_log_df)
                            else:
                                st.info("No trades were executed during the backtest period with the selected indicators.")
                        else:
                            st.info("No backtest results returned. Ensure your strategy parameters are valid or enough data is available.")
                except Exception as e:
                    st.error(f"Error during backtest: {e}")
                    st.exception(e)


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
        entry_details_expander = f"**Entry Criteria Details for {row['Ticker']}**\n\n{row['Entry Criteria Details']}"
        exit_details_expander = f"**Exit Criteria Details for {row['Ticker']}**\n\n{row['Exit Criteria Details']}"

        display_data.append({
            "Ticker": row['Ticker'],
            "Style": row['Trading Style'],
            "Confidence": row['Overall Confidence'],
            "Direction": row['Direction'],
            "Price": row['Current Price'],
            "ATR": row['ATR'],
            "Target": row['Target Price'],
            "Stop Loss": row['Stop Loss'],
            "Entry Zone": row['Entry Zone'],
            "R/R": row['Reward/Risk'],
            "Pivot (P)": row['Pivot (P)'],
            "R1": row['Resistance 1 (R1)'],
            "S1": row['Support 1 (S1)'],
            "R2": row['Resistance 2 (R2)'],
            "S2": row['Support 2 (S2)'],
            "Entry Details": entry_details_expander, # This will be expanded
            "Exit Details": exit_details_expander,   # This will be expanded
            "Rationale": row['Rationale']
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


def display_backtest_tab(ticker, indicator_selection, current_price, prev_close, overall_confidence, backtest_direction):
    """
    Displays the backtesting tab, allowing users to run backtests on historical data.
    """
    _display_common_header(ticker, current_price, prev_close, overall_confidence, backtest_direction.capitalize()) # Pass backtest_direction to header
    st.subheader(f"🧪 Backtesting Results for {ticker}")
    st.info("Run a backtest on historical data to evaluate strategy performance.")

    st.markdown("---")
    st.markdown("#### Backtest Parameters")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"), key=f"backtest_start_date_{ticker}")
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("today"), key=f"backtest_end_date_{ticker}")

    # Backtest interval should ideally be tied to the selected trading style's interval
    backtest_interval_map = {
        "Long (Bullish)": "1d",
        "Short (Bearish)": "1d" # Assuming daily for now, can be made configurable
    }
    selected_backtest_interval = backtest_interval_map.get(f"{backtest_direction.capitalize()} ({backtest_direction.capitalize()})", "1d")

    if st.button(f"Run {backtest_direction.capitalize()} Backtest", key=f"run_backtest_{ticker}_{backtest_direction}"):
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            with st.spinner(f"Running {backtest_direction.lower()} backtest... This may take a moment."):
                # Fetch data for the selected period for backtesting
                try:
                    hist_data_backtest, _ = get_data(ticker, f"{end_date - start_date}", selected_backtest_interval, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                    if hist_data_backtest.empty:
                        st.warning("Could not fetch historical data for the selected period. Please adjust dates or ticker.")
                    else:
                        trades, performance = backtest_strategy(
                            hist_data_backtest.copy(), 
                            indicator_selection, 
                            trade_direction=backtest_direction # Pass the direction to the backtest function
                        )

                        if trades and performance:
                            st.subheader("Backtest Performance Summary")
                            col_p1, col_p2, col_p3 = st.columns(3)
                            col_p1.metric("Total Trades", performance.get('Total Trades', 0))
                            col_p2.metric("Winning Trades", performance.get('Winning Trades', 0))
                            col_p3.metric("Losing Trades", performance.get('Losing Trades', 0))

                            col_p4, col_p5, col_p6 = st.columns(3)
                            col_p4.metric("Win Rate", performance.get('Win Rate', "0.00%"))
                            col_p5.metric("Gross Profit", f"${performance.get('Gross Profit', 0):.2f}")
                            col_p6.metric("Gross Loss", f"${performance.get('Gross Loss', 0):.2f}")
                            
                            col_p7, col_p8 = st.columns(2)
                            col_p7.metric("Profit Factor", performance.get('Profit Factor', "0.00"))
                            col_p8.metric("Net PnL", f"${performance.get('Net PnL', 0):.2f}")


                            if trades:
                                st.subheader("Detailed Trade Log")
                                # Convert list of dicts to DataFrame for display
                                trade_log_df = pd.DataFrame(trades)
                                st.dataframe(trade_log_df)
                            else:
                                st.info("No trades were executed during the backtest period with the selected indicators.")
                        else:
                            st.info("No backtest results returned. Ensure your strategy parameters are valid or enough data is available.")
                except Exception as e:
                    st.error(f"Error during backtest: {e}")
                    st.exception(e)


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

