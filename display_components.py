# display_components.py - Version 1.40

import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf # Ensure yfinance is imported here
import os # For chart saving/removing, though direct plot is preferred
import numpy as np # For numerical operations in payoff chart

# Import functions from utils.py
from utils import backtest_strategy, calculate_indicators, generate_signals_for_row, generate_option_trade_plan, get_options_chain, get_data, get_finviz_data, calculate_pivot_points, EXPERT_RATING_MAP, get_moneyness, analyze_options_chain # Import get_moneyness and analyze_options_chain
from glossary_components import INDICATOR_DESCRIPTIONS, OPTIONS_GREEKS_DESCRIPTIONS

# === Helper for Indicator Display ===
def format_indicator_display(signal_key, current_value, selected, signals_dict):
    """
    Formats and displays a single technical indicat or's concise information.
    """
    if not selected:
        return ""

    is_fired = signals_dict.get(signal_key, False)
    status_icon = '🟢' if is_fired else '🔴'
    display_name = signal_key.split('(')[0].strip()

    value_str = ""
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            value_str = f"Current: `{current_value:.2f}`"
        else:
            value_str = "Current: N/A"
    else:
        value_str = "Current: N/A"

    return f"{status_icon} **{display_name}** ({value_str})"


# === Option Payoff Chart Functions ===

def calculate_payoff_from_legs(stock_prices, legs):
    """
    Calculates the total payoff for a given set of option legs across a range of stock prices.
    Each leg is expected to be a dictionary: {'type': 'call'/'put', 'strike': float, 'premium': float, 'action': 'buy'/'sell'}
    """
    total_payoff = np.zeros_like(stock_prices, dtype=float) # Ensure float type

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
            # This case should ideally be prevented by input validation
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
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Breakeven Line') # Breakeven line

    ax.plot(stock_prices, payoffs, label=f'{strategy_name} Payoff', color='blue')

    # Mark strikes
    for leg in legs:
        color = 'green' if leg['action'] == 'buy' else 'red'
        linestyle = ':'
        ax.axvline(leg['strike'], color=color, linestyle=linestyle, label=f"{leg['action'].capitalize()} {leg['type'].capitalize()} Strike: ${leg['strike']:.2f}")

    # Mark current stock price
    ax.axvline(current_stock_price, color='orange', linestyle='-', linewidth=1.5, label=f'Current Price: ${current_stock_price:.2f}')

    # Find breakeven points (where payoff crosses zero)
    breakeven_points = []
    # Check for sign changes in payoff
    for i in range(1, len(payoffs)):
        if (payoffs[i-1] < 0 and payoffs[i] >= 0) or (payoffs[i-1] > 0 and payoffs[i] <= 0):
            # Linear interpolation for a more precise breakeven point
            x1, y1 = stock_prices[i-1], payoffs[i-1]
            x2, y2 = stock_prices[i], payoffs[i]
            if (y2 - y1) != 0: # Avoid division by zero
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakeven_points.append(breakeven)

    # Filter unique breakeven points and plot them
    unique_breakeven_points = sorted(list(set(round(bp, 2) for bp in breakeven_points)))
    for bp in unique_breakeven_points:
        ax.axvline(bp, color='purple', linestyle='--', label=f'Breakeven: ${bp:.2f}')

    # Calculate and mark max profit/loss from the payoff array
    max_payoff = np.max(payoffs)
    min_payoff = np.min(payoffs)

    # Add text annotations for max profit/loss
    # Adjust position slightly to avoid overlap with plot line
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

def plot_automated_strategy_payoff(strategy_details, current_stock_price, ticker):
    """
    Generates and displays an option payoff chart for the automated recommended strategy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Breakeven Line') # Breakeven line

    strategy_type = strategy_details['Strategy']

    # Determine the range of stock prices for the chart
    # Use current stock price and relevant strikes to set a good range
    min_price = current_stock_price * 0.8
    max_price = current_stock_price * 1.2

    if strategy_type == "Bull Call Spread":
        buy_strike = strategy_details['Contracts']['Buy']['strike']
        sell_strike = strategy_details['Contracts']['Sell']['strike']
        # Safely convert net_debit string to float
        net_debit_str = strategy_details['Net Debit'].replace('~', '').replace('$', '')
        net_debit = float(net_debit_str) if net_debit_str else 0.0

        min_price = min(buy_strike, sell_strike, current_stock_price) * 0.9
        max_price = max(buy_strike, sell_strike, current_stock_price) * 1.1

        stock_prices = np.linspace(min_price, max_price, 200)

        # Calculate payoff for Bull Call Spread
        payoff = np.maximum(0, stock_prices - buy_strike) - np.maximum(0, stock_prices - sell_strike) - net_debit

        ax.plot(stock_prices, payoff, label=f'{strategy_type} Payoff', color='blue')

        # Mark key points
        ax.axvline(buy_strike, color='green', linestyle=':', label=f'Buy Strike: ${buy_strike:.2f}')
        ax.axvline(sell_strike, color='red', linestyle=':', label=f'Sell Strike: ${sell_strike:.2f}')

        breakeven = buy_strike + net_debit
        ax.axvline(breakeven, color='purple', linestyle='--', label=f'Breakeven: ${breakeven:.2f}')

        max_profit = (sell_strike - buy_strike) - net_debit
        max_loss = -net_debit

        ax.text(stock_prices[-1], max_profit, f'Max Profit: ${max_profit:.2f}', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=9)
        ax.text(stock_prices[-1], max_loss, f'Max Loss: ${max_loss:.2f}', verticalalignment='top', horizontalalignment='right', color='red', fontsize=9)

    elif strategy_type in ["Buy ITM Call", "Buy ATM Call"]:
        strike = float(strategy_details['Strike'].replace('$', ''))
        # Safely convert entry_price string to float
        entry_premium_str = strategy_details['Entry Price'].replace('~', '').replace('$', '')
        entry_premium = float(entry_premium_str) if entry_premium_str else 0.0

        min_price = min(strike, current_stock_price) * 0.9
        max_price = max(strike, current_stock_price) * 1.1 + entry_premium * 2 # Extend range for unlimited profit

        stock_prices = np.linspace(min_price, max_price, 200)

        # Calculate payoff for Buy Call
        payoff = np.maximum(0, stock_prices - strike) - entry_premium

        ax.plot(stock_prices, payoff, label=f'{strategy_type} Payoff', color='blue')

        # Mark key points
        ax.axvline(strike, color='green', linestyle=':', label=f'Strike: ${strike:.2f}')

        breakeven = strike + entry_premium
        ax.axvline(breakeven, color='purple', linestyle='--', label=f'Breakeven: ${breakeven:.2f}')

        max_loss = -entry_premium

        ax.text(stock_prices[-1], payoff[-1], 'Max Profit: Unlimited', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=9)
        ax.text(stock_prices[-1], max_loss, f'Max Loss: ${max_loss:.2f}', verticalalignment='top', horizontalalignment='right', color='red', fontsize=9)

    else:
        ax.text(0.5, 0.5, "Payoff chart not available for this strategy.", transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        plt.close(fig) # Close the figure if no chart is drawn to prevent memory leaks
        return None # Return None if no chart is generated

    ax.set_title(f'{ticker} {strategy_type} Payoff Chart')
    ax.set_xlabel('Stock Price at Expiration ($)')
    ax.set_ylabel('Profit/Loss ($)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    return fig

def display_interactive_payoff_calculator(current_stock_price, ticker):
    """
    Displays an interactive Streamlit module for calculating and plotting option payoff charts.
    """
    st.subheader("🧮 Interactive Payoff Calculator")
    st.info("Build and visualize your own options strategies.")

    strategy_options = {
        "Long Call": "Buy a call option. Bullish, unlimited profit, defined risk.",
        "Long Put": "Buy a put option. Bearish, unlimited profit, defined risk.",
        "Bull Call Spread": "Buy ITM call, Sell OTM call. Bullish, defined risk/reward.",
        "Bear Put Spread": "Buy OTM put, Sell ITM put. Bearish, defined risk/reward."
    }

    selected_strategy = st.selectbox("Select Strategy", list(strategy_options.keys()), key="payoff_strategy_select")
    st.markdown(f"*{strategy_options[selected_strategy]}*")

    legs = []

    # Default range for stock prices on the chart
    stock_prices_for_chart = np.linspace(current_stock_price * 0.7, current_stock_price * 1.3, 200)

    if selected_strategy == "Long Call":
        col1, col2 = st.columns(2)
        with col1:
            strike = st.number_input("Call Strike Price", min_value=0.01, value=round(current_stock_price * 1.05, 2), format="%.2f", key="lc_strike")
        with col2:
            premium = st.number_input("Call Premium Paid", min_value=0.01, value=2.00, format="%.2f", key="lc_premium")
        legs.append({'type': 'call', 'strike': strike, 'premium': premium, 'action': 'buy'})

        # Adjust chart range based on inputs
        stock_prices_for_chart = np.linspace(min(strike, current_stock_price) * 0.9, max(strike, current_stock_price) * 1.1 + premium * 2, 200)

    elif selected_strategy == "Long Put":
        col1, col2 = st.columns(2)
        with col1:
            strike = st.number_input("Put Strike Price", min_value=0.01, value=round(current_stock_price * 0.95, 2), format="%.2f", key="lp_strike")
        with col2:
            premium = st.number_input("Put Premium Paid", min_value=0.01, value=2.00, format="%.2f", key="lp_premium")
        legs.append({'type': 'put', 'strike': strike, 'premium': premium, 'action': 'buy'})

        # Adjust chart range based on inputs
        stock_prices_for_chart = np.linspace(min(strike, current_stock_price) * 0.9 - premium * 2, max(strike, current_stock_price) * 1.1, 200)

    elif selected_strategy == "Bull Call Spread":
        col1, col2 = st.columns(2)
        with col1:
            buy_strike = st.number_input("Buy Call Strike", min_value=0.01, value=round(current_stock_price * 0.95, 2), format="%.2f", key="bcs_buy_strike")
            buy_premium = st.number_input("Buy Call Premium", min_value=0.01, value=3.00, format="%.2f", key="bcs_buy_premium")
        with col2:
            sell_strike = st.number_input("Sell Call Strike", min_value=0.01, value=round(current_stock_price * 1.05, 2), format="%.2f", key="bcs_sell_strike")
            sell_premium = st.number_input("Sell Call Premium", min_value=0.01, value=1.00, format="%.2f", key="bcs_sell_premium")

        if sell_strike <= buy_strike:
            st.warning("Sell Call Strike must be strictly greater than Buy Call Strike for a Bull Call Spread.")
        else:
            legs.append({'type': 'call', 'strike': buy_strike, 'premium': buy_premium, 'action': 'buy'})
            legs.append({'type': 'call', 'strike': sell_strike, 'premium': sell_premium, 'action': 'sell'})

            stock_prices_for_chart = np.linspace(min(buy_strike, sell_strike, current_stock_price) * 0.9, max(buy_strike, sell_strike, current_stock_price) * 1.1, 200)

    elif selected_strategy == "Bear Put Spread":
        col1, col2 = st.columns(2)
        with col1:
            buy_strike = st.number_input("Buy Put Strike", min_value=0.01, value=round(current_stock_price * 1.05, 2), format="%.2f", key="bps_buy_strike")
            buy_premium = st.number_input("Buy Put Premium", min_value=0.01, value=3.00, format="%.2f", key="bps_buy_premium")
        with col2:
            sell_strike = st.number_input("Sell Put Strike", min_value=0.01, value=round(current_stock_price * 0.95, 2), format="%.2f", key="bps_sell_strike")
            sell_premium = st.number_input("Sell Put Premium", min_value=0.01, value=1.00, format="%.2f", key="bps_sell_premium")

        if sell_strike >= buy_strike:
            st.warning("Sell Put Strike must be strictly less than Buy Put Strike for a Bear Put Spread.")
        else:
            legs.append({'type': 'put', 'strike': buy_strike, 'premium': buy_premium, 'action': 'buy'})
            legs.append({'type': 'put', 'strike': sell_strike, 'premium': sell_premium, 'action': 'sell'})

            stock_prices_for_chart = np.linspace(min(buy_strike, sell_strike, current_stock_price) * 0.9, max(buy_strike, sell_strike, current_stock_price) * 1.1, 200)

    st.markdown("---")
    if legs:
        payoffs = calculate_payoff_from_legs(stock_prices_for_chart, legs)
        payoff_fig = plot_generic_payoff_chart(stock_prices_for_chart, payoffs, legs, selected_strategy, ticker, current_stock_price)
        if payoff_fig:
            st.pyplot(payoff_fig, clear_figure=True)
            plt.close(payoff_fig)
        else:
            st.info("Payoff chart could not be generated with the provided inputs. Please check your strike and premium values.")
    else:
        st.info("Please configure the options legs using the inputs above to see the payoff chart.")


# === Dashboard Tab Display Functions ===
def display_main_analysis_tab(ticker, df, info, params, selection, overall_confidence, scores, final_weights, sentiment_score, expert_score, df_pivots, show_finviz_link):
    """Displays the main technical analysis and confidence score tab."""
    is_intraday = params['interval'] in ['5m', '60m']
    last = df.iloc[-1]
    signals = generate_signals_for_row(last, selection, df, is_intraday)

    col1, col2 = st.columns([1, 2])
    with col1:
        # --- Ticker Price and General Info (Moved to Top) ---
        st.subheader(f"📊 {info.get('longName', ticker)}")
        st.write(f"**Ticker:** {ticker}")

        current_price = last['Close']
        prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
        price_delta = current_price - prev_close

        # Determine bullish/bearish based on overall confidence
        sentiment_status = "Neutral"
        sentiment_icon = "⚪"
        if overall_confidence >= 65:
            sentiment_status = "Bullish"
            sentiment_icon = "⬆️"
        elif overall_confidence <= 35:
            sentiment_status = "Bearish"
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

        if show_finviz_link:
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
            if selection.get("EMA Trend"):
                st.markdown(format_indicator_display("Uptrend (21>50>200 EMA)", None, selection.get("EMA Trend"), signals))

            if selection.get("Parabolic SAR"):
                st.markdown(format_indicator_display("Bullish PSAR", last.get('psar'), selection.get("Parabolic SAR"), signals))

            if selection.get("ADX"):
                st.markdown(format_indicator_display("Strong Trend (ADX > 25)", last.get("adx"), selection.get("ADX"), signals))

        with st.expander("💨 Momentum & Volume Indicators", expanded=True):
            if selection.get("RSI Momentum"):
                st.markdown(format_indicator_display("Bullish Momentum (RSI > 50)", last.get("RSI"), selection.get("RSI Momentum"), signals))

            if selection.get("Stochastic"):
                st.markdown(format_indicator_display("Bullish Stoch Cross", last.get("stoch_k"), selection.get("Stochastic"), signals))

            if selection.get("CCI"):
                st.markdown(format_indicator_display("Bullish CCI (>0)", last.get("cci"), selection.get("CCI"), signals))

            if selection.get("ROC"):
                st.markdown(format_indicator_display("Positive ROC (>0)", last.get("roc"), selection.get("ROC"), signals))

            if selection.get("Volume Spike"):
                st.markdown(format_indicator_display("Volume Spike (>1.5x Avg)", last.get("Volume"), selection.get("Volume Spike"), signals))

            if selection.get("OBV"):
                st.markdown(format_indicator_display("OBV Rising", last.get("obv"), selection.get("OBV"), signals))

            if is_intraday and selection.get("VWAP"):
                st.markdown(format_indicator_display("Price > VWAP", last.get("vwap"), selection.get("VWAP"), signals))

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

                # Removed 'legend' kwarg from make_addplot calls
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
                addplot=ap, # ap is now guaranteed to be a list
                title=f"{ticker} - {params['interval']} chart",
                returnfig=True
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.info("Not enough data to generate chart.")

def display_trade_plan_options_tab(ticker, df, overall_confidence):
    """Displays the suggested trade plan and options strategy."""
    last = df.iloc[-1]
    current_stock_price = last['Close'] # Get current stock price for moneyness calculation

    st.subheader("📋 Suggested Stock Trade Plan (Bullish Swing)")
    entry_zone_start = last['EMA21'] * 0.99 if 'EMA21' in last and not pd.isna(last['EMA21']) else last['Close'] * 0.99
    entry_zone_end = last['EMA21'] * 1.01 if 'EMA21' in last and not pd.isna(last['EMA21']) else last['Close'] * 1.01

    stop_loss_val = last['Low'] - last['ATR'] if 'Low' in last and 'ATR' in last and not pd.isna(last['ATR']) and last['ATR'] > 0 else last['Close'] * 0.95
    profit_target_val = last['Close'] + (2 * (last['Close'] - stop_loss_val)) if 'Close' in last and stop_loss_val and not pd.isna(stop_loss_val) else last['Close'] * 1.1

    st.info(f"**Based on {overall_confidence:.0f}% Overall Confidence:**\n\n"
            f"**Entry Zone:** Between **${entry_zone_start:.2f}** and **${entry_zone_end:.2f}**.\n"
            f"**Stop-Loss:** A close below **${stop_loss_val:.2f}**.\n"
            f"**Profit Target:** Around **${profit_target_val:.2f}** (2:1 Reward/Risk).")
    st.markdown("---")

    st.subheader("🎭 Automated Options Strategy")
    stock_obj = yf.Ticker(ticker)
    expirations = stock_obj.options
    if not expirations:
        st.warning("No options data available for this ticker.")
    else:
        trade_plan = generate_option_trade_plan(ticker, overall_confidence, current_stock_price, expirations)

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

                # Check if 'Buy' leg exists before accessing
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

                # Check if 'Sell' leg exists before accessing
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

            elif trade_plan['Strategy'] in ["Buy ITM Call", "Buy ATM Call"]: # Corrected: use elif here
                rec_option = trade_plan['Contract']
                entry_premium = rec_option.get('ask', rec_option.get('lastPrice', 0))
                moneyness = get_moneyness(rec_option.get('strike'), current_stock_price, "call")

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

            # --- Display Payoff Chart for Automated Strategy ---
            st.markdown("---")
            st.subheader("📊 Automated Strategy Payoff Chart")
            payoff_fig = plot_automated_strategy_payoff(trade_plan, current_stock_price, ticker)
            if payoff_fig: # Only display if a figure was successfully generated
                st.pyplot(payoff_fig, clear_figure=True)
                plt.close(payoff_fig) # Close the figure to free up memory
            else:
                st.info("Automated strategy payoff chart could not be generated.")

        else: # This 'else' correctly belongs to the 'if trade_plan['status'] == 'success':
            st.warning(trade_plan['message'])

        st.markdown("---")
        st.subheader("⛓️ Full Option Chain")
        option_type = st.radio("Select Option Type", ["Calls", "Puts"], horizontal=True, key=f"option_type_{ticker}")
        exp_date_str = st.selectbox("Select Expiration Date to View", expirations, key=f"exp_date_select_{ticker}")
        if exp_date_str:
            calls, puts = get_options_chain(ticker, exp_date_str)

            # --- New Options Chain Analysis ---
            st.markdown("##### Options Chain Highlights & Suggestions")
            # Pass exp_date_str to analyze_options_chain
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
                'gamma': 'Measures the rate of change of an option\'s delta. High gamma = faster delta changes.',
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
    # Call the interactive payoff calculator
    display_interactive_payoff_calculator(current_stock_price, ticker)


def display_backtest_tab(ticker, selection):
    """Displays the historical backtest results."""
    st.subheader(f"🧪 Historical Backtest for {ticker}")
    st.info(f"Simulating trades based on your **currently selected indicators**. Entry is triggered if ALL selected signals are positive.")

    daily_hist, _ = get_data(ticker, "2y", "1d")
    if daily_hist is not None and not daily_hist.empty:
        daily_df_calculated = calculate_indicators(daily_hist.copy(), is_intraday=False)

        trades, wins, losses = backtest_strategy(daily_df_calculated, selection)
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

def display_news_info_tab(ticker, info, finviz_data):
    """Displays news and company information."""
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

def display_trade_log_tab(LOG_FILE, ticker, timeframe, overall_confidence):
    """Displays and manages the trade log."""
    st.subheader("📝 Log Your Trade Analysis")
    user_notes = st.text_area("Add your personal notes or trade thesis here:", key=f"trade_notes_{ticker}")

    st.info("Trade log functionality is pending implementation.")

def display_glossary(current_stock_price=None):
    """
    Displays a comprehensive glossary of all technical indicators and options greeks.
    """
    st.subheader("📚 Glossary: Technical Indicators & Options Concepts")
    st.info("This section provides detailed explanations, ideal criteria, and examples for the various technical indicators and options-related terms used in this application.")

    st.markdown("---")
    st.markdown("### Technical Indicators")
    for key, data in INDICATOR_DESCRIPTIONS.items():
        with st.expander(f"**{key}**"):
            st.markdown(f"**Description:** {data['description']}")
            st.markdown(f"**Ideal (Bullish):** {data['ideal']}")
            if 'example' in data:
                st.markdown(f"**Example:** {data['example']}")

    st.markdown("---")
    st.markdown("### Options Concepts & Greeks")
    st.info("The 'Greeks' (Delta, Theta, Gamma, Vega, Rho) are measures of an option's sensitivity to various factors. Understanding them is crucial for managing options risk.")
    for key, data in OPTIONS_GREEKS_DESCRIPTIONS.items():
        with st.expander(f"**{key}**"):
            st.markdown(f"**Description:** {data['description']}")
            st.markdown(f"**Ideal for Buyers/Sellers:** {data['ideal']}")
            if 'example' in data:
                st.markdown(f"**Example:** {data['example']}")
