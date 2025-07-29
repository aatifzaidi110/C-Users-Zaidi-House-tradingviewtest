# display_components.py - Fully Merged and Corrected

import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import yfinance as yf
import os
import numpy as np
from datetime import datetime, timedelta
import pytz # Import pytz for timezone handling

# Import functions from utils.py (ensure calculate_indicators is imported)
from utils import (
    backtest_strategy, calculate_indicators, generate_signals_for_row,
    suggest_options_strategy, get_options_chain, get_data, get_finviz_data,
    calculate_pivot_points, get_moneyness, analyze_options_chain,
    generate_directional_trade_plan, get_indicator_summary_text,
    get_economic_data_fred, get_vix_data, calculate_economic_score, calculate_sentiment_score
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
    
    base_display = ""
    if "ADX" in signal_name_base:
        base_display = f"{bullish_icon} {bearish_icon} **{signal_name_base}** ({value_str})"
    else:
        base_display = f"{bullish_icon} **{signal_name_base} Bullish** | {bearish_icon} **{signal_name_base} Bearish** ({value_str})"
    
    return f"{base_display}\n    - {details_text.replace(f'**{signal_name_base}:** ', '')}"


# === Option Payoff Chart Functions ===
def calculate_payoff_from_legs(stock_prices, legs):
    """
    Calculates the total payoff for a given set of option legs across a range of stock prices.
    Each leg is expected to be a dictionary: {'type': 'call'/'put', 'strike': float, 'premium': float, 'action': 'buy'/'sell', 'contracts': int}
    """
    total_payoff = np.zeros_like(stock_prices, dtype=float)

    for leg in legs:
        option_type = leg['type']
        strike = leg['strike']
        premium = leg['premium']
        action = leg['action']
        contracts = leg.get('contracts', 1) # Default to 1 contract if not specified

        if option_type == 'call':
            payoff_per_share = np.maximum(0, stock_prices - strike)
        elif option_type == 'put':
            payoff_per_share = np.maximum(0, strike - stock_prices)
        else:
            continue

        if action == 'buy':
            total_payoff += (payoff_per_share - premium) * contracts * 100 # Multiply by contracts and 100 shares/contract
        elif action == 'sell':
            total_payoff += (premium - payoff_per_share) * contracts * 100 # Multiply by contracts and 100 shares/contract
    return total_payoff

def plot_generic_payoff_chart(stock_prices, payoffs, legs, strategy_name, ticker, current_stock_price):
    """
    Generates and displays an option payoff chart for a generic strategy
    based on calculated payoffs and individual legs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Breakeven Line')

    ax.plot(stock_prices, payoffs, label=f'{strategy_name} Payoff', color='blue')

    # Plot strike price lines
    for leg in legs:
        color = 'green' if leg['action'] == 'buy' else 'red'
        linestyle = ':'
        ax.axvline(leg['strike'], color=color, linestyle=linestyle, label=f"{leg['action'].capitalize()} {leg['type'].capitalize()} Strike: ${leg['strike']:.2f}")

    ax.axvline(current_stock_price, color='orange', linestyle='-', linewidth=1.5, label=f'Current Price: ${current_stock_price:.2f}')

    # Calculate and plot breakeven points
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
    
    if max_payoff > 0 and max_payoff != np.inf:
        ax.text(stock_prices[-1], max_payoff * 0.9, f'Max Profit: ${max_payoff:.2f}', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=9)
    elif max_payoff == np.inf:
        ax.text(stock_prices[-1], ax.get_ylim()[1] * 0.9, 'Max Profit: Unlimited', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=9)

    if min_payoff < 0 and min_payoff != -np.inf:
        ax.text(stock_prices[-1], min_payoff * 1.1, f'Max Loss: ${-min_payoff:.2f}', verticalalignment='top', horizontalalignment='right', color='red', fontsize=9)
    elif min_payoff == -np.inf:
        ax.text(stock_prices[-1], ax.get_ylim()[0] * 0.9, 'Max Loss: Unlimited', verticalalignment='top', horizontalalignment='right', color='red', fontsize=9)


    ax.set_title(f'{ticker} {strategy_name} Payoff Chart')
    ax.set_xlabel('Stock Price at Expiration ($)')
    ax.set_ylabel('Profit/Loss ($)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig


# === Display Functions for Tabs ===

def display_technical_analysis_tab(ticker, df_calculated, is_intraday, indicator_selection):
    st.markdown("### 📊 Technical Analysis & Chart")

    if df_calculated.empty:
        st.info("No data available to display technical analysis.")
        return

    # Ensure the DataFrame index is a DatetimeIndex, coercing errors
    df_calculated.index = pd.to_datetime(df_calculated.index, errors='coerce')
    df_calculated = df_calculated[df_calculated.index.notna()]

    if df_calculated.empty:
        st.info("No valid date data in the DataFrame index after cleaning. Cannot display chart.")
        return

    # --- Standardize to UTC for robust comparison ---
    if df_calculated.index.tz is not None:
        df_calculated.index = df_calculated.index.tz_convert('UTC')
    else:
        try:
            df_calculated.index = df_calculated.index.tz_localize('America/New_York', errors='coerce').tz_convert('UTC')
        except pytz.exceptions.NonExistentTimeError:
            st.warning("Could not localize DataFrame index to 'America/New_York' for UTC conversion. Proceeding with existing timezone or naive index.", icon="⚠️")
            if df_calculated.index.tz is None:
                df_calculated.index = df_calculated.index.tz_localize('UTC', errors='coerce')

    current_time_utc = pd.Timestamp.now(tz='UTC')
    df_calculated = df_calculated[df_calculated.index <= current_time_utc]

    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit',
        wick='inherit',
        volume='in',
        ohlc='i'
    )
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc
    )

    add_plots = []
    current_panel_index = 0 # Panel 0 is reserved for the main OHLCV chart + Volume

    # --- Add Indicator Plots (only if selected and data is not all NaN) ---

    # EMAs (always on main panel 0)
    if indicator_selection.get("EMA Trend"):
        if not df_calculated['EMA21'].dropna().empty and \
           not df_calculated['EMA50'].dropna().empty and \
           not df_calculated['EMA200'].dropna().empty:
            add_plots.append(mpf.make_addplot(df_calculated['EMA21'], color='blue', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['EMA50'], color='orange', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['EMA200'], color='purple', panel=0, width=0.7, secondary_y=False))

    # Ichimoku Cloud (always on main panel 0)
    if indicator_selection.get("Ichimoku Cloud"):
        if not df_calculated['ichimoku_a'].dropna().empty and \
           not df_calculated['ichimoku_b'].dropna().empty and \
           not df_calculated['ichimoku_conversion_line'].dropna().empty and \
           not df_calculated['ichimoku_base_line'].dropna().empty:
            add_plots.append(mpf.make_addplot(df_calculated['ichimoku_a'], color='green', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['ichimoku_b'], color='red', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['ichimoku_conversion_line'], color='cyan', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['ichimoku_base_line'], color='magenta', panel=0, width=0.7, secondary_y=False))

    # Parabolic SAR (always on main panel 0)
    if indicator_selection.get("Parabolic SAR"):
        if not df_calculated['psar'].dropna().empty:
            add_plots.append(mpf.make_addplot(df_calculated['psar'], type='scatter', marker='.', markersize=50, color='lime', panel=0, secondary_y=False))

    # Bollinger Bands (always on main panel 0)
    if indicator_selection.get("Bollinger Bands"):
        if not df_calculated['BB_upper'].dropna().empty and \
           not df_calculated['BB_lower'].dropna().empty and \
           not df_calculated['BB_mavg'].dropna().empty:
            add_plots.append(mpf.make_addplot(df_calculated['BB_upper'], color='gray', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['BB_lower'], color='gray', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['BB_mavg'], color='blue', panel=0, width=0.7, secondary_y=False))

    # VWAP (if intraday, always on main panel 0)
    if is_intraday and indicator_selection.get("VWAP"):
        if 'VWAP' in df_calculated.columns and not df_calculated['VWAP'].dropna().empty:
            add_plots.append(mpf.make_addplot(df_calculated['VWAP'], color='darkred', panel=0, width=1.0, secondary_y=False))


    # MACD (separate panel)
    if indicator_selection.get("MACD"):
        if not df_calculated['MACD'].dropna().empty and \
           not df_calculated['MACD_Signal'].dropna().empty and \
           not df_calculated['MACD_Hist'].dropna().empty:
            current_panel_index += 1 # Increment for a new panel
            add_plots.extend([
                mpf.make_addplot(df_calculated['MACD'], panel=current_panel_index, color='red', secondary_y=False, width=0.7),
                mpf.make_addplot(df_calculated['MACD_Signal'], panel=current_panel_index, color='blue', secondary_y=False, width=0.7),
                mpf.make_addplot(df_calculated['MACD_Hist'], panel=current_panel_index, type='bar', color='green', secondary_y=False, width=0.7)
            ])

    # RSI (separate panel)
    if indicator_selection.get("RSI"):
        if not df_calculated['RSI'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['RSI'], panel=current_panel_index, color='purple', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(70, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(30, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # Stochastic Oscillator (separate panel)
    if indicator_selection.get("Stoch"):
        if not df_calculated['Stoch_K'].dropna().empty and not df_calculated['Stoch_D'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['Stoch_K'], panel=current_panel_index, color='magenta', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['Stoch_D'], panel=current_panel_index, color='cyan', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(80, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(20, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # ADX (separate panel)
    if indicator_selection.get("ADX"):
        if not df_calculated['adx'].dropna().empty and \
           not df_calculated['plus_di'].dropna().empty and \
           not df_calculated['minus_di'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['adx'], panel=current_panel_index, color='red', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['plus_di'], panel=current_panel_index, color='green', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['minus_di'], panel=current_panel_index, color='orange', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(25, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # CCI (separate panel)
    if indicator_selection.get("CCI"):
        if not df_calculated['CCI'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['CCI'], panel=current_panel_index, color='brown', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(100, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(-100, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # ROC (separate panel)
    if indicator_selection.get("ROC"):
        if not df_calculated['ROC'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['ROC'], panel=current_panel_index, color='darkgreen', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(0, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # OBV (separate panel)
    if indicator_selection.get("OBV"):
        if not df_calculated['obv'].dropna().empty and not df_calculated['obv_ema'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['obv'], panel=current_panel_index, color='blue', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['obv_ema'], color='orange', panel=current_panel_index, secondary_y=False, width=0.7))


    # --- Construct panels_list and panel_ratios_list for mpf.plot ---
    # Get all unique panel numbers used in add_plots (including 0 for the main chart)
    all_panels_in_addplots = []
    for i, ap in enumerate(add_plots):
        print(f"DEBUG: Item {i} in add_plots: Type={type(ap)}, Content={ap}")
        if hasattr(ap, 'panel'):
            all_panels_in_addplots.append(ap.panel)
        elif isinstance(ap, dict) and 'panel' in ap:
            # This case should ideally not happen if make_addplot is consistently used
            all_panels_in_addplots.append(ap['panel'])
        else:
            print(f"ERROR: Unexpected item in add_plots at index {i}. Does not have 'panel' attribute or 'panel' key.")
            # You might want to raise an error or handle it differently here
            # For now, we'll skip it to allow the code to run
            pass # Or all_panels_in_addplots.append(0) as a fallback

    
    # The panels list must include 0 for the main chart, plus any other unique panels from add_plots
    panels_list = sorted(list(set([0] + all_panels_in_addplots)))

    # The panel_ratios_list must have the same length as panels_list
    # Give panel 0 (main chart + volume) a larger ratio (e.g., 3)
    # Give all other indicator panels a smaller ratio (e.g., 1)
    panel_ratios_list = [3] + [1] * (len(panels_list) - 1)

    # Adjust figure size dynamically based on the number of panels
    figure_height = 8 + len(panels_list) * 2 # Base height + 2 inches for each panel

    # --- Generate the Plot ---
    try:
        fig, axlist = mpf.plot(
            df_calculated,
            type='candle',
            style=s,
            title=f"Technical Analysis for {ticker}",
            ylabel='Price',
            addplot=add_plots if add_plots else None,
            panel_ratios=panel_ratios_list,
            figscale=1.2,
            figsize=(12, figure_height),
            volume=True,
            volume_panel=0,
            returnfig=True,
            panels=panels_list # Explicitly pass the list of panel indices to use
        )
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning(f"⚠️ Could not render chart. Please check data and indicator selections. Error: {e}")
        st.exception(e)
        if "panel_ratios" in str(e) or "num panels" in str(e):
            st.error(f"Panel Configuration Error: Determined panel_ratios_list length: {len(panel_ratios_list)}, determined panels_list length: {len(panels_list)}. Error details: {e}")
            st.info("This indicates a mismatch between the number of panels mplfinance expects and what it can actually plot. This often happens if an indicator's data for the current view is entirely NaN after internal mplfinance filtering.")
            st.info("Please verify the data in your DataFrame columns for the selected indicators, especially the latest values. If data is sparse, try a wider date range.")

    # Indicator Summary
    if not df_calculated.empty:
        st.markdown("---")
        st.subheader("💡 Indicator Summary")
        # Ensure get_indicator_summary_text can handle the latest_row directly
        summary_text = get_indicator_summary_text(df_calculated.iloc[-1], indicator_selection)
        st.write(summary_text)

    # Display Pivot Points
    if 'Pivot (P)' in df_calculated.columns and not df_calculated['Pivot (P)'].empty:
        last_pivot = df_calculated.iloc[-1]
        st.markdown("---")
        st.subheader("🎯 Pivot Points (Classic)")
        st.write(f"**P:** {last_pivot.get('Pivot (P)', 'N/A'):.2f} | "
                 f"**R1:** {last_pivot.get('R1', 'N/A'):.2f} | "
                 f"**R2:** {last_pivot.get('R2', 'N/A'):.2f} | "
                 f"**S1:** {last_pivot.get('S1', 'N/A'):.2f} | "
                 f"**S2:** {last_pivot.get('S2', 'N/A'):.2f}")

    # Display Trade Signals
    if 'Trade Signal (Bullish)' in df_calculated.columns or 'Trade Signal (Bearish)' in df_calculated.columns:
        last_row_signals = generate_signals_for_row(df_calculated.iloc[-1])
        st.markdown("---")
        st.subheader("🚦 Current Trade Signals")
        if last_row_signals['bullish_signals']:
            st.success("📈 Bullish Signals Detected:")
            for signal, rationale in last_row_signals['bullish_signals'].items():
                st.markdown(f"- **{signal}:** {rationale}")
        else:
            st.info("No immediate bullish signals detected.")

        if last_row_signals['bearish_signals']:
            st.error("📉 Bearish Signals Detected:")
            for signal, rationale in last_row_signals['bearish_signals'].items():
                st.markdown(f"- **{signal}:** {rationale}")
        else:
            st.info("No immediate bearish signals detected.")

    # Backtesting Section
    st.markdown("---")
    st.subheader("📈 Strategy Backtesting")

    # Assuming backtest_strategy is defined in utils.py and handles indicator_selection
    backtest_data = backtest_strategy(df_calculated.copy(), indicator_selection)

    if backtest_data:
        total_trades = backtest_data.get('total_trades', 0)
        winning_trades = backtest_data.get('winning_trades', 0)
        losing_trades = backtest_data.get('losing_trades', 0)
        win_rate = backtest_data.get('win_rate', 0.0)
        total_profit = backtest_data.get('total_profit', 0.0)
        avg_profit_per_trade = backtest_data.get('avg_profit_per_trade', 0.0)
        cagr = backtest_data.get('cagr', 0.0)
        max_drawdown = backtest_data.get('max_drawdown', 0.0)

        st.write(f"**Total Trades:** {total_trades}")
        st.write(f"**Winning Trades:** {winning_trades}")
        st.write(f"**Losing Trades:** {losing_trades}")
        st.write(f"**Win Rate:** {win_rate:.2f}%")
        st.write(f"**Total Profit/Loss:** ${total_profit:.2f}")
        st.write(f"**Average Profit/Loss per Trade:** ${avg_profit_per_trade:.2f}")
        st.write(f"**CAGR (Compound Annual Growth Rate):** {cagr:.2f}%")
        st.write(f"**Max Drawdown:** {max_drawdown:.2f}%")

        # Display trade log
        trade_log_df = backtest_data.get('trade_log', pd.DataFrame())
        if not trade_log_df.empty:
            st.markdown("---")
            st.subheader("Detailed Trade Log")
            st.dataframe(trade_log_df)
        else:
            st.info("No trades executed during backtesting period.")
    else:
        st.info("Backtesting data not available. Ensure sufficient data for indicators and signals.")

    # Directional Trade Plan
    st.markdown("---")
    st.subheader("🗺️ Directional Trade Plan (Based on Current Data)")
    if not df_calculated.empty:
        try:
            last_row = df_calculated.iloc[-1]
            trade_plan_result = generate_directional_trade_plan(last_row, indicator_selection)

            if trade_plan_result:
                st.write(f"**Direction:** {trade_plan_result.get('direction', 'N/A')}")
                st.write(f"**Confidence Score:** {trade_plan_result.get('confidence_score', 'N/A'):.2f}%")
                st.write(f"**Entry Zone:** ${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} - ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}")
                st.write(f"**Target Price:** ${trade_plan_result.get('target_price', 'N/A'):.2f}")
                st.write(f"**Stop Loss:** ${trade_plan_result.get('stop_loss', 'N/A'):.2f}")
                st.write(f"**Reward/Risk Ratio:** {trade_plan_result.get('reward_risk_ratio', 'N/A'):.1f}:1")
                st.markdown("---")
                st.write("**Key Rationale:**")
                st.write(trade_plan_result.get('key_rationale', 'No specific rationale available.'))
                st.write("**Detailed Entry Criteria:**")
                for criteria in trade_plan_result.get('entry_criteria_details', []):
                    st.markdown(f"- {criteria}")
                st.write("**Detailed Exit Criteria:**")
                for criteria in trade_plan_result.get('exit_criteria_details', []):
                    st.markdown(f"- {criteria}")
            else:
                st.info("Could not generate a directional trade plan. Ensure sufficient data and valid indicator selections.")
        except Exception as e:
            st.error(f"An error occurred while generating the trade plan: {e}")
            st.exception(e) # For debugging


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
        legs_to_calculate.append(leg) # Append the dictionary, contracts will be handled in calculate_payoff_from_legs

    st.markdown("---")

    # --- Calculation and Display ---
    if st.button("Calculate Payoff"):
        if not stock_action == "None" and num_shares == 0:
            st.warning("Please enter the number of shares for the stock leg, or set action to 'None'.")
        elif not legs_to_calculate and stock_action == "None":
            st.warning("Please add at least one stock or option leg to calculate the payoff.")
        else:
            # Determine the range of stock prices for the chart
            min_price_range = current_stock_price * 0.8
            max_price_range = current_stock_price * 1.2
            if legs_to_calculate:
                strikes = [leg['strike'] for leg in legs_to_calculate]
                min_price_range = min(min_price_range, min(strikes) * 0.9)
                max_price_range = max(max_price_range, max(strikes) * 1.1)
            
            # Extend range for potential unlimited profit/loss
            if any(leg['type'] == 'call' and leg['action'] == 'buy' for leg in legs_to_calculate):
                max_price_range += current_stock_price * 0.5
            if any(leg['type'] == 'put' and leg['action'] == 'buy' for leg in legs_to_calculate):
                min_price_range -= current_stock_price * 0.5

            stock_prices_range = np.linspace(min_price_range, max_price_range, 200)

            # Calculate payoff from stock leg
            stock_payoff = np.zeros_like(stock_prices_range, dtype=float)
            if stock_action == "Buy":
                stock_payoff = (stock_prices_range - stock_purchase_price) * num_shares
            elif stock_action == "Sell":
                stock_payoff = (stock_purchase_price - stock_prices_range) * num_shares

            # Calculate payoff from option legs
            option_payoff = calculate_payoff_from_legs(stock_prices_range, legs_to_calculate)

            total_payoff = stock_payoff + option_payoff # Option payoff already includes contracts * 100

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
            table_stock_prices = np.linspace(min_price_range, max_price_range, 20).round(2) # 20 points for table
            
            for price in table_stock_prices:
                # Recalculate payoff for each specific price point
                current_stock_payoff = 0
                if stock_action == "Buy":
                    current_stock_payoff = (price - stock_purchase_price) * num_shares
                elif stock_action == "Sell":
                    current_stock_payoff = (stock_purchase_price - price) * num_shares
                
                current_option_payoff = calculate_payoff_from_legs(np.array([price]), legs_to_calculate)[0]
                
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
                st.markdown(format_indicator_display("Stochastic Oscillator", last.get("Stoch_K"), bullish_signals.get("Stochastic", False), bearish_signals.get("Stochastic", False), selection.get("Stochastic")))

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
                if 'BB_upper' in last and 'BB_lower' in last and not pd.isna(last['BB_upper']) and not pd.isna(last['BB_lower']):
                    if last['Close'] > last['BB_upper']:
                        bb_status = '🔴 **Price Above Upper Band** (Overbought/Strong Uptrend)'
                    elif last['Close'] < last['BB_lower']:
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
                    if 'Pivot (P)' in last_pivot and not pd.isna(last_pivot['Pivot (P)']):
                        if last['Close'] > last_pivot['R1']:
                            pivot_status = '🟢 **Price Above R1** (Strong Bullish)'
                        elif last['Close'] > last_pivot['Pivot (P)']:
                            pivot_status = '🟡 **Price Above Pivot** (Bullish)'
                        elif last['Close'] < last_pivot['S1']:
                            pivot_status = '🔴 **Price Below S1** (Strong Bearish)'
                        elif last['Close'] < last_pivot['Pivot (P)']:
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
        # The main chart in display_main_analysis_tab does not use the complex add_plots/panels logic
        # of display_technical_analysis_tab. It uses a simpler `mav` parameter.
        mav_tuple = (21, 50, 200) if selection.get("EMA Trend") else None
        
        ap = [] # Initialize addplot as an empty list for this chart
        
        # Add Bollinger Bands to addplot if selected and data is available
        if selection.get("Bollinger Bands"):
            # Check if BB columns exist and are not all NaN in the tail data
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns and not df[['BB_upper', 'BB_lower']].tail(120).isnull().all().all():
                ap.append(mpf.make_addplot(df.tail(120)[['BB_upper', 'BB_lower']]))
            else:
                st.warning("Bollinger Bands data not available or all NaN for plotting.", icon="⚠️")

        # Add Pivot Points to addplot if selected and data is available (for daily/weekly)
        if selection.get("Pivot Points") and not is_intraday and not df_pivots.empty and len(df_pivots) > 1:
            last_pivot = df_pivots.iloc[-1]
            # Ensure pivot values are not NaN before attempting to plot
            if not pd.isna(last_pivot.get('Pivot (P)')):
                # Create Series aligned with the chart's index (df.tail(120).index)
                # This ensures the horizontal lines span the visible chart
                chart_index = df.tail(120).index
                
                pivot_values = pd.Series(last_pivot['Pivot (P)'], index=chart_index)
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
                df.tail(120), # Display last 120 data points for clarity
                type='candle',
                style='yahoo',
                mav=mav_tuple, # Apply MAs if selected
                volume=True,
                addplot=ap, # Add other plots like BB and Pivots
                title=f"{ticker} - {params['interval']} chart",
                returnfig=True
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough data to generate chart.")


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

def display_options_analysis_tab(ticker, current_stock_price, expirations, prev_close, overall_confidence, trade_direction):
    st.markdown("### 📈 Options Analysis")
    _display_common_header(ticker, current_stock_price, prev_close, overall_confidence, trade_direction)

    if not expirations:
        st.info("No options data available for this ticker or expiration dates are missing.")
        return

    selected_expiry = st.selectbox("Select Expiration Date for Chain", expirations)

    if selected_expiry:
        options_chain = get_options_chain(ticker, selected_expiry)

        if options_chain and (not options_chain['calls'].empty or not options_chain['puts'].empty):
            st.subheader(f"Options Chain for {selected_expiry}")

            col_chain1, col_chain2 = st.columns(2)
            with col_chain1:
                st.markdown("#### Calls")
                # Display relevant call option columns
                if not options_chain['calls'].empty:
                    st.dataframe(options_chain['calls'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].round(2))
                else:
                    st.info("No call options available for this expiration.")

            with col_chain2:
                st.markdown("#### Puts")
                # Display relevant put option columns
                if not options_chain['puts'].empty:
                    st.dataframe(options_chain['puts'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].round(2))
                else:
                    st.info("No put options available for this expiration.")

            st.markdown("---")
            st.subheader("Options Insights & Strategy Suggestions")
            
            analyzed_options = analyze_options_chain(options_chain, current_stock_price)
            suggested_strategies = suggest_options_strategy(current_stock_price, overall_confidence, trade_direction, analyzed_options)

            if suggested_strategies:
                st.markdown("##### Suggested Strategies:")
                for strategy_name, details in suggested_strategies.items():
                    with st.expander(f"**{strategy_name}** - *{details.get('rationale', 'No rationale provided.')}*"):
                        st.write(f"**Max Profit:** {details.get('max_profit', 'N/A')}")
                        st.write(f"**Max Loss:** {details.get('max_loss', 'N/A')}")
                        st.write(f"**Breakeven(s):** {details.get('breakevens', 'N/A')}")
                        
                        st.markdown("###### Legs:")
                        legs_data = []
                        for leg in details.get('legs', []):
                            st.write(f"- {leg['action'].capitalize()} {leg['quantity']} {leg['type'].capitalize()} @ ${leg['strike']:.2f} (Premium: ${leg['premium']:.2f})")
                            legs_data.append({
                                'type': leg['type'],
                                'strike': leg['strike'],
                                'premium': leg['premium'],
                                'action': leg['action'],
                                'contracts': leg.get('quantity', 1) # Use quantity as contracts
                            })

                        # Plotting Payoff Diagram for the suggested strategy
                        if legs_data:
                            # Define a range of stock prices around the current price and strikes
                            min_price = min([leg['strike'] for leg in legs_data]) * 0.8
                            max_price = max([leg['strike'] for leg in legs_data]) * 1.2
                            if current_stock_price < min_price: min_price = current_stock_price * 0.8
                            if current_stock_price > max_price: max_price = current_stock_price * 1.2

                            stock_prices_range = np.linspace(min_price, max_price, 200)
                            payoffs = calculate_payoff_from_legs(stock_prices_range, legs_data)
                            plot_generic_payoff_chart(stock_prices_range, payoffs, legs_data, strategy_name, ticker, current_stock_price)

            else:
                st.info("No specific options strategies suggested based on current analysis and sentiment.")

        else:
            st.info(f"Could not retrieve options chain for {ticker} on {selected_expiry}. Please check the ticker and date.")
    else:
        st.info("Please select an expiration date to view the options chain.")


def display_backtesting_tab(backtest_data, indicator_selection_current_run):
    st.markdown("### 📈 Backtesting Results")

    if not backtest_data:
        st.info("No backtesting data available. Please run analysis on a ticker first.")
        return

    # Display overall backtesting metrics
    st.subheader("Overall Strategy Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", backtest_data.get('total_trades', 0))
        st.metric("Winning Trades", backtest_data.get('winning_trades', 0))
    with col2:
        st.metric("Losing Trades", backtest_data.get('losing_trades', 0))
        st.metric("Win Rate", f"{backtest_data.get('win_rate', 0.0):.2f}%")
    with col3:
        st.metric("Total Profit/Loss", f"${backtest_data.get('total_profit', 0.0):.2f}")
        st.metric("Avg Profit/Loss per Trade", f"${backtest_data.get('avg_profit_per_trade', 0.0):.2f}")
    
    st.metric("CAGR (Compound Annual Growth Rate)", f"{backtest_data.get('cagr', 0.0):.2f}%")
    st.metric("Max Drawdown", f"{backtest_data.get('max_drawdown', 0.0):.2f}%")

    st.markdown("---")
    st.subheader("Strategy Parameters Used")
    st.write("The backtest was run with the following indicator selections:")
    selected_indicators_list = [
        key for key, value in indicator_selection_current_run.items() if value
    ]
    if selected_indicators_list:
        st.write(", ".join(selected_indicators_list))
    else:
        st.write("No specific indicators were selected for this backtest (or default settings were used).")

    # Display detailed trade log
    trade_log_df = backtest_data.get('trade_log', pd.DataFrame())
    if not trade_log_df.empty:
        st.markdown("---")
        st.subheader("Detailed Trade Log")
        # Format PnL column to 2 decimal places and ensure it's numeric
        if 'PnL' in trade_log_df.columns:
            trade_log_df['PnL'] = pd.to_numeric(trade_log_df['PnL'], errors='coerce').fillna(0.0).map('${:,.2f}'.format)
        
        # Format 'Entry Price' and 'Exit Price' if they exist
        for col in ['Entry Price', 'Exit Price']:
            if col in trade_log_df.columns:
                trade_log_df[col] = pd.to_numeric(trade_log_df[col], errors='coerce').map('${:,.2f}'.format)

        st.dataframe(trade_log_df)
    else:
        st.info("No trades were executed during the backtesting period with the selected strategy and data.")


def display_trade_log_tab(ticker, current_price, timeframe, overall_confidence, trade_direction):
    st.markdown("### 📜 Trade Log")

    log_file = f"trade_log_{ticker}_{timeframe.replace(' ', '_')}.csv"

    # Load existing log
    if os.path.exists(log_file):
        trade_log_df = pd.read_csv(log_file)
    else:
        trade_log_df = pd.DataFrame(columns=["Timestamp", "Ticker", "Timeframe", "Confidence", "Direction", "Price", "PnL", "Notes"])

    # Display existing trade log
    if not trade_log_df.empty:
        st.dataframe(trade_log_df)
    else:
        st.info("No trade entries yet. Add a new trade using the form below.")

    st.markdown("#### Add New Trade Entry")
    with st.form("new_trade_form"):
        trade_type = st.selectbox("Trade Type", ["Long", "Short", "Exit Long", "Exit Short"])
        # Ensure current_price is a float for initial value
        price_value = float(current_price) if current_price is not None else 0.0
        price = st.number_input("Price", value=price_value, format="%.2f")
        pnl = st.number_input("PnL (if exit)", value=0.0, format="%.2f")
        notes = st.text_area("Notes")

        submitted = st.form_submit_button("Add Trade to Log")
        if submitted:
            if price <= 0: # Check for valid price
                st.error("Price must be a positive value.")
            else:
                current_datetime = datetime.now()
                new_entry = {
                    "Timestamp": current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
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
    
    st.markdown("---")
    st.subheader("Your Trade History")
    if not trade_log_df.empty:
        # Filter log to show only trades for the current ticker
        df_ticker_log = trade_log_df[trade_log_df['Ticker'].str.upper() == ticker.upper()].copy()

        if not df_ticker_log.empty:
            st.dataframe(df_ticker_log)
            # Option to download full log
            csv = trade_log_df.to_csv(index=False).encode('utf-8')
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
def display_scanner_tab(scanner_results_df): # Renamed from display_scanner_results_tab for consistency with app.py
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
            "Direction": row.get('Trade Direction', 'N/A'), # Use 'Trade Direction' as in utils.py scanner output
            "Price": f"${row.get('Current Price', 'N/A'):.2f}" if pd.notna(row.get('Current Price')) else 'N/A',
            "ATR": f"{row.get('ATR', 'N/A'):.2f}" if pd.notna(row.get('ATR')) else 'N/A',
            "Target": f"${row.get('Target Price', 'N/A'):.2f}" if pd.notna(row.get('Target Price')) else 'N/A',
            "Stop Loss": f"${row.get('Stop Loss', 'N/A'):.2f}" if pd.notna(row.get('Stop Loss')) else 'N/A',
            "Entry Zone": f"${row.get('Entry Zone Start', 'N/A'):.2f} - ${row.get('Entry Zone End', 'N/A'):.2f}" if pd.notna(row.get('Entry Zone Start')) and pd.notna(row.get('Entry Zone End')) else 'N/A',
            "R/R": f"{row.get('Reward/Risk', 'N/A')}",
            "Pivot (P)": f"${row.get('Pivot (P)', 'N/A'):.2f}" if pd.notna(row.get('Pivot (P)')) else 'N/A',
            "R1": f"${row.get('Resistance 1 (R1)', 'N/A'):.2f}" if pd.notna(row.get('Resistance 1 (R1)')) else 'N/A',
            "S1": f"${row.get('Support 1 (S1)', 'N/A'):.2f}" if pd.notna(row.get('Support 1 (S1)')) else 'N/A',
            "R2": f"${row.get('Resistance 2 (R2)', 'N/A'):.2f}" if pd.notna(row.get('Resistance 2 (R2)')) else 'N/A',
            "S2": f"${row.get('Support 2 (S2)', 'N/A'):.2f}" if pd.notna(row.get('Support 2 (S2)')) else 'N/A',
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

