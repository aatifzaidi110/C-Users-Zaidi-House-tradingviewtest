import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt # Import matplotlib for plt.close()
import numpy as np # Import numpy for np.linspace and np.inf
from datetime import datetime, timedelta # Import datetime and timedelta for date operations
import pytz # Import pytz for timezone handling
import os # Import the os module for path operations

# Import functions from utils.py (ensure calculate_indicators is imported)
from utils import (
    backtest_strategy, calculate_indicators, generate_signals_for_row,
    suggest_options_strategy, get_options_chain, get_data, get_finviz_data,
    calculate_pivot_points, analyze_options_chain,
    generate_directional_trade_plan,
    get_economic_data_fred, get_vix_data, calculate_economic_score, calculate_sentiment_score,
    calculate_confidence_score # Added this import as it's used in display_backtesting_tab
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
    
    # Ensure prev_close is a valid number for delta calculation
    price_delta = current_price - prev_close if prev_close is not None and pd.notna(prev_close) else 0.0
    
    sentiment_status = trade_direction # Use the determined trade_direction
    sentiment_icon = "⚪"
    if trade_direction == "Bullish":
        sentiment_icon = "⬆️"
    elif trade_direction == "Bearish":
        sentiment_icon = "⬇️"
        
    col1, col2 = st.columns([1, 1])
    with col1:
        # Safely format current_price and price_delta
        current_price_str = f"${current_price:.2f}" if current_price is not None and pd.notna(current_price) else "N/A"
        price_delta_str = f"${price_delta:.2f}" if price_delta is not None and pd.notna(price_delta) else "N/A"
        st.metric(label="Current Price", value=current_price_str, delta=price_delta_str)
    with col2:
        st.markdown(f"**Overall Sentiment:** {sentiment_icon} {sentiment_status} ({overall_confidence:.2f}%)")
    st.markdown("---")


# === Helper for Indicator Display ===
def format_indicator_display(signal_name_base, current_value, bullish_fired, bearish_fired):
    """
    Formats and displays a single technical indicator's concise information,
    showing both bullish and bearish status, plus qualitative insights.
    This function is now aligned with the get_indicator_summary_text signature in utils.py.
    """
    bullish_icon = '🟢' if bullish_fired else '⚪'
    bearish_icon = '🔴' if bearish_fired else '⚪'
    
    value_str = ""
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            value_str = f"Current: `{current_value:.2f}`"
        else:
            value_str = f"Current: `{current_value}`" # For boolean (Volume Spike)
    else:
        value_str = "Current: N/A"

    # The get_indicator_summary_text now directly returns the full summary text
    # We pass a dummy 'row' as it's not used in get_indicator_summary_text's current logic
    # but it's part of its signature.
    dummy_row = pd.Series({'Close': current_value}) # Create a dummy row for the function
    details_text = get_indicator_summary_text(signal_name_base, dummy_row, {signal_name_base: bullish_fired}, {signal_name_base: bearish_fired})
    
    # Remove the initial "**Indicator Name:** " part from the summary text
    clean_details_text = details_text.replace(f"**{signal_name_base}:** ", "")

    base_display = f"{bullish_icon} {bearish_icon} **{signal_name_base}** ({value_str})"
    
    return f"{base_display}\n    - {clean_details_text}"


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

def display_technical_analysis_tab(ticker, df_calculated, is_intraday, indicator_selection, normalized_weights): # Added normalized_weights
    """
    Displays the technical analysis chart with selected indicators.
    Args:
        ticker (str): The stock ticker symbol.
        df_calculated (pd.DataFrame): DataFrame with historical data and calculated indicators.
        is_intraday (bool): True if the data is intraday, False otherwise.
        indicator_selection (dict): Dictionary of selected indicators from Streamlit session state.
        normalized_weights (dict): Dictionary of normalized weights for confidence scoring.
    """
    st.subheader(f"Technical Analysis for {ticker}")

    if df_calculated.empty:
        st.info("No data available to display technical analysis chart.")
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
        if 'psar' in df_calculated.columns and not df_calculated['psar'].dropna().empty:
            add_plots.append(mpf.make_addplot(df_calculated['psar'], type='scatter', marker='.', markersize=50, color='lime', panel=0, secondary_y=False))

    # Bollinger Bands (always on main panel 0)
    if indicator_selection.get("Bollinger Bands"):
        bb_cols = ['BB_upper', 'BB_lower', 'BB_mavg']
        if all(col in df_calculated.columns and not df_calculated[col].dropna().empty for col in bb_cols):
            add_plots.append(mpf.make_addplot(df_calculated['BB_upper'], color='gray', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['BB_lower'], color='gray', panel=0, width=0.7, secondary_y=False))
            add_plots.append(mpf.make_addplot(df_calculated['BB_mavg'], color='blue', panel=0, width=0.7, secondary_y=False))

    # VWAP (if intraday, always on main panel 0)
    if is_intraday and indicator_selection.get("VWAP"):
        if 'VWAP' in df_calculated.columns and not df_calculated['VWAP'].dropna().empty:
            add_plots.append(mpf.make_addplot(df_calculated['VWAP'], color='darkred', panel=0, width=1.0, secondary_y=False))


    # MACD (separate panel)
    if indicator_selection.get("MACD"):
        macd_cols = ['MACD', 'MACD_Signal', 'MACD_Hist']
        if all(col in df_calculated.columns and not df_calculated[col].dropna().empty for col in macd_cols):
            current_panel_index += 1 # Increment for a new panel
            add_plots.extend([
                mpf.make_addplot(df_calculated['MACD'], panel=current_panel_index, color='red', secondary_y=False, width=0.7),
                mpf.make_addplot(df_calculated['MACD_Signal'], panel=current_panel_index, color='blue', secondary_y=False, width=0.7),
                mpf.make_addplot(df_calculated['MACD_Hist'], panel=current_panel_index, type='bar', color='green', secondary_y=False, width=0.7)
            ])

    # RSI (separate panel)
    if indicator_selection.get("RSI Momentum"): # Corrected from "RSI" to "RSI Momentum"
        if 'RSI' in df_calculated.columns and not df_calculated['RSI'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['RSI'], panel=current_panel_index, color='purple', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(70, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(30, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # Stochastic Oscillator (separate panel)
    if indicator_selection.get("Stochastic"): # Corrected from "Stoch" to "Stochastic"
        stoch_cols = ['Stoch_K', 'Stoch_D']
        if all(col in df_calculated.columns and not df_calculated[col].dropna().empty for col in stoch_cols):
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['Stoch_K'], panel=current_panel_index, color='magenta', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['Stoch_D'], panel=current_panel_index, color='cyan', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(80, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(20, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # ADX (separate panel)
    if indicator_selection.get("ADX"):
        adx_cols = ['adx', 'plus_di', 'minus_di']
        if all(col in df_calculated.columns and not df_calculated[col].dropna().empty for col in adx_cols):
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['adx'], panel=current_panel_index, color='red', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['plus_di'], panel=current_panel_index, color='green', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['minus_di'], panel=current_panel_index, color='orange', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(25, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # CCI (separate panel)
    if indicator_selection.get("CCI"):
        if 'CCI' in df_calculated.columns and not df_calculated['CCI'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['CCI'], panel=current_panel_index, color='brown', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(100, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(-100, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # ROC (separate panel)
    if indicator_selection.get("ROC"):
        if 'ROC' in df_calculated.columns and not df_calculated['ROC'].dropna().empty:
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['ROC'], panel=current_panel_index, color='darkgreen', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(pd.Series(0, index=df_calculated.index), panel=current_panel_index, color='gray', linestyle=':', secondary_y=False, width=0.7))

    # OBV (separate panel)
    if indicator_selection.get("OBV"):
        obv_cols = ['obv', 'obv_ema']
        if all(col in df_calculated.columns and not df_calculated[col].dropna().empty for col in obv_cols):
            current_panel_index += 1
            add_plots.append(mpf.make_addplot(df_calculated['obv'], panel=current_panel_index, color='blue', secondary_y=False, width=0.7))
            add_plots.append(mpf.make_addplot(df_calculated['obv_ema'], color='orange', panel=current_panel_index, secondary_y=False, width=0.7))


    # --- Construct panels_list and panel_ratios_list for mpf.plot ---
    # Get all unique panel numbers used in add_plots (including 0 for the main chart)
    all_panels_in_addplots = [ap['panel'] for ap in add_plots if isinstance(ap, dict) and 'panel' in ap]
    
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
            figscale=1.2, # figscale is correctly here
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
        
        latest_row = df_calculated.iloc[-1]
        bullish_signals_summary, bearish_signals_summary, _ = generate_signals_for_row(
            latest_row,
            indicator_selection,
            normalized_weights
        )

        summary_texts = []
        all_indicator_names = [
            "EMA Trend", "Ichimoku Cloud", "Parabolic SAR", "ADX",
            "RSI Momentum", "Stochastic", "MACD", "Bollinger Bands",
            "Volume Spike", "CCI", "ROC", "OBV", "VWAP", "Pivot Points"
        ]

        for ind_name in all_indicator_names:
            if indicator_selection.get(ind_name, False):
                current_ind_value = None
                # Map indicator name to its corresponding value in latest_row
                if ind_name == "RSI Momentum": current_ind_value = latest_row.get("RSI")
                elif ind_name == "Stochastic": current_ind_value = latest_row.get("Stoch_K")
                elif ind_name == "ADX": current_ind_value = latest_row.get("adx")
                elif ind_name == "CCI": current_ind_value = latest_row.get("CCI")
                elif ind_name == "ROC": current_ind_value = latest_row.get("ROC")
                elif ind_name == "OBV": current_ind_value = latest_row.get("obv")
                elif ind_name == "VWAP": current_ind_value = latest_row.get("VWAP")
                elif ind_name == "Parabolic SAR": current_ind_value = latest_row.get("psar")
                elif ind_name == "Bollinger Bands": current_ind_value = latest_row.get("BB_mavg") # Using middle band for value
                elif ind_name == "Ichimoku Cloud": current_ind_value = latest_row.get("ichimoku_conversion_line") # Using conversion line for value
                elif ind_name == "EMA Trend": current_ind_value = latest_row.get("EMA21") # Using EMA21 for value
                elif ind_name == "Volume Spike": current_ind_value = latest_row.get("Volume_Spike") # Volume Spike is boolean
                elif ind_name == "Pivot Points": current_ind_value = latest_row.get("Pivot") # Pivot Points value

                # Call get_indicator_summary_text for each selected indicator
                # The get_indicator_summary_text function in utils.py expects (indicator_name, row, bullish_signals, bearish_signals)
                summary_line = get_indicator_summary_text(
                    ind_name,
                    latest_row, # Pass the full row
                    bullish_signals_summary, # Pass the full bullish signals dict
                    bearish_signals_summary # Pass the full bearish signals dict
                )
                summary_texts.append(summary_line)
        
        if summary_texts:
            st.markdown("\n".join(summary_texts))
        else:
            st.info("No selected indicators to summarize.")


    # Display Pivot Points
    # Ensure df_calculated has 'Pivot' column (calculated in utils.py)
    if 'Pivot' in df_calculated.columns and not df_calculated['Pivot'].empty: # Corrected column name to 'Pivot'
        last_pivot = df_calculated.iloc[-1]
        st.markdown("---")
        st.subheader("🎯 Pivot Points (Classic)")
        st.write(f"**P:** {last_pivot.get('Pivot', 'N/A'):.2f} | " # Corrected column name to 'Pivot'
                 f"**R1:** {last_pivot.get('R1', 'N/A'):.2f} | "
                 f"**R2:** {last_pivot.get('R2', 'N/A'):.2f} | "
                 f"**S1:** {last_pivot.get('S1', 'N/A'):.2f} | "
                 f"**S2:** {last_pivot.get('S2', 'N/A'):.2f}")
    else:
        st.info("Pivot Points data not available for display.")


    # Display Trade Signals (This section is redundant if using the overall summary, but keeping for now)
    # The generate_signals_for_row is already called above for the summary.
    # This part can be simplified or removed if the "Indicator Summary" is sufficient.
    # For now, let's just display the raw bullish/bearish signals if they are present.
    st.markdown("---")
    st.subheader("🚦 Current Trade Signals")
    
    latest_row_signals = generate_signals_for_row(
        df_calculated.iloc[-1],
        indicator_selection,
        normalized_weights
    )
    
    bullish_signals_raw = latest_row_signals[0]
    bearish_signals_raw = latest_row_signals[1]

    if any(bullish_signals_raw.values()):
        st.success("📈 Bullish Signals Detected:")
        for signal, fired in bullish_signals_raw.items():
            if fired:
                st.markdown(f"- **{signal}**")
    else:
        st.info("No immediate bullish signals detected.")

    if any(bearish_signals_raw.values()):
        st.error("📉 Bearish Signals Detected:")
        for signal, fired in bearish_signals_raw.items():
            if fired:
                st.markdown(f"- **{signal}**")
    else:
        st.info("No immediate bearish signals detected.")


def display_options_analysis_tab(ticker, current_stock_price, expirations, trade_direction, overall_confidence, target_price, stop_loss_price):
    """
    Displays options analysis, allowing selection of expiration date.
    Args:
        ticker (str): The stock ticker symbol.
        current_stock_price (float): The current price of the stock.
        expirations (list): List of available options expiration dates.
        trade_direction (str): The predicted trade direction ("Bullish", "Bearish", "Neutral").
        overall_confidence (float): The overall confidence score for the trade plan.
        target_price (float): The target price for the trade.
        stop_loss_price (float): The stop loss price for the trade.
    """
    # For options tab, prev_close might not be directly available without fetching df_calculated again.
    # Let's pass None for prev_close to _display_common_header as it's not crucial here.
    _display_common_header(ticker, current_stock_price, None, overall_confidence, trade_direction)
    st.subheader(f"📈 Options Analysis for {ticker}")

    if not expirations:
        st.info("No options data available for this ticker or expiration dates are missing.")
        return

    selected_expiry = st.selectbox("Select Expiration Date for Chain", expirations)

    if selected_expiry:
        calls_df, puts_df = get_options_chain(ticker, selected_expiry) # get_options_chain returns two DFs

        if not calls_df.empty or not puts_df.empty:
            st.subheader(f"Options Chain for {selected_expiry}")

            col_chain1, col_chain2 = st.columns(2)
            with col_chain1:
                st.markdown("#### Calls")
                # Display relevant call option columns
                if not calls_df.empty:
                    st.dataframe(calls_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].round(2))
                else:
                    st.info("No call options available for this expiration.")

            with col_chain2:
                st.markdown("#### Puts")
                # Display relevant put option columns
                if not puts_df.empty:
                    st.dataframe(puts_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']].round(2))
                else:
                    st.info("No put options available for this expiration.")

            st.markdown("---")
            st.subheader("Options Insights & Strategy Suggestions")
            
            # analyze_options_chain expects (calls_df, puts_df, current_price, target_price, stop_loss_price, trade_direction)
            analyzed_options = analyze_options_chain(calls_df, puts_df, current_stock_price, target_price, stop_loss_price, trade_direction)
            
            # suggest_options_strategy expects (suitable_options, trade_direction, current_price, target_price, stop_loss_price)
            suggested_strategy_result = suggest_options_strategy(analyzed_options, trade_direction, current_stock_price, target_price, stop_loss_price)
            
            # Check if suggested_strategy_result is a dictionary before accessing its keys
            if isinstance(suggested_strategy_result, dict) and suggested_strategy_result.get('type') != 'N/A': # Check if a valid strategy was suggested
                st.markdown("##### Suggested Strategy:")
                strategy_name = suggested_strategy_result.get('type', 'N/A')
                message = suggested_strategy_result.get('rationale', 'No message.') # Use 'rationale' from utils.py
                
                with st.expander(f"**{strategy_name}** - *{message}*"):
                    st.write(f"**Direction:** {trade_direction}") # Use the overall trade_direction
                    st.write(f"**Expiration:** {suggested_strategy_result.get('expiration', 'N/A')}")
                    
                    # Safely format strike and premium
                    strike_val = suggested_strategy_result.get('strike')
                    premium_val = suggested_strategy_result.get('premium')
                    
                    st.write(f"**Strike:** ${strike_val:.2f}" if strike_val is not None and pd.notna(strike_val) else "**Strike:** N/A")
                    st.write(f"**Premium:** ${premium_val:.2f}" if premium_val is not None and pd.notna(premium_val) else "**Premium:** N/A")
                    
                    # Net Debit/Credit, Max Profit/Risk, Reward/Risk are not directly returned by suggest_options_strategy
                    # in utils.py. If these are desired, they need to be added to the suggested_strategy_result dict
                    # within the suggest_options_strategy function in utils.py.
                    # For now, I'll comment them out or provide N/A.
                    # st.write(f"**Net Debit/Credit:** {suggested_strategy_result.get('Net Debit', 'N/A')}")
                    # st.write(f"**Max Profit:** {suggested_strategy_result.get('Max Profit', 'N/A')}")
                    # st.write(f"**Max Risk:** {suggested_strategy_result.get('Max Risk', 'N/A')}")
                    # st.write(f"**Reward / Risk:** {suggested_strategy_result.get('Reward / Risk', 'N/A')}")
                    
                    st.write(f"**Rationale:** {suggested_strategy_result.get('rationale', 'N/A')}")
                    
                    # Contracts are not directly returned in the current suggested_strategy_result structure.
                    # If detailed contract info is needed, it should be added to the dict in utils.py.
                    # For now, commenting out.
                    # st.markdown("###### Contracts:")
                    # contracts = suggested_strategy_result.get('Contracts', {})
                    # for action, details in contracts.items():
                    #     st.write(f"- **{action.capitalize()}**: {details.get('type', 'N/A').capitalize()} @ ${details.get('strike', 'N/A'):.2f} (Premium: ${details.get('lastPrice', 'N/A'):.2f})")

                    # Plotting Payoff Diagram for the suggested strategy
                    # The `suggest_options_strategy` in utils.py does not return `option_legs_for_chart`.
                    # This needs to be added to utils.py if you want to plot the payoff.
                    # For now, this part will not function correctly without that data.
                    # I will keep the structure but note the dependency.
                    legs_for_chart = [] # Placeholder: This needs to come from utils.py's suggested_strategy_result
                    if legs_for_chart: # Placeholder condition
                        # Define a range of stock prices around the current price and strikes
                        min_price = min([leg['strike'] for leg in legs_for_chart]) * 0.8
                        max_price = max([leg['strike'] for leg in legs_for_chart]) * 1.2
                        if current_stock_price < min_price: min_price = current_stock_price * 0.8
                        if current_stock_price > max_price: max_price = current_stock_price * 1.2

                        stock_prices_range = np.linspace(min_price, max_price, 200)
                        payoffs = calculate_payoff_from_legs(stock_prices_range, legs_for_chart)
                        payoff_fig = plot_generic_payoff_chart(stock_prices_range, payoffs, legs_for_chart, strategy_name, ticker, current_stock_price)
                        if payoff_fig:
                            st.pyplot(payoff_fig, clear_figure=True)
                            plt.close(payoff_fig)
                        else:
                            st.error("Could not generate payoff chart for suggested strategy.")
                    else:
                        st.info("Payoff chart data not available for the suggested strategy.")

            else:
                # If suggested_strategy_result is a string, display it directly as an info message
                st.info(suggested_strategy_result if isinstance(suggested_strategy_result, str) else "No specific options strategies suggested based on current analysis and sentiment.")

        else:
            st.info(f"Could not retrieve options chain for {ticker} on {selected_expiry}. Please check the ticker and date.")
    else:
        st.info("Please select an expiration date to view the options chain.")


def display_trade_log_tab(LOG_FILE, ticker, selected_timeframe, overall_confidence, current_price, prev_close, trade_direction):
    # Import os locally to ensure it's available
    import os 

    st.markdown("### 📜 Trade Log")

    # Use a unique log file name based on ticker and timeframe to avoid conflicts
    log_file_path = f"trade_log_{ticker.replace('/', '_')}_{selected_timeframe.replace(' ', '_')}.csv"

    # Load existing log
    if os.path.exists(log_file_path):
        trade_log_df = pd.read_csv(log_file_path)
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
        # Ensure current_price is a float for initial value, and at least min_value
        price_value = float(current_price) if current_price is not None and pd.notna(current_price) else 0.01 # Set to min_value if None
        # Set explicit min_value and max_value for price to prevent JSNumberBoundsException
        price = st.number_input("Price", value=price_value, format="%.2f", min_value=0.01, max_value=1000000.00)
        # Set explicit min_value and max_value for PnL
        pnl = st.number_input("PnL (if exit)", value=0.0, format="%.2f", min_value=-1000000.00, max_value=1000000.00)
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
                    "Timeframe": selected_timeframe,
                    "Confidence": f"{overall_confidence:.0f}%",
                    "Direction": trade_direction,
                    "Price": price,
                    "PnL": pnl,
                    "Notes": notes
                }
                new_entry_df = pd.DataFrame([new_entry])
                trade_log_df = pd.concat([trade_log_df, new_entry_df], ignore_index=True)
                trade_log_df.to_csv(log_file_path, index=False) # Save to the unique file
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
        # Access the scalar value from the Series before formatting
        gdp_value = latest_gdp.iloc[-1] if latest_gdp is not None and not latest_gdp.empty else None
        st.metric("Latest GDP Growth (Annualized)", f"{gdp_value:.2f}%" if gdp_value is not None else "N/A")
        st.markdown("*(Source: FRED - Gross Domestic Product)*")
    with col2:
        # Access the scalar value from the Series before formatting
        cpi_value = latest_cpi.iloc[-1] if latest_cpi is not None and not latest_cpi.empty else None
        st.metric("Latest CPI (Inflation)", f"{cpi_value:.2f}" if cpi_value is not None else "N/A")
        st.markdown("*(Source: FRED - Consumer Price Index)*")
    with col3:
        # Access the scalar value from the Series before formatting
        unemployment_value = latest_unemployment.iloc[-1] if latest_unemployment is not None and not latest_unemployment.empty else None
        st.metric("Latest Unemployment Rate", f"{unemployment_value:.2f}%" if unemployment_value is not None else "N/A")
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
    st.subheader("Analyst Recommendations (from Finviz)")
    # Access finviz_recommendation from session_state.trade_plan_result
    finviz_recom = st.session_state.trade_plan_result['sentiment_analysis'].get('finviz_recommendation')
    if finviz_recom is not None:
        st.write(f"**Average Recommendation Score:** {finviz_recom:.2f} (1.0 = Strong Buy, 5.0 = Strong Sell)")
    else:
        st.info("No analyst recommendation data available from Finviz.")

    st.subheader("News Sentiment (from Finviz)")
    # Access finviz_news_sentiment from session_state.trade_plan_result
    finviz_news_sentiment = st.session_state.trade_plan_result['sentiment_analysis'].get('finviz_news_sentiment')
    if finviz_news_sentiment is not None:
        st.write(f"**Average News Sentiment Score:** {finviz_news_sentiment:.2f}% (100% = Very Positive)")
    else:
        st.info("No news sentiment data available from Finviz.")

    st.markdown("---")
    st.info("Note: Sentiment indicators are often contrarian. Extreme fear can be a buying opportunity, and extreme complacency a selling opportunity.")


def display_backtesting_tab(df_hist, indicator_selection, normalized_weights): # Renamed from backtest_data and indicator_selection_current_run
    st.markdown("### 📈 Backtesting Results")

    # Backtesting parameters (can be made configurable in sidebar if desired)
    atr_multiplier = st.slider("ATR Multiplier for Stop Loss", 0.5, 3.0, 1.5, 0.1)
    reward_risk_ratio = st.slider("Reward/Risk Ratio for Take Profit", 1.0, 5.0, 2.0, 0.1)
    signal_threshold_percentage = st.slider("Signal Threshold Percentage", 0.1, 1.0, 0.7, 0.05)
    trade_direction_bt = st.selectbox("Backtest Trade Direction", ["long", "short"])
    exit_strategy_bt = st.selectbox("Backtest Exit Strategy", ["fixed_rr", "trailing_psar"])

    run_backtest_button = st.button("Run Backtest")

    if run_backtest_button:
        with st.spinner("Running backtest... This may take a moment for longer periods."):
            # Define a simple trade plan function for backtesting
            # This function needs to mimic the output of generate_directional_trade_plan
            # but operate on a single row (or small slice) of historical data.
            def simple_backtest_trade_plan_func(current_data_slice, indicators_selected, weights_normalized):
                # Ensure current_data_slice is not empty and has enough data for indicators
                if current_data_slice.empty or len(current_data_slice) < 200: # Minimum data for some indicators
                    return {
                        "trade_direction": "Neutral",
                        "entry_zone_start": None, "entry_zone_end": None,
                        "target_price": None, "stop_loss": None,
                        "current_price": current_data_slice['Close'].iloc[-1] if not current_data_slice.empty else None
                    }

                latest_row_bt = current_data_slice.iloc[-1]
                current_price_bt = latest_row_bt['Close']

                # Generate signals for this historical row
                bullish_signals_bt, bearish_signals_bt, signal_strength_bt = generate_signals_for_row(
                    latest_row_bt, indicators_selected, weights_normalized
                )
                
                overall_confidence_bt, direction_bt, _ = calculate_confidence_score(
                    bullish_signals_bt, bearish_signals_bt, signal_strength_bt, weights_normalized
                )

                # Simple ATR calculation for this slice
                if 'ATR' not in current_data_slice.columns or pd.isna(current_data_slice['ATR'].iloc[-1]):
                     # Recalculate ATR for the slice if not present
                    current_data_slice['ATR'] = ta.volatility.average_true_range(
                        current_data_slice['High'], current_data_slice['Low'], current_data_slice['Close'], window=14
                    )
                atr_bt = current_data_slice['ATR'].iloc[-1] if 'ATR' in current_data_slice.columns and not pd.isna(current_data_slice['ATR'].iloc[-1]) else (current_price_bt * 0.02)

                # Determine entry/exit based on backtest parameters
                target_bt = None
                stop_loss_bt = None
                entry_start_bt = None
                entry_end_bt = None

                if direction_bt == "Bullish" and overall_confidence_bt >= (signal_threshold_percentage * 100):
                    entry_start_bt = current_price_bt * (1 - 0.005) # Small buffer for entry
                    entry_end_bt = current_price_bt * (1 + 0.005)
                    target_bt = current_price_bt + (reward_risk_ratio * atr_multiplier * atr_bt)
                    stop_loss_bt = current_price_bt - (atr_multiplier * atr_bt)
                elif direction_bt == "Bearish" and overall_confidence_bt >= (signal_threshold_percentage * 100):
                    entry_start_bt = current_price_bt * (1 + 0.005)
                    entry_end_bt = current_price_bt * (1 - 0.005)
                    target_bt = current_price_bt - (reward_risk_ratio * atr_multiplier * atr_bt)
                    stop_loss_bt = current_price_bt + (atr_multiplier * atr_bt)
                
                return {
                    "trade_direction": direction_bt,
                    "overall_confidence": overall_confidence_bt,
                    "entry_zone_start": entry_start_bt,
                    "entry_zone_end": entry_end_bt,
                    "target_price": target_bt,
                    "stop_loss": stop_loss_bt,
                    "current_price": current_price_bt
                }

            # Pass all required arguments to backtest_strategy
            # The backtest_strategy function needs to be updated to accept `indicator_selection` and `normalized_weights`
            # and pass them to the `trade_plan_func` (which is `simple_backtest_trade_plan_func` here).
            # For now, let's adapt the call to match the expected signature in utils.py.
            trades_log, performance_metrics = backtest_strategy(
                df_hist.copy(), # Pass a copy to avoid modifying original df_hist
                lambda df_slice: simple_backtest_trade_plan_func(df_slice, indicator_selection, normalized_weights),
                # The lambda function ensures that simple_backtest_trade_plan_func receives the correct arguments
                atr_multiplier=atr_multiplier, # Pass these as keyword arguments if backtest_strategy expects them
                reward_risk_ratio=reward_risk_ratio,
                signal_threshold_percentage=signal_threshold_percentage,
                trade_direction_bt=trade_direction_bt,
                exit_strategy_bt=exit_strategy_bt
            )

            if performance_metrics.get("error"):
                st.warning(f"Backtest could not be completed: {performance_metrics['error']}")
                return

            st.subheader("Overall Strategy Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", performance_metrics.get('Total Trades', 0))
                st.metric("Winning Trades", performance_metrics.get('Winning Trades', 0))
            with col2:
                st.metric("Losing Trades", performance_metrics.get('Losing Trades', 0))
                st.metric("Win Rate", performance_metrics.get('Win Rate', "0.00%"))
            with col3:
                st.metric("Gross Profit", f"${performance_metrics.get('Gross Profit', 0.0):.2f}")
                st.metric("Gross Loss", f"${performance_metrics.get('Gross Loss', 0.0):.2f}")

            st.metric("Net PnL", f"${performance_metrics.get('Net PnL', 0.0):.2f}")
            st.metric("Profit Factor", performance_metrics.get('Profit Factor', "0.00"))

            # CAGR and Max Drawdown are not directly returned by backtest_strategy in utils.py
            # If you want these, you'll need to calculate them within backtest_strategy
            # or add them to the performance_metrics dictionary in utils.py.
            # For now, commenting them out to avoid errors.
            # st.metric("CAGR (Compound Annual Growth Rate)", f"{performance_metrics.get('CAGR', 0.0):.2f}%")
            # st.metric("Max Drawdown", f"{performance_metrics.get('Max Drawdown', 0.0):.2f}%")

            st.markdown("---")
            st.subheader("Strategy Parameters Used")
            st.write("The backtest was run with the following indicator selections:")
            selected_indicators_list = [
                key for key, value in indicator_selection.items() if value
            ]
            if selected_indicators_list:
                st.write(", ".join(selected_indicators_list))
            else:
                st.write("No specific indicators were selected for this backtest (or default settings were used).")
            st.write(f"ATR Multiplier: {atr_multiplier}")
            st.write(f"Reward/Risk Ratio: {reward_risk_ratio}")
            st.write(f"Signal Threshold: {signal_threshold_percentage:.0%}")
            st.write(f"Trade Direction: {trade_direction_bt.capitalize()}")
            st.write(f"Exit Strategy: {exit_strategy_bt.replace('_', ' ').title()}")


            # Display detailed trade log
            if trades_log:
                trade_log_df = pd.DataFrame(trades_log)
                st.markdown("---")
                st.subheader("Detailed Trade Log")
                # Format PnL column to 2 decimal places and ensure it's numeric
                if 'profit_loss' in trade_log_df.columns: # Changed from 'PnL' to 'profit_loss'
                    trade_log_df['profit_loss'] = pd.to_numeric(trade_log_df['profit_loss'], errors='coerce').fillna(0.0).map('${:,.2f}'.format)

                # Format 'Entry Price' and 'Exit Price' if they exist
                for col in ['price']: # Assuming 'price' column holds both entry/exit prices
                    if col in trade_log_df.columns:
                        trade_log_df[col] = pd.to_numeric(trade_log_df[col], errors='coerce').map('${:,.2f}'.format)

                st.dataframe(trade_log_df)
            else:
                st.info("No trades executed during backtesting period with the selected strategy and data.")
    else:
        st.info("Configure backtest parameters and click 'Run Backtest' to see results.")


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
        entry_criteria_details_str = row.get('Entry Criteria Details', 'N/A')
        exit_criteria_details_str = row.get('Exit Criteria Details', 'N/A')

        display_data.append({
            "Ticker": row.get('Ticker', 'N/A'),
            "Style": row.get('Trading Style', 'N/A'),
            "Confidence": row.get('Overall Confidence', 'N/A'),
            "Direction": row.get('Direction', 'N/A'), # Corrected column name from 'Trade Direction' to 'Direction'
            "Price": f"${row.get('Current Price', 'N/A')}" if pd.notna(row.get('Current Price')) else 'N/A', # Removed .2f for string
            "ATR": f"{row.get('ATR', 'N/A')}" if pd.notna(row.get('ATR')) else 'N/A', # Removed .2f for string
            "Target": f"${row.get('Target Price', 'N/A')}" if pd.notna(row.get('Target Price')) else 'N/A', # Removed .2f for string
            "Stop Loss": f"${row.get('Stop Loss', 'N/A')}" if pd.notna(row.get('Stop Loss')) else 'N/A', # Removed .2f for string
            "Entry Zone": f"{row.get('Entry Zone Start', 'N/A')} - {row.get('Entry Zone End', 'N/A')}", # Combine start/end
            "R/R": f"{row.get('Reward/Risk', 'N/A')}",
            "Pivot (P)": f"${row.get('Pivot (P)', 'N/A')}" if pd.notna(row.get('Pivot (P)')) else 'N/A', # Removed .2f for string
            "R1": f"${row.get('Resistance 1 (R1)', 'N/A')}" if pd.notna(row.get('Resistance 1 (R1)')) else 'N/A', # Removed .2f for string
            "S1": f"${row.get('Support 1 (S1)', 'N/A')}" if pd.notna(row.get('Support 1 (S1)')) else 'N/A', # Removed .2f for string
            "R2": f"${row.get('Resistance 2 (R2)', 'N/A')}" if pd.notna(row.get('Resistance 2 (R2)')) else 'N/A', # Removed .2f for string
            "S2": f"${row.get('Support 2 (S2)', 'N/A')}" if pd.notna(row.get('Support 2 (S2)')) else 'N/A', # Removed .2f for string
            "Entry Details": entry_criteria_details_str, # This will be expanded
            "Exit Details": exit_criteria_details_str,   # This will be expanded
            "Rationale": row.get('Rationale', 'N/A')
        })

    # Use st.expander for each row to show details
    for item in display_data:
        with st.expander(f"**{item['Ticker']}** | {item['Style']} | Confidence: {item['Confidence']:.2f}% | Direction: {item['Direction']}"):
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
