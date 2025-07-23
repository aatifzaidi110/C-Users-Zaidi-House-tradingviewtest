# utils.py - Version 2.6

import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import nltk

# --- NLTK VADER Lexicon Download ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError: # Catch LookupError if the resource is not found
    nltk.download('vader_lexicon')

# === Data Fetching Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    """Fetches analyst recommendations and news sentiment from Finviz."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with st.spinner(f"Fetching Finviz data for {ticker}..."):
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            recom_tag = soup.find('td', text='Recom')
            analyst_recom = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
            headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
            analyzer = SentimentIntensityAnalyzer()
            compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
            return {"recom": analyst_recom, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception as e:
        st.error(f"Error fetching Finviz data: {e}", icon="🚫")
        return {"recom": "N/A", "headlines": [], "sentiment_compound": 0, "error": str(e)} # Added 'error' key


@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    """Fetches historical stock data and basic info from Yahoo Finance."""
    with st.spinner(f"Fetching {period} of {interval} data for {symbol}..."):
        stock = yf.Ticker(symbol)
        try: # Added try-except for yfinance data fetching
            hist = stock.history(period=period, interval=interval, auto_adjust=True)
            return (hist, stock.info) if not hist.empty else (None, None)
        except Exception as e:
            st.error(f"YFinance error fetching data for {symbol}: {e}", icon="🚫")
            return None, None


@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    """Fetches call and put options data for a given ticker and expiry."""
    with st.spinner(f"Fetching options chain for {ticker} ({expiry_date})..."):
        stock_obj = yf.Ticker(ticker)
        try: # Added try-except for yfinance option chain fetching
            options = stock_obj.option_chain(expiry_date)
            return options.calls, options.puts
        except Exception as e:
            st.warning(f"Could not fetch options chain for {ticker} on {expiry_date}: {e}", icon="⚠️")
            return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames on error

# === Indicator Calculation Functions ===
def calculate_indicators(df, is_intraday=False):
    """Calculates various technical indicators for a given DataFrame."""
    initial_len = len(df) # Keep track of initial length
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing one or more required columns for indicator calculation: {required_cols}", icon="🚫")
        return df

    df_cleaned = df.dropna(subset=['High', 'Low', 'Close', 'Volume']).copy()
    if df_cleaned.empty:
        st.warning("DataFrame is empty after dropping NaN values, cannot calculate indicators.", icon="⚠️")
        return df_cleaned
    
    if len(df_cleaned) < initial_len: # Inform if rows were dropped
        st.info(f"Dropped {initial_len - len(df_cleaned)} rows due to NaN values in OHLCV data before indicator calculation.", icon="ℹ️")

    # Use .loc for safe assignment to avoid SettingWithCopyWarning
    try:
        df_cleaned.loc[:, "EMA21"] = ta.trend.ema_indicator(df_cleaned["Close"], 21)
        df_cleaned.loc[:, "EMA50"] = ta.trend.ema_indicator(df_cleaned["Close"], 50)
        df_cleaned.loc[:, "EMA200"] = ta.trend.ema_indicator(df_cleaned["Close"], 200)
    except Exception as e:
        st.warning(f"Could not calculate EMA indicators: {e}", icon="⚠️")
        df_cleaned.loc[:, "EMA21"] = pd.NA; df_cleaned.loc[:, "EMA50"] = pd.NA; df_cleaned.loc[:, "EMA200"] = pd.NA # Ensure columns exist

    # --- Corrected Ichimoku Cloud Calculation (using direct functions) ---
    try:
        # For ta 0.11.0, Ichimoku components are often calculated using direct functions
        # which take high, low, close, and window parameters.
        df_cleaned.loc[:, 'ichimoku_conversion_line'] = ta.trend.ichimoku_conversion_line(
            high=df_cleaned['High'],
            low=df_cleaned['Low'],
            close=df_cleaned['Close'],
            window1=9,
            window2=26,
            fillna=True
        )
        df_cleaned.loc[:, 'ichimoku_base_line'] = ta.trend.ichimoku_base_line(
            high=df_cleaned['High'],
            low=df_cleaned['Low'],
            close=df_cleaned['Close'],
            window1=9, # conversion_line uses window1
            window2=26, # base_line uses window2
            fillna=True
        )
        df_cleaned.loc[:, 'ichimoku_a'] = ta.trend.ichimoku_a(
            high=df_cleaned['High'],
            low=df_cleaned['Low'],
            window1=9, # window1 for tenkan
            window2=26, # window2 for kijun
            fillna=True
        )
        df_cleaned.loc[:, 'ichimoku_b'] = ta.trend.ichimoku_b(
            high=df_cleaned['High'],
            low=df_cleaned['Low'],
            window2=26, # window2 for senkou span B
            window3=52, # window3 for senkou span B
            fillna=True
        )
    except Exception as e:
        st.warning(f"Could not calculate Ichimoku Cloud: {e}", icon="⚠️")
        # Ensure columns are added even if calculation fails to prevent KeyError later
        df_cleaned.loc[:, 'ichimoku_a'] = pd.NA
        df_cleaned.loc[:, 'ichimoku_b'] = pd.NA
        df_cleaned.loc[:, 'ichimoku_conversion_line'] = pd.NA
        df_cleaned.loc[:, 'ichimoku_base_line'] = pd.NA
    # --- End Corrected Ichimoku Cloud Calculation ---

    try: df_cleaned.loc[:, 'psar'] = ta.trend.PSARIndicator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']).psar()
    except Exception as e: st.warning(f"Could not calculate Parabolic SAR: {e}", icon="⚠️"); df_cleaned.loc[:, 'psar'] = pd.NA
    try: df_cleaned.loc[:, 'adx'] = ta.trend.ADXIndicator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']).adx()
    except Exception as e: st.warning(f"Could not calculate ADX: {e}", icon="⚠️"); df_cleaned.loc[:, 'adx'] = pd.NA
    try: df_cleaned.loc[:, "RSI"]=ta.momentum.RSIIndicator(df_cleaned["Close"]).rsi()
    except Exception as e: st.warning(f"Could not calculate RSI: {e}", icon="⚠️"); df_cleaned.loc[:, "RSI"] = pd.NA
    try: stoch = ta.momentum.StochasticOscillator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']); df_cleaned.loc[:, 'stoch_k'] = stoch.stoch(); df_cleaned.loc[:, 'stoch_d'] = stoch.stoch_signal()
    except Exception as e: st.warning(f"Could not calculate Stochastic Oscillator: {e}", icon="⚠️"); df_cleaned.loc[:, 'stoch_k'] = pd.NA; df_cleaned.loc[:, 'stoch_d'] = pd.NA
    
    # --- Corrected CCI Calculation ---
    try:
        # Explicitly use ta.momentum.CCIIndicator and then its cci() method
        cci_indicator = ta.momentum.CCIIndicator(high=df_cleaned['High'], low=df_cleaned['Low'], close=df_cleaned['Close'])
        df_cleaned.loc[:, 'cci'] = cci_indicator.cci()
    except Exception as e: st.warning(f"Could not calculate CCI: {e}", icon="⚠️"); df_cleaned.loc[:, 'cci'] = pd.NA
    # --- End Corrected CCI Calculation ---

    try: df_cleaned.loc[:, 'roc'] = ta.momentum.ROCIndicator(df_cleaned['Close']).roc()
    except Exception as e: st.warning(f"Could not calculate ROC: {e}", icon="⚠️"); df_cleaned.loc[:, 'roc'] = pd.NA
    try: df_cleaned.loc[:, 'obv'] = ta.volume.OnBalanceVolumeIndicator(df_cleaned['Close'], df_cleaned['Volume']).on_balance_volume()
    except Exception as e: st.warning(f"Could not calculate OBV: {e}", icon="⚠️"); df_cleaned.loc[:, 'obv'] = pd.NA
    
    if is_intraday:
        try: df_cleaned.loc[:, 'vwap'] = ta.volume.VolumeWeightedAveragePrice(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close'], df_cleaned['Volume']).volume_weighted_average_price()
        except Exception as e: st.warning(f"Could not calculate VWAP: {e}", icon="⚠️"); df_cleaned.loc[:, 'vwap'] = pd.NA
    else: # If not intraday, ensure 'vwap' column exists with NA to prevent KeyError in backtest
        df_cleaned.loc[:, 'vwap'] = pd.NA
    
    try: df_cleaned.loc[:, "ATR"]=ta.volatility.AverageTrueRange(df_cleaned["High"],df_cleaned["Low"],df_cleaned["Close"]).average_true_range()
    except Exception as e: st.warning(f"Could not calculate ATR: {e}", icon="⚠️"); df_cleaned.loc[:, "ATR"] = pd.NA
    
    try: bb=ta.volatility.BollingerBands(df_cleaned["Close"]); df_cleaned.loc[:, "BB_low"]=bb.bollinger_lband(); df_cleaned.loc[:, "BB_high"]=bb.bollinger_hband()
    except Exception as e: st.warning(f"Could not calculate Bollinger Bands: {e}", icon="⚠️"); df_cleaned.loc[:, "BB_low"] = pd.NA; df_cleaned.loc[:, "BB_high"] = pd.NA
    
    try: df_cleaned.loc[:, "Vol_Avg_50"]=df_cleaned["Volume"].rolling(50).mean()
    except Exception as e: st.warning(f"Could not calculate Volume Average: {e}", icon="⚠️"); df_cleaned.loc[:, "Vol_Avg_50"] = pd.NA
    
    return df_cleaned

def calculate_pivot_points(df):
    """Calculates classical pivot points for a DataFrame."""
    if df.empty or not all(col in df.columns for col in ['High', 'Low', 'Close']):
        st.warning("Insufficient data for pivot point calculation: Missing OHLC columns or empty DataFrame.", icon="⚠️") # Added warning
        return pd.DataFrame(index=df.index)

    df_pivots = pd.DataFrame(index=df.index)
    try: # Added try-except for pivot point calculation
        df_pivots['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df_pivots['R1'] = (2 * df_pivots['Pivot']) - df['Low']
        df_pivots['S1'] = (2 * df_pivots['Pivot']) - df['High']
        df_pivots['R2'] = df_pivots['Pivot'] + (df['High'] - df['Low'])
        df_pivots['S2'] = df_pivots['Pivot'] - (df['High'] - df['Low'])
        return df_pivots.shift(1)
    except Exception as e:
        st.warning(f"Error calculating pivot points: {e}", icon="⚠️")
        return pd.DataFrame(index=df.index)


# === Signal Generation ===
def generate_signals_for_row(row_data, selection, full_df=None, is_intraday=False):
    """
    Generates bullish and bearish signals for a single row of data based on selected indicators.
    Returns two dictionaries: one for bullish signals and one for bearish signals.
    """
    bullish_signals = {}
    bearish_signals = {}
    
    # Helper to safely get value and check for NaN
    def get_val(key):
        return row_data.get(key) if key in row_data and not pd.isna(row_data.get(key)) else None

    # EMA Trend
    ema21 = get_val("EMA21")
    ema50 = get_val("EMA50")
    ema200 = get_val("EMA200")
    if ema21 is not None and ema50 is not None and ema200 is not None:
        bullish_signals["Uptrend (21>50>200 EMA)"] = ema21 > ema50 and ema50 > ema200
        bearish_signals["Downtrend (21<50<200 EMA)"] = ema21 < ema50 and ema50 < ema200
    else:
        bullish_signals["Uptrend (21>50>200 EMA)"] = False
        bearish_signals["Downtrend (21<50<200 EMA)"] = False

    # Ichimoku Cloud
    ichimoku_a = get_val("ichimoku_a")
    ichimoku_b = get_val("ichimoku_b")
    close_price = get_val("Close")
    if ichimoku_a is not None and ichimoku_b is not None and close_price is not None:
        bullish_signals["Bullish Ichimoku"] = close_price > ichimoku_a and close_price > ichimoku_b
        bearish_signals["Bearish Ichimoku"] = close_price < ichimoku_a and close_price < ichimoku_b
    else:
        bullish_signals["Bullish Ichimoku"] = False
        bearish_signals["Bearish Ichimoku"] = False

    # Parabolic SAR
    psar = get_val("psar")
    if psar is not None and close_price is not None:
        bullish_signals["Bullish PSAR"] = close_price > psar
        bearish_signals["Bearish PSAR"] = close_price < psar
    else:
        bullish_signals["Bullish PSAR"] = False
        bearish_signals["Bearish PSAR"] = False

    # ADX
    adx = get_val("adx")
    if adx is not None:
        bullish_signals["Strong Trend (ADX > 25)"] = adx > 25 # ADX is non-directional, so same for both
        bearish_signals["Strong Trend (ADX > 25)"] = adx > 25 # High ADX means strong trend, could be up or down
    else:
        bullish_signals["Strong Trend (ADX > 25)"] = False
        bearish_signals["Strong Trend (ADX > 25)"] = False

    # RSI Momentum
    rsi = get_val("RSI")
    if rsi is not None:
        bullish_signals["Bullish Momentum (RSI > 50)"] = rsi > 50
        bearish_signals["Bearish Momentum (RSI < 50)"] = rsi < 50
    else:
        bullish_signals["Bullish Momentum (RSI > 50)"] = False
        bearish_signals["Bearish Momentum (RSI < 50)"] = False

    # Stochastic
    stoch_k = get_val("stoch_k")
    stoch_d = get_val("stoch_d")
    if stoch_k is not None and stoch_d is not None:
        bullish_signals["Bullish Stoch Cross"] = stoch_k > stoch_d
        bearish_signals["Bearish Stoch Cross"] = stoch_k < stoch_d
    else:
        bullish_signals["Bullish Stoch Cross"] = False
        bearish_signals["Bearish Stoch Cross"] = False

    # CCI
    cci = get_val("cci")
    if cci is not None:
        bullish_signals["Bullish CCI (>0)"] = cci > 0
        bearish_signals["Bearish CCI (<0)"] = cci < 0
    else:
        bullish_signals["Bullish CCI (>0)"] = False
        bearish_signals["Bearish CCI (<0)"] = False

    # ROC
    roc = get_val("roc")
    if roc is not None:
        bullish_signals["Positive ROC (>0)"] = roc > 0
        bearish_signals["Negative ROC (<0)"] = roc < 0
    else:
        bullish_signals["Positive ROC (>0)"] = False
        bearish_signals["Negative ROC (<0)"] = False

    # Volume Spike (Directional)
    volume = get_val("Volume")
    vol_avg_50 = get_val("Vol_Avg_50")
    prev_close = full_df['Close'].iloc[-2] if full_df is not None and len(full_df) >= 2 else None
    if volume is not None and vol_avg_50 is not None and prev_close is not None and close_price is not None:
        is_spike = volume > vol_avg_50 * 1.5
        bullish_signals["Volume Spike (Up Move)"] = is_spike and close_price > prev_close
        bearish_signals["Volume Spike (Down Move)"] = is_spike and close_price < prev_close
    else:
        bullish_signals["Volume Spike (Up Move)"] = False
        bearish_signals["Volume Spike (Down Move)"] = False

    # OBV (Directional)
    obv = get_val("obv")
    if selection.get("OBV") and obv is not None and full_df is not None and len(full_df) >= 10:
        try:
            current_index_loc = full_df.index.get_loc(row_data.name)
            if current_index_loc >= 10:
                prev_data_for_rolling = full_df.iloc[current_index_loc - 10 : current_index_loc]
                if not prev_data_for_rolling.empty and 'obv' in prev_data_for_rolling.columns:
                    obv_rolling_mean = prev_data_for_rolling['obv'].rolling(10).mean().iloc[-1]
                    if not pd.isna(obv_rolling_mean):
                        bullish_signals["OBV Rising"] = obv > obv_rolling_mean
                        bearish_signals["OBV Falling"] = obv < obv_rolling_mean
                    else:
                        bullish_signals["OBV Rising"] = False; bearish_signals["OBV Falling"] = False
                else:
                    bullish_signals["OBV Rising"] = False; bearish_signals["OBV Falling"] = False
            else:
                bullish_signals["OBV Rising"] = False; bearish_signals["OBV Falling"] = False
        except KeyError:
            bullish_signals["OBV Rising"] = False; bearish_signals["OBV Falling"] = False
    else:
        bullish_signals["OBV Rising"] = False
        bearish_signals["OBV Falling"] = False

    # VWAP (Intraday only)
    vwap = get_val("vwap")
    if is_intraday and vwap is not None and close_price is not None:
        bullish_signals["Price > VWAP"] = close_price > vwap
        bearish_signals["Price < VWAP"] = close_price < vwap
    else:
        bullish_signals["Price > VWAP"] = False
        bearish_signals["Price < VWAP"] = False

    return bullish_signals, bearish_signals

# === Backtesting Logic ===
def backtest_strategy(df_historical_calculated, selection, atr_multiplier=1.5, reward_risk_ratio=2.0, signal_threshold_percentage=0.7, trade_direction="long"):
    """
    Simulates trades based on selected indicators and a simple entry/exit strategy.
    Assumes df_historical_calculated has all indicators pre-calculated and NaNs handled.
    signal_threshold_percentage: % of selected signals that must be active for entry.
    trade_direction: 'long' or 'short' to specify which type of trades to backtest.
    """
    trades = []
    in_trade = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    min_data_points_for_backtest = 200 # Safe minimum for EMA200 and other lookbacks

    if len(df_historical_calculated) < min_data_points_for_backtest + 1:
        st.info(f"Not enough complete historical data for robust backtesting after indicator calculation. (Need at least {min_data_points_for_backtest+1} data points after NaN removal). Found: {len(df_historical_calculated)}. Please select a longer period for backtesting (e.g., 2 years daily).", icon="⚠️")
        return [], 0, 0

    required_cols_for_signals = ['Close', 'Low', 'High', 'Open', 'Volume', 'ATR'] # Always needed
    
    # Add indicator-specific columns if selected
    if selection.get("EMA Trend"): required_cols_for_signals.extend(["EMA21", "EMA50", "EMA200"])
    if selection.get("Ichimoku Cloud"): required_cols_for_signals.extend(["ichimoku_a", "ichimoku_b", "ichimoku_conversion_line", "ichimoku_base_line"])
    if selection.get("Parabolic SAR"): required_cols_for_signals.append("psar")
    if selection.get("ADX"): required_cols_for_signals.append("adx")
    if selection.get("RSI Momentum"): required_cols_for_signals.append("RSI")
    if selection.get("Stochastic"): required_cols_for_signals.extend(["stoch_k", "stoch_d"])
    if selection.get("CCI"): required_cols_for_signals.append("cci")
    if selection.get("ROC"): required_cols_for_signals.append("roc")
    if selection.get("Volume Spike"): required_cols_for_signals.append("Vol_Avg_50")
    if selection.get("OBV"): required_cols_for_signals.append("obv")
    # VWAP is for intraday, backtest is daily, so it won't be used in daily backtest signals

    required_cols_for_signals = list(set(required_cols_for_signals))
    
    missing_cols = [col for col in required_cols_for_signals if col not in df_historical_calculated.columns]
    if missing_cols:
        st.warning(f"Backtest cannot proceed: Missing required columns in historical data: {missing_cols}. This might be due to indicator calculation failures or selection of indicators not applicable to daily data.", icon="⚠️")
        return [], 0, 0

    initial_clean_len = len(df_historical_calculated)
    df_historical_calculated_clean = df_historical_calculated.dropna(subset=required_cols_for_signals).copy()
    
    if len(df_historical_calculated_clean) < initial_clean_len:
        st.info(f"Dropped {initial_clean_len - len(df_historical_calculated_clean)} rows due to NaN values in required indicator columns for backtesting.", icon="ℹ️")

    first_valid_idx = df_historical_calculated_clean[required_cols_for_signals].first_valid_index()
    if first_valid_idx is None:
        st.warning("No valid data points found after indicator calculation and cleaning for backtesting (all required columns are NaN at the start). This often happens if indicators require a lot of history or data is very sparse.", icon="⚠️")
        return [], 0, 0

    start_i = df_historical_calculated_clean.index.get_loc(first_valid_idx)
    if start_i == 0: start_i = 1 # Ensure we can look at prev_day_data

    if len(df_historical_calculated_clean) <= start_i:
        st.warning(f"Not enough data points after cleaning ({len(df_historical_calculated_clean)} available) to start backtesting from index {start_i}.", icon="⚠️")
        return [], 0, 0

    for i in range(start_i, len(df_historical_calculated_clean)):
        current_day_data = df_historical_calculated_clean.iloc[i]
        prev_day_data = df_historical_calculated_clean.iloc[i-1]

        if pd.isna(prev_day_data.get('ATR')) or prev_day_data['ATR'] <= 0 or pd.isna(current_day_data['Close']):
            continue

        if in_trade:
            # Exit conditions
            if trade_direction == "long":
                if current_day_data['Low'] <= stop_loss:
                    pnl = stop_loss - entry_price
                    trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Loss)", "Price": round(stop_loss, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                    in_trade = False
                elif current_day_data['High'] >= take_profit:
                    pnl = take_profit - entry_price
                    trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Win)", "Price": round(take_profit, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                    in_trade = False
            elif trade_direction == "short":
                if current_day_data['High'] >= stop_loss: # Stop loss for short is above entry
                    pnl = entry_price - stop_loss # PnL for short: entry - exit
                    trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Loss)", "Price": round(stop_loss, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                    in_trade = False
                elif current_day_data['Low'] <= take_profit: # Take profit for short is below entry
                    pnl = entry_price - take_profit
                    trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": "Exit (Win)", "Price": round(take_profit, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})
                    in_trade = False

        if not in_trade:
            bullish_signals, bearish_signals = generate_signals_for_row(prev_day_data, selection, df_historical_calculated_clean.iloc[:i], is_intraday=False)

            # Determine selected and fired signals for the chosen direction
            selected_directional_signals = {}
            if trade_direction == "long":
                for k, v in selection.items():
                    if v and k not in ["Bollinger Bands", "Pivot Points", "VWAP"]: # Exclude display-only and intraday-only for daily backtest
                        # Map selection key to actual signal name
                        signal_name_map = {
                            "EMA Trend": "Uptrend (21>50>200 EMA)",
                            "Ichimoku Cloud": "Bullish Ichimoku",
                            "Parabolic SAR": "Bullish PSAR",
                            "ADX": "Strong Trend (ADX > 25)",
                            "RSI Momentum": "Bullish Momentum (RSI > 50)",
                            "Stochastic": "Bullish Stoch Cross",
                            "CCI": "Bullish CCI (>0)",
                            "ROC": "Positive ROC (>0)",
                            "Volume Spike": "Volume Spike (Up Move)",
                            "OBV": "OBV Rising",
                        }
                        if signal_name_map.get(k) in bullish_signals:
                            selected_directional_signals[signal_name_map.get(k)] = bullish_signals[signal_name_map.get(k)]
            elif trade_direction == "short":
                for k, v in selection.items():
                    if v and k not in ["Bollinger Bands", "Pivot Points", "VWAP"]: # Exclude display-only and intraday-only for daily backtest
                        # Map selection key to actual signal name
                        signal_name_map = {
                            "EMA Trend": "Downtrend (21<50<200 EMA)",
                            "Ichimoku Cloud": "Bearish Ichimoku",
                            "Parabolic SAR": "Bearish PSAR",
                            "ADX": "Strong Trend (ADX > 25)",
                            "RSI Momentum": "Bearish Momentum (RSI < 50)",
                            "Stochastic": "Bearish Stoch Cross",
                            "CCI": "Bearish CCI (<0)",
                            "ROC": "Negative ROC (<0)",
                            "Volume Spike": "Volume Spike (Down Move)",
                            "OBV": "OBV Falling",
                        }
                        if signal_name_map.get(k) in bearish_signals:
                            selected_directional_signals[signal_name_map.get(k)] = bearish_signals[signal_name_map.get(k)]

            selected_indicator_count = len(selected_directional_signals)
            fired_directional_signals_count = sum(1 for v in selected_directional_signals.values() if v)
            
            # Entry condition: enough signals fired for the chosen direction
            if selected_indicator_count > 0 and (fired_directional_signals_count / selected_indicator_count) >= signal_threshold_percentage:
                entry_price = current_day_data['Open']
                if not pd.isna(prev_day_data['ATR']) and prev_day_data['ATR'] > 0:
                    if trade_direction == "long":
                        stop_loss = entry_price - (prev_day_data['ATR'] * atr_multiplier)
                        take_profit = entry_price + (prev_day_data['ATR'] * atr_multiplier * reward_risk_ratio)
                    elif trade_direction == "short":
                        stop_loss = entry_price + (prev_day_data['ATR'] * atr_multiplier) # Stop loss for short is above entry
                        take_profit = entry_price - (prev_day_data['ATR'] * atr_multiplier * reward_risk_ratio) # Take profit for short is below entry
                    
                    trades.append({"Date": current_day_data.name.strftime('%Y-%m-%d'), "Type": f"Entry ({trade_direction.capitalize()})", "Price": round(entry_price, 2)})
                    in_trade = True

    if in_trade:
        final_exit_price = df_historical_calculated_clean.iloc[-1]['Close']
        if trade_direction == "long":
            pnl = final_exit_price - entry_price
        elif trade_direction == "short":
            pnl = entry_price - final_exit_price
        trades.append({"Date": df_historical_calculated_clean.index[-1].strftime('%Y-%m-%d'), "Type": "Exit (End of Backtest)", "Price": round(final_exit_price, 2), "Entry Price": round(entry_price, 2), "PnL": round(pnl, 2)})

    wins = len([t for t in trades if t['Type'] == 'Exit (Win)'])
    losses = len([t for t in trades if t['Type'] == 'Exit (Loss)'])
    return trades, wins, losses

# === Options Strategy Logic ===
EXPERT_RATING_MAP = {"Strong Buy": 100, "Buy": 85, "Hold": 50, "N/A": 50, "Sell": 15, "Strong Sell": 0}

def convert_compound_to_100_scale(compound_score): return int((compound_score + 1) * 50)

def convert_finviz_recom_to_score(recom_str):
    """Converts Finviz numerical recommendation string to a score (0-100)."""
    try:
        recom_val = float(recom_str)
        if recom_val <= 1.5: return 100 # Strong Buy
        elif recom_val <= 2.5: return 85 # Buy
        elif recom_val <= 3.5: return 50 # Hold
        elif recom_val <= 4.5: return 15 # Sell
        else: return 0 # Strong Sell
    except ValueError:
        return 50 # Default to Hold if not a valid number

def get_moneyness(strike, current_stock_price, option_type):
    """Determines if an option is In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM)."""
    if option_type == "call":
        if current_stock_price > strike:
            return "ITM"
        elif current_stock_price == strike:
            return "ATM"
        else:
            return "OTM"
    elif option_type == "put":
        if current_stock_price < strike:
            return "ITM"
        elif current_stock_price == strike:
            return "ATM"
        else:
            return "OTM"
    return "N/A"

def analyze_options_chain(calls_df, puts_df, current_stock_price, expiry_date): # Added expiry_date parameter
    """
    Analyzes the options chain to highlight key options based on various metrics
    and provides suggestions for ITM, ATM, OTM.
    """
    analysis_results = {
        "Highest Volume Options": [],
        "Highest Open Interest Options": [],
        "Highest Implied Volatility Options": [],
        "Highest Delta Calls": [],
        "Lowest Theta Calls": [],
        "Highest Gamma Calls": [],
        "Highest Vega Calls": [],
        "ITM Call Suggestions": [],
        "ATM Call Suggestions": [],
        "OTM Call Suggestions": [],
        # Added Put specific analysis categories
        "Highest Delta Puts": [],
        "Lowest Theta Puts": [],
        "Highest Gamma Puts": [],
        "Highest Vega Puts": [],
        "ITM Put Suggestions": [],
        "ATM Put Suggestions": [],
        "OTM Put Suggestions": []
    }

    # Helper to format option summary
    def format_option_summary(option_row, opt_type, reason, expiration_date_for_summary): # Added expiration_date_for_summary
        return {
            "Type": f"{opt_type} Option",
            "Strike": option_row.get('strike', pd.NA),
            "Expiration": expiration_date_for_summary, # Use the passed expiration_date_for_summary
            "Value": option_row.get('lastPrice', pd.NA),
            "Reason": reason
        }

    # Analyze Calls
    if not calls_df.empty:
        # Add moneyness for filtering
        calls_df_copy = calls_df.copy()
        calls_df_copy['Moneyness'] = calls_df_copy.apply(
            lambda row: get_moneyness(row['strike'], current_stock_price, "call"), axis=1
        )
        
        # Highest Volume
        if 'volume' in calls_df_copy.columns and not calls_df_copy['volume'].isnull().all():
            highest_vol_call = calls_df_copy.loc[calls_df_copy['volume'].idxmax()]
            analysis_results["Highest Volume Options"].append(format_option_summary(
                highest_vol_call, "Call", f"Highest volume ({highest_vol_call['volume']:,}) indicates strong current interest.", expiry_date # Pass expiry_date
            ))

        # Highest Open Interest
        if 'openInterest' in calls_df_copy.columns and not calls_df_copy['openInterest'].isnull().all():
            highest_oi_call = calls_df_copy.loc[calls_df_copy['openInterest'].idxmax()]
            analysis_results["Highest Open Interest Options"].append(format_option_summary(
                highest_oi_call, "Call", f"Highest open interest ({highest_oi_call['openInterest']:,}) suggests significant market positioning.", expiry_date # Pass expiry_date
            ))

        # Highest Implied Volatility
        if 'impliedVolatility' in calls_df_copy.columns and not calls_df_copy['impliedVolatility'].isnull().all():
            highest_iv_call = calls_df_copy.loc[calls_df_copy['impliedVolatility'].idxmax()]
            analysis_results["Highest Implied Volatility Options"].append(format_option_summary(
                highest_iv_call, "Call", f"Highest IV ({highest_iv_call['impliedVolatility']:.2%}) indicates high expected price movement.", expiry_date # Pass expiry_date
            ))
        
        # Highest Delta Calls
        if 'delta' in calls_df_copy.columns and not calls_df_copy['delta'].isnull().all():
            highest_delta_call = calls_df_copy.loc[calls_df_copy['delta'].idxmax()]
            analysis_results["Highest Delta Calls"].append(format_option_summary(
                highest_delta_call, "Call", f"Highest Delta ({highest_delta_call['delta']:.2f}) means price moves most with stock.", expiry_date # Pass expiry_date
            ))

        # Lowest Theta Calls (for buyers)
        if 'theta' in calls_df_copy.columns and not calls_df_copy['theta'].isnull().all():
            lowest_theta_call = calls_df_copy.loc[calls_df_copy['theta'].idxmax()] # Theta is typically negative, so max is closest to zero
            analysis_results["Lowest Theta Calls"].append(format_option_summary(
                lowest_theta_call, "Call", f"Lowest Theta ({lowest_theta_call['theta']:.3f}) means less time decay.", expiry_date # Pass expiry_date
            ))
        
        # Highest Gamma Calls
        if 'gamma' in calls_df_copy.columns and not calls_df_copy['gamma'].isnull().all():
            highest_gamma_call = calls_df_copy.loc[calls_df_copy['gamma'].idxmax()]
            analysis_results["Highest Gamma Calls"].append(format_option_summary(
                highest_gamma_call, "Call", f"Highest Gamma ({highest_gamma_call['gamma']:.3f}) means fastest changing Delta.", expiry_date # Pass expiry_date
            ))

        # Highest Vega Calls
        if 'vega' in calls_df_copy.columns and not calls_df_copy['vega'].isnull().all():
            highest_vega_call = calls_df_copy.loc[calls_df_copy['vega'].idxmax()]
            analysis_results["Highest Vega Calls"].append(format_option_summary(
                highest_vega_call, "Call", f"Highest Vega ({highest_vega_call['vega']:.3f}) means most sensitive to IV changes.", expiry_date # Pass expiry_date
            ))

        # ITM Call Suggestions
        itm_calls = calls_df_copy[calls_df_copy['Moneyness'] == 'ITM'].sort_values(by='strike', ascending=False)
        if not itm_calls.empty:
            best_itm = itm_calls.iloc[0]
            analysis_results["ITM Call Suggestions"].append(format_option_summary(
                best_itm, "Call", "Deep ITM calls offer high delta, behaving more like stock.", expiry_date # Pass expiry_date
            ))
        
        # ATM Call Suggestions
        atm_calls = calls_df_copy[calls_df_copy['Moneyness'] == 'ATM']
        if not atm_calls.empty:
            best_atm = atm_calls.iloc[0]
            analysis_results["ATM Call Suggestions"].append(format_option_summary(
                best_atm, "Call", "ATM calls balance cost and directional exposure.", expiry_date # Pass expiry_date
            ))

        # OTM Call Suggestions
        otm_calls = calls_df_copy[calls_df_copy['Moneyness'] == 'OTM'].sort_values(by='strike', ascending=True)
        if not otm_calls.empty:
            best_otm = otm_calls.iloc[0]
            analysis_results["OTM Call Suggestions"].append(format_option_summary(
                best_otm, "Call", "OTM calls are cheaper and offer high leverage, but lower probability.", expiry_date # Pass expiry_date
            ))

    # Analyze Puts (similar logic, but for puts)
    if not puts_df.empty:
        # Add moneyness for filtering
        puts_df_copy = puts_df.copy()
        puts_df_copy['Moneyness'] = puts_df_copy.apply(
            lambda row: get_moneyness(row['strike'], current_stock_price, "put"), axis=1
        )

        # Highest Volume (Puts)
        if 'volume' in puts_df_copy.columns and not puts_df_copy['volume'].isnull().all():
            highest_vol_put = puts_df_copy.loc[puts_df_copy['volume'].idxmax()]
            analysis_results["Highest Volume Options"].append(format_option_summary(
                highest_vol_put, "Put", f"Highest volume ({highest_vol_put['volume']:,}) indicates strong current interest.", expiry_date # Pass expiry_date
            ))

        # Highest Open Interest (Puts)
        if 'openInterest' in puts_df_copy.columns and not puts_df_copy['openInterest'].isnull().all():
            highest_oi_put = puts_df_copy.loc[puts_df_copy['openInterest'].idxmax()]
            analysis_results["Highest Open Interest Options"].append(format_option_summary(
                highest_oi_put, "Put", f"Highest open interest ({highest_oi_put['openInterest']:,}) suggests significant market positioning.", expiry_date # Pass expiry_date
            ))

        # Highest Implied Volatility (Puts)
        if 'impliedVolatility' in puts_df_copy.columns and not puts_df_copy['impliedVolatility'].isnull().all():
            highest_iv_put = puts_df_copy.loc[puts_df_copy['impliedVolatility'].idxmax()]
            analysis_results["Highest Implied Volatility Options"].append(format_option_summary(
                highest_iv_put, "Put", f"Highest IV ({highest_iv_put['impliedVolatility']:.2%}) indicates high expected price movement.", expiry_date # Pass expiry_date
            ))
        
        # Highest Delta Puts (most negative delta, i.e., closest to -1.0)
        if 'delta' in puts_df_copy.columns and not puts_df_copy['delta'].isnull().all():
            # For puts, highest delta (in magnitude) is the most negative
            highest_delta_put = puts_df_copy.loc[puts_df_copy['delta'].idxmin()] 
            analysis_results["Highest Delta Puts"].append(format_option_summary(
                highest_delta_put, "Put", f"Highest Delta ({highest_delta_put['delta']:.2f}) means price moves most with stock (inversely).", expiry_date # Pass expiry_date
            ))

        # Lowest Theta Puts (for buyers)
        if 'theta' in puts_df_copy.columns and not puts_df_copy['theta'].isnull().all():
            lowest_theta_put = puts_df_copy.loc[puts_df_copy['theta'].idxmax()]
            analysis_results["Lowest Theta Puts"].append(format_option_summary(
                lowest_theta_put, "Put", f"Lowest Theta ({lowest_theta_put['theta']:.3f}) means less time decay.", expiry_date # Pass expiry_date
            ))

        # Highest Gamma Puts
        if 'gamma' in puts_df_copy.columns and not puts_df_copy['gamma'].isnull().all():
            highest_gamma_put = puts_df_copy.loc[puts_df_copy['gamma'].idxmax()]
            analysis_results["Highest Gamma Puts"].append(format_option_summary(
                highest_gamma_put, "Put", f"Highest Gamma ({highest_gamma_put['gamma']:.3f}) means fastest changing Delta.", expiry_date # Pass expiry_date
            ))

        # Highest Vega Puts
        if 'vega' in puts_df_copy.columns and not puts_df_copy['vega'].isnull().all():
            highest_vega_put = puts_df_copy.loc[puts_df_copy['vega'].idxmax()]
            analysis_results["Highest Vega Puts"].append(format_option_summary(
                highest_vega_put, "Put", f"Highest Vega ({highest_vega_put['vega']:.3f}) means most sensitive to IV changes.", expiry_date # Pass expiry_date
            ))

        # ITM Put Suggestions
        itm_puts = puts_df_copy[puts_df_copy['Moneyness'] == 'ITM'].sort_values(by='strike', ascending=True)
        if not itm_puts.empty:
            best_itm_put = itm_puts.iloc[0]
            analysis_results["ITM Put Suggestions"].append(format_option_summary(
                best_itm_put, "Put", "Deep ITM puts offer high delta, behaving more like stock (inversely).", expiry_date # Pass expiry_date
            ))
        
        # ATM Put Suggestions
        atm_puts = puts_df_copy[puts_df_copy['Moneyness'] == 'ATM']
        if not atm_puts.empty:
            best_atm_put = atm_puts.iloc[0]
            analysis_results["ATM Put Suggestions"].append(format_option_summary(
                best_atm_put, "Put", "ATM puts balance cost and directional exposure.", expiry_date # Pass expiry_date
            ))

        # OTM Put Suggestions
        otm_puts = puts_df_copy[puts_df_copy['Moneyness'] == 'OTM'].sort_values(by='strike', ascending=False)
        if not otm_puts.empty:
            best_otm_put = otm_puts.iloc[0]
            analysis_results["OTM Put Suggestions"].append(format_option_summary(
                best_otm_put, "Put", "OTM puts are cheaper and offer high leverage, but lower probability.", expiry_date # Pass expiry_date
            ))

    return analysis_results


def generate_directional_trade_plan(current_price, atr_value, trade_direction, timeframe, period_interval):
    """
    Generates a dynamic stock trade plan (entry, target, stop) based on direction and timeframe.
    """
    if pd.isna(atr_value) or atr_value <= 0:
        return {"status": "error", "message": "ATR data not available or invalid for trade plan generation."}

    # Define Reward/Risk ratios based on timeframe (can be customized)
    # These are illustrative and can be fine-tuned based on backtesting
    if timeframe == "Scalp Trading":
        reward_risk_ratio = 1.5
        risk_multiplier = 0.5 # Smaller risk for scalping
    elif timeframe == "Day Trading":
        reward_risk_ratio = 2.0
        risk_multiplier = 1.0 # Standard 1 ATR risk
    elif timeframe == "Swing Trading":
        reward_risk_ratio = 2.5
        risk_multiplier = 1.5 # Larger risk for wider swings
    elif timeframe == "Position Trading":
        reward_risk_ratio = 3.0
        risk_multiplier = 2.0 # Even larger risk for long-term positions
    else: # Default
        reward_risk_ratio = 2.0
        risk_multiplier = 1.0

    entry_buffer_percent = 0.001 # 0.1% buffer around current price for entry zone

    if trade_direction == "Bullish":
        entry_zone_start = current_price * (1 - entry_buffer_percent)
        entry_zone_end = current_price * (1 + entry_buffer_percent)
        
        stop_loss_val = current_price - (atr_value * risk_multiplier)
        profit_target_val = current_price + (atr_value * risk_multiplier * reward_risk_ratio)
        
        trade_type_label = f"Bullish {timeframe.replace(' Trading', ' Trade')}"
        return {
            "status": "success",
            "direction": "Bullish",
            "label": trade_type_label,
            "entry_zone_start": entry_zone_start,
            "entry_zone_end": entry_zone_end,
            "stop_loss": stop_loss_val,
            "profit_target": profit_target_val,
            "reward_risk_ratio": reward_risk_ratio
        }
    elif trade_direction == "Bearish":
        entry_zone_start = current_price * (1 + entry_buffer_percent)
        entry_zone_end = current_price * (1 - entry_buffer_percent) # Entry above current for short
        
        stop_loss_val = current_price + (atr_value * risk_multiplier) # Stop above entry for short
        profit_target_val = current_price - (atr_value * risk_multiplier * reward_risk_ratio) # Target below entry for short
        
        trade_type_label = f"Bearish {timeframe.replace(' Trading', ' Trade')}"
        return {
            "status": "success",
            "direction": "Bearish",
            "label": trade_type_label,
            "entry_zone_start": entry_zone_end, # For display, start low, end high
            "entry_zone_end": entry_zone_start,
            "stop_loss": stop_loss_val,
            "profit_target": profit_target_val,
            "reward_risk_ratio": reward_risk_ratio
        }
    else: # Neutral
        return {"status": "warning", "message": "No strong directional bias detected for a stock trade plan. Consider a neutral strategy or further analysis."}


def generate_option_trade_plan(ticker, confidence, stock_price, expirations, trade_direction): # Added trade_direction
    """Generates an options trade plan based on confidence, stock price, expirations, and trade direction."""
    
    # If confidence is too low, or if it's neutral, don't recommend options
    if confidence < 60 or trade_direction == "Neutral":
        return {"status": "warning", "message": "Confidence score is too low or sentiment is neutral. No options trade is recommended."}
    
    today = datetime.now()
    suitable_expirations = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
        days_to_expiry = (exp_date - today).days
        if 45 <= days_to_expiry <= 365: # Up to 1 year
            suitable_expirations.append((days_to_expiry, exp_str))
            
    target_exp_date = None

    if suitable_expirations:
        suitable_expirations.sort()
        target_exp_date = suitable_expirations[0][1]
    else:
        # Fallback: If no expirations found within 45-365 days, try the very next available one
        if expirations:
            all_exp_dates = sorted([datetime.strptime(e, '%Y-%m-%d') for e in expirations])
            if all_exp_dates:
                target_exp_date = all_exp_dates[0].strftime('%Y-%m-%d')
                st.info(f"No expirations found between 45-365 days. Falling back to nearest available expiration: {target_exp_date}", icon="ℹ️")
            else:
                return {"status": "warning", "message": "No expiration dates available for this ticker at all."}
        else:
            return {"status": "warning", "message": "No expiration dates available for this ticker at all."}

    if target_exp_date is None:
        return {"status": "error", "message": "Could not determine a valid expiration date for options analysis."}

    calls, puts = get_options_chain(ticker, target_exp_date)
    if calls.empty and puts.empty:
        return {"status": "error", "message": f"No options found for {target_exp_date}."}

    strategy = "N/A"
    reason = ""
    recommended_option = pd.Series() # Initialize as empty Series

    if trade_direction == "Bullish":
        if confidence >= 75: # High bullish confidence -> Bull Call Spread or ITM Call
            # Attempt Bull Call Spread
            if 'delta' in calls.columns:
                itm_calls = calls[(calls['inTheMoney']) & (calls['delta'] > 0.60)].sort_values(by='strike', ascending=False)
            else:
                st.warning("Delta data not available for calls. Filtering ITM calls by 'inTheMoney' only.", icon="⚠️")
                itm_calls = calls[calls['inTheMoney']].sort_values(by='strike', ascending=False)

            if not itm_calls.empty:
                buy_leg = itm_calls.iloc[0]
                otm_calls_for_spread = calls[(calls['strike'] > buy_leg['strike']) & (calls['inTheMoney'] == False)].sort_values(by='strike', ascending=True)
                otm_calls_for_spread = otm_calls_for_spread[(otm_calls_for_spread['volume'] > 5) | (otm_calls_for_spread['openInterest'] > 10)]

                if not otm_calls_for_spread.empty and len(otm_calls_for_spread) > 0:
                    sell_leg = None
                    for j in range(1, min(len(otm_calls_for_spread), 5)):
                        if otm_calls_for_spread.iloc[j]['strike'] > buy_leg['strike'] * 1.02:
                            sell_leg = otm_calls_for_spread.iloc[j]
                            break
                    
                    if sell_leg is not None:
                        strategy = "Bull Call Spread"
                        reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move with defined risk. A Bull Call Spread limits both upside and downside, reducing premium cost."
                        
                        buy_price = buy_leg.get('ask', buy_leg.get('lastPrice', 0))
                        sell_price = sell_leg.get('bid', sell_leg.get('lastPrice', 0))

                        if buy_price > 0 and sell_price > 0: # Ensure valid prices for spread
                            spread_cost = buy_price - sell_price
                            strike_difference = sell_leg['strike'] - buy_leg['strike']
                            max_profit = (strike_difference) - spread_cost
                            max_risk = spread_cost

                            if spread_cost > 0 and max_risk > 0 and strike_difference > 0:
                                return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                                        "Buy Strike": f"${buy_leg['strike']:.2f}", "Sell Strike": f"${sell_leg['strike']:.2f}",
                                        "Net Debit": f"~${spread_cost:.2f}",
                                        "Max Profit": f"~${max_profit:.2f}", "Max Risk": f"~${max_risk:.2f}", "Reward / Risk": f"{max_profit/max_risk:.1f} to 1" if max_risk > 0 else "N/A",
                                        "Contracts": {"Buy": buy_leg, "Sell": sell_leg}}
                            else:
                                strategy = "Buy ITM Call" # Fallback
                                reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (Bull Call Spread not feasible due to invalid spread metrics)."
                                recommended_option = itm_calls.iloc[0] if not itm_calls.empty else pd.Series()
                        else:
                            strategy = "Buy ITM Call" # Fallback
                            reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (Bull Call Spread not feasible due to illiquid prices)."
                            recommended_option = itm_calls.iloc[0] if not itm_calls.empty else pd.Series()
                    else: # No suitable sell leg found
                        strategy = "Buy ITM Call"
                        reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (No suitable OTM calls for spread found)."
                        recommended_option = itm_calls.iloc[0] if not itm_calls.empty else pd.Series()
                else: # No OTM calls for spread
                    strategy = "Buy ITM Call"
                    reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (No suitable OTM calls for spread to create spread)."
                    recommended_option = itm_calls.iloc[0] if not itm_calls.empty else pd.Series()
            else: # No ITM calls for spread
                strategy = "Buy ITM Call"
                reason = f"High confidence ({confidence:.0f}% bullish) suggests a strong directional move. An ITM call (Delta > 0.60) provides good leverage with a higher probability of success. (No suitable ITM calls for spread)."
                recommended_option = calls[calls['inTheMoney']].iloc[0] if not calls[calls['inTheMoney']].empty else pd.Series()

        elif 60 <= confidence < 75: # Moderate bullish confidence -> Buy ATM Call
            strategy = "Buy ATM Call"
            reason = f"Moderate confidence ({confidence:.0f}% bullish) favors an At-the-Money call to balance cost and potential upside."
            if 'strike' in calls.columns:
                recommended_option = calls.iloc[[(calls['strike'] - stock_price).abs().idxmin()]]
                if not recommended_option.empty:
                    recommended_option = recommended_option.iloc[0]
                else:
                    st.warning("Could not find ATM call. Falling back to ITM if available.", icon="⚠️")
                    recommended_option = calls[calls['inTheMoney']].iloc[0] if not calls[calls['inTheMoney']].empty else pd.Series()
            else:
                return {"status": "error", "message": "Could not find strike price column in calls options data."}
        
        # If a single call option strategy was determined, calculate its PnL
        if strategy in ["Buy ITM Call", "Buy ATM Call"] and not recommended_option.empty:
            entry_price = recommended_option.get('ask', recommended_option.get('lastPrice'))
            if entry_price is None or entry_price == 0:
                entry_price = recommended_option.get('lastPrice')
                if entry_price is None or entry_price == 0:
                    return {"status": "error", "message": "Could not determine a valid entry price for the recommended call option."}
            
            risk_per_share = entry_price * 0.50 # Example: Risk 50% of premium
            stop_loss = entry_price - risk_per_share
            profit_target = entry_price + (risk_per_share * 2) # Example: 2:1 Reward/Risk

            return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                    "Strike": f"${recommended_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2f}",
                    "Stop-Loss": f"~${stop_loss:.2f} (50% loss)", "Profit Target": f"~${profit_target:.2f} (100% gain)",
                    "Max Risk / Share": f"${risk_per_share:.2f}", "Reward / Risk": "2 to 1", "Contract": recommended_option}

    elif trade_direction == "Bearish":
        if confidence >= 75: # High bearish confidence -> Bear Put Spread or ITM Put
            # Attempt Bear Put Spread
            if 'delta' in puts.columns:
                itm_puts = puts[(puts['inTheMoney']) & (puts['delta'] < -0.60)].sort_values(by='strike', ascending=True) # ITM puts have negative delta
            else:
                st.warning("Delta data not available for puts. Filtering ITM puts by 'inTheMoney' only.", icon="⚠️")
                itm_puts = puts[puts['inTheMoney']].sort_values(by='strike', ascending=True)

            if not itm_puts.empty:
                buy_leg = itm_puts.iloc[0]
                otm_puts_for_spread = puts[(puts['strike'] < buy_leg['strike']) & (puts['inTheMoney'] == False)].sort_values(by='strike', ascending=False)
                otm_puts_for_spread = otm_puts_for_spread[(otm_puts_for_spread['volume'] > 5) | (otm_puts_for_spread['openInterest'] > 10)]

                if not otm_puts_for_spread.empty and len(otm_puts_for_spread) > 0:
                    sell_leg = None
                    for j in range(1, min(len(otm_puts_for_spread), 5)):
                        if otm_puts_for_spread.iloc[j]['strike'] < buy_leg['strike'] * 0.98: # Sell leg strike lower than buy leg
                            sell_leg = otm_puts_for_spread.iloc[j]
                            break
                    
                    if sell_leg is not None:
                        strategy = "Bear Put Spread"
                        reason = f"High confidence ({confidence:.0f}% bearish) suggests a strong directional move with defined risk. A Bear Put Spread limits both upside and downside, reducing premium cost."
                        
                        buy_price = buy_leg.get('ask', buy_leg.get('lastPrice', 0))
                        sell_price = sell_leg.get('bid', sell_leg.get('lastPrice', 0))

                        if buy_price > 0 and sell_price > 0: # Ensure valid prices for spread
                            spread_cost = buy_price - sell_price # Net debit for put spread
                            strike_difference = buy_leg['strike'] - sell_leg['strike']
                            max_profit = (strike_difference) - spread_cost
                            max_risk = spread_cost

                            if spread_cost > 0 and max_risk > 0 and strike_difference > 0:
                                return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                                        "Buy Strike": f"${buy_leg['strike']:.2f}", "Sell Strike": f"${sell_leg['strike']:.2f}",
                                        "Net Debit": f"~${spread_cost:.2f}",
                                        "Max Profit": f"~${max_profit:.2f}", "Max Risk": f"~${max_risk:.2f}", "Reward / Risk": f"{max_profit/max_risk:.1f} to 1" if max_risk > 0 else "N/A",
                                        "Contracts": {"Buy": buy_leg, "Sell": sell_leg}}
                            else:
                                strategy = "Buy ITM Put" # Fallback
                                reason = f"High confidence ({confidence:.0f}% bearish) suggests a strong directional move. An ITM put (Delta < -0.60) provides good leverage with a higher probability of success. (Bear Put Spread not feasible due to invalid spread metrics)."
                                recommended_option = itm_puts.iloc[0] if not itm_puts.empty else pd.Series()
                        else:
                            strategy = "Buy ITM Put" # Fallback
                            reason = f"High confidence ({confidence:.0f}% bearish) suggests a strong directional move. An ITM put (Delta < -0.60) provides good leverage with a higher probability of success. (Bear Put Spread not feasible due to illiquid prices)."
                            recommended_option = itm_puts.iloc[0] if not itm_puts.empty else pd.Series()
                    else: # No suitable sell leg found
                        strategy = "Buy ITM Put"
                        reason = f"High confidence ({confidence:.0f}% bearish) suggests a strong directional move. An ITM put (Delta < -0.60) provides good leverage with a higher probability of success. (No suitable OTM puts for spread found)."
                        recommended_option = itm_puts.iloc[0] if not itm_puts.empty else pd.Series()
                else: # No OTM puts for spread
                    strategy = "Buy ITM Put"
                    reason = f"High confidence ({confidence:.0f}% bearish) suggests a strong directional move. An ITM put (Delta < -0.60) provides good leverage with a higher probability of success. (No suitable OTM puts for spread to create spread)."
                    recommended_option = itm_puts.iloc[0] if not itm_puts.empty else pd.Series()
            else: # No ITM puts for spread
                strategy = "Buy ITM Put"
                reason = f"High confidence ({confidence:.0f}% bearish) suggests a strong directional move. An ITM put (Delta < -0.60) provides good leverage with a higher probability of success. (No suitable ITM puts for spread)."
                recommended_option = puts[puts['inTheMoney']].iloc[0] if not puts[puts['inTheMoney']].empty else pd.Series()

        elif 60 <= confidence < 75: # Moderate bearish confidence -> Buy ATM Put
            strategy = "Buy ATM Put"
            reason = f"Moderate confidence ({confidence:.0f}% bearish) favors an At-the-Money put to balance cost and potential upside."
            if 'strike' in puts.columns:
                recommended_option = puts.iloc[[(puts['strike'] - stock_price).abs().idxmin()]]
                if not recommended_option.empty:
                    recommended_option = recommended_option.iloc[0]
                else:
                    st.warning("Could not find ATM put. Falling back to ITM if available.", icon="⚠️")
                    recommended_option = puts[puts['inTheMoney']].iloc[0] if not puts[puts['inTheMoney']].empty else pd.Series()
            else:
                return {"status": "error", "message": "Could not find strike price column in puts options data."}
        
        # If a single put option strategy was determined, calculate its PnL
        if strategy in ["Buy ITM Put", "Buy ATM Put"] and not recommended_option.empty:
            entry_price = recommended_option.get('ask', recommended_option.get('lastPrice'))
            if entry_price is None or entry_price == 0:
                entry_price = recommended_option.get('lastPrice')
                if entry_price is None or entry_price == 0:
                    return {"status": "error", "message": "Could not determine a valid entry price for the recommended put option."}
            
            risk_per_share = entry_price * 0.50 # Example: Risk 50% of premium
            stop_loss = entry_price + risk_per_share # Stop loss for put is above entry
            profit_target = entry_price - (risk_per_share * 2) # Target for put is below entry

            return {"status": "success", "Strategy": strategy, "Reason": reason, "Expiration": target_exp_date,
                    "Strike": f"${recommended_option['strike']:.2f}", "Entry Price": f"~${entry_price:.2f}",
                    "Stop-Loss": f"~${stop_loss:.2f} (50% loss)", "Profit Target": f"~${profit_target:.2f} (100% gain)",
                    "Max Risk / Share": f"${risk_per_share:.2f}", "Reward / Risk": "2 to 1", "Contract": recommended_option}

    return {"status": "error", "message": "No suitable option strategy found based on current confidence and direction."}


def get_options_suggestion(confidence, stock_price, calls_df, puts_df, trade_direction): # Added puts_df and trade_direction
    """Provides more specific options chain suggestions based on confidence and direction."""
    if calls_df.empty and puts_df.empty:
        return "warning", "No options available for detailed suggestion.", "", None

    if trade_direction == "Bullish":
        if confidence >= 75:
            if 'inTheMoney' in calls_df.columns and 'volume' in calls_df.columns:
                itm_calls = calls_df[calls_df['inTheMoney'] & (calls_df['volume'] > 10)]
                if not itm_calls.empty:
                    target_call = itm_calls.iloc[0]
                    return "success", f"High Confidence ({confidence:.0f}% Bullish): Consider a deep In-The-Money (ITM) call for strong directional play.", "Look for calls with high delta and good liquidity.", target_call
            return "info", f"High Confidence ({confidence:.0f}% Bullish), but specific ITM call not found.", "Consider ATM calls or further research.", None
        elif 60 <= confidence < 75:
            atm_call = calls_df.iloc[[(calls_df['strike'] - stock_price).abs().idxmin()]]
            if not atm_call.empty:
                return "info", f"Moderate Confidence ({confidence:.0f}% Bullish): An At-The-Money (ATM) call balances cost and potential upside.", "This is a good general strategy for moderate bullishness.", atm_call.iloc[0]
            return "warning", f"Moderate Confidence ({confidence:.0f}% Bullish), but ATM call not found.", "Consider OTM calls or re-evaluate.", None
    elif trade_direction == "Bearish":
        if confidence >= 75:
            if 'inTheMoney' in puts_df.columns and 'volume' in puts_df.columns:
                itm_puts = puts_df[puts_df['inTheMoney'] & (puts_df['volume'] > 10)]
                if not itm_puts.empty:
                    target_put = itm_puts.iloc[0]
                    return "success", f"High Confidence ({confidence:.0f}% Bearish): Consider a deep In-The-Money (ITM) put for strong directional play.", "Look for puts with high delta and good liquidity.", target_put
            return "info", f"High Confidence ({confidence:.0f}% Bearish), but specific ITM put not found.", "Consider ATM puts or further research.", None
        elif 60 <= confidence < 75:
            atm_put = puts_df.iloc[[(puts_df['strike'] - stock_price).abs().idxmin()]]
            if not atm_put.empty:
                return "info", f"Moderate Confidence ({confidence:.0f}% Bearish): An At-The-Money (ATM) put balances cost and potential upside.", "This is a good general strategy for moderate bearishness.", atm_put.iloc[0]
            return "warning", f"Moderate Confidence ({confidence:.0f}% Bearish), but ATM put not found.", "Consider OTM puts or re-evaluate.", None
    
    return "warning", f"Low Confidence ({confidence:.0f}%): Options trading is not recommended at this time due to low overall confidence or neutral sentiment.", "Focus on further analysis or paper trading.", None

