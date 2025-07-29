# utils.py - Final Version (with distutils workaround for Python 3.10+)
print("--- utils.py VERSION CHECK: Loading Final Version with all functions and scanner (v4.1) ---")

# IMPORTANT: Import setuptools first to provide distutils compatibility for older libraries like pandas_datareader
import setuptools
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, date, timedelta # ADDED 'date' here
import nltk

# --- NLTK VADER Lexicon Download ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# === Data Fetching Functions ===
@st.cache_data(ttl=900)
def get_finviz_data(ticker):
    """Fetches analyst recommendations and news sentiment from Finviz."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # Finviz scraping can be unstable; handle errors gracefully
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        recom_tag = soup.find('td', text='Avg. Recom')
        recom_score = None
        if recom_tag:
            recom_value = recom_tag.find_next_sibling('td').text.strip()
            # Convert to float and then to a score (e.g., 1.00 Strong Buy -> 100)
            try:
                recom_score = convert_finviz_recom_to_score(float(recom_value))
            except ValueError:
                recom_score = None # Handle cases where conversion fails

        # Fetch news sentiment
        news_table = soup.find('table', class_='fullview-news-outer')
        news_sentiment_score = None
        if news_table:
            news_headlines = []
            for row in news_table.find_all('tr'):
                link = row.find('a', class_='tab-link-news')
                if link:
                    news_headlines.append(link.text)
            
            if news_headlines:
                analyzer = SentimentIntensityAnalyzer()
                sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in news_headlines]
                # Average compound score, scaled to 0-100
                if sentiment_scores:
                    avg_sentiment = np.mean(sentiment_scores)
                    news_sentiment_score = (avg_sentiment + 1) / 2 * 100 # Normalize from -1 to 1 to 0 to 100
        
        return {"recom_score": recom_score, "news_sentiment_score": news_sentiment_score}

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Finviz data for {ticker}: {e}")
        return {"recom_score": None, "news_sentiment_score": None}
    except Exception as e:
        st.error(f"An unexpected error occurred while processing Finviz data for {ticker}: {e}")
        return {"recom_score": None, "news_sentiment_score": None}


@st.cache_data(ttl=900)
def get_data(ticker, interval, start_date, end_date):
    """
    Fetches historical stock data using yfinance.
    Includes robust error handling and data validation.
    """
    try:
        # Convert date objects to datetime objects for yfinance if they are not already
        if isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.min.time())

        print(f"Attempting to fetch {ticker} data from {start_date} to {end_date} with interval {interval}")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            print(f"No data returned for {ticker} with interval {interval} from {start_date} to {end_date}. Check ticker/dates/interval.")
            return pd.DataFrame() # Return an empty DataFrame
        
        print(f"Successfully fetched {len(df)} rows for {ticker}.")
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        st.error(f"Error fetching data for {ticker}. Please check the ticker symbol and selected date range/interval. Details: {e}")
        return pd.DataFrame() # Return an empty DataFrame

@st.cache_data(ttl=900)
def get_options_chain(ticker, expiration_date):
    """Fetches options chain for a given ticker and expiration date."""
    try:
        tk = yf.Ticker(ticker)
        # Ensure expiration_date is in 'YYYY-MM-DD' format if it's a datetime object
        if isinstance(expiration_date, datetime) or isinstance(expiration_date, date):
            expiration_date_str = expiration_date.strftime('%Y-%m-%d')
        else:
            expiration_date_str = str(expiration_date) # Assume it's already a string

        opt = tk.option_chain(expiration_date_str)
        return opt.calls, opt.puts
    except Exception as e:
        st.error(f"Error fetching options chain for {ticker} on {expiration_date}: {e}")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames on error

@st.cache_data(ttl=900)
def get_economic_data_fred(series_id, start_date, end_date):
    """
    Fetches economic data from FRED.
    Args:
        series_id (str): The FRED series ID (e.g., "GDP", "CPIAUCSL", "UNRATE").
        start_date (datetime.date): Start date for data.
        end_date (datetime.date): End date for data.
    Returns:
        pd.Series: A pandas Series containing the economic data.
    """
    # Map common names to FRED series IDs
    fred_series_map = {
        "GDP": "GDP", # Gross Domestic Product
        "CPI": "CPIAUCSL", # Consumer Price Index for All Urban Consumers: All Items
        "UNRATE": "UNRATE" # Civilian Unemployment Rate
    }
    
    actual_series_id = fred_series_map.get(series_id.upper(), series_id) # Use .upper() for robustness

    try:
        # pandas_datareader requires datetime objects for start/end
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())

        data = pdr.DataReader(actual_series_id, 'fred', start_dt, end_dt)
        if data.empty:
            print(f"No FRED data for {actual_series_id} between {start_date} and {end_date}.")
            return pd.Series(dtype=float)
        
        # FRED data often comes with a single column named after the series_id
        # We want to return a Series for consistency with previous uses
        return data[actual_series_id]
    except Exception as e:
        st.warning(f"Could not fetch FRED data for {series_id} ({actual_series_id}): {e}")
        return pd.Series(dtype=float) # Return empty Series on error

@st.cache_data(ttl=900)
def get_vix_data(start_date, end_date):
    """
    Fetches VIX (CBOE Volatility Index) historical data using yfinance.
    Args:
        start_date (datetime.date): Start date for data.
        end_date (datetime.date): End date for data.
    Returns:
        pd.DataFrame: DataFrame with VIX data, or empty DataFrame on error.
    """
    try:
        # yfinance expects datetime objects for start/end
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())

        vix_df = yf.download("^VIX", start=start_dt, end=end_dt)
        if vix_df.empty:
            print(f"No VIX data returned for {start_date} to {end_date}.")
            return pd.DataFrame()
        return vix_df
    except Exception as e:
        st.warning(f"Could not fetch VIX data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


# === Indicator Calculation Functions ===

def calculate_indicators(df, indicator_selection, is_intraday):
    """
    Calculates selected technical indicators for the given DataFrame.
    Args:
        df (pd.DataFrame): Historical stock data.
        indicator_selection (dict): Dictionary of selected indicators.
        is_intraday (bool): True if data is intraday, False otherwise.
    Returns:
        pd.DataFrame: DataFrame with calculated indicators.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying original DataFrame

    # Ensure columns are numeric, coercing errors
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df_copy.columns:
            # Check if the column is already numeric or if it can be converted
            # and that it's not a scalar value, which would cause the TypeError
            if not pd.api.types.is_numeric_dtype(df_copy[col]) and not df_copy[col].empty:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            elif df_copy[col].empty:
                # If the column is empty, it cannot be converted, so set to NaN or handle
                df_copy[col] = np.nan
        else:
            # If column is missing, add it with NaNs to prevent errors in later calculations
            df_copy[col] = np.nan

    df_copy = df_copy.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for indicator calculation.")
        return pd.DataFrame()

    # EMA
    if indicator_selection.get("EMA Trend"):
        df_copy['EMA21'] = ta.trend.ema_indicator(df_copy['Close'], window=21)
        df_copy['EMA50'] = ta.trend.ema_indicator(df_copy['Close'], window=50)
        df_copy['EMA200'] = ta.trend.ema_indicator(df_copy['Close'], window=200)

    # MACD
    if indicator_selection.get("MACD"):
        df_copy['MACD'] = ta.trend.macd(df_copy['Close'])
        df_copy['MACD_Signal'] = ta.trend.macd_signal(df_copy['Close'])
        df_copy['MACD_Hist'] = ta.trend.macd_diff(df_copy['Close'])

    # RSI
    if indicator_selection.get("RSI Momentum"):
        df_copy['RSI'] = ta.momentum.rsi(df_copy['Close'])

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands"):
        df_copy['BB_upper'], df_copy['BB_mavg'], df_copy['BB_lower'] = ta.volatility.bollinger_hband(df_copy['Close']), ta.volatility.bollinger_mavg(df_copy['Close']), ta.volatility.bollinger_lband(df_copy['Close'])

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic"):
        df_copy['Stoch_K'] = ta.momentum.stoch(df_copy['High'], df_copy['Low'], df_copy['Close'])
        df_copy['Stoch_D'] = ta.momentum.stoch_signal(df_copy['High'], df_copy['Low'], df_copy['Close'])
        
   
# Ichimoku Cloud
if indicator_selection.get("Ichimoku Cloud"):
    # Ichimoku requires longer data history, handle NaNs
    ichimoku_df = ta.trend.ichimoku_cloud(df_copy['High'], df_copy['Low'], df_copy['Close'],
                                          window1=9, window2=26, window3=52, visual=True)
    df_copy['ichimoku_base_line'] = ichimoku_df['ichimoku_base_line']
    df_copy['ichimoku_conversion_line'] = ichimoku_df['ichimoku_conversion_line']
    df_copy['ichimoku_a'] = ichimoku_df['ichimoku_a']
    df_copy['ichimoku_b'] = ichimoku_df['ichimoku_b']
    df_copy['ichimoku_leading_span_a'] = ichimoku_df['ichimoku_leading_span_a']
    df_copy['ichimoku_leading_span_b'] = ichimoku_df['ichimoku_leading_span_b']


# Parabolic SAR
if indicator_selection.get("Parabolic SAR"):
    df_copy['psar'] = ta.trend.psar(df_copy['High'], df_copy['Low'], df_copy['Close'])

# ADX
if indicator_selection.get("ADX"):
    df_copy['adx'] = ta.trend.adx(df_copy['High'], df_copy['Low'], df_copy['Close'])
    df_copy['plus_di'] = ta.trend.adx_pos(df_copy['High'], df_copy['Low'], df_copy['Close'])
    df_copy['minus_di'] = ta.trend.adx_neg(df_copy['High'], df_copy['Low'], df_copy['Close'])

# Volume Spike (simple check)
if indicator_selection.get("Volume Spike"):
    # Define a rolling window for average volume (e.g., 20 periods)
    window = 20
    if len(df_copy) >= window:
        df_copy['Volume_MA'] = df_copy['Volume'].rolling(window=window).mean()
        # A spike is typically defined as volume significantly higher than average (e.g., 1.5x)
        df_copy['Volume_Spike'] = df_copy['Volume'] > (df_copy['Volume_MA'] * 1.5)
    else:
        df_copy['Volume_Spike'] = False # Not enough data for calculation

# CCI (Commodity Channel Index)
if indicator_selection.get("CCI"):
    df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

# ROC (Rate of Change)
if indicator_selection.get("ROC"):
    df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

# OBV (On-Balance Volume)
if indicator_selection.get("OBV"):
    df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
    df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


# VWAP (Volume Weighted Average Price) - Only for intraday
if indicator_selection.get("VWAP") and is_intraday:
    # VWAP typically needs to be calculated per day for intraday data
    # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
    # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
    # For simplicity here, we'll calculate a cumulative VWAP.
    # Ensure 'Volume' column exists and is numeric
    if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
        df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
    else:
        df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
elif "VWAP" in df_copy.columns:
    df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

# Drop rows with NaN values that result from indicator calculations
df_copy = df_copy.dropna()

return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns] # Corrected from pivot_copy.columns
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D' in row:
        if row['Stoch_K'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold, K crosses above D
            bullish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (20 - row['Stoch_K']) / 20
        elif row['Stoch_K'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought, K crosses below D
            bearish_signals["Stochastic"] = True
            signal_strength["Stochastic"] = (row['Stoch_K'] - 80) / 20

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud"):
        # Ichimoku requires longer data history, handle NaNs
        ichimoku_df = ta.trend.ichimoku_cloud(row['High'], row['Low'], row['Close'],
                                              window1=9, window2=26, window3=52, visual=True)
        # Check if ichimoku_df is a DataFrame and has the expected columns
        if not ichimoku_df.empty and 'ichimoku_base_line' in ichimoku_df.columns:
            # Access values from the single row DataFrame
            if not ichimoku_df.empty:
                # Assuming ichimoku_df has only one row or we care about the last one
                ichimoku_base_line = ichimoku_df['ichimoku_base_line'].iloc[-1]
                ichimoku_conversion_line = ichimoku_df['ichimoku_conversion_line'].iloc[-1]
                ichimoku_leading_span_a = ichimoku_df['ichimoku_leading_span_a'].iloc[-1]
                ichimoku_leading_span_b = ichimoku_df['ichimoku_leading_span_b'].iloc[-1]

                # Bullish: Price above cloud, Conversion Line above Base Line, Leading Span A above Leading Span B
                if (close > ichimoku_leading_span_a and close > ichimoku_leading_span_b) and \
                   (ichimoku_conversion_line > ichimoku_base_line):
                    bullish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal
                # Bearish: Price below cloud, Conversion Line below Base Line, Leading Span A below Leading Span B
                elif (close < ichimoku_leading_span_a and close < ichimoku_leading_span_b) and \
                     (ichimoku_conversion_line < ichimoku_base_line):
                    bearish_signals["Ichimoku Cloud"] = True
                    signal_strength["Ichimoku Cloud"] = 1.0 # Strong signal


    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'psar' in row:
        if close > row['psar']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0
        elif close < row['psar']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = 1.0

    # ADX
    if indicator_selection.get("ADX") and 'adx' in row and 'plus_di' in row and 'minus_di' in row:
        if row['adx'] > 25: # Strong trend
            if row['plus_di'] > row['minus_di']:
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75 # Scale strength by ADX value
            elif row['minus_di'] > row['plus_di']:
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = (row['adx'] - 25) / 75
    
    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row:
        if row['Volume_Spike']:
            # Volume spike itself isn't directional, but can confirm other signals
            # Assign a neutral or confirming strength
            signal_strength["Volume Spike"] = 0.5 # Neutral confirmation

    # CCI (Commodity Channel Index)
    if indicator_selection.get("CCI"):
        df_copy['CCI'] = ta.trend.cci(df_copy['High'], df_copy['Low'], df_copy['Close'])

    # ROC (Rate of Change)
    if indicator_selection.get("ROC"):
        df_copy['ROC'] = ta.momentum.roc(df_copy['Close'])

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV"):
        df_copy['obv'] = ta.volume.on_balance_volume(df_copy['Close'], df_copy['Volume'])
        df_copy['obv_ema'] = ta.trend.ema_indicator(df_copy['obv'], window=10) # Corrected to OBV EMA


    # VWAP (Volume Weighted Average Price) - Only for intraday
    if indicator_selection.get("VWAP") and is_intraday:
        # VWAP typically needs to be calculated per day for intraday data
        # This implementation assumes df_copy is already intraday data for a single day or handles daily resets.
        # For multi-day intraday data, a more complex group-by-day VWAP calculation would be needed.
        # For simplicity here, we'll calculate a cumulative VWAP.
        # Ensure 'Volume' column exists and is numeric
        if 'Volume' in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy['Volume']):
            df_copy['VWAP'] = (df_copy['Close'] * df_copy['Volume']).cumsum() / df_copy['Volume'].cumsum()
        else:
            df_copy['VWAP'] = np.nan # Set to NaN if Volume is missing or not numeric
    elif "VWAP" in df_copy.columns:
        df_copy = df_copy.drop(columns=['VWAP']) # Drop if not intraday and VWAP was somehow calculated

    # Drop rows with NaN values that result from indicator calculations
    df_copy = df_copy.dropna()
    
    return df_copy


def calculate_pivot_points(df):
    """
    Calculates Classic Pivot Points (P, R1, R2, S1, S2) for each period in the DataFrame.
    Assumes df has 'High', 'Low', 'Close' columns.
    """
    df_copy = df.copy()
    
    # Ensure columns are numeric
    for col in ['High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # Drop rows with NaNs in critical columns for pivot point calculation
    df_copy = df_copy.dropna(subset=['High', 'Low', 'Close'])

    if df_copy.empty:
        print("DataFrame is empty after cleaning for pivot point calculation.")
        return pd.DataFrame() # Return empty DataFrame if no valid data

    # Calculate Pivot Point (P)
    df_copy['Pivot'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3

    # Calculate Resistance 1 (R1)
    df_copy['R1'] = (2 * df_copy['Pivot']) - df_copy['Low']

    # Calculate Support 1 (S1)
    df_copy['S1'] = (2 * df_copy['Pivot']) - df_copy['High']

    # Calculate Resistance 2 (R2)
    df_copy['R2'] = df_copy['Pivot'] + (df_copy['High'] - df_copy['Low'])

    # Calculate Support 2 (S2)
    df_copy['S2'] = df_copy['Pivot'] - (df_copy['High'] - df_copy['Low'])

    # Select only the pivot point columns to return
    pivot_cols = ['Pivot', 'R1', 'S1', 'R2', 'S2']
    # Ensure all pivot_cols exist before selecting
    existing_pivot_cols = [col for col in pivot_cols if col in df_copy.columns]
    
    return df_copy[existing_pivot_cols]


# === Signal Generation and Confidence Scoring ===

def get_indicator_summary_text(indicator_name, current_value, bullish_fired, bearish_fired):
    """
    Generates a qualitative summary text for a given indicator.
    """
    summary = f"**{indicator_name}:** "
    if current_value is not None and not pd.isna(current_value):
        if isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
            summary += f"Current Value: `{current_value:.2f}`. "
        else:
            summary += "Current Value: N/A. "
    else:
        summary += "Current Value: N/A. "

    if bullish_fired and bearish_fired:
        summary += "Conflicting signals (both bullish and bearish detected)."
    elif bullish_fired:
        summary += "Bullish signal detected."
    elif bearish_fired:
        summary += "Bearish signal detected."
    else:
        summary += "Neutral or no clear signal."
    return summary


def generate_signals_for_row(row, indicator_selection, normalized_weights):
    """
    Generates bullish and bearish signals based on the latest row of data
    and selected indicators.
    Args:
        row (pd.Series): The latest row of the DataFrame with calculated indicators.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (bullish_signals, bearish_signals, signal_strength)
               bullish_signals (dict): True/False for each bullish signal.
               bearish_signals (dict): True/False for each bearish signal.
               signal_strength (dict): Raw strength for each signal (0-1).
    """
    bullish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    bearish_signals = {
        "EMA Trend": False, "MACD": False, "RSI Momentum": False,
        "Bollinger Bands": False, "Stochastic": False, "Ichimoku Cloud": False,
        "Parabolic SAR": False, "ADX": False, "Volume Spike": False,
        "CCI": False, "ROC": False, "OBV": False, "VWAP": False,
        "Pivot Points": False
    }
    signal_strength = {
        "EMA Trend": 0.0, "MACD": 0.0, "RSI Momentum": 0.0,
        "Bollinger Bands": 0.0, "Stochastic": 0.0, "Ichimoku Cloud": 0.0,
        "Parabolic SAR": 0.0, "ADX": 0.0, "Volume Spike": 0.0,
        "CCI": 0.0, "ROC": 0.0, "OBV": 0.0, "VWAP": 0.0,
        "Pivot Points": 0.0
    }

    close = row['Close']
    
    # EMA Trend
    if indicator_selection.get("EMA Trend") and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
        if close > row['EMA21'] > row['EMA50'] > row['EMA200']:
            bullish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0
        elif close < row['EMA21'] < row['EMA50'] < row['EMA200']:
            bearish_signals["EMA Trend"] = True
            signal_strength["EMA Trend"] = 1.0

    # MACD
    if indicator_selection.get("MACD") and 'MACD' in row and 'MACD_Signal' in row and 'MACD_Hist' in row:
        if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01)) # Scale by 1% of price
        elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = min(1.0, abs(row['MACD_Hist']) / (close * 0.01))

    # RSI Momentum
    if indicator_selection.get("RSI Momentum") and 'RSI' in row:
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (30 - row['RSI']) / 30
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI Momentum"] = True
            signal_strength["RSI Momentum"] = (row['RSI'] - 70) / 30

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_upper' in row and 'BB_lower' in row:
        if close < row['BB_lower']:
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (row['BB_lower'] - close) / row['BB_lower']
        elif close > row['BB_upper']:
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = (close - row['BB_upper']) / row['BB_upper']

    # Stochastic Oscillator
    if indicator_selection.get("Stochastic") and 'Stoch_K' in row and 'Stoch_D'
