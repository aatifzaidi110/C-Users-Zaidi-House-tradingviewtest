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
        df_copy['obv_ema'] = ta.volume.volume_weighted_average_price(df_copy['High'], df_copy['Low'], df_copy['Close'], df_copy['Volume']) # This is VWAP, not OBV EMA. Correcting.
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
    existing_pivot_cols = [col for col in pivot_copy.columns] # Changed from df_copy to pivot_copy
    
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
    if indicator_selection.get("CCI") and 'CCI' in row:
        if row['CCI'] > 100: # Overbought, potential bearish reversal or strong bullish momentum
            # Can be interpreted as bullish if trend is strong, or bearish if overextended
            # For simplicity, let's say >100 is bullish momentum, < -100 is bearish momentum
            bullish_signals["CCI"] = True
            signal_strength["CCI"] = min(1.0, (row['CCI'] - 100) / 100)
        elif row['CCI'] < -100: # Oversold, potential bullish reversal or strong bearish momentum
            bearish_signals["CCI"] = True
            signal_strength["CCI"] = min(1.0, abs(row['CCI'] + 100) / 100)

    # ROC (Rate of Change)
    if indicator_selection.get("ROC") and 'ROC' in row:
        if row['ROC'] > 0: # Price is increasing
            bullish_signals["ROC"] = True
            signal_strength["ROC"] = min(1.0, row['ROC'] / 10) # Scale by 10% change
        elif row['ROC'] < 0: # Price is decreasing
            bearish_signals["ROC"] = True
            signal_strength["ROC"] = min(1.0, abs(row['ROC']) / 10)

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV") and 'obv' in row and 'obv_ema' in row:
        if row['obv'] > row['obv_ema']: # OBV rising, confirming price trend
            bullish_signals["OBV"] = True
            signal_strength["OBV"] = 0.7 # Moderate strength
        elif row['obv'] < row['obv_ema']: # OBV falling, confirming price trend
            bearish_signals["OBV"] = True
            signal_strength["OBV"] = 0.7

    # VWAP (Volume Weighted Average Price) - only for intraday, signal if price is above/below
    if indicator_selection.get("VWAP") and 'VWAP' in row and not pd.isna(row['VWAP']):
        if close > row['VWAP']:
            bullish_signals["VWAP"] = True
            signal_strength["VWAP"] = 0.8 # Strong intraday signal
        elif close < row['VWAP']:
            bearish_signals["VWAP"] = True
            signal_strength["VWAP"] = 0.8

    # Pivot Points (signals based on current price relative to P, R1, S1 etc.)
    # This assumes pivot points are calculated for the current period (e.g., daily pivots for daily data)
    if indicator_selection.get("Pivot Points") and 'Pivot' in row:
        p = row.get('Pivot')
        r1 = row.get('R1')
        s1 = row.get('S1')
        r2 = row.get('R2')
        s2 = row.get('S2')

        if p is not None:
            if close > p: # Price above pivot
                bullish_signals["Pivot Points"] = True
                signal_strength["Pivot Points"] = 0.5
            elif close < p: # Price below pivot
                bearish_signals["Pivot Points"] = True
                signal_strength["Pivot Points"] = 0.5
            
            # More nuanced signals based on resistance/support levels could be added
            # e.g., if close breaks above R1, stronger bullish signal
            if r1 is not None and close > r1:
                bullish_signals["Pivot Points"] = True # Stronger bullish
                signal_strength["Pivot Points"] = 0.8
            if s1 is not None and close < s1:
                bearish_signals["Pivot Points"] = True # Stronger bearish
                signal_strength["Pivot Points"] = 0.8

    return bullish_signals, bearish_signals, signal_strength


def calculate_confidence_score(
    latest_row, news_sentiment_score, recom_score,
    latest_gdp, latest_cpi, latest_unemployment,
    latest_vix, historical_vix_avg,
    indicator_selection, normalized_weights
):
    """
    Calculates an overall confidence score based on various factors.
    Args:
        latest_row (pd.Series): The latest row of the DataFrame with calculated indicators.
        news_sentiment_score (float): News sentiment score (0-100).
        recom_score (float): Analyst recommendation score (0-100).
        latest_gdp (float): Latest GDP growth rate.
        latest_cpi (float): Latest CPI value.
        latest_unemployment (float): Latest unemployment rate.
        latest_vix (float): Latest VIX value.
        historical_vix_avg (float): Historical average VIX.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (scores, overall_confidence, trade_direction)
    """
    scores = {
        "technical": 0, "sentiment": 0, "expert": 0,
        "economic": 0, "investor_sentiment": 0
    }
    
    # --- Technical Score ---
    bullish_tech_signals, bearish_tech_signals, tech_signal_strength = generate_signals_for_row(
        latest_row, indicator_selection, normalized_weights
    )
    
    total_tech_strength = 0
    total_bullish_tech_strength = 0
    total_bearish_tech_strength = 0

    for indicator, selected in indicator_selection.items():
        if selected:
            weight = normalized_weights.get("technical", 0) / sum(1 for s in indicator_selection.values() if s) # Distribute technical weight among selected indicators
            
            if bullish_tech_signals.get(indicator):
                total_bullish_tech_strength += tech_signal_strength.get(indicator, 0) * weight
            if bearish_tech_signals.get(indicator):
                total_bearish_tech_strength += tech_signal_strength.get(indicator, 0) * weight
    
    # Simple aggregation for now:
    if total_bullish_tech_strength > total_bearish_tech_strength:
        scores["technical"] = total_bullish_tech_strength * 100 # Max 100
    elif total_bearish_tech_strength > total_bullish_tech_strength:
        scores["technical"] = -total_bearish_tech_strength * 100 # Max -100 (for bearish)
    # If no strong direction, technical score remains 0

    # --- Sentiment Score (from Finviz news) ---
    if news_sentiment_score is not None:
        scores["sentiment"] = news_sentiment_score # Already 0-100

    # --- Expert Score (from Finviz recommendations) ---
    if recom_score is not None:
        scores["expert"] = recom_score # Already 0-100

    # --- Economic Score ---
    economic_score = calculate_economic_score(latest_gdp, latest_cpi, latest_unemployment)
    scores["economic"] = economic_score

    # --- Investor Sentiment Score (from VIX) ---
    sentiment_score_vix = calculate_sentiment_score(latest_vix, historical_vix_avg)
    scores["investor_sentiment"] = sentiment_score_vix

    # --- Calculate Overall Confidence and Direction ---
    weighted_sum = 0
    total_possible_positive_score = 0
    total_possible_negative_score = 0
    
    # Determine overall trade direction based on weighted sum of directional scores
    directional_sum = 0 # Positive for bullish, negative for bearish

    # Technical component contributes directly to directional_sum
    directional_sum += scores["technical"] * normalized_weights.get("technical", 0)

    # Other components are 0-100. Normalize to -1 to 1 for direction contribution.
    # 0-100 -> -1 to 1 (50 is neutral, 100 is +1, 0 is -1)
    
    # Sentiment (news)
    if scores["sentiment"] is not None:
        directional_sum += ((scores["sentiment"] / 100) * 2 - 1) * normalized_weights.get("sentiment", 0) * 100 # Scale to -100 to 100
        weighted_sum += scores["sentiment"] * normalized_weights.get("sentiment", 0)
        total_possible_positive_score += 100 * normalized_weights.get("sentiment", 0)
        total_possible_negative_score += 0 * normalized_weights.get("sentiment", 0) # Min score is 0

    # Expert (analyst rec)
    if scores["expert"] is not None:
        directional_sum += ((scores["expert"] / 100) * 2 - 1) * normalized_weights.get("expert", 0) * 100 # Scale to -100 to 100
        weighted_sum += scores["expert"] * normalized_weights.get("expert", 0)
        total_possible_positive_score += 100 * normalized_weights.get("expert", 0)
        total_possible_negative_score += 0 * normalized_weights.get("expert", 0)

    # Economic
    if scores["economic"] is not None:
        directional_sum += ((scores["economic"] / 100) * 2 - 1) * normalized_weights.get("economic", 0) * 100
        weighted_sum += scores["economic"] * normalized_weights.get("economic", 0)
        total_possible_positive_score += 100 * normalized_weights.get("economic", 0)
        total_possible_negative_score += 0 * normalized_weights.get("economic", 0)

    # Investor Sentiment (VIX) - often contrarian
    if scores["investor_sentiment"] is not None:
        # If VIX sentiment is high (fear), it's bullish (contrarian). If low (complacency), it's bearish.
        # So, invert the VIX sentiment score for directional sum
        directional_sum += (((100 - scores["investor_sentiment"]) / 100) * 2 - 1) * normalized_weights.get("investor_sentiment", 0) * 100
        weighted_sum += scores["investor_sentiment"] * normalized_weights.get("investor_sentiment", 0)
        total_possible_positive_score += 100 * normalized_weights.get("investor_sentiment", 0)
        total_possible_negative_score += 0 * normalized_weights.get("investor_sentiment", 0)


    overall_confidence = 0
    trade_direction = "Neutral"

    if directional_sum > 0:
        trade_direction = "Bullish"
        # Scale confidence based on how far above 0 the directional sum is, relative to max positive
        # Max directional_sum is 100 (if all components are max bullish and weights sum to 1)
        # Min directional_sum is -100 (if all components are max bearish and weights sum to 1)
        overall_confidence = (directional_sum / 100) * 100 # Scale to 0-100
    elif directional_sum < 0:
        trade_direction = "Bearish"
        overall_confidence = (abs(directional_sum) / 100) * 100 # Scale to 0-100
    else:
        trade_direction = "Neutral"
        overall_confidence = 0 # No clear direction, so confidence is 0

    # Ensure overall_confidence is between 0 and 100
    overall_confidence = max(0, min(100, overall_confidence))

    return scores, overall_confidence, trade_direction


def calculate_economic_score(gdp, cpi, unemployment):
    """
    Calculates an economic score based on GDP, CPI, and Unemployment.
    Scores are normalized to 0-100. Higher is better for stocks.
    """
    score = 50 # Start neutral

    # GDP: Higher is better. Assume typical range 0-5%.
    if gdp is not None and not pd.isna(gdp):
        if gdp > 3.0:
            score += 20 # Very strong
        elif gdp > 1.5:
            score += 10 # Moderate
        elif gdp < 0:
            score -= 20 # Contraction
        else:
            score -= 10 # Slow growth

    # CPI: Lower (stable) is better. Assume typical range 0-10%. Target ~2-3%.
    if cpi is not None and not pd.isna(cpi):
        if cpi < 2.0:
            score += 15 # Low inflation, good
        elif 2.0 <= cpi <= 3.5:
            score += 5 # Moderate, healthy inflation
        elif cpi > 5.0:
            score -= 20 # High inflation, bad
        else:
            score -= 10 # Elevated inflation

    # Unemployment: Lower is better. Assume typical range 3-10%. Target ~3-5%.
    if unemployment is not None and not pd.isna(unemployment):
        if unemployment < 4.0:
            score += 15 # Very low, strong labor market
        elif 4.0 <= unemployment <= 5.5:
            score += 5 # Healthy labor market
        elif unemployment > 7.0:
            score -= 20 # High unemployment, weak labor market
        else:
            score -= 10 # Elevated unemployment

    return max(0, min(100, score)) # Clamp between 0 and 100


def calculate_sentiment_score(latest_vix, historical_vix_avg):
    """
    Calculates an investor sentiment score based on VIX.
    Normalized to 0-100. Lower VIX (complacency) -> lower score (bearish contrarian).
    Higher VIX (fear) -> higher score (bullish contrarian).
    """
    score = 50 # Neutral

    if latest_vix is None or pd.isna(latest_vix) or historical_vix_avg is None or pd.isna(historical_vix_avg):
        return score # Return neutral if data is missing

    # VIX is a fear gauge. High VIX = high fear (often market bottoms, bullish contrarian).
    # Low VIX = complacency (often market tops, bearish contrarian).

    # Normalize VIX relative to its historical average or a typical range (e.g., 10-30)
    # A simple approach:
    # If VIX < 15: Very low fear (complacency) -> lower score
    # If 15 <= VIX <= 20: Normal range -> neutral score
    # If 20 < latest_vix <= 30: Elevated fear -> higher score
    # If latest_vix > 30: High fear -> very high score

    if latest_vix < 15:
        score = 20 # Complacency, bearish contrarian
    elif 15 <= latest_vix <= 20:
        score = 50 # Neutral
    elif 20 < latest_vix <= 30:
        score = 75 # Elevated fear, bullish contrarian
    else: # VIX > 30
        score = 90 # High fear, very bullish contrarian

    # You could also use the historical average for a more dynamic comparison:
    # if latest_vix < historical_vix_avg * 0.8: # Significantly below average
    #     score = 20
    # elif latest_vix > historical_vix_avg * 1.2: # Significantly above average
    #     score = 80
    
    return max(0, min(100, score)) # Clamp between 0 and 100


def convert_finviz_recom_to_score(recom_value):
    """Converts Finviz analyst recommendation (1.00-5.00) to a 0-100 score."""
    # 1.00 (Strong Buy) -> 100
    # 3.00 (Hold) -> 50
    # 5.00 (Strong Sell) -> 0
    # Linear interpolation: score = 100 - (recom_value - 1) * (100 / 4)
    return max(0, min(100, 100 - (recom_value - 1) * 25))


# === Options Analysis Functions ===

def get_moneyness(current_price, strike, option_type):
    """Determines if an option is In-the-Money (ITM), At-the-Money (ATM), or Out-of-the-Money (OTM)."""
    if option_type == 'call':
        if current_price > strike:
            return "ITM"
        elif current_price == strike:
            return "ATM"
        else:
            return "OTM"
    elif option_type == 'put':
        if current_price < strike:
            return "ITM"
        elif current_price == strike:
            return "ATM"
        else:
            return "OTM"
    return "N/A"

def analyze_options_chain(calls_df, puts_df, current_price):
    """
    Performs basic analysis on options chain data.
    Adds 'Moneyness' column and identifies high volume/open interest strikes.
    """
    analysis = {}

    if not calls_df.empty:
        calls_df['Moneyness'] = calls_df.apply(lambda row: get_moneyness(current_price, row['strike'], 'call'), axis=1)
        # Example: Find the call with highest open interest
        if 'openInterest' in calls_df.columns and not calls_df['openInterest'].empty:
            max_oi_call = calls_df.loc[calls_df['openInterest'].idxmax()]
            analysis['max_oi_call'] = max_oi_call.to_dict()
        if 'volume' in calls_df.columns and not calls_df['volume'].empty:
            max_vol_call = calls_df.loc[calls_df['volume'].idxmax()]
            analysis['max_vol_call'] = max_vol_call.to_dict()

    if not puts_df.empty:
        puts_df['Moneyness'] = puts_df.apply(lambda row: get_moneyness(current_price, row['strike'], 'put'), axis=1)
        # Example: Find the put with highest open interest
        if 'openInterest' in puts_df.columns and not puts_df['openInterest'].empty:
            max_oi_put = puts_df.loc[puts_df['openInterest'].idxmax()]
            analysis['max_oi_put'] = max_oi_put.to_dict()
        if 'volume' in puts_df.columns and not puts_df['volume'].empty:
            max_vol_put = puts_df.loc[puts_df['volume'].idxmax()]
            analysis['max_vol_put'] = max_vol_put.to_dict()

    return analysis


def suggest_options_strategy(ticker, confidence_score_value, current_stock_price, expirations, trade_direction):
    """
    Suggests a basic options strategy based on confidence score and trade direction.
    This is a simplified example and should be expanded for real-world use.
    """
    suggested_strategy = {
        "status": "fail",
        "message": "No strategy suggested based on current parameters.",
        "Strategy": "N/A",
        "Direction": "N/A",
        "Expiration": "N/A",
        "Net Debit": "N/A",
        "Max Profit": "N/A",
        "Max Risk": "N/A",
        "Reward / Risk": "N/A",
        "Notes": "N/A",
        "Contracts": {},
        "option_legs_for_chart": [] # For payoff chart
    }

    if not expirations:
        suggested_strategy["message"] = "No expiration dates available for options."
        return suggested_strategy

    # Prioritize shorter-term expirations for swing/day trading, longer for long-term
    # For simplicity, let's pick the first available expiration for now.
    target_expiration = expirations[0] if expirations else None

    if not target_expiration:
        suggested_strategy["message"] = "No valid expiration date found."
        return suggested_strategy

    calls_df, puts_df = get_options_chain(ticker, target_expiration)

    if calls_df.empty and puts_df.empty:
        suggested_strategy["message"] = f"No options data for {ticker} on {target_expiration}."
        return suggested_strategy

    # Strategy logic based on trade direction and confidence
    if trade_direction == "Bullish" and confidence_score_value >= 60:
        # Suggest a Call Debit Spread or Long Call
        # Find an ITM call and an OTM call for a debit spread
        itm_calls = calls_df[calls_df['strike'] < current_stock_price].sort_values(by='strike', ascending=False)
        otm_calls = calls_df[calls_df['strike'] > current_stock_price].sort_values(by='strike', ascending=True)

        if not itm_calls.empty and not otm_calls.empty:
            buy_strike = itm_calls.iloc[0]['strike']
            sell_strike = otm_calls.iloc[0]['strike']
            
            # Ensure we have valid premiums
            buy_premium = itm_calls.iloc[0]['lastPrice']
            sell_premium = otm_calls.iloc[0]['lastPrice']

            if buy_premium and sell_premium:
                net_debit = (buy_premium - sell_premium) * 100
                max_profit = (sell_strike - buy_strike - (buy_premium - sell_premium)) * 100
                max_risk = net_debit

                if net_debit > 0: # Ensure it's a debit spread
                    suggested_strategy.update({
                        "status": "success",
                        "Strategy": "Bull Call Debit Spread",
                        "Direction": "Bullish",
                        "Expiration": target_expiration,
                        "Net Debit": f"${net_debit:.2f}",
                        "Max Profit": f"${max_profit:.2f}",
                        "Max Risk": f"${max_risk:.2f}",
                        "Reward / Risk": f"{(max_profit / max_risk):.1f}:1" if max_risk > 0 else "N/A",
                        "Notes": f"Buy {ticker} Call @ ${buy_strike:.2f}, Sell {ticker} Call @ ${sell_strike:.2f}. Expects moderate bullish movement.",
                        "Contracts": {
                            "buy_call": {"type": "call", "strike": buy_strike, "lastPrice": buy_premium},
                            "sell_call": {"type": "call", "strike": sell_strike, "lastPrice": sell_premium}
                        },
                        "option_legs_for_chart": [
                            {'type': 'call', 'strike': buy_strike, 'premium': buy_premium, 'action': 'buy', 'contracts': 1},
                            {'type': 'call', 'strike': sell_strike, 'premium': sell_premium, 'action': 'sell', 'contracts': 1}
                        ]
                    })
                    return suggested_strategy
        
        # Fallback to Long Call if spread not feasible
        otm_call = otm_calls.iloc[0] if not otm_calls.empty else None
        if otm_call is not None:
            suggested_strategy.update({
                "status": "success",
                "Strategy": "Long Call",
                "Direction": "Bullish",
                "Expiration": target_expiration,
                "Net Debit": f"${otm_call['lastPrice'] * 100:.2f}",
                "Max Profit": "Unlimited",
                "Max Risk": f"${otm_call['lastPrice'] * 100:.2f}",
                "Reward / Risk": "Unlimited",
                "Notes": f"Buy {ticker} Call @ ${otm_call['strike']:.2f}. Expects strong bullish movement.",
                "Contracts": {
                    "buy_call": {"type": "call", "strike": otm_call['strike'], "lastPrice": otm_call['lastPrice']}
                },
                "option_legs_for_chart": [
                    {'type': 'call', 'strike': otm_call['strike'], 'premium': otm_call['lastPrice'], 'action': 'buy', 'contracts': 1}
                ]
            })
            return suggested_strategy


    elif trade_direction == "Bearish" and confidence_score_value >= 60:
        # Suggest a Put Debit Spread or Long Put
        itm_puts = puts_df[puts_df['strike'] > current_stock_price].sort_values(by='strike', ascending=True)
        otm_puts = puts_df[puts_df['strike'] < current_stock_price].sort_values(by='strike', ascending=False)

        if not itm_puts.empty and not otm_puts.empty:
            buy_strike = itm_puts.iloc[0]['strike']
            sell_strike = otm_puts.iloc[0]['strike']

            buy_premium = itm_puts.iloc[0]['lastPrice']
            sell_premium = otm_puts.iloc[0]['lastPrice']

            if buy_premium and sell_premium:
                net_debit = (buy_premium - sell_premium) * 100
                max_profit = (buy_strike - sell_strike - (buy_premium - sell_premium)) * 100
                max_risk = net_debit

                if net_debit > 0:
                    suggested_strategy.update({
                        "status": "success",
                        "Strategy": "Bear Put Debit Spread",
                        "Direction": "Bearish",
                        "Expiration": target_expiration,
                        "Net Debit": f"${net_debit:.2f}",
                        "Max Profit": f"${max_profit:.2f}",
                        "Max Risk": f"${max_risk:.2f}",
                        "Reward / Risk": f"{(max_profit / max_risk):.1f}:1" if max_risk > 0 else "N/A",
                        "Notes": f"Buy {ticker} Put @ ${buy_strike:.2f}, Sell {ticker} Put @ ${sell_strike:.2f}. Expects moderate bearish movement.",
                        "Contracts": {
                            "buy_put": {"type": "put", "strike": buy_strike, "lastPrice": buy_premium},
                            "sell_put": {"type": "put", "strike": sell_strike, "lastPrice": sell_premium}
                        },
                        "option_legs_for_chart": [
                            {'type': 'put', 'strike': buy_strike, 'premium': buy_premium, 'action': 'buy', 'contracts': 1},
                            {'type': 'put', 'strike': sell_strike, 'premium': sell_premium, 'action': 'sell', 'contracts': 1}
                        ]
                    })
                    return suggested_strategy
        
        # Fallback to Long Put if spread not feasible
        otm_put = otm_puts.iloc[0] if not otm_puts.empty else None
        if otm_put is not None:
            suggested_strategy.update({
                "status": "success",
                "Strategy": "Long Put",
                "Direction": "Bearish",
                "Expiration": target_expiration,
                "Net Debit": f"${otm_put['lastPrice'] * 100:.2f}",
                "Max Profit": "Unlimited",
                "Max Risk": f"${otm_put['lastPrice'] * 100:.2f}",
                "Reward / Risk": "Unlimited",
                "Notes": f"Buy {ticker} Put @ ${otm_put['strike']:.2f}. Expects strong bearish movement.",
                "Contracts": {
                    "buy_put": {"type": "put", "strike": otm_put['strike'], "lastPrice": otm_put['lastPrice']}
                },
                "option_legs_for_chart": [
                    {'type': 'put', 'strike': otm_put['strike'], 'premium': otm_put['lastPrice'], 'action': 'buy', 'contracts': 1}
                ]
            })
            return suggested_strategy

    return suggested_strategy


# === Trade Planning Functions ===

def generate_directional_trade_plan(latest_row, indicator_selection, normalized_weights):
    """
    Generates a directional trade plan (entry, target, stop-loss) based on
    technical signals and confidence.
    """
    trade_plan = {
        "direction": "Neutral",
        "confidence_score": 0,
        "entry_zone_start": None,
        "entry_zone_end": None,
        "target_price": None,
        "stop_loss": None,
        "reward_risk_ratio": None,
        "key_rationale": "No clear trade plan generated.",
        "entry_criteria_details": [],
        "exit_criteria_details": [],
        "atr": None # Add ATR to the trade plan
    }

    close = latest_row['Close']
    high = latest_row['High']
    low = latest_row['Low']

    # Calculate ATR for dynamic stop-loss and target
    # Ensure 'High', 'Low', 'Close' are available for ATR calculation
    if all(col in latest_row.index for col in ['High', 'Low', 'Close']):
        # Create a tiny DataFrame for ta.volatility.average_true_range
        # This is a workaround as ta functions are designed for DataFrames
        temp_df = pd.DataFrame([latest_row[['High', 'Low', 'Close']]])
        atr = ta.volatility.average_true_range(temp_df['High'], temp_df['Low'], temp_df['Close'], window=14).iloc[-1]
        trade_plan["atr"] = atr
    else:
        atr = None
        trade_plan["atr"] = "N/A"

    # Get signals and confidence
    scores, overall_confidence, trade_direction = calculate_confidence_score(
        latest_row,
        None, # news_sentiment_score (not available in latest_row)
        None, # recom_score (not available in latest_row)
        None, None, None, # economic data (not available in latest_row)
        None, None, # vix data (not available in latest_row)
        indicator_selection,
        normalized_weights
    )
    
    trade_plan["direction"] = trade_direction
    trade_plan["confidence_score"] = overall_confidence

    if trade_direction == "Bullish" and overall_confidence >= 50:
        # Entry: Slightly below current price or at a support level
        # Target: Resistance level or ATR-based extension
        # Stop Loss: Below a recent low or ATR-based
        
        entry_start = close * 0.99 # Slight dip
        entry_end = close * 1.01 # Slight rise
        
        if atr is not None:
            entry_start = close - (atr * 0.5)
            entry_end = close + (atr * 0.5)
            target = close + (atr * 2) # 2x ATR target
            stop_loss = close - (atr * 1.5) # 1.5x ATR stop
        else:
            target = close * 1.03 # 3% target
            stop_loss = close * 0.98 # 2% stop

        # Adjust based on pivot points if available and selected
        if indicator_selection.get("Pivot Points") and 'Pivot' in latest_row:
            p = latest_row.get('Pivot')
            s1 = latest_row.get('S1')
            r1 = latest_row.get('R1')

            if s1 is not None and close > s1: # If above S1, S1 can be support
                entry_start = min(entry_start, s1) # Entry can be near S1
            if r1 is not None and close < r1: # If below R1, R1 can be target
                target = max(target, r1) # Target can be R1
            if p is not None and close > p: # Price above pivot is bullish
                # Entry could be retest of pivot
                entry_start = min(entry_start, p)


        trade_plan.update({
            "entry_zone_start": entry_start,
            "entry_zone_end": entry_end,
            "target_price": target,
            "stop_loss": stop_loss,
            "key_rationale": f"Strong bullish signals ({overall_confidence:.0f}% confidence). Looking for entry near current price, targeting resistance/ATR extension.",
            "entry_criteria_details": [
                f"Price enters between ${entry_start:.2f} and ${entry_end:.2f}",
                "Confirmation from selected bullish indicators (e.g., EMA cross, RSI bounce from oversold)."
            ],
            "exit_criteria_details": [
                f"Price reaches target of ${target:.2f}",
                f"Price falls to stop loss at ${stop_loss:.2f}",
                "Bearish reversal signal from selected indicators."
            ]
        })

    elif trade_direction == "Bearish" and overall_confidence >= 50:
        # Entry: Slightly above current price or at a resistance level
        # Target: Support level or ATR-based extension
        # Stop Loss: Above a recent high or ATR-based

        entry_start = close * 1.01 # Slight bounce
        entry_end = close * 0.99 # Slight dip

        if atr is not None:
            entry_start = close + (atr * 0.5)
            entry_end = close - (atr * 0.5)
            target = close - (atr * 2) # 2x ATR target
            stop_loss = close + (atr * 1.5) # 1.5x ATR stop
        else:
            target = close * 0.97 # 3% target
            stop_loss = close * 1.02 # 2% stop

        # Adjust based on pivot points if available and selected
        if indicator_selection.get("Pivot Points") and 'Pivot' in latest_row:
            p = latest_row.get('Pivot')
            s1 = latest_row.get('S1')
            r1 = latest_row.get('R1')

            if r1 is not None and close < r1: # If below R1, R1 can be resistance
                entry_start = max(entry_start, r1) # Entry can be near R1
            if s1 is not None and close > s1: # If above S1, S1 can be target
                target = min(target, s1) # Target can be S1
            if p is not None and close < p: # Price below pivot is bearish
                # Entry could be retest of pivot
                entry_start = max(entry_start, p)


        trade_plan.update({
            "entry_zone_start": entry_start,
            "entry_zone_end": entry_end,
            "target_price": target,
            "stop_loss": stop_loss,
            "key_rationale": f"Strong bearish signals ({overall_confidence:.0f}% confidence). Looking for entry near current price, targeting support/ATR extension.",
            "entry_criteria_details": [
                f"Price enters between ${entry_start:.2f} and ${entry_end:.2f}",
                "Confirmation from selected bearish indicators (e.g., EMA cross, RSI fall from overbought)."
            ],
            "exit_criteria_details": [
                f"Price reaches target of ${target:.2f}",
                f"Price rises to stop loss at ${stop_loss:.2f}",
                "Bullish reversal signal from selected indicators."
            ]
        })
    
    # Calculate Reward/Risk Ratio
    if trade_plan["target_price"] is not None and trade_plan["stop_loss"] is not None and trade_plan["entry_zone_start"] is not None:
        if trade_plan["direction"] == "Bullish":
            reward = trade_plan["target_price"] - trade_plan["entry_zone_start"]
            risk = trade_plan["entry_zone_start"] - trade_plan["stop_loss"]
        else: # Bearish
            reward = trade_plan["entry_zone_start"] - trade_plan["target_price"]
            risk = trade_plan["stop_loss"] - trade_plan["entry_zone_start"]
        
        if risk > 0:
            trade_plan["reward_risk_ratio"] = reward / risk
        else:
            trade_plan["reward_risk_ratio"] = float('inf') if reward > 0 else 0 # Infinite if no risk and profit

    return trade_plan


# === Backtesting Functions ===

def backtest_strategy(df, indicator_selection, atr_multiplier, reward_risk_ratio, signal_threshold_percentage, trade_direction_bt, exit_strategy_bt):
    """
    Performs a simple backtest of the selected strategy.
    Args:
        df (pd.DataFrame): Historical data with indicators.
        indicator_selection (dict): Selected indicators for signal generation.
        atr_multiplier (float): Multiplier for ATR to set stop loss.
        reward_risk_ratio (float): Target reward/risk for take profit.
        signal_threshold_percentage (float): Minimum confidence score (0-1) to take a trade.
        trade_direction_bt (str): "long" or "short" for backtest.
        exit_strategy_bt (str): "fixed_rr" or "trailing_psar".
    Returns:
        tuple: (trades_log, performance_metrics)
    """
    trades_log = []
    in_trade = False
    entry_price = 0
    trade_entry_date = None
    trade_direction = "" # "long" or "short"
    stop_loss = 0
    take_profit = 0

    # Ensure df is sorted by date
    df = df.sort_index()

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        current_date = df.index[i]

        # Calculate ATR for current row
        atr_series = ta.volatility.average_true_range(df['High'].iloc[:i+1], df['Low'].iloc[:i+1], df['Close'].iloc[:i+1], window=14)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else np.nan

        # Generate signals and confidence for the current row
        # Note: For backtesting, we use the historical signals at each point.
        # We need to ensure that the confidence score calculation also uses historical economic/sentiment data
        # or simplify it for the backtest. For now, we'll assume a simplified confidence based on tech signals.
        
        # In a real backtest, you'd fetch/calculate economic/sentiment data for each historical date.
        # For this simplified backtest, let's just use technical signals for 'confidence'.
        bullish_signals, bearish_signals, signal_strength = generate_signals_for_row(
            current_row, indicator_selection, {} # Pass empty weights for simplicity in backtest signal_strength
        )
        
        # Calculate a simple technical confidence for backtest entry
        num_bullish = sum(1 for k, v in bullish_signals.items() if v and indicator_selection.get(k))
        num_bearish = sum(1 for k, v in bearish_signals.items() if v and indicator_selection.get(k))
        
        tech_confidence = 0
        if num_bullish + num_bearish > 0:
            if trade_direction_bt == "long":
                tech_confidence = (num_bullish / (num_bullish + num_bearish)) * 100
            elif trade_direction_bt == "short":
                tech_confidence = (num_bearish / (num_bullish + num_bearish)) * 100

        # Entry Logic
        if not in_trade:
            if trade_direction_bt == "long" and num_bullish > 0 and tech_confidence >= (signal_threshold_percentage * 100):
                in_trade = True
                entry_price = current_row['Open'] # Enter at next open
                trade_entry_date = current_date
                trade_direction = "long"
                if not np.isnan(current_atr):
                    stop_loss = entry_price - (current_atr * atr_multiplier)
                    take_profit = entry_price + (current_atr * atr_multiplier * reward_risk_ratio)
                else: # Fallback if ATR is NaN
                    stop_loss = entry_price * 0.98 # 2% stop
                    take_profit = entry_price * 1.03 # 3% target

            elif trade_direction_bt == "short" and num_bearish > 0 and tech_confidence >= (signal_threshold_percentage * 100):
                in_trade = True
                entry_price = current_row['Open'] # Enter at next open
                trade_entry_date = current_date
                trade_direction = "short"
                if not np.isnan(current_atr):
                    stop_loss = entry_price + (current_atr * atr_multiplier)
                    take_profit = entry_price - (current_atr * atr_multiplier * reward_risk_ratio)
                else: # Fallback if ATR is NaN
                    stop_loss = entry_price * 1.02 # 2% stop
                    take_profit = entry_price * 0.97 # 3% target

        # Exit Logic
        if in_trade:
            pnl = 0
            exit_reason = ""
            exit_price = 0

            if trade_direction == "long":
                # Check Stop Loss
                if current_row['Low'] <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * 1 # Assuming 1 share for simplicity
                    exit_reason = "Stop Loss Hit"
                    in_trade = False
                # Check Take Profit (Fixed R/R)
                elif exit_strategy_bt == "fixed_rr" and current_row['High'] >= take_profit:
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) * 1
                    exit_reason = "Take Profit Hit (Fixed R/R)"
                    in_trade = False
                # Check Trailing PSAR
                elif exit_strategy_bt == "trailing_psar" and 'psar' in current_row and current_row['Close'] < current_row['psar']:
                    exit_price = current_row['Close'] # Exit at close if PSAR flips
                    pnl = (exit_price - entry_price) * 1
                    exit_reason = "Trailing PSAR Exit"
                    in_trade = False

            elif trade_direction == "short":
                # Check Stop Loss
                if current_row['High'] >= stop_loss:
                    exit_price = stop_loss
                    pnl = (entry_price - exit_price) * 1
                    exit_reason = "Stop Loss Hit"
                    in_trade = False
                # Check Take Profit (Fixed R/R)
                elif exit_strategy_bt == "fixed_rr" and current_row['Low'] <= take_profit:
                    exit_price = take_profit
                    pnl = (entry_price - exit_price) * 1
                    exit_reason = "Take Profit Hit (Fixed R/R)"
                    in_trade = False
                # Check Trailing PSAR
                elif exit_strategy_bt == "trailing_psar" and 'psar' in current_row and current_row['Close'] > current_row['psar']:
                    exit_price = current_row['Close'] # Exit at close if PSAR flips
                    pnl = (entry_price - exit_price) * 1
                    exit_reason = "Trailing PSAR Exit"
                    in_trade = False
            
            # Record trade if exited
            if not in_trade and exit_reason:
                trades_log.append({
                    "Entry Date": trade_entry_date.strftime('%Y-%m-%d'),
                    "Exit Date": current_date.strftime('%Y-%m-%d'),
                    "Direction": trade_direction.capitalize(),
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL": pnl,
                    "Exit Reason": exit_reason
                })
                # Reset trade variables
                entry_price = 0
                trade_entry_date = None
                trade_direction = ""
                stop_loss = 0
                take_profit = 0

    # Calculate Performance Metrics
    total_trades = len(trades_log)
    winning_trades = sum(1 for trade in trades_log if trade['PnL'] > 0)
    losing_trades = total_trades - winning_trades
    gross_profit = sum(trade['PnL'] for trade in trades_log if trade['PnL'] > 0)
    gross_loss = sum(trade['PnL'] for trade in trades_log if trade['PnL'] < 0)
    net_pnl = gross_profit + gross_loss
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else (float('inf') if gross_profit > 0 else 0)

    performance_metrics = {
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss,
        "Net PnL": net_pnl,
        "Profit Factor": f"{profit_factor:.2f}"
    }

    return trades_log, performance_metrics


# === Stock Scanner Function ===

def run_stock_scanner(ticker_list, trading_style, min_confidence, indicator_selection, normalized_weights):
    """
    Scans a list of tickers and returns those that meet the trading style criteria
    with a minimum confidence score.
    Args:
        ticker_list (list): List of ticker symbols to scan.
        trading_style (str): "Swing", "Day", or "Long-Term".
        min_confidence (int): Minimum overall confidence score (0-100) required.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for confidence scoring.
    Returns:
        pd.DataFrame: DataFrame of qualifying stocks with trade plan details.
    """
    scanned_results = []
    
    # Determine interval and period based on trading style for scanner
    interval_map = {
        "Day": "15m",  # Intraday for day trading
        "Swing": "1d",   # Daily for swing trading
        "Long-Term": "1wk" # Weekly for long-term
    }
    period_map = {
        "Day": "7d",   # Last 7 days for intraday
        "Swing": "1y",   # Last 1 year for swing
        "Long-Term": "5y" # Last 5 years for long-term
    }

    selected_interval = interval_map.get(trading_style, "1d")
    selected_period = period_map.get(trading_style, "1y")
    is_intraday_scanner = "m" in selected_interval or "h" in selected_interval # Check if it's an intraday interval

    for ticker in ticker_list:
        try:
            # For scanner, we fetch data for a fixed period based on trading style
            # and let yfinance determine the start/end dates based on the period.
            # Convert period to start_date and end_date for get_data function
            end_date = datetime.now()
            if selected_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif selected_period == "1y":
                start_date = end_date - timedelta(days=365)
            elif selected_period == "5y":
                start_date = end_date - timedelta(days=365 * 5)
            else: # Default if period not recognized
                start_date = end_date - timedelta(days=365)


            df = get_data(ticker, selected_interval, start_date, end_date)
            
            # Ensure df is a DataFrame and not empty
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"Skipping {ticker}: No data or invalid data format from get_data.")
                continue

            df_calculated = calculate_indicators(df.copy(), indicator_selection, is_intraday_scanner)
            
            if df_calculated.empty:
                print(f"Skipping {ticker}: No calculated indicators after processing.")
                continue

            # Ensure pivot points are calculated if selected and not intraday
            last_pivot = {}
            if indicator_selection.get("Pivot Points") and not is_intraday_scanner:
                df_pivots = calculate_pivot_points(df.copy())
                if not df_pivots.empty:
                    last_pivot = df_pivots.iloc[-1].to_dict()
                else:
                    print(f"No pivot points calculated for {ticker}.")

            last_row = df_calculated.iloc[-1]
            current_price = last_row['Close']

            # Fetch Finviz data for scanner as well
            finviz_data = get_finviz_data(ticker)
            
            # Fetch economic data for scanner (using a fixed recent period if not tied to stock data dates)
            # For scanner, let's assume we use very recent economic data (e.g., last 3 months)
            economic_start_date = datetime.now() - timedelta(days=90)
            economic_end_date = datetime.now()
            latest_gdp = get_economic_data_fred("GDP", economic_start_date, economic_end_date)
            latest_cpi = get_economic_data_fred("CPI", economic_start_date, economic_end_date)
            latest_unemployment = get_economic_data_fred("UNRATE", economic_start_date, economic_end_date)

            # Fetch VIX data for scanner
            vix_data_raw = get_vix_data(economic_start_date, economic_end_date)
            vix_data = None
            if isinstance(vix_data_raw, tuple) and len(vix_data_raw) > 0:
                if isinstance(vix_data_raw[0], pd.DataFrame):
                    vix_data = vix_data_raw[0]
            elif isinstance(vix_data_raw, pd.DataFrame):
                vix_data = vix_data_raw

            latest_vix = None
            historical_vix_avg = None
            if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
                latest_vix = vix_data['Close'].iloc[-1]
                historical_vix_avg = vix_data['Close'].mean()


            scores, overall_confidence, trade_direction = calculate_confidence_score(
                last_row,
                finviz_data.get('news_sentiment_score'),
                finviz_data.get('recom_score'),
                latest_gdp.iloc[-1] if latest_gdp is not None and not latest_gdp.empty else None,
                latest_cpi.iloc[-1] if latest_cpi is not None and not latest_cpi.empty else None,
                latest_unemployment.iloc[-1] if latest_unemployment is not None and not latest_unemployment.empty else None,
                latest_vix,
                historical_vix_avg,
                indicator_selection,
                normalized_weights
            )

            # Check if confidence meets the minimum and direction matches style
            meets_confidence = overall_confidence >= min_confidence
            meets_direction = False
            if trading_style == "Swing" or trading_style == "Day":
                # For short-term, both bullish and bearish opportunities are relevant
                if trade_direction != "Neutral":
                    meets_direction = True
            elif trading_style == "Long-Term":
                # For long-term, typically look for bullish opportunities
                if trade_direction == "Bullish":
                    meets_direction = True

            if meets_confidence and meets_direction:
                trade_plan_result = generate_directional_trade_plan(last_row, indicator_selection, normalized_weights)
                
                # Append results if a valid trade plan is generated
                if trade_plan_result and trade_plan_result.get('direction') != "Neutral":
                    entry_criteria_details = trade_plan_result.get('entry_criteria_details', [])
                    exit_criteria_details = trade_plan_result.get('exit_criteria_details', [])

                    scanned_results.append({
                        "Ticker": ticker,
                        "Trading Style": trading_style,
                        "Overall Confidence": f"{overall_confidence:.0f}",
                        "Direction": trade_direction,
                        "Current Price": f"{current_price:.2f}",
                        "ATR": f"{trade_plan_result.get('atr', 'N/A'):.2f}",
                        "Entry Zone": f"${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} - ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}",
                        "Target Price": f"{trade_plan_result.get('target_price', 'N/A'):.2f}",
                        "Stop Loss": f"{trade_plan_result.get('stop_loss', 'N/A'):.2f}",
                        "Reward/Risk": f"{trade_plan_result.get('reward_risk_ratio', 'N/A'):.1f}:1",
                        "Pivot (P)": f"{last_pivot.get('Pivot', 'N/A'):.2f}",
                        "Resistance 1 (R1)": f"{last_pivot.get('R1', 'N/A'):.2f}",
                        "Resistance 2 (R2)": f"{last_pivot.get('R2', 'N/A'):.2f}",
                        "Support 1 (S1)": f"{last_pivot.get('S1', 'N/A'):.2f}",
                        "Support 2 (S2)": f"{last_pivot.get('S2', 'N/A'):.2f}",
                        "Entry Criteria Details": "\n".join(entry_criteria_details),
                        "Exit Criteria Details": "\n".join(exit_criteria_details),
                        "Rationale": trade_plan_result.get('key_rationale', '')
                    })

        except Exception as e:
            print(f"Error scanning {ticker}: {e}") # For debugging
            import traceback
            print(traceback.format_exc()) # Uncomment for full traceback during debugging
            continue
    
    # Sort results by confidence (highest first)
    if scanned_results:
        df_results = pd.DataFrame(scanned_results)
        df_results['Overall Confidence'] = pd.to_numeric(df_results['Overall Confidence'], errors='coerce')
        df_results = df_results.sort_values(by='Overall Confidence', ascending=False).reset_index(drop=True)
        return df_results
    else:
        return pd.DataFrame() # Return empty DataFrame if no results


# --- Simple Test for yfinance ---
def test_yfinance_data_fetch():
    """
    A simple function to test if yfinance can fetch data for a known ticker.
    """
    print("--- Running yfinance connectivity test ---")
    try:
        test_ticker = "SPY"
        test_df = yf.download(test_ticker, period="1d", interval="1m")
        if not test_df.empty:
            print(f"yfinance test successful: Fetched {len(test_df)} rows for {test_ticker}.")
            return True
        else:
            print(f"yfinance test failed: No data returned for {test_ticker}. This might indicate network issues or data availability problems for yfinance.")
            return False
    except Exception as e:
        print(f"yfinance test failed with an error: {e}")
        print("This often indicates network connectivity issues or problems with the yfinance library installation/access.")
        return False

# Run the test when utils.py is loaded
if __name__ == "__main__":
    test_yfinance_data_fetch()

I'm acknowledging your request to refer to the provided immersive artifact for `utils_py_final`.

The traceback you provided indicates a `TypeError` in the `calculate_indicators` function within `utils.py`, specifically at the line `df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')`. This error means that `pd.to_numeric` is receiving an argument that is not a list, tuple, 1-D array, or Series. This usually happens if `df_copy[col]` is an empty Series or a scalar value that `pd.to_numeric` cannot process in the expected way.

Additionally, I noticed a potential issue in the `calculate_pivot_points` function where `existing_pivot_cols` is defined using `pivot_copy.columns` instead of `df_copy.columns`. This would lead to a `NameError` if `pivot_copy` is not defined in that scope.

I will update the `utils.py` Canvas to address these two issues:

1.  **Robust `pd.to_numeric` handling:** I will add a check to ensure that `df_copy[col]` is not empty before attempting `pd.to_numeric`. If it's empty, it will be set to `np.nan`.
2.  **Corrected `existing_pivot_cols` definition:** I will change `pivot_copy.columns` to `df_copy.columns` in `calculate_pivot_points`.

Here's the updated `utils.py` Canvas:


```python
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
    url = f"[https://finviz.com/quote.ashx?t=](https://finviz.com/quote.ashx?t=){ticker}"
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
            # Check if the column is a Series and not empty before converting
            if isinstance(df_copy[col], pd.Series) and not df_copy[col].empty:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            else:
                # If it's not a Series or is empty, set to NaN to avoid TypeError
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
    if indicator_selection.get("CCI") and 'CCI' in row:
        if row['CCI'] > 100: # Overbought, potential bearish reversal or strong bullish momentum
            # Can be interpreted as bullish if trend is strong, or bearish if overextended
            # For simplicity, let's say >100 is bullish momentum, < -100 is bearish momentum
            bullish_signals["CCI"] = True
            signal_strength["CCI"] = min(1.0, (row['CCI'] - 100) / 100)
        elif row['CCI'] < -100: # Oversold, potential bullish reversal or strong bearish momentum
            bearish_signals["CCI"] = True
            signal_strength["CCI"] = min(1.0, abs(row['CCI'] + 100) / 100)

    # ROC (Rate of Change)
    if indicator_selection.get("ROC") and 'ROC' in row:
        if row['ROC'] > 0: # Price is increasing
            bullish_signals["ROC"] = True
            signal_strength["ROC"] = min(1.0, row['ROC'] / 10) # Scale by 10% change
        elif row['ROC'] < 0: # Price is decreasing
            bearish_signals["ROC"] = True
            signal_strength["ROC"] = min(1.0, abs(row['ROC']) / 10)

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV") and 'obv' in row and 'obv_ema' in row:
        if row['obv'] > row['obv_ema']: # OBV rising, confirming price trend
            bullish_signals["OBV"] = True
            signal_strength["OBV"] = 0.7 # Moderate strength
        elif row['obv'] < row['obv_ema']: # OBV falling, confirming price trend
            bearish_signals["OBV"] = True
            signal_strength["OBV"] = 0.7

    # VWAP (Volume Weighted Average Price) - only for intraday, signal if price is above/below
    if indicator_selection.get("VWAP") and 'VWAP' in row and not pd.isna(row['VWAP']):
        if close > row['VWAP']:
            bullish_signals["VWAP"] = True
            signal_strength["VWAP"] = 0.8 # Strong intraday signal
        elif close < row['VWAP']:
            bearish_signals["VWAP"] = True
            signal_strength["VWAP"] = 0.8

    # Pivot Points (signals based on current price relative to P, R1, S1 etc.)
    # This assumes pivot points are calculated for the current period (e.g., daily pivots for daily data)
    if indicator_selection.get("Pivot Points") and 'Pivot' in row:
        p = row.get('Pivot')
        r1 = row.get('R1')
        s1 = row.get('S1')
        r2 = row.get('R2')
        s2 = row.get('S2')

        if p is not None:
            if close > p: # Price above pivot
                bullish_signals["Pivot Points"] = True
                signal_strength["Pivot Points"] = 0.5
            elif close < p: # Price below pivot
                bearish_signals["Pivot Points"] = True
                signal_strength["Pivot Points"] = 0.5
            
            # More nuanced signals based on resistance/support levels could be added
            # e.g., if close breaks above R1, stronger bullish signal
            if r1 is not None and close > r1:
                bullish_signals["Pivot Points"] = True # Stronger bullish
                signal_strength["Pivot Points"] = 0.8
            if s1 is not None and close < s1:
                bearish_signals["Pivot Points"] = True # Stronger bearish
                signal_strength["Pivot Points"] = 0.8

    return bullish_signals, bearish_signals, signal_strength


def calculate_confidence_score(
    latest_row, news_sentiment_score, recom_score,
    latest_gdp, latest_cpi, latest_unemployment,
    latest_vix, historical_vix_avg,
    indicator_selection, normalized_weights
):
    """
    Calculates an overall confidence score based on various factors.
    Args:
        latest_row (pd.Series): The latest row of the DataFrame with calculated indicators.
        news_sentiment_score (float): News sentiment score (0-100).
        recom_score (float): Analyst recommendation score (0-100).
        latest_gdp (float): Latest GDP growth rate.
        latest_cpi (float): Latest CPI value.
        latest_unemployment (float): Latest unemployment rate.
        latest_vix (float): Latest VIX value.
        historical_vix_avg (float): Historical average VIX.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (scores, overall_confidence, trade_direction)
    """
    scores = {
        "technical": 0, "sentiment": 0, "expert": 0,
        "economic": 0, "investor_sentiment": 0
    }
    
    # --- Technical Score ---
    bullish_tech_signals, bearish_tech_signals, tech_signal_strength = generate_signals_for_row(
        latest_row, indicator_selection, normalized_weights
    )
    
    total_tech_strength = 0
    total_bullish_tech_strength = 0
    total_bearish_tech_strength = 0

    for indicator, selected in indicator_selection.items():
        if selected:
            weight = normalized_weights.get("technical", 0) / sum(1 for s in indicator_selection.values() if s) # Distribute technical weight among selected indicators
            
            if bullish_tech_signals.get(indicator):
                total_bullish_tech_strength += tech_signal_strength.get(indicator, 0) * weight
            if bearish_tech_signals.get(indicator):
                total_bearish_tech_strength += tech_signal_strength.get(indicator, 0) * weight
    
    # Simple aggregation for now:
    if total_bullish_tech_strength > total_bearish_tech_strength:
        scores["technical"] = total_bullish_tech_strength * 100 # Max 100
    elif total_bearish_tech_strength > total_bullish_tech_strength:
        scores["technical"] = -total_bearish_tech_strength * 100 # Max -100 (for bearish)
    # If no strong direction, technical score remains 0

    # --- Sentiment Score (from Finviz news) ---
    if news_sentiment_score is not None:
        scores["sentiment"] = news_sentiment_score # Already 0-100

    # --- Expert Score (from Finviz recommendations) ---
    if recom_score is not None:
        scores["expert"] = recom_score # Already 0-100

    # --- Economic Score ---
    economic_score = calculate_economic_score(latest_gdp, latest_cpi, latest_unemployment)
    scores["economic"] = economic_score

    # --- Investor Sentiment Score (from VIX) ---
    sentiment_score_vix = calculate_sentiment_score(latest_vix, historical_vix_avg)
    scores["investor_sentiment"] = sentiment_score_vix

    # --- Calculate Overall Confidence and Direction ---
    weighted_sum = 0
    total_possible_positive_score = 0
    total_possible_negative_score = 0
    
    # Determine overall trade direction based on weighted sum of directional scores
    directional_sum = 0 # Positive for bullish, negative for bearish

    # Technical component contributes directly to directional_sum
    directional_sum += scores["technical"] * normalized_weights.get("technical", 0)

    # Other components are 0-100. Normalize to -1 to 1 for direction contribution.
    # 0-100 -> -1 to 1 (50 is neutral, 100 is +1, 0 is -1)
    
    # Sentiment (news)
    if scores["sentiment"] is not None:
        directional_sum += ((scores["sentiment"] / 100) * 2 - 1) * normalized_weights.get("sentiment", 0) * 100 # Scale to -100 to 100
        weighted_sum += scores["sentiment"] * normalized_weights.get("sentiment", 0)
        total_possible_positive_score += 100 * normalized_weights.get("sentiment", 0)
        total_possible_negative_score += 0 * normalized_weights.get("sentiment", 0) # Min score is 0

    # Expert (analyst rec)
    if scores["expert"] is not None:
        directional_sum += ((scores["expert"] / 100) * 2 - 1) * normalized_weights.get("expert", 0) * 100 # Scale to -100 to 100
        weighted_sum += scores["expert"] * normalized_weights.get("expert", 0)
        total_possible_positive_score += 100 * normalized_weights.get("expert", 0)
        total_possible_negative_score += 0 * normalized_weights.get("expert", 0)

    # Economic
    if scores["economic"] is not None:
        directional_sum += ((scores["economic"] / 100) * 2 - 1) * normalized_weights.get("economic", 0) * 100
        weighted_sum += scores["economic"] * normalized_weights.get("economic", 0)
        total_possible_positive_score += 100 * normalized_weights.get("economic", 0)
        total_possible_negative_score += 0 * normalized_weights.get("economic", 0)

    # Investor Sentiment (VIX) - often contrarian
    if scores["investor_sentiment"] is not None:
        # If VIX sentiment is high (fear), it's bullish (contrarian). If low (complacency), it's bearish.
        # So, invert the VIX sentiment score for directional sum
        directional_sum += (((100 - scores["investor_sentiment"]) / 100) * 2 - 1) * normalized_weights.get("investor_sentiment", 0) * 100
        weighted_sum += scores["investor_sentiment"] * normalized_weights.get("investor_sentiment", 0)
        total_possible_positive_score += 100 * normalized_weights.get("investor_sentiment", 0)
        total_possible_negative_score += 0 * normalized_weights.get("investor_sentiment", 0)


    overall_confidence = 0
    trade_direction = "Neutral"

    if directional_sum > 0:
        trade_direction = "Bullish"
        # Scale confidence based on how far above 0 the directional sum is, relative to max positive
        # Max directional_sum is 100 (if all components are max bullish and weights sum to 1)
        # Min directional_sum is -100 (if all components are max bearish and weights sum to 1)
        overall_confidence = (directional_sum / 100) * 100 # Scale to 0-100
    elif directional_sum < 0:
        trade_direction = "Bearish"
        overall_confidence = (abs(directional_sum) / 100) * 100 # Scale to 0-100
    else:
        trade_direction = "Neutral"
        overall_confidence = 0 # No clear direction, so confidence is 0

    # Ensure overall_confidence is between 0 and 100
    overall_confidence = max(0, min(100, overall_confidence))

    return scores, overall_confidence, trade_direction


def calculate_economic_score(gdp, cpi, unemployment):
    """
    Calculates an economic score based on GDP, CPI, and Unemployment.
    Scores are normalized to 0-100. Higher is better for stocks.
    """
    score = 50 # Start neutral

    # GDP: Higher is better. Assume typical range 0-5%.
    if gdp is not None and not pd.isna(gdp):
        if gdp > 3.0:
            score += 20 # Very strong
        elif gdp > 1.5:
            score += 10 # Moderate
        elif gdp < 0:
            score -= 20 # Contraction
        else:
            score -= 10 # Slow growth

    # CPI: Lower (stable) is better. Assume typical range 0-10%. Target ~2-3%.
    if cpi is not None and not pd.isna(cpi):
        if cpi < 2.0:
            score += 15 # Low inflation, good
        elif 2.0 <= cpi <= 3.5:
            score += 5 # Moderate, healthy inflation
        elif cpi > 5.0:
            score -= 20 # High inflation, bad
        else:
            score -= 10 # Elevated inflation

    # Unemployment: Lower is better. Assume typical range 3-10%. Target ~3-5%.
    if unemployment is not None and not pd.isna(unemployment):
        if unemployment < 4.0:
            score += 15 # Very low, strong labor market
        elif 4.0 <= unemployment <= 5.5:
            score += 5 # Healthy labor market
        elif unemployment > 7.0:
            score -= 20 # High unemployment, weak labor market
        else:
            score -= 10 # Elevated unemployment

    return max(0, min(100, score)) # Clamp between 0 and 100


def calculate_sentiment_score(latest_vix, historical_vix_avg):
    """
    Calculates an investor sentiment score based on VIX.
    Normalized to 0-100. Lower VIX (complacency) -> lower score (bearish contrarian).
    Higher VIX (fear) -> higher score (bullish contrarian).
    """
    score = 50 # Neutral

    if latest_vix is None or pd.isna(latest_vix) or historical_vix_avg is None or pd.isna(historical_vix_avg):
        return score # Return neutral if data is missing

    # VIX is a fear gauge. High VIX = high fear (often market bottoms, bullish contrarian).
    # Low VIX = complacency (often market tops, bearish contrarian).

    # Normalize VIX relative to its historical average or a typical range (e.g., 10-30)
    # A simple approach:
    # If VIX < 15: Very low fear (complacency) -> lower score
    # If 15 <= VIX <= 20: Normal range -> neutral score
    # If 20 < latest_vix <= 30: Elevated fear -> higher score
    # If latest_vix > 30: High fear -> very high score

    if latest_vix < 15:
        score = 20 # Complacency, bearish contrarian
    elif 15 <= latest_vix <= 20:
        score = 50 # Neutral
    elif 20 < latest_vix <= 30:
        score = 75 # Elevated fear, bullish contrarian
    else: # VIX > 30
        score = 90 # High fear, very bullish contrarian

    # You could also use the historical average for a more dynamic comparison:
    # if latest_vix < historical_vix_avg * 0.8: # Significantly below average
    #     score = 20
    # elif latest_vix > historical_vix_avg * 1.2: # Significantly above average
    #     score = 80
    
    return max(0, min(100, score)) # Clamp between 0 and 100


def convert_finviz_recom_to_score(recom_value):
    """Converts Finviz analyst recommendation (1.00-5.00) to a 0-100 score."""
    # 1.00 (Strong Buy) -> 100
    # 3.00 (Hold) -> 50
    # 5.00 (Strong Sell) -> 0
    # Linear interpolation: score = 100 - (recom_value - 1) * (100 / 4)
    return max(0, min(100, 100 - (recom_value - 1) * 25))


# === Options Analysis Functions ===

def get_moneyness(current_price, strike, option_type):
    """Determines if an option is In-the-Money (ITM), At-the-Money (ATM), or Out-of-the-Money (OTM)."""
    if option_type == 'call':
        if current_price > strike:
            return "ITM"
        elif current_price == strike:
            return "ATM"
        else:
            return "OTM"
    elif option_type == 'put':
        if current_price < strike:
            return "ITM"
        elif current_price == strike:
            return "ATM"
        else:
            return "OTM"
    return "N/A"

def analyze_options_chain(calls_df, puts_df, current_price):
    """
    Performs basic analysis on options chain data.
    Adds 'Moneyness' column and identifies high volume/open interest strikes.
    """
    analysis = {}

    if not calls_df.empty:
        calls_df['Moneyness'] = calls_df.apply(lambda row: get_moneyness(current_price, row['strike'], 'call'), axis=1)
        # Example: Find the call with highest open interest
        if 'openInterest' in calls_df.columns and not calls_df['openInterest'].empty:
            max_oi_call = calls_df.loc[calls_df['openInterest'].idxmax()]
            analysis['max_oi_call'] = max_oi_call.to_dict()
        if 'volume' in calls_df.columns and not calls_df['volume'].empty:
            max_vol_call = calls_df.loc[calls_df['volume'].idxmax()]
            analysis['max_vol_call'] = max_vol_call.to_dict()

    if not puts_df.empty:
        puts_df['Moneyness'] = puts_df.apply(lambda row: get_moneyness(current_price, row['strike'], 'put'), axis=1)
        # Example: Find the put with highest open interest
        if 'openInterest' in puts_df.columns and not puts_df['openInterest'].empty:
            max_oi_put = puts_df.loc[puts_df['openInterest'].idxmax()]
            analysis['max_oi_put'] = max_oi_put.to_dict()
        if 'volume' in puts_df.columns and not puts_df['volume'].empty:
            max_vol_put = puts_df.loc[puts_df['volume'].idxmax()]
            analysis['max_vol_put'] = max_vol_put.to_dict()

    return analysis


def suggest_options_strategy(ticker, confidence_score_value, current_stock_price, expirations, trade_direction):
    """
    Suggests a basic options strategy based on confidence score and trade direction.
    This is a simplified example and should be expanded for real-world use.
    """
    suggested_strategy = {
        "status": "fail",
        "message": "No strategy suggested based on current parameters.",
        "Strategy": "N/A",
        "Direction": "N/A",
        "Expiration": "N/A",
        "Net Debit": "N/A",
        "Max Profit": "N/A",
        "Max Risk": "N/A",
        "Reward / Risk": "N/A",
        "Notes": "N/A",
        "Contracts": {},
        "option_legs_for_chart": [] # For payoff chart
    }

    if not expirations:
        suggested_strategy["message"] = "No expiration dates available for options."
        return suggested_strategy

    # Prioritize shorter-term expirations for swing/day trading, longer for long-term
    # For simplicity, let's pick the first available expiration for now.
    target_expiration = expirations[0] if expirations else None

    if not target_expiration:
        suggested_strategy["message"] = "No valid expiration date found."
        return suggested_strategy

    calls_df, puts_df = get_options_chain(ticker, target_expiration)

    if calls_df.empty and puts_df.empty:
        suggested_strategy["message"] = f"No options data for {ticker} on {target_expiration}."
        return suggested_strategy

    # Strategy logic based on trade direction and confidence
    if trade_direction == "Bullish" and confidence_score_value >= 60:
        # Suggest a Call Debit Spread or Long Call
        # Find an ITM call and an OTM call for a debit spread
        itm_calls = calls_df[calls_df['strike'] < current_stock_price].sort_values(by='strike', ascending=False)
        otm_calls = calls_df[calls_df['strike'] > current_stock_price].sort_values(by='strike', ascending=True)

        if not itm_calls.empty and not otm_calls.empty:
            buy_strike = itm_calls.iloc[0]['strike']
            sell_strike = otm_calls.iloc[0]['strike']
            
            # Ensure we have valid premiums
            buy_premium = itm_calls.iloc[0]['lastPrice']
            sell_premium = otm_calls.iloc[0]['lastPrice']

            if buy_premium and sell_premium:
                net_debit = (buy_premium - sell_premium) * 100
                max_profit = (sell_strike - buy_strike - (buy_premium - sell_premium)) * 100
                max_risk = net_debit

                if net_debit > 0: # Ensure it's a debit spread
                    suggested_strategy.update({
                        "status": "success",
                        "Strategy": "Bull Call Debit Spread",
                        "Direction": "Bullish",
                        "Expiration": target_expiration,
                        "Net Debit": f"${net_debit:.2f}",
                        "Max Profit": f"${max_profit:.2f}",
                        "Max Risk": f"${max_risk:.2f}",
                        "Reward / Risk": f"{(max_profit / max_risk):.1f}:1" if max_risk > 0 else "N/A",
                        "Notes": f"Buy {ticker} Call @ ${buy_strike:.2f}, Sell {ticker} Call @ ${sell_strike:.2f}. Expects moderate bullish movement.",
                        "Contracts": {
                            "buy_call": {"type": "call", "strike": buy_strike, "lastPrice": buy_premium},
                            "sell_call": {"type": "call", "strike": sell_strike, "lastPrice": sell_premium}
                        },
                        "option_legs_for_chart": [
                            {'type': 'call', 'strike': buy_strike, 'premium': buy_premium, 'action': 'buy', 'contracts': 1},
                            {'type': 'call', 'strike': sell_strike, 'premium': sell_premium, 'action': 'sell', 'contracts': 1}
                        ]
                    })
                    return suggested_strategy
        
        # Fallback to Long Call if spread not feasible
        otm_call = otm_calls.iloc[0] if not otm_calls.empty else None
        if otm_call is not None:
            suggested_strategy.update({
                "status": "success",
                "Strategy": "Long Call",
                "Direction": "Bullish",
                "Expiration": target_expiration,
                "Net Debit": f"${otm_call['lastPrice'] * 100:.2f}",
                "Max Profit": "Unlimited",
                "Max Risk": f"${otm_call['lastPrice'] * 100:.2f}",
                "Reward / Risk": "Unlimited",
                "Notes": f"Buy {ticker} Call @ ${otm_call['strike']:.2f}. Expects strong bullish movement.",
                "Contracts": {
                    "buy_call": {"type": "call", "strike": otm_call['strike'], "lastPrice": otm_call['lastPrice']}
                },
                "option_legs_for_chart": [
                    {'type': 'call', 'strike': otm_call['strike'], 'premium': otm_call['lastPrice'], 'action': 'buy', 'contracts': 1}
                ]
            })
            return suggested_strategy


    elif trade_direction == "Bearish" and confidence_score_value >= 60:
        # Suggest a Put Debit Spread or Long Put
        itm_puts = puts_df[puts_df['strike'] > current_stock_price].sort_values(by='strike', ascending=True)
        otm_puts = puts_df[puts_df['strike'] < current_stock_price].sort_values(by='strike', ascending=False)

        if not itm_puts.empty and not otm_puts.empty:
            buy_strike = itm_puts.iloc[0]['strike']
            sell_strike = otm_puts.iloc[0]['strike']

            buy_premium = itm_puts.iloc[0]['lastPrice']
            sell_premium = otm_puts.iloc[0]['lastPrice']

            if buy_premium and sell_premium:
                net_debit = (buy_premium - sell_premium) * 100
                max_profit = (buy_strike - sell_strike - (buy_premium - sell_premium)) * 100
                max_risk = net_debit

                if net_debit > 0:
                    suggested_strategy.update({
                        "status": "success",
                        "Strategy": "Bear Put Debit Spread",
                        "Direction": "Bearish",
                        "Expiration": target_expiration,
                        "Net Debit": f"${net_debit:.2f}",
                        "Max Profit": f"${max_profit:.2f}",
                        "Max Risk": f"${max_risk:.2f}",
                        "Reward / Risk": f"{(max_profit / max_risk):.1f}:1" if max_risk > 0 else "N/A",
                        "Notes": f"Buy {ticker} Put @ ${buy_strike:.2f}, Sell {ticker} Put @ ${sell_strike:.2f}. Expects moderate bearish movement.",
                        "Contracts": {
                            "buy_put": {"type": "put", "strike": buy_strike, "lastPrice": buy_premium},
                            "sell_put": {"type": "put", "strike": sell_strike, "lastPrice": sell_premium}
                        },
                        "option_legs_for_chart": [
                            {'type': 'put', 'strike': buy_strike, 'premium': buy_premium, 'action': 'buy', 'contracts': 1},
                            {'type': 'put', 'strike': sell_strike, 'premium': sell_premium, 'action': 'sell', 'contracts': 1}
                        ]
                    })
                    return suggested_strategy
        
        # Fallback to Long Put if spread not feasible
        otm_put = otm_puts.iloc[0] if not otm_puts.empty else None
        if otm_put is not None:
            suggested_strategy.update({
                "status": "success",
                "Strategy": "Long Put",
                "Direction": "Bearish",
                "Expiration": target_expiration,
                "Net Debit": f"${otm_put['lastPrice'] * 100:.2f}",
                "Max Profit": "Unlimited",
                "Max Risk": f"${otm_put['lastPrice'] * 100:.2f}",
                "Reward / Risk": "Unlimited",
                "Notes": f"Buy {ticker} Put @ ${otm_put['strike']:.2f}. Expects strong bearish movement.",
                "Contracts": {
                    "buy_put": {"type": "put", "strike": otm_put['strike'], "lastPrice": otm_put['lastPrice']}
                },
                "option_legs_for_chart": [
                    {'type': 'put', 'strike': otm_put['strike'], 'premium': otm_put['lastPrice'], 'action': 'buy', 'contracts': 1}
                ]
            })
            return suggested_strategy

    return suggested_strategy


# === Trade Planning Functions ===

def generate_directional_trade_plan(latest_row, indicator_selection, normalized_weights):
    """
    Generates a directional trade plan (entry, target, stop-loss) based on
    technical signals and confidence.
    """
    trade_plan = {
        "direction": "Neutral",
        "confidence_score": 0,
        "entry_zone_start": None,
        "entry_zone_end": None,
        "target_price": None,
        "stop_loss": None,
        "reward_risk_ratio": None,
        "key_rationale": "No clear trade plan generated.",
        "entry_criteria_details": [],
        "exit_criteria_details": [],
        "atr": None # Add ATR to the trade plan
    }

    close = latest_row['Close']
    high = latest_row['High']
    low = latest_row['Low']

    # Calculate ATR for dynamic stop-loss and target
    # Ensure 'High', 'Low', 'Close' are available for ATR calculation
    if all(col in latest_row.index for col in ['High', 'Low', 'Close']):
        # Create a tiny DataFrame for ta.volatility.average_true_range
        # This is a workaround as ta functions are designed for DataFrames
        temp_df = pd.DataFrame([latest_row[['High', 'Low', 'Close']]])
        atr = ta.volatility.average_true_range(temp_df['High'], temp_df['Low'], temp_df['Close'], window=14).iloc[-1]
        trade_plan["atr"] = atr
    else:
        atr = None
        trade_plan["atr"] = "N/A"

    # Get signals and confidence
    scores, overall_confidence, trade_direction = calculate_confidence_score(
        latest_row,
        None, # news_sentiment_score (not available in latest_row)
        None, # recom_score (not available in latest_row)
        None, None, None, # economic data (not available in latest_row)
        None, None, # vix data (not available in latest_row)
        indicator_selection,
        normalized_weights
    )
    
    trade_plan["direction"] = trade_direction
    trade_plan["confidence_score"] = overall_confidence

    if trade_direction == "Bullish" and overall_confidence >= 50:
        # Entry: Slightly below current price or at a support level
        # Target: Resistance level or ATR-based extension
        # Stop Loss: Below a recent low or ATR-based
        
        entry_start = close * 0.99 # Slight dip
        entry_end = close * 1.01 # Slight rise
        
        if atr is not None:
            entry_start = close - (atr * 0.5)
            entry_end = close + (atr * 0.5)
            target = close + (atr * 2) # 2x ATR target
            stop_loss = close - (atr * 1.5) # 1.5x ATR stop
        else:
            target = close * 1.03 # 3% target
            stop_loss = close * 0.98 # 2% stop

        # Adjust based on pivot points if available and selected
        if indicator_selection.get("Pivot Points") and 'Pivot' in latest_row:
            p = latest_row.get('Pivot')
            s1 = latest_row.get('S1')
            r1 = latest_row.get('R1')

            if s1 is not None and close > s1: # If above S1, S1 can be support
                entry_start = min(entry_start, s1) # Entry can be near S1
            if r1 is not None and close < r1: # If below R1, R1 can be target
                target = max(target, r1) # Target can be R1
            if p is not None and close > p: # Price above pivot is bullish
                # Entry could be retest of pivot
                entry_start = min(entry_start, p)


        trade_plan.update({
            "entry_zone_start": entry_start,
            "entry_zone_end": entry_end,
            "target_price": target,
            "stop_loss": stop_loss,
            "key_rationale": f"Strong bullish signals ({overall_confidence:.0f}% confidence). Looking for entry near current price, targeting resistance/ATR extension.",
            "entry_criteria_details": [
                f"Price enters between ${entry_start:.2f} and ${entry_end:.2f}",
                "Confirmation from selected bullish indicators (e.g., EMA cross, RSI bounce from oversold)."
            ],
            "exit_criteria_details": [
                f"Price reaches target of ${target:.2f}",
                f"Price falls to stop loss at ${stop_loss:.2f}",
                "Bearish reversal signal from selected indicators."
            ]
        })

    elif trade_direction == "Bearish" and overall_confidence >= 50:
        # Entry: Slightly above current price or at a resistance level
        # Target: Support level or ATR-based extension
        # Stop Loss: Above a recent high or ATR-based

        entry_start = close * 1.01 # Slight bounce
        entry_end = close * 0.99 # Slight dip

        if atr is not None:
            entry_start = close + (atr * 0.5)
            entry_end = close - (atr * 0.5)
            target = close - (atr * 2) # 2x ATR target
            stop_loss = close + (atr * 1.5) # 1.5x ATR stop
        else:
            target = close * 0.97 # 3% target
            stop_loss = close * 1.02 # 2% stop

        # Adjust based on pivot points if available and selected
        if indicator_selection.get("Pivot Points") and 'Pivot' in latest_row:
            p = latest_row.get('Pivot')
            s1 = latest_row.get('S1')
            r1 = latest_row.get('R1')

            if r1 is not None and close < r1: # If below R1, R1 can be resistance
                entry_start = max(entry_start, r1) # Entry can be near R1
            if s1 is not None and close > s1: # If above S1, S1 can be target
                target = min(target, s1) # Target can be S1
            if p is not None and close < p: # Price below pivot is bearish
                # Entry could be retest of pivot
                entry_start = max(entry_start, p)


        trade_plan.update({
            "entry_zone_start": entry_start,
            "entry_zone_end": entry_end,
            "target_price": target,
            "stop_loss": stop_loss,
            "key_rationale": f"Strong bearish signals ({overall_confidence:.0f}% confidence). Looking for entry near current price, targeting support/ATR extension.",
            "entry_criteria_details": [
                f"Price enters between ${entry_start:.2f} and ${entry_end:.2f}",
                "Confirmation from selected bearish indicators (e.g., EMA cross, RSI fall from overbought)."
            ],
            "exit_criteria_details": [
                f"Price reaches target of ${target:.2f}",
                f"Price rises to stop loss at ${stop_loss:.2f}",
                "Bullish reversal signal from selected indicators."
            ]
        })
    
    # Calculate Reward/Risk Ratio
    if trade_plan["target_price"] is not None and trade_plan["stop_loss"] is not None and trade_plan["entry_zone_start"] is not None:
        if trade_plan["direction"] == "Bullish":
            reward = trade_plan["target_price"] - trade_plan["entry_zone_start"]
            risk = trade_plan["entry_zone_start"] - trade_plan["stop_loss"]
        else: # Bearish
            reward = trade_plan["entry_zone_start"] - trade_plan["target_price"]
            risk = trade_plan["stop_loss"] - trade_plan["entry_zone_start"]
        
        if risk > 0:
            trade_plan["reward_risk_ratio"] = reward / risk
        else:
            trade_plan["reward_risk_ratio"] = float('inf') if reward > 0 else 0 # Infinite if no risk and profit

    return trade_plan


# === Backtesting Functions ===

def backtest_strategy(df, indicator_selection, atr_multiplier, reward_risk_ratio, signal_threshold_percentage, trade_direction_bt, exit_strategy_bt):
    """
    Performs a simple backtest of the selected strategy.
    Args:
        df (pd.DataFrame): Historical data with indicators.
        indicator_selection (dict): Selected indicators for signal generation.
        atr_multiplier (float): Multiplier for ATR to set stop loss.
        reward_risk_ratio (float): Target reward/risk for take profit.
        signal_threshold_percentage (float): Minimum confidence score (0-1) to take a trade.
        trade_direction_bt (str): "long" or "short" for backtest.
        exit_strategy_bt (str): "fixed_rr" or "trailing_psar".
    Returns:
        tuple: (trades_log, performance_metrics)
    """
    trades_log = []
    in_trade = False
    entry_price = 0
    trade_entry_date = None
    trade_direction = "" # "long" or "short"
    stop_loss = 0
    take_profit = 0

    # Ensure df is sorted by date
    df = df.sort_index()

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        current_date = df.index[i]

        # Calculate ATR for current row
        atr_series = ta.volatility.average_true_range(df['High'].iloc[:i+1], df['Low'].iloc[:i+1], df['Close'].iloc[:i+1], window=14)
        current_atr = atr_series.iloc[-1] if not atr_series.empty else np.nan

        # Generate signals and confidence for the current row
        # Note: For backtesting, we use the historical signals at each point.
        # We need to ensure that the confidence score calculation also uses historical economic/sentiment data
        # or simplify it for the backtest. For now, we'll assume a simplified confidence based on tech signals.
        
        # In a real backtest, you'd fetch/calculate economic/sentiment data for each historical date.
        # For this simplified backtest, let's just use technical signals for 'confidence'.
        bullish_signals, bearish_signals, signal_strength = generate_signals_for_row(
            current_row, indicator_selection, {} # Pass empty weights for simplicity in backtest signal_strength
        )
        
        # Calculate a simple technical confidence for backtest entry
        num_bullish = sum(1 for k, v in bullish_signals.items() if v and indicator_selection.get(k))
        num_bearish = sum(1 for k, v in bearish_signals.items() if v and indicator_selection.get(k))
        
        tech_confidence = 0
        if num_bullish + num_bearish > 0:
            if trade_direction_bt == "long":
                tech_confidence = (num_bullish / (num_bullish + num_bearish)) * 100
            elif trade_direction_bt == "short":
                tech_confidence = (num_bearish / (num_bullish + num_bearish)) * 100

        # Entry Logic
        if not in_trade:
            if trade_direction_bt == "long" and num_bullish > 0 and tech_confidence >= (signal_threshold_percentage * 100):
                in_trade = True
                entry_price = current_row['Open'] # Enter at next open
                trade_entry_date = current_date
                trade_direction = "long"
                if not np.isnan(current_atr):
                    stop_loss = entry_price - (current_atr * atr_multiplier)
                    take_profit = entry_price + (current_atr * atr_multiplier * reward_risk_ratio)
                else: # Fallback if ATR is NaN
                    stop_loss = entry_price * 0.98 # 2% stop
                    take_profit = entry_price * 1.03 # 3% target

            elif trade_direction_bt == "short" and num_bearish > 0 and tech_confidence >= (signal_threshold_percentage * 100):
                in_trade = True
                entry_price = current_row['Open'] # Enter at next open
                trade_entry_date = current_date
                trade_direction = "short"
                if not np.isnan(current_atr):
                    stop_loss = entry_price + (current_atr * atr_multiplier)
                    take_profit = entry_price - (current_atr * atr_multiplier * reward_risk_ratio)
                else: # Fallback if ATR is NaN
                    stop_loss = entry_price * 1.02 # 2% stop
                    take_profit = entry_price * 0.97 # 3% target

        # Exit Logic
        if in_trade:
            pnl = 0
            exit_reason = ""
            exit_price = 0

            if trade_direction == "long":
                # Check Stop Loss
                if current_row['Low'] <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * 1 # Assuming 1 share for simplicity
                    exit_reason = "Stop Loss Hit"
                    in_trade = False
                # Check Take Profit (Fixed R/R)
                elif exit_strategy_bt == "fixed_rr" and current_row['High'] >= take_profit:
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) * 1
                    exit_reason = "Take Profit Hit (Fixed R/R)"
                    in_trade = False
                # Check Trailing PSAR
                elif exit_strategy_bt == "trailing_psar" and 'psar' in current_row and current_row['Close'] < current_row['psar']:
                    exit_price = current_row['Close'] # Exit at close if PSAR flips
                    pnl = (exit_price - entry_price) * 1
                    exit_reason = "Trailing PSAR Exit"
                    in_trade = False

            elif trade_direction == "short":
                # Check Stop Loss
                if current_row['High'] >= stop_loss:
                    exit_price = stop_loss
                    pnl = (entry_price - exit_price) * 1
                    exit_reason = "Stop Loss Hit"
                    in_trade = False
                # Check Take Profit (Fixed R/R)
                elif exit_strategy_bt == "fixed_rr" and current_row['Low'] <= take_profit:
                    exit_price = take_profit
                    pnl = (entry_price - exit_price) * 1
                    exit_reason = "Take Profit Hit (Fixed R/R)"
                    in_trade = False
                # Check Trailing PSAR
                elif exit_strategy_bt == "trailing_psar" and 'psar' in current_row and current_row['Close'] > current_row['psar']:
                    exit_price = current_row['Close'] # Exit at close if PSAR flips
                    pnl = (entry_price - exit_price) * 1
                    exit_reason = "Trailing PSAR Exit"
                    in_trade = False
            
            # Record trade if exited
            if not in_trade and exit_reason:
                trades_log.append({
                    "Entry Date": trade_entry_date.strftime('%Y-%m-%d'),
                    "Exit Date": current_date.strftime('%Y-%m-%d'),
                    "Direction": trade_direction.capitalize(),
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "PnL": pnl,
                    "Exit Reason": exit_reason
                })
                # Reset trade variables
                entry_price = 0
                trade_entry_date = None
                trade_direction = ""
                stop_loss = 0
                take_profit = 0

    # Calculate Performance Metrics
    total_trades = len(trades_log)
    winning_trades = sum(1 for trade in trades_log if trade['PnL'] > 0)
    losing_trades = total_trades - winning_trades
    gross_profit = sum(trade['PnL'] for trade in trades_log if trade['PnL'] > 0)
    gross_loss = sum(trade['PnL'] for trade in trades_log if trade['PnL'] < 0)
    net_pnl = gross_profit + gross_loss
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss < 0 else (float('inf') if gross_profit > 0 else 0)

    performance_metrics = {
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss,
        "Net PnL": net_pnl,
        "Profit Factor": f"{profit_factor:.2f}"
    }

    return trades_log, performance_metrics


# === Stock Scanner Function ===

def run_stock_scanner(ticker_list, trading_style, min_confidence, indicator_selection, normalized_weights):
    """
    Scans a list of tickers and returns those that meet the trading style criteria
    with a minimum confidence score.
    Args:
        ticker_list (list): List of ticker symbols to scan.
        trading_style (str): "Swing", "Day", or "Long-Term".
        min_confidence (int): Minimum overall confidence score (0-100) required.
        indicator_selection (dict): Dictionary of selected indicators.
        normalized_weights (dict): Dictionary of normalized weights for confidence scoring.
    Returns:
        pd.DataFrame: DataFrame of qualifying stocks with trade plan details.
    """
    scanned_results = []
    
    # Determine interval and period based on trading style for scanner
    interval_map = {
        "Day": "15m",  # Intraday for day trading
        "Swing": "1d",   # Daily for swing trading
        "Long-Term": "1wk" # Weekly for long-term
    }
    period_map = {
        "Day": "7d",   # Last 7 days for intraday
        "Swing": "1y",   # Last 1 year for swing
        "Long-Term": "5y" # Last 5 years for long-term
    }

    selected_interval = interval_map.get(trading_style, "1d")
    selected_period = period_map.get(trading_style, "1y")
    is_intraday_scanner = "m" in selected_interval or "h" in selected_interval # Check if it's an intraday interval

    for ticker in ticker_list:
        try:
            # For scanner, we fetch data for a fixed period based on trading style
            # and let yfinance determine the start/end dates based on the period.
            # Convert period to start_date and end_date for get_data function
            end_date = datetime.now()
            if selected_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif selected_period == "1y":
                start_date = end_date - timedelta(days=365)
            elif selected_period == "5y":
                start_date = end_date - timedelta(days=365 * 5)
            else: # Default if period not recognized
                start_date = end_date - timedelta(days=365)


            df = get_data(ticker, selected_interval, start_date, end_date)
            
            # Ensure df is a DataFrame and not empty
            if not isinstance(df, pd.DataFrame) or df.empty:
                print(f"Skipping {ticker}: No data or invalid data format from get_data.")
                continue

            df_calculated = calculate_indicators(df.copy(), indicator_selection, is_intraday_scanner)
            
            if df_calculated.empty:
                print(f"Skipping {ticker}: No calculated indicators after processing.")
                continue

            # Ensure pivot points are calculated if selected and not intraday
            last_pivot = {}
            if indicator_selection.get("Pivot Points") and not is_intraday_scanner:
                df_pivots = calculate_pivot_points(df.copy())
                if not df_pivots.empty:
                    last_pivot = df_pivots.iloc[-1].to_dict()
                else:
                    print(f"No pivot points calculated for {ticker}.")

            last_row = df_calculated.iloc[-1]
            current_price = last_row['Close']

            # Fetch Finviz data for scanner as well
            finviz_data = get_finviz_data(ticker)
            
            # Fetch economic data for scanner (using a fixed recent period if not tied to stock data dates)
            # For scanner, let's assume we use very recent economic data (e.g., last 3 months)
            economic_start_date = datetime.now() - timedelta(days=90)
            economic_end_date = datetime.now()
            latest_gdp = get_economic_data_fred("GDP", economic_start_date, economic_end_date)
            latest_cpi = get_economic_data_fred("CPI", economic_start_date, economic_end_date)
            latest_unemployment = get_economic_data_fred("UNRATE", economic_start_date, economic_end_date)

            # Fetch VIX data for scanner
            vix_data_raw = get_vix_data(economic_start_date, economic_end_date)
            vix_data = None
            if isinstance(vix_data_raw, tuple) and len(vix_data_raw) > 0:
                if isinstance(vix_data_raw[0], pd.DataFrame):
                    vix_data = vix_data_raw[0]
            elif isinstance(vix_data_raw, pd.DataFrame):
                vix_data = vix_data_raw

            latest_vix = None
            historical_vix_avg = None
            if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
                latest_vix = vix_data['Close'].iloc[-1]
                historical_vix_avg = vix_data['Close'].mean()


            scores, overall_confidence, trade_direction = calculate_confidence_score(
                last_row,
                finviz_data.get('news_sentiment_score'),
                finviz_data.get('recom_score'),
                latest_gdp.iloc[-1] if latest_gdp is not None and not latest_gdp.empty else None,
                latest_cpi.iloc[-1] if latest_cpi is not None and not latest_cpi.empty else None,
                latest_unemployment.iloc[-1] if latest_unemployment is not None and not latest_unemployment.empty else None,
                latest_vix,
                historical_vix_avg,
                indicator_selection,
                normalized_weights
            )

            # Check if confidence meets the minimum and direction matches style
            meets_confidence = overall_confidence >= min_confidence
            meets_direction = False
            if trading_style == "Swing" or trading_style == "Day":
                # For short-term, both bullish and bearish opportunities are relevant
                if trade_direction != "Neutral":
                    meets_direction = True
            elif trading_style == "Long-Term":
                # For long-term, typically look for bullish opportunities
                if trade_direction == "Bullish":
                    meets_direction = True

            if meets_confidence and meets_direction:
                trade_plan_result = generate_directional_trade_plan(last_row, indicator_selection, normalized_weights)
                
                # Append results if a valid trade plan is generated
                if trade_plan_result and trade_plan_result.get('direction') != "Neutral":
                    entry_criteria_details = trade_plan_result.get('entry_criteria_details', [])
                    exit_criteria_details = trade_plan_result.get('exit_criteria_details', [])

                    scanned_results.append({
                        "Ticker": ticker,
                        "Trading Style": trading_style,
                        "Overall Confidence": f"{overall_confidence:.0f}",
                        "Direction": trade_direction,
                        "Current Price": f"{current_price:.2f}",
                        "ATR": f"{trade_plan_result.get('atr', 'N/A'):.2f}",
                        "Entry Zone": f"${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} - ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}",
                        "Target Price": f"{trade_plan_result.get('target_price', 'N/A'):.2f}",
                        "Stop Loss": f"{trade_plan_result.get('stop_loss', 'N/A'):.2f}",
                        "Reward/Risk": f"{trade_plan_result.get('reward_risk_ratio', 'N/A'):.1f}:1",
                        "Pivot (P)": f"{last_pivot.get('Pivot', 'N/A'):.2f}",
                        "Resistance 1 (R1)": f"{last_pivot.get('R1', 'N/A'):.2f}",
                        "Resistance 2 (R2)": f"{last_pivot.get('R2', 'N/A'):.2f}",
                        "Support 1 (S1)": f"{last_pivot.get('S1', 'N/A'):.2f}",
                        "Support 2 (S2)": f"{last_pivot.get('S2', 'N/A'):.2f}",
                        "Entry Criteria Details": "\n".join(entry_criteria_details),
                        "Exit Criteria Details": "\n".join(exit_criteria_details),
                        "Rationale": trade_plan_result.get('key_rationale', '')
                    })

        except Exception as e:
            print(f"Error scanning {ticker}: {e}") # For debugging
            import traceback
            print(traceback.format_exc()) # Uncomment for full traceback during debugging
            continue
    
    # Sort results by confidence (highest first)
    if scanned_results:
        df_results = pd.DataFrame(scanned_results)
        df_results['Overall Confidence'] = pd.to_numeric(df_results['Overall Confidence'], errors='coerce')
        df_results = df_results.sort_values(by='Overall Confidence', ascending=False).reset_index(drop=True)
        return df_results
    else:
        return pd.DataFrame() # Return empty DataFrame if no results


# --- Simple Test for yfinance ---
def test_yfinance_data_fetch():
    """
    A simple function to test if yfinance can fetch data for a known ticker.
    """
    print("--- Running yfinance connectivity test ---")
    try:
        test_ticker = "SPY"
        test_df = yf.download(test_ticker, period="1d", interval="1m")
        if not test_df.empty:
            print(f"yfinance test successful: Fetched {len(test_df)} rows for {test_ticker}.")
            return True
        else:
            print(f"yfinance test failed: No data returned for {test_ticker}. This might indicate network issues or data availability problems for yfinance.")
            return False
    except Exception as e:
        print(f"yfinance test failed with an error: {e}")
        print("This often indicates network connectivity issues or problems with the yfinance library installation/access.")
        return False

# Run the test when utils.py is loaded
if __name__ == "__main__":
    test_yfinance_data_fetch()
