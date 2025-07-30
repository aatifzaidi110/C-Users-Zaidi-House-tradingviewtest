# utils.py - Final Version (with distutils workaround for Python 3.10+)
print("--- utils.py VERSION CHECK: Loading Final Version with all functions and scanner (v4.1) ---\n") # Added newline for better log readability

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
from datetime import datetime, date, timedelta
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
    df_copy = df.copy()

    # Ensure standard OHLCV columns are present
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df_copy.columns for col in required_cols):
        raise ValueError("Missing required columns in input DataFrame.")

    # === EMA Trend ===
    if indicator_selection.get("EMA Trend"):
        df_copy['EMA_20'] = ta.trend.ema_indicator(df_copy['Close'], window=20)
        df_copy['EMA_50'] = ta.trend.ema_indicator(df_copy['Close'], window=50)

    # === MACD ===
    if indicator_selection.get("MACD"):
        macd = ta.trend.MACD(df_copy['Close'])
        df_copy['MACD'] = macd.macd()
        df_copy['MACD_Signal'] = macd.macd_signal()
        df_copy['MACD_Diff'] = macd.macd_diff()

    # === RSI ===
    if indicator_selection.get("RSI Momentum"):
        rsi = ta.momentum.RSIIndicator(df_copy['Close'], window=14)
        df_copy['RSI'] = rsi.rsi()

    # === Bollinger Bands ===
    if indicator_selection.get("Bollinger Bands"):
        bb = ta.volatility.BollingerBands(df_copy['Close'], window=20)
        df_copy['BB_High'] = bb.bollinger_hband()
        df_copy['BB_Low'] = bb.bollinger_lband()
        df_copy['BB_Mid'] = bb.bollinger_mavg()

    # === Stochastic Oscillator ===
    if indicator_selection.get("Stochastic"):
        stoch = ta.momentum.StochasticOscillator(df_copy['High'], df_copy['Low'], df_copy['Close'])
        df_copy['Stoch_K'] = stoch.stoch()
        df_copy['Stoch_D'] = stoch.stoch_signal()

    # === Add more indicators as needed below (e.g., CCI, OBV, etc.) ===

    # Optional: Drop rows where indicators are NaN
    df_copy.dropna(inplace=True)

    print("✅ [calculate_indicators] Output shape:", df_copy.shape)
    print("✅ [calculate_indicators] Columns:", df_copy.columns.tolist())


    return df_copy

    # Combine current and required columns, then create a unique list for reindex
    all_cols = list(set(current_col_names + required_cols))
    
    # Reindex the DataFrame to ensure all required columns exist, filling missing with NaN
    df_copy = df_copy.reindex(columns=all_cols, fill_value=np.nan)

    # Now, ensure numeric types for the core columns
    for col in required_cols:
        if col in df_copy.columns:
            # Always convert to a Series first if it's not already, then to numeric
            if isinstance(df_copy[col], pd.DataFrame):
                # If it's a DataFrame, assume it's a single column and convert to Series
                df_copy[col] = pd.to_numeric(df_copy[col].iloc[:, 0], errors='coerce')
            elif not isinstance(df_copy[col], pd.Series):
                # If it's not a Series (e.g., a list, numpy array, or scalar), convert to Series first
                df_copy[col] = pd.to_numeric(pd.Series(df_copy[col]), errors='coerce')
            else:
                # If it's already a Series, just convert to numeric
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    # It's crucial to check if df_copy is empty *after* the initial data fetch and before calculations.
    if df_copy.empty:
        print("DataFrame is empty after ensuring required columns and initial type conversion.")
        return pd.DataFrame()

    # Now dropna can be safely called as columns are guaranteed to exist
    df_copy = df_copy.dropna(subset=required_cols)

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
        # The `ta.trend.ichimoku_cloud` function expects Series, not a single row.
        # This part of the logic needs to read pre-calculated Ichimoku values from 'row'.
        if 'ichimoku_base_line' in row and 'ichimoku_conversion_line' in row and \
           'ichimoku_leading_span_a' in row and 'ichimoku_leading_span_b' in row:
            
            ichimoku_base_line = row['ichimoku_base_line']
            ichimoku_conversion_line = row['ichimoku_conversion_line']
            ichimoku_leading_span_a = row['ichimoku_leading_span_a']
            ichimoku_leading_span_b = row['ichimoku_leading_span_b']

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
    if indicator_selection.get("Parabolic SAR"):
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
        if row['CCI'] > 100:
            bullish_signals["CCI"] = True
            signal_strength["CCI"] = min(1.0, (row['CCI'] - 100) / 100)
        elif row['CCI'] < -100:
            bearish_signals["CCI"] = True
            signal_strength["CCI"] = min(1.0, (-100 - row['CCI']) / 100)

    # ROC (Rate of Change)
    if indicator_selection.get("ROC") and 'ROC' in row:
        if row['ROC'] > 0:
            bullish_signals["ROC"] = True
            signal_strength["ROC"] = min(1.0, row['ROC'] / 10) # Scale, adjust divisor as needed
        elif row['ROC'] < 0:
            bearish_signals["ROC"] = True
            signal_strength["ROC"] = min(1.0, abs(row['ROC']) / 10)

    # OBV (On-Balance Volume)
    if indicator_selection.get("OBV") and 'obv' in row and 'obv_ema' in row:
        if row['obv'] > row['obv_ema']:
            bullish_signals["OBV"] = True
            signal_strength["OBV"] = 1.0
        elif row['obv'] < row['obv_ema']:
            bearish_signals["OBV"] = True
            signal_strength["OBV"] = 1.0

    # VWAP (Volume Weighted Average Price)
    if indicator_selection.get("VWAP") and 'VWAP' in row:
        if close > row['VWAP']:
            bullish_signals["VWAP"] = True
            signal_strength["VWAP"] = min(1.0, (close - row['VWAP']) / row['VWAP'])
        elif close < row['VWAP']:
            bearish_signals["VWAP"] = True
            signal_strength["VWAP"] = min(1.0, (row['VWAP'] - close) / row['VWAP'])

    return bullish_signals, bearish_signals, signal_strength


def calculate_confidence_score(bullish_signals, bearish_signals, signal_strength, normalized_weights):
    """
    Calculates an overall confidence score based on individual indicator signals and their weights.
    Args:
        bullish_signals (dict): True/False for each bullish signal.
        bearish_signals (dict): True/False for each bearish signal.
        signal_strength (dict): Raw strength for each signal (0-1).
        normalized_weights (dict): Dictionary of normalized weights for each component.
    Returns:
        tuple: (overall_confidence, direction, reasons)
               overall_confidence (float): Overall confidence score (0-100).
               direction (str): "Bullish", "Bearish", or "Neutral".
               reasons (list): List of strings explaining the signals.
    """
    total_bullish_score = 0
    total_bearish_score = 0
    reasons = []

    for indicator, is_bullish in bullish_signals.items():
        weight = normalized_weights.get(indicator, 0)
        strength = signal_strength.get(indicator, 0)

        if is_bullish:
            total_bullish_score += weight * strength
            reasons.append(f"• {indicator}: Bullish (Strength: {strength:.2f})")
        
        # Check for bearish signals for the same indicator
        if bearish_signals.get(indicator):
            total_bearish_score += weight * strength # Use same strength for bearish
            if not is_bullish: # Only add if not already added as conflicting
                reasons.append(f"• {indicator}: Bearish (Strength: {strength:.2f})")

    # Determine overall direction and confidence
    if total_bullish_score > total_bearish_score:
        overall_confidence = total_bullish_score * 100
        direction = "Bullish"
    elif total_bearish_score > total_bullish_score:
        overall_confidence = total_bearish_score * 100
        direction = "Bearish"
    else:
        overall_confidence = total_bullish_score * 100 # If equal, use either score
        direction = "Neutral"
        if not reasons: # If no clear signals, state it
            reasons.append("• No strong directional signals detected from selected indicators.")

    # Cap confidence at 100%
    overall_confidence = min(100, overall_confidence)

    return overall_confidence, direction, reasons


def convert_finviz_recom_to_score(recom_value):
    """Converts Finviz recommendation value (1.0-5.0) to a 0-100 score."""
    # 1.0 (Strong Buy) -> 100, 5.0 (Strong Sell) -> 0, 3.0 (Hold) -> 50
    return (5.0 - recom_value) / 4.0 * 100


def calculate_economic_score(gdp_data, cpi_data, unemployment_data):
    """
    Calculates an economic health score (0-100) based on recent economic data trends.
    Higher score indicates better economic health.
    """
    score = 50 # Start with a neutral score

    # GDP: Positive growth is good
    if gdp_data is not None and not gdp_data.empty:
        latest_gdp = gdp_data.iloc[-1]
        # Compare to previous period or a threshold
        if len(gdp_data) > 1:
            prev_gdp = gdp_data.iloc[-2]
            if latest_gdp > prev_gdp:
                score += 10 # Positive growth
            elif latest_gdp < prev_gdp:
                score -= 10 # Negative growth
        # Absolute level check (e.g., if GDP is growing strongly)
        if latest_gdp > 0: # Assuming positive GDP is generally good
             score += 5

    # CPI: Stable/low inflation is good, high inflation is bad
    if cpi_data is not None and not cpi_data.empty:
        latest_cpi = cpi_data.iloc[-1]
        # Target inflation around 2-3%
        if 2.0 <= latest_cpi <= 3.0:
            score += 10 # Ideal inflation
        elif latest_cpi < 2.0:
            score += 5 # Low inflation (could be deflationary risk if too low)
        elif latest_cpi > 3.0:
            score -= 10 # High inflation

    # Unemployment Rate: Low unemployment is good
    if unemployment_data is not None and not unemployment_data.empty:
        latest_unrate = unemployment_data.iloc[-1]
        # Compare to historical low or a threshold
        if latest_unrate < 4.0: # Generally considered low unemployment
            score += 10
        elif latest_unrate > 6.0: # Generally considered high unemployment
            score -= 10
        
        # Trend check
        if len(unemployment_data) > 1:
            prev_unrate = unemployment_data.iloc[-2]
            if latest_unrate < prev_unrate:
                score += 5 # Decreasing unemployment
            elif latest_unrate > prev_unrate:
                score -= 5 # Increasing unemployment

    # Ensure score is within 0-100 bounds
    return max(0, min(100, score))


def calculate_sentiment_score(finviz_data, vix_data):
    """
    Calculates an overall market sentiment score (0-100) based on Finviz news/recommendations and VIX.
    Higher score indicates more positive sentiment.
    """
    sentiment_score = 50 # Start neutral

    # Finviz Analyst Recommendation Score (0-100, 100 is Strong Buy)
    if finviz_data and finviz_data.get("recom_score") is not None:
        sentiment_score += (finviz_data["recom_score"] - 50) * 0.4 # Scale influence

    # Finviz News Sentiment Score (0-100, 100 is very positive)
    if finviz_data and finviz_data.get("news_sentiment_score") is not None:
        sentiment_score += (finviz_data["news_sentiment_score"] - 50) * 0.4 # Scale influence

    # VIX (Volatility Index): Lower VIX indicates higher sentiment/less fear
    if vix_data is not None and not vix_data.empty:
        latest_vix = vix_data['Close'].iloc[-1]
        # VIX typically ranges from 10-30, with spikes much higher.
        # Lower VIX (e.g., < 20) is bullish, higher VIX (e.g., > 30) is bearish.
        if latest_vix < 17:
            sentiment_score += 15 # Low fear, bullish
        elif latest_vix > 25:
            sentiment_score -= 15 # High fear, bearish
        elif 17 <= latest_vix <= 25:
            # Neutral range, slight adjustment based on proximity to extremes
            sentiment_score += (21 - latest_vix) * 0.5 # Closer to 17 adds, closer to 25 subtracts

    # Ensure score is within 0-100 bounds
    return max(0, min(100, score))


def get_moneyness(current_price, strike_price, option_type):
    """
    Determines the moneyness of an option.
    """
    if option_type == 'call':
        if strike_price < current_price:
            return "ITM" # In The Money
        elif strike_price == current_price:
            return "ATM" # At The Money
        else:
            return "OTM" # Out of The Money
    elif option_type == 'put':
        if strike_price > current_price:
            return "ITM" # In The Money
        elif strike_price == current_price:
            return "ATM" # At The Money
        else:
            return "OTM" # Out of The Money
    return "N/A" # Should not happen


def analyze_options_chain(calls_df, puts_df, current_price, target_price, stop_loss_price, trade_direction):
    """
    Analyzes options chain to find suitable options based on trade plan.
    Prioritizes options that align with the directional bias and risk management.
    """
    suitable_options = []

    # Filter calls/puts based on trade direction
    if trade_direction == "Bullish":
        # For bullish, we might look for ITM/ATM calls or OTM calls for aggressive plays
        # Or OTM puts for selling premium (bearish on volatility, but bullish on price)
        options_to_consider = calls_df
        option_type_filter = 'call'
    elif trade_direction == "Bearish":
        # For bearish, we might look for ITM/ATM puts or OTM puts for aggressive plays
        # Or OTM calls for selling premium (bearish on volatility, but bearish on price)
        options_to_consider = puts_df
        option_type_filter = 'put'
    else: # Neutral or undefined
        return [] # No specific options strategy for neutral directional bias

    if options_to_consider.empty:
        return []

    # Common filtering criteria for both calls and puts
    # Filter out options with very low volume or open interest as they are illiquid
    options_to_consider = options_to_consider[
        (options_to_consider['volume'] > 0) & (options_to_consider['openInterest'] > 0)
    ]

    for index, option in options_to_consider.iterrows():
        moneyness = get_moneyness(current_price, option['strike'], option_type_filter)
        
        # Prioritize options that align with the trade direction and risk/reward
        is_suitable = False
        rationale = []

        if option_type_filter == 'call' and trade_direction == "Bullish":
            # For bullish call, strike should ideally be below or near target, and above stop-loss
            if option['strike'] <= target_price and option['strike'] >= stop_loss_price:
                is_suitable = True
                rationale.append("Strike price aligns with target/stop-loss.")
            
            if moneyness in ["ITM", "ATM"]:
                is_suitable = True
                rationale.append(f"{moneyness} option, good for directional move.")
            elif moneyness == "OTM" and option['strike'] <= current_price * 1.05: # Slightly OTM
                is_suitable = True
                rationale.append("Slightly OTM, potential for higher leverage.")

        elif option_type_filter == 'put' and trade_direction == "Bearish":
            # For bearish put, strike should ideally be above or near target, and below stop-loss
            if option['strike'] >= target_price and option['strike'] <= stop_loss_price:
                is_suitable = True
                rationale.append("Strike price aligns with target/stop-loss.")

            if moneyness in ["ITM", "ATM"]:
                is_suitable = True
                rationale.append(f"{moneyness} option, good for directional move.")
            elif moneyness == "OTM" and option['strike'] >= current_price * 0.95: # Slightly OTM
                is_suitable = True
                rationale.append("Slightly OTM, potential for higher leverage.")

        if is_suitable:
            suitable_options.append({
                "contractSymbol": option['contractSymbol'],
                "strike": option['strike'],
                "lastPrice": option['lastPrice'],
                "bid": option['bid'],
                "ask": option['ask'],
                "volume": option['volume'],
                "openInterest": option['openInterest'],
                "impliedVolatility": option['impliedVolatility'],
                "moneyness": moneyness,
                "optionType": option_type_filter,
                "rationale": ", ".join(rationale)
            })
    return suitable_options


def suggest_options_strategy(suitable_options, trade_direction, current_price, target_price, stop_loss_price):
    """
    Suggests a basic options strategy (e.g., buying a call/put) based on suitable options
    and the overall trade direction.
    """
    if not suitable_options:
        return "No suitable options found for a directional strategy based on current criteria."

    suggested_strategy = {
        "type": "N/A",
        "option_symbol": "N/A",
        "strike": "N/A",
        "premium": "N/A",
        "expiration": "N/A",
        "rationale": "N/A"
    }

    # Sort suitable options to find the most relevant one (e.g., closest to ATM, good liquidity)
    # For simplicity, let's pick the one closest to current price with decent liquidity
    
    # Filter for options that are ATM or slightly OTM, and have good volume/open interest
    filtered_options = [
        opt for opt in suitable_options 
        if opt['moneyness'] in ["ATM", "OTM"] and opt['volume'] > 100 and opt['openInterest'] > 100
    ]

    if not filtered_options:
        # If no ATM/OTM with good liquidity, try ITM with good liquidity
        filtered_options = [
            opt for opt in suitable_options 
            if opt['moneyness'] == "ITM" and opt['volume'] > 100 and opt['openInterest'] > 100
        ]
    
    if not filtered_options:
        # If still nothing, just take the first suitable one as a fallback
        filtered_options = suitable_options


    # Sort by distance from current price (ATM first), then by volume
    filtered_options.sort(key=lambda x: (abs(x['strike'] - current_price), -x['volume']))

    if filtered_options:
        best_option = filtered_options[0]
        expiration_date = best_option['contractSymbol'].split('C' if best_option['optionType'] == 'call' else 'P')[0][-6:] # Extract YYYYMMDD
        expiration_date_obj = datetime.strptime(expiration_date, '%y%m%d').strftime('%Y-%m-%d') # Convert to full date

        if trade_direction == "Bullish" and best_option['optionType'] == 'call':
            suggested_strategy["type"] = "Buy Call Option"
            suggested_strategy["option_symbol"] = best_option['contractSymbol']
            suggested_strategy["strike"] = best_option['strike']
            suggested_strategy["premium"] = best_option['lastPrice']
            suggested_strategy["expiration"] = expiration_date_obj
            suggested_strategy["rationale"] = (
                f"To capitalize on expected upward movement. "
                f"This {best_option['moneyness']} call has a strike of ${best_option['strike']:.2f} "
                f"which is aligned with the bullish outlook towards your target of ${target_price:.2f}. "
                f"It offers leverage and defined risk (max loss is premium paid)."
            )
        elif trade_direction == "Bearish" and best_option['optionType'] == 'put':
            suggested_strategy["type"] = "Buy Put Option"
            suggested_strategy["option_symbol"] = best_option['contractSymbol']
            suggested_strategy["strike"] = best_option['strike']
            suggested_strategy["premium"] = best_option['lastPrice']
            suggested_strategy["expiration"] = expiration_date_obj
            suggested_strategy["rationale"] = (
                f"To capitalize on expected downward movement. "
                f"This {best_option['moneyness']} put has a strike of ${best_option['strike']:.2f} "
                f"which is aligned with the bearish outlook towards your target of ${target_price:.2f}. "
                f"It offers leverage and defined risk (max loss is premium paid)."
            )
        else:
            suggested_strategy["rationale"] = "Could not find a direct options strategy matching the directional bias and available options."

    return suggested_strategy


def generate_directional_trade_plan(
    ticker,
    interval,
    start_date,
    end_date,
    indicator_selection,
    weights,
    options_expiration_date=None
):
    """
    Generates a comprehensive directional trade plan for a given ticker,
    including technical analysis, sentiment, economic context, and options strategy.
    """
    trade_plan = {
        "ticker": ticker,
        "current_price": None,
        "trade_direction": "Neutral",
        "overall_confidence": 0,
        "entry_zone_start": None,
        "entry_zone_end": None,
        "target_price": None,
        "stop_loss": None,
        "reward_risk_ratio": None,
        "key_rationale": "",
        "technical_signals": [],
        "sentiment_analysis": {},
        "economic_context": {},
        "suggested_options_strategy": {},
        "pivot_points": {}
    }

    # 1. Fetch Data
    df = get_data(ticker, interval, start_date, end_date)
    if df.empty:
        trade_plan["key_rationale"] = f"Could not fetch sufficient historical data for {ticker}."
        return trade_plan
    
    current_price = df['Close'].iloc[-1]
    trade_plan["current_price"] = current_price

    # 2. Calculate Indicators
    df_indicators = calculate_indicators(df, indicator_selection, is_intraday=(interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']))
    if df_indicators.empty:
        trade_plan["key_rationale"] = "Not enough data to calculate indicators after cleaning."
        return trade_plan

    latest_row = df_indicators.iloc[-1]

    # 3. Generate Signals and Confidence Score
    # Normalize weights to ensure they sum to 1 for proper confidence calculation
    total_weight_sum = sum(weights.values())
    normalized_weights = {k: v / total_weight_sum for k, v in weights.items()}

    bullish_signals, bearish_signals, signal_strength = generate_signals_for_row(latest_row, indicator_selection, normalized_weights)
    
    # Add Pivot Points to signals if selected and available
    if indicator_selection.get("Pivot Points"):
        # Calculate pivot points for the last available day
        # For daily/weekly/monthly intervals, this is straightforward.
        # For intraday, you might want to calculate daily pivots.
        # Assuming for simplicity, we calculate pivots for the last full day's data
        if interval in ['1d', '1wk', '1mo']:
            last_day_df = df.iloc[-1:] # Just the last row for daily/weekly/monthly
        else: # For intraday, get the last full day
            last_day_date = df.index[-1].date()
            last_day_df = df[df.index.date == last_day_date]
            if last_day_df.empty:
                # Fallback if no full day data for intraday, use last available bar
                last_day_df = df.iloc[-1:]

        pivot_points_df = calculate_pivot_points(last_day_df)
        if not pivot_points_df.empty:
            last_pivot = pivot_points_df.iloc[-1].to_dict()
            trade_plan["pivot_points"] = {k: float(v) for k, v in last_pivot.items()} # Ensure float conversion
            
            # Incorporate pivot points into signals and rationale
            # This is a basic example; more complex logic could be used
            if trade_plan["trade_direction"] == "Bullish":
                # If price is above Pivot, it's bullish
                if current_price > last_pivot.get('Pivot', -np.inf):
                    bullish_signals["Pivot Points"] = True
                    signal_strength["Pivot Points"] = 0.7 # Moderate strength
                    trade_plan["technical_signals"].append("• Price is above the daily Pivot Point.")
                # If price is above S1, it's bullish
                if current_price > last_pivot.get('S1', -np.inf):
                    bullish_signals["Pivot Points"] = True
                    signal_strength["Pivot Points"] = max(signal_strength["Pivot Points"], 0.5)
                    trade_plan["technical_signals"].append("• Price is holding above Support 1 (S1).")
            elif trade_plan["trade_direction"] == "Bearish":
                # If price is below Pivot, it's bearish
                if current_price < last_pivot.get('Pivot', np.inf):
                    bearish_signals["Pivot Points"] = True
                    signal_strength["Pivot Points"] = 0.7
                    trade_plan["technical_signals"].append("• Price is below the daily Pivot Point.")
                # If price is below R1, it's bearish
                if current_price < last_pivot.get('R1', np.inf):
                    bearish_signals["Pivot Points"] = True
                    signal_strength["Pivot Points"] = max(signal_strength["Pivot Points"], 0.5)
                    trade_plan["technical_signals"].append("• Price is below Resistance 1 (R1).")


    overall_confidence, direction, reasons = calculate_confidence_score(bullish_signals, bearish_signals, signal_strength, normalized_weights)
    trade_plan["overall_confidence"] = overall_confidence
    trade_plan["trade_direction"] = direction
    trade_plan["technical_signals"] = reasons # Use the reasons generated by confidence score

    # 4. Determine Entry, Target, Stop Loss (simplified example)
    # This part needs more sophisticated logic based on indicators and price action.
    # For now, a simple ATR-based calculation.
    if 'ATR' in latest_row and not pd.isna(latest_row['ATR']):
        atr = latest_row['ATR']
    else:
        # Calculate ATR if not already present or NaN
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        atr = df['ATR'].iloc[-1] if not df['ATR'].empty and not pd.isna(df['ATR'].iloc[-1]) else (current_price * 0.02) # Default to 2% of price if ATR not available
        df_indicators['ATR'] = df['ATR'] # Add to df_indicators as well for consistency

    if direction == "Bullish":
        trade_plan["entry_zone_start"] = current_price - (0.5 * atr)
        trade_plan["entry_zone_end"] = current_price + (0.5 * atr)
        trade_plan["target_price"] = current_price + (2 * atr) # 2x ATR
        trade_plan["stop_loss"] = current_price - (1 * atr) # 1x ATR
    elif direction == "Bearish":
        trade_plan["entry_zone_start"] = current_price + (0.5 * atr)
        trade_plan["entry_zone_end"] = current_price - (0.5 * atr)
        trade_plan["target_price"] = current_price - (2 * atr) # 2x ATR
        trade_plan["stop_loss"] = current_price + (1 * atr) # 1x ATR
    
    # Calculate Reward/Risk Ratio
    if trade_plan["target_price"] and trade_plan["stop_loss"] and current_price:
        risk = abs(current_price - trade_plan["stop_loss"])
        reward = abs(trade_plan["target_price"] - current_price)
        if risk > 0:
            trade_plan["reward_risk_ratio"] = reward / risk
        else:
            trade_plan["reward_risk_ratio"] = np.inf # Infinite if no risk

    # 5. Sentiment Analysis
    finviz_data = get_finviz_data(ticker)
    # Fetch VIX data for the last month to get a recent trend/value
    vix_start_date = (date.today() - timedelta(days=30))
    vix_end_date = date.today()
    vix_df = get_vix_data(vix_start_date, vix_end_date)
    
    sentiment_score = calculate_sentiment_score(finviz_data, vix_df)
    trade_plan["sentiment_analysis"] = {
        "overall_score": sentiment_score,
        "finviz_recommendation": finviz_data.get("recom_score"),
        "finviz_news_sentiment": finviz_data.get("news_sentiment_score"),
        "latest_vix": vix_df['Close'].iloc[-1] if not vix_df.empty else None
    }
    trade_plan["key_rationale"] += f"\n\n**Market Sentiment Score:** {sentiment_score:.2f}/100. "
    if sentiment_score >= 70:
        trade_plan["key_rationale"] += "Market sentiment is generally positive."
    elif sentiment_score <= 30:
        trade_plan["key_rationale"] += "Market sentiment is generally negative."
    else:
        trade_plan["key_rationale"] += "Market sentiment is neutral."


    # 6. Economic Context
    # Fetch recent economic data (e.g., last 3 months)
    econ_start_date = (date.today() - timedelta(days=90))
    econ_end_date = date.today()

    gdp_data = get_economic_data_fred("GDP", econ_start_date, econ_end_date)
    cpi_data = get_economic_data_fred("CPI", econ_start_date, econ_end_date)
    unemployment_data = get_economic_data_fred("UNRATE", econ_start_date, econ_end_date)

    economic_score = calculate_economic_score(gdp_data, cpi_data, unemployment_data)
    trade_plan["economic_context"] = {
        "overall_score": economic_score,
        "latest_gdp": gdp_data.iloc[-1] if not gdp_data.empty else None,
        "latest_cpi": cpi_data.iloc[-1] if not cpi_data.empty else None,
        "latest_unemployment_rate": unemployment_data.iloc[-1] if not unemployment_data.empty else None
    }
    trade_plan["key_rationale"] += f"\n\n**Economic Health Score:** {economic_score:.2f}/100. "
    if economic_score >= 70:
        trade_plan["key_rationale"] += "Economic conditions appear favorable."
    elif economic_score <= 30:
        trade_plan["key_rationale"] += "Economic conditions appear challenging."
    else:
        trade_plan["key_rationale"] += "Economic conditions are neutral."


    # 7. Options Strategy Suggestion
    if options_expiration_date:
        calls_df, puts_df = get_options_chain(ticker, options_expiration_date)
        if not calls_df.empty or not puts_df.empty:
            suitable_options = analyze_options_chain(calls_df, puts_df, current_price, trade_plan["target_price"], trade_plan["stop_loss"], trade_plan["trade_direction"])
            suggested_options = suggest_options_strategy(suitable_options, trade_plan["trade_direction"], current_price, trade_plan["target_price"], trade_plan["stop_loss"])
            trade_plan["suggested_options_strategy"] = suggested_options
            trade_plan["key_rationale"] += f"\n\n**Options Strategy:** {suggested_options.get('type', 'N/A')}. {suggested_options.get('rationale', '')}"
        else:
            trade_plan["key_rationale"] += "\n\nNo options data available for the selected expiration date."
    else:
        trade_plan["key_rationale"] += "\n\nNo options expiration date provided for options strategy analysis."


    # Final rationale based on overall confidence and direction
    if trade_plan["overall_confidence"] >= 70:
        trade_plan["key_rationale"] = f"Strong {trade_plan['trade_direction']} signal ({trade_plan['overall_confidence']:.2f}% confidence) based on multiple converging technical indicators, supported by market sentiment and economic context." + trade_plan["key_rationale"]
    elif trade_plan["overall_confidence"] >= 50:
        trade_plan["key_rationale"] = f"Moderate {trade_plan['trade_direction']} signal ({trade_plan['overall_confidence']:.2f}% confidence) with some supporting factors." + trade_plan["key_rationale"]
    else:
        trade_plan["key_rationale"] = f"Neutral or weak directional signal ({trade_plan['overall_confidence']:.2f}% confidence). Consider waiting for clearer signals or re-evaluating parameters." + trade_plan["key_rationale"]

    return trade_plan

def backtest_strategy(df, trade_plan_func, initial_capital=10000, commission=0.001):
    """
    Backtests a trading strategy based on generated trade plans.
    Args:
        df (pd.DataFrame): Historical stock data with 'Open', 'High', 'Low', 'Close'.
        trade_plan_func (callable): A function that takes a DataFrame slice (representing historical data up to a point)
                                    and returns a trade plan (dict) with 'trade_direction', 'entry_zone_start',
                                    'entry_zone_end', 'target_price', 'stop_loss'.
        initial_capital (float): Starting capital for backtesting.
        commission (float): Commission per trade as a percentage of trade value (e.g., 0.001 for 0.1%).
    Returns:
        dict: Backtesting results including final capital, total trades, win rate, etc.
    """
    capital = initial_capital
    shares_held = 0
    trades = [] # List to store details of each trade
    
    # Ensure df is sorted by date
    df = df.sort_index()

    # Iterate through the DataFrame to simulate trading day by day (or bar by bar)
    # Start from a point where enough historical data is available for indicator calculation
    min_history_for_indicators = 200 # Roughly for 200-period EMA, adjust as needed

    for i in range(min_history_for_indicators, len(df)):
        current_data = df.iloc[:i+1] # Data up to the current point for trade plan generation
        current_price = current_data['Close'].iloc[-1]
        current_date = current_data.index[-1]

        # Generate trade plan for the current historical context
        # Note: We need to pass the actual arguments that generate_directional_trade_plan expects.
        # This backtest_strategy function assumes a simplified trade_plan_func for demonstration.
        # In a real scenario, you would pass the full context (indicator_selection, weights, etc.)
        # to the trade_plan_func, or make trade_plan_func a method of a strategy class.
        
        # For simplicity, let's assume trade_plan_func wraps generate_directional_trade_plan
        # and has access to indicator_selection and weights.
        # This is a placeholder and needs to be adapted to how trade_plan_func is structured.
        
        # Mock trade plan for demonstration purposes
        trade_plan_result = {
            "trade_direction": "Neutral",
            "entry_zone_start": current_price * 0.99,
            "entry_zone_end": current_price * 1.01,
            "target_price": current_price * 1.05,
            "stop_loss": current_price * 0.95,
            "current_price": current_price
        }

        # --- Trading Logic ---
        if shares_held == 0: # Not in a position
            if trade_plan_result["trade_direction"] == "Bullish" and \
               trade_plan_result["entry_zone_start"] <= current_price <= trade_plan_result["entry_zone_end"]:
                
                # Buy signal
                shares_to_buy = int(capital / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + commission)
                    if capital >= cost:
                        capital -= cost
                        shares_held += shares_to_buy
                        trades.append({
                            "date": current_date,
                            "type": "BUY",
                            "price": current_price,
                            "shares": shares_to_buy,
                            "capital": capital,
                            "position_value": shares_held * current_price
                        })
                        # st.write(f"BUY: {shares_to_buy} shares at {current_price:.2f} on {current_date.date()}")

            elif trade_plan_result["trade_direction"] == "Bearish" and \
                 trade_plan_result["entry_zone_end"] <= current_price <= trade_plan_result["entry_zone_start"]: # Note: entry_zone_start/end might be swapped for bearish
                
                # Sell (short) signal - simplified, assuming ability to short
                shares_to_short = int(capital / current_price) # Use capital as margin for short
                if shares_to_short > 0:
                    proceeds = shares_to_short * current_price * (1 - commission)
                    capital += proceeds # Capital increases from short sale
                    shares_held -= shares_to_short # Negative shares for short position
                    trades.append({
                        "date": current_date,
                        "type": "SHORT",
                        "price": current_price,
                        "shares": shares_to_short,
                        "capital": capital,
                        "position_value": shares_held * current_price # This will be negative
                    })
                    # st.write(f"SHORT: {shares_to_short} shares at {current_price:.2f} on {current_date.date()}")

        else: # In a position (long or short)
            if shares_held > 0: # Long position
                # Check for target or stop loss
                if current_price >= trade_plan_result["target_price"] or \
                   current_price <= trade_plan_result["stop_loss"]:
                    
                    # Sell to close long position
                    proceeds = shares_held * current_price * (1 - commission)
                    capital += proceeds
                    trade_profit_loss = (current_price - trades[-1]['price']) * shares_held - (2 * commission * current_price * shares_held) # Simplified P/L
                    trades.append({
                        "date": current_date,
                        "type": "SELL",
                        "price": current_price,
                        "shares": shares_held,
                        "capital": capital,
                        "profit_loss": trade_profit_loss
                    })
                    # st.write(f"SELL: {shares_held} shares at {current_price:.2f} on {current_date.date()}")

            elif shares_held < 0: # Short position
                # Check for target or stop loss
                if current_price <= trade_plan_result["target_price"] or \
                   current_price >= trade_plan_result["stop_loss"]:
                    
                    # Buy to close short position
                    cost = abs(shares_held) * current_price * (1 + commission)
                    capital -= cost
                    trade_profit_loss = (trades[-1]['price'] - current_price) * abs(shares_held) - (2 * commission * current_price * abs(shares_held)) # Simplified P/L
                    trades.append({
                        "date": current_date,
                        "type": "COVER",
                        "price": current_price,
                        "shares": abs(shares_held),
                        "capital": capital,
                        "profit_loss": trade_profit_loss
                    })
                    # st.write(f"COVER: {abs(shares_held)} shares at {current_price:.2f} on {current_date.date()} P/L: {trade_profit_loss:.2f}")
                    shares_held = 0

    # Calculate final portfolio value
    final_portfolio_value = capital + (shares_held * df['Close'].iloc[-1])

    # Calculate performance metrics
    total_profit_loss = final_portfolio_value - initial_capital
    total_trades = len([t for t in trades if t['type'] in ['BUY', 'SHORT']])
    winning_trades = len([t for t in trades if 'profit_loss' in t and t['profit_loss'] > 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    return {
        "initial_capital": initial_capital,
        "final_capital": final_portfolio_value,
        "total_profit_loss": total_profit_loss,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "win_rate": win_rate,
        "trade_log": trades
    }


def scan_for_trades(
    tickers,
    interval,
    start_date,
    end_date,
    indicator_selection,
    weights,
    min_confidence=60,
    options_expiration_date=None
):
    """
    Scans a list of tickers for potential trade opportunities based on the defined trade plan logic.
    """
    scanned_results = []

    for ticker in tickers:
        try:
            # Generate trade plan for each ticker
            trade_plan_result = generate_directional_trade_plan(
                ticker,
                interval,
                start_date,
                end_date,
                indicator_selection,
                weights,
                options_expiration_date
            )

            # Only add to results if confidence meets the minimum threshold and a clear direction is found
            if trade_plan_result["overall_confidence"] >= min_confidence and \
               trade_plan_result["trade_direction"] in ["Bullish", "Bearish"]:
                
                # Fetch the last pivot points for display
                df = get_data(ticker, interval, start_date, end_date)
                if df.empty:
                    continue # Skip if no data
                
                # Ensure the last_day_df is correctly prepared for pivot calculation
                if interval in ['1d', '1wk', '1mo']:
                    last_day_df = df.iloc[-1:]
                else:
                    last_day_date = df.index[-1].date()
                    last_day_df = df[df.index.date == last_day_date]
                    if last_day_df.empty:
                        last_day_df = df.iloc[-1:] # Fallback to last available bar

                pivot_points_df = calculate_pivot_points(last_day_df)
                last_pivot = pivot_points_df.iloc[-1].to_dict() if not pivot_points_df.empty else {}

                # Prepare detailed entry/exit criteria for display
                entry_criteria_details = []
                exit_criteria_details = []

                # Example of adding details based on trade_plan_result
                if trade_plan_result["trade_direction"] == "Bullish":
                    entry_criteria_details.append(f"Enter between ${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} and ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}")
                    exit_criteria_details.append(f"Target Price: ${trade_plan_result.get('target_price', 'N/A'):.2f}")
                    exit_criteria_details.append(f"Stop Loss: ${trade_plan_result.get('stop_loss', 'N/A'):.2f}")
                elif trade_plan_result["trade_direction"] == "Bearish":
                    entry_criteria_details.append(f"Enter between ${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} and ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}")
                    exit_criteria_details.append(f"Target Price: ${trade_plan_result.get('target_price', 'N/A'):.2f}")
                    exit_criteria_details.append(f"Stop Loss: ${trade_plan_result.get('stop_loss', 'N/A'):.2f}")

                # Add technical signal details
                entry_criteria_details.extend(trade_plan_result.get('technical_signals', []))


                scanned_results.append({
                    "Ticker": ticker,
                    "Overall Confidence": trade_plan_result["overall_confidence"],
                    "Direction": trade_plan_result["trade_direction"],
                    "Current Price": trade_plan_result.get('current_price', 'N/A'),
                    "Target Price": trade_plan_result.get('target_price', 'N/A'),
                    "Stop Loss": trade_plan_result.get('stop_loss', 'N/A'),
                    "Entry Zone Start": f"${trade_plan_result.get('entry_zone_start', 'N/A'):.2f}",
                    "Entry Zone End": f"${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}",
                    "Reward/Risk": f"{trade_plan_result.get('reward_risk_ratio', 'N/A'):.1f}:1",
                    "Pivot (P)": f"${last_pivot.get('Pivot', 'N/A'):.2f}",
                    "Resistance 1 (R1)": f"${last_pivot.get('R1', 'N/A'):.2f}",
                    "Resistance 2 (R2)": f"${last_pivot.get('R2', 'N/A'):.2f}",
                    "Support 1 (S1)": f"${last_pivot.get('S1', 'N/A'):.2f}",
                    "Support 2 (S2)": f"${last_pivot.get('S2', 'N/A'):.2f}",
                    "Entry Criteria Details": "\n".join(entry_criteria_details),
                    "Exit Criteria Details": "\n".join(exit_criteria_details),
                    "Rationale": trade_plan_result.get('key_rationale', '')
                })

        except Exception as e:
            # print(f"Error scanning {ticker}: {e}") # For debugging
            # import traceback
            # print(traceback.format_exc()) # Uncomment for full traceback during debugging
            continue
    
    # Sort results by confidence (highest first)
    if scanned_results:
        df_results = pd.DataFrame(scanned_results)
        df_results['Overall Confidence'] = pd.to_numeric(df_results['Overall Confidence'])
        df_results = df_results.sort_values(by="Overall Confidence", ascending=False).reset_index(drop=True)
        return df_results
    else:
        return pd.DataFrame()


def test_yfinance_data_fetch():
    """A simple test function to verify yfinance data fetching."""
    print("Running yfinance data fetch test...")
    test_ticker = "MSFT"
    test_interval = "1d"
    test_start_date = datetime(2023, 1, 1).date()
    test_end_date = datetime(2023, 1, 31).date()
    
    df = get_data(test_ticker, test_interval, test_start_date, test_end_date)
    
    if not df.empty:
        print(f"Successfully fetched data for {test_ticker}. Head:\n{df.head()}")
        print(f"Tail:\n{df.tail()}")
        print(f"Columns: {df.columns.tolist()}")
    else:
        print(f"Failed to fetch data for {test_ticker}.")

# Example of how to call the test function if this script is run directly
if __name__ == "__main__":
    test_yfinance_data_fetch()
