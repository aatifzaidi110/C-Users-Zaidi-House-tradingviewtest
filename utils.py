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
from datetime import datetime, timedelta
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
        
        recom_tag = soup.find('td', text='Recom')
        analyst_recom_str = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
        
        # Extract headlines
        headlines_tags = soup.findAll('a', class_='news-link-left')
        headlines = [tag.text for tag in headlines_tags[:10]] # Get top 10 headlines
        
        analyzer = SentimentIntensityAnalyzer()
        compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        
        return {"recom_str": analyst_recom_str, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception as e:
        # st.error(f"Error fetching Finviz data for {ticker}: {e}", icon="🚨") # Don't show error in scanner loop
        return {"recom_str": "N/A", "headlines": [], "sentiment_compound": 0, "error": str(e)}

@st.cache_data(ttl=60)
def get_data(symbol, period, interval, start_date=None, end_date=None):
    """Fetches historical stock data and basic info from Yahoo Finance."""
    stock = yf.Ticker(symbol)
    try:
        # Use start/end date if provided, otherwise use period/interval
        if start_date and end_date:
            hist = stock.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
        else:
            hist = stock.history(period=period, interval=interval, auto_adjust=True)
        
        # Fetch info separately as stock.history might not always return it directly
        info = stock.info
        
        return (hist, info) if not hist.empty else (pd.DataFrame(), {})
    except Exception as e:
        # st.error(f"YFinance error fetching data for {symbol}: {e}", icon="🚫") # Don't show error in scanner loop
        return pd.DataFrame(), {}

@st.cache_data(ttl=300)
def get_options_chain(ticker, expiry_date):
    """Fetches call and put options data for a given ticker and expiry."""
    stock_obj = yf.Ticker(ticker)
    try:
        options = stock_obj.option_chain(expiry_date)
        return options.calls, options.puts
    except Exception as e:
        # st.warning(f"Could not fetch options chain for {ticker} on {expiry_date}: {e}", icon="⚠️") # Don't show error in scanner loop
        return pd.DataFrame(), pd.DataFrame()

# --- NEW: Economic Data Fetching ---
@st.cache_data(ttl=3600) # Cache for longer as economic data updates less frequently
def get_economic_data_fred(series_id, start_date, end_date):
    """
    Fetches economic data from FRED (Federal Reserve Economic Data).
    
    Args:
        series_id (str): The FRED series ID (e.g., 'GDP', 'CPIAUCSL', 'UNRATE').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.Series: A pandas Series with the economic data, or None if fetching fails.
    """
    try:
        # FRED data is often daily, weekly, monthly, or quarterly.
        # pandas_datareader handles date ranges.
        data = pdr.data.DataReader(series_id, 'fred', start_date, end_date)
        return data[series_id] # Return the specific series
    except Exception as e:
        # print(f"Error fetching FRED data for {series_id}: {e}") # For debugging
        return None

# --- NEW: Investor Sentiment Data Fetching ---
@st.cache_data(ttl=300) # VIX updates frequently, but not as fast as stock prices
def get_vix_data(start_date, end_date):
    """
    Fetches VIX (CBOE Volatility Index) data using yfinance.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: DataFrame with VIX historical data, or None.
    """
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(start=start_date, end=end_date)
        if not vix_data.empty:
            return vix_data
        return None
    except Exception as e:
        # print(f"Error fetching VIX data: {e}") # For debugging
        return None

# === Indicator Calculation ===

def calculate_indicators(df, is_intraday=False):
    """Calculates various technical indicators for a given DataFrame."""
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # --- Initial check for required columns and robust return for empty/incomplete data ---
    # Prepare a DataFrame for cleaning, or use original if already bad
    df_processed = df.copy()

    if not all(col in df_processed.columns for col in required_cols):
        print("Warning: Missing required columns for indicator calculation. Attempting to add missing indicator columns with NaN.")
        # Ensure all indicator columns are added with NaN even if initial data is incomplete
        all_indicator_cols = [
            "EMA21", "EMA50", "EMA200",
            'ichimoku_a', 'ichimoku_b', 'ichimoku_conversion_line', 'ichimoku_base_line',
            'psar', "BB_upper", "BB_lower", "BB_mavg", "RSI", "MACD", "MACD_Signal",
            "MACD_Hist", "Stoch_K", "Stoch_D", "adx", "plus_di", "minus_di", "CCI", # Existing
            "ROC" # ADD THIS LINE
        ]
        for col in all_indicator_cols:
            if col not in df_processed.columns:
                df_processed.loc[:, col] = np.nan
        # If essential columns are missing, we can't calculate anything meaningful. Return with NaNs.
        return df_processed.set_index(df.index) # Ensure index is preserved if coming from yf

    df_cleaned = df_processed.dropna(subset=required_cols).copy()

    if df_cleaned.empty:
        print("Warning: DataFrame is empty after dropping NA values. Attempting to add missing indicator columns with NaN.")
        # Ensure all indicator columns are added with NaN even if DataFrame is empty after dropna
        all_indicator_cols = [
            "EMA21", "EMA50", "EMA200",
            'ichimoku_a', 'ichimoku_b', 'ichimoku_conversion_line', 'ichimoku_base_line',
            'psar', "BB_upper", "BB_lower", "BB_mavg", "RSI", "MACD", "MACD_Signal",
            "MACD_Hist", "Stoch_K", "Stoch_D", "adx", "plus_di", "minus_di", "CCI",
            "ROC" # <--- ADD THIS LINE
        ]
        for col in all_indicator_cols:
            if col not in df_cleaned.columns:
                df_cleaned.loc[:, col] = np.nan
        return df_cleaned # Return the empty DataFrame with expected columns

    # --- End of initial checks ---


    # --- Initialize all indicator columns to NaN to ensure they always exist before calculation attempts ---
    all_indicator_cols = [
        "EMA21", "EMA50", "EMA200",
        'ichimoku_a', 'ichimoku_b', 'ichimoku_conversion_line', 'ichimoku_base_line',
        'psar', "BB_upper", "BB_lower", "BB_mavg", "RSI", "MACD", "MACD_Signal",
        "MACD_Hist", "Stoch_K", "Stoch_D", "adx", "plus_di", "minus_di", "CCI",
        "ROC" # <--- ADD THIS LINE
    ]
    for col in all_indicator_cols:
        if col not in df_cleaned.columns: # Only add if not already present
             df_cleaned.loc[:, col] = np.nan
        # Otherwise, existing (NaN or otherwise) values will be overwritten by calculations below


    # --- Indicator Calculations ---

    # EMAs
    try:
        if not df_cleaned["Close"].empty:
            df_cleaned.loc[:, "EMA21"] = ta.trend.ema_indicator(df_cleaned["Close"], 21, fillna=True)
            df_cleaned.loc[:, "EMA50"] = ta.trend.ema_indicator(df_cleaned["Close"], 50, fillna=True)
            df_cleaned.loc[:, "EMA200"] = ta.trend.ema_indicator(df_cleaned["Close"], 200, fillna=True)
    except Exception as e:
        print(f"Error calculating EMA indicators: {e}")

    # Ichimoku
    try:
        if not df_cleaned['High'].empty and not df_cleaned['Low'].empty and not df_cleaned['Close'].empty:
            df_cleaned.loc[:, 'ichimoku_a'] = ta.trend.ichimoku_a(df_cleaned['High'], df_cleaned['Low'], fillna=True)
            df_cleaned.loc[:, 'ichimoku_b'] = ta.trend.ichimoku_b(df_cleaned['High'], df_cleaned['Low'], fillna=True)
            df_cleaned.loc[:, 'ichimoku_conversion_line'] = ta.trend.ichimoku_conversion_line(df_cleaned['High'], df_cleaned['Low'], fillna=True)
            df_cleaned.loc[:, 'ichimoku_base_line'] = ta.trend.ichimoku_base_line(df_cleaned['High'], df_cleaned['Low'], fillna=True)
    except Exception as e:
        print(f"Error calculating Ichimoku indicators: {e}")

    # PSAR
    try:
        if not df_cleaned['High'].empty and not df_cleaned['Low'].empty and not df_cleaned['Close'].empty:
            df_cleaned.loc[:, 'psar'] = ta.trend.PSARIndicator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close'], fillna=True).psar()
    except Exception as e:
        print(f"Error calculating PSAR indicator: {e}")

    # Bollinger Bands
    try:
        if not df_cleaned["Close"].empty:
            bollinger_bands = ta.volatility.BollingerBands(df_cleaned["Close"], window=20, window_dev=2, fillna=True)
            df_cleaned.loc[:, "BB_upper"] = bollinger_bands.bollinger_hband()
            df_cleaned.loc[:, "BB_lower"] = bollinger_bands.bollinger_lband()
            df_cleaned.loc[:, "BB_mavg"] = bollinger_bands.bollinger_mavg()
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")

    # RSI
    try:
        if not df_cleaned["Close"].empty:
            df_cleaned.loc[:, "RSI"] = ta.momentum.rsi(df_cleaned["Close"], window=14, fillna=True)
    except Exception as e:
        print(f"Error calculating RSI: {e}")

    # MACD
    try:
        if not df_cleaned["Close"].empty:
            macd = ta.trend.MACD(df_cleaned["Close"], window_fast=12, window_slow=26, window_sign=9, fillna=True)
            df_cleaned.loc[:, "MACD"] = macd.macd()
            df_cleaned.loc[:, "MACD_Signal"] = macd.macd_signal()
            df_cleaned.loc[:, "MACD_Hist"] = macd.macd_diff()
    except Exception as e:
        print(f"Error calculating MACD: {e}")

    # Stochastic Oscillator
    try:
        if not df_cleaned["High"].empty and not df_cleaned["Low"].empty and not df_cleaned["Close"].empty:
            stoch = ta.momentum.StochasticOscillator(df_cleaned["High"], df_cleaned["Low"], df_cleaned["Close"], window=14, smooth_window=3, fillna=True)
            df_cleaned.loc[:, "Stoch_K"] = stoch.stoch()
            df_cleaned.loc[:, "Stoch_D"] = stoch.stoch_signal()
    except Exception as e:
        print(f"Error calculating Stochastic Oscillator: {e}")

    # ADX (Average Directional Index)
    try:
        if not df_cleaned["High"].empty and not df_cleaned["Low"].empty and not df_cleaned["Close"].empty:
            adx_indicator = ta.trend.ADXIndicator(df_cleaned["High"], df_cleaned["Low"], df_cleaned["Close"], window=14, fillna=True)
            df_cleaned.loc[:, "adx"] = adx_indicator.adx()
            df_cleaned.loc[:, "plus_di"] = adx_indicator.adx_pos()
            df_cleaned.loc[:, "minus_di"] = adx_indicator.adx_neg()
    except Exception as e:
        print(f"Error calculating ADX indicators: {e}")

    # CCI (Commodity Channel Index) - Added this to the list of all_indicator_cols, so adding calculation here
    try:
        if not df_cleaned["High"].empty and not df_cleaned["Low"].empty and not df_cleaned["Close"].empty:
            df_cleaned.loc[:, "CCI"] = ta.trend.cci(df_cleaned["High"], df_cleaned["Low"], df_cleaned["Close"], window=20, fillna=True)
    except Exception as e:
        print(f"Error calculating CCI indicator: {e}")
   # ROC (Rate of Change)
    try:
        if not df_cleaned["Close"].empty:
            df_cleaned.loc[:, "ROC"] = ta.momentum.roc(df_cleaned["Close"], window=14, fillna=True)
    except Exception as e:
        print(f"Error calculating ROC indicator: {e}")
    # --- END OF ROC BLOCK ---

    return df_cleaned
   
    return df_cleaned

# === Signal Generation ===
def generate_signals_for_row(row_data):
    """Generates bullish and bearish signals for a single row of data."""
    bullish_signals = {}
    bearish_signals = {}
    
    close_price = row_data.get("Close")

    # EMA Trend
    ema21 = row_data.get("EMA21")
    ema50 = row_data.get("EMA50")
    ema200 = row_data.get("EMA200")
    if ema21 and ema50 and ema200 and not pd.isna(ema21) and not pd.isna(ema50) and not pd.isna(ema200):
        bullish_signals["EMA Trend"] = ema21 > ema50 > ema200
        bearish_signals["EMA Trend"] = ema21 < ema50 < ema200
        
    # Ichimoku Cloud
    ichimoku_a = row_data.get("ichimoku_a")
    ichimoku_b = row_data.get("ichimoku_b")
    if close_price and ichimoku_a and ichimoku_b and not pd.isna(close_price) and not pd.isna(ichimoku_a) and not pd.isna(ichimoku_b):
        bullish_signals["Ichimoku Cloud"] = close_price > ichimoku_a and close_price > ichimoku_b
        bearish_signals["Ichimoku Cloud"] = close_price < ichimoku_a and close_price < ichimoku_b
    
    # Parabolic SAR
    psar = row_data.get("psar")
    if close_price and psar and not pd.isna(close_price) and not pd.isna(psar):
        bullish_signals["Parabolic SAR"] = close_price > psar
        bearish_signals["Parabolic SAR"] = close_price < psar
    
    # ADX (requires +DI and -DI from calculate_indicators)
    adx = row_data.get("adx")
    plus_di = row_data.get('plus_di')
    minus_di = row_data.get('minus_di')
    if adx and plus_di and minus_di and not pd.isna(adx) and not pd.isna(plus_di) and not pd.isna(minus_di):
        bullish_signals["ADX"] = adx > 25 and plus_di > minus_di # Strong trend and bullish direction
        bearish_signals["ADX"] = adx > 25 and minus_di > plus_di # Strong trend and bearish direction

    # RSI Momentum
    rsi = row_data.get("RSI")
    if rsi is not None and not pd.isna(rsi):
        bullish_signals["RSI Momentum"] = rsi > 50 # General bullish momentum
        bearish_signals["RSI Momentum"] = rsi < 50 # General bearish momentum

    # Stochastic
    stoch_k = row_data.get("stoch_k")
    stoch_d = row_data.get("stoch_d")
    if stoch_k is not None and stoch_d is not None and not pd.isna(stoch_k) and not pd.isna(stoch_d):
        bullish_signals["Stochastic"] = stoch_k > stoch_d and stoch_k < 80 # K-line crossing above D-line, not overbought
        bearish_signals["Stochastic"] = stoch_k < stoch_d and stoch_k > 20 # K-line crossing below D-line, not oversold

    # MACD (using crosses of MACD and Signal line)
    macd = row_data.get("macd")
    macd_signal = row_data.get("macd_signal")
    if macd is not None and macd_signal is not None and not pd.isna(macd) and not pd.isna(macd_signal):
        bullish_signals["MACD"] = macd > macd_signal # Bullish cross
        bearish_signals["MACD"] = macd < macd_signal # Bearish cross

    # Volume Spike (simplified: current volume much higher than recent average)
    volume = row_data.get("Volume")
    volume_ma = row_data.get("Volume_MA")
    if volume is not None and volume_ma is not None and not pd.isna(volume) and not pd.isna(volume_ma):
        # A bullish volume spike is high volume on an up day (close > open)
        # A bearish volume spike is high volume on a down day (close < open)
        if volume > (volume_ma * 1.5): # 50% above average
            if close_price > row_data.get("Open", close_price): # Check if it's an up day
                bullish_signals["Volume Spike"] = True
            elif close_price < row_data.get("Open", close_price): # Check if it's a down day
                bearish_signals["Volume Spike"] = True
        else:
            bullish_signals["Volume Spike"] = False
            bearish_signals["Volume Spike"] = False
    else:
        bullish_signals["Volume Spike"] = False
        bearish_signals["Volume Spike"] = False


    # CCI
    cci = row_data.get("CCI")
    if cci is not None and not pd.isna(cci):
        bullish_signals["CCI"] = cci > 100 # Overbought, but can indicate strong trend
        bearish_signals["CCI"] = cci < -100 # Oversold, but can indicate strong trend

    # ROC
    roc = row_data.get("ROC")
    if roc is not None and not pd.isna(roc):
        bullish_signals["ROC"] = roc > 0 # Positive rate of change
        bearish_signals["ROC"] = roc < 0 # Negative rate of change

    # OBV (needs previous OBV for comparison, assuming OBV values are cumulative)
    obv = row_data.get("obv")
    obv_ema = row_data.get("obv_ema")
    if obv is not None and obv_ema is not None and not pd.isna(obv) and not pd.isna(obv_ema):
        bullish_signals["OBV"] = obv > obv_ema # OBV rising above its EMA
        bearish_signals["OBV"] = obv < obv_ema # OBV falling below its EMA
    else:
        bullish_signals["OBV"] = False
        bearish_signals["OBV"] = False

    # VWAP (Intraday only - needs current price vs VWAP)
    vwap = row_data.get("VWAP")
    if vwap is not None and close_price is not None and not pd.isna(vwap) and not pd.isna(close_price):
        bullish_signals["VWAP"] = close_price > vwap
        bearish_signals["VWAP"] = close_price < vwap
    else:
        bullish_signals["VWAP"] = False
        bearish_signals["VWAP"] = False
    
    return bullish_signals, bearish_signals


# --- NEW: Helper for converting Finviz expert recommendation string to a numerical score ---
def convert_finviz_recom_to_score(recom_str):
    """Converts a Finviz recommendation string (e.g., "2.50") to a numerical score (0-100)."""
    if recom_str is None or not isinstance(recom_str, str):
        return 50 # Default to neutral if no recommendation
    
    try:
        recom_val = float(recom_str)
        # Map 1.00 (Strong Buy) to 100, 5.00 (Strong Sell) to 0
        # Linear interpolation: score = 100 - (recom_val - 1) * (100 / 4)
        score = 100 - (recom_val - 1) * 25
        return max(0, min(100, score)) # Ensure score is within 0-100
    except ValueError:
        return 50 # Default to neutral if conversion fails

# --- NEW: Function to calculate Economic Score ---
def calculate_economic_score(latest_gdp_growth, latest_cpi, latest_unemployment_rate):
    """
    Calculates an economic score (e.g., 0-100) based on key economic indicators.
    This is a simplified example; real models use more complex logic.
    
    Args:
        latest_gdp_growth (float): Latest GDP growth rate (e.g., quarterly, annualized).
        latest_cpi (float): Latest CPI (inflation) reading.
        latest_unemployment_rate (float): Latest unemployment rate.
        
    Returns:
        float: Economic score between 0 and 100.
    """
    score = 50 # Start neutral

    # Example logic:
    # GDP Growth: Positive is good for economy/market.
    if latest_gdp_growth is not None and not pd.isna(latest_gdp_growth):
        if latest_gdp_growth > 2.0: # Strong growth
            score += 15
        elif latest_gdp_growth < 0: # Contraction
            score -= 15
    
    # CPI (Inflation): Moderate is good, too high or too low is bad.
    if latest_cpi is not None and not pd.isna(latest_cpi):
        if 2.0 <= latest_cpi <= 3.0: # Ideal inflation range
            score += 10
        elif latest_cpi > 5.0: # High inflation
            score -= 15
        elif latest_cpi < 0: # Deflation
            score -= 10

    # Unemployment Rate: Lower is generally better.
    if latest_unemployment_rate is not None and not pd.isna(latest_unemployment_rate):
        if latest_unemployment_rate < 4.0: # Low unemployment
            score += 15
        elif latest_unemployment_rate > 6.0: # High unemployment
            score -= 15
            
    return max(0, min(100, score)) # Cap score between 0 and 100

# --- NEW: Function to calculate Investor Sentiment Score ---
def calculate_sentiment_score(latest_vix, historical_vix_avg=None):
    """
    Calculates an investor sentiment score (0-100) based on VIX.
    Lower VIX usually means higher sentiment (less fear).
    
    Args:
        latest_vix (float): Latest VIX reading.
        historical_vix_avg (float, optional): Historical average VIX for context.
        
    Returns:
        float: Sentiment score between 0 and 100.
    """
    score = 50 # Start neutral

    if latest_vix is None or pd.isna(latest_vix):
        return 50 # Return neutral if no VIX data

    # Example VIX thresholds (these can be adjusted)
    # VIX < 15: Low fear, high sentiment
    # VIX 15-20: Moderate fear
    # VIX > 20: High fear, low sentiment
    
    if latest_vix < 15:
        score += 30 # Very bullish sentiment
    elif 15 <= latest_vix <= 20:
        score += 10 # Moderately bullish
    elif 20 < latest_vix <= 30:
        score -= 10 # Moderately bearish
    else: # VIX > 30
        score -= 30 # Very bearish sentiment (high fear)

    # You could also compare to historical average if available
    if historical_vix_avg is not None and historical_vix_avg > 0:
        if latest_vix < historical_vix_avg * 0.8: # Significantly below average
            score += 10
        elif latest_vix > historical_vix_avg * 1.2: # Significantly above average
            score -= 10
            
    return max(0, min(100, score)) # Cap score between 0 and 100


# --- MODIFIED: calculate_confidence_score to include new components ---
def calculate_confidence_score(technical_score, sentiment_score, expert_score, economic_score, investor_sentiment_score, weights):
    """
    Calculates the overall confidence score by combining technical, sentiment,
    expert, economic, and investor sentiment scores with user-defined weights.
    
    Args:
        technical_score (float): Score from technical indicators (0-100).
        sentiment_score (float): Score from news/social sentiment (0-100).
        expert_score (float): Score from expert ratings (0-100).
        economic_score (float): Score from economic data (0-100).
        investor_sentiment_score (float): Score from investor sentiment indicators (0-100).
        weights (dict): Dictionary of weights for each component
                        (e.g., {'technical': 0.4, 'sentiment': 0.2, 'expert': 0.2, 'economic': 0.1, 'investor_sentiment': 0.1}).
                        Weights should sum to 1.
                        
    Returns:
        dict: A dictionary containing the overall score, direction, and component scores.
    """
    # Ensure scores are not None before multiplication
    technical_score = technical_score if technical_score is not None else 50
    sentiment_score = sentiment_score if sentiment_score is not None else 50
    expert_score = expert_score if expert_score is not None else 50
    economic_score = economic_score if economic_score is not None else 50
    investor_sentiment_score = investor_sentiment_score if investor_sentiment_score is not None else 50

    # Apply weights
    weighted_technical = technical_score * weights.get('technical', 0)
    weighted_sentiment = sentiment_score * weights.get('sentiment', 0)
    weighted_expert = expert_score * weights.get('expert', 0)
    weighted_economic = economic_score * weights.get('economic', 0)
    weighted_investor_sentiment = investor_sentiment_score * weights.get('investor_sentiment', 0)

    # Sum weighted scores
    overall_score = (
        weighted_technical +
        weighted_sentiment +
        weighted_expert +
        weighted_economic +
        weighted_investor_sentiment
    )

    # Determine overall direction based on the final score
    if overall_score >= 60:
        trade_direction = "Bullish"
    elif overall_score <= 40:
        trade_direction = "Bearish"
    else:
        trade_direction = "Neutral"

    return {
        'score': overall_score,
        'direction': trade_direction,
        'components': {
            'Technical': technical_score,
            'Sentiment': sentiment_score,
            'Expert': expert_score,
            'Economic': economic_score,
            'Investor Sentiment': investor_sentiment_score
        }
    }

def generate_directional_trade_plan(confidence_score, current_price, latest_row, period_interval):
    """
    Generates a detailed directional trade plan (e.g., for stocks, not options)
    based on confidence score and current market conditions.
    
    Args:
        confidence_score (dict): A dictionary with 'score' (float) and 'band' (str, e.g., "Bullish", "Bearish").
        current_price (float): The current stock price.
        latest_row (pd.Series): The latest row of the DataFrame containing indicator values (e.g., 'ATR').
        period_interval (str): The interval of the data (e.g., '1d', '5m').
        
    Returns:
        dict: A dictionary containing the trade plan details, including 'status' and 'message'.
    """
    score = confidence_score['score']
    trade_direction = confidence_score['band'] # Renamed 'band' to 'trade_direction' for clarity
    
    # Get ATR from the latest_row. Provide a default if not found.
    # ATR is crucial for calculating volatility-based entry/exit points.
    atr_value = latest_row.get('ATR')
    
    # Define a default ATR if it's missing or NaN, to prevent errors in calculations
    if atr_value is None or pd.isna(atr_value) or atr_value == 0:
        # Fallback: Use a small percentage of current price as a rough volatility estimate
        # Or, ideally, ensure ATR is calculated properly upstream.
        atr_value = current_price * 0.01 # 1% of current price as a very rough default
        if atr_value == 0: # Ensure it's not zero if current_price is zero
            atr_value = 0.1 # Minimum default ATR

    # Initialize the plan with a default error status
    plan = {
        "status": "error",
        "message": "Could not generate a trade plan. Missing data or neutral outlook.",
        "direction": "Neutral",
        "entry_zone_start": None,
        "entry_zone_end": None,
        "stop_loss": None,
        "profit_target": None,
        "reward_risk_ratio": None,
        "key_rationale": f"Overall outlook: {trade_direction} ({score:.0f}/100)."
    }

    # Define multipliers for ATR for entry, stop-loss, and profit targets
    # These are examples and should be tuned based on strategy and risk tolerance
    entry_atr_multiplier = 0.5
    stop_loss_atr_multiplier = 1.5
    profit_target_atr_multiplier_1 = 2.0

    # Only generate a detailed plan if confidence is high enough and direction is clear
    if trade_direction == "Bullish" and score >= 60: # Example threshold
        plan["status"] = "success"
        plan["direction"] = "Bullish"
        plan["Trade Type"] = "Long"
        
        # Entry Zone: Slightly below current price, based on ATR
        plan["entry_zone_start"] = current_price - (atr_value * entry_atr_multiplier)
        plan["entry_zone_end"] = current_price + (atr_value * entry_atr_multiplier) # Small range around current price
        
        # Stop Loss: Below entry, based on ATR
        plan["stop_loss"] = current_price - (atr_value * stop_loss_atr_multiplier)
        
        # Profit Target: Above entry, based on ATR
        plan["profit_target"] = current_price + (atr_value * profit_target_atr_multiplier_1)
        
        # Calculate Reward/Risk Ratio
        risk = current_price - plan["stop_loss"]
        reward = plan["profit_target"] - current_price
        plan["reward_risk_ratio"] = reward / risk if risk > 0 else float('inf')

        plan["key_rationale"] = f"Bullish outlook ({score:.0f}/100). Price expected to rise from current levels."

    elif trade_direction == "Bearish" and score <= 40: # Example threshold
        plan["status"] = "success"
        plan["direction"] = "Bearish"
        plan["Trade Type"] = "Short"
        
        # Entry Zone: Slightly above current price, based on ATR
        plan["entry_zone_start"] = current_price + (atr_value * entry_atr_multiplier)
        plan["entry_zone_end"] = current_price - (atr_value * entry_atr_multiplier) # Small range around current price
        
        # Stop Loss: Above entry, based on ATR
        plan["stop_loss"] = current_price + (atr_value * stop_loss_atr_multiplier)
        
        # Profit Target: Below entry, based on ATR
        plan["profit_target"] = current_price - (atr_value * profit_target_atr_multiplier_1)
        
        # Calculate Reward/Risk Ratio
        risk = plan["stop_loss"] - current_price
        reward = current_price - plan["profit_target"]
        plan["reward_risk_ratio"] = reward / risk if risk > 0 else float('inf')

        plan["key_rationale"] = f"Bearish outlook ({score:.0f}/100). Price expected to fall from current levels."
        
    # Ensure numerical values are rounded for display
    for key in ["entry_zone_start", "entry_zone_end", "stop_loss", "profit_target", "reward_risk_ratio"]:
        if plan[key] is not None and isinstance(plan[key], (float, int)):
            plan[key] = round(plan[key], 2)

    return plan

def get_moneyness(current_price, strike_price, option_type):
    """
    Calculates the moneyness of an option based on current stock price and strike price.
    """
    if option_type == 'call':
        if current_price > strike_price:
            return "In-the-Money (ITM)"
        elif current_price < strike_price:
            return "Out-of-the-Money (OTM)"
        else:
            return "At-the-Money (ATM)"
    elif option_type == 'put':
        if current_price < strike_price:
            return "In-the-Money (ITM)"
        elif current_price > strike_price:
            return "Out-of-the-Money (OTM)"
        else:
            return "At-the-Money (ATM)"
    return "Invalid Option Type"


def analyze_options_chain(calls_df, puts_df, current_price):
    """
    Analyzes an options chain DataFrame to identify key characteristics
    like implied volatility, open interest, and volume for calls and puts.
    """
    analysis_results = {
        "calls": {},
        "puts": {}
    }

    if not calls_df.empty:
        # Calls analysis
        calls_df['moneyness'] = calls_df.apply(lambda row: get_moneyness(current_price, row['strike'], 'call'), axis=1)
        
        # Example metrics for calls
        analysis_results["calls"]["total_volume"] = calls_df['volume'].sum()
        analysis_results["calls"]["total_open_interest"] = calls_df['openInterest'].sum()
        
        # Average IV for ITM, OTM, ATM calls
        itm_calls_iv = calls_df[calls_df['moneyness'] == "In-the-Money (ITM)"]['impliedVolatility'].mean()
        otm_calls_iv = calls_df[calls_df['moneyness'] == "Out-of-the-Money (OTM)"]['impliedVolatility'].mean()
        atm_calls_iv = calls_df[calls_df['moneyness'] == "At-the-Money (ATM)"]['impliedVolatility'].mean()
        analysis_results["calls"]["avg_iv_itm"] = itm_calls_iv if not pd.isna(itm_calls_iv) else 0
        analysis_results["calls"]["avg_iv_otm"] = otm_calls_iv if not pd.isna(otm_calls_iv) else 0
        analysis_results["calls"]["avg_iv_atm"] = atm_calls_iv if not pd.isna(atm_calls_iv) else 0

    if not puts_df.empty:
        # Puts analysis
        puts_df['moneyness'] = puts_df.apply(lambda row: get_moneyness(current_price, row['strike'], 'put'), axis=1)
        
        # Example metrics for puts
        analysis_results["puts"]["total_volume"] = puts_df['volume'].sum()
        analysis_results["puts"]["total_open_interest"] = puts_df['openInterest'].sum()

        # Average IV for ITM, OTM, ATM puts
        itm_puts_iv = puts_df[puts_df['moneyness'] == "In-the-Money (ITM)"]['impliedVolatility'].mean()
        otm_puts_iv = puts_df[puts_df['moneyness'] == "Out-of-the-Money (OTM)"]['impliedVolatility'].mean()
        atm_puts_iv = puts_df[puts_df['moneyness'] == "At-the-Money (ATM)"]['impliedVolatility'].mean()
        analysis_results["puts"]["avg_iv_itm"] = itm_puts_iv if not pd.isna(itm_puts_iv) else 0
        analysis_results["puts"]["avg_iv_otm"] = otm_puts_iv if not pd.isna(otm_puts_iv) else 0
        analysis_results["puts"]["avg_iv_atm"] = atm_puts_iv if not pd.isna(atm_puts_iv) else 0

    return analysis_results


def suggest_options_strategy(ticker, confidence_score_value, current_stock_price, expirations, trade_direction):
    """
    Suggests an options strategy based on confidence score, current price, and trade direction.
    
    Args:
        ticker (str): Stock ticker symbol.
        confidence_score_value (float): The overall confidence score (e.g., 0-100).
        current_stock_price (float): The current price of the stock.
        expirations (list): List of available expiration dates.
        trade_direction (str): The anticipated trade direction ("Bullish", "Bearish", "Neutral").
        
    Returns:
        dict: A dictionary containing the suggested strategy details or an error message.
    """
    score = confidence_score_value 

    plan = {
        "status": "error",
        "message": "Could not suggest an options strategy. Neutral outlook or insufficient data.",
        "Strategy": "N/A",
        "Direction": "Neutral",
        "Expiration": "N/A",
        "Net Debit/Credit": "N/A",
        "Max Profit": "N/A",
        "Max Loss": "N/A",
        "Break-even": "N/A",
        "Notes": "No specific strategy recommended for this outlook.",
        "Contracts": {},
        "option_legs_for_chart": []
    }

    if trade_direction == "Bullish" and score >= 60:
        if not expirations:
            plan["message"] = "No expiration dates available for options strategy."
            return plan
        
        selected_expiration = expirations[0]

        buy_strike = current_stock_price * 0.98
        sell_strike = current_stock_price * 1.02

        if buy_strike >= sell_strike:
            buy_strike = current_stock_price * 0.95
            sell_strike = current_stock_price * 1.05

        buy_premium = (current_stock_price - buy_strike) * 0.5 + 1.0
        sell_premium = (current_stock_price - sell_strike) * 0.1 + 0.5
        
        if buy_premium <= 0: buy_premium = 1.0
        if sell_premium <= 0: sell_premium = 0.5

        net_debit = (buy_premium - sell_premium) * 100

        buy_contract_details = {
            'strike': round(buy_strike, 2),
            'lastPrice': round(buy_premium, 2),
            'bid': round(buy_premium * 0.95, 2),
            'ask': round(buy_premium * 1.05, 2),
            'volume': 1000,
            'openInterest': 5000,
            'impliedVolatility': 0.30,
            'delta': 0.65,
            'theta': -0.03,
            'gamma': 0.01
        }
        sell_contract_details = {
            'strike': round(sell_strike, 2),
            'lastPrice': round(sell_premium, 2),
            'bid': round(sell_premium * 0.95, 2),
            'ask': round(sell_premium * 1.05, 2),
            'volume': 800,
            'openInterest': 4000,
            'impliedVolatility': 0.28,
            'delta': 0.35,
            'theta': -0.05,
            'gamma': 0.005
        }

        max_profit_per_share = (sell_strike - buy_strike) - (buy_premium - sell_premium)
        max_risk_per_share = (buy_premium - sell_premium)
        
        max_profit = max_profit_per_share * 100
        max_risk = max_risk_per_share * 100

        max_profit_display = f"${max_profit:.2f}" if max_profit > 0 else "N/A"
        max_risk_display = f"${max_risk:.2f}" if max_risk > 0 else "N/A"

        plan.update({
            "status": "success",
            "message": "Bull Call Spread recommended for a moderately bullish outlook.",
            "Strategy": "Bull Call Spread",
            "Direction": "Bullish",
            "Expiration": selected_expiration,
            "Buy Strike": buy_contract_details['strike'],
            "Sell Strike": sell_contract_details['strike'],
            "Net Debit": f"${net_debit:.2f}",
            "Max Profit": max_profit_display,
            "Max Risk": max_risk_display,
            "Reward / Risk": f"{max_profit_per_share / max_risk_per_share:.1f}:1" if max_risk_per_share > 0 else "Unlimited",
            "Contracts": {
                "Buy": buy_contract_details,
                "Sell": sell_contract_details
            },
            "option_legs_for_chart": [
                {'strike': buy_contract_details['strike'], 'type': 'call', 'action': 'buy', 'premium': buy_contract_details['lastPrice']},
                {'strike': sell_contract_details['strike'], 'type': 'call', 'action': 'sell', 'premium': sell_contract_details['lastPrice']}
            ]
        })

    elif trade_direction == "Bearish" and score <= 40:
        if not expirations:
            plan["message"] = "No expiration dates available for options strategy."
            return plan
        
        selected_expiration = expirations[0]

        buy_strike = current_stock_price * 1.02
        sell_strike = current_stock_price * 0.98

        if buy_strike <= sell_strike:
            buy_strike = current_stock_price * 1.05
            sell_strike = current_stock_price * 0.95

        buy_premium = (buy_strike - current_stock_price) * 0.5 + 1.0
        sell_premium = (sell_strike - current_stock_price) * 0.1 + 0.5

        if buy_premium <= 0: buy_premium = 1.0
        if sell_premium <= 0: sell_premium = 0.5

        net_debit = (buy_premium - sell_premium) * 100

        buy_contract_details = {
            'strike': round(buy_strike, 2),
            'lastPrice': round(buy_premium, 2),
            'bid': round(buy_premium * 0.95, 2),
            'ask': round(buy_premium * 1.05, 2),
            'volume': 900,
            'openInterest': 4500,
            'impliedVolatility': 0.32,
            'delta': -0.65,
            'theta': -0.04,
            'gamma': 0.01
        }
        sell_contract_details = {
            'strike': round(sell_strike, 2),
            'lastPrice': round(sell_premium, 2),
            'bid': round(sell_premium * 0.95, 2),
            'ask': round(sell_premium * 1.05, 2),
            'volume': 700,
            'openInterest': 3500,
            'impliedVolatility': 0.30,
            'delta': -0.35,
            'theta': -0.06,
            'gamma': 0.005
        }

        max_profit_per_share = (buy_strike - sell_strike) - (buy_premium - sell_premium)
        max_risk_per_share = (buy_premium - sell_premium)
        
        max_profit = max_profit_per_share * 100
        max_risk = max_risk_per_share * 100

        max_profit_display = f"${max_profit:.2f}" if max_profit > 0 else "N/A"
        max_risk_display = f"${max_risk:.2f}" if max_risk > 0 else "N/A"

        plan.update({
            "status": "success",
            "message": "Bear Put Spread recommended for a moderately bearish outlook.",
            "Strategy": "Bear Put Spread",
            "Direction": "Bearish",
            "Expiration": selected_expiration,
            "Buy Strike": buy_contract_details['strike'],
            "Sell Strike": sell_contract_details['strike'],
            "Net Debit": f"${net_debit:.2f}",
            "Max Profit": max_profit_display,
            "Max Risk": max_risk_display,
            "Reward / Risk": f"{max_profit_per_share / max_risk_per_share:.1f}:1" if max_risk_per_share > 0 else "Unlimited",
            "Contracts": {
                "Buy": buy_contract_details,
                "Sell": sell_contract_details
            },
            "option_legs_for_chart": [
                {'strike': buy_contract_details['strike'], 'type': 'put', 'action': 'buy', 'premium': buy_contract_details['lastPrice']},
                {'strike': sell_contract_details['strike'], 'type': 'put', 'action': 'sell', 'premium': sell_contract_details['lastPrice']}
            ]
        })

    return plan


# === Backtesting Logic ===

def backtest_strategy(df_historical, selection, atr_multiplier=1.5, reward_risk_ratio=2.0, signal_threshold_percentage=0.7, trade_direction="long", exit_strategy="fixed_rr"):
    """
    Simulates trades and returns a list of trades and key performance metrics.
    exit_strategy: 'fixed_rr' or 'trailing_psar'.
    """
    trades = []
    in_trade = False
    
    # Clean data and ensure required columns are present
    df_clean = df_historical.dropna().copy()
    if len(df_clean) < 200:
        # st.info("Not enough data for robust backtesting.") # Don't show in scanner loop
        return [], {"error": "Insufficient data"}

    # Ensure ATR is calculated before starting the loop for backtesting
    if 'ATR' not in df_clean.columns:
        df_clean.loc[:, 'ATR'] = ta.volatility.AverageTrueRange(df_clean['High'], df_clean['Low'], df_clean['Close'], fillna=True).average_true_range()
        df_clean.dropna(subset=['ATR'], inplace=True) # Drop rows where ATR is NaN
        if df_clean.empty:
            # st.error("Not enough data after calculating ATR for backtesting.") # Don't show in scanner loop
            return [], {"error": "Insufficient data after ATR calculation"}


    for i in range(1, len(df_clean)):
        current_day = df_clean.iloc[i]
        prev_day = df_clean.iloc[i-1]

        # --- Exit Logic ---
        if in_trade:
            exit_reason = None
            if trade_direction == "long":
                # NEW: Trailing PSAR exit
                if exit_strategy == 'trailing_psar' and 'psar' in df_clean.columns and prev_day.get('psar') is not None and not pd.isna(prev_day.get('psar')):
                    stop_loss = max(stop_loss, prev_day['psar'])

                if current_day['Low'] <= stop_loss:
                    exit_price, exit_reason = stop_loss, "Stop-Loss"
                elif current_day['High'] >= take_profit and exit_strategy == 'fixed_rr':
                    exit_price, exit_reason = take_profit, "Take-Profit"
            
            elif trade_direction == "short":
                 # NEW: Trailing PSAR exit for short
                if exit_strategy == 'trailing_psar' and 'psar' in df_clean.columns and prev_day.get('psar') is not None and not pd.isna(prev_day.get('psar')):
                    stop_loss = min(stop_loss, prev_day['psar']) # For short, PSAR trailing stop moves down

                if current_day['High'] >= stop_loss: # For short, if price goes above SL
                    exit_price, exit_reason = stop_loss, "Stop-Loss"
                elif current_day['Low'] <= take_profit and exit_strategy == 'fixed_rr': # For short, if price goes below TP
                    exit_price, exit_reason = take_profit, "Take-Profit"

            if exit_reason:
                pnl = exit_price - entry_price if trade_direction == "long" else entry_price - exit_price
                trades.append({
                    "Exit Date": current_day.name.strftime('%Y-%m-%d'),
                    "Type": f"Exit ({'Win' if pnl > 0 else 'Loss'})",
                    "Price": round(exit_price, 2), "PnL": round(pnl, 2)
                })
                in_trade = False
            
        # --- Entry Logic ---
        if not in_trade:
            bullish_signals, bearish_signals = generate_signals_for_row(prev_day)
            
            fired_signals_count = 0
            total_selected_directional_indicators = 0

            # Count signals based on `selection` (from app.py) and `trade_direction`
            for indicator_name_full, is_selected in selection.items():
                if is_selected:
                    # Map full indicator name from selection to simplified signal name
                    signal_key = indicator_name_full.split('(')[0].strip() # e.g., "EMA Trend" from "EMA Trend (21, 50, 200)"

                    # Only consider directional indicators for the count
                    if signal_key not in ["Bollinger Bands", "Pivot Points"]: # These are display-only or non-directional for signal counting
                        total_selected_directional_indicators += 1
                        if trade_direction == "long" and bullish_signals.get(signal_key, False):
                            fired_signals_count += 1
                        elif trade_direction == "short" and bearish_signals.get(signal_key, False):
                            fired_signals_count += 1
                            
                        # Special handling for VWAP if selected and applicable
                        if signal_key == "VWAP" and prev_day.get('VWAP') is not None and not pd.isna(prev_day.get('VWAP')): # VWAP is intraday, check if data exists
                             if trade_direction == "long" and bullish_signals.get("VWAP", False):
                                 fired_signals_count += 1
                             elif trade_direction == "short" and bearish_signals.get("VWAP", False):
                                 fired_signals_count += 1
                                 
            # Backtest entry criteria
            if total_selected_directional_indicators > 0 and (fired_signals_count / total_selected_directional_indicators) >= signal_threshold_percentage:
                entry_price = current_day['Open']
                atr = prev_day['ATR']
                
                if atr > 0:
                    if trade_direction == "long":
                        stop_loss = entry_price - (atr * atr_multiplier)
                        take_profit = entry_price + (atr * atr_multiplier * reward_risk_ratio)
                    elif trade_direction == "short":
                        stop_loss = entry_price + (atr * atr_multiplier)
                        take_profit = entry_price - (atr * atr_multiplier * reward_risk_ratio)
                    
                    trades.append({
                        "Entry Date": current_day.name.strftime('%Y-%m-%d'),
                        "Type": f"Entry ({trade_direction.capitalize()})", "Price": round(entry_price, 2)
                    })
                    in_trade = True
            
    # --- Calculate Performance Metrics ---
    wins = [t['PnL'] for t in trades if 'PnL' in t and t['PnL'] > 0]
    losses = [t['PnL'] for t in trades if 'PnL' in t and t['PnL'] < 0]
    
    total_completed_trades = len(wins) + len(losses)
    win_rate = len(wins) / total_completed_trades if total_completed_trades > 0 else 0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (float('inf') if gross_win > 0 else 0)
    
    performance = {
        "Total Trades": total_completed_trades,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate": f"{win_rate:.2%}",
        "Gross Profit": round(gross_win, 2),
        "Gross Loss": round(gross_loss, 2),
        "Profit Factor": round(profit_factor, 2) if profit_factor != float('inf') else "Infinite",
        "Net PnL": round(sum(wins) + sum(losses), 2)
    }

    return trades, performance


def calculate_pivot_points(df):
    """Calculates classical pivot points for a DataFrame."""
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        # st.warning("DataFrame is missing High, Low, or Close columns for pivot point calculation.") # Don't show in scanner loop
        return pd.DataFrame(index=df.index)

    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_close = df['Close'].shift(1)

    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = (2 * pivot) - prev_low
    s1 = (2 * pivot) - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)

    pivots_df = pd.DataFrame({
        'Pivot': pivot,
        'R1': r1,
        'S1': s1,
        'R2': r2,
        'S2': s2,
        'R3': r3,
        'S3': s3
    }, index=df.index)

    return pivots_df

# --- NEW HELPER: Get Indicator Summary Text for Scanner ---
def get_indicator_summary_text(signal_name_base, current_value, bullish_fired, bearish_fired):
    """
    Generates a concise text summary for a single technical indicator, suitable for scanner results.
    """
    summary = f"**{signal_name_base}:** "
    value_str = f"Current: {current_value:.2f}" if current_value is not None and not pd.isna(current_value) else "Current: N/A"

    if "ADX" in signal_name_base:
        if current_value is not None and not pd.isna(current_value):
            if current_value > 25:
                status = "Strong Trend"
            elif current_value < 20:
                status = "Weak/No Trend"
            else:
                status = "Developing Trend"
            summary += f"{status} ({value_str}). Ideal Strong Trend: >25."
        else:
            summary += f"N/A ({value_str})."
    elif "EMA Trend" in signal_name_base:
        if bullish_fired:
            summary += f"Bullish Trend Confirmed. Ideal: 21>50>200 EMA."
        elif bearish_fired:
            summary += f"Bearish Trend Confirmed. Ideal: 21<50<200 EMA."
        else:
            summary += f"Neutral/Consolidating Trend. Ideal: Clear EMA alignment."
    elif "Ichimoku Cloud" in signal_name_base:
        if bullish_fired:
            summary += f"Bullish Ichimoku Signal. Ideal: Price above Cloud, Tenkan above Kijun."
        elif bearish_fired:
            summary += f"Bearish Ichimoku Signal. Ideal: Price below Cloud, Tenkan below Kijun."
        else:
            summary += f"Neutral/Mixed Ichimoku Signals. Ideal: Clear alignment."
    elif "Parabolic SAR" in signal_name_base:
        if bullish_fired:
            summary += f"Bullish PSAR (dots below price). Ideal: PSAR dots below price."
        elif bearish_fired:
            summary += f"Bearish PSAR (dots above price). Ideal: PSAR dots above price."
        else:
            summary += f"N/A (no clear signal) ({value_str})."
    elif "RSI Momentum" in signal_name_base:
        if current_value is not None and not pd.isna(current_value):
            if current_value > 70:
                status = "Overbought"
            elif current_value < 30:
                status = "Oversold"
            elif current_value > 50:
                status = "Bullish Momentum"
            else:
                status = "Bearish Momentum"
            summary += f"{status} ({value_str}). Ideal Bullish: Rising from 30-70. Ideal Bearish: Falling from 70-30."
        else:
            summary += f"N/A ({value_str})."
    elif "Stochastic" in signal_name_base: # Assuming this is Stochastic Oscillator
        if current_value is not None and not pd.isna(current_value):
            if current_value > 80:
                status = "Overbought"
            elif current_value < 20:
                status = "Oversold"
            else:
                status = "Neutral"
            summary += f"{status} ({value_str}). Ideal Bullish: %K above %D (below 20). Ideal Bearish: %K below %D (above 80)."
        else:
            summary += f"N/A ({value_str})."
    elif "CCI" in signal_name_base:
        if current_value is not None and not pd.isna(current_value):
            if current_value > 100:
                status = "Strong Bullish"
            elif current_value < -100:
                status = "Strong Bearish"
            elif current_value > 0:
                status = "Bullish Bias"
            else:
                status = "Bearish Bias"
            summary += f"{status} ({value_str}). Ideal Bullish: >100. Ideal Bearish: <-100."
        else:
            summary += f"N/A ({value_str})."
    elif "ROC" in signal_name_base:
        if current_value is not None and not pd.isna(current_value):
            if current_value > 0:
                status = "Positive Momentum"
            else:
                status = "Negative Momentum"
            summary += f"{status} ({value_str}). Ideal Bullish: >0. Ideal Bearish: <0."
        else:
            summary += f"N/A ({value_str})."
    elif "Volume Spike" in signal_name_base:
        if bullish_fired:
            summary += f"Bullish Volume Spike Detected. Ideal: High volume on rising prices."
        elif bearish_fired:
            summary += f"Bearish Volume Spike Detected. Ideal: High volume on falling prices."
        else:
            summary += f"Normal Volume. Ideal: High volume on breakouts."
    elif "OBV" in signal_name_base:
        if bullish_fired:
            summary += f"Rising OBV (Accumulation). Ideal: Rising OBV."
        elif bearish_fired:
            summary += f"Falling OBV (Distribution). Ideal: Falling OBV."
        else:
            summary += f"Sideways OBV (Indecision). Ideal: OBV confirms price trend."
    elif "VWAP" in signal_name_base:
        if current_value is not None and not pd.isna(current_value):
            if bullish_fired:
                status = "Price Above VWAP"
            elif bearish_fired:
                status = "Price Below VWAP"
            else:
                status = "Price Near VWAP"
            summary += f"{status} ({value_str}). Ideal Bullish: Price consistently above VWAP. Ideal Bearish: Price consistently below VWAP."
        else:
            summary += f"N/A ({value_str})."

    return summary


# --- NEW: Stock Scanner Function to include detailed trade plan ---
def run_stock_scanner(
    ticker_list,
    trading_style,
    min_confidence,
    indicator_selection,
    confidence_weights
):
    """
    Scans a list of tickers for trading opportunities based on selected style and confidence,
    including detailed trade plan elements.

    Args:
        ticker_list (list): List of stock ticker symbols to scan.
        trading_style (str): Desired trading style (e.g., "Day Trading Long", "Swing Trading Call").
        min_confidence (int): Minimum overall confidence score required (0-100).
        indicator_selection (dict): Dictionary of selected technical indicators.
        confidence_weights (dict): Weights for confidence score components.

    Returns:
        pd.DataFrame: A DataFrame of qualifying tickers with relevant metrics and trade plan details.
    """
    scanned_results = []
    today = datetime.today()
    one_year_ago = today - timedelta(days=365) # Use timedelta directly r historical data context

    for ticker in ticker_list:
        try:
            # 1. Fetch Data
            # get_data now returns (hist_df, info_dict)
            df_hist, info_data = get_data(ticker, "1y", "1d", one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
            
            if df_hist.empty:
                # print(f"Skipping {ticker}: No historical data found.") # For debugging
                continue

            current_price = df_hist['Close'].iloc[-1]
            if pd.isna(current_price):
                # print(f"Skipping {ticker}: Current price is NaN.") # For debugging
                continue

            # Calculate indicators for the full historical data
            df_calculated = (df_hist.copy())
            
            # Get Finviz data
            finviz_data = get_finviz_data(ticker)

            # 2. Calculate Confidence Scores
            last_row = df_calculated.iloc[-1]
            bullish_signals, bearish_signals = generate_signals_for_row(last_row)
            
            tech_score_raw = 0
            total_possible_tech_points = 0
            
            # Define all possible indicator names for consistent scoring
            all_indicator_names = [
                "EMA Trend", "Ichimoku Cloud", "Parabolic SAR", "ADX",
                "RSI Momentum", "Stochastic", "MACD", "Volume Spike",
                "CCI", "ROC", "OBV", "VWAP"
            ]

            for ind_name in all_indicator_names:
                is_selected = indicator_selection.get(ind_name, False) # Check if user selected it
                if is_selected:
                    total_possible_tech_points += 1 
                    # Check for specific bullish signals based on indicator name
                    if ind_name == "EMA Trend" and bullish_signals.get("EMA Trend", False):
                        tech_score_raw += 1
                    elif ind_name == "Ichimoku Cloud" and bullish_signals.get("Ichimoku Cloud", False):
                        tech_score_raw += 1
                    elif ind_name == "Parabolic SAR" and bullish_signals.get("Parabolic SAR", False):
                        tech_score_raw += 1
                    elif ind_name == "ADX" and bullish_signals.get("ADX", False): # ADX signal includes direction
                        tech_score_raw += 1
                    elif ind_name == "RSI Momentum" and bullish_signals.get("RSI Momentum", False):
                        tech_score_raw += 1
                    elif ind_name == "Stochastic" and bullish_signals.get("Stochastic", False):
                        tech_score_raw += 1
                    elif ind_name == "MACD" and bullish_signals.get("MACD", False):
                        tech_score_raw += 1
                    elif ind_name == "Volume Spike" and bullish_signals.get("Volume Spike", False):
                        tech_score_raw += 1
                    elif ind_name == "CCI" and bullish_signals.get("CCI", False):
                        tech_score_raw += 1
                    elif ind_name == "ROC" and bullish_signals.get("ROC", False):
                        tech_score_raw += 1
                    elif ind_name == "OBV" and bullish_signals.get("OBV", False):
                        tech_score_raw += 1
                    elif ind_name == "VWAP" and bullish_signals.get("VWAP", False):
                        tech_score_raw += 1
                    
                    # Check for specific bearish signals and subtract
                    if ind_name == "EMA Trend" and bearish_signals.get("EMA Trend", False):
                        tech_score_raw -= 1
                    elif ind_name == "Ichimoku Cloud" and bearish_signals.get("Ichimoku Cloud", False):
                        tech_score_raw -= 1
                    elif ind_name == "Parabolic SAR" and bearish_signals.get("Parabolic SAR", False):
                        tech_score_raw -= 1
                    elif ind_name == "ADX" and bearish_signals.get("ADX", False):
                        tech_score_raw -= 1
                    elif ind_name == "RSI Momentum" and bearish_signals.get("RSI Momentum", False):
                        tech_score_raw -= 1
                    elif ind_name == "Stochastic" and bearish_signals.get("Stochastic", False):
                        tech_score_raw -= 1
                    elif ind_name == "MACD" and bearish_signals.get("MACD", False):
                        tech_score_raw -= 1
                    elif ind_name == "Volume Spike" and bearish_signals.get("Volume Spike", False):
                        tech_score_raw -= 1
                    elif ind_name == "CCI" and bearish_signals.get("CCI", False):
                        tech_score_raw -= 1
                    elif ind_name == "ROC" and bearish_signals.get("ROC", False):
                        tech_score_raw -= 1
                    elif ind_name == "OBV" and bearish_signals.get("OBV", False):
                        tech_score_raw -= 1
                    elif ind_name == "VWAP" and bearish_signals.get("VWAP", False):
                        tech_score_raw -= 1

            if total_possible_tech_points > 0:
                technical_score_current = ((tech_score_raw + total_possible_tech_points) / (2 * total_possible_tech_points)) * 100
            else:
                technical_score_current = 50 # Neutral if no selected indicators were directional

            sentiment_score_current = finviz_data.get("sentiment_compound", 0) * 100 # Use the compound score directly

            expert_recom_str = info_data.get('recommendationMean', None)
            expert_score_current = convert_finviz_recom_to_score(str(expert_recom_str))

            latest_gdp = get_economic_data_fred('GDP', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
            latest_cpi = get_economic_data_fred('CPIAUCSL', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
            latest_unemployment = get_economic_data_fred('UNRATE', one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))

            economic_score_current = calculate_economic_score(
                latest_gdp.iloc[-1] if latest_gdp is not None and not latest_gdp.empty else None,
                latest_cpi.iloc[-1] if latest_cpi is not None and not latest_cpi.empty else None,
                latest_unemployment.iloc[-1] if latest_unemployment is not None and not latest_unemployment.empty else None
            )

            vix_data = get_vix_data(one_year_ago.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
            latest_vix = vix_data['Close'].iloc[-1] if vix_data is not None and not vix_data.empty else None
            historical_vix_avg = vix_data['Close'].mean() if vix_data is not None and not vix_data.empty else None
            investor_sentiment_score_current = calculate_sentiment_score(latest_vix, historical_vix_avg)

            confidence_results = calculate_confidence_score(
                technical_score_current,
                sentiment_score_current,
                expert_score_current,
                economic_score_current,
                investor_sentiment_score_current,
                confidence_weights
            )
            overall_confidence = confidence_results['score']
            trade_direction = confidence_results['direction']

            # 3. Calculate Trade Plan Details and Support/Resistance
            confidence_for_plan = {
                'score': overall_confidence,
                'band': trade_direction
            }
            
            trade_plan_interval = '1d' # Default for swing
            if "Day Trading" in trading_style:
                trade_plan_interval = '60m' # Assuming hourly data for day trading plans
                # For more accurate day trading, you'd ideally fetch and calculate indicators on 60m data here
                # For simplicity, we'll use daily ATR from last_row, but it's a limitation for intraday
            
            atr_val = last_row.get('ATR')
            if atr_val is None or pd.isna(atr_val) or atr_val == 0:
                atr_val = (last_row['High'] - last_row['Low']) * 0.01 # Fallback for ATR

            trade_plan_result = generate_directional_trade_plan(
                confidence_for_plan,
                current_price,
                last_row, # Pass the full latest_row
                trade_plan_interval
            )

            df_pivots = calculate_pivot_points(df_hist.copy())
            last_pivot = df_pivots.iloc[-1] if not df_pivots.empty else {}

            # 4. Generate Detailed Indicator Descriptions for Entry/Exit Rationale
            entry_criteria_details = []
            exit_criteria_details = []

            entry_criteria_details.append(f"**Overall Outlook:** {trade_plan_result.get('key_rationale', 'N/A')}")
            entry_criteria_details.append(f"**Entry Zone:** Between ${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} and ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}.")
            exit_criteria_details.append(f"**Stop-Loss:** Close {'below' if trade_direction == 'Bullish' else 'above'} ${trade_plan_result.get('stop_loss', 'N/A'):.2f}.")
            exit_criteria_details.append(f"**Profit Target:** Around ${trade_plan_result.get('profit_target', 'N/A'):.2f} ({trade_plan_result.get('reward_risk_ratio', 'N/A'):.1f}:1 Reward/Risk).")
            
            entry_criteria_details.append("\n**Current Indicator Status:**")
            for ind_name in all_indicator_names: # Iterate through all indicator names
                is_selected = indicator_selection.get(ind_name, False)
                if is_selected:
                    # Map indicator name from selection to the key in last_row
                    current_ind_value = None
                    if ind_name == "RSI Momentum": current_ind_value = last_row.get("RSI")
                    elif ind_name == "Stochastic": current_ind_value = last_row.get("stoch_k")
                    elif ind_name == "ADX": current_ind_value = last_row.get("adx")
                    elif ind_name == "CCI": current_ind_value = last_row.get("cci")
                    elif ind_name == "ROC": current_ind_value = last_row.get("roc")
                    elif ind_name == "OBV": current_ind_value = last_row.get("obv")
                    elif ind_name == "VWAP": current_ind_value = last_row.get("VWAP")
                    # For EMA Trend, Ichimoku, Parabolic SAR, Volume Spike, their 'current value' is often implicit in signal
                    # For these, get_indicator_summary_text relies more on bullish_fired/bearish_fired
                    
                    summary_text = get_indicator_summary_text(
                        ind_name,
                        current_ind_value,
                        bullish_signals.get(ind_name, False), # Pass signal directly
                        bearish_signals.get(ind_name, False) # Pass signal directly
                    )
                    entry_criteria_details.append(f"- {summary_text}")

            # 5. Apply Filtering Logic and Store Results
            if overall_confidence >= min_confidence:
                if trade_plan_result['status'] != 'success':
                    # print(f"Skipping {ticker}: Trade plan generation failed for {trade_direction} direction.") # For debugging
                    continue

                qualifies = False
                if trading_style == "Day Trading Long" and trade_direction == "Bullish":
                    if atr_val is not None and not pd.isna(atr_val) and atr_val > 0.5: # ATR threshold for day trading
                        qualifies = True
                elif trading_style == "Day Trading Short" and trade_direction == "Bearish":
                    if atr_val is not None and not pd.isna(atr_val) and atr_val > 0.5:
                        qualifies = True
                elif trading_style == "Swing Trading Call" and trade_direction == "Bullish":
                    qualifies = True
                elif trading_style == "Swing Trading Put" and trade_direction == "Bearish":
                    qualifies = True

                if qualifies:
                    scanned_results.append({
                        "Ticker": ticker,
                        "Trading Style": trading_style,
                        "Overall Confidence": f"{overall_confidence:.0f}",
                        "Direction": trade_direction,
                        "Current Price": f"${current_price:.2f}",
                        "ATR": f"{atr_val:.2f}",
                        "Target Price": f"${trade_plan_result.get('profit_target', 'N/A'):.2f}",
                        "Stop Loss": f"${trade_plan_result.get('stop_loss', 'N/A'):.2f}",
                        "Entry Zone": f"${trade_plan_result.get('entry_zone_start', 'N/A'):.2f} - ${trade_plan_result.get('entry_zone_end', 'N/A'):.2f}",
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
        df_results = df_results.sort_values(by='Overall Confidence', ascending=False).reset_index(drop=True)
        return df_results
    return pd.DataFrame() # Return empty DataFrame if no results

