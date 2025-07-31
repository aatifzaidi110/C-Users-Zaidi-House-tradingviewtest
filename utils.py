# utils.py - Final Version (with distutils workaround for Python 3.10+ 0730) 
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

        data = {}
        # Attempt to extract recommendation score (this might vary)
        # Example: looking for a specific table or div that contains analyst ratings
        # This part might need frequent adjustment due to website changes
        
        # A more robust way might be to look for text near "Analyst Recommendation"
        # For demonstration, let's assume a simple lookup.
        # Finviz structure is complex; this is a simplified placeholder.
        
        # Example: Extracting news sentiment (this is illustrative, Finviz doesn't provide a direct score)
        # We'll simulate this with VADER on news headlines if available later.
        data['recom_score'] = "N/A" # Placeholder
        data['news_sentiment_score'] = "N/A" # Placeholder

        # Try to find a table containing descriptive data
        main_table = soup.find('table', class_='snapshot-table2')
        if main_table:
            rows = main_table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                for i in range(len(cols)):
                    text = cols[i].get_text(strip=True)
                    if "Analyst Recommendation" in text and i + 1 < len(cols):
                        data['recom_score'] = cols[i+1].get_text(strip=True)
                        break
        
        # For news sentiment, you'd typically scrape news headlines and run VADER
        # This is beyond a simple Finviz scrape for a single score.
        # Let's add a placeholder for now and assume a more complex news scraping for sentiment analysis if needed.

        return data
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch Finviz data for {ticker}: {e}. Skipping sentiment analysis from Finviz.")
        return {}
    except Exception as e:
        st.warning(f"Error parsing Finviz data for {ticker}: {e}. Skipping sentiment analysis from Finviz.")
        return {}


@st.cache_data(ttl=900)
def get_vix_data(start_date, end_date):
    """Fetches VIX (CBOE Volatility Index) historical data."""
    try:
        vix_df = yf.download('^VIX', start=start_date, end=end_date)
        return vix_df
    except Exception as e:
        st.warning(f"Could not fetch VIX data: {e}. Skipping VIX analysis.")
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_economic_data_fred(series_id, start_date, end_date):
    """Fetches economic data from FRED."""
    try:
        # FRED data reader often expects datetime objects
        data = pdr.DataReader(series_id, 'fred', start_date, end_date)
        return data
    except Exception as e:
        st.warning(f"Could not fetch FRED data for {series_id}: {e}. Skipping economic analysis for this series.")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_data(ticker, interval, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    try:
        # Convert start_date and end_date to datetime objects for yf.download
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        df = yf.download(ticker, start=start_datetime, end=end_datetime, interval=interval)
        if df.empty:
            st.warning(f"No data fetched for {ticker} with interval {interval} from {start_date} to {end_date}. Please check the ticker, dates, and interval.")
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=900)
def get_options_chain(ticker, expiration_date=None):
    """
    Fetches options chain data for a given ticker and optional expiration date.
    Returns two DataFrames: one for calls and one for puts.
    """
    try:
        tk = yf.Ticker(ticker)
        
        if expiration_date:
            # Ensure expiration_date is in 'YYYY-MM-DD' format
            exp_str = expiration_date.strftime('%Y-%m-%d')
            if exp_str not in tk.options:
                st.warning(f"Expiration date {exp_str} not found for {ticker}. Available dates: {tk.options}")
                return pd.DataFrame(), pd.DataFrame()
            options_chain = tk.option_chain(exp_str)
        else:
            # Get the first available expiration date if none specified
            if not tk.options:
                st.warning(f"No options expiration dates found for {ticker}.")
                return pd.DataFrame(), pd.DataFrame()
            first_expiration = tk.options[0]
            options_chain = tk.option_chain(first_expiration)
        
        calls = options_chain.calls if hasattr(options_chain, 'calls') else pd.DataFrame()
        puts = options_chain.puts if hasattr(options_chain, 'puts') else pd.DataFrame()
        
        return calls, puts
    except Exception as e:
        st.warning(f"Could not fetch options chain for {ticker}: {e}. Options strategy will be skipped.")
        return pd.DataFrame(), pd.DataFrame()

# === Indicator Calculation Functions ===

def calculate_sma(df, window):
    return ta.trend.sma_indicator(df["Close"], window=window)

def calculate_ema(df, window):
    return ta.trend.ema_indicator(df["Close"], window=window)

def calculate_macd(df):
    macd = ta.trend.macd(df["Close"])
    macd_signal = ta.trend.macd_signal(df["Close"])
    macd_diff = ta.trend.macd_diff(df["Close"])
    return macd, macd_signal, macd_diff

def calculate_rsi(df, window):
    return ta.momentum.rsi(df["Close"], window=window)

def calculate_bollinger_bands(df):
    bb = ta.volatility.BollingerBands(df["Close"])
    return bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mband()

def calculate_stochastic_oscillator(df):
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    return stoch.stoch(), stoch.stoch_signal()

def calculate_ichimoku_cloud(df):
    ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"], window1=9, window2=26, window3=52)
    return (ichimoku.ichimoku_conversion_line(), 
            ichimoku.ichimoku_base_line(), 
            ichimoku.ichimoku_a(), 
            ichimoku.ichimoku_b())

def calculate_parabolic_sar(df):
    return ta.trend.psar_indicator(df["High"], df["Low"], df["Close"])

def calculate_adx(df):
    adx_indicator = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    return adx_indicator.adx(), adx_indicator.adx_pos(), adx_indicator.adx_neg()

def calculate_cci(df):
    return ta.trend.cci(df["High"], df["Low"], df["Close"])

def calculate_roc(df):
    return ta.momentum.roc(df["Close"])

def calculate_obv(df):
    return ta.volume.on_balance_volume(df["Close"], df["Volume"])

def calculate_volume_spike(df, window=5):
    """Detects if current volume is significantly higher than recent average."""
    if len(df) < window:
        return False
    
    # Ensure 'Volume' column exists
    if 'Volume' not in df.columns:
        return False

    avg_volume = df['Volume'].iloc[-window-1:-1].mean()
    current_volume = df['Volume'].iloc[-1]
    return current_volume > (avg_volume * 1.5) # 1.5x average volume as a spike


def calculate_pivot_points(df):
    """
    Calculates Classical Pivot Points for the last period in the DataFrame.
    Assumes df contains 'High', 'Low', 'Close' for the period.
    """
    if df.empty:
        return pd.DataFrame()

    last_period = df.iloc[-1]
    high = last_period['High']
    low = last_period['Low']
    close = last_period['Close']

    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return pd.DataFrame([{
        'Pivot': pivot,
        'R1': r1, 'R2': r2, 'R3': r3,
        'S1': s1, 'S2': s2, 'S3': s3
    }])


def calculate_indicators(df, indicator_selection, is_intraday=False):
    """
    Calculates selected technical indicators and adds them to the DataFrame.
    Returns the DataFrame with new indicator columns.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying original DataFrame
    
    # Ensure there's enough data for indicator calculation
    if len(df_copy) < 20: # Minimum data for most common indicators (e.g., 14-period RSI)
        return pd.DataFrame() # Return empty if not enough data

    if indicator_selection.get("SMA"):
        df_copy["SMA_20"] = calculate_sma(df_copy, 20)
        df_copy["SMA_50"] = calculate_sma(df_copy, 50)
    if indicator_selection.get("EMA"):
        df_copy["EMA_20"] = calculate_ema(df_copy, 20)
        df_copy["EMA_50"] = calculate_ema(df_copy, 50)
    if indicator_selection.get("MACD"):
        df_copy["MACD"], df_copy["MACD_Signal"], df_copy["MACD_Diff"] = calculate_macd(df_copy)
    if indicator_selection.get("RSI"):
        df_copy["RSI"] = calculate_rsi(df_copy, 14)
    if indicator_selection.get("Bollinger Bands"):
        df_copy["BB_High"], df_copy["BB_Low"], df_copy["BB_Mid"] = calculate_bollinger_bands(df_copy)
    if indicator_selection.get("Stochastic Oscillator"):
        df_copy["Stoch_K"], df_copy["Stoch_D"] = calculate_stochastic_oscillator(df_copy)
    if indicator_selection.get("Ichimoku Cloud"):
        df_copy["Ichimoku_Conversion"], df_copy["Ichimoku_Base"], df_copy["Ichimoku_A"], df_copy["Ichimoku_B"] = calculate_ichimoku_cloud(df_copy)
    if indicator_selection.get("Parabolic SAR"):
        df_copy["PSAR"] = calculate_parabolic_sar(df_copy)
    if indicator_selection.get("ADX"):
        df_copy["ADX"], df_copy["ADX_Pos"], df_copy["ADX_Neg"] = calculate_adx(df_copy)
    if indicator_selection.get("CCI"):
        df_copy["CCI"] = calculate_cci(df_copy)
    if indicator_selection.get("ROC"):
        df_copy["ROC"] = calculate_roc(df_copy)
    if indicator_selection.get("OBV"):
        df_copy["OBV"] = calculate_obv(df_copy)
    if indicator_selection.get("Volume Spike"):
        df_copy["Volume_Spike"] = calculate_volume_spike(df_copy) # This will be boolean or based on threshold
    
    # Calculate ATR for potential stop-loss/target calculations, always include it if possible
    df_copy['ATR'] = ta.volatility.average_true_range(df_copy['High'], df_copy['Low'], df_copy['Close'], window=14)
    
    # For Pivot Points, calculate them for the last full daily period, especially for intraday data
    if indicator_selection.get("Pivot Points"):
        if is_intraday:
            # For intraday data, calculate daily pivots from the last full day's data
            last_day_date = df_copy.index[-1].date()
            daily_df = df_copy[df_copy.index.date == last_day_date]
            if daily_df.empty and len(df_copy) > 1: # If no full day, use last available bar
                daily_df = df_copy.iloc[-1:]
            
            if not daily_df.empty:
                pivot_data = calculate_pivot_points(daily_df)
                if not pivot_data.empty:
                    # Add pivot points to the last row of the main DataFrame
                    for col in pivot_data.columns:
                        df_copy.loc[df_copy.index[-1], col] = pivot_data[col].iloc[0]
        else:
            # For daily/weekly/monthly, just calculate for the last row
            pivot_data = calculate_pivot_points(df_copy.iloc[-1:])
            if not pivot_data.empty:
                 for col in pivot_data.columns:
                    df_copy.loc[df_copy.index[-1], col] = pivot_data[col].iloc[0]

    # Drop rows with NaN values that result from indicator calculations
    df_cleaned = df_copy.dropna().copy()
    if df_cleaned.empty and len(df_copy) > 0:
        # If dropping NaNs makes it empty, but original had data,
        # it means not enough data for ANY indicator. Return original df or a slice
        return df_copy.iloc[-1:] if not df_copy.empty else pd.DataFrame()
    return df_cleaned


# === Signal Generation and Confidence Scoring ===

def generate_signals_for_row(row, indicator_selection, weights):
    """
    Generates bullish/bearish signals and their strength for a single row of data
    (i.e., the most recent data point).
    """
    bullish_signals = {}
    bearish_signals = {}
    signal_strength = {}

    current_price = row['Close']

    # MACD
    if indicator_selection.get("MACD") and 'MACD_Diff' in row and not pd.isna(row['MACD_Diff']):
        if row['MACD_Diff'] > 0:
            bullish_signals["MACD"] = True
            signal_strength["MACD"] = weights.get("MACD", 0) # Use actual weight for strength
        elif row['MACD_Diff'] < 0:
            bearish_signals["MACD"] = True
            signal_strength["MACD"] = weights.get("MACD", 0)

    # RSI
    if indicator_selection.get("RSI") and 'RSI' in row and not pd.isna(row['RSI']):
        if row['RSI'] < 30: # Oversold
            bullish_signals["RSI"] = True
            signal_strength["RSI"] = weights.get("RSI", 0)
        elif row['RSI'] > 70: # Overbought
            bearish_signals["RSI"] = True
            signal_strength["RSI"] = weights.get("RSI", 0)
    
    # SMA (Crossover)
    if indicator_selection.get("SMA") and 'SMA_20' in row and 'SMA_50' in row and \
       not pd.isna(row['SMA_20']) and not pd.isna(row['SMA_50']):
        if row['SMA_20'] > row['SMA_50']:
            bullish_signals["SMA"] = True
            signal_strength["SMA"] = weights.get("SMA", 0)
        elif row['SMA_20'] < row['SMA_50']:
            bearish_signals["SMA"] = True
            signal_strength["SMA"] = weights.get("SMA", 0)

    # EMA (Crossover)
    if indicator_selection.get("EMA") and 'EMA_20' in row and 'EMA_50' in row and \
       not pd.isna(row['EMA_20']) and not pd.isna(row['EMA_50']):
        if row['EMA_20'] > row['EMA_50']:
            bullish_signals["EMA"] = True
            signal_strength["EMA"] = weights.get("EMA", 0)
        elif row['EMA_20'] < row['EMA_50']:
            bearish_signals["EMA"] = True
            signal_strength["EMA"] = weights.get("EMA", 0)

    # Bollinger Bands
    if indicator_selection.get("Bollinger Bands") and 'BB_High' in row and 'BB_Low' in row and \
       not pd.isna(row['BB_High']) and not pd.isna(row['BB_Low']):
        if current_price < row['BB_Low']: # Price below lower band
            bullish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = weights.get("Bollinger Bands", 0)
        elif current_price > row['BB_High']: # Price above upper band
            bearish_signals["Bollinger Bands"] = True
            signal_strength["Bollinger Bands"] = weights.get("Bollinger Bands", 0)
            
    # Stochastic Oscillator
    if indicator_selection.get("Stochastic Oscillator") and 'Stoch_K' in row and 'Stoch_D' in row and \
       not pd.isna(row['Stoch_K']) and not pd.isna(row['Stoch_D']):
        if row['Stoch_K'] < 20 and row['Stoch_D'] < 20 and row['Stoch_K'] > row['Stoch_D']: # Oversold and K crosses above D
            bullish_signals["Stochastic Oscillator"] = True
            signal_strength["Stochastic Oscillator"] = weights.get("Stochastic Oscillator", 0)
        elif row['Stoch_K'] > 80 and row['Stoch_D'] > 80 and row['Stoch_K'] < row['Stoch_D']: # Overbought and K crosses below D
            bearish_signals["Stochastic Oscillator"] = True
            signal_strength["Stochastic Oscillator"] = weights.get("Stochastic Oscillator", 0)

    # Ichimoku Cloud
    if indicator_selection.get("Ichimoku Cloud") and 'Ichimoku_Conversion' in row and 'Ichimoku_Base' in row and \
       'Ichimoku_A' in row and 'Ichimoku_B' in row and \
       not pd.isna(row['Ichimoku_Conversion']) and not pd.isna(row['Ichimoku_Base']) and \
       not pd.isna(row['Ichimoku_A']) and not pd.isna(row['Ichimoku_B']):
        # Price above Cloud (A and B)
        if current_price > row['Ichimoku_A'] and current_price > row['Ichimoku_B']:
            bullish_signals["Ichimoku Cloud"] = True
            signal_strength["Ichimoku Cloud"] = weights.get("Ichimoku Cloud", 0)
        # Price below Cloud (A and B)
        elif current_price < row['Ichimoku_A'] and current_price < row['Ichimoku_B']:
            bearish_signals["Ichimoku Cloud"] = True
            signal_strength["Ichimoku Cloud"] = weights.get("Ichimoku Cloud", 0)

    # Parabolic SAR
    if indicator_selection.get("Parabolic SAR") and 'PSAR' in row and not pd.isna(row['PSAR']):
        if current_price > row['PSAR']:
            bullish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = weights.get("Parabolic SAR", 0)
        elif current_price < row['PSAR']:
            bearish_signals["Parabolic SAR"] = True
            signal_strength["Parabolic SAR"] = weights.get("Parabolic SAR", 0)

    # ADX
    if indicator_selection.get("ADX") and 'ADX' in row and 'ADX_Pos' in row and 'ADX_Neg' in row and \
       not pd.isna(row['ADX']) and not pd.isna(row['ADX_Pos']) and not pd.isna(row['ADX_Neg']):
        if row['ADX'] > 25: # Strong trend
            if row['ADX_Pos'] > row['ADX_Neg']: # Bullish trend
                bullish_signals["ADX"] = True
                signal_strength["ADX"] = weights.get("ADX", 0)
            elif row['ADX_Neg'] > row['ADX_Pos']: # Bearish trend
                bearish_signals["ADX"] = True
                signal_strength["ADX"] = weights.get("ADX", 0)

    # CCI
    if indicator_selection.get("CCI") and 'CCI' in row and not pd.isna(row['CCI']):
        if row['CCI'] > 100: # Overbought
            bullish_signals["CCI"] = True # Can signal strong uptrend or reversal
            signal_strength["CCI"] = weights.get("CCI", 0)
        elif row['CCI'] < -100: # Oversold
            bearish_signals["CCI"] = True # Can signal strong downtrend or reversal
            signal_strength["CCI"] = weights.get("CCI", 0)
            
    # ROC
    if indicator_selection.get("ROC") and 'ROC' in row and not pd.isna(row['ROC']):
        if row['ROC'] > 0:
            bullish_signals["ROC"] = True
            signal_strength["ROC"] = weights.get("ROC", 0)
        elif row['ROC'] < 0:
            bearish_signals["ROC"] = True
            signal_strength["ROC"] = weights.get("ROC", 0)

    # OBV
    if indicator_selection.get("OBV") and 'OBV' in row and not pd.isna(row['OBV']):
        # Simple OBV signal: rising OBV is bullish, falling OBV is bearish
        # This requires historical OBV to compare, so a simple check on current value might not be enough
        # For simplicity, let's assume a "trend" in OBV could be inferred if compared to a moving average of OBV,
        # but for a single row, we just check its value relative to a conceptual past.
        # A more robust OBV signal would compare current OBV to previous OBV or its SMA.
        pass # Not implementing a simple single-row OBV signal due to complexity

    # Volume Spike
    if indicator_selection.get("Volume Spike") and 'Volume_Spike' in row and not pd.isna(row['Volume_Spike']):
        if row['Volume_Spike']:
            # A volume spike is neutral on its own; needs context.
            # If price is up with spike: bullish. If price is down with spike: bearish.
            if row['Close'] > row['Open']:
                bullish_signals["Volume Spike"] = True
                signal_strength["Volume Spike"] = weights.get("Volume Spike", 0)
            elif row['Close'] < row['Open']:
                bearish_signals["Volume Spike"] = True
                signal_strength["Volume Spike"] = weights.get("Volume Spike", 0)
    
    # Pivot Points (already added to the row if selected, now generate signal based on current price)
    if indicator_selection.get("Pivot Points") and 'Pivot' in row and not pd.isna(row['Pivot']):
        if current_price > row['Pivot']:
            bullish_signals["Pivot Points"] = True
            signal_strength["Pivot Points"] = weights.get("Pivot Points", 0)
        elif current_price < row['Pivot']:
            bearish_signals["Pivot Points"] = True
            signal_strength["Pivot Points"] = weights.get("Pivot Points", 0)


    return bullish_signals, bearish_signals, signal_strength


def calculate_confidence_score(bullish_signals, bearish_signals, signal_strength, normalized_weights):
    """
    Calculates an overall confidence score and determines trade direction.
    """
    total_bullish_strength = sum(signal_strength.get(sig, 0) for sig in bullish_signals if bullish_signals[sig])
    total_bearish_strength = sum(signal_strength.get(sig, 0) for sig in bearish_signals if bearish_signals[sig])

    overall_confidence = 0
    trade_direction = "Neutral"
    reasons = []

    if total_bullish_strength > total_bearish_strength:
        trade_direction = "Bullish"
        overall_confidence = total_bullish_strength * 100
        for signal, is_active in bullish_signals.items():
            if is_active:
                reasons.append(f"• {signal} indicates bullish momentum.")
    elif total_bearish_strength > total_bullish_strength:
        trade_direction = "Bearish"
        overall_confidence = total_bearish_strength * 100
        for signal, is_active in bearish_signals.items():
            if is_active:
                reasons.append(f"• {signal} indicates bearish momentum.")
    else: # Equal or no strong signals
        trade_direction = "Neutral"
        overall_confidence = 0
        reasons.append("• No clear directional signals from selected indicators.")

    # Cap confidence at 100%
    overall_confidence = min(overall_confidence, 100)
    return overall_confidence, trade_direction, reasons


# === Sentiment and Economic Analysis ===

def calculate_sentiment_score(finviz_data, vix_df):
    """
    Calculates an overall sentiment score based on Finviz data and VIX.
    Score from 0-100 (0 very bearish, 100 very bullish).
    """
    score = 50 # Start neutral

    # Finviz Recommendation Score (Very Basic Interpretation)
    # This part is highly dependent on how finviz_data['recom_score'] is structured
    # and needs specific parsing based on Finviz's actual output.
    # For now, a placeholder logic:
    recom_score = finviz_data.get("recom_score")
    if recom_score and isinstance(recom_score, str):
        if "Buy" in recom_score or "Strong Buy" in recom_score:
            score += 15
        elif "Sell" in recom_score or "Strong Sell" in recom_score:
            score -= 15
        elif "Hold" in recom_score:
            pass # Neutral

    # VIX (Fear Index) - Lower VIX is generally bullish for stocks, higher is bearish
    if not vix_df.empty:
        latest_vix = vix_df['Close'].iloc[-1]
        # VIX typically ranges from 10-30 in normal markets, spikes higher in volatility
        if latest_vix < 15: # Low volatility, generally bullish
            score += 10
        elif latest_vix > 25: # High volatility, generally bearish
            score -= 10
        # Interpolate for values in between
        else:
            # Scale VIX (15-25 range) to a -5 to +5 score
            scaled_vix_impact = ((25 - latest_vix) / 10) * 5
            score += scaled_vix_impact
            
    # News Sentiment (if available, e.g., from a separate news scraper and VADER)
    # For now, using a placeholder, assuming finviz_data might have a news_sentiment_score
    # which would typically come from an actual NLP process.
    news_sentiment = finviz_data.get("news_sentiment_score")
    if isinstance(news_sentiment, (int, float)):
        # Assuming news_sentiment is already a score, e.g., -1 to 1 or 0 to 100
        if news_sentiment > 0.5: # Example positive threshold
            score += 5
        elif news_sentiment < -0.5: # Example negative threshold
            score -= 5

    return max(0, min(100, score)) # Ensure score is within 0-100

def calculate_economic_score(gdp_data, cpi_data, unemployment_data):
    """
    Calculates an overall economic health score (0-100).
    Higher score indicates a healthier economy.
    """
    score = 50 # Start neutral

    # GDP Growth (Positive is good)
    if not gdp_data.empty:
        latest_gdp = gdp_data.iloc[-1]
        # Assuming typical quarterly GDP growth rates
        if latest_gdp > 2.0: # Good growth
            score += 10
        elif latest_gdp < 0: # Recessionary
            score -= 10
        elif latest_gdp > 0: # Modest growth
            score += 5

    # CPI (Inflation - Moderate is good, too high or too low is bad)
    if not cpi_data.empty:
        latest_cpi = cpi_data.iloc[-1]
        # Target inflation around 2-3%
        if 1.5 <= latest_cpi <= 3.0:
            score += 10
        elif latest_cpi > 5.0 or latest_cpi < 0: # High inflation or deflation
            score -= 10
        elif latest_cpi > 3.0 or latest_cpi < 1.5:
            score -= 5

    # Unemployment Rate (Lower is good)
    if not unemployment_data.empty:
        latest_unemployment = unemployment_data.iloc[-1]
        # Natural rate of unemployment often considered around 4-5%
        if latest_unemployment < 4.0: # Very low unemployment
            score += 10
        elif latest_unemployment > 6.0: # High unemployment
            score -= 10
        elif latest_unemployment >= 4.0 and latest_unemployment <= 6.0:
            score += 5

    return max(0, min(100, score)) # Ensure score is within 0-100

# === Options Strategy Analysis ===

def analyze_options_chain(calls_df, puts_df, current_price, target_price, stop_loss_price, trade_direction):
    """
    Analyzes options chain to find suitable options based on trade direction and price targets.
    Adds a 'moneyness' column (ITM, ATM, OTM).
    """
    suitable_options = []

    # Process Call Options
    for _, option in calls_df.iterrows():
        moneyness = ""
        # Assuming calls are sorted by strike or can be sorted later
        if option['strike'] < current_price:
            moneyness = "ITM"
        elif option['strike'] > current_price:
            moneyness = "OTM"
        else:
            moneyness = "ATM" # At The Money

        # Add moneyness to the option dictionary/series
        option_data = option.to_dict()
        option_data['moneyness'] = moneyness
        option_data['optionType'] = 'call'
        suitable_options.append(option_data)

    # Process Put Options
    for _, option in puts_df.iterrows():
        moneyness = ""
        if option['strike'] > current_price:
            moneyness = "ITM"
        elif option['strike'] < current_price:
            moneyness = "OTM"
        else:
            moneyness = "ATM" # At The Money
        
        # Add moneyness to the option dictionary/series
        option_data = option.to_dict()
        option_data['moneyness'] = moneyness
        option_data['optionType'] = 'put'
        suitable_options.append(option_data)
        
    # Further filter based on trade direction and proximity to target/stop loss
    directional_options = []
    for opt in suitable_options:
        if trade_direction == "Bullish" and opt['optionType'] == 'call':
            # For bullish call, ideally look for OTM/ATM calls that become ITM if target is hit
            if opt['strike'] <= target_price: # Strike is at or below target
                directional_options.append(opt)
        elif trade_direction == "Bearish" and opt['optionType'] == 'put':
            # For bearish put, ideally look for OTM/ATM puts that become ITM if target is hit
            if opt['strike'] >= target_price: # Strike is at or above target
                directional_options.append(opt)
    
    return directional_options


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
