# utils.py - Version 3.0
print("--- utils.py VERSION CHECK: Loading Version 3.0 with all functions ---")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np # Imported for calculations
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

# === Data Fetching Functions (Largely Unchanged) ===
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
            analyst_recom_str = recom_tag.find_next_sibling('td').text if recom_tag else "N/A"
            headlines = [tag.text for tag in soup.findAll('a', class_='news-link-left')[:10]]
            analyzer = SentimentIntensityAnalyzer()
            compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            avg_compound = sum(compound_scores) / len(compound_scores) if compound_scores else 0
            return {"recom_str": analyst_recom_str, "headlines": headlines, "sentiment_compound": avg_compound}
    except Exception as e:
        st.error(f"Error fetching Finviz data: {e}", icon="🚨")
        return {"recom_str": "N/A", "headlines": [], "sentiment_compound": 0, "error": str(e)}

@st.cache_data(ttl=60)
def get_data(symbol, period, interval):
    """Fetches historical stock data and basic info from Yahoo Finance."""
    with st.spinner(f"Fetching {period} of {interval} data for {symbol}..."):
        stock = yf.Ticker(symbol)
        try:
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
        try:
            options = stock_obj.option_chain(expiry_date)
            return options.calls, options.puts
        except Exception as e:
            st.warning(f"Could not fetch options chain for {ticker} on {expiry_date}: {e}", icon="⚠️")
            return pd.DataFrame(), pd.DataFrame()

# === Indicator Calculation (Largely Unchanged) ===
def calculate_indicators(df, is_intraday=False):
    """Calculates various technical indicators for a given DataFrame."""
    initial_len = len(df)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return df
    df_cleaned = df.dropna(subset=required_cols).copy()
    if df_cleaned.empty: return df_cleaned
    
    # EMAs
    df_cleaned.loc[:, "EMA21"] = ta.trend.ema_indicator(df_cleaned["Close"], 21)
    df_cleaned.loc[:, "EMA50"] = ta.trend.ema_indicator(df_cleaned["Close"], 50)
    df_cleaned.loc[:, "EMA200"] = ta.trend.ema_indicator(df_cleaned["Close"], 200)
    
    # Ichimoku
    df_cleaned.loc[:, 'ichimoku_a'] = ta.trend.ichimoku_a(df_cleaned['High'], df_cleaned['Low'], fillna=True)
    df_cleaned.loc[:, 'ichimoku_b'] = ta.trend.ichimoku_b(df_cleaned['High'], df_cleaned['Low'], fillna=True)
    
    # Other indicators
    df_cleaned.loc[:, 'psar'] = ta.trend.PSARIndicator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']).psar()
    df_cleaned.loc[:, 'adx'] = ta.trend.ADXIndicator(df_cleaned['High'], df_cleaned['Low'], df_cleaned['Close']).adx()
    df_cleaned.loc[:, "RSI"] = ta.momentum.RSIIndicator(df_cleaned["Close"]).rsi()
    df_cleaned.loc[:, "Stochastic_K"] = ta.momentum.StochasticOscillator(df_cleaned["High"], df_cleaned["Low"], df_cleaned["Close"]).stoch()
    df_cleaned.loc[:, "Stochastic_D"] = ta.momentum.StochasticOscillator(df_cleaned["High"], df_cleaned["Low"], df_cleaned["Close"]).stoch_signal()
    df_cleaned.loc[:, "MACD"] = ta.trend.MACD(df_cleaned["Close"]).macd()
    df_cleaned.loc[:, "MACD_Signal"] = ta.trend.MACD(df_cleaned["Close"]).macd_signal()
    df_cleaned.loc[:, "Bollinger_HBand"] = ta.volatility.BollingerBands(df_cleaned["Close"]).bollinger_hband()
    df_cleaned.loc[:, "Bollinger_LBand"] = ta.volatility.BollingerBands(df_cleaned["Close"]).bollinger_lband()
    df_cleaned.loc[:, "Volume_MA"] = ta.volume.volume_trend(df_cleaned["Volume"]) # Placeholder for volume spike
    df_cleaned.loc[:, "CCI"] = ta.trend.CCIIndicator(df_cleaned["High"], df_cleaned["Low"], df_cleaned["Close"]).cci()
    df_cleaned.loc[:, "ROC"] = ta.momentum.ROCIndicator(df_cleaned["Close"]).roc()
    df_cleaned.loc[:, "OBV"] = ta.volume.OnBalanceVolumeIndicator(df_cleaned["Close"], df_cleaned["Volume"]).on_balance_volume()

    if is_intraday:
        df_cleaned.loc[:, "VWAP"] = (df_cleaned['Volume'] * (df_cleaned['High'] + df_cleaned['Low'] + df_cleaned['Close']) / 3).cumsum() / df_cleaned['Volume'].cumsum()
    else:
        df_cleaned.loc[:, "VWAP"] = np.nan # Not applicable for daily/weekly

    return df_cleaned


# === Signal Generation (Largely Unchanged) ===
def generate_signals_for_row(row_data):
    """Generates bullish and bearish signals for a single row of data."""
    bullish_signals = {}
    bearish_signals = {}
    
    close_price = row_data.get("Close")

    # EMA Trend
    ema21 = row_data.get("EMA21")
    ema50 = row_data.get("EMA50")
    ema200 = row_data.get("EMA200")
    if ema21 and ema50 and ema200:
        bullish_signals["EMA Trend"] = ema21 > ema50 > ema200
        bearish_signals["EMA Trend"] = ema21 < ema50 < ema200
        
    # Ichimoku Cloud
    ichimoku_a = row_data.get("ichimoku_a")
    ichimoku_b = row_data.get("ichimoku_b")
    if close_price and ichimoku_a and ichimoku_b:
        bullish_signals["Ichimoku Cloud"] = close_price > ichimoku_a and close_price > ichimoku_b
        bearish_signals["Ichimoku Cloud"] = close_price < ichimoku_a and close_price < ichimoku_b
    
    # Parabolic SAR
    psar = row_data.get("psar")
    if close_price and psar:
        bullish_signals["Parabolic SAR"] = close_price > psar
        bearish_signals["Parabolic SAR"] = close_price < psar
    
    # ADX
    adx = row_data.get("adx")
    if adx and row_data.get('plus_di') and row_data.get('minus_di'): # Assuming plus_di and minus_di are also available from calculate_indicators
        bullish_signals["ADX"] = adx > 25 and row_data['plus_di'] > row_data['minus_di']
        bearish_signals["ADX"] = adx > 25 and row_data['minus_di'] > row_data['plus_di']

    # RSI Momentum
    rsi = row_data.get("RSI")
    if rsi:
        bullish_signals["RSI Momentum"] = rsi < 30 # Oversold, bullish
        bearish_signals["RSI Momentum"] = rsi > 70 # Overbought, bearish

    # Stochastic
    stoch_k = row_data.get("Stochastic_K")
    stoch_d = row_data.get("Stochastic_D")
    if stoch_k and stoch_d:
        bullish_signals["Stochastic"] = stoch_k < 20 and stoch_k > stoch_d # Oversold and K-line crossing above D-line
        bearish_signals["Stochastic"] = stoch_k > 80 and stoch_k < stoch_d # Overbought and K-line crossing below D-line

    # MACD (using crosses of MACD and Signal line)
    macd = row_data.get("MACD")
    macd_signal = row_data.get("MACD_Signal")
    if macd and macd_signal and row_data.get('MACD_Hist'): # Assuming MACD_Hist is also available
        # Simple cross logic; more complex divergence could be added
        bullish_signals["MACD"] = macd > macd_signal # Bullish cross
        bearish_signals["MACD"] = macd < macd_signal # Bearish cross

    # Volume Spike (simplified: current volume much higher than recent average)
    volume = row_data.get("Volume")
    volume_ma = row_data.get("Volume_MA") # Volume_MA could be a simple moving average of volume
    if volume and volume_ma:
        bullish_signals["Volume Spike"] = volume > (volume_ma * 1.5) # 50% above average
        bearish_signals["Volume Spike"] = volume < (volume_ma * 0.5) # 50% below average (unusual low volume)

    # CCI
    cci = row_data.get("CCI")
    if cci:
        bullish_signals["CCI"] = cci < -100 # Oversold
        bearish_signals["CCI"] = cci > 100 # Overbought

    # ROC
    roc = row_data.get("ROC")
    if roc:
        bullish_signals["ROC"] = roc > 0 # Positive rate of change
        bearish_signals["ROC"] = roc < 0 # Negative rate of change

    # OBV (needs previous OBV for comparison, assuming OBV values are cumulative)
    obv = row_data.get("OBV")
    prev_obv = row_data.get("OBV_prev") # You would need to add a previous OBV calculation to df
    if obv and prev_obv:
        bullish_signals["OBV"] = obv > prev_obv # OBV rising
        bearish_signals["OBV"] = obv < prev_obv # OBV falling

    # VWAP (Intraday only - needs current price vs VWAP)
    vwap = row_data.get("VWAP")
    if vwap and close_price: # Check if VWAP is calculated (i.e., intraday)
        bullish_signals["VWAP"] = close_price > vwap
        bearish_signals["VWAP"] = close_price < vwap
    
    return bullish_signals, bearish_signals


# === NEW: Confidence Score and Strategy Engine ===

def calculate_confidence_score(signals, latest_row, finviz_data):
    """
    Calculates a weighted confidence score based on technicals, sentiment, and analyst ratings.
    """
    # 1. Technical Score
    bullish_signals, bearish_signals = signals
    active_bull = sum(1 for v in bullish_signals.values() if v)
    active_bear = sum(1 for v in bearish_signals.values() if v)
    total_signals = len(bullish_signals)
    
    tech_score = 0
    if total_signals > 0:
        tech_score = ((active_bull - active_bear) / total_signals) * 100
        
    # 2. Sentiment Score (from -100 to 100)
    sentiment_compound = finviz_data.get("sentiment_compound", 0)
    sentiment_score = sentiment_compound * 100

    # 3. Analyst Rating Score (from -100 to 100)
    def convert_finviz_recom_to_score(recom_str):
        try:
            recom_val = float(recom_str)
            # Scale from 1 (Strong Buy) to 5 (Strong Sell) to -100 to 100
            # 1 -> 100, 3 -> 0, 5 -> -100
            return -50 * (recom_val - 3)
        except (ValueError, TypeError):
            return 0 # Neutral/Hold
    
    analyst_score = convert_finviz_recom_to_score(finviz_data.get("recom_str", "N/A"))

    # 4. Market Regime Filter (using EMA 200)
    regime_multiplier = 1.0
    close = latest_row.get("Close")
    ema200 = latest_row.get("EMA200")
    if close is not None and ema200 is not None and not pd.isna(close) and not pd.isna(ema200):
        if close > ema200:
            regime_multiplier = 1.2 # Boost score in an uptrend
        else:
            regime_multiplier = 0.8 # Penalize score in a downtrend

    # 5. Weighted Final Score
    weights = {"tech": 0.5, "sentiment": 0.25, "analyst": 0.25}
    
    # Apply regime filter only to the technical component before averaging
    final_score_filtered = (weights["tech"] * tech_score * regime_multiplier +
                            weights["sentiment"] * sentiment_score +
                            weights["analyst"] * analyst_score)
    final_score_filtered = max(-100, min(100, final_score_filtered)) # Clamp between -100 and 100

    # 6. Qualitative Confidence Bands
    band = "Neutral / Unclear"
    if final_score_filtered > 70: band = "Very High Confidence (Bullish)"
    elif final_score_filtered > 40: band = "High Confidence (Bullish)"
    elif final_score_filtered > 15: band = "Moderate Confidence (Bullish)"
    elif final_score_filtered < -70: band = "Very High Confidence (Bearish)"
    elif final_score_filtered < -40: band = "High Confidence (Bearish)"
    elif final_score_filtered < -15: band = "Moderate Confidence (Bearish)"
    
    return {
        "score": round(final_score_filtered, 2),
        "band": band,
        "components": {
            "Technical Score": round(tech_score, 2),
            "News Sentiment Score": round(sentiment_score, 2),
            "Analyst Rating Score": round(analyst_score, 2),
            "Market Regime": "Uptrend" if regime_multiplier > 1 else "Downtrend" if regime_multiplier < 1 else "Neutral"
        }
    }

def generate_directional_trade_plan(confidence_score, current_price, latest_row, period_interval):
    """
    Generates a detailed directional trade plan (e.g., for stocks, not options)
    based on confidence score and current market conditions.
    """
    score = confidence_score['score']
    band = confidence_score['band']
    
    plan = {
        "Trade Type": "N/A",
        "Direction": "Neutral",
        "Entry Zone": "N/A",
        "Stop Loss": "N/A",
        "Take Profit 1": "N/A",
        "Take Profit 2": "N/A",
        "Key Rationale": f"Overall outlook: {band} ({score:.2f})."
    }

    # Use ATR from latest_row for dynamic levels, if available
    atr = latest_row.get('ATR')
    if atr is None or pd.isna(atr) or atr == 0:
        # Fallback if ATR is not directly available or valid (e.g., for very short periods or start of data)
        # Consider a percentage of current price as a rough estimate for volatile assets
        atr = current_price * 0.02 # Example: 2% of price as a default
        if "5m" in period_interval or "15m" in period_interval:
            atr = current_price * 0.005 # Smaller for intraday
        elif "1d" in period_interval:
             atr = current_price * 0.02 # Larger for daily
        
    
    # Define thresholds for different confidence levels
    HIGH_CONF_BULLISH = 40
    MOD_CONF_BULLISH = 15
    HIGH_CONF_BEARISH = -40
    MOD_CONF_BEARISH = -15

    if score > HIGH_CONF_BULLISH:
        plan["Trade Type"] = "Long Trade (Buy Stock)"
        plan["Direction"] = "Bullish"
        plan["Entry Zone"] = f"Around {current_price:.2f} or on slight pullback to {current_price - (0.5 * atr):.2f}"
        plan["Stop Loss"] = f"{current_price - (2 * atr):.2f}" # 2x ATR below entry
        plan["Take Profit 1"] = f"{current_price + (2 * atr):.2f}" # 2x ATR above entry
        plan["Take Profit 2"] = f"{current_price + (4 * atr):.2f}" # 4x ATR above entry
        plan["Key Rationale"] += " Strong bullish technicals and positive sentiment support a long position. Target higher prices."
        
    elif score > MOD_CONF_BULLISH:
        plan["Trade Type"] = "Conservative Long Trade"
        plan["Direction"] = "Bullish"
        plan["Entry Zone"] = f"On pullback to {current_price - (1 * atr):.2f}"
        plan["Stop Loss"] = f"{current_price - (2.5 * atr):.2f}"
        plan["Take Profit 1"] = f"{current_price + (1.5 * atr):.2f}"
        plan["Take Profit 2"] = f"{current_price + (3 * atr):.2f}"
        plan["Key Rationale"] += " Moderate bullish signals. Look for a retracement before entry to improve risk/reward."

    elif score < HIGH_CONF_BEARISH:
        plan["Trade Type"] = "Short Trade (Sell Stock)"
        plan["Direction"] = "Bearish"
        plan["Entry Zone"] = f"Around {current_price:.2f} or on slight bounce to {current_price + (0.5 * atr):.2f}"
        plan["Stop Loss"] = f"{current_price + (2 * atr):.2f}" # 2x ATR above entry
        plan["Take Profit 1"] = f"{current_price - (2 * atr):.2f}" # 2x ATR below entry
        plan["Take Profit 2"] = f"{current_price - (4 * atr):.2f}" # 4x ATR below entry
        plan["Key Rationale"] += " Strong bearish technicals and negative sentiment support a short position. Target lower prices."

    elif score < MOD_CONF_BEARISH:
        plan["Trade Type"] = "Conservative Short Trade"
        plan["Direction"] = "Bearish"
        plan["Entry Zone"] = f"On bounce to {current_price + (1 * atr):.2f}"
        plan["Stop Loss"] = f"{current_price + (2.5 * atr):.2f}"
        plan["Take Profit 1"] = f"{current_price - (1.5 * atr):.2f}"
        plan["Take Profit 2"] = f"{current_price - (3 * atr):.2f}"
        plan["Key Rationale"] += " Moderate bearish signals. Look for a bounce before entry to improve risk/reward."
    
    # Format prices to 2 decimal places if they are numbers
    for key in ["Entry Zone", "Stop Loss", "Take Profit 1", "Take Profit 2"]:
        try:
            # Check if the string contains a number, then format it
            if isinstance(plan[key], str) and any(char.isdigit() for char in plan[key]):
                parts = plan[key].split()
                formatted_parts = []
                for part in parts:
                    try:
                        formatted_parts.append(f"{float(part):.2f}")
                    except ValueError:
                        formatted_parts.append(part)
                plan[key] = " ".join(formatted_parts)
            elif isinstance(plan[key], (int, float)):
                plan[key] = f"{plan[key]:.2f}"
        except Exception:
            pass # Keep as is if conversion fails

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


def suggest_options_strategy(confidence_score, calls_df, puts_df, current_stock_price):
    """Suggests an options strategy based on confidence and implied volatility."""
    score = confidence_score['score']
    
    # Calculate average IV for At-The-Money (ATM) options
    atm_calls = calls_df[abs(calls_df['strike'] - current_stock_price) < (current_stock_price * 0.05)]
    atm_puts = puts_df[abs(puts_df['strike'] - current_stock_price) < (current_stock_price * 0.05)]
    avg_iv = pd.concat([atm_calls['impliedVolatility'], atm_puts['impliedVolatility']]).mean()
    
    if pd.isna(avg_iv):
        return "IV data not available to suggest a strategy.", "N/A"

    iv_is_high = avg_iv > 0.50 # Threshold for high IV, can be adjusted
    
    strategy, rationale = "N/A", "N/A"

    if score > 40: # High Bullish Confidence
        if iv_is_high:
            strategy = "Buy Calls"
            rationale = "High bullish confidence and high IV suggest a large upward move is expected. Buying calls offers leveraged upside."
        else:
            strategy = "Sell Cash-Secured Puts"
            rationale = "High bullish confidence but low IV makes buying calls expensive. Selling puts collects premium with a bullish outlook."
    elif score < -40: # High Bearish Confidence
        if iv_is_high:
            strategy = "Buy Puts"
            rationale = "High bearish confidence and high IV suggest a large downward move is expected. Buying puts offers leveraged downside."
        else:
            strategy = "Sell Covered Calls (if you own shares) or Buy Put Spreads"
            rationale = "High bearish confidence but low IV. A put spread limits cost while providing downside exposure."
    else: # Neutral / Moderate Confidence
        if iv_is_high:
            strategy = "Iron Condor or Straddle/Strangle"
            rationale = "Neutral outlook with high IV suggests profiting from a large move in either direction (Straddle) or from IV crush after an event (Iron Condor)."
        else:
            strategy = "Credit Spreads (Bull Put or Bear Call)"
            rationale = "Neutral outlook with low IV. Use credit spreads to collect premium with a slight directional bias and defined risk."
            
    return strategy, rationale


# === ENHANCED: Backtesting Logic ===

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
        st.info("Not enough data for robust backtesting after cleaning.")
        return [], {"error": "Insufficient data"}

    # Ensure ATR is calculated before starting the loop for backtesting
    if 'ATR' not in df_clean.columns:
        df_clean['ATR'] = ta.volatility.AverageTrueRange(df_clean['High'], df_clean['Low'], df_clean['Close']).average_true_range()
        df_clean.dropna(subset=['ATR'], inplace=True) # Drop rows where ATR is NaN
        if df_clean.empty:
            st.error("Not enough data after calculating ATR for backtesting.")
            return [], {"error": "Insufficient data after ATR calculation"}


    for i in range(1, len(df_clean)):
        current_day = df_clean.iloc[i]
        prev_day = df_clean.iloc[i-1]

        # --- Exit Logic ---
        if in_trade:
            exit_reason = None
            if trade_direction == "long":
                # NEW: Trailing PSAR exit
                if exit_strategy == 'trailing_psar' and 'psar' in df_clean.columns and prev_day.get('psar') is not None:
                    stop_loss = max(stop_loss, prev_day['psar'])

                if current_day['Low'] <= stop_loss:
                    exit_price, exit_reason = stop_loss, "Stop-Loss"
                elif current_day['High'] >= take_profit and exit_strategy == 'fixed_rr':
                    exit_price, exit_reason = take_profit, "Take-Profit"
            
            elif trade_direction == "short":
                 # NEW: Trailing PSAR exit for short
                if exit_strategy == 'trailing_psar' and 'psar' in df_clean.columns and prev_day.get('psar') is not None:
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
                        if signal_key == "VWAP" and prev_day.get('VWAP') is not None: # VWAP is intraday, check if data exists
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
    # Ensure we have the required columns
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        st.warning("DataFrame is missing High, Low, or Close columns for pivot point calculation.")
        return pd.DataFrame(index=df.index) # Return empty DF to prevent errors

    # Use the previous period's data to calculate the current period's pivots
    prev_high = df['High'].shift(1)
    prev_low = df['Low'].shift(1)
    prev_close = df['Close'].shift(1)

    # Calculate Pivot Points
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = (2 * pivot) - prev_low
    s1 = (2 * pivot) - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)

    # Create a new DataFrame with the pivot levels
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
