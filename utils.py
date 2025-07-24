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
    df_cleaned.loc[:, "Volume_MA"] = df_cleaned["Volume"].rolling(window=20).mean() # Example: 20-period rolling mean of Volume
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
    profit_target_atr_multiplier_2 = 3.5 # For a second target, if desired

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
    # Now, confidence_score_value is directly the score, not a dictionary.
    # We can use it directly.
    score = confidence_score_value 

    # Initialize a default plan for neutral or unhandled cases
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

    # Logic for suggesting strategies based on trade_direction and score
    if trade_direction == "Bullish" and score >= 60: # Example threshold for bullish strategy
        # Implement logic for Bull Call Spread or other bullish strategies
        
        # Select a near-term expiration (e.g., the first available)
        if not expirations:
            plan["message"] = "No expiration dates available for options strategy."
            return plan
        
        selected_expiration = expirations[0] # Simplistic selection (you might want more logic here)

        # Simulate options data for a bull call spread
        # In a real application, you'd fetch actual options data here (e.g., using yfinance)
        
        # Assume we find options around the current price
        # Buy an ITM call, Sell an OTM call
        buy_strike = current_stock_price * 0.98 # Slightly ITM
        sell_strike = current_stock_price * 1.02 # Slightly OTM

        # Ensure strikes are reasonable (buy_strike < sell_strike for bull call spread)
        if buy_strike >= sell_strike:
            buy_strike = current_stock_price * 0.95
            sell_strike = current_stock_price * 1.05

        # Simulate premiums (buy ITM call is more expensive than sell OTM call)
        buy_premium = (current_stock_price - buy_strike) * 0.5 + 1.0 # Example premium
        sell_premium = (current_stock_price - sell_strike) * 0.1 + 0.5 # Example premium (lower)
        
        if buy_premium <= 0: buy_premium = 1.0 # Ensure positive
        if sell_premium <= 0: sell_premium = 0.5 # Ensure positive

        net_debit = (buy_premium - sell_premium) * 100 # Multiplied by 100 for contracts

        # Define contract details for the suggested strategy
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

        # Calculate Max Profit and Max Risk for Bull Call Spread
        max_profit_per_share = (sell_strike - buy_strike) - (buy_premium - sell_premium)
        max_risk_per_share = (buy_premium - sell_premium)
        
        max_profit = max_profit_per_share * 100
        max_risk = max_risk_per_share * 100

        # Ensure max_profit and max_risk are positive for display
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

    elif trade_direction == "Bearish" and score <= 40: # Example threshold for bearish strategy
        # Implement logic for Bear Put Spread or other bearish strategies
        
        if not expirations:
            plan["message"] = "No expiration dates available for options strategy."
            return plan
        
        selected_expiration = expirations[0] # Simplistic selection

        # Buy an OTM put, Sell an ITM put
        buy_strike = current_stock_price * 1.02 # Higher strike (OTM)
        sell_strike = current_stock_price * 0.98 # Lower strike (ITM)

        if buy_strike <= sell_strike:
            buy_strike = current_stock_price * 1.05
            sell_strike = current_stock_price * 0.95

        buy_premium = (buy_strike - current_stock_price) * 0.5 + 1.0 # Example premium
        sell_premium = (sell_strike - current_stock_price) * 0.1 + 0.5 # Example premium (lower)

        if buy_premium <= 0: buy_premium = 1.0 # Ensure positive
        if sell_premium <= 0: sell_premium = 0.5 # Ensure positive

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
