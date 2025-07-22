# glossary_components.py - Version 1.2

import streamlit as st

# === Indicator Descriptions Glossary ===
INDICATOR_DESCRIPTIONS = {
    "Uptrend (21>50>200 EMA)": {
        "description": "Exponential Moving Averages (EMAs) smooth price data to identify trend direction. A bullish trend is indicated when shorter EMAs (e.g., 21-day) are above longer EMAs (e.g., 50-day), and both are above the longest EMA (e.g., 200-day).",
        "ideal": "21 EMA > 50 EMA > 200 EMA",
        "example": "If 21 EMA is $105, 50 EMA is $100, and 200 EMA is $95, this indicates a strong uptrend."
    },
    "Bullish PSAR": {
        "description": "Parabolic Stop and Reverse (PSAR) is a time and price based trading system used to identify potential reversals in the price movement of traded assets. Bullish when dots are below price.",
        "ideal": "Dots below price",
        "example": "When the PSAR dots appear below the price candles, it suggests an ongoing or strengthening uptrend."
    },
    "Strong Trend (ADX > 25)": {
        "description": "The Average Directional Index (ADX) quantifies the strength of a trend. Values above 25 generally indicate a strong trend (either up or down), while values below 20 suggest a weak or non-trending market.",
        "ideal": "ADX > 25",
        "example": "An ADX reading of 35 suggests a strong trend is in place, regardless of direction."
    },
    "Bullish Momentum (RSI > 50)": {
        "description": "The Relative Strength Index (RSI) is a momentum oscillator measuring the speed and change of price movements. An RSI above 50 generally suggests bullish momentum, while below 50 indicates bearish momentum. Overbought is typically above 70, oversold below 30.",
        "ideal": "RSI > 50 (for bullish momentum), ideally not overbought (>70)",
        "example": "An RSI of 60 indicates strong bullish momentum, while an RSI of 80 suggests the asset might be overbought and due for a pullback."
    },
    "Bullish Stoch Cross": {
        "description": "The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period. A bullish cross occurs when %K (fast line) crosses above %D (slow line), often below 50. Overbought is typically above 80, oversold below 20.",
        "ideal": "%K line crosses above %D (preferably below 50, but can signal continuation above 50)",
        "example": "If %K crosses above %D while both are below 50, it's a strong bullish signal from an oversold condition."
    },
    "Bullish CCI (>0)": {
        "description": "The Commodity Channel Index (CCI) measures the current price level relative to an average price level over a given period. A CCI above zero generally indicates the price is above its average, suggesting an uptrend. Readings above +100 suggest strong uptrend, below -100 strong downtrend.",
        "ideal": "CCI > 0, ideally breaking above +100 for strong bullishness",
        "example": "A CCI moving from -50 to +70 indicates a shift to bullish momentum."
    },
    "Positive ROC (>0)": {
        "description": "Rate of Change (ROC) is a momentum indicator that measures the percentage change between the current price and the price a certain number of periods ago. A positive ROC indicates upward momentum.",
        "ideal": "ROC > 0",
        "example": "An ROC of +5% means the price is 5% higher than it was 'N' periods ago, indicating positive momentum."
    },
    "Volume Spike (>1.5x Avg)": {
        "description": "A volume spike indicates an unusual increase in trading activity, which often precedes or accompanies significant price movements. A volume greater than 1.5 times the average suggests strong interest.",
        "ideal": "Volume > 1.5x 50-day Average Volume (often confirms price moves)",
        "example": "A stock breaking out of resistance on a volume spike 2x its average volume confirms strong buying interest."
    },
    "OBV Rising": {
        "description": "On-Balance Volume (OBV) relates volume to price changes. A rising OBV indicates that positive volume pressure is increasing and confirms an uptrend. Divergences between OBV and price can signal reversals.",
        "ideal": "OBV is rising (higher than its recent average), confirming price uptrend",
        "example": "If price is making new highs and OBV is also making new highs, it confirms the strength of the uptrend."
    },
    "Price > VWAP": {
        "description": "Volume Weighted Average Price (VWAP) is a trading benchmark that represents the average price a security has traded at throughout the day, based on both volume and price. Price trading above VWAP is considered bullish.",
        "ideal": "Price > VWAP",
        "example": "If a stock opens below VWAP but then crosses above it, it often signals intraday strength."
    },
    "Bollinger Bands": {
        "description": "Bollinger Bands consist of a middle band (typically a 20-period Simple Moving Average) and two outer bands (typically two standard deviations above and below the middle band). They measure volatility and identify overbought/oversold conditions relative to the average price.",
        "ideal": "Price bouncing off lower band (bullish reversal), or price walking up the upper band (strong uptrend)",
        "example": "Price touching the lower band after a downtrend can signal a potential bounce. Price staying outside the upper band indicates strong momentum."
    },
    "Pivot Points": {
        "description": "Pivot Points are technical analysis indicators used to determine potential support and resistance levels based on the previous day's (or period's) high, low, and close prices. They are often used by intraday traders.",
        "ideal": "Price holding above Pivot Point (bullish), Price holding above R1 (very bullish), Price bouncing off S1/S2 (support)",
        "example": "If a stock opens above its daily Pivot Point and holds it, traders might look for a move towards R1."
    }
}

# === Options Greeks Descriptions ===
# Re-categorized for better organization
OPTIONS_CORE_CONCEPTS = {
    "Strike": {
        "description": "The predetermined price at which the underlying asset can be bought (for a call) or sold (for a put) when the option is exercised.",
        "ideal": "Depends on strategy (e.g., lower for long calls, higher for long puts).",
        "example": "A $150 Call option gives the holder the right to buy the stock at $150."
    },
    "Moneyness (ITM, ATM, OTM)": {
        "description": "Describes an option's relationship between its strike price and the underlying asset's current price.",
        "ideal": "ITM (In-The-Money) for higher probability, ATM (At-The-Money) for balanced risk/reward, OTM (Out-of-The-Money) for higher leverage/lower cost.",
        "example": "If a stock is at ${current_price:.2f}, a ${current_price * 0.9:.2f} Call is ITM, a ${current_price:.2f} Call is ATM, and a ${current_price * 1.1:.2f} Call is OTM."
    },
    "Expiration": {
        "description": "The date on which an option contract ceases to exist. After this date, the option can no longer be exercised.",
        "ideal": "Typically 45-90 days out for swing trades, longer for position trades, shorter for scalping.",
        "example": "A 2025-07-18 expiration means the option expires on July 18, 2025."
    },
    "Entry Premium": {
        "description": "The price paid per share for the option contract. Option prices are quoted per share, but one contract typically represents 100 shares.",
        "ideal": "Lower to enter (for buyers), higher to enter (for sellers).",
        "example": "If an option's premium is $2.50, one contract costs $250 ($2.50 x 100 shares)."
    },
    "Max Loss (per share)": {
        "description": "The maximum theoretical loss for an option trade. For a long option (buying a call or put), this is typically the premium paid.",
        "ideal": "As low as possible, or defined and acceptable for the strategy.",
        "example": "If you buy a call for $3.00, your max loss is $300 per contract."
    },
    "Profit Target (per share)": {
        "description": "The target price for the option to reach for a desired profit. This is often based on a reward-to-risk ratio.",
        "ideal": "At least 1:1 or 2:1 Reward/Risk ratio.",
        "example": "If your max risk is $200, a $400 profit target represents a 2:1 Reward/Risk."
    },
    "Bid": {
        "description": "The highest price a buyer is currently willing to pay for the option. If you want to sell, this is the price you'll get.",
        "ideal": "Higher (if selling), lower (if buying and looking for entry).",
        "example": "If the bid is $1.50, you can sell your option for $1.50 per share."
    },
    "Ask": {
        "description": "The lowest price a seller is currently willing to accept for the option. If you want to buy, this is the price you'll pay.",
        "ideal": "Lower (if buying), higher (if selling and looking for entry).",
        "example": "If the ask is $1.60, you can buy the option for $1.60 per share."
    },
}

OPTIONS_GREEKS = {
    "Delta": {
        "description": "Measures how much an option's price is expected to move for every $1 change in the underlying stock's price. Also represents the approximate probability of an option expiring in-the-money.",
        "ideal": "0.50-0.80 for ITM calls (more directional), 0.20-0.40 for OTM calls (leverage).",
        "example": "A call option with a Delta of 0.60 means its price will increase by $0.60 for every $1 increase in the stock price."
    },
    "Theta": {
        "description": "Measures the rate at which an option's price decays over time (time decay). Theta is typically negative, meaning the option loses value as it approaches expiration.",
        "ideal": "As low as possible when buying options (less time decay), higher when selling options (more time decay works in your favor).",
        "example": "A Theta of -0.05 means the option loses $0.05 per share per day due to time decay."
    },
    "Gamma": {
        "description": "Measures the rate of change of an option's Delta. High Gamma means Delta will change rapidly as the stock price moves, leading to accelerated gains or losses.",
        "ideal": "Higher for directional plays (faster delta changes), lower for stability.",
        "example": "If Delta is 0.50 and Gamma is 0.05, a $1 stock move will increase Delta to 0.55."
    },
    "Vega": {
        "description": "Measures how much an option's price is expected to change for every 1% change in implied volatility. Options with higher Vega are more sensitive to changes in IV.",
        "ideal": "Lower if you expect IV to fall (for long options), higher if you expect IV to fall (for short options).",
        "example": "A Vega of 0.10 means the option price will increase by $0.10 for every 1% increase in implied volatility."
    },
    "Rho": {
        "description": "Measures how much an option's price is expected to change for every 1% change in interest rates. This is typically less significant for short-term options.",
        "ideal": "Less significant for most short-term strategies.",
        "example": "A positive Rho for calls means their value increases with rising interest rates."
    },
}

OPTIONS_LIQUIDITY_VOLATILITY = {
    "Implied Volatility (IV)": {
        "description": "The market's forecast of the likely movement in the underlying stock's price. Higher IV generally means higher option premiums.",
        "ideal": "Lower when buying options (cheaper premiums), higher when selling options (more premium collected).",
        "example": "An IV of 30% suggests the market expects a 30% price swing over the next year. A sudden jump in IV can make options more expensive."
    },
    "Open Interest": {
        "description": "The total number of outstanding option contracts that have not yet been closed or exercised. A high number indicates good liquidity and market interest.",
        "ideal": "> 100s, preferably > 500 for good liquidity.",
        "example": "An Open Interest of 5,000 means there are 5,000 contracts currently held by traders."
    },
    "Volume (Today)": {
        "description": "The number of option contracts traded for that specific strike and expiration today. High volume indicates high trading activity.",
        "ideal": "Higher (>100) for good liquidity and ease of entry/exit.",
        "example": "A Volume of 200 means 200 contracts have been traded today."
    },
}

OPTIONS_CHAIN_SUGGESTIONS = {
    "Highest Volume Options": {
        "description": "Options contracts with the highest number of trades executed today. High volume indicates strong current market interest and liquidity for that specific contract.",
        "ideal": "Look for high volume to ensure you can easily enter and exit positions without significant slippage.",
        "example": "A call option with 5,000 contracts traded today has very high volume, suggesting active trading."
    },
    "Highest Open Interest Options": {
        "description": "Options contracts with the largest number of outstanding contracts that have not yet been closed or exercised. High open interest suggests significant market positioning and potential for larger institutional involvement.",
        "ideal": "High open interest indicates good liquidity and that many traders believe this strike/expiration is important.",
        "example": "A put option with 10,000 open interest means 10,000 contracts are currently held by traders, indicating strong conviction."
    },
    "Highest Implied Volatility Options": {
        "description": "Options contracts with the highest implied volatility (IV). High IV means the market expects large price swings in the underlying stock, leading to higher option premiums. Often seen before earnings or major news.",
        "ideal": "Consider selling high IV options if you expect volatility to decrease (e.g., after an event). Consider buying low IV options if you expect volatility to increase.",
        "example": "An option with 150% IV is significantly more expensive than one with 30% IV, reflecting higher expected price movement."
    },
    "Highest Delta Calls": {
        "description": "Call options with the highest Delta value (closest to 1.0). These options behave most like the underlying stock, meaning their price changes significantly with small movements in the stock price.",
        "ideal": "For strong bullish conviction, as they offer high leverage and move almost dollar-for-dollar with the stock (if deep ITM).",
        "example": "A call with a Delta of 0.90 will gain approximately $0.90 for every $1 increase in the stock price."
    },
    "Lowest Theta Calls": {
        "description": "Call options with the lowest (least negative) Theta value. Theta represents time decay, so options with lower Theta lose less value each day as they approach expiration.",
        "ideal": "For long option positions (buying calls or puts) to minimize the impact of time decay.",
        "example": "A call with Theta of -0.01 loses $1 per contract per day, while one with -0.05 loses $5 per contract per day."
    },
    "Highest Gamma Calls": {
        "description": "Call options with the highest Gamma value. Gamma measures the rate of change of Delta. High Gamma means Delta will accelerate rapidly as the stock price moves, leading to amplified gains or losses.",
        "ideal": "For highly directional trades where you expect a strong, fast move in the underlying stock.",
        "example": "If a call has a Delta of 0.50 and Gamma of 0.10, a $1 stock move could increase its Delta to 0.60."
    },
    "Highest Vega Calls": {
        "description": "Call options with the highest Vega value. Vega measures an option's sensitivity to changes in implied volatility. High Vega means the option's price will change significantly with shifts in IV.",
        "ideal": "If you anticipate a significant increase in implied volatility, buying high Vega options can be profitable.",
        "example": "A call with Vega of 0.15 will gain $0.15 for every 1% increase in implied volatility."
    },
    "ITM Call Suggestions": {
        "description": "In-The-Money (ITM) call options are those where the strike price is below the current stock price. They have intrinsic value and generally higher Delta.",
        "ideal": "For aggressive bullish positions, as they have a higher probability of expiring profitably and behave more like the stock.",
        "example": "If NVDA is at ${current_price:.2f}, an ITM call suggestion might be the ${current_price * 0.95:.2f} strike."
    },
    "ATM Call Suggestions": {
        "description": "At-The-Money (ATM) call options are those where the strike price is very close to the current stock price. They have a balanced risk/reward profile.",
        "ideal": "For moderate bullish positions, offering a good balance between cost, leverage, and probability of success.",
        "example": "If NVDA is at ${current_price:.2f}, an ATM call suggestion might be the ${current_price:.2f} strike."
    },
    "OTM Call Suggestions": {
        "description": "Out-of-The-Money (OTM) call options are those where the strike price is above the current stock price. They have no intrinsic value and are cheaper but have lower probability of expiring profitably.",
        "ideal": "For highly speculative bullish positions, offering high leverage for a relatively low cost, but with higher risk.",
        "example": "If NVDA is at ${current_price:.2f}, an OTM call suggestion might be the ${current_price * 1.05:.2f} strike."
    },
    "Highest Delta Puts": {
        "description": "Put options with the highest (most negative) Delta value (closest to -1.0). These options behave most like shorting the underlying stock, meaning their price changes significantly with small movements in the stock price in the opposite direction.",
        "ideal": "For strong bearish conviction, as they offer high leverage and move almost dollar-for-dollar (inversely) with the stock (if deep ITM).",
        "example": "A put with a Delta of -0.80 will gain approximately $0.80 for every $1 decrease in the stock price."
    },
    "Lowest Theta Puts": {
        "description": "Put options with the lowest (least negative) Theta value. Theta represents time decay, so options with lower Theta lose less value each day as they approach expiration.",
        "ideal": "For long option positions (buying calls or puts) to minimize the impact of time decay.",
        "example": "A put with Theta of -0.01 loses $1 per contract per day, while one with -0.05 loses $5 per contract per day."
    },
    "Highest Gamma Puts": {
        "description": "Put options with the highest Gamma value. Gamma measures the rate of change of Delta. High Gamma means Delta will accelerate rapidly as the stock price moves, leading to amplified gains or losses.",
        "ideal": "For highly directional trades where you expect a strong, fast move in the underlying stock.",
        "example": "If a put has a Delta of -0.50 and Gamma of 0.10, a $1 stock move down could increase its Delta to -0.60."
    },
    "Highest Vega Puts": {
        "description": "Put options with the highest Vega value. Vega measures an option's sensitivity to changes in implied volatility. High Vega means the option's price will change significantly with shifts in IV.",
        "ideal": "If you anticipate a significant increase in implied volatility, buying high Vega options can be profitable.",
        "example": "A put with Vega of 0.15 will gain $0.15 for every 1% increase in implied volatility."
    },
    "ITM Put Suggestions": {
        "description": "In-The-Money (ITM) put options are those where the strike price is above the current stock price. They have intrinsic value and generally higher (more negative) Delta.",
        "ideal": "For aggressive bearish positions, as they have a higher probability of expiring profitably and behave more like shorting the stock.",
        "example": "If NVDA is at ${current_price:.2f}, an ITM put suggestion might be the ${current_price * 1.05:.2f} strike."
    },
    "ATM Put Suggestions": {
        "description": "At-The-Money (ATM) put options are those where the strike price is very close to the current stock price. They have a balanced risk/reward profile.",
        "ideal": "For moderate bearish positions, offering a good balance between cost, leverage, and probability of success.",
        "example": "If NVDA is at ${current_price:.2f}, an ATM put suggestion might be the ${current_price:.2f} strike."
    },
    "OTM Put Suggestions": {
        "description": "Out-of-The-Money (OTM) put options are those where the strike price is below the current stock price. They have no intrinsic value and are cheaper but have lower probability of expiring profitably.",
        "ideal": "For highly speculative bearish positions, offering high leverage for a relatively low cost, but with higher risk.",
        "example": "If NVDA is at ${current_price:.2f}, an OTM put suggestion might be the ${current_price * 0.95:.2f} strike."
    }
}


def display_glossary_tab(current_stock_price=None):
    """Displays a comprehensive glossary of technical indicators and options Greeks."""
    st.subheader("📚 Glossary: Technical Indicators & Options Concepts")

    st.markdown("This section provides detailed explanations, ideal criteria, and examples for the various technical indicators and options-related terms used in this application.")

    st.markdown("---")

    # --- Search Bar ---
    search_query = st.text_input("🔍 Search Glossary (e.g., 'EMA', 'Delta', 'Volume')", "").lower()
    st.markdown("---")

    # --- Technical Indicators Section ---
    st.markdown("### Technical Indicators")
    found_tech_indicators = False
    for key, data in INDICATOR_DESCRIPTIONS.items():
        if search_query in key.lower() or search_query in data['description'].lower():
            found_tech_indicators = True
            with st.expander(f"**{key}**"):
                st.markdown(f"**Description:** {data['description']}")
                st.markdown(f"**Ideal (Bullish):** {data['ideal']}")
                if 'example' in data:
                    st.markdown(f"**Example:** {data['example']}")
    if not found_tech_indicators and search_query:
        st.info("No technical indicators found matching your search query.")
    elif not found_tech_indicators and not search_query:
        st.info("No technical indicators defined.") # Should not happen with current data

    st.markdown("---")

    # --- Options Concepts & Greeks Section ---
    st.markdown("### Options Concepts & Greeks")
    st.info("The 'Greeks' (Delta, Theta, Gamma, Vega, Rho) are measures of an option's sensitivity to various factors. Understanding them is crucial for managing options risk.")

    found_options_concepts = False

    # Helper function to display glossary items
    def display_glossary_items(items_dict, title, current_stock_price, search_query):
        nonlocal found_options_concepts # Declare intent to modify outer scope variable
        
        filtered_items = {
            key: data for key, data in items_dict.items()
            if search_query in key.lower() or search_query in data['description'].lower()
        }
        
        if filtered_items:
            found_options_concepts = True
            st.markdown(f"#### {title}")
            for key, data in filtered_items.items():
                with st.expander(f"**{key}**"):
                    st.markdown(f"**Description:** {data['description']}")
                    st.markdown(f"**Ideal for Buyers/Sellers:** {data['ideal']}")
                    if 'example' in data:
                        # Format dynamic examples if current_stock_price is available
                        if current_stock_price is not None:
                            try:
                                formatted_example = data['example'].format(current_price=current_stock_price)
                                st.markdown(f"**Example:** {formatted_example}")
                            except KeyError: # Fallback if format string doesn't match
                                st.markdown(f"**Example:** {data['example']}")
                        else:
                            st.markdown(f"**Example:** {data['example']}")

    # Display sub-sections for Options
    display_glossary_items(OPTIONS_CORE_CONCEPTS, "Core Concepts", current_stock_price, search_query)
    display_glossary_items(OPTIONS_GREEKS, "The Greeks (Sensitivity Measures)", current_stock_price, search_query)
    display_glossary_items(OPTIONS_LIQUIDITY_VOLATILITY, "Liquidity & Volatility", current_stock_price, search_query)
    display_glossary_items(OPTIONS_CHAIN_SUGGESTIONS, "Options Chain Analysis & Suggestions", current_stock_price, search_query)

    if not found_options_concepts and search_query:
        st.info("No options concepts or Greeks found matching your search query.")
    elif not found_options_concepts and not search_query:
        st.info("No options concepts or Greeks defined.") # Should not happen with current data
