"""
FINA4350 Stock Sentiment Analysis Feature engineering module
Enhanced with advanced financial indicators and regime detection
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from modules.sentiment_analysis import analyze_sentiment
import config
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import scipy.signal as signal

# Set up logging
logger = logging.getLogger(__name__)

def filter_features(df, min_valid_ratio=None):
    """Only keep features with at least min_valid_ratio non-NaN values"""
    if min_valid_ratio is None:
        min_valid_ratio = config.MIN_VALID_RATIO
    valid_cols = []
    for col in df.columns:
        valid_ratio = df[col].notna().mean()
        if valid_ratio >= min_valid_ratio:
            valid_cols.append(col)
        else:
            logging.warning(f"Dropping column {col} with only {valid_ratio:.1%} valid values")
    return df[valid_cols]

def add_news_nlp_features(news_df):
    """Add enhanced NLP and event features to news_df (per headline)"""
    if 'title' not in news_df.columns:
        return news_df
        
    # Basic metrics
    news_df['headline_length'] = news_df['title'].apply(lambda x: len(str(x).split()))
    news_df['headline_chars'] = news_df['title'].apply(lambda x: len(str(x)))
    
    # Advanced event detection with confidence levels
    # Earnings related
    news_df['earnings_confidence'] = news_df['title'].str.lower().apply(
        lambda x: sum(1 for word in ['earnings', 'eps', 'profit', 'revenue', 'quarter', 'guidance'] 
                     if word in str(x).lower()) / 6
    )
    news_df['has_earnings'] = (news_df['earnings_confidence'] > 0.1).astype(int)
    
    # M&A related
    news_df['mna_confidence'] = news_df['title'].str.lower().apply(
        lambda x: sum(1 for word in ['acquire', 'merger', 'buyout', 'deal', 'acquisition', 'takeover', 'bid'] 
                     if word in str(x).lower()) / 7
    )
    news_df['has_mna'] = (news_df['mna_confidence'] > 0.1).astype(int)
    
    # Analyst actions
    news_df['analyst_confidence'] = news_df['title'].str.lower().apply(
        lambda x: sum(1 for word in ['upgrade', 'downgrade', 'outperform', 'overweight', 'underweight', 'buy', 'sell', 'target'] 
                     if word in str(x).lower()) / 8
    )
    news_df['has_analyst'] = (news_df['analyst_confidence'] > 0.1).astype(int)
    
    # Numerical detections
    news_df['has_numbers'] = news_df['title'].str.contains(r'\d+').astype(int)
    news_df['has_pct'] = news_df['title'].str.contains(r'\d+%|\d+\.\d+%').astype(int)
    news_df['has_dollar'] = news_df['title'].str.contains(r'\$\d+|\$\d+\.\d+|\$\d+ (billion|million)').astype(int)
    
    return news_df

def detect_market_regime(price_series, window=20):
    """Detect market regime based on multiple indicators
       Returns regime as: 1 (Bull), 0 (Neutral), -1 (Bear)
    """
    if len(price_series) < window*2:
        return pd.Series(0, index=price_series.index)  # Not enough data
    
    # Calculate key indicators
    returns = price_series.pct_change()
    
    # 1. Trend indicator (short vs long MA)
    ma_short = price_series.rolling(window=window).mean()
    ma_long = price_series.rolling(window=window*2).mean()
    trend = (ma_short > ma_long).astype(int) * 2 - 1  # 1 if uptrend, -1 if downtrend
    
    # 2. Volatility regime
    vol = returns.rolling(window=window).std()
    vol_mean = vol.rolling(window=window*3).mean()
    high_vol = (vol > vol_mean * 1.2).astype(int) * -1  # High volatility often indicates bearish
    
    # 3. Momentum
    momentum = returns.rolling(window=window).sum().apply(np.sign)
    
    # 4. Mean reversion pressure
    zscore = (price_series - price_series.rolling(window=window).mean()) / price_series.rolling(window=window).std()
    mean_reversion = -zscore.apply(lambda x: np.sign(x) * min(abs(x), 1))
    
    # 5. Volume-weighted indicators if volume available
    vol_weight = pd.Series(1, index=price_series.index)  # Default to 1
    
    # Combine indicators with weights
    regime = (
        0.35 * trend +         # Trend (most important)
        0.25 * momentum +      # Momentum
        0.15 * high_vol +      # Volatility
        0.15 * mean_reversion + # Mean reversion
        0.10 * vol_weight       # Volume confirmation
    )
    
    # Discretize into three regimes
    regime_discrete = pd.Series(0, index=regime.index)  # Neutral default
    regime_discrete[regime > 0.2] = 1     # Bull
    regime_discrete[regime < -0.2] = -1   # Bear
    
    return regime_discrete

def add_advanced_technical_indicators(df, close_col, high_col=None, low_col=None, volume_col=None):
    """Add advanced technical indicators using TA-Lib and custom calculations"""
    price = df[close_col]
    
    # Base indicators
    df['MA_10'] = price.rolling(window=10).mean()
    df['MA_20'] = price.rolling(window=20).mean()
    df['MA_50'] = price.rolling(window=50).mean()
    df['MA_200'] = price.rolling(window=200).mean()
    
    # Trend indicators
    df['EMA_12'] = price.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = price.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Detect golden/death crosses
    df['GoldenCross'] = ((df['MA_50'] > df['MA_200']) & 
                          (df['MA_50'].shift(1) <= df['MA_200'].shift(1))).astype(int)
    df['DeathCross'] = ((df['MA_50'] < df['MA_200']) & 
                          (df['MA_50'].shift(1) >= df['MA_200'].shift(1))).astype(int)
    
    # Moving average crossovers
    df['MA_10_20_Cross'] = np.where(
        (df['MA_10'] > df['MA_20']) & (df['MA_10'].shift(1) <= df['MA_20'].shift(1)), 1,
        np.where((df['MA_10'] < df['MA_20']) & (df['MA_10'].shift(1) >= df['MA_20'].shift(1)), -1, 0)
    )
    
    # Momentum indicators
    delta = price.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Identify RSI divergences
    # Price making higher high but RSI making lower high (bearish)
    price_higher_high = (price > price.shift(1)) & (price.shift(1) > price.shift(2))
    rsi_lower_high = (df['RSI'] < df['RSI'].shift(1)) & (df['RSI'].shift(1) > df['RSI'].shift(2))
    df['RSI_Bearish_Div'] = (price_higher_high & rsi_lower_high).astype(int)
    
    # Price making lower low but RSI making higher low (bullish)
    price_lower_low = (price < price.shift(1)) & (price.shift(1) < price.shift(2))
    rsi_higher_low = (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'].shift(1) < df['RSI'].shift(2))
    df['RSI_Bullish_Div'] = (price_lower_low & rsi_higher_low).astype(int)
    
    # Volatility indicators
    df['BB_middle'] = price.rolling(window=20).mean()
    df['BB_std'] = price.rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct'] = (price - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Detect Bollinger Band squeeze and break
    df['BB_squeeze'] = (df['BB_width'] < df['BB_width'].rolling(window=50).min() * 1.2).astype(int)
    df['BB_break_upper'] = ((price > df['BB_upper']) & (price.shift(1) <= df['BB_upper'].shift(1))).astype(int)
    df['BB_break_lower'] = ((price < df['BB_lower']) & (price.shift(1) >= df['BB_lower'].shift(1))).astype(int)
    
    # Long-term volatility
    df['Volatility_20d'] = price.pct_change().rolling(window=20).std() * 100
    df['Volatility_50d'] = price.pct_change().rolling(window=50).std() * 100
    
    # Compared to historical volatility
    hist_vol_mean = df['Volatility_20d'].rolling(window=252).mean()
    hist_vol_std = df['Volatility_20d'].rolling(window=252).std()
    df['VolatilityRegime'] = (df['Volatility_20d'] - hist_vol_mean) / hist_vol_std.replace(0, 1)
    
    # Return momentum and reversal indicators
    for period in [5, 10, 20, 50]:
        # Return over past period
        df[f'Return_{period}d'] = price.pct_change(periods=period) * 100
        
        # Z-score of returns (mean reversion indicator)
        df[f'Return_{period}d_zscore'] = (
            df[f'Return_{period}d'] - df[f'Return_{period}d'].rolling(window=252).mean()
        ) / df[f'Return_{period}d'].rolling(window=252).std()
    
    # Market regime detection
    df['MarketRegime'] = detect_market_regime(price)
    
    # Add high, low, volume based indicators if available
    if high_col and low_col:
        high = df[high_col]
        low = df[low_col]
        
        # Add Average True Range (ATR)
        tr1 = abs(high - low)
        tr2 = abs(high - price.shift())
        tr3 = abs(low - price.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(window=14).mean()
        
        # Stochastic Oscillator
        df['Stoch_K'] = 100 * ((price - low.rolling(window=14).min()) / 
                              (high.rolling(window=14).max() - low.rolling(window=14).min()))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high.rolling(window=14).max() - price) / 
                                  (high.rolling(window=14).max() - low.rolling(window=14).min()))
    
    if volume_col:
        volume = df[volume_col]
        
        # On-Balance Volume (OBV)
        obv = pd.Series(0, index=price.index)
        for i in range(1, len(price)):
            if price.iloc[i] > price.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price.iloc[i] < price.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        df['OBV'] = obv
        
        # Volume-Price Trend
        vpt = pd.Series(0, index=price.index)
        for i in range(1, len(price)):
            vpt.iloc[i] = vpt.iloc[i-1] + volume.iloc[i] * (price.iloc[i] - price.iloc[i-1]) / price.iloc[i-1]
        df['VPT'] = vpt
        
        # Abnormal volume detection
        df['Volume_MA_20'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_MA_20']
        df['High_Volume'] = (df['Volume_Ratio'] > 2.0).astype(int)
    
    # Detect price patterns
    return df

def generate_lagged_features(df, target_col, lags=[1, 2, 3, 5, 10]):
    """Generate lagged features for time series forecasting"""
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    return df

def generate_rolling_stats(df, target_col, windows=[5, 10, 20]):
    """Generate rolling statistics for a column"""
    for window in windows:
        # Mean
        df[f'{target_col}_ma{window}'] = df[target_col].rolling(window=window).mean()
        
        # Standard deviation (volatility)
        df[f'{target_col}_std{window}'] = df[target_col].rolling(window=window).std()
        
        # Min/Max
        df[f'{target_col}_min{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_max{window}'] = df[target_col].rolling(window=window).max()
        
        # Range
        df[f'{target_col}_range{window}'] = (
            df[f'{target_col}_max{window}'] - df[f'{target_col}_min{window}']
        )
        
        # Z-score (for mean reversion)
        df[f'{target_col}_zscore{window}'] = (
            df[target_col] - df[f'{target_col}_ma{window}']
        ) / df[f'{target_col}_std{window}'].replace(0, np.nan)
    
    return df

def add_market_context(df, market_indices=['SPY', '^VIX'], lookback_days=365):
    """Add broader market context and correlation to sentiment/price"""
    try:
        # Get start date
        start_date = df.index.min() - pd.Timedelta(days=lookback_days)
        end_date = df.index.max() + pd.Timedelta(days=1)
        
        market_dfs = {}
        
        # Download data for each market index
        for index in market_indices:
            market_data = yf.download(index, start=start_date, end=end_date)
            if not market_data.empty:
                market_dfs[index] = market_data
        
        # Add base market return
        if 'SPY' in market_dfs:
            spy_data = market_dfs['SPY']
            # Calculate returns
            df['Market_Return'] = spy_data['Close'].pct_change() * 100
            
            # Calculate excess returns (stock return - market return)
            if 'Return_1d' in df.columns:
                df['Excess_Return'] = df['Return_1d'] - df['Market_Return']
                
                # Next day excess return (target variable)
                df['Next_Excess_Return'] = df['Excess_Return'].shift(-1)
                df['Up_Excess_Return'] = (df['Next_Excess_Return'] > 0).astype(int)
            
            # Add market moving average trends
            df['Market_MA_10'] = spy_data['Close'].rolling(window=10).mean()
            df['Market_MA_50'] = spy_data['Close'].rolling(window=50).mean()
            df['Market_MA_200'] = spy_data['Close'].rolling(window=200).mean()
            
            # Market trend indicator
            df['Market_Trend'] = np.where(
                df['Market_MA_10'] > df['Market_MA_50'], 1,
                np.where(df['Market_MA_50'] < df['Market_MA_200'], -1, 0)
            )
            
            # Market momentum
            df['Market_Momentum'] = spy_data['Close'].pct_change(periods=10) * 100
            
            # Calculate beta (requires stock returns)
            if 'Return_1d' in df.columns:
                # Create matching index for calculating beta
                stock_returns = df['Return_1d'] / 100  # Convert to decimal
                market_returns = df['Market_Return'] / 100  # Convert to decimal
                
                # Calculate rolling 60-day beta
                cov = stock_returns.rolling(window=60).cov(market_returns)
                market_var = market_returns.rolling(window=60).var()
                df['Beta_60d'] = cov / market_var
        
        # Add volatility index (VIX) if available
        if '^VIX' in market_dfs:
            vix_data = market_dfs['^VIX']
            df['VIX'] = vix_data['Close']
            
            # VIX moving averages
            df['VIX_MA_10'] = vix_data['Close'].rolling(window=10).mean()
            
            # VIX regime (high volatility when VIX > VIX 50-day MA)
            df['VIX_Regime'] = (vix_data['Close'] > vix_data['Close'].rolling(window=50).mean()).astype(int)
            
            # VIX spikes
            df['VIX_1d_Change'] = vix_data['Close'].pct_change() * 100
            df['VIX_Spike'] = (df['VIX_1d_Change'] > 10).astype(int)
        
        # Add correlation between stock and market
        if 'Return_1d' in df.columns and 'Market_Return' in df.columns:
            # Calculate rolling 30-day correlation
            df['Market_Correlation'] = df['Return_1d'].rolling(window=30).corr(df['Market_Return'])
        
        return df
    except Exception as e:
        logging.error(f"Error adding market context: {e}")
        return df

def extract_seasonal_features(df):
    """Extract seasonal and calendar features"""
    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Month
    df['month'] = df.index.month
    
    # Quarter end
    df['quarter_end'] = ((df.index.month % 3 == 0) & (df.index.day >= 25)).astype(int)
    
    # Is around earnings season (Jan, Apr, Jul, Oct)
    df['earnings_season'] = df.index.month.isin([1, 4, 7, 10]).astype(int)
    
    return df

def generate_interaction_features(df, sentiment_cols, price_cols, market_cols):
    """Generate interaction features between sentiment, price, and market indicators"""
    for sent_col in sentiment_cols:
        if sent_col not in df.columns:
            continue
            
        # Sentiment x Market interactions
        for market_col in market_cols:
            if market_col in df.columns:
                df[f'{sent_col}_x_{market_col}'] = df[sent_col] * df[market_col]
        
        # Sentiment x Price momentum interactions
        for price_col in price_cols:
            if price_col in df.columns:
                df[f'{sent_col}_x_{price_col}'] = df[sent_col] * df[price_col]
    
    return df

def reduce_dimensions(df, feature_columns, n_components=5):
    """Reduce dimensionality of features using PCA"""
    # Get only numeric columns without NaN
    X = df[feature_columns].select_dtypes(include=[np.number])
    X = X.dropna(axis=1)
    
    if len(X.columns) <= n_components:
        return df  # Not enough features for PCA
    
    try:
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)
        
        # Add PCA components to dataframe
        for i in range(n_components):
            df[f'PCA_{i+1}'] = components[:, i]
        
        # Log explained variance
        explained_variance = pca.explained_variance_ratio_
        total_variance = np.sum(explained_variance)
        logging.info(f"PCA with {n_components} components explains {total_variance:.2%} of variance")
        
        return df
    except Exception as e:
        logging.error(f"PCA reduction failed: {e}")
        return df

def enhanced_feature_engineering(news_df, stock_df, min_valid_ratio=0.3):
    """
    Main feature engineering with improved robustness, financial domain knowledge,
    and regime-adaptive features.
    """
    logging.info("==== Enhanced Feature Engineering ====")

    # 1. Make copies to avoid modifying originals
    news_df_copy = news_df.copy()
    stock_df_copy = stock_df.copy()
    
    # 2. NLP features: Ensure we have the title column 
    if 'title' not in news_df_copy.columns and len(news_df_copy.columns) > 0:
        news_df_copy['title'] = news_df_copy[news_df_copy.columns[0]]
    
    # 3. Add advanced NLP features that capture more dimensions
    news_df_copy = add_news_nlp_features(news_df_copy)

    # 4. Get daily sentiment with multiple metrics (not just one value)
    daily_sentiment = analyze_sentiment(news_df_copy)
    
    # 5. Generate sentiment features from raw sentiment
    sentiment_features = pd.DataFrame(index=daily_sentiment.index)
    
    # Copy all sentiment metrics from daily_sentiment
    for col in daily_sentiment.columns:
        sentiment_features[col] = daily_sentiment[col]
    
    # Generate rolling stats for raw sentiment
    if 'raw_sentiment' in sentiment_features.columns:
        sentiment_features = generate_rolling_stats(
            sentiment_features, 'raw_sentiment', windows=[3, 7, 14, 30]
        )
    
    # Generate lagged features for raw sentiment
    if 'raw_sentiment' in sentiment_features.columns:
        sentiment_features = generate_lagged_features(
            sentiment_features, 'raw_sentiment', lags=[1, 2, 3, 5, 7]
        )
    
    # Add momentum and change features
    if 'raw_sentiment' in sentiment_features.columns:
        # Calculate absolute and percentage change
        sentiment_features['sentiment_change'] = sentiment_features['raw_sentiment'].diff()
        sentiment_features['sentiment_pct_change'] = sentiment_features['raw_sentiment'].pct_change().fillna(0)
        
        # Calculate momentum (sum of changes over past N days)
        sentiment_features['sentiment_momentum_3d'] = sentiment_features['sentiment_change'].rolling(window=3).sum()
        sentiment_features['sentiment_momentum_7d'] = sentiment_features['sentiment_change'].rolling(window=7).sum()
        
        # Detect sentiment regime changes (positive to negative or vice versa)
        sentiment_features['sentiment_regime_change'] = (
            (np.sign(sentiment_features['raw_sentiment']) != 
             np.sign(sentiment_features['raw_sentiment'].shift(1))) & 
            (sentiment_features['raw_sentiment'].abs() > 0.2)
        ).astype(int)
    
    # 6. Ensure stock data has price columns
    if isinstance(stock_df_copy.columns, pd.MultiIndex):
        stock_df_copy.columns = ["_".join([str(x) for x in col if x]) for col in stock_df_copy.columns]
    
    # Find essential price columns
    close_col = next((col for col in stock_df_copy.columns if 'Close' in col or 'close' in col), None)
    high_col = next((col for col in stock_df_copy.columns if 'High' in col or 'high' in col), None)
    low_col = next((col for col in stock_df_copy.columns if 'Low' in col or 'low' in col), None)
    volume_col = next((col for col in stock_df_copy.columns if 'Volume' in col or 'volume' in col), None)
    
    if not close_col:
        logging.warning("No Close column found. Adding synthetic prices.")
        stock_df_copy['Close'] = np.linspace(100, 120, len(stock_df_copy))
        close_col = 'Close'
    
    # 7. Create return features with multiple timeframes
    stock_df_copy['Return_1d'] = stock_df_copy[close_col].pct_change() * 100
    
    # Add different return horizons
    for period in [1, 2, 3, 5, 10, 20]:
        stock_df_copy[f'Return_{period}d'] = stock_df_copy[close_col].pct_change(period) * 100
    
    # Create target variables (shifted returns) with horizons
    for period in [1, 2, 3, 5, 10, 20]:
        # Next period return
        stock_df_copy[f'Next_Return_{period}d'] = stock_df_copy[f'Return_{period}d'].shift(-period)
        
        # Direction (binary classification target)
        stock_df_copy[f'Up_Next_{period}d'] = (stock_df_copy[f'Next_Return_{period}d'] > 0).astype(int)
        
        # Lagged returns
        for lag in range(1, 6):
            stock_df_copy[f'Return_{period}d_lag{lag}'] = stock_df_copy[f'Return_{period}d'].shift(lag)
    
    # 8. Add advanced technical indicators
    stock_df_copy = add_advanced_technical_indicators(
        stock_df_copy, close_col, high_col, low_col, volume_col
    )
    
    # 9. Add market data and context
    stock_df_copy = add_market_context(stock_df_copy)
    
    # 10. Add seasonal features
    stock_df_copy = extract_seasonal_features(stock_df_copy)
    
    # 11. Join sentiment with stock data
    stock_df_copy.index = pd.to_datetime(stock_df_copy.index)
    sentiment_features.index = pd.to_datetime(sentiment_features.index)
    
    # Align dates and merge
    combined = pd.merge(stock_df_copy, sentiment_features, 
                      left_index=True, right_index=True, how='left')
    
    # 12. Add interaction features
    sentiment_cols = ['raw_sentiment', 'sentiment_ma7_mean', 'sentiment_momentum_3d']
    price_cols = ['Return_1d', 'Return_5d', 'RSI']
    market_cols = ['Market_Return', 'Market_Trend']
    
    combined = generate_interaction_features(
        combined, sentiment_cols, price_cols, market_cols
    )
    
    # 13. Fill missing values with appropriate strategies
    # For sentiment features - carry forward (with decay for older values)
    sentiment_cols = [col for col in combined.columns if 'sentiment' in col]
    for col in sentiment_cols:
        if col in combined.columns:
            # First forward fill with a limit
            combined[col] = combined[col].fillna(method='ffill', limit=5)
            
            # Apply exponential decay to remaining NAs
            na_mask = combined[col].isna()
            if na_mask.any():
                # Find last valid value before NA sequence
                for i in range(len(combined)):
                    if na_mask.iloc[i]:
                        # Find most recent non-NA value
                        last_valid_idx = na_mask[:i].argmin() if any(~na_mask[:i]) else None
                        if last_valid_idx is not None:
                            last_valid = combined[col].iloc[last_valid_idx]
                            
                            # Apply exponential decay
                            days_gap = (combined.index[i] - combined.index[last_valid_idx]).days
                            decay_factor = np.exp(-0.05 * days_gap)  # 5% decay per day
                            combined[col].iloc[i] = last_valid * decay_factor
    
    # For price and technical features - use median for missing values
    numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in sentiment_cols]
    combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())
    
    # 14. PCA to reduce dimensionality of similar features
    # Group features by type for PCA
    sentiment_cols = [col for col in combined.columns if 'sentiment' in col]
    technical_cols = [col for col in combined.columns if any(x in col for x in ['MA_', 'RSI', 'MACD', 'BB_'])]
    return_cols = [col for col in combined.columns if 'Return_' in col and not 'Next_' in col]
    
    # Apply PCA to each group if they have sufficient features
    if len(sentiment_cols) > 5:
        combined = reduce_dimensions(combined, sentiment_cols, n_components=3)
    
    if len(technical_cols) > 5:
        combined = reduce_dimensions(combined, technical_cols, n_components=4)
    
    if len(return_cols) > 5:
        combined = reduce_dimensions(combined, return_cols, n_components=3)
    
    # 15. Filter to valid features
    combined = filter_features(combined, min_valid_ratio)
    
    logging.info(f"Final engineered feature set shape: {combined.shape}")
    return combined