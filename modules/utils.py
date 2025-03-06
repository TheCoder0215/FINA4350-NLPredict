"""
Utility functions for stock sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def filter_features(df, min_valid_ratio=0.3):
    """Only keep features with at least min_valid_ratio non-NaN values"""
    valid_cols = []
    for col in df.columns:
        valid_ratio = df[col].notna().mean()
        if valid_ratio >= min_valid_ratio:
            valid_cols.append(col)
        else:
            print(f"Dropping column {col} with only {valid_ratio:.1%} valid values")
    
    return df[valid_cols]

def create_synthetic_data(n_samples=30, start_date=None, positive_trend=True):
    """Create synthetic data for demonstrations or fallback.
    
    Args:
        n_samples: Number of days of data to create
        start_date: Starting date (defaults to 30 days ago)
        positive_trend: Whether to use a positive drift in the prices
    
    Returns:
        DataFrame with synthetic price and sentiment data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_samples)
    
    dates = pd.date_range(start=start_date, periods=n_samples)
    synthetic_df = pd.DataFrame(index=dates)
    
    # Create sentiment with slight autocorrelation
    sentiment = np.zeros(n_samples)
    sentiment[0] = np.random.normal(0.1, 0.3)  # Start with random value
    
    for i in range(1, n_samples):
        # AR(1) process with mean reversion
        sentiment[i] = 0.8 * sentiment[i-1] + 0.2 * 0.1 + np.random.normal(0, 0.2)
    
    synthetic_df['raw_sentiment'] = sentiment
    
    # Add sentiment moving averages
    synthetic_df['sentiment_ma3'] = synthetic_df['raw_sentiment'].rolling(3).mean().fillna(method='bfill')
    synthetic_df['sentiment_ma7'] = synthetic_df['raw_sentiment'].rolling(7).mean().fillna(method='bfill')
    
    # Add sentiment derived features
    synthetic_df['sentiment_change'] = synthetic_df['raw_sentiment'].diff().fillna(0)
    synthetic_df['sentiment_volatility'] = synthetic_df['raw_sentiment'].rolling(5).std().fillna(method='bfill')
    
    # Generate price data that correlates with sentiment
    drift = 0.001 if positive_trend else -0.001
    volatility = 0.015
    
    # Create returns including sentiment impact and random noise
    returns = np.zeros(n_samples)
    for i in range(1, n_samples):
        # Returns include drift, some sentiment impact, and random noise
        sentiment_impact = 0.01 * sentiment[i-1]  # Yesterday's sentiment affects today's return
        random_component = np.random.normal(0, volatility)
        returns[i] = drift + sentiment_impact + random_component
        
    # Calculate price from returns
    price = 100.0  # Starting price
    prices = [price]
    for i in range(1, n_samples):
        price = price * (1 + returns[i])
        prices.append(price)
    
    synthetic_df['Close'] = prices
    synthetic_df['Return'] = returns[1:] * 100  # Convert to percentage
    synthetic_df['Return'] = synthetic_df['Return'].shift(-1)  # Align with correct day
    
    # Add market and excess returns
    synthetic_df['Market_Return'] = np.random.normal(drift*100, volatility*100, n_samples)
    synthetic_df['Excess_Return'] = synthetic_df['Return'] - synthetic_df['Market_Return']
    
    # Add next-day returns (targets)
    for col in ['Return', 'Market_Return', 'Excess_Return']:
        synthetic_df[f'Next_{col}'] = synthetic_df[col].shift(-1)
    
    return synthetic_df

def multi_stock_analysis(tickers, company_names, start_date=None, end_date=None):
    """
    Shortened version of main analysis function to avoid duplicate code
    
    Args:
        tickers: List of stock ticker symbols
        company_names: List of company names
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Dictionary with analysis results and best model info
    """
    from .data_collection import get_news, get_stock_data
    from .feature_engineering import enhanced_feature_engineering
    from .modeling import advanced_model_training, evaluate_models, find_best_model
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
        
    results = {}
    
    for ticker, company_name in zip(tickers, company_names):
        print(f"\n{'='*50}")
        print(f"Analyzing {company_name} ({ticker})...")
        print(f"{'='*50}")
        
        try:
            # Get news and stock data
            news_df = get_news(ticker, company_name)
            stock_data = get_stock_data(ticker, start_date, end_date)
            
            # Enhanced feature engineering
            enhanced_data = enhanced_feature_engineering(news_df, stock_data)
            
            # Advanced model training
            model, features, target = advanced_model_training(enhanced_data)
            
            # Store results
            results[ticker] = {
                'company': company_name,
                'model': model,
                'features': features,
                'target': target,
                'data': enhanced_data
            }
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            print("Skipping to next stock...")
    
    if not results:
        print("No successful analyses to compare. Exiting.")
        return None
    
    # Evaluate all models
    model_metrics = evaluate_models(results)
    
    # Find the best model
    best_model_info = find_best_model(results, model_metrics)
    
    return {
        'results': results,
        'model_metrics': model_metrics,
        'best_model': best_model_info
    }