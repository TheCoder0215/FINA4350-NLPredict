"""
FINA4350 Stock Sentiment Analysis
Main entry point for the application
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import nltk
import sklearn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from modules.data_collection import get_news, get_stock_data
from modules.sentiment_analysis import analyze_sentiment
from modules.feature_engineering import enhanced_feature_engineering
from modules.modeling import advanced_model_training, evaluate_models, find_best_model
from modules.visualization import visualize_enhanced_results, visualize_model_comparison

def setup_logging():
    """Configure logging to both file and console"""
    os.makedirs('logs', exist_ok=True)
    log_filename = f"logs/stock_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def log_system_info():
    """Log system and package information"""
    logging.info("=" * 80)
    logging.info("STOCK SENTIMENT ANALYSIS STARTED")
    logging.info("=" * 80)
    logging.info(f"Python version: {sys.version}")
    logging.info(f"NumPy version: {np.__version__}")
    logging.info(f"Pandas version: {pd.__version__}")
    logging.info(f"Scikit-learn version: {sklearn.__version__}")
    logging.info(f"NLTK version: {nltk.__version__}")
    logging.info(f"YFinance version: {yf.__version__}")

def main():
    """Main function to analyze multiple tech giant stocks with improved logging."""
    log_filename = setup_logging()
    log_system_info()
    
    try:
        # Define the 5 tech giants
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        company_names = ["Apple", "Microsoft", "Google", "Amazon", "Meta"]
        
        logging.info(f"Analyzing {len(tickers)} stocks: {', '.join(tickers)}")
        
        results = {}
        model_metrics = {}
        
        # Process each stock individually with timing and error tracking
        for ticker, company_name in zip(tickers, company_names):
            logging.info("\n" + "=" * 60)
            logging.info(f"ANALYZING {company_name} ({ticker})")
            logging.info("=" * 60)
            
            start_time = time.time()
            
            try:
                # Define date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                logging.info(f"Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
                # Step 1: Get news data
                logging.info(f"Step 1: Fetching news for {ticker}...")
                news_start_time = time.time()
                news_df = get_news(ticker, company_name)
                logging.info(f"Found {len(news_df)} news items in {time.time() - news_start_time:.2f} seconds")
                
                # Step 2: Get stock price data
                logging.info(f"Step 2: Fetching stock data for {ticker}...")
                stock_start_time = time.time()
                stock_data = get_stock_data(ticker, start_date, end_date)
                logging.info(f"Retrieved {len(stock_data)} days of stock data in {time.time() - stock_start_time:.2f} seconds")

                if news_df.empty:
                    logging.warning(f"No news data found for {ticker}. Creating synthetic news data.")
                    # Generate synthetic news
                    news_df = pd.DataFrame([
                        {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                        'title': f"News about {company_name} - Day {i}"} 
                        for i in range(30)
                    ])
                    
                # Step 3: Feature engineering
                logging.info(f"Step 3: Performing feature engineering...")
                fe_start_time = time.time()
                enhanced_data = enhanced_feature_engineering(news_df, stock_data)
                logging.info(f"Generated {len(enhanced_data)} data points with {len(enhanced_data.columns)} features in {time.time() - fe_start_time:.2f} seconds")
                
                # Log feature statistics
                for col in enhanced_data.columns:
                    non_nan_count = enhanced_data[col].notna().sum()
                    logging.info(f"  - Feature '{col}': {non_nan_count}/{len(enhanced_data)} valid values ({non_nan_count/len(enhanced_data)*100:.1f}%)")
                
                # Step 4: Model training
                logging.info(f"Step 4: Training models...")
                model_start_time = time.time()
                model, features, target = advanced_model_training(enhanced_data)
                logging.info(f"Model training completed in {time.time() - model_start_time:.2f} seconds")
                logging.info(f"Selected model: {type(model).__name__} with {len(features)} features targeting {target}")
                
                # Store results
                results[ticker] = {
                    'company': company_name,
                    'model': model,
                    'features': features,
                    'target': target,
                    'data': enhanced_data
                }
                
                # Calculate basic metrics
                from sklearn.metrics import mean_squared_error, r2_score
                
                X = enhanced_data[features].dropna()
                valid_indices = X.index.intersection(enhanced_data[enhanced_data[target].notna()].index)
                
                if len(valid_indices) >= 5:
                    X_valid = X.loc[valid_indices]
                    y_valid = enhanced_data.loc[valid_indices, target]
                    
                    try:
                        predictions = model.predict(X_valid)
                        mse = mean_squared_error(y_valid, predictions)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_valid, predictions)
                        
                        logging.info(f"Model performance metrics:")
                        logging.info(f"  - MSE: {mse:.4f}")
                        logging.info(f"  - RMSE: {rmse:.4f}")
                        logging.info(f"  - R²: {r2:.4f}")
                        logging.info(f"  - Data points: {len(X_valid)}")
                        
                        # Save prediction visualization
                        plt.figure(figsize=(10, 6))
                        plt.scatter(y_valid, predictions, alpha=0.7)
                        plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
                        plt.xlabel(f'Actual {target}')
                        plt.ylabel(f'Predicted {target}')
                        plt.title(f'{ticker} - Actual vs Predicted {target}')
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.savefig(f'logs/{ticker}_predictions.png')
                        plt.close()
                        
                        model_metrics[ticker] = {
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2,
                            'model_type': type(model).__name__,
                            'features': features,
                            'target': target,
                            'data_points': len(X_valid)
                        }
                    except Exception as metric_err:
                        logging.error(f"Error calculating metrics: {metric_err}")
                        
                # Generate detailed visualizations
                visualize_enhanced_results(enhanced_data, model, features, target)
            
            except Exception as e:
                logging.error(f"Error analyzing {ticker}: {str(e)}")
                logging.exception("Stack trace:")
            
            finally:
                total_time = time.time() - start_time
                logging.info(f"Analysis of {ticker} completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Perform cross-stock evaluation
        if results:
            logging.info("\n" + "=" * 60)
            logging.info("EVALUATING ALL MODELS")
            logging.info("=" * 60)
            
            if not model_metrics:  # If metrics weren't calculated during individual analysis
                model_metrics = evaluate_models(results)
            
            # Find the best model
            best_model_info = find_best_model(results, model_metrics)
            
            if best_model_info:
                logging.info("\nANALYSIS COMPLETE!")
                logging.info(f"Best model found for {best_model_info['company']} ({best_model_info['ticker']})")
                logging.info(f"Model type: {best_model_info['model_type']}")
                logging.info(f"Features: {best_model_info['features']}")
                logging.info(f"Target: {best_model_info['target']}")
                logging.info(f"R² Score: {best_model_info['metrics']['r2']:.4f}")
                
                # Create summary visualization
                visualize_model_comparison(results, model_metrics, best_model_info)
                
                # Print to console for user convenience (log file may be long)
                print("\nAnalysis complete!")
                print(f"Best model found for {best_model_info['company']} ({best_model_info['ticker']})")
                print(f"Model type: {best_model_info['model_type']}")
                print(f"Features: {best_model_info['features']}")
                print(f"Target: {best_model_info['target']}")
                print(f"R² Score: {best_model_info['metrics']['r2']:.4f}")
            else:
                logging.warning("No best model could be determined.")
                print("\nAnalysis complete, but no best model could be determined.")
        else:
            logging.error("No successful analyses to evaluate.")
            print("\nNo successful analyses to evaluate.")
            
    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}")
        logging.exception("Stack trace:")
        print(f"Critical error occurred: {str(e)}")
        
    finally:
        logging.info("=" * 80)
        logging.info("STOCK SENTIMENT ANALYSIS COMPLETED")
        logging.info("=" * 80)
        print(f"Complete log available at: {log_filename}")

# Execute the main function
if __name__ == "__main__":
    main()