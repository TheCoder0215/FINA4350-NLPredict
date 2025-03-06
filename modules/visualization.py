"""
FINA4350 Stock Sentiment Analysis Visualization module
Contains functions for creating plots and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def visualize_results(data, model):
    """Visualize sentiment vs. returns and predictions."""
    print("Generating visualizations...")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Sentiment Time Series
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data['Sentiment'], 'b-', label='Sentiment')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Score Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Returns Time Series
    plt.subplot(2, 2, 2)
    plt.plot(data.index, data['Next_Day_Return'], 'g-', label='Next-Day Return')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.title('Stock Returns Over Time')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 3: Sentiment vs. Returns Scatterplot
    plt.subplot(2, 2, 3)
    plt.scatter(data['Sentiment'], data['Next_Day_Return'], alpha=0.7)
    plt.xlabel('News Sentiment Score')
    plt.ylabel('Next-Day Return (%)')
    plt.title('Sentiment vs. Next-Day Returns')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add regression line
    x = np.linspace(data['Sentiment'].min(), data['Sentiment'].max(), 100)
    y = model.predict(x.reshape(-1, 1))
    plt.plot(x, y, 'r--', label='Model Prediction')
    plt.legend()
    
    # Plot 4: Predicted vs. Actual Returns
    plt.subplot(2, 2, 4)
    predictions = model.predict(data[['Sentiment']])
    plt.scatter(predictions, data['Next_Day_Return'], alpha=0.7)
    plt.xlabel('Predicted Return (%)')
    plt.ylabel('Actual Return (%)')
    plt.title('Predicted vs. Actual Returns')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add perfect prediction line
    max_val = max(predictions.max(), data['Next_Day_Return'].max())
    min_val = min(predictions.min(), data['Next_Day_Return'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', label='Perfect Prediction')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis_results.png')  # Save the figure
    plt.show()

def visualize_enhanced_results(data, model, features, target):
    """
    Visualize the results of the enhanced model with advanced features.
    Handles NaN values properly for visualization.
    """
    
    # Create a clean copy of the data for visualization
    viz_data = data.copy()
    
    # Generate predictions for all data points
    X_all = viz_data[features]
    y_true = viz_data[target]
    
    # Get predictions, handling potential NaN values in input
    y_pred = np.full(len(X_all), np.nan)  # Initialize with NaNs
    
    # Get valid indices (rows without NaN in features)
    valid_indices = X_all.dropna().index
    
    # Only predict for valid feature rows
    if len(valid_indices) > 0:
        X_valid = X_all.loc[valid_indices]
        y_pred_valid = model.predict(X_valid)
        
        # Assign predictions to corresponding positions
        for i, idx in enumerate(valid_indices):
            y_pred[viz_data.index.get_indexer([idx])[0]] = y_pred_valid[i]
    
    # Store predictions in the dataframe
    viz_data['Predicted'] = y_pred
    
    # Create a mask for valid rows (both target and prediction available)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    
    # Create a larger figure for all visualizations
    plt.figure(figsize=(15, 15))
    
    # 1. Time Series of Enhanced Features
    plt.subplot(3, 2, 1)
    for feature in features:
        plt.plot(viz_data.index, viz_data[feature], label=feature)
    plt.title('Enhanced Features Over Time')
    plt.xlabel('Date')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Feature Importance (if available)
    plt.subplot(3, 2, 2)
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.title('Feature Importances')
    elif hasattr(model, 'coef_'):
        # For linear models
        coefs = model.coef_
        plt.barh(range(len(coefs)), coefs, align='center')
        plt.yticks(range(len(coefs)), features)
        plt.title('Feature Coefficients')
    else:
        plt.text(0.5, 0.5, 'Feature importance not available for this model type', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Feature Importance Not Available')
    plt.xlabel('Importance')
    
    # 3. Target Variable vs Prediction (only for valid data points)
    plt.subplot(3, 2, 3)
    if np.sum(valid_mask) > 0:
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        plt.scatter(y_true_valid, y_pred_valid, alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(y_true_valid.min(), y_pred_valid.min())
        max_val = max(y_true_valid.max(), y_pred_valid.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.title(f'Actual vs. Predicted {target}')
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'Not enough valid data points for comparison', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Insufficient Data for Comparison Plot')
    
    # 4. Correlation Heatmap (only for numeric columns with sufficient data)
    plt.subplot(3, 2, 4)
    
    # Get numeric columns with at least 50% non-null values
    numeric_cols = viz_data.select_dtypes(include=[np.number]).columns
    valid_numeric_cols = [col for col in numeric_cols if viz_data[col].isna().mean() < 0.5]
    
    if len(valid_numeric_cols) > 1:
        # Calculate correlation for valid columns
        corr = viz_data[valid_numeric_cols].corr(min_periods=3)  # Require at least 3 valid pairs
        
        # Plot heatmap
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
    else:
        plt.text(0.5, 0.5, 'Not enough valid numeric data for correlation matrix', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Insufficient Data for Correlation Plot')
    
    # 5. Prediction Error Distribution (only for valid data points)
    plt.subplot(3, 2, 5)
    if np.sum(valid_mask) > 0:
        prediction_errors = y_true_valid - y_pred_valid
        sns.histplot(prediction_errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'Not enough valid data points for error distribution', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Insufficient Data for Error Distribution')
    
    # 6. Prediction Time Series
    plt.subplot(3, 2, 6)
    plt.plot(viz_data.index, y_true, 'b-', label=f'Actual {target}', alpha=0.7)
    plt.plot(viz_data.index, y_pred, 'r--', label=f'Predicted {target}', alpha=0.7)
    plt.title(f'{target} - Actual vs. Predicted Over Time')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate performance metrics only if we have enough valid data points
    if np.sum(valid_mask) >= 3:
        try:
            mse = mean_squared_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_valid, y_pred_valid)
            
            plt.figtext(0.5, 0.01, 
                        f'Model Performance Metrics (on {np.sum(valid_mask)} valid samples):\n'
                        f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}',
                        ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        except Exception as e:
            plt.figtext(0.5, 0.01, 
                        f'Could not calculate metrics due to error: {str(e)}',
                        ha='center', fontsize=12, bbox={"facecolor":"red", "alpha":0.2, "pad":5})
    else:
        plt.figtext(0.5, 0.01, 
                    'Insufficient valid data points for reliable metrics calculation',
                    ha='center', fontsize=12, bbox={"facecolor":"red", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Enhanced Model Analysis - {target} Prediction', fontsize=16)
    
    try:
        plt.savefig('enhanced_model_results.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Could not save visualization image: {e}")
    
    plt.show()
    
    # Create a simplified trading strategy visualization if we have enough data
    if np.sum(valid_mask) >= 5:
        try:
            plt.figure(figsize=(12, 8))
            
            # Create a simple trading strategy based on predictions
            viz_data['Strategy_Signal'] = np.where(y_pred > 0, 1, -1)
            
            # Extract the base return column (without 'Next_' prefix)
            base_return_col = target.replace('Next_', '')
            if base_return_col not in viz_data.columns and 'Return' in viz_data.columns:
                base_return_col = 'Return'  # Fallback to generic Return column
                
            if base_return_col in viz_data.columns:
                # Only calculate returns where we have valid data
                valid_return_mask = ~np.isnan(viz_data[base_return_col]) & ~np.isnan(viz_data['Strategy_Signal'])
                
                # Skip if not enough valid points
                if np.sum(valid_return_mask) >= 5:
                    # Create a copy for valid data points only
                    strategy_data = viz_data.loc[valid_return_mask].copy()
                    
                    strategy_data['Strategy_Return'] = strategy_data['Strategy_Signal'] * strategy_data[base_return_col]
                    
                    # Calculate cumulative returns
                    strategy_data['Cumulative_Actual'] = (1 + strategy_data[base_return_col] / 100).cumprod()
                    strategy_data['Cumulative_Strategy'] = (1 + strategy_data['Strategy_Return'] / 100).cumprod()
                    
                    plt.plot(strategy_data.index, strategy_data['Cumulative_Actual'], 'b-', label='Buy & Hold')
                    plt.plot(strategy_data.index, strategy_data['Cumulative_Strategy'], 'g-', label='Sentiment Strategy')
                    plt.title('Cumulative Returns: Strategy vs. Buy & Hold')
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative Return (1 = starting value)')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # Print trading strategy performance statistics
                    strategy_return = strategy_data['Cumulative_Strategy'].iloc[-1] - 1
                    buy_hold_return = strategy_data['Cumulative_Actual'].iloc[-1] - 1
                    
                    plt.figtext(0.5, 0.01, 
                                f"Strategy Return: {strategy_return*100:.2f}%\n"
                                f"Buy & Hold Return: {buy_hold_return*100:.2f}%\n"
                                f"Outperformance: {(strategy_return-buy_hold_return)*100:.2f}%",
                                ha='center', fontsize=12, bbox={"facecolor":"lightgreen", "alpha":0.2, "pad":5})
                    
                    try:
                        plt.savefig('trading_strategy_results.png', dpi=300, bbox_inches='tight')
                    except Exception as e:
                        print(f"Warning: Could not save strategy image: {e}")
                        
                    plt.show()
                else:
                    print("Not enough valid return data points for trading strategy visualization")
            else:
                print(f"Could not find base return column '{base_return_col}' for strategy analysis")
        except Exception as e:
            print(f"Error creating trading strategy visualization: {e}")
    else:
        print("Not enough valid data points for trading strategy visualization")

def visualize_model_comparison(results, model_metrics, best_model_info):
    """Create visualizations comparing models across different stocks."""
    plt.figure(figsize=(15, 10))
    
    # 1. R² Scores Comparison
    plt.subplot(2, 2, 1)
    
    # Get tickers with valid metrics
    valid_tickers = [ticker for ticker in results if ticker in model_metrics and 'error' not in model_metrics[ticker]]
    
    if valid_tickers:
        r2_scores = [model_metrics[ticker]['r2'] for ticker in valid_tickers]
        
        bars = plt.bar(valid_tickers, r2_scores, color='skyblue')
        
        # Highlight the best model
        if best_model_info and 'ticker' in best_model_info:
            best_ticker = best_model_info['ticker']
            if best_ticker in valid_tickers:
                best_index = valid_tickers.index(best_ticker)
                bars[best_index].set_color('gold')
        
        plt.title('Model Performance Comparison (R² Score)')
        plt.ylabel('R² Score')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No valid models to compare', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Model Performance Comparison')
    
    # 2. Feature Importance for Best Model
    plt.subplot(2, 2, 2)
    
    if best_model_info and 'model' in best_model_info:
        best_model = best_model_info['model']
        best_features = best_model_info['features']
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [best_features[i] for i in indices])
            plt.title(f'Feature Importances for Best Model ({best_model_info["ticker"]})')
        elif hasattr(best_model, 'coef_'):
            coefs = best_model.coef_
            if not isinstance(coefs, np.ndarray):
                coefs = np.array(coefs)
            if coefs.ndim > 1:
                coefs = coefs[0]  # Take first row if multiple outputs
            plt.barh(range(len(coefs)), coefs, align='center')
            plt.yticks(range(len(coefs)), best_features)
            plt.title(f'Feature Coefficients for Best Model ({best_model_info["ticker"]})')
        else:
            plt.text(0.5, 0.5, 'Feature importance not available for this model type', 
                     horizontalalignment='center', verticalalignment='center')
            plt.title('Feature Importance Not Available')
    else:
        plt.text(0.5, 0.5, 'No best model identified', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Feature Importance')
    
    # 3. Model Types Comparison
    plt.subplot(2, 2, 3)
    
    # Get model types for all valid models
    valid_models = {k: v for k, v in model_metrics.items() if 'error' not in v}
    
    if valid_models:
        model_types = [valid_models[ticker]['model_type'] for ticker in valid_models]
        
        # Count occurrences of each model type
        model_counts = {}
        for model_type in model_types:
            if model_type in model_counts:
                model_counts[model_type] += 1
            else:
                model_counts[model_type] = 1
        
        plt.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.1f%%', 
                startangle=90, shadow=True)
        plt.title('Model Types Distribution')
    else:
        plt.text(0.5, 0.5, 'No valid models to compare', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Model Types Distribution')
    
    # 4. Feature Usage Comparison
    plt.subplot(2, 2, 4)
    
    # Get all unique features across valid models
    all_features = set()
    for ticker in results:
        if ticker in model_metrics and 'error' not in model_metrics[ticker]:
            all_features.update(results[ticker]['features'])
    
    if all_features:
        # Count feature usage across models
        feature_usage = {feature: 0 for feature in all_features}
        for ticker in results:
            if ticker in model_metrics and 'error' not in model_metrics[ticker]:
                for feature in results[ticker]['features']:
                    feature_usage[feature] += 1
        
        # Sort by usage
        feature_usage = dict(sorted(feature_usage.items(), key=lambda x: x[1], reverse=True))
        
        plt.barh(list(feature_usage.keys()), list(feature_usage.values()), color='lightgreen')
        plt.title('Feature Usage Across Models')
        plt.xlabel('Number of Models Using Feature')
    else:
        plt.text(0.5, 0.5, 'No valid features to compare', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Feature Usage Comparison')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()