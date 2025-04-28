"""
FINA4350 Stock Sentiment Analysis Visualization module
Enhanced with interactive visualizations and advanced analytics displays
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import config
import logging
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import base64
from io import BytesIO
import matplotlib.ticker as mtick

# Configure plot style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Setup logging
logger = logging.getLogger(__name__)

def ensure_output_directory():
    """Ensure the visualizations directory exists"""
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string for HTML embedding"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def plotly_to_base64(fig):
    """Convert plotly figure to base64 encoded string for HTML embedding"""
    img_bytes = pio.to_image(fig, format='png', scale=2)
    img_str = base64.b64encode(img_bytes).decode('utf-8')
    return img_str

def visualize_data_quality(data, title="Data Quality Overview"):
    """Visualize data quality metrics like missing values and distributions"""
    # Set up the figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Missing values heatmap for top feature groups
    missing_data = data.isnull().sum() / len(data)
    missing_data = missing_data.sort_values(ascending=False)
    
    # Only show top 30 columns with highest missing rate
    top_missing = missing_data.head(30)
    
    # Plot heatmap
    sns.heatmap(pd.DataFrame(top_missing).T, 
                cmap="YlGnBu", 
                vmin=0, vmax=1,
                cbar_kws={'label': 'Fraction Missing'},
                ax=axes[0])
    axes[0].set_title("Missing Data by Feature (Top 30)")
    axes[0].set_ylabel("Features")
    axes[0].set_xlabel("Missing Fraction")
    
    # Plot 2: Distribution of selected numeric features
    # Select a few key features
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    selected_cols = []
    
    # Try to select representative features from different categories
    for prefix in ['Return', 'sentiment', 'RSI', 'MA_', 'MACD', 'Volatility']:
        matching_cols = [col for col in numeric_cols if prefix in col]
        if matching_cols:
            selected_cols.append(matching_cols[0])
    
    # Limit to 6 columns for readability
    selected_cols = selected_cols[:6]
    
    # Create histograms
    if selected_cols:
        data[selected_cols].hist(bins=20, ax=axes[1], figsize=(12, 4), layout=(2, 3))
        axes[1].set_title("Distribution of Key Features")
    else:
        axes[1].text(0.5, 0.5, "No numeric features available", 
                    horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    return fig

def visualize_time_series(dates, series_dict, title="Time Series Analysis", y_label="Value"):
    """Plot multiple time series on the same axis"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each series
    for label, series in series_dict.items():
        ax.plot(dates, series, label=label)
    
    # Format x-axis to show dates properly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add other elements
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add data points count
    ax.text(0.01, 0.01, f"N={len(dates)} observations", transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    return fig

def visualize_sentiment_analysis(sentiment_data, price_data, title="Sentiment vs. Stock Price"):
    """Visualize sentiment analysis results vs. stock price"""
    # Create aligned dataframe
    df = pd.DataFrame()
    
    # Add sentiment (align with price dates if needed)
    if isinstance(sentiment_data, pd.Series):
        df['Sentiment'] = sentiment_data
    else:
        # Extract sentiment value column
        sentiment_col = 'raw_sentiment'
        if sentiment_col in sentiment_data.columns:
            df['Sentiment'] = sentiment_data[sentiment_col]
        else:
            # Use the first column as sentiment
            df['Sentiment'] = sentiment_data.iloc[:, 0]
    
    # Add price data (align with sentiment dates if needed)
    if 'Close' in price_data.columns:
        df['Price'] = price_data['Close']
    else:
        # Find the close column
        close_cols = [col for col in price_data.columns if 'close' in col.lower()]
        if close_cols:
            df['Price'] = price_data[close_cols[0]]
        else:
            # Use first numeric column as price
            numeric_cols = price_data.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                df['Price'] = price_data[numeric_cols[0]]
    
    # Ensure data is not empty
    if df.empty or df['Sentiment'].isna().all() or df['Price'].isna().all():
        logger.warning("Insufficient data for sentiment visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data for visualization", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Create subplot figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot sentiment on top subplot
    dates = df.index
    ax1.plot(dates, df['Sentiment'], color='blue', label='Sentiment Score')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)  # Add zero line
    ax1.set_title(f"News Sentiment over Time")
    ax1.set_ylabel("Sentiment Score")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add sentiment moving average
    if len(df) >= 7:
        sentiment_ma = df['Sentiment'].rolling(window=7).mean()
        ax1.plot(dates, sentiment_ma, color='green', linestyle='-', 
                 label='7-day MA', alpha=0.7)
        ax1.legend(loc='upper left')
    
    # Plot price on bottom subplot
    color = 'green'
    ax2.plot(dates, df['Price'], color=color, label='Stock Price')
    ax2.set_title(f"Stock Price over Time")
    ax2.set_ylabel("Price")
    ax2.set_xlabel("Date")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Calculate correlation
    correlation = df['Sentiment'].corr(df['Price'])
    correlation_text = f"Correlation: {correlation:.3f}"
    
    # Add correlation annotation
    ax1.text(0.02, 0.05, correlation_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_correlation_matrix(data, title="Feature Correlation Matrix"):
    """Visualize correlation matrix with highlighted sentiment correlations"""
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr = numeric_data.corr()
    
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Draw heatmap
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={'shrink': .5},
                vmin=-1, vmax=1, annot=False)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Optional: highlight sentiment-related correlations if they exist
    sentiment_cols = [col for col in corr.columns if 'sentiment' in col.lower()]
    
    if sentiment_cols:
        # Create a second figure focusing on sentiment correlations
        plt.figure(figsize=(10, 8))
        
        # Get correlation with sentiment features
        sentiment_corr = corr[sentiment_cols].drop(sentiment_cols)
        
        # Sort by absolute correlation with first sentiment column
        main_sentiment_col = sentiment_cols[0]
        sentiment_corr = sentiment_corr.reindex(
            sentiment_corr[main_sentiment_col].abs().sort_values(ascending=False).index
        )
        
        # Take top 20 correlations for readability
        sentiment_corr = sentiment_corr.head(20)
        
        # Plot horizontal bar chart
        ax = sns.heatmap(sentiment_corr, cmap='coolwarm', center=0,
                    linewidths=.5, cbar_kws={'shrink': .5},
                    vmin=-1, vmax=1, annot=True, fmt=".2f")
        
        plt.title(f"Top 20 Feature Correlations with Sentiment")
        plt.tight_layout()
        
        return ax.figure
    
    return plt.gcf()

def visualize_model_performance(y_true, y_pred, dates, model_type='regression', title=None):
    """
    Create comprehensive model performance visualization
    
    Args:
        y_true: actual values
        y_pred: predicted values
        dates: corresponding dates
        model_type: 'regression' or 'classification'
        title: plot title
    
    Returns:
        matplotlib figure
    """
    if title is None:
        title = f"{model_type.capitalize()} Model Performance"
    
    if model_type == 'regression':
        fig = plt.figure(figsize=(12, 10))
        
        # Layout with 4 subplots
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, :])  # Time series plot (full width)
        ax2 = fig.add_subplot(gs[1, 0])  # Scatter plot (left)
        ax3 = fig.add_subplot(gs[1, 1])  # Histogram (right)
        ax4 = fig.add_subplot(gs[2, 0])  # Cumulative returns (left)
        ax5 = fig.add_subplot(gs[2, 1])  # Performance metrics (right)
        
        # 1. Time series plot of actual vs. predicted
        ax1.plot(dates, y_true, 'b-', label='Actual', alpha=0.7)
        ax1.plot(dates, y_pred, 'r--', label='Predicted')
        ax1.set_title('Actual vs. Predicted Values Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Target Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Scatter plot of actual vs. predicted
        ax2.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax2.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.8)
        
        ax2.set_title('Predicted vs. Actual')
        ax2.set_xlabel('Actual Value')
        ax2.set_ylabel('Predicted Value')
        ax2.grid(True, alpha=0.3)
        
        # Add R² value
        r2 = r2_score(y_true, y_pred)
        ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Prediction error histogram
        errors = y_true - y_pred
        sns.histplot(errors, kde=True, ax=ax3)
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.set_title('Prediction Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Add error statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax3.text(0.05, 0.95, f'Mean = {mean_error:.4f}\nStd = {std_error:.4f}', 
                transform=ax3.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Cumulative returns plot
        # Trading strategy: go long/short based on prediction sign
        strategy_returns = np.sign(y_pred) * y_true
        
        # Calculate cumulative returns
        cum_actual = (1 + y_true/100).cumprod()
        cum_strategy = (1 + strategy_returns/100).cumprod()
        
        # Plot cumulative returns
        ax4.plot(dates, cum_actual, 'b-', label='Buy & Hold')
        ax4.plot(dates, cum_strategy, 'g-', label='Strategy')
        ax4.set_title('Trading Strategy Performance')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return (1.0 = start)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Add performance metrics
        final_strategy = cum_strategy.iloc[-1] if len(cum_strategy) > 0 else 1.0
        final_bh = cum_actual.iloc[-1] if len(cum_actual) > 0 else 1.0
        strategy_return = (final_strategy - 1) * 100
        bh_return = (final_bh - 1) * 100
        outperf = strategy_return - bh_return
        
        # Add metrics text
        ax4.text(0.05, 0.05, 
                f'Strategy: {strategy_return:.1f}%\nB&H: {bh_return:.1f}%\nα: {outperf:.1f}%', 
                transform=ax4.transAxes, fontsize=10, va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Performance metrics table
        ax5.axis('off')
        
        # Calculate additional metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / (y_true + 1e-7))) * 100  # Add small number to avoid div by zero
        
        # Direction accuracy
        direction_accuracy = np.mean((y_true * y_pred) > 0)
        
        # Calculate Sharpe ratio (annualized)
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-7) * np.sqrt(252)
        
        # Create metrics text
        metrics_text = (
            "Model Performance Metrics:\n\n"
            f"MSE: {mse:.4f}\n"
            f"RMSE: {rmse:.4f}\n"
            f"MAE: {mae:.4f}\n"
            f"MAPE: {mape:.2f}%\n"
            f"R²: {r2:.4f}\n\n"
            f"Direction Accuracy: {direction_accuracy:.2%}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
            f"Strategy Return: {strategy_return:.2f}%\n"
            f"Buy & Hold Return: {bh_return:.2f}%\n"
            f"Outperformance (α): {outperf:.2f}%"
        )
        
        ax5.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
        
    else:  # Classification
        fig = plt.figure(figsize=(12, 10))
        
        # Layout with 4 subplots
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, :])  # Time series plot (full width)
        ax2 = fig.add_subplot(gs[1, 0])  # Confusion matrix (left)
        ax3 = fig.add_subplot(gs[1, 1])  # ROC curve (right)
        ax4 = fig.add_subplot(gs[2, 0])  # Performance over time (left)
        ax5 = fig.add_subplot(gs[2, 1])  # Performance metrics (right)
        
        # 1. Time series plot of actual vs. predicted classes
        ax1.plot(dates, y_true, 'b-', label='Actual')
        ax1.plot(dates, y_pred, 'r--', label='Predicted')
        ax1.set_title('Actual vs. Predicted Classes Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Class (1=Up, 0=Down)')
        ax1.set_yticks([0, 1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title('Confusion Matrix')
        
        # 3. ROC curve (only if we have probability estimates)
        # For now, just show random baseline
        ax3.plot([0, 1], [0, 1], 'r--', label='Random')
        
        # If we have probabilities (placeholder)
        # fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        # roc_auc = auc(fpr, tpr)
        # ax3.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance over time
        # Track correct/incorrect predictions over time
        correct_predictions = (y_true == y_pred).astype(int)
        
        # Calculate cumulative accuracy over time
        cumulative_correct = np.cumsum(correct_predictions)
        observation_count = np.arange(1, len(correct_predictions) + 1)
        cumulative_accuracy = cumulative_correct / observation_count
        
        ax4.plot(dates, cumulative_accuracy, 'g-')
        ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Guess')
        ax4.set_title('Cumulative Classification Accuracy')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Cap y-axis at [0, 1]
        ax4.set_ylim([0, 1])
        
        # Add final accuracy
        final_accuracy = cumulative_accuracy[-1] if len(cumulative_accuracy) > 0 else 0
        ax4.text(0.05, 0.05, f'Final Accuracy: {final_accuracy:.2%}', 
                 transform=ax4.transAxes, fontsize=10, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Classification metrics
        ax5.axis('off')
        
        # Calculate metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Class distribution
        class_balance = np.mean(y_true)
        
        # Precision, recall for class 1
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create metrics text
        metrics_text = (
            "Classification Performance Metrics:\n\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n\n"
            f"Class Balance: {class_balance:.2%} positive\n"
            f"Prediction Balance: {np.mean(y_pred):.2%} positive\n\n"
            f"True Positives: {tp}\n"
            f"False Positives: {fp}\n"
            f"False Negatives: {fn}\n"
            f"True Negatives: {cm[0, 0]}"
        )
        
        ax5.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, y=0.99)
    plt.tight_layout()
    
    return fig

def visualize_feature_importance(model, feature_names, top_n=20, title="Feature Importance"):
    """Visualize feature importance from the model"""
    # Check if model has feature importance attribute
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models like Random Forest
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        if len(model.coef_.shape) > 1:
            importance = np.abs(model.coef_[0])
        else:
            importance = np.abs(model.coef_)
    elif hasattr(model, 'steps'):
        # Pipeline with a model at the end
        final_step = model.steps[-1][1]
        if hasattr(final_step, 'feature_importances_'):
            importance = final_step.feature_importances_
        elif hasattr(final_step, 'coef_'):
            if len(final_step.coef_.shape) > 1:
                importance = np.abs(final_step.coef_[0])
            else:
                importance = np.abs(final_step.coef_)
    
    if importance is None or len(importance) != len(feature_names):
        logger.warning("Could not extract feature importance from model")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Feature importance not available for this model type", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Create DataFrame with feature names and importance values
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top N features
    if top_n < len(importance_df):
        importance_df = importance_df.head(top_n)
    
    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
    
    # Plot bars
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center')
    
    # Formatting
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    # Adjust layout to ensure feature names are visible
    plt.tight_layout()
    
    return fig

def visualize_model_comparison(results, model_metrics, best_model_info):
    """Create visualizations comparing models across different stocks."""
    output_dir = ensure_output_directory()
    
    # Filter out models with errors
    valid_metrics = {ticker: metrics for ticker, metrics in model_metrics.items() 
                    if 'error' not in metrics}
    
    if not valid_metrics:
        logger.warning("No valid models to compare")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid models to compare", 
                horizontalalignment='center', verticalalignment='center')
        return [fig]
    
    # 1. Model Performance Comparison
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Detect if we have regression or classification models
    has_regression = any('r2' in metrics for metrics in valid_metrics.values())
    has_classification = any('accuracy' in metrics for metrics in valid_metrics.values())
    
    ticker_labels = []
    performance_values = []
    
    if has_regression:
        # Use R² for regression models
        for ticker, metrics in valid_metrics.items():
            if 'r2' in metrics:
                ticker_labels.append(f"{ticker} ({results[ticker]['company']})")
                performance_values.append(metrics['r2'])
                
        metric_name = 'R² Score'
    else:
        # Use accuracy for classification models
        for ticker, metrics in valid_metrics.items():
            if 'accuracy' in metrics:
                ticker_labels.append(f"{ticker} ({results[ticker]['company']})")
                performance_values.append(metrics['accuracy'])
                
        metric_name = 'Accuracy'
    
    # Create bars
    colors = ['skyblue'] * len(ticker_labels)
    
    # Highlight the best model
    if best_model_info and 'ticker' in best_model_info:
        best_ticker = best_model_info['ticker']
        for i, ticker in enumerate([t.split(' ')[0] for t in ticker_labels]):
            if ticker == best_ticker:
                colors[i] = 'gold'
                break
    
    # Plot bars
    bars = ax1.bar(ticker_labels, performance_values, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Add a reference line for R² (not for accuracy)
    if has_regression:
        ax1.axhline(y=0, color='red', linestyle='-', alpha=0.3, label='No predictive power')
        ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.3, label='Minimal predictive power')
    else:
        # For classification, add random guess baseline
        ax1.axhline(y=0.5, color='red', linestyle='-', alpha=0.3, label='Random guess')
    
    # Format
    ax1.set_title(f'Model Performance Comparison ({metric_name})')
    ax1.set_ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 2. Feature Usage Comparison
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Get all unique features across models
    all_features = set()
    for ticker in valid_metrics:
        if 'features' in valid_metrics[ticker]:
            all_features.update(valid_metrics[ticker]['features'])
    
    if all_features:
        # Count feature usage
        feature_counts = {feature: 0 for feature in all_features}
        for ticker in valid_metrics:
            if 'features' in valid_metrics[ticker]:
                for feature in valid_metrics[ticker]['features']:
                    feature_counts[feature] += 1
        
        # Convert to DataFrame and sort
        feature_df = pd.DataFrame({
            'Feature': list(feature_counts.keys()),
            'Count': list(feature_counts.values())
        }).sort_values('Count', ascending=False)
        
        # Limit to top 20 features for readability
        if len(feature_df) > 20:
            feature_df = feature_df.head(20)
        
        # Plot horizontal bars
        bars = ax2.barh(feature_df['Feature'], feature_df['Count'], color='lightgreen')
        
        # Add counts
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(int(width)), va='center')
        
        ax2.set_title('Most Commonly Used Features Across Models')
        ax2.set_xlabel('Number of Models Using Feature')
        ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No feature data available", 
                 horizontalalignment='center', verticalalignment='center')
        ax2.set_title('Feature Usage Comparison')
    
    plt.tight_layout()
    
    # 3. Model Types Comparison
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    model_types = [metrics.get('model_type', 'Unknown') for metrics in valid_metrics.values()]
    model_type_counts = {}
    for model_type in model_types:
        if model_type in model_type_counts:
            model_type_counts[model_type] += 1
        else:
            model_type_counts[model_type] = 1
    
    if model_type_counts:
        # Create pie chart
        ax3.pie(model_type_counts.values(), labels=model_type_counts.keys(), 
                autopct='%1.1f%%', startangle=90, shadow=True)
        ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax3.set_title('Model Types Distribution')
    else:
        ax3.text(0.5, 0.5, "No model type data available", 
                 horizontalalignment='center', verticalalignment='center')
        ax3.set_title('Model Types Comparison')
    
    # 4. Best Model Details
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.axis('off')
    
    if best_model_info:
        features_text = ", ".join(best_model_info['features'][:5])
        if len(best_model_info['features']) > 5:
            features_text += f"... and {len(best_model_info['features']) - 5} more"
            
        best_model_text = (
            f"Best Model Summary\n\n"
            f"Company: {best_model_info['company']} ({best_model_info['ticker']})\n"
            f"Model Type: {best_model_info['model_type']}\n"
            f"Target Variable: {best_model_info['target']}\n\n"
            f"Key Features: {features_text}\n\n"
            "Performance Metrics:\n"
        )
        
        # Add metrics based on model type
        metrics = best_model_info['metrics']
        if 'r2' in metrics:
            best_model_text += (
                f"R² Score: {metrics['r2']:.4f}\n"
                f"RMSE: {metrics['rmse']:.4f}\n"
                f"MSE: {metrics['mse']:.4f}\n"
            )
            if 'direction_accuracy' in metrics:
                best_model_text += f"Direction Accuracy: {metrics['direction_accuracy']:.4f}\n"
            if 'sharpe_ratio' in metrics:
                best_model_text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n"
        elif 'accuracy' in metrics:
            best_model_text += (
                f"Accuracy: {metrics['accuracy']:.4f}\n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"Recall: {metrics['recall']:.4f}\n"
                f"F1 Score: {metrics['f1']:.4f}\n"
            )
        
        # Add sample size
        if 'data_points' in metrics:
            best_model_text += f"\nSample Size: {metrics['data_points']} observations"
        
        ax4.text(0.5, 0.5, best_model_text, ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, "No best model information available", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Save all figures
    fig1.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'feature_usage_comparison.png'), dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'model_types_comparison.png'), dpi=300, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'best_model_details.png'), dpi=300, bbox_inches='tight')
    
    return [fig1, fig2, fig3, fig4]

def generate_html_report(results, model_metrics, best_model_info, output_file="stock_sentiment_report.html"):
    """Generate a comprehensive HTML report with all visualizations"""
    output_dir = ensure_output_directory()
    report_path = os.path.join(output_dir, output_file)
    
    # Generate all visualizations
    model_comparison_figs = visualize_model_comparison(results, model_metrics, best_model_info)
    
    # Convert figures to base64 for embedding in HTML
    model_comparison_imgs = [fig_to_base64(fig) for fig in model_comparison_figs]
    
    # Best model visualization if available
    best_model_fig = None
    best_model_img = ""
    best_feature_img = ""
    best_data_img = ""
    
    if best_model_info and 'ticker' in best_model_info:
        ticker = best_model_info['ticker']
        
        if ticker in results:
            # Feature importance visualization
            if len(best_model_info['features']) > 0:
                try:
                    feature_fig = visualize_feature_importance(
                        best_model_info['model'], 
                        best_model_info['features'],
                        title=f"Feature Importance for {ticker}"
                    )
                    best_feature_img = fig_to_base64(feature_fig)
                except Exception as e:
                    logger.error(f"Error generating feature importance plot: {e}")
            
            # Model performance visualization
            try:
                model = results[ticker]['model']
                features = results[ticker]['features']
                target = results[ticker]['target']
                data = results[ticker]['data']
                
                # Get valid samples for visualization
                valid = data[features].notna().all(axis=1) & data[target].notna()
                X = data.loc[valid, features]
                y = data.loc[valid, target]
                dates = X.index
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Determine model type
                model_type = 'classification' if target.startswith('Up_') else 'regression'
                
                # Create visualization
                best_model_fig = visualize_model_performance(
                    y, y_pred, dates, 
                    model_type=model_type,
                    title=f"Model Performance for {ticker} ({results[ticker]['company']})"
                )
                best_model_img = fig_to_base64(best_model_fig)
            except Exception as e:
                logger.error(f"Error generating model performance plot: {e}")
            
            # Data quality visualization
            try:
                data_fig = visualize_data_quality(
                    results[ticker]['data'],
                    title=f"Data Quality for {ticker} ({results[ticker]['company']})"
                )
                best_data_img = fig_to_base64(data_fig)
            except Exception as e:
                logger.error(f"Error generating data quality plot: {e}")
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Sentiment Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .report-header {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                border-left: 5px solid #3498db;
            }}
            .section {{
                margin-bottom: 40px;
                padding: 20px;
                background-color: #fff;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .img-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .img-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
                font-size: 0.9em;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>Stock Sentiment Analysis Report</h1>
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Analysis of {len(results)} stocks with sentiment-based prediction models</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report presents the results of sentiment analysis on financial news and its relationship to stock price movements. 
            The analysis includes data collection from various financial news sources, sentiment extraction, and predictive modeling.</p>
            
            <h3>Key Findings</h3>
            <ul>
    """
    
    # Add key findings based on best model
    if best_model_info:
        metrics = best_model_info['metrics']
        if 'r2' in metrics:
            r2 = metrics['r2']
            if r2 > 0.2:
                prediction_quality = "strong"
            elif r2 > 0.1:
                prediction_quality = "moderate"
            elif r2 > 0.05:
                prediction_quality = "weak"
            else:
                prediction_quality = "very weak"
                
            html_content += f"""
                <li>The best model shows a {prediction_quality} predictive relationship between news sentiment and stock returns (R² = {r2:.3f}).</li>
                <li>News sentiment appears most effective for predicting {best_model_info['company']} ({best_model_info['ticker']}) stock movements.</li>
                <li>The model achieves {metrics.get('direction_accuracy', 0):.1%} accuracy in predicting price movement direction.</li>
            """
        elif 'accuracy' in metrics:
            accuracy = metrics['accuracy']
            if accuracy > 0.7:
                prediction_quality = "strong"
            elif accuracy > 0.6:
                prediction_quality = "moderate"
            elif accuracy > 0.55:
                prediction_quality = "weak"
            else:
                prediction_quality = "very weak"
                
            html_content += f"""
                <li>The best classification model shows a {prediction_quality} ability to predict market direction (Accuracy = {accuracy:.1%}).</li>
                <li>Market direction prediction works best for {best_model_info['company']} ({best_model_info['ticker']}).</li>
                <li>The F1 score of {metrics.get('f1', 0):.3f} indicates the model's balance between precision and recall.</li>
            """
    else:
        html_content += f"""
            <li>No strong predictive relationship was found between news sentiment and stock price movements.</li>
            <li>The analysis suggests that using sentiment alone is insufficient for reliable stock price prediction.</li>
        """
    
    html_content += f"""
            </ul>
        </div>
        
        <div class="section">
            <h2>Model Comparison Across Stocks</h2>
            <p>The analysis compares model performance across different stocks to identify which companies' price movements are most predictable using sentiment data.</p>
            
            <div class="img-container">
                <h3>Performance Metrics Comparison</h3>
                <img src="data:image/png;base64,{model_comparison_imgs[0]}" alt="Model Performance Comparison">
            </div>
            
            <div class="img-container">
                <h3>Most Commonly Used Features</h3>
                <img src="data:image/png;base64,{model_comparison_imgs[1]}" alt="Feature Usage Comparison">
            </div>
            
            <div class="img-container">
                <h3>Model Types Distribution</h3>
                <img src="data:image/png;base64,{model_comparison_imgs[2]}" alt="Model Types Comparison">
            </div>
            
            <div class="img-container">
                <h3>Best Model Details</h3>
                <img src="data:image/png;base64,{model_comparison_imgs[3]}" alt="Best Model Details">
            </div>
        </div>
    """
    
    # Add best model detail section if available
    if best_model_img:
        html_content += f"""
        <div class="section">
            <h2>Best Model Detailed Analysis</h2>
            <p>Detailed performance analysis of the best performing model: {best_model_info['model_type']} for {best_model_info['company']} ({best_model_info['ticker']})</p>
            
            <div class="img-container">
                <h3>Model Performance</h3>
                <img src="data:image/png;base64,{best_model_img}" alt="Best Model Performance">
            </div>
        """
        
        if best_feature_img:
            html_content += f"""
            <div class="img-container">
                <h3>Feature Importance</h3>
                <img src="data:image/png;base64,{best_feature_img}" alt="Feature Importance">
            </div>
            """
            
        if best_data_img:
            html_content += f"""
            <div class="img-container">
                <h3>Data Quality</h3>
                <img src="data:image/png;base64,{best_data_img}" alt="Data Quality">
            </div>
            """
            
        html_content += f"""
        </div>
        """
    
    # Add results table
    html_content += f"""
        <div class="section">
            <h2>Results Summary Table</h2>
            <table>
                <tr>
                    <th>Company</th>
                    <th>Ticker</th>
                    <th>Model Type</th>
                    <th>Performance</th>
                    <th>Features</th>
                </tr>
    """
    
    for ticker, info in results.items():
        if ticker in model_metrics and 'error' not in model_metrics[ticker]:
            metrics = model_metrics[ticker]
            
            # Determine performance metric
            if 'r2' in metrics:
                performance = f"R² = {metrics['r2']:.4f}"
            elif 'accuracy' in metrics:
                performance = f"Accuracy = {metrics['accuracy']:.2%}"
            else:
                performance = "N/A"
            
            # Get features (limit to first 3 for readability)
            features = info.get('features', [])
            if len(features) > 3:
                feature_text = ", ".join(features[:3]) + f" and {len(features)-3} more"
            else:
                feature_text = ", ".join(features)
            
            # Add row
            html_content += f"""
                <tr>
                    <td>{info['company']}</td>
                    <td>{ticker}</td>
                    <td>{metrics.get('model_type', 'Unknown')}</td>
                    <td>{performance}</td>
                    <td>{feature_text}</td>
                </tr>
            """
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Conclusions and Recommendations</h2>
            <p>Based on the analysis of news sentiment and stock price data, the following conclusions can be drawn:</p>
            
            <ul>
    """
    
    # Add conclusions based on results
    if best_model_info and 'metrics' in best_model_info:
        metrics = best_model_info['metrics']
        if 'r2' in metrics and metrics['r2'] > 0.1:
            html_content += f"""
                <li>News sentiment shows some predictive power for stock price movements, particularly for {best_model_info['company']}.</li>
                <li>The {best_model_info['model_type']} model performs best, suggesting this approach is suitable for sentiment-based prediction.</li>
                <li>Features like {', '.join(best_model_info['features'][:3]) if len(best_model_info['features']) >= 3 else ', '.join(best_model_info['features'])} are most informative for prediction.</li>
            """
        elif 'accuracy' in metrics and metrics['accuracy'] > 0.55:
            html_content += f"""
                <li>News sentiment can help predict market direction with {metrics['accuracy']:.1%} accuracy, particularly for {best_model_info['company']}.</li>
                <li>Classification models are effective for sentiment-based direction prediction.</li>
                <li>Features like {', '.join(best_model_info['features'][:3]) if len(best_model_info['features']) >= 3 else ', '.join(best_model_info['features'])} are most informative for direction prediction.</li>
            """
        else:
            html_content += f"""
                <li>News sentiment alone provides limited predictive power for stock price movements.</li>
                <li>More sophisticated models or additional data sources may be needed for reliable prediction.</li>
                <li>The relationship between news sentiment and stock prices appears to be weak or inconsistent.</li>
            """
    else:
        html_content += f"""
            <li>The analysis did not find a significant relationship between news sentiment and stock price movements.</li>
            <li>News sentiment may be already priced in by the time it becomes publicly available.</li>
            <li>Alternative approaches incorporating additional data sources are recommended.</li>
        """
    
    html_content += f"""
            </ul>
            
            <h3>Recommendations for Further Research</h3>
            <ul>
                <li>Incorporate real-time news data for more timely sentiment analysis.</li>
                <li>Explore more sophisticated NLP techniques for better sentiment extraction.</li>
                <li>Combine sentiment data with technical indicators for improved prediction.</li>
                <li>Investigate sector-specific sentiment effects.</li>
                <li>Explore non-linear relationships between sentiment and price movements.</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>Stock Sentiment Analysis Project - Generated {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at {report_path}")
    
    return report_path

def create_interactive_dashboard(results, model_metrics, best_model_info, output_file="interactive_dashboard.html"):
    """Create an interactive dashboard using Plotly"""
    output_dir = ensure_output_directory()
    dashboard_path = os.path.join(output_dir, output_file)
    
    # Create the dashboard layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Model Performance Comparison", 
            "Feature Importance",
            "Performance by Stock", 
            "Sentiment vs Return Correlation"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Model Performance Comparison (top left)
    tickers = []
    performance_values = []
    
    # Detect model type (regression or classification)
    has_regression = any('r2' in metrics for ticker, metrics in model_metrics.items() if 'error' not in metrics)
    
    if has_regression:
        # Collect R² values
        for ticker, metrics in model_metrics.items():
            if 'error' not in metrics and 'r2' in metrics:
                tickers.append(ticker)
                performance_values.append(metrics['r2'])
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=performance_values,
                name="R² Score",
                marker_color=['gold' if ticker == best_model_info.get('ticker') else 'royalblue' for ticker in tickers]
            ),
            row=1, col=1
        )
        
        # Add reference line
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(tickers)-0.5, y0=0, y1=0,
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Update layout
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
    else:
        # Collect accuracy values
        for ticker, metrics in model_metrics.items():
            if 'error' not in metrics and 'accuracy' in metrics:
                tickers.append(ticker)
                performance_values.append(metrics['accuracy'])
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=performance_values,
                name="Accuracy",
                marker_color=['gold' if ticker == best_model_info.get('ticker') else 'royalblue' for ticker in tickers]
            ),
            row=1, col=1
        )
        
        # Add reference line for random guessing
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(tickers)-0.5, y0=0.5, y1=0.5,
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Update layout
        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    
    # 2. Feature Importance (top right)
    if best_model_info and 'features' in best_model_info:
        # Get feature importance if possible
        try:
            model = best_model_info['model']
            features = best_model_info['features']
            
            # Extract importance values
            importance_values = []
            
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) > 1:
                    importance_values = np.abs(model.coef_[0])
                else:
                    importance_values = np.abs(model.coef_)
            
            # If we have pipeline, try to extract from the last step
            elif hasattr(model, 'steps'):
                final_step = model.steps[-1][1]
                if hasattr(final_step, 'feature_importances_'):
                    importance_values = final_step.feature_importances_
                elif hasattr(final_step, 'coef_'):
                    if len(final_step.coef_.shape) > 1:
                        importance_values = np.abs(final_step.coef_[0])
                    else:
                        importance_values = np.abs(final_step.coef_)
            
            if len(importance_values) == len(features):
                # Create DataFrame and sort
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance_values
                }).sort_values('Importance', ascending=False)
                
                # Take top 10 features
                top_features = importance_df.head(10)
                
                # Add horizontal bar chart
                fig.add_trace(
                    go.Bar(
                        y=top_features['Feature'],
                        x=top_features['Importance'],
                        orientation='h',
                        marker_color='lightgreen',
                        name="Feature Importance"
                    ),
                    row=1, col=2
                )
                
                # Update layout
                fig.update_xaxes(title_text="Importance", row=1, col=2)
                fig.update_layout(height=800)  # Make room for feature names
            else:
                # Add text if no importance available
                fig.add_annotation(
                    text="Feature importance not available",
                    x=0.5, y=0.5,
                    xref="x2", yref="y2",
                    showarrow=False,
                    font=dict(size=14)
                )
        except Exception as e:
            # Add text if extraction failed
            fig.add_annotation(
                text="Could not extract feature importance",
                x=0.5, y=0.5,
                xref="x2", yref="y2",
                showarrow=False,
                font=dict(size=14)
            )
    else:
        # Add text if no best model
        fig.add_annotation(
            text="No best model selected",
            x=0.5, y=0.5,
            xref="x2", yref="y2",
            showarrow=False,
            font=dict(size=14)
        )
    
    # 3. Performance by Stock Scatter Plot (bottom left)
    # Create a scatter plot of accuracy vs another metric
    stock_metrics = []
    
    if has_regression:
        # For regression, plot R² vs Direction Accuracy
        for ticker, metrics in model_metrics.items():
            if 'error' not in metrics and 'r2' in metrics:
                stock_name = results[ticker]['company'] if ticker in results else ticker
                stock_metrics.append({
                    'Ticker': ticker,
                    'Name': stock_name,
                    'R²': metrics['r2'],
                    'Direction Accuracy': metrics.get('direction_accuracy', 0.5),
                    'Is Best': ticker == best_model_info.get('ticker', '')
                })
        
        if stock_metrics:
            df = pd.DataFrame(stock_metrics)
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df['R²'],
                    y=df['Direction Accuracy'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=['gold' if is_best else 'royalblue' for is_best in df['Is Best']],
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    text=df['Ticker'],
                    textposition="top center",
                    name="Stocks"
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_xaxes(title_text="R² Score", row=2, col=1)
            fig.update_yaxes(title_text="Direction Accuracy", row=2, col=1)
    else:
        # For classification, plot Accuracy vs F1 Score
        for ticker, metrics in model_metrics.items():
            if 'error' not in metrics and 'accuracy' in metrics:
                stock_name = results[ticker]['company'] if ticker in results else ticker
                stock_metrics.append({
                    'Ticker': ticker,
                    'Name': stock_name,
                    'Accuracy': metrics['accuracy'],
                    'F1 Score': metrics.get('f1', 0),
                    'Is Best': ticker == best_model_info.get('ticker', '')
                })
        
        if stock_metrics:
            df = pd.DataFrame(stock_metrics)
            
            # Create scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df['Accuracy'],
                    y=df['F1 Score'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=['gold' if is_best else 'royalblue' for is_best in df['Is Best']],
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    text=df['Ticker'],
                    textposition="top center",
                    name="Stocks"
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_xaxes(title_text="Accuracy", row=2, col=1)
            fig.update_yaxes(title_text="F1 Score", row=2, col=1)
    
    # 4. Sentiment vs Return Correlation Heatmap (bottom right)
    # Create correlation matrix between sentiment and returns across stocks
    correlation_data = []
    
    for ticker, info in results.items():
        if 'data' not in info:
            continue
        
        data = info['data']
        
        # Find sentiment and return columns
        sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
        return_cols = [col for col in data.columns if 'return' in col.lower() and 'next' in col.lower()]
        
        if not sentiment_cols or not return_cols:
            continue
        
        # Take first sentiment and return columns
        sentiment_col = sentiment_cols[0]
        return_cols = [col for col in return_cols if data[col].notna().sum() > 10]
        
        if not return_cols:
            continue
            
        return_col = return_cols[0]
        
        # Calculate correlation
        valid_data = data[[sentiment_col, return_col]].dropna()
        if len(valid_data) < 10:
            continue
            
        corr = valid_data[sentiment_col].corr(valid_data[return_col])
        
        correlation_data.append({
            'Ticker': ticker,
            'Sentiment': sentiment_col,
            'Return': return_col,
            'Correlation': corr
        })
    
    if correlation_data:
        df = pd.DataFrame(correlation_data)
        
        # Create a pivot table for heatmap
        if len(df) > 1:
            # If we have multiple stocks, create heatmap
            tickers = df['Ticker'].unique()
            z_values = df['Correlation'].values
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=[z_values],
                    x=tickers,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=[[f"{z:.3f}" for z in z_values]],
                    texttemplate="%{text}",
                    showscale=True
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_xaxes(title_text="Stock", row=2, col=2)
            fig.update_yaxes(title_text="Sentiment-Return Correlation", row=2, col=2, showticklabels=False)
        else:
            # If only one stock, show correlation value as text
            fig.add_annotation(
                text=f"Sentiment-Return Correlation:<br>{df['Ticker'].iloc[0]}: {df['Correlation'].iloc[0]:.3f}",
                x=0.5, y=0.5,
                xref="x4", yref="y4",
                showarrow=False,
                font=dict(size=16)
            )
    else:
        # Show message if no correlation data
        fig.add_annotation(
            text="Insufficient data for correlation analysis",
            x=0.5, y=0.5,
            xref="x4", yref="y4",
            showarrow=False,
            font=dict(size=14)
        )
    
    # Update overall layout
    fig.update_layout(
        title_text="Stock Sentiment Analysis Dashboard",
        height=800,
        width=1200,
        showlegend=False,
    )
    
    # Write to HTML file
    fig.write_html(dashboard_path)
    
    return dashboard_path