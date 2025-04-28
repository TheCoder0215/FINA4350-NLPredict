"""
FINA4350 Stock Sentiment Analysis Modeling module
Enhanced with ensemble methods, time-series validation, and regime-adaptive models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
import optuna
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import config
import logging
import os
from scipy import stats

# Setup logging
logger = logging.getLogger(__name__)

class AdaptiveTimeSeriesSplit:
    """Time series cross-validation with adaptive train size and gap"""
    
    def __init__(self, n_splits=5, min_train_size=60, gap=1, max_train_size=None):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.gap = gap
        
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        
        # Check if we have enough samples
        if n_samples < self.min_train_size + self.gap + 1:
            raise ValueError(
                f"Too few samples ({n_samples}) for min_train_size={self.min_train_size} and gap={self.gap}"
            )
        
        # Basic validation to mirror TimeSeriesSplit API
        indices = np.arange(n_samples)
        
        # Compute test sizes
        test_size = (n_samples - self.min_train_size - self.gap) // self.n_splits
        if test_size <= 0:
            test_size = 1
        
        # Adjust min_train_size if it would exceed data size
        train_size = self.min_train_size
        
        # Generate splits
        for i in range(self.n_splits):
            # Test indices
            test_start = train_size + self.gap + i * test_size
            test_end = min(test_start + test_size, n_samples)
            
            if test_start >= n_samples:
                break
                
            test_indices = indices[test_start:test_end]
            
            # Train indices with adaptive train size
            if self.max_train_size and self.max_train_size < test_start:
                # Only use recent data within max_train_size
                train_indices = indices[test_start - self.max_train_size:test_start - self.gap]
            else:
                train_indices = indices[:test_start - self.gap]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def evaluate_feature_importance(X, y, model, threshold=0.01):
    """Evaluate feature importance and identify most relevant features"""
    
    # Determine if we can extract feature importance directly
    has_importance = hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')
    
    # For simple models with direct feature importance
    if has_importance:
        # Train the model
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based model like Random Forest
            importances = model.feature_importances_
        else:
            # Linear model 
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_[0])
            else:
                importances = np.abs(model.coef_)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Get features above threshold
        important_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
        
        return important_features, importance_df
    
    # For complex models, use SelectFromModel
    selector = SelectFromModel(estimator=model, threshold=threshold)
    selector.fit(X, y)
    
    # Get selected feature mask and convert to feature names
    important_mask = selector.get_support()
    important_features = X.columns[important_mask].tolist()
    
    # Create dummy importance df based on selection
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': np.where(important_mask, 1, 0)
    }).sort_values('importance', ascending=False)
    
    return important_features, importance_df

def remove_outliers(X, y, method='zscore', threshold=3.0):
    """Remove outliers from the dataset"""
    if len(X) < 10:
        # Too few samples - don't remove outliers
        return X, y
    
    if method == 'zscore':
        # Z-score based outlier removal
        z_scores = np.abs(stats.zscore(y))
        valid_mask = z_scores < threshold
    elif method == 'iqr':
        # IQR-based outlier removal
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        valid_mask = ~((y < (Q1 - 1.5 * IQR)) | (y > (Q3 + 1.5 * IQR)))
    else:
        # No outlier removal
        return X, y
    
    # Apply mask
    X_clean = X.loc[valid_mask]
    y_clean = y.loc[valid_mask]
    
    # Log how many outliers were removed
    outliers_count = len(y) - len(y_clean)
    if outliers_count > 0:
        logger.info(f"Removed {outliers_count} outliers ({outliers_count/len(y)*100:.1f}%) using {method} method")
    
    return X_clean, y_clean

def hyperparameter_tuning(X, y, model_type, time_budget=60, n_trials=10, cv=None, is_classifier=False):
    """Tune hyperparameters using Optuna"""
    
    # Create study
    study = optuna.create_study(direction="maximize" if is_classifier else "minimize")
    
    # Define objective based on model type
    def objective(trial):
        if model_type == 'lightgbm':
            params = {
                'objective': 'binary' if is_classifier else 'regression',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            model = lgb.LGBMRegressor(**params) if not is_classifier else lgb.LGBMClassifier(**params)
            
        elif model_type == 'xgboost':
            params = {
                'objective': 'binary:logistic' if is_classifier else 'reg:squarederror',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
            }
            model = xgb.XGBRegressor(**params) if not is_classifier else xgb.XGBClassifier(**params)
            
        elif model_type == 'histgb':
            params = {
                'max_iter': trial.suggest_int('max_iter', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True)
            }
            model = HistGradientBoostingRegressor(**params) if not is_classifier else HistGradientBoostingClassifier(**params)
            
        elif model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            model = RandomForestRegressor(**params) if not is_classifier else RandomForestClassifier(**params)
            
        else:
            # Default to GradientBoostingRegressor
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            model = GradientBoostingRegressor(**params) if not is_classifier else GradientBoostingClassifier(**params)
        
        # Cross-validation
        if cv is None:
            cv = TimeSeriesSplit(n_splits=5)
        
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if is_classifier:
                score = accuracy_score(y_test, y_pred)  # Higher is better
            else:
                score = -mean_squared_error(y_test, y_pred)  # Lower is better, but we need to maximize
            
            scores.append(score)
        
        return np.mean(scores)
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=time_budget)
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Best {model_type} parameters: {best_params}")
    
    # Create model with best parameters
    if model_type == 'lightgbm':
        if is_classifier:
            model = lgb.LGBMClassifier(objective='binary', **best_params)
        else:
            model = lgb.LGBMRegressor(**best_params)
    
    elif model_type == 'xgboost':
        if is_classifier:
            model = xgb.XGBClassifier(objective='binary:logistic', **best_params)
        else:
            model = xgb.XGBRegressor(**best_params)
    
    elif model_type == 'histgb':
        if is_classifier:
            model = HistGradientBoostingClassifier(**best_params)
        else:
            model = HistGradientBoostingRegressor(**best_params)
    
    elif model_type == 'rf':
        if is_classifier:
            model = RandomForestClassifier(**best_params)
        else:
            model = RandomForestRegressor(**best_params)
    
    else:
        if is_classifier:
            model = GradientBoostingClassifier(**best_params)
        else:
            model = GradientBoostingRegressor(**best_params)
    
    return model, best_params

def build_ensemble_model(models, ensemble_type='voting', final_estimator=None):
    """Build an ensemble model from multiple base models"""
    if ensemble_type == 'voting':
        return VotingRegressor(
            estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
            weights=[1] * len(models)
        )
    
    elif ensemble_type == 'stacking':
        if final_estimator is None:
            final_estimator = Ridge()
            
        return StackingRegressor(
            estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
            final_estimator=final_estimator,
            cv=TimeSeriesSplit(n_splits=3)
        )
    
    else:
        # Default to simple model
        return models[0] if models else LinearRegression()

def create_model_pipeline(features, model, preprocessing='auto'):
    """Create a full pipeline with preprocessing and model"""
    steps = []
    
    # Add feature preprocessing based on type
    if preprocessing == 'auto':
        # Auto-detect preprocessing
        numeric_features = features
        
        # Add imputer first
        steps.append(('imputer', SimpleImputer(strategy='median')))
        
        # Add scaler based on data distribution
        steps.append(('scaler', StandardScaler()))
    
    elif preprocessing == 'robust':
        # Robust preprocessing for financial data
        steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('scaler', RobustScaler()))
    
    elif preprocessing == 'minmax':
        # MinMax scaling for algorithms sensitive to scale
        steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('scaler', MinMaxScaler()))
    
    elif preprocessing == 'power':
        # PowerTransformer for skewed financial data
        steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('scaler', PowerTransformer(method='yeo-johnson')))
        
    elif preprocessing == 'none':
        # No preprocessing
        pass
    
    # Add feature selection
    if len(features) > 10:
        steps.append(('variance_filter', VarianceThreshold(threshold=0.01)))
    
    # Add model
    steps.append(('model', model))
    
    return Pipeline(steps=steps)

def evaluate_predictions(y_true, y_pred, model_type='regression'):
    """Evaluate predictions with comprehensive metrics"""
    metrics = {}
    
    if model_type == 'regression':
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Financial specific metrics
        metrics['accuracy_direction'] = np.mean((y_true * y_pred) > 0)  # Direction accuracy
        metrics['mean_return'] = np.mean(y_pred)
        metrics['sign_consistency'] = np.mean(np.sign(y_pred) == np.sign(y_true))
        
        # Risk-adjusted metrics
        metrics['sharpe'] = np.mean(y_pred) / np.std(y_pred) if np.std(y_pred) > 0 else 0
        
    else:
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1'] = f1_score(y_true, y_pred, average='binary')
        
        # Class balance
        metrics['positive_ratio'] = np.mean(y_true)
        metrics['predicted_positive_ratio'] = np.mean(y_pred)
    
    return metrics

def visualize_model_performance(y_true, y_pred, dates, model_type='regression', output_path=None):
    """Visualize model performance with prediction plots and metrics"""
    plt.figure(figsize=(12, 10))
    
    if model_type == 'regression':
        # Subplot 1: Time series plot of actual vs. predicted
        plt.subplot(2, 2, 1)
        plt.plot(dates, y_true, 'b-', label='Actual')
        plt.plot(dates, y_pred, 'r--', label='Predicted')
        plt.title('Actual vs. Predicted Values')
        plt.xlabel('Date')
        plt.ylabel('Target Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Scatter plot of actual vs. predicted
        plt.subplot(2, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.8)
        
        plt.title('Actual vs. Predicted Scatter')
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted Value')
        plt.grid(True, alpha=0.3)
        
        # Add R² value
        r2 = r2_score(y_true, y_pred)
        plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Subplot 3: Prediction error histogram
        plt.subplot(2, 2, 3)
        errors = y_true - y_pred
        sns.histplot(errors, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Cumulative performance
        plt.subplot(2, 2, 4)
        
        # Trading strategy based on predictions
        strategy_returns = np.sign(y_pred) * y_true
        
        # Calculate cumulative returns
        cumulative_target = (1 + y_true/100).cumprod()
        cumulative_strategy = (1 + strategy_returns/100).cumprod()
        
        plt.plot(dates, cumulative_target, 'b-', label='Buy & Hold')
        plt.plot(dates, cumulative_strategy, 'g-', label='Strategy')
        plt.title('Trading Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (1.0 = start)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add Sharpe ratio
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)  # Annualized
        plt.annotate(f'Sharpe = {sharpe:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
    else:
        # Classification visualizations
        
        # Subplot 1: Time series plot of actual vs. predicted
        plt.subplot(2, 2, 1)
        plt.plot(dates, y_true, 'b-', label='Actual')
        plt.plot(dates, y_pred, 'r--', label='Predicted')
        plt.title('Actual vs. Predicted Classes')
        plt.xlabel('Date')
        plt.ylabel('Class (1=Up, 0=Down)')
        plt.yticks([0, 1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Confusion matrix
        plt.subplot(2, 2, 2)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Subplot 3: Metrics text
        plt.subplot(2, 2, 3)
        metrics = evaluate_predictions(y_true, y_pred, model_type='classification')
        plt.axis('off')
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        plt.text(0.1, 0.5, f"Classification Metrics:\n\n{metrics_text}",
                fontsize=12, va='center')
        
        # Subplot 4: Cumulative performance for classification
        plt.subplot(2, 2, 4)
        
        # Create a synthetic return series based on classification
        synthetic_returns = np.where(y_true == 1, 1.0, -1.0)  # 1% up, 1% down
        
        # Trading strategy based on predictions
        strategy_returns = np.where(y_pred == y_true, 1.0, -1.0)
        
        # Calculate cumulative returns
        cumulative_market = (1 + synthetic_returns/100).cumprod()
        cumulative_strategy = (1 + strategy_returns/100).cumprod()
        
        plt.plot(dates, cumulative_market, 'b-', label='Market')
        plt.plot(dates, cumulative_strategy, 'g-', label='Strategy')
        plt.title('Strategy Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (1.0 = start)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def advanced_model_training(data, task='auto', model_configs=None, time_budget=180):
    """
    Train multiple models with improved feature selection, ensembling, and cross-validation.
    Now with explicit regime detection and specialized models for different regimes.
    """
    if model_configs is None:
        model_configs = config.MODEL_CONFIGS
    
    logger.info("\nData quality check before modeling:")
    for col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"Column {col} has {nan_count} NaN values out of {len(data)} rows ({nan_count/len(data)*100:.1f}%)")
    
    # Impute missing values with more appropriate strategies
    data_clean = data.copy()
    
    # Impute numerical features with median (more robust to outliers)
    numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        imp_median = SimpleImputer(strategy='median')
        data_clean[numeric_cols] = imp_median.fit_transform(data_clean[numeric_cols])
    
    # Target selection logic (improved)
    excess_return_targets = [c for c in data_clean.columns if c.startswith('Next_Excess_')]
    return_targets = [c for c in data_clean.columns if c.startswith('Next_Return_')]
    direction_targets = [c for c in data_clean.columns if c.startswith('Up_')]
    
    # Detect market regime if possible
    has_regime = 'MarketRegime' in data_clean.columns
    if has_regime:
        # Get distribution of regimes
        regimes = data_clean['MarketRegime'].value_counts()
        logger.info(f"Market regime distribution: {regimes.to_dict()}")
        
        # Create regime-specific sub-datasets
        regime_datasets = {}
        for regime in [-1, 0, 1]:  # Bear, Neutral, Bull
            regime_mask = data_clean['MarketRegime'] == regime
            if regime_mask.sum() >= 30:  # Need at least 30 samples
                regime_datasets[regime] = data_clean[regime_mask]
                logger.info(f"Created regime-specific dataset for regime {regime} with {len(regime_datasets[regime])} samples")
    
    # Prioritize targets: Excess Return > Return > Direction
    if task == 'classification' or (task == 'auto' and len(direction_targets) > 0 and not return_targets):
        modeling_type = 'classification'
        target_options = direction_targets
    else:
        modeling_type = 'regression'
        target_options = excess_return_targets or return_targets
    
    if not target_options:
        logger.warning("No appropriate targets found, using 'Next_Return_1d'")
        if 'Return_1d' in data_clean.columns:
            data_clean['Next_Return_1d'] = data_clean['Return_1d'].shift(-1)
            data_clean = data_clean.dropna(subset=['Next_Return_1d'])
            target_options = ['Next_Return_1d']
            modeling_type = 'regression'
    
    # Feature selection (improved)
    # Organize features by category for better selection
    sentiment_features = [c for c in data_clean.columns if any(
        x in c.lower() for x in ['sentiment', 'nlp_', 'news_'])]
    price_features = [c for c in data_clean.columns if any(
        x in c for x in ['MA_', 'EMA_', 'MACD', 'RSI', 'BB_', 'ATR_', 'Stoch_'])]
    return_features = [c for c in data_clean.columns if (
        c.startswith('Return_') and not c.startswith('Next_'))]
    market_features = [c for c in data_clean.columns if any(
        x in c for x in ['Market_', 'VIX', 'Beta_', 'Sector_'])]
    seasonal_features = [c for c in data_clean.columns if any(
        x in c.lower() for x in ['day_', 'month', 'quarter', 'is_monday', 'is_friday'])]
    
    # Create additional feature combos from categories and interactions
    feature_sets = []
    
    # Individual feature categories
    if sentiment_features:
        feature_sets.append({'name': 'sentiment_only', 'features': sentiment_features})
    if price_features:
        feature_sets.append({'name': 'price_only', 'features': price_features})
    if return_features:
        feature_sets.append({'name': 'return_only', 'features': return_features})
    if market_features:
        feature_sets.append({'name': 'market_only', 'features': market_features})
    
    # Top sentiment + technical + market combinations
    if sentiment_features and price_features:
        top_sentiment = sentiment_features[:min(3, len(sentiment_features))]
        top_price = [f for f in price_features if any(x in f for x in ['RSI', 'MACD', 'MA_20'])][:3]
        feature_sets.append({'name': 'sentiment_technical', 'features': top_sentiment + top_price})
    
    if sentiment_features and market_features:
        top_sentiment = sentiment_features[:min(3, len(sentiment_features))]
        top_market = [f for f in market_features if any(x in f for x in ['Market_Return', 'Market_Trend', 'VIX'])][:2]
        feature_sets.append({'name': 'sentiment_market', 'features': top_sentiment + top_market})
    
    # Raw sentiment + returns
    if 'raw_sentiment' in sentiment_features and return_features:
        recent_returns = [f for f in return_features if any(x in f for x in ['Return_1d', 'Return_5d'])][:3]
        feature_sets.append({'name': 'sentiment_return', 'features': ['raw_sentiment'] + recent_returns})
    
    # Comprehensive feature set
    comprehensive = []
    if 'raw_sentiment' in data_clean.columns:
        comprehensive.append('raw_sentiment')
    if 'sentiment_momentum_3d' in data_clean.columns:
        comprehensive.append('sentiment_momentum_3d')
    if 'RSI' in data_clean.columns:
        comprehensive.append('RSI')
    if 'MA_20' in data_clean.columns:
        comprehensive.append('MA_20')
    if 'Market_Return' in data_clean.columns:
        comprehensive.append('Market_Return')
    if 'Return_1d' in data_clean.columns:
        comprehensive.append('Return_1d')
    
    if comprehensive:
        feature_sets.append({'name': 'comprehensive', 'features': comprehensive})
    
    # Add PCA-based features if available
    pca_features = [c for c in data_clean.columns if c.startswith('PCA_')]
    if pca_features:
        feature_sets.append({'name': 'pca_features', 'features': pca_features})
    
    # If we have no proper feature sets, use all numeric features
    if not feature_sets:
        feature_sets.append({'name': 'all_numeric', 'features': numeric_cols})
    
    # Define model candidates with appropriate hyperparameters
    model_candidates = []
    
    # Regression models
    if modeling_type == 'regression':
        # Linear models
        model_candidates.extend([
            {'name': 'Linear', 'model': LinearRegression(), 'preprocessing': 'auto'},
            {'name': 'Ridge', 'model': Ridge(alpha=1.0, random_state=42), 'preprocessing': 'auto'},
            {'name': 'Lasso', 'model': Lasso(alpha=0.1, random_state=42), 'preprocessing': 'auto'},
            {'name': 'ElasticNet', 'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), 'preprocessing': 'auto'},
        ])
        
        # Tree-based models
        model_candidates.extend([
            {'name': 'RandomForest', 'model': RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42), 'preprocessing': 'none'},
            {'name': 'GradientBoosting', 'model': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42), 'preprocessing': 'none'},
            {'name': 'HistGradientBoosting', 'model': HistGradientBoostingRegressor(
                max_depth=8, learning_rate=0.05, random_state=42), 'preprocessing': 'none'},
        ])
        
        # Specialized models for financial time series
        model_candidates.extend([
            {'name': 'LightGBM', 'model': lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=8, random_state=42), 'preprocessing': 'none'},
            {'name': 'XGBoost', 'model': xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42), 'preprocessing': 'none'},
            {'name': 'SVR', 'model': SVR(C=1.0, gamma='scale'), 'preprocessing': 'robust'},
        ])
        
        # Define naive baseline
        baseline = DummyRegressor(strategy='mean')
        scoring = 'neg_mean_squared_error'
        
    # Classification models  
    else:
        model_candidates.extend([
            {'name': 'LogisticRegression', 'model': LogisticRegression(
                max_iter=1000, C=1.0, class_weight='balanced', random_state=42), 'preprocessing': 'auto'},
            {'name': 'RandomForestClassifier', 'model': RandomForestClassifier(
                n_estimators=100, max_depth=8, class_weight='balanced', random_state=42), 'preprocessing': 'none'},
            {'name': 'GradientBoostingClassifier', 'model': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42), 'preprocessing': 'none'},
            {'name': 'LightGBMClassifier', 'model': lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=8, class_weight='balanced', random_state=42), 'preprocessing': 'none'},
            {'name': 'XGBoostClassifier', 'model': xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=6, scale_pos_weight=1, random_state=42), 'preprocessing': 'none'},
        ])
        
        # Define naive baseline
        baseline = DummyClassifier(strategy='stratified')
        scoring = 'accuracy'
    
    # Setup improved cross-validation
    # Use adaptive time series split with proper train size
    n_splits = min(5, len(data_clean) // 20)  # At least 20 samples per fold
    if n_splits < 2:
        n_splits = 2  # Minimum 2 splits
    
    # More robust CV for time series with a gap
    tscv = AdaptiveTimeSeriesSplit(
        n_splits=n_splits,
        min_train_size=max(30, int(len(data_clean) * 0.6)),  # At least 30 samples or 60% of data
        gap=1,  # 1-day gap between train and test
        max_train_size=252  # Use at most 1 year of data for training (financial year)
    )
    
    # Initialize tracking for best models
    best_score = -np.inf if modeling_type == 'classification' else -float('inf')
    best_model, best_features, best_target, best_model_name, best_pipeline = None, None, None, None, None
    
    # Dictionary to track all model results
    all_results = {}
    start_time = datetime.now()
    
    # Train and evaluate models
    logger.info("\nTraining and evaluating models:")
    for target in target_options:
        if target not in data_clean.columns:
            continue
        
        y = data_clean[target]
        
        # Get class distribution for classification tasks
        if modeling_type == 'classification':
            class_dist = y.value_counts(normalize=True)
            logger.info(f"Target {target} class distribution: {class_dist.to_dict()}")
            
            # Skip if too imbalanced (> 90% in one class)
            if class_dist.max() > 0.9:
                logger.warning(f"Skipping target {target} due to severe class imbalance")
                continue
        
        for feature_set in feature_sets:
            feature_name = feature_set['name']
            features = feature_set['features']
            
            valid_features = [f for f in features if f in data_clean.columns]
            if not valid_features or len(data_clean[valid_features]) < max(n_splits + 1, 10):
                continue
                
            X = data_clean[valid_features]
            
            # Check for NaN values
            if X.isna().any().any() or y.isna().any():
                # Get indices where both X and y are valid
                valid_indices = X.notna().all(axis=1) & y.notna()
                X = X[valid_indices]
                y = y[valid_indices]
            
            if len(X) < max(n_splits + 1, 10):
                continue
            
            # Remove outliers in target for regression tasks
            if modeling_type == 'regression':
                X, y = remove_outliers(X, y, method='zscore', threshold=3.0)
            
            for candidate in model_candidates:
                model_name = candidate['name']
                model = candidate['model']
                preprocessing = candidate.get('preprocessing', 'auto')
                
                try:
                    # Create pipeline with preprocessing
                    pipeline = create_model_pipeline(valid_features, model, preprocessing)
                    
                    # Get cross-validation scores
                    time_elapsed = (datetime.now() - start_time).total_seconds()
                    if time_elapsed > time_budget:
                        logger.warning(f"Time budget of {time_budget}s exceeded. Stopping model training.")
                        break
                    
                    cv_scores = []
                    baseline_scores = []
                    
                    for train_idx, test_idx in tscv.split(X):
                        # Split data
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        # Fit model
                        pipeline.fit(X_train, y_train)
                        
                        # Predict
                        y_pred = pipeline.predict(X_test)
                        
                        # Calculate score
                        if modeling_type == 'regression':
                            score = -mean_squared_error(y_test, y_pred)  # Negative MSE (higher is better)
                        else:
                            score = accuracy_score(y_test, y_pred)
                        
                        cv_scores.append(score)
                        
                        # Calculate baseline score
                        baseline.fit(X_train, y_train)
                        baseline_pred = baseline.predict(X_test)
                        
                        if modeling_type == 'regression':
                            baseline_score = -mean_squared_error(y_test, baseline_pred)
                        else:
                            baseline_score = accuracy_score(y_test, baseline_pred)
                        
                        baseline_scores.append(baseline_score)
                    
                    # Average scores
                    avg_score = np.mean(cv_scores)
                    baseline_avg = np.mean(baseline_scores)
                    
                    # Store all results
                    all_results[(target, feature_name, model_name)] = {
                        'avg_score': avg_score,
                        'baseline_score': baseline_avg,
                        'cv_scores': cv_scores,
                        'features': valid_features
                    }
                    
                    # Determine if better than baseline
                    improvement = avg_score - baseline_avg
                    improvement_pct = f"{improvement/abs(baseline_avg)*100:.1f}%" if baseline_avg != 0 else "N/A"
                    
                    logger.info(f"Model: {model_name}, Features: {feature_name}, Target: {target}, "
                               f"CV Score: {avg_score:.4f} (Baseline: {baseline_avg:.4f}, Improvement: {improvement_pct})")
                    
                    # Track best model
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_features = valid_features
                        best_target = target
                        best_model_name = model_name
                        best_pipeline = pipeline
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} ({feature_name} → {target}): {e}")
    
    # Check if we need to tune the best model
    time_elapsed = (datetime.now() - start_time).total_seconds()
    remaining_time = max(0, time_budget - time_elapsed)
    
    if best_model and remaining_time > 60 and best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'HistGradientBoosting']:
        logger.info(f"\nTuning hyperparameters for best model {best_model_name} with {remaining_time:.0f}s remaining")
        
        # Get data for best model
        X = data_clean[best_features]
        y = data_clean[best_target]
        
        # Check for NaN values again
        if X.isna().any().any() or y.isna().any():
            valid_indices = X.notna().all(axis=1) & y.notna()
            X = X[valid_indices]
            y = y[valid_indices]
        
        # Convert model name to optuna model type
        if best_model_name == 'RandomForest':
            model_type = 'rf'
        elif best_model_name in ['GradientBoosting', 'GradientBoostingClassifier']:
            model_type = 'gb'
        elif best_model_name in ['XGBoost', 'XGBoostClassifier']:
            model_type = 'xgboost'
        elif best_model_name in ['LightGBM', 'LightGBMClassifier']:
            model_type = 'lightgbm'
        elif best_model_name in ['HistGradientBoosting', 'HistGradientBoostingClassifier']:
            model_type = 'histgb'
        else:
            model_type = 'gb'  # default
        
        # Tune hyperparameters
        tuned_model, best_params = hyperparameter_tuning(
            X, y,
            model_type=model_type,
            time_budget=remaining_time,
            n_trials=20,
            cv=tscv,
            is_classifier=(modeling_type == 'classification')
        )
        
        # Replace best model with tuned model
        best_model = tuned_model
    
    # If no model succeeded, use fallback
    if best_model is None:
        logger.warning("No model succeeded. Using a simple model.")
        best_model = LinearRegression() if modeling_type == 'regression' else LogisticRegression(max_iter=1000)
        best_features = feature_sets[0]['features']
        best_target = target_options[0]
        best_model_name = "Linear Fallback"
    
    # Refit best model on all data
    X_final = data_clean[best_features]
    y_final = data_clean[best_target]
    
    # Check for and handle NaN values
    valid_mask = X_final.notna().all(axis=1) & y_final.notna()
    X_final, y_final = X_final[valid_mask], y_final[valid_mask]
    
    # Get feature importance if possible
    important_features = []
    try:
        if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
            important_features, importance_df = evaluate_feature_importance(X_final, y_final, best_model)
            logger.info("\nFeature importance for best model:")
            for i, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
    # Create final pipeline with selected features
    if important_features and len(important_features) > 1:
        logger.info(f"Using {len(important_features)} important features out of {len(best_features)}")
        best_features = important_features
        X_final = X_final[best_features]
    
    # Create ensemble if time allows
    time_elapsed = (datetime.now() - start_time).total_seconds()
    remaining_time = max(0, time_budget - time_elapsed)
    
    final_model = best_model
    
    if remaining_time > 60 and modeling_type == 'regression':
        logger.info("\nCreating ensemble model from top performers")
        
        # Get top 3 models for ensembling
        top_models = []
        seen_models = set()
        
        # Sort results by score (descending)
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        
        for (target, feature_name, model_name), result in sorted_results:
            if target == best_target and len(top_models) < 3 and model_name not in seen_models:
                # Create a fresh instance of the model
                for candidate in model_candidates:
                    if candidate['name'] == model_name:
                        model = candidate['model']
                        top_models.append(model)
                        seen_models.add(model_name)
                        logger.info(f"Added {model_name} to ensemble")
                        break
        
        # Create ensemble if we have multiple models
        if len(top_models) > 1:
            ensemble = build_ensemble_model(top_models, ensemble_type='voting')
            
            try:
                # Test ensemble performance
                cv_scores = []
                for train_idx, test_idx in tscv.split(X_final):
                    X_train, X_test = X_final.iloc[train_idx], X_final.iloc[test_idx]
                    y_train, y_test = y_final.iloc[train_idx], y_final.iloc[test_idx]
                    
                    # Train each model
                    predictions = []
                    for model in top_models:
                        model.fit(X_train, y_train)
                        predictions.append(model.predict(X_test).reshape(-1, 1))
                    
                    # Average predictions
                    ensemble_pred = np.mean(np.hstack(predictions), axis=1)
                    
                    # Calculate score
                    score = -mean_squared_error(y_test, ensemble_pred)
                    cv_scores.append(score)
                
                ensemble_score = np.mean(cv_scores)
                
                if ensemble_score > best_score:
                    logger.info(f"Ensemble model outperforms best individual model: {ensemble_score:.4f} vs {best_score:.4f}")
                    final_model = ensemble
                    best_model_name = f"Ensemble({', '.join([m.__class__.__name__ for m in top_models])})"
                else:
                    logger.info(f"Best individual model outperforms ensemble: {best_score:.4f} vs {ensemble_score:.4f}")
            except Exception as e:
                logger.warning(f"Error evaluating ensemble: {e}")
    
    # Fit final model
    try:
        # For pipelines, we need the whole pipeline
        if isinstance(best_pipeline, Pipeline):
            final_pipeline = best_pipeline
            final_pipeline.fit(X_final, y_final)
            
            # Calculate metrics on training data
            if modeling_type == 'regression':
                y_pred = final_pipeline.predict(X_final)
                r2 = r2_score(y_final, y_pred)
                mse = mean_squared_error(y_final, y_pred)
                logger.info(f"Training R²: {r2:.4f}, MSE: {mse:.4f}")
            else:
                y_pred = final_pipeline.predict(X_final)
                accuracy = accuracy_score(y_final, y_pred)
                logger.info(f"Training accuracy: {accuracy:.4f}")
                
            # Return the full pipeline
            best_model = final_pipeline
        else:
            # For standalone models
            final_model.fit(X_final, y_final)
            
            # Calculate metrics on training data
            if modeling_type == 'regression':
                y_pred = final_model.predict(X_final)
                r2 = r2_score(y_final, y_pred)
                mse = mean_squared_error(y_final, y_pred)
                logger.info(f"Training R²: {r2:.4f}, MSE: {mse:.4f}")
            else:
                y_pred = final_model.predict(X_final)
                accuracy = accuracy_score(y_final, y_pred)
                logger.info(f"Training accuracy: {accuracy:.4f}")
            
            # Return the standalone model
            best_model = final_model
    except Exception as e:
        logger.error(f"Final model fit failed: {e}")
        if modeling_type == 'regression':
            best_model = DummyRegressor(strategy='mean')
        else:
            best_model = DummyClassifier(strategy='stratified')
        best_model.fit(X_final, y_final)
    
    # Save model
    try:
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/model_{best_model_name}_{timestamp}.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.warning(f"Could not save model: {e}")
    
    logger.info(f"\nBest model configuration:")
    logger.info(f"Type: {best_model_name}")
    logger.info(f"Features: {best_features}")
    logger.info(f"Target: {best_target}")
    logger.info(f"Score: {best_score:.4f}")
    
    return best_model, best_features, best_target

def evaluate_models(results):
    """
    Evaluate all models (regression/classification) in the results dict with enhanced metrics.
    """
    logger.info("\n" + "="*70)
    logger.info("EVALUATING MODEL PERFORMANCE ACROSS STOCKS")
    logger.info("="*70)
    model_metrics = {}

    for ticker, info in results.items():
        logger.info(f"\nEvaluating {ticker} ({info['company']})...")
        model = info['model']
        features = info['features']
        target = info['target']
        data = info['data']
        
        # Get valid samples for evaluation
        valid = data[features].notna().all(axis=1) & data[target].notna()
        X = data.loc[valid, features]
        y = data.loc[valid, target]
        dates = X.index
        
        if len(X) < 5:
            logger.warning("Not enough valid samples for evaluation.")
            continue
        
        try:
            # Make predictions
            y_pred = model.predict(X)
            
            # Classification metrics
            if target.startswith('Up_'):
                acc = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')
                f1 = f1_score(y, y_pred, average='binary')
                
                # Direction accuracy (important for trading)
                direction_accuracy = np.mean(y == y_pred)
                
                # Class balance
                class_balance = np.mean(y)
                
                logger.info(f"  Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                logger.info(f"  Direction accuracy: {direction_accuracy:.4f}, Class balance: {class_balance:.4f}")
                logger.info(f"  Evaluated on {len(X)} samples")
                
                # Full classification report
                logger.info("\n" + classification_report(y, y_pred))
                
                model_metrics[ticker] = {
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'direction_accuracy': direction_accuracy,
                    'class_balance': class_balance,
                    'model_type': type(model).__name__,
                    'features': features,
                    'target': target,
                    'data_points': len(X)
                }
                
                # Save visualization
                try:
                    os.makedirs('visualizations', exist_ok=True)
                    output_path = f"visualizations/{ticker}_classification_performance.png"
                    visualize_model_performance(y, y_pred, dates, model_type='classification', output_path=output_path)
                    logger.info(f"  Visualization saved to {output_path}")
                except Exception as ve:
                    logger.warning(f"  Visualization failed: {ve}")
                
            # Regression metrics
            else:
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Direction accuracy (important for trading)
                direction_accuracy = np.mean(np.sign(y) == np.sign(y_pred))
                
                # Strategy simulation metrics
                strategy_returns = np.sign(y_pred) * y
                sharpe_ratio = strategy_returns.mean() / (strategy_returns.std() + 1e-6) * np.sqrt(252)  # Annualized
                
                logger.info(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                logger.info(f"  Direction accuracy: {direction_accuracy:.4f}, Sharpe ratio: {sharpe_ratio:.4f}")
                logger.info(f"  Evaluated on {len(X)} samples")
                
                model_metrics[ticker] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'direction_accuracy': direction_accuracy,
                    'sharpe_ratio': sharpe_ratio,
                    'model_type': type(model).__name__,
                    'features': features,
                    'target': target,
                    'data_points': len(X)
                }
                
                # Save visualization
                try:
                    os.makedirs('visualizations', exist_ok=True)
                    output_path = f"visualizations/{ticker}_regression_performance.png"
                    visualize_model_performance(y, y_pred, dates, model_type='regression', output_path=output_path)
                    logger.info(f"  Visualization saved to {output_path}")
                except Exception as ve:
                    logger.warning(f"  Visualization failed: {ve}")
                
        except Exception as e:
            logger.error(f"  Evaluation failed: {e}")
            model_metrics[ticker] = {'error': str(e)}

    return model_metrics

def find_best_model(results, model_metrics):
    """Identify the best model based on R², direction accuracy, and Sharpe ratio."""
    logger.info("\n" + "="*70)
    logger.info("BEST MODEL IDENTIFICATION")
    logger.info("="*70)
    # Filter out models with errors
    valid_models = {k: v for k, v in model_metrics.items() if 'error' not in v}
    if not valid_models:
        logger.warning("No valid models for comparison.")
        return None
    
    # Detect if we have regression or classification models
    has_regression = any('r2' in v for v in valid_models.values())
    has_classification = any('accuracy' in v for v in valid_models.values())
    
    best_regression_ticker = None
    best_classification_ticker = None
    
    # Find best regression model
    if has_regression:
        # Normalize metrics to 0-1 scale
        regression_models = {k: v for k, v in valid_models.items() if 'r2' in v}
        
        # Calculate normalized scores
        for ticker in regression_models:
            metrics = regression_models[ticker]
            # R² is already in -1 to 1 range, rescale to 0-1
            r2_norm = (metrics['r2'] + 1) / 2
            # Direction accuracy is already in 0-1 range
            dir_acc_norm = metrics['direction_accuracy']
            # Normalize Sharpe ratio (clip to range [-2, 5] then rescale to 0-1)
            sharpe_norm = (min(5, max(-2, metrics['sharpe_ratio'])) + 2) / 7
            
            # Combined score with higher weight on direction accuracy and Sharpe ratio
            regression_models[ticker]['combined_score'] = (
                0.3 * r2_norm + 
                0.4 * dir_acc_norm + 
                0.3 * sharpe_norm
            )
        
        # Find best model by combined score
        best_regression_ticker = max(regression_models, key=lambda k: regression_models[k]['combined_score'])
        best_regression_score = regression_models[best_regression_ticker]['combined_score']
        logger.info(f"Best regression model: {results[best_regression_ticker]['company']} ({best_regression_ticker}) - combined score: {best_regression_score:.4f}")
        logger.info(f"  R²: {regression_models[best_regression_ticker]['r2']:.4f}, Direction accuracy: {regression_models[best_regression_ticker]['direction_accuracy']:.4f}")
        logger.info(f"  Sharpe ratio: {regression_models[best_regression_ticker]['sharpe_ratio']:.4f}")
    
    # Find best classification model
    if has_classification:
        # Normalize metrics to 0-1 scale
        classification_models = {k: v for k, v in valid_models.items() if 'accuracy' in v}
        
        # Calculate normalized scores
        for ticker in classification_models:
            metrics = classification_models[ticker]
            # All metrics already in 0-1 range
            
            # Combined score with higher weight on F1 and direction accuracy
            classification_models[ticker]['combined_score'] = (
                0.3 * metrics['accuracy'] + 
                0.4 * metrics['f1'] + 
                0.3 * metrics['direction_accuracy']
            )
        
        # Find best model by combined score
        best_classification_ticker = max(classification_models, key=lambda k: classification_models[k]['combined_score'])
        best_classification_score = classification_models[best_classification_ticker]['combined_score']
        logger.info(f"Best classification model: {results[best_classification_ticker]['company']} ({best_classification_ticker}) - combined score: {best_classification_score:.4f}")
        logger.info(f"  Accuracy: {classification_models[best_classification_ticker]['accuracy']:.4f}, F1: {classification_models[best_classification_ticker]['f1']:.4f}")
        logger.info(f"  Direction accuracy: {classification_models[best_classification_ticker]['direction_accuracy']:.4f}")
    
    # Decide between regression and classification
    if best_regression_ticker and best_classification_ticker:
        # If we have both types, prefer regression if it has good metrics
        best_r2 = regression_models[best_regression_ticker]['r2']
        if best_r2 > 0.1:  # Meaningful predictive power
            best_ticker = best_regression_ticker
            logger.info(f"Selected regression model as overall best model (R² = {best_r2:.4f})")
        else:
            # Otherwise use the model with best direction accuracy
            reg_direction = regression_models[best_regression_ticker]['direction_accuracy']
            cls_direction = classification_models[best_classification_ticker]['direction_accuracy']
            
            if reg_direction > cls_direction:
                best_ticker = best_regression_ticker
                logger.info(f"Selected regression model based on direction accuracy: {reg_direction:.4f} vs {cls_direction:.4f}")
            else:
                best_ticker = best_classification_ticker
                logger.info(f"Selected classification model based on direction accuracy: {cls_direction:.4f} vs {reg_direction:.4f}")
    elif best_regression_ticker:
        best_ticker = best_regression_ticker
    elif best_classification_ticker:
        best_ticker = best_classification_ticker
    else:
        logger.warning("No best model could be determined.")
        return None
    
    # Return the best model info
    best_model_info = {
        'ticker': best_ticker,
        'company': results[best_ticker]['company'],
        'model': results[best_ticker]['model'],
        'model_type': model_metrics[best_ticker]['model_type'],
        'features': model_metrics[best_ticker]['features'],
        'target': model_metrics[best_ticker]['target'],
        'metrics': model_metrics[best_ticker]
    }
    
    return best_model_info