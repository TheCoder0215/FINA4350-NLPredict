"""
FINA4350 Stock Sentiment Analysis Modeling module
Contains functions for model training and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor

def build_model(data):
    """Build a prediction model with robustness for small datasets."""
    print("Building prediction model...")
    
    # Extract features and target
    X = data[['Sentiment']]
    y = data['Next_Day_Return']
    
    # Determine if we have enough data for a meaningful train/test split
    if len(data) < 10:
        print("Warning: Very few data points. Using all data for training.")
        X_train, y_train = X, y
        X_test, y_test = X, y  # Use same data for testing as a fallback
        train_size = len(X)
        test_size = 0
    else:
        # Use a smaller test proportion for smaller datasets
        test_size = min(0.2, 5/len(data))  # At most 20%, but ensure at least 5 samples in training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        train_size = len(X_train)
        test_size = len(X_test)
    
    print(f"Training with {train_size} samples, testing with {test_size} samples")
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    
    if test_size > 0:
        test_score = model.score(X_test, y_test)
        print(f"Model R-squared (train): {train_score:.4f}, (test): {test_score:.4f}")
    else:
        print(f"Model R-squared (train): {train_score:.4f}")
    
    # Calculate and display the model coefficients
    print(f"Sentiment coefficient: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Interpretation: A 0.1 increase in sentiment score is associated with a {model.coef_[0] * 0.1:.4f}% change in next-day returns")
    
    return model

def advanced_model_training(data):
    """Train multiple models and select the best performer with proper handling of missing values"""
    # Print data quality information
    print("\nData quality check before modeling:")
    for col in data.columns:
        nan_count = data[col].isna().sum()
        if nan_count > 0:
            print(f"Column {col} has {nan_count} NaN values out of {len(data)} rows ({nan_count/len(data)*100:.1f}%)")
    
    # Make a clean copy for modeling
    data_clean = data.copy()
    
    # Drop rows with NaNs for better model stability
    data_no_nan = data_clean.dropna()
    print(f"Original data shape: {data_clean.shape}, After dropping NaNs: {data_no_nan.shape}")
    
    # If we've lost too much data, we'll use imputation instead
    if len(data_no_nan) < max(10, len(data_clean) * 0.5):
        print("Too many rows lost when dropping NaNs. Using imputation instead.")
        
        # Separate numeric and non-numeric columns
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create imputers
        mean_imputer = SimpleImputer(strategy='mean')
        
        # Apply imputation to numeric columns
        if numeric_cols:
            data_clean[numeric_cols] = mean_imputer.fit_transform(data_clean[numeric_cols])
    else:
        # Use the cleaned data without NaNs if we haven't lost too much
        data_clean = data_no_nan
    
    # Choose target variable (try different return horizons)
    target_options = [col for col in data_clean.columns if col.startswith('Next_')]
    
    # If no Next_* columns are available, create a basic one
    if not target_options:
        print("No target columns found. Creating a basic Next_Return target.")
        if 'Return' in data_clean.columns:
            data_clean['Next_Return'] = data_clean['Return'].shift(-1)
            data_clean = data_clean.dropna(subset=['Next_Return'])
            target_options = ['Next_Return']
        else:
            print("Error: No Return data available for prediction.")
            # Create a dummy target as last resort
            data_clean['Next_Return'] = np.random.normal(0, 1, len(data_clean))
            target_options = ['Next_Return']
    
    # Choose features to use
    feature_sets = []
    
    # Basic features - always include raw_sentiment
    if 'raw_sentiment' in data_clean.columns:
        feature_sets.append(['raw_sentiment'])
        
        # Add additional feature sets if columns exist
        ma_cols = [col for col in ['sentiment_ma3', 'sentiment_ma7'] if col in data_clean.columns]
        if ma_cols:
            feature_sets.append(['raw_sentiment'] + ma_cols)
            
        change_cols = [col for col in ['sentiment_change', 'sentiment_volatility'] if col in data_clean.columns]
        if change_cols:
            feature_sets.append(['raw_sentiment'] + change_cols)
            
        market_cols = [col for col in ['Market_Return'] if col in data_clean.columns]
        if ma_cols and market_cols:
            feature_sets.append(['raw_sentiment', ma_cols[0]] + market_cols)
    else:
        print("Error: raw_sentiment not available. Using all available features.")
        # Use all columns that aren't targets as features
        non_target_cols = [col for col in data_clean.columns if not col.startswith('Next_')]
        if non_target_cols:
            feature_sets.append(non_target_cols)
        else:
            print("Critical error: No features available for modeling.")
            # Return a dummy model
            dummy_model = LinearRegression()
            dummy_features = ['dummy']
            data_clean['dummy'] = np.random.normal(0, 1, len(data_clean))
            dummy_target = target_options[0] if target_options else 'Next_Return'
            if dummy_target not in data_clean.columns:
                data_clean[dummy_target] = np.random.normal(0, 1, len(data_clean))
            dummy_model.fit(data_clean[['dummy']], data_clean[dummy_target])
            return dummy_model, dummy_features, dummy_target
    
    # Prepare models to test
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        'SVR': Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=1.0, epsilon=0.1))]),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    best_score = -float('inf')
    best_score_threshold = -1.0  # Models need to be at least this good
    best_model = None
    best_model_name = None
    best_features = None
    best_target = None
    
    # Find the best combination
    for target in target_options:
        if target not in data_clean.columns:
            continue
            
        y = data_clean[target]
        
        for features in feature_sets:
            # Ensure all features exist in the dataframe
            valid_features = [f for f in features if f in data_clean.columns]
            if not valid_features:
                continue
                
            X = data_clean[valid_features]
            
            # Sanity check for data
            if len(X) < 5:
                print(f"Skipping {features} → {target} due to insufficient data")
                continue
                
            for model_name, model in models.items():
                try:
                    # Use cross-validation for more reliable scoring
                    cv_folds = min(5, len(X) - 1)  # Make sure we have enough samples
                    cv_folds = max(2, cv_folds)  # At least 2-fold CV
                    
                    # Use a more robust scoring metric (neg_mean_squared_error)
                    cv_scores = cross_val_score(
                        model, X, y, 
                        cv=cv_folds, 
                        scoring='neg_mean_squared_error',
                        error_score='raise'
                    )
                    avg_score = np.mean(cv_scores)
                    
                    print(f"Model: {model_name}, Features: {valid_features}, Target: {target}, Score: {avg_score:.4f}")
                    
                    # Keep track of best model configuration
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                        best_model_name = model_name
                        best_features = valid_features
                        best_target = target
                except Exception as e:
                    print(f"Error with {model_name} on {target}: {e}")
    
    # If no model performed well enough, create a fallback linear model
    if best_model is None or best_score < best_score_threshold:
        print("\nNo successful models found. Using fallback linear model.")
        
        # Use linear regression for fallback
        best_model = LinearRegression()
        best_model_name = 'LinearRegression'
        
        # Use the simplest feature set and target
        if feature_sets and target_options:
            best_features = feature_sets[0]
            best_target = target_options[0]
        else:
            print("Critical error: No valid features or targets available.")
            # Create a dummy model
            if 'raw_sentiment' not in data_clean.columns:
                data_clean['raw_sentiment'] = np.random.normal(0, 1, len(data_clean))
            if 'Next_Return' not in data_clean.columns:
                data_clean['Next_Return'] = np.random.normal(0, 1, len(data_clean))
            best_features = ['raw_sentiment']
            best_target = 'Next_Return'
    
    # Now fit the best model on the full dataset
    print(f"\nFitting final {best_model_name} model on {len(data_clean)} samples...")
    
    X_final = data_clean[best_features]
    y_final = data_clean[best_target]
    
    try:
        best_model.fit(X_final, y_final)
        
        # Check model fit
        train_score = best_model.score(X_final, y_final)
        print(f"Training R²: {train_score:.4f}")
        
        # Print feature importances if available
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            for feature, importance in zip(best_features, importances):
                print(f"Feature {feature}: importance = {importance:.4f}")
        elif best_model_name == 'LinearRegression' and hasattr(best_model, 'coef_'):
            coefs = best_model.coef_
            for feature, coef in zip(best_features, coefs):
                print(f"Feature {feature}: coefficient = {coef:.4f}")
    except Exception as e:
        print(f"Error fitting final model: {e}")
        # Create an extremely simple model as last resort
        best_model = DummyRegressor(strategy='mean')
        best_model.fit(X_final, y_final)
        best_model_name = 'DummyRegressor'
        print("Fallback to DummyRegressor (predicts mean value)")
    
    # Print summary information
    print(f"\nBest model configuration:")
    print(f"Model: {best_model_name}")
    print(f"Features: {best_features}")
    print(f"Target: {best_target}")
    print(f"Score: {best_score:.4f}")
    
    return best_model, best_features, best_target

def evaluate_models(results):
    """Calculate performance metrics for all models with robust error handling."""
    print("\n" + "="*70)
    print("EVALUATING MODEL PERFORMANCE ACROSS STOCKS")
    print("="*70)
    
    model_metrics = {}
    
    for ticker, info in results.items():
        print(f"\nEvaluating {ticker} ({info['company']})...")
        
        try:
            # Extract model, features, target and data
            model = info['model']
            features = info['features']
            target = info['target']
            data = info['data']
            
            # Check if features exist in data
            missing_features = [f for f in features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Features {missing_features} not found in data")
                
            # Get valid rows without NaN in features or target
            valid_X = data[features].dropna()
            if target in data.columns:
                valid_indices = valid_X.index.intersection(data[data[target].notna()].index)
                valid_X = valid_X.loc[valid_indices]
                valid_y = data.loc[valid_indices, target]
            else:
                raise ValueError(f"Target {target} not found in data")
            
            if len(valid_X) < 3:  # Need minimum samples for evaluation
                raise ValueError(f"Only {len(valid_X)} valid samples - insufficient for evaluation")
                
            # Get predictions and compute metrics
            y_pred = model.predict(valid_X)
            
            # Compute metrics with more robustness
            try:
                mse = mean_squared_error(valid_y, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(valid_y, y_pred)
            except Exception as metric_error:
                print(f"Error calculating metrics: {metric_error}")
                # Calculate metrics manually as fallback
                errors = valid_y - y_pred
                mse = np.mean(errors**2)
                rmse = np.sqrt(mse)
                ss_total = np.sum((valid_y - np.mean(valid_y))**2)
                ss_residual = np.sum(errors**2)
                r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            model_metrics[ticker] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'model_type': type(model).__name__,
                'num_features': len(features),
                'features': features,
                'target': target,
                'data_points': len(valid_X)
            }
            
            print(f"  Model: {type(model).__name__}")
            print(f"  Features: {features}")
            print(f"  Target: {target}")
            print(f"  Data points: {len(valid_X)}")
            print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        except Exception as e:
            print(f"  Error evaluating model: {e}")
            print(f"  Attempting fallback evaluation...")
            
            # Try a simplified evaluation if possible
            try:
                # Find any valid target and feature
                available_features = [col for col in data.columns if data[col].notna().sum() > 5 
                                     and not col.startswith('Next_')]
                available_targets = [col for col in data.columns if col.startswith('Next_') 
                                   and data[col].notna().sum() > 5]
                
                if available_features and available_targets:
                    simple_feature = available_features[0]
                    simple_target = available_targets[0]
                    
                    # Create a simple linear model as fallback
                    fallback_model = LinearRegression()
                    
                    # Get valid rows
                    mask = data[simple_feature].notna() & data[simple_target].notna()
                    X_simple = data.loc[mask, [simple_feature]]
                    y_simple = data.loc[mask, simple_target]
                    
                    if len(X_simple) >= 3:
                        fallback_model.fit(X_simple, y_simple)
                        y_pred = fallback_model.predict(X_simple)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_simple, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_simple, y_pred)
                        
                        print(f"  Fallback model using {simple_feature} → {simple_target}")
                        print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f} with {len(X_simple)} samples")
                        
                        model_metrics[ticker] = {
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2,
                            'model_type': 'FallbackLinearRegression',
                            'features': [simple_feature],
                            'target': simple_target,
                            'data_points': len(X_simple),
                            'is_fallback': True
                        }
                    else:
                        raise ValueError("Insufficient data even for fallback model")
                else:
                    raise ValueError("No valid features or targets for fallback model")
            except Exception as fallback_e:
                print(f"  Fallback evaluation also failed: {fallback_e}")
                model_metrics[ticker] = {
                    'error': str(e),
                    'model_type': type(model).__name__,
                    'features': features,
                    'target': target
                }
    
    return model_metrics

def find_best_model(results, model_metrics):
    """Identify the best model based on metrics."""
    print("\n" + "="*70)
    print("BEST MODEL IDENTIFICATION")
    print("="*70)
    
    # Filter out models with errors
    valid_models = {k: v for k, v in model_metrics.items() if 'error' not in v}
    
    if not valid_models:
        print("No valid models found for comparison.")
        return None
    
    # Determine the best model based on R² score
    best_ticker = max(valid_models, key=lambda k: valid_models[k]['r2'])
    
    print(f"Best model found for {results[best_ticker]['company']} ({best_ticker}):")
    print(f"  Model type: {valid_models[best_ticker]['model_type']}")
    print(f"  Features: {valid_models[best_ticker]['features']}")
    print(f"  Target: {valid_models[best_ticker]['target']}")
    print(f"  R² Score: {valid_models[best_ticker]['r2']:.4f}")
    
    return {
        'ticker': best_ticker,
        'company': results[best_ticker]['company'],
        'model': results[best_ticker]['model'],
        'model_type': valid_models[best_ticker]['model_type'],
        'features': valid_models[best_ticker]['features'],
        'target': valid_models[best_ticker]['target'],
        'metrics': valid_models[best_ticker]
    }