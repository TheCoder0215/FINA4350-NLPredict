"""
FINA4350 Stock Sentiment Analysis
Configuration settings for stock sentiment analysis
"""

# Data collection
NEWS_SOURCES = {
    'yahoo': True,
    'finviz': True,
    'marketwatch': True,
    'seekingalpha': True,
}

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Sentiment analysis
POSITIVE_WORDS = [
    'up', 'gain', 'grow', 'rise', 'positive', 'profit', 'success', 
    'increase', 'strong', 'higher', 'bullish', 'beat', 'exceed', 'upgrade'
]

NEGATIVE_WORDS = [
    'down', 'loss', 'drop', 'fall', 'negative', 'decline', 'weak', 
    'lower', 'bearish', 'concern', 'risk', 'miss', 'downgrade', 'below'
]

# Feature engineering
MIN_VALID_RATIO = 0.3  # Minimum required valid values in a column

# Model training
MODEL_CONFIGS = {
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_leaf': 5,
        'random_state': 42
    },
    'GradientBoosting': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'SVR': {
        'C': 1.0,
        'epsilon': 0.1
    },
    'HistGradientBoosting': {
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42
    }
}

# Visualization
VISUALIZATION_SETTINGS = {
    'dpi': 300,
    'figsize_large': (15, 15),
    'figsize_medium': (12, 8),
    'cmap': 'coolwarm',
    'line_color': 'skyblue',
    'highlight_color': 'gold',
    'grid_alpha': 0.7
}

# Default stocks to analyze
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_COMPANIES = ["Apple", "Microsoft", "Google", "Amazon", "Meta"]