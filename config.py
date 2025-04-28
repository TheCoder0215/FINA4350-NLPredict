"""
FINA4350 Stock Sentiment Analysis
Configuration settings for stock sentiment analysis
"""

# News API Key (if available)
NEWS_API_KEY = ""  # Add your key here

# Alpha Vantage API Key (if available)
ALPHA_VANTAGE_API_KEY = ""  # Add your key here

# Data collection
NEWS_SOURCES = {
    'yahoo': True,
    'finviz': True,
    'marketwatch': True,
    'seekingalpha': True,
    'reuters': True,
    'cnbc': True,
    'bloomberg': False,
}

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Sentiment analysis
POSITIVE_WORDS = [
    'up', 'gain', 'grow', 'rise', 'positive', 'profit', 'success', 
    'increase', 'strong', 'higher', 'bullish', 'beat', 'exceed', 'upgrade',
    'opportunity', 'promising', 'outperform', 'robust', 'innovative',
    'breakthrough', 'surpass', 'leadership', 'momentum', 'confident'
]

NEGATIVE_WORDS = [
    'down', 'loss', 'drop', 'fall', 'negative', 'decline', 'weak', 
    'lower', 'bearish', 'concern', 'risk', 'miss', 'downgrade', 'below',
    'disappoint', 'warning', 'caution', 'struggle', 'underperform', 'fail',
    'plunge', 'tumble', 'slump', 'investigation', 'lawsuit', 'penalty'
]

# Feature engineering
MIN_VALID_RATIO = 0.3  # Minimum required valid values in a column

# Model training
MODEL_CONFIGS = {
    'RandomForest': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 5,
        'random_state': 42
    },
    'GradientBoosting': {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'random_state': 42
    },
    'SVR': {
        'C': 1.0,
        'epsilon': 0.1,
        'kernel': 'rbf'
    },
    'HistGradientBoosting': {
        'max_depth': 10,
        'learning_rate': 0.05,
        'max_leaf_nodes': 31,
        'random_state': 42
    }
}

# Enhanced visualization settings
VISUALIZATION_SETTINGS = {
    'dpi': 300,
    'figsize_large': (15, 15),
    'figsize_medium': (12, 8),
    'cmap': 'coolwarm',
    'line_color': 'skyblue',
    'highlight_color': 'gold',
    'grid_alpha': 0.7,
    'interactive': True  # Whether to generate interactive visualizations
}

# Report generation settings
REPORT_SETTINGS = {
    'generate_html': True,
    'include_summary': True,
    'include_images': True,
    'detailed_metrics': True
}

# Additional technical indicators to include
TECHNICAL_INDICATORS = {
    'moving_averages': [5, 10, 20, 50, 200],
    'rsi': True,
    'macd': True,
    'bollinger': True,
    'fibonacci': False,
    'ichimoku': False
}

# Default stocks to analyze
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_COMPANIES = ["Apple", "Microsoft", "Google", "Amazon", "Meta"]

# Time periods
DEFAULT_DAYS = 365  # How many days of data to analyze