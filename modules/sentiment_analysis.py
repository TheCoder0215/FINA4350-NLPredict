"""
FINA4350 Stock Sentiment Analysis Sentiment analysis module
Enhanced with financial-specific lexicons and contextual sentiment analysis
"""

import pandas as pd
import numpy as np
import nltk
import torch
import logging
import re
import string
from datetime import datetime, timedelta
from dateutil import parser
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import config
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import textblob

# Setup logging
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Enhanced financial keywords with expanded vocabulary
POSITIVE_FINANCIAL_WORDS = set([
    'beat', 'exceeded', 'surpass', 'outperform', 'upgrade', 'rise', 'growth', 'profit', 
    'positive', 'strong', 'higher', 'bullish', 'opportunity', 'innovation', 'leadership',
    'overweight', 'buy', 'momentum', 'upside', 'breakthrough', 'succeed', 'potential',
    'raise', 'dividend', 'expansion', 'recovery', 'strength', 'gain', 'improved',
    'record', 'boost', 'advance', 'jump', 'accelerate', 'increasing', 'soar', 'rally',
    'progress', 'advantage', 'favorable', 'lucrative', 'optimistic', 'promising',
    'thrive', 'robust', 'prosper', 'excel', 'peak', 'expand', 'elevate'
])

NEGATIVE_FINANCIAL_WORDS = set([
    'miss', 'downgrade', 'underperform', 'decline', 'drop', 'fall', 'weak', 'loss',
    'lower', 'bearish', 'concern', 'risk', 'warning', 'below', 'disappointing', 'caution',
    'struggle', 'fail', 'downside', 'selloff', 'investigation', 'lawsuit', 'penalty',
    'underweight', 'sell', 'cut', 'bankruptcy', 'debt', 'litigation', 'recession',
    'headwind', 'slump', 'tumble', 'plunge', 'pressure', 'trouble', 'crisis', 'default',
    'layoff', 'downturn', 'slowdown', 'depreciate', 'negative', 'shrink', 'collapse',
    'underachieve', 'delay', 'adverse', 'plummet', 'deteriorate', 'worsen', 'deficit'
])

# Patterns that intensify financial sentiment
INTENSIFIER_PATTERNS = [
    r'significantly (higher|lower|better|worse)',
    r'(far|much|substantially) (above|below|better|worse)',
    r'(strong|weak) (growth|decline|performance)',
    r'(major|sharp|dramatic) (increase|decrease|rise|drop)',
    r'(beat|miss).*consensus',
    r'(exceed|disappoint).*expectation',
    r'guidance (raised|lowered)',
    r'rating (upgraded|downgraded)',
    r'record (high|low|revenue|profit)',
    r'highest|lowest.*(since|in history|ever)',
    r'(double|triple)-digit (growth|decline)',
    r'(surge|plummet).*(percent|sales|profit)',
    r'outperform.*market',
    r'worst|best.*(performance|quarter)'
]

class EnhancedFinancialSentimentAnalyzer:
    """Enhanced sentiment analyzer with financial domain adaptation"""
    
    def __init__(self, use_finbert=True):
        # Initialize VADER with custom financial lexicon
        self.vader = VaderSentimentIntensityAnalyzer()
        self.vader.lexicon.update({word: 3.5 for word in POSITIVE_FINANCIAL_WORDS})
        self.vader.lexicon.update({word: -3.5 for word in NEGATIVE_FINANCIAL_WORDS})
        
        # Flag for FinBERT availability
        self.finbert_available = False
        
        # Load FinBERT if requested
        if use_finbert:
            try:
                # Using the financial sentiment analysis model
                # Try to use FinBERT first, fall back to regular sentiment model if unavailable
                model_names = [
                    'ProsusAI/finbert', 
                    'yiyanghkust/finbert-tone',
                    'distilbert-base-uncased-finetuned-sst-2-english'
                ]
                
                for model_name in model_names:
                    try:
                        logger.info(f"Attempting to load {model_name}")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                        self.model.eval()
                        
                        # Move to GPU if available
                        if torch.cuda.is_available():
                            self.model = self.model.cuda()
                            logger.info(f"Using GPU for {model_name}")
                        
                        self.finbert_available = True
                        logger.info(f"Successfully loaded {model_name}")
                        
                        # Create sentiment pipeline
                        self.nlp = pipeline(
                            "sentiment-analysis", 
                            model=self.model, 
                            tokenizer=self.tokenizer,
                            device=0 if torch.cuda.is_available() else -1
                        )
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name}: {e}")
                        continue
                        
                if not self.finbert_available:
                    logger.warning("All models failed to load. Falling back to VADER only.")
            except Exception as e:
                logger.warning(f"Error setting up transformer models: {e}")
        
        # Initialize TextBlob
        self.textblob_analyzer = textblob.TextBlob
        
        # Compile regex patterns for efficiency
        self.intensifier_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in INTENSIFIER_PATTERNS]
        
        # Stop words for cleaning
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters but keep sentence structure and dollar signs
        text = re.sub(r'[^\w\s.,!?$%]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def detect_financial_intensity(self, text):
        """Detect patterns that intensify financial sentiment"""
        intensity_score = 0
        
        for pattern in self.intensifier_patterns:
            matches = pattern.findall(text.lower())
            intensity_score += len(matches) * 0.5  # Add 0.5 for each match
            
        return intensity_score
    
    def analyze(self, text):
        """Multi-model sentiment analysis with financial context adjustment"""
        if not isinstance(text, str) or not text.strip():
            return 0.0
            
        text = self.clean_text(text)
        if not text:
            return 0.0
            
        scores = []
        weights = []
        
        # 1. VADER with financial lexicon
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        scores.append(vader_compound)
        weights.append(1.0)  # Base weight for VADER
        
        # 2. FinBERT if available
        if self.finbert_available:
            try:
                # Use the sentiment pipeline
                result = self.nlp(text[:512])  # Truncate to max length
                
                # Map sentiment label to score
                if hasattr(result[0], 'label') and hasattr(result[0], 'score'):
                    label = result[0]['label'].lower()
                    score = result[0]['score']
                    
                    # Convert label to numeric sentiment score
                    if 'positive' in label:
                        finbert_score = score
                    elif 'negative' in label:
                        finbert_score = -score
                    else:  # neutral
                        finbert_score = 0.0
                        
                    scores.append(finbert_score)
                    weights.append(1.5)  # Higher weight for domain-specific model
            except Exception as e:
                logger.warning(f"FinBERT inference failed: {str(e)[:100]}...")
        
        # 3. TextBlob sentiment
        try:
            textblob_score = self.textblob_analyzer(text).sentiment.polarity
            scores.append(textblob_score)
            weights.append(0.8)  # Lower weight for general model
        except Exception as e:
            logger.debug(f"TextBlob analysis failed: {str(e)[:100]}...")
        
        # 4. Calculate custom financial features
        # Count positive and negative financial words
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        pos_count = len(words.intersection(POSITIVE_FINANCIAL_WORDS))
        neg_count = len(words.intersection(NEGATIVE_FINANCIAL_WORDS))
        
        financial_words_score = 0
        if pos_count + neg_count > 0:
            financial_words_score = (pos_count - neg_count) / (pos_count + neg_count)
            scores.append(financial_words_score)
            weights.append(1.2)  # Higher weight for financial lexicon
            
        # 5. Apply financial intensity multiplier
        intensity = self.detect_financial_intensity(text)
        
        # No scores collected? Return neutral
        if not scores:
            return 0.0
            
        # Weighted average of all scores
        if sum(weights) > 0:
            final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            final_score = sum(scores) / len(scores)
        
        # Apply intensity multiplier (increases magnitude but preserves sign)
        if intensity > 0:
            final_score = final_score * (1 + min(intensity, 1.0))
            
        return max(min(final_score, 1.0), -1.0)  # Clamp between -1 and 1

# Initialize enhanced analyzer
enhanced_analyzer = EnhancedFinancialSentimentAnalyzer(use_finbert=True)

def extract_numerical_features(text):
    """Extract numerical values that might indicate financial performance"""
    if not isinstance(text, str):
        return {}
        
    features = {}
    
    # Extract percentages
    percentages = re.findall(r'(\-?\d+\.?\d*)%', text)
    if percentages:
        features['pct_mentioned'] = 1
        features['max_pct'] = max([abs(float(p)) for p in percentages])
    else:
        features['pct_mentioned'] = 0
        features['max_pct'] = 0
        
    # Extract dollar amounts
    dollar_amounts = re.findall(r'\$(\d+(?:\.\d+)?)(?: ?(?:million|billion|trillion|m|b|t|M|B|T))?', text)
    if dollar_amounts:
        features['dollar_mentioned'] = 1
    else:
        features['dollar_mentioned'] = 0
        
    # Look for financial timeframes
    if re.search(r'(quarter|annual|fiscal|year)', text, re.IGNORECASE):
        features['timeframe_mentioned'] = 1
    else:
        features['timeframe_mentioned'] = 0
        
    return features

def analyze_sentiment(news_df):
    """
    Analyze sentiment of news headlines using advanced financial methods.
    Returns daily aggregated sentiment with multiple features.
    """
    logging.info(f"Analyzing sentiment for {len(news_df)} news items with enhanced financial context")
    
    if news_df.empty:
        logging.warning("Empty news dataframe provided")
        # Return a minimal dataframe with today's date as index
        return pd.DataFrame({'raw_sentiment': [0]}, index=[pd.Timestamp.now()])
    
    # 1. Parse dates properly
    if 'date' in news_df.columns:
        try:
            news_df['date_parsed'] = pd.to_datetime(news_df['date'], errors='coerce')
            # Fill missing dates with a reasonable fallback
            news_df.loc[news_df['date_parsed'].isna(), 'date_parsed'] = datetime.now()
        except Exception as e:
            logging.error(f"Error parsing dates: {e}")
            news_df['date_parsed'] = datetime.now()
    else:
        news_df['date_parsed'] = datetime.now()
    
    # 2. Apply enhanced sentiment analysis
    if 'title' in news_df.columns:
        # Enhanced sentiment score
        try:
            # Process in batches to avoid memory issues with large news sets
            batch_size = 100
            sentiment_results = []
            
            for i in range(0, len(news_df), batch_size):
                batch = news_df['title'].iloc[i:i+batch_size]
                batch_sentiments = batch.apply(lambda x: enhanced_analyzer.analyze(str(x)))
                sentiment_results.extend(batch_sentiments)
                
            news_df['enhanced_sentiment'] = sentiment_results
            
            # Fill NAs with neutral sentiment
            news_df['enhanced_sentiment'] = news_df['enhanced_sentiment'].fillna(0)
            
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
            news_df['enhanced_sentiment'] = 0
        
        # Extract financial numerical features
        try:
            numerical_features = news_df['title'].apply(extract_numerical_features)
            for feat in ['pct_mentioned', 'max_pct', 'dollar_mentioned', 'timeframe_mentioned']:
                news_df[feat] = numerical_features.apply(lambda x: x.get(feat, 0))
        except Exception as e:
            logging.error(f"Error extracting numerical features: {e}")
            # Add empty columns
            for feat in ['pct_mentioned', 'max_pct', 'dollar_mentioned', 'timeframe_mentioned']:
                news_df[feat] = 0
                
        # Topic classification
        news_df['has_earnings'] = news_df['title'].str.lower().str.contains(
            'earnings|revenue|profit|eps|income|quarter|results', regex=True).astype(int)
        news_df['has_ratings'] = news_df['title'].str.lower().str.contains(
            'upgrade|downgrade|initiate|coverage|rating|price target', regex=True).astype(int)
        news_df['has_products'] = news_df['title'].str.lower().str.contains(
            'launch|product|announce|unveil|release|new', regex=True).astype(int)
        news_df['has_management'] = news_df['title'].str.lower().str.contains(
            'ceo|executive|management|appoint|hire|resign', regex=True).astype(int)
    
    # 3. Aggregate to daily level with multiple metrics
    agg_dict = {
        'enhanced_sentiment': ['mean', 'median', 'min', 'max', 'count', 'std'],
        'pct_mentioned': 'sum',
        'max_pct': ['min', 'max', 'mean'],
        'dollar_mentioned': 'sum',
        'timeframe_mentioned': 'sum',
        'has_earnings': 'sum',
        'has_ratings': 'sum',
        'has_products': 'sum',
        'has_management': 'sum'
    }
    
    try:
        # Group by date and aggregate
        daily_sentiment = news_df.groupby('date_parsed').agg(agg_dict)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
        
        # Rename main sentiment column for backward compatibility
        daily_sentiment.rename(columns={'enhanced_sentiment_mean': 'raw_sentiment'}, inplace=True)
        
        # Add headline volume metrics
        daily_sentiment['news_volume'] = daily_sentiment['enhanced_sentiment_count']
        
        # Calculate sentiment volatility
        daily_sentiment['sentiment_volatility'] = daily_sentiment['enhanced_sentiment_std']
        
        # Calculate sentiment consensus (how unanimous is the sentiment)
        daily_sentiment['sentiment_consensus'] = 1.0 - (
            (daily_sentiment['enhanced_sentiment_max'] - daily_sentiment['enhanced_sentiment_min']) / 2.0
        ).fillna(0)
        
        # Calculate sentiment dispersion (another measure of consensus)
        daily_sentiment['sentiment_dispersion'] = daily_sentiment['enhanced_sentiment_std'] / (
            daily_sentiment['enhanced_sentiment_count'].apply(lambda x: max(x, 1))
        )
        
        # 4. Smooth and fill sentiment time series
        # Use last 6 months of trading days as index
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Reindex to include all trading days
        daily_sentiment = daily_sentiment.reindex(trading_days)
        
        # Forward fill with decay - more robust method
        for col in daily_sentiment.columns:
            # First forward fill values up to 5 days
            daily_sentiment[col] = daily_sentiment[col].fillna(method='ffill', limit=5)
            
        # Add exponential decay for sentiment longer than 5 days old
        for col in ['raw_sentiment', 'enhanced_sentiment_median', 'enhanced_sentiment_min', 
                   'enhanced_sentiment_max', 'sentiment_volatility']:
            if col in daily_sentiment.columns:
                mask = daily_sentiment[col].isna()
                if mask.sum() > 0:
                    # Find all non-NA values
                    valid_values = daily_sentiment.loc[~mask, col]
                    if len(valid_values) > 0:
                        last_valid = valid_values.iloc[-1]
                        # Apply exponential decay
                        for i, (idx, row) in enumerate(daily_sentiment.loc[mask].iterrows()):
                            days_gap = min(i+1, 30)  # Cap at 30 days
                            decay_factor = np.exp(-0.1 * days_gap)
                            daily_sentiment.loc[idx, col] = last_valid * decay_factor
        
        # Fill any remaining NAs with appropriate values
        fill_values = {
            'raw_sentiment': 0,
            'enhanced_sentiment_median': 0, 
            'enhanced_sentiment_min': 0,
            'enhanced_sentiment_max': 0,
            'enhanced_sentiment_count': 0,
            'enhanced_sentiment_std': 0,
            'pct_mentioned_sum': 0, 
            'max_pct_min': 0,
            'max_pct_max': 0, 
            'max_pct_mean': 0,
            'dollar_mentioned_sum': 0,
            'timeframe_mentioned_sum': 0,
            'has_earnings_sum': 0,
            'has_ratings_sum': 0,
            'has_products_sum': 0,
            'has_management_sum': 0,
            'news_volume': 0,
            'sentiment_volatility': 0,
            'sentiment_consensus': 0.5,
            'sentiment_dispersion': 0
        }
        
        for col, fill_value in fill_values.items():
            if col in daily_sentiment.columns:
                daily_sentiment[col] = daily_sentiment[col].fillna(fill_value)
        
        # Generate rolling statistics for sentiment
        if 'raw_sentiment' in daily_sentiment.columns:
            # Moving averages
            for window in [3, 7, 14, 30]:
                daily_sentiment[f'raw_sentiment_ma{window}'] = (
                    daily_sentiment['raw_sentiment'].rolling(window=window).mean().fillna(0)
                )
                
                # Standard deviation (volatility)
                daily_sentiment[f'raw_sentiment_std{window}'] = (
                    daily_sentiment['raw_sentiment'].rolling(window=window).std().fillna(0)
                )
                
                # Min/Max
                daily_sentiment[f'raw_sentiment_min{window}'] = (
                    daily_sentiment['raw_sentiment'].rolling(window=window).min().fillna(0)
                )
                daily_sentiment[f'raw_sentiment_max{window}'] = (
                    daily_sentiment['raw_sentiment'].rolling(window=window).max().fillna(0)
                )
                
                # Range
                daily_sentiment[f'raw_sentiment_range{window}'] = (
                    daily_sentiment[f'raw_sentiment_max{window}'] - 
                    daily_sentiment[f'raw_sentiment_min{window}']
                )
                
                # Z-score
                daily_sentiment[f'raw_sentiment_zscore{window}'] = (
                    (daily_sentiment['raw_sentiment'] - daily_sentiment[f'raw_sentiment_ma{window}']) / 
                    daily_sentiment[f'raw_sentiment_std{window}'].replace(0, 1)
                ).fillna(0)
            
            # Generate lagged sentiment features
            for lag in [1, 2, 3, 5, 7]:
                daily_sentiment[f'raw_sentiment_lag{lag}'] = daily_sentiment['raw_sentiment'].shift(lag).fillna(0)
            
            # Calculate changes
            daily_sentiment['sentiment_change'] = daily_sentiment['raw_sentiment'].diff().fillna(0)
            daily_sentiment['sentiment_pct_change'] = daily_sentiment['raw_sentiment'].pct_change().fillna(0)
            
            # Momentum
            daily_sentiment['sentiment_momentum_3d'] = daily_sentiment['raw_sentiment'].diff(3).fillna(0)
            daily_sentiment['sentiment_momentum_7d'] = daily_sentiment['raw_sentiment'].diff(7).fillna(0)
        
        logging.info(f"Sentiment analysis complete: {len(daily_sentiment)} days with {len(daily_sentiment.columns)} metrics")
        return daily_sentiment
        
    except Exception as e:
        logging.error(f"Sentiment aggregation failed: {e}")
        # Return minimal dataframe with raw sentiment to maintain compatibility
        minimal_df = pd.DataFrame({'raw_sentiment': [0]}, index=[pd.Timestamp.now()])
        return minimal_df

def combine_data(sentiment_df, stock_df):
    """Combine sentiment and stock price data with robust alignment."""
    logging.info(f"Combining sentiment data ({len(sentiment_df)} records) with stock data ({len(stock_df)} records)")
    
    # Convert sentiment index to datetime
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    
    # Ensure stock index is datetime 
    stock_df.index = pd.to_datetime(stock_df.index)
    
    # Get all unique dates from both dataframes
    all_dates = sorted(set(sentiment_df.index).union(set(stock_df.index)))
    
    # Create a new DataFrame with the combined index
    combined_df = pd.DataFrame(index=all_dates)
    
    # Add sentiment data
    for col in sentiment_df.columns:
        combined_df[col] = sentiment_df.loc[sentiment_df.index.intersection(all_dates), col]
    
    # Forward fill sentiment data (with limit)
    sentiment_cols = list(sentiment_df.columns)
    combined_df[sentiment_cols] = combined_df[sentiment_cols].fillna(method='ffill', limit=5)
    
    # Add stock price data
    for col in stock_df.columns:
        combined_df[col] = stock_df.loc[stock_df.index.intersection(all_dates), col]
    
    # Filter to only keep dates where we have stock data
    combined_df = combined_df.loc[combined_df.index.intersection(stock_df.index)]
    
    # Generate target variables with different horizons
    if 'Return_1d' in stock_df.columns:
        for horizon in [1, 3, 5, 10]:
            col = f'Return_{horizon}d'
            if col in stock_df.columns:
                # Use stock_df to get returns for forecast horizons
                target_values = {}
                for i, date in enumerate(combined_df.index):
                    if date in stock_df.index:
                        idx = stock_df.index.get_loc(date)
                        if idx + horizon < len(stock_df.index):
                            target_date = stock_df.index[idx + horizon]
                            if col in stock_df.columns and not pd.isna(stock_df.loc[target_date, col]):
                                target_values[date] = stock_df.loc[target_date, col]
                
                combined_df[f'Next_{col}'] = pd.Series(target_values)
    
    # Fill any missing values in sentiment columns
    for col in sentiment_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna(0)  # Fill with neutral sentiment
    
    logging.info(f"Combined data has {len(combined_df)} records after alignment")
    return combined_df
