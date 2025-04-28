"""
Data collection module for stock sentiment analysis
Enhanced with robust data sources, error handling, and API integrations
"""

import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import config
import logging
import re
import json
import os
import random
from urllib.parse import quote_plus
import ssl
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Set up logging
logger = logging.getLogger(__name__)

# Enhanced user agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

def get_random_user_agent():
    """Return a random user agent from the list"""
    return random.choice(USER_AGENTS)

def get_enhanced_headers():
    """Get enhanced headers that look more like a real browser"""
    return {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'DNT': '1',
        'Pragma': 'no-cache',
    }

def create_requests_session():
    """Create a requests session with retries and proper settings"""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=1,  # Time factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"]
    )
    
    # Apply retry strategy to session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def fetch_url(url, headers=None, timeout=15, session=None):
    """Fetch URL with retry logic and proper error handling"""
    if headers is None:
        headers = get_enhanced_headers()
    
    if session is None:
        session = create_requests_session()
    
    try:
        response = session.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.text
        else:
            logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None

def parse_date(date_text):
    """Parse various date formats into a standard YYYY-MM-DD format"""
    now = datetime.now()
    
    try:
        # Handle relative dates
        if isinstance(date_text, str) and 'ago' in date_text.lower():
            if 'min' in date_text.lower():
                # Minutes ago
                minutes = int(re.search(r'(\d+)', date_text).group(1))
                return (now - timedelta(minutes=minutes)).strftime('%Y-%m-%d')
            elif 'hour' in date_text.lower():
                # Hours ago
                hours = int(re.search(r'(\d+)', date_text).group(1))
                return (now - timedelta(hours=hours)).strftime('%Y-%m-%d')
            elif 'day' in date_text.lower():
                # Days ago
                days = int(re.search(r'(\d+)', date_text).group(1))
                return (now - timedelta(days=days)).strftime('%Y-%m-%d')
            else:
                return now.strftime('%Y-%m-%d')
        
        # Try standard date parsing
        parsed_date = pd.to_datetime(date_text, errors='coerce')
        if pd.isna(parsed_date):
            return now.strftime('%Y-%m-%d')
        return parsed_date.strftime('%Y-%m-%d')
    except:
        # Default to today's date
        return now.strftime('%Y-%m-%d')

def extract_newsapi_results(api_key, ticker, company_name, days=30):
    """Get news from NewsAPI.org if API key is provided"""
    if not api_key:
        return []
    
    logger.info(f"Fetching news from NewsAPI for {company_name} ({ticker})")
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create query terms (company name OR ticker)
    query = f"({quote_plus(company_name)}) OR {ticker}"
    
    # Endpoint URL
    url = "https://newsapi.org/v2/everything"
    
    # Parameters
    params = {
        'q': query,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] != 'ok':
                logger.warning(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            logger.info(f"Received {len(articles)} articles from NewsAPI")
            
            news_items = []
            for article in articles:
                published_at = article.get('publishedAt', '')
                title = article.get('title', '')
                url = article.get('url', '')
                source = article.get('source', {}).get('name', 'NewsAPI')
                
                # Skip if no title
                if not title:
                    continue
                
                # Parse date
                date = parse_date(published_at)
                
                news_items.append({
                    'date': date,
                    'title': title,
                    'source': source,
                    'url': url
                })
            
            return news_items
        else:
            logger.warning(f"NewsAPI request failed: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching from NewsAPI: {e}")
        return []

def extract_alpha_vantage_news(api_key, ticker, days=30):
    """Get news from Alpha Vantage News API if API key is provided"""
    if not api_key:
        return []
    
    logger.info(f"Fetching news from Alpha Vantage for {ticker}")
    
    # Endpoint URL
    url = "https://www.alphavantage.co/query"
    
    # Parameters
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'time_from': (datetime.now() - timedelta(days=days)).strftime('%Y%m%dT0000'),
        'limit': 200,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            
            if 'feed' not in data:
                logger.warning(f"Alpha Vantage error: No feed data returned")
                return []
            
            articles = data.get('feed', [])
            logger.info(f"Received {len(articles)} articles from Alpha Vantage")
            
            news_items = []
            for article in articles:
                published_at = article.get('time_published', '')
                title = article.get('title', '')
                url = article.get('url', '')
                source = article.get('source', 'Alpha Vantage')
                
                # Skip if no title
                if not title:
                    continue
                
                # Parse date (format: YYYYMMDDTHHMM)
                try:
                    date = datetime.strptime(published_at, '%Y%m%dT%H%M').strftime('%Y-%m-%d')
                except:
                    date = datetime.now().strftime('%Y-%m-%d')
                
                news_items.append({
                    'date': date,
                    'title': title,
                    'source': source,
                    'url': url
                })
            
            return news_items
        else:
            logger.warning(f"Alpha Vantage request failed: HTTP {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching from Alpha Vantage: {e}")
        return []

def get_news_from_yahoo_finance(ticker, max_articles=40):
    """Get news from Yahoo Finance with robust parsing"""
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    logger.info(f"Fetching news from Yahoo Finance: {url}")
    
    html = fetch_url(url)
    if not html:
        return []
        
    soup = BeautifulSoup(html, 'html.parser')
    news_items = []
    
    # Various possible selectors for Yahoo's changing layout
    selectors = [
        'li.js-stream-content',
        'div.Ov\(h\)',
        'div[data-test="CARD"]',
        'div.caas-card',
        'li[data-uuid]',
        'div.StreamMegaItem, div.js-stream-content'
    ]
    
    articles = []
    for selector in selectors:
        articles = soup.select(selector)
        if articles:
            break
            
    # Fallback to any div with title/link
    if not articles:
        articles = [div for div in soup.find_all('div') if div.find('h3') and div.find('a')]
    
    logger.info(f"Found {len(articles)} articles on Yahoo Finance")
    
    for article in articles[:max_articles]:
        try:
            # Extract title
            title_element = None
            for tag in ['h3', 'h4', 'h2']:
                title_element = article.select_one(f'{tag}, a > {tag}, div > {tag}')
                if title_element:
                    break
            
            # Fallback to any prominent link text
            if not title_element:
                links = article.select('a')
                for link in links:
                    if link.text and len(link.text.strip()) > 15:
                        title_element = link
                        break
            
            # Extract date
            date_element = None
            date_selectors = [
                'span[class="C(#959595)"]', 
                'span[class="C(#5b5b5b)"]', 
                'span.bl-5', 
                'time'
            ]
            
            for selector in date_selectors:
                date_element = article.select_one(selector)
                if date_element:
                    break
                    
            # Fallback to any span containing time-related text
            if not date_element:
                spans = article.select('span')
                for span in spans:
                    text = span.text.lower()
                    if any(time_word in text for time_word in ['ago', 'min', 'hour', 'day', 'am', 'pm', 'gmt']):
                        date_element = span
                        break
            
            # Extract URL
            link_element = article.find('a', href=True)
            url = link_element['href'] if link_element else ""
            
            # Only add if we have a title
            if title_element:
                title = title_element.get_text(strip=True)
                date_text = date_element.get_text(strip=True) if date_element else datetime.now().strftime('%Y-%m-%d')
                
                # Parse date
                date = parse_date(date_text)
                
                news_items.append({
                    'date': date,
                    'title': title,
                    'source': 'Yahoo Finance',
                    'url': url
                })
        except Exception as e:
            logger.error(f"Error processing Yahoo article: {e}")
    
    return news_items

def get_news_from_finviz(ticker, max_articles=40):
    """Get news from Finviz with robust parsing"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    logger.info(f"Fetching news from Finviz: {url}")
    
    html = fetch_url(url)
    if not html:
        return []
        
    soup = BeautifulSoup(html, 'html.parser')
    news_items = []
    
    # Find the news table
    news_table = soup.find('table', class_='news-table')
    if not news_table:
        return []
        
    rows = news_table.find_all('tr')
    logger.info(f"Found {len(rows)} news rows on Finviz")
    
    current_date = None
    for row in rows[:max_articles]:
        try:
            date_td = row.find('td', align="right")
            title_td = row.find('td', align="left")
            
            if date_td and title_td:
                date_text = date_td.get_text(strip=True)
                
                # Handle date format
                if len(date_text) > 5 and ('-' in date_text or '/' in date_text):
                    # It's a full date like "Nov-24-20"
                    current_date = date_text
                    
                title_element = title_td.find('a')
                
                if title_element:
                    title = title_element.get_text(strip=True)
                    url = title_element.get('href', '')
                    
                    # Use stored date or time
                    date = current_date if current_date else datetime.now().strftime('%Y-%m-%d')
                    
                    # Parse date
                    parsed_date = parse_date(date)
                    
                    news_items.append({
                        'date': parsed_date,
                        'title': title,
                        'source': 'Finviz',
                        'url': url
                    })
        except Exception as e:
            logger.error(f"Error processing Finviz news row: {e}")
    
    return news_items

def get_news_from_wsj(ticker, max_articles=20):
    """Get news from Wall Street Journal Search"""
    url = f"https://www.wsj.com/search?query={ticker}"
    logger.info(f"Fetching news from WSJ: {url}")
    
    html = fetch_url(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    news_items = []
    
    # Find article listings
    articles = soup.select('article, .WSJTheme--story--XB4V2mLz')
    logger.info(f"Found {len(articles)} articles on WSJ")
    
    for article in articles[:max_articles]:
        try:
            # Find headline
            headline = article.select_one('h2, h3, .WSJTheme--headline--unZqjb45')
            if not headline:
                continue
                
            title = headline.get_text(strip=True)
            
            # Find date
            date_element = article.select_one('time, .WSJTheme--timestamp--22j2KzIE')
            date_text = date_element.get_text(strip=True) if date_element else datetime.now().strftime('%Y-%m-%d')
            date = parse_date(date_text)
            
            # Find URL
            link = headline.find_parent('a', href=True)
            url = link['href'] if link else ""
            
            news_items.append({
                'date': date,
                'title': title,
                'source': 'Wall Street Journal',
                'url': url
            })
        except Exception as e:
            logger.error(f"Error processing WSJ article: {e}")
    
    return news_items

def get_news_from_marketwatch(ticker, max_articles=20):
    """Get news from MarketWatch"""
    url = f"https://www.marketwatch.com/investing/stock/{ticker}"
    logger.info(f"Fetching news from MarketWatch: {url}")
    
    html = fetch_url(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    news_items = []
    
    # Find article listings - try multiple selectors
    selectors = [
        'div.article__content', 
        'div.element--article',
        'div.collection__elements > div',
        'div.story__body'
    ]
    
    articles = []
    for selector in selectors:
        articles = soup.select(selector)
        if articles:
            break
    
    # Fallback
    if not articles:
        articles = [div for div in soup.find_all('div') if div.find('h3') and div.find('a')]
    
    logger.info(f"Found {len(articles)} articles on MarketWatch")
    
    for article in articles[:max_articles]:
        try:
            # Find headline
            headline = article.select_one('h3.article__headline, h3.headline, a.headline, h3, h2')
            if not headline:
                continue
                
            title = headline.get_text(strip=True)
            if not title and headline.has_attr('title'):
                title = headline['title']
            
            # Find date
            date_element = article.select_one('span.article__timestamp, span.timestamp, time')
            date_text = date_element.get_text(strip=True) if date_element else datetime.now().strftime('%Y-%m-%d')
            date = parse_date(date_text)
            
            # Find URL
            if headline.name == 'a' and headline.has_attr('href'):
                url = headline['href']
            else:
                link = headline.find_parent('a', href=True) or article.find('a', href=True)
                url = link['href'] if link else ""
            
            news_items.append({
                'date': date,
                'title': title,
                'source': 'MarketWatch',
                'url': url
            })
        except Exception as e:
            logger.error(f"Error processing MarketWatch article: {e}")
    
    return news_items

def get_news_from_reuters(company_name, max_articles=20):
    """Get news from Reuters"""
    url = f"https://www.reuters.com/search/news?blob={quote_plus(company_name)}"
    logger.info(f"Fetching news from Reuters: {url}")
    
    html = fetch_url(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    news_items = []
    
    # Find article listings - try multiple selectors
    selectors = [
        'div.search-result-content',
        'div.media-story-card',
        'article.story-card'
    ]
    
    articles = []
    for selector in selectors:
        articles = soup.select(selector)
        if articles:
            break
    
    logger.info(f"Found {len(articles)} articles on Reuters")
    
    for article in articles[:max_articles]:
        try:
            # Find headline
            headline = article.select_one('h3.search-result-title, h3.media-story-card__heading__eqhp9, h3, a[data-testid="Heading"]')
            if not headline:
                continue
                
            title = headline.get_text(strip=True)
            
            # Find date
            date_element = article.select_one('span.timestamp, time, span[data-testid="published-timestamp"]')
            date_text = date_element.get_text(strip=True) if date_element else datetime.now().strftime('%Y-%m-%d')
            date = parse_date(date_text)
            
            # Find URL
            if headline.name == 'a' and headline.has_attr('href'):
                url = headline['href']
            else:
                link = article.find('a', href=True)
                url = link['href'] if link else ""
                
            # Make URL absolute if needed
            if url and url.startswith('/'):
                url = f"https://www.reuters.com{url}"
            
            news_items.append({
                'date': date,
                'title': title,
                'source': 'Reuters',
                'url': url
            })
        except Exception as e:
            logger.error(f"Error processing Reuters article: {e}")
    
    return news_items

def get_news_from_bloomberg(company_name, max_articles=15):
    """Get news from Bloomberg (note: might have limited success due to paywall)"""
    url = f"https://www.bloomberg.com/search?query={quote_plus(company_name)}"
    logger.info(f"Fetching news from Bloomberg: {url}")
    
    html = fetch_url(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    news_items = []
    
    # Find article listings - try multiple selectors
    selectors = [
        'div.search-result-story',
        'article.story-package-module',
        'div.story-package-module',
        'div.story-list-story'
    ]
    
    articles = []
    for selector in selectors:
        articles = soup.select(selector)
        if articles:
            break
    
    # Fallback
    if not articles:
        articles = [div for div in soup.find_all('div') if 'story' in div.get('class', []) and div.find('a')]
    
    logger.info(f"Found {len(articles)} articles on Bloomberg")
    
    for article in articles[:max_articles]:
        try:
            # Find headline
            headline_selectors = [
                'h1.search-result-story__headline', 
                'h1.story-package-module__headline',
                'a[data-component="headline"]',
                'h3.story-package-module__headline'
            ]
            
            headline = None
            for selector in headline_selectors:
                headline = article.select_one(selector)
                if headline:
                    break
            
            # Fallback to any heading
            if not headline:
                for tag in ['h1', 'h2', 'h3']:
                    headline = article.find(tag)
                    if headline:
                        break
            
            if not headline:
                continue
                
            title = headline.get_text(strip=True)
            
            # Find date
            date_element = article.select_one('time, span.date-time, .published-at')
            date_text = date_element.get_text(strip=True) if date_element else datetime.now().strftime('%Y-%m-%d')
            date = parse_date(date_text)
            
            # Find URL
            if headline.name == 'a' and headline.has_attr('href'):
                url = headline['href']
            else:
                link = headline.find_parent('a', href=True) or article.find('a', href=True)
                url = link['href'] if link else ""
                
            # Make URL absolute if needed
            if url and url.startswith('/'):
                url = f"https://www.bloomberg.com{url}"
            
            news_items.append({
                'date': date,
                'title': title,
                'source': 'Bloomberg',
                'url': url
            })
        except Exception as e:
            logger.error(f"Error processing Bloomberg article: {e}")
    
    return news_items

def get_news_from_financial_times(ticker, company_name, max_articles=15):
    """Get news from Financial Times"""
    # Try company name first, then ticker
    url = f"https://www.ft.com/search?q={quote_plus(company_name)}"
    logger.info(f"Fetching news from Financial Times: {url}")
    
    html = fetch_url(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    news_items = []
    
    # Find article listings
    articles = soup.select('li.search-results__list-item')
    
    # Fallback
    if not articles:
        articles = soup.select('div.o-teaser')
    
    logger.info(f"Found {len(articles)} articles on Financial Times")
    
    for article in articles[:max_articles]:
        try:
            # Find headline
            headline = article.select_one('a.js-teaser-heading-link, h3.o-teaser__heading')
            if not headline:
                continue
                
            title = headline.get_text(strip=True)
            
            # Find date
            date_element = article.select_one('time, .o-date')
            date_text = date_element.get_text(strip=True) if date_element else datetime.now().strftime('%Y-%m-%d')
            date = parse_date(date_text)
            
            # Find URL
            if headline.name == 'a' and headline.has_attr('href'):
                url = headline['href']
            else:
                link = headline.find_parent('a', href=True) or article.find('a', href=True)
                url = link['href'] if link else ""
                
            # Make URL absolute if needed
            if url and url.startswith('/'):
                url = f"https://www.ft.com{url}"
            
            news_items.append({
                'date': date,
                'title': title,
                'source': 'Financial Times',
                'url': url
            })
        except Exception as e:
            logger.error(f"Error processing Financial Times article: {e}")
    
    return news_items

def generate_synthetic_articles(ticker, company_name, count=30):
    """Generate synthetic news articles when real data is insufficient"""
    logger.info(f"Generating {count} synthetic news articles for {company_name} ({ticker})")
    
    # Base date for articles
    now = datetime.now()
    
    # Theme templates with placeholders
    templates = [
        # Positive templates
        "{company} reports strong quarterly results with revenue growth",
        "{company} stock rises after analyst upgrade",
        "{company} announces new product line, shares up",
        "Analysts raise price target on {company} citing growth potential",
        "{company} exceeds expectations, raises guidance",
        "{company} partners with {partner} on new initiative",
        "{company} expands into {market} market, investors optimistic",
        "{company} sees increased demand for {product} product line",
        "{company} CEO discusses future growth strategy at investor conference",
        "{company} beats consensus estimates by {percent}%",
        
        # Neutral templates
        "{company} to report earnings next week, analysts mixed",
        "{company} maintains market position despite competition",
        "{company} stock trades sideways following product announcement",
        "What's next for {company}? Analysts weigh in",
        "{company} holds annual shareholder meeting",
        "{company} addresses supply chain challenges",
        "{company} reworks strategy for {market} market",
        "{company} neither surprises nor disappoints investors",
        "{company} keeps dividend unchanged at quarterly meeting",
        "{company} completes previously announced restructuring",
        
        # Negative templates
        "{company} misses earnings expectations, shares down",
        "Analysts downgrade {company} citing competitive pressures",
        "{company} faces challenges in {market} market",
        "{company} lowers guidance for coming quarters",
        "{company} stock drops amid broader market selloff",
        "{company} announces layoffs as part of cost-cutting measures",
        "{company} faces regulatory scrutiny over {issue}",
        "{company} loses market share to competitors",
        "{company} product launch delayed, investors concerned",
        "{company} grapples with increasing costs, margin pressure"
    ]
    
    # Random partners for partnership news
    partners = ["Microsoft", "Amazon", "Google", "Apple", "Meta", "IBM", "Oracle", "Samsung", "Intel", "Nvidia"]
    
    # Random markets
    markets = ["European", "Asian", "Latin American", "emerging", "enterprise", "consumer", "mobile", "cloud", "AI", "healthcare"]
    
    # Random products
    products = ["flagship", "new", "core", "premium", "budget", "enterprise", "consumer", "latest", "innovative", "next-generation"]
    
    # Random issues
    issues = ["data practices", "pricing", "competitive behavior", "environmental impact", "labor practices", "tax strategy", "product safety", "privacy concerns", "user rights", "market dominance"]
    
    # Random percentages
    percentages = ["5", "8", "10", "12", "15", "20", "25", "3", "7", "18"]
    
    synthetic_articles = []
    
    for i in range(count):
        # Select a template
        template = random.choice(templates)
        
        # Create placeholders dictionary
        placeholders = {
            "company": company_name,
            "ticker": ticker,
            "partner": random.choice(partners),
            "market": random.choice(markets),
            "product": random.choice(products),
            "issue": random.choice(issues),
            "percent": random.choice(percentages)
        }
        
        # Fill in the template
        title = template.format(**placeholders)
        
        # Generate a date within the past 30 days, weighted toward more recent
        days_ago = int(np.random.exponential(scale=7))
        days_ago = min(days_ago, 30)  # Cap at 30 days
        date = (now - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Add random source for variety
        sources = [
            "Market Analysis", "Stock News Daily", "Financial Reporter", 
            "Wall Street Today", "Investor's Journal", "Market Watch Digest",
            "Finance Insider", "The Street View", "Business Observer", "Market Update"
        ]
        source = random.choice(sources)
        
        synthetic_articles.append({
            'date': date,
            'title': title,
            'source': f"{source} (Synthetic)",
            'url': ""
        })
    
    return synthetic_articles

def deduplicate_news(news_items):
    """Remove duplicate news items based on title similarity"""
    if not news_items:
        return []
        
    # Function to normalize title for comparison
    def normalize_title(title):
        if not isinstance(title, str):
            return ""
        # Convert to lowercase, remove punctuation
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    # Add normalized titles
    for item in news_items:
        item['normalized_title'] = normalize_title(item['title'])
    
    # Sort by date (newest first) to keep most recent version of duplicates
    news_items.sort(key=lambda x: x['date'], reverse=True)
    
    # Use set to track seen titles
    seen_titles = set()
    unique_items = []
    
    for item in news_items:
        normalized = item['normalized_title']
        # Skip if empty title
        if not normalized:
            continue
            
        # Check for near duplicates
        is_duplicate = False
        for seen_title in seen_titles:
            # Use string similarity (could use more sophisticated methods like Levenshtein)
            # Here we're just checking if one is a substring of the other, which catches many duplicates
            if len(normalized) > 20 and (normalized in seen_title or seen_title in normalized):
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_titles.add(normalized)
            unique_items.append(item)
    
    # Remove the temporary normalized titles field
    for item in unique_items:
        del item['normalized_title']
    
    # Sort by date (oldest to newest) for processing
    unique_items.sort(key=lambda x: x['date'])
    
    return unique_items

def get_news(ticker, company_name, days=30, news_sources=None, headers=None):
    """Collect financial news articles for the specified company with enhanced source coverage."""
    logger.info(f"Collecting news for {company_name} ({ticker})")
    news_items = []

    if news_sources is None:
        news_sources = config.NEWS_SOURCES
    if headers is None:
        headers = config.DEFAULT_HEADERS
    
    # Create a session for reuse
    session = create_requests_session()
    
    # Try API sources first (if keys provided)
    if hasattr(config, 'NEWS_API_KEY') and config.NEWS_API_KEY:
        api_news = extract_newsapi_results(config.NEWS_API_KEY, ticker, company_name, days)
        news_items.extend(api_news)
        logger.info(f"Collected {len(api_news)} news items from NewsAPI")
    
    if hasattr(config, 'ALPHA_VANTAGE_API_KEY') and config.ALPHA_VANTAGE_API_KEY:
        alpha_news = extract_alpha_vantage_news(config.ALPHA_VANTAGE_API_KEY, ticker, days)
        news_items.extend(alpha_news)
        logger.info(f"Collected {len(alpha_news)} news items from Alpha Vantage")
    
    # Traditional news sources
    source_functions = {
        'yahoo': (get_news_from_yahoo_finance, [ticker]),
        'finviz': (get_news_from_finviz, [ticker]),
        'marketwatch': (get_news_from_marketwatch, [ticker]),
        'wsj': (get_news_from_wsj, [ticker]),
        'reuters': (get_news_from_reuters, [company_name]),
        'bloomberg': (get_news_from_bloomberg, [company_name]),
        'ft': (get_news_from_financial_times, [ticker, company_name])
    }
    
    # Track success of each source
    source_success = {source: False for source in source_functions.keys()}
    
    # Try all enabled sources
    for source, (func, args) in source_functions.items():
        if not news_sources.get(source, True):
            continue
            
        try:
            source_news = func(*args)
            if source_news:
                news_items.extend(source_news)
                source_success[source] = True
                logger.info(f"Collected {len(source_news)} news items from {source}")
            
            # Small delay between requests
            time.sleep(random.uniform(1.0, 3.0))
        except Exception as e:
            logger.error(f"Error with {source} scraping: {e}")
    
    # Log results of scraping
    logger.info(f"News collection results:")
    for source, success in source_success.items():
        if source in news_sources and news_sources[source]:
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  - {source.capitalize()}: {status}")
    
    # Deduplicate news
    unique_news = deduplicate_news(news_items)
    logger.info(f"After deduplication: {len(unique_news)} unique news items")
    
    # Generate synthetic news if we didn't find enough
    if len(unique_news) < 30:
        logger.warning(f"Only found {len(unique_news)} news items. Adding synthetic data...")
        
        # Calculate how many synthetic items to add
        needed_synthetic = max(0, 100 - len(unique_news))
        synthetic_news = generate_synthetic_articles(ticker, company_name, count=needed_synthetic)
        unique_news.extend(synthetic_news)
        
        logger.info(f"Added {len(synthetic_news)} synthetic news items to reach {len(unique_news)} total")
    
    # Ensure we have the expected columns in the DataFrame
    df = pd.DataFrame(unique_news)
    logger.info(f"Final news dataframe shape: {df.shape}")
    
    # Ensure the dataframe has required columns
    if 'date' not in df.columns:
        df['date'] = datetime.now().strftime('%Y-%m-%d')
    if 'title' not in df.columns:
        df['title'] = f"News about {company_name}"
    if 'source' not in df.columns:
        df['source'] = 'Unknown'
    if 'url' not in df.columns:
        df['url'] = ''
    
    # Add date_parsed column for convenience
    df['date_parsed'] = pd.to_datetime(df['date'])
    
    return df

def get_stock_data(ticker, start_date, end_date):
    """Get historical stock price data with enhanced reliability."""
    logger.info(f"Getting stock data for {ticker} from {start_date} to {end_date}")
    
    stock_data = pd.DataFrame()
    
    # Ensure dates are datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Add a retry mechanism
    for attempt in range(3):
        try:
            # Try to get data from Yahoo Finance
            stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not stock_data.empty:
                # Check if we have enough data
                if len(stock_data) >= max(5, (end_date - start_date).days / 10):
                    logger.info(f"Successfully downloaded {len(stock_data)} rows of data for {ticker}")
                    
                    # Add ticker suffix to column names - fixing the naming here
                    stock_data = stock_data.copy()  # Create a copy to avoid SettingWithCopyWarning
                    stock_data.columns = [f"{col}_{ticker}" for col in stock_data.columns]
                    break
                    
            logger.warning(f"Attempt {attempt+1}: Insufficient data retrieved. Retrying...")
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    
    # Create fallback data if needed
    if stock_data.empty or len(stock_data) < 5:
        logger.warning(f"No reliable stock data found for {ticker}. Using simulated data.")
        # Create better simulated data that follows a random walk
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        np.random.seed(42)  # For reproducibility
        
        # Generate prices following a random walk with drift
        # Using reasonable parameters for a typical stock
        initial_price = 100
        daily_drift = 0.0005  # Small positive drift (0.05% per day)
        daily_volatility = 0.015  # 1.5% daily volatility
        
        # Generate returns with realistic autocorrelation
        # First generate random returns
        random_returns = np.random.normal(daily_drift, daily_volatility, len(date_range))
        
        # Add mild autocorrelation (momentum effect)
        autocorrelation = 0.1  # Mild positive autocorrelation
        for i in range(1, len(random_returns)):
            random_returns[i] += autocorrelation * random_returns[i-1]
        
        # Convert returns to prices
        prices = initial_price * np.exp(np.cumsum(random_returns))
        
        # Create dataframe with all required columns
        stock_data = pd.DataFrame(index=date_range)
        stock_data[f'Close_{ticker}'] = prices
        
        # Generate realistic OHLC data
        stock_data[f'Open_{ticker}'] = prices * np.exp(np.random.normal(0, 0.005, len(date_range)))
        stock_data[f'High_{ticker}'] = np.maximum(
            stock_data[f'Open_{ticker}'], 
            stock_data[f'Close_{ticker}']
        ) * np.exp(np.random.normal(0.005, 0.005, len(date_range)))
        stock_data[f'Low_{ticker}'] = np.minimum(
            stock_data[f'Open_{ticker}'], 
            stock_data[f'Close_{ticker}']
        ) * np.exp(np.random.normal(-0.005, 0.005, len(date_range)))
        
        # Generate volume data with realistic patterns
        # Higher volume on trend days, lower volume on ranging days
        returns_abs = np.abs(random_returns)
        normalized_returns = (returns_abs - np.mean(returns_abs)) / np.std(returns_abs)
        volume_factor = np.exp(normalized_returns * 0.5)  # Volume increases with return magnitude
        base_volume = 1000000  # Base volume level
        stock_data[f'Volume_{ticker}'] = (base_volume * volume_factor).astype(int)
        
        stock_data[f'Adj Close_{ticker}'] = stock_data[f'Close_{ticker}']
        
        logger.info(f"Created simulated price data with {len(stock_data)} rows")
    
    # Calculate daily returns
    close_col = f'Close_{ticker}'
    
    # Check if the close column exists
    if close_col not in stock_data.columns:
        # Get the first column that matches "Close" pattern
        close_cols = [col for col in stock_data.columns if 'Close' in col]
        if close_cols:
            close_col = close_cols[0]
            logger.info(f"Using {close_col} as the close price column")
        else:
            # If no Close column, try Adj Close
            adj_close_cols = [col for col in stock_data.columns if 'Adj Close' in col]
            if adj_close_cols:
                close_col = adj_close_cols[0]
                logger.info(f"Using {close_col} as the close price column")
            else:
                # Last resort - use first column
                if len(stock_data.columns) > 0:
                    close_col = stock_data.columns[0]
                    logger.warning(f"No Close column found. Using {close_col} instead")
                else:
                    # Create a dummy close column
                    stock_data[f'Close_{ticker}'] = 100.0
                    close_col = f'Close_{ticker}'
                    logger.warning(f"No columns found in data. Created dummy {close_col}")

    # Now we can safely use the close_col
    stock_data['Return'] = stock_data[close_col].pct_change() * 100
    
    # Add additional basic features
    # Simple moving averages
    for period in [5, 10, 20, 50, 200]:
        stock_data[f'SMA_{period}d'] = stock_data[close_col].rolling(window=period).mean()
    
    # Exponential moving averages
    for period in [5, 10, 20, 50]:
        stock_data[f'EMA_{period}d'] = stock_data[close_col].ewm(span=period, adjust=False).mean()
    
    # Simple volatility
    stock_data['Volatility_20d'] = stock_data['Return'].rolling(window=20).std()
    
    # Fill any NAs at the beginning
    stock_data = stock_data.fillna(method='bfill')
    
    return stock_data

def get_market_data(start_date, end_date, indices=None):
    """Get market index data for the given period"""
    if indices is None:
        indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow Jones, NASDAQ, VIX
    
    logger.info(f"Getting market data for indices: {', '.join(indices)}")
    
    market_data = {}
    
    for index in indices:
        try:
            # Download index data
            index_data = yf.download(index, start=start_date, end=end_date, progress=False)
            
            if not index_data.empty:
                # Calculate returns
                index_data['Return'] = index_data['Close'].pct_change() * 100
                
                # Save in dictionary
                market_data[index] = index_data
                logger.info(f"Successfully downloaded {len(index_data)} rows for {index}")
        except Exception as e:
            logger.error(f"Error downloading data for {index}: {e}")
    
    return market_data

def get_economic_indicators(start_date, end_date):
    """Get economic indicators from FRED (simplified placeholder version)"""
    logger.info("This function would download economic indicators if implemented")
    
    # In a real implementation, this would fetch data from FRED API
    # Examples of useful indicators:
    # - GDP growth rate
    # - Unemployment rate
    # - Consumer sentiment
    # - Manufacturing indices
    # - Interest rates
    
    # For now, return empty DataFrame
    return pd.DataFrame()

def get_earnings_calendar(ticker, company_name):
    """Get upcoming and past earnings dates"""
    logger.info(f"Getting earnings calendar for {company_name} ({ticker})")
    
    try:
        # Use yfinance to get earnings dates
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        
        if calendar is not None and not calendar.empty:
            logger.info(f"Found earnings calendar data for {ticker}")
            return calendar
    except Exception as e:
        logger.error(f"Error getting earnings calendar: {e}")
    
    logger.warning(f"No earnings calendar found for {ticker}")
    return pd.DataFrame()