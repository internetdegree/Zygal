# twitter_integration/twitter_monitor.py
import tweepy
import logging
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List
import json
from config import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
from .sentiment_analyzer import SentimentAnalyzer
from .tweet_processor import TweetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterMonitor:
    def __init__(self):
        # Initialize Twitter API client
        auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        self.client = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN,
            consumer_key=TWITTER_API_KEY,
            consumer_secret=TWITTER_API_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_SECRET
        )
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.tweet_processor = TweetProcessor()
        self.tracked_tokens = set()
        self.sentiment_cache = {}
        
    async def start_monitoring(self):
        """Start monitoring Twitter for relevant signals"""
        try:
            # Start different monitoring tasks
            await asyncio.gather(
                self._monitor_trending_tokens(),
                self._monitor_influential_accounts(),
                self._monitor_token_mentions()
            )
        except Exception as e:
            logger.error(f"Error in Twitter monitoring: {e}")
            raise

    async def _monitor_trending_tokens(self):
        """Monitor for trending token discussions"""
        while True:
            try:
                # Search for tweets about Solana tokens
                query = "solana OR $sol OR #solana lang:en -is:retweet"
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=100,
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                for tweet in tweets.data:
                    await self._process_tweet(tweet)
                    
            except Exception as e:
                logger.error(f"Error monitoring trending tokens: {e}")
                
            await asyncio.sleep(60)  # Rate limit compliance

    async def _process_tweet(self, tweet) -> Dict:
        """Process individual tweets for relevant information"""
        try:
            # Extract tweet data
            tweet_data = {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at,
                'metrics': tweet.public_metrics,
                'tokens': self.tweet_processor.extract_token_mentions(tweet.text)
            }
            
            # Analyze sentiment
            if tweet_data['tokens']:
                sentiment = await self.sentiment_analyzer.analyze_tweet(tweet.text)
                tweet_data['sentiment'] = sentiment
                
                # Update sentiment cache for each mentioned token
                for token in tweet_data['tokens']:
                    if token not in self.sentiment_cache:
                        self.sentiment_cache[token] = []
                    self.sentiment_cache[token].append(sentiment)
                    
            return tweet_data
            
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return None

    async def get_token_sentiment(self, token_address: str) -> Dict:
        """Get aggregated sentiment data for a specific token"""
        try:
            if token_address not in self.sentiment_cache:
                return {
                    'score': 0,
                    'volume': 0,
                    'trending': False
                }
                
            sentiments = self.sentiment_cache[token_address]
            
            return {
                'score': sum(sentiments) / len(sentiments),
                'volume': len(sentiments),
                'trending': len(sentiments) > 100  # Arbitrary threshold
            }
            
        except Exception as e:
            logger.error(f"Error getting token sentiment: {e}")
            return None

# twitter_integration/sentiment_analyzer.py
from textblob import TextBlob
import re
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.emoji_sentiment = self._load_emoji_sentiment()
        
    def _load_emoji_sentiment(self) -> Dict:
        """Load emoji sentiment mappings"""
        return {
            '🚀': 1.0,
            '🌙': 0.8,
            '💎': 0.7,
            '📈': 0.6,
            '🔥': 0.5,
            '📉': -0.6,
            '💀': -0.7,
            '⚰️': -0.8
        }
        
    async def analyze_tweet(self, text: str) -> float:
        """Analyze sentiment of tweet text"""
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Get TextBlob sentiment
            blob = TextBlob(cleaned_text)
            text_sentiment = blob.sentiment.polarity
            
            # Get emoji sentiment
            emoji_sentiment = self._analyze_emoji_sentiment(text)
            
            # Combine sentiments (weighted average)
            combined_sentiment = (text_sentiment * 0.7) + (emoji_sentiment * 0.3)
            
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing tweet sentiment: {e}")
            return 0.0
            
    def _clean_text(self, text: str) -> str:
        """Clean tweet text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
        
    def _analyze_emoji_sentiment(self, text: str) -> float:
        """Analyze sentiment based on emojis"""
        emoji_sentiments = []
        
        for emoji, sentiment in self.emoji_sentiment.items():
            if emoji in text:
                emoji_sentiments.append(sentiment)
                
        if not emoji_sentiments:
            return 0.0
            
        return sum(emoji_sentiments) / len(emoji_sentiments)

# twitter_integration/tweet_processor.py
import re
from typing import List, Set
import logging

logger = logging.getLogger(__name__)

class TweetProcessor:
    def __init__(self):
        self.token_patterns = {
            'address': r'[1-9A-HJ-NP-Za-km-z]{32,44}',
            'cashtag': r'\$[A-Za-z]+',
            'hashtag': r'#[A-Za-z]+'
        }
        
    def extract_token_mentions(self, text: str) -> Set[str]:
        """Extract token mentions from tweet text"""
        try:
            mentions = set()
            
            # Find Solana addresses
            addresses = re.findall(self.token_patterns['address'], text)
            mentions.update(addresses)
            
            # Find cashtags
            cashtags = re.findall(self.token_patterns['cashtag'], text)
            mentions.update(tag[1:] for tag in cashtags)  # Remove $ prefix
            
            # Find relevant hashtags
            hashtags = re.findall(self.token_patterns['hashtag'], text)
            # Filter for token-related hashtags
            token_hashtags = {tag[1:] for tag in hashtags if self._is_token_hashtag(tag)}
            mentions.update(token_hashtags)
            
            return mentions
            
        except Exception as e:
            logger.error(f"Error extracting token mentions: {e}")
            return set()
            
    def _is_token_hashtag(self, hashtag: str) -> bool:
        """Check if hashtag is likely token-related"""
        # Remove # prefix
        tag = hashtag[1:].lower()
        
        # Common token-related patterns
        token_indicators = [
            'token',
            'coin',
            'sol',
            'solana',
            'launch',
            'presale'
        ]
        
        return any(indicator in tag for indicator in token_indicators)

# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Twitter API credentials
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# requirements.txt
tweepy==4.12.1
textblob==0.17.1
python-dotenv==1.0.0
asyncio==3.4.3
