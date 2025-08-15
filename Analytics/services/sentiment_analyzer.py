"""
Sentiment Analysis Service using FinBERT
Analyzes financial text sentiment with confidence scoring and batch processing.
"""

import logging
import time
import re
import unicodedata
import threading
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from functools import lru_cache
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from django.conf import settings
from django.core.cache import cache

TORCH_AVAILABLE = True

logger = logging.getLogger(__name__)


# Stocks that are known to cause sentiment analysis timeouts - use fallback
PROBLEMATIC_STOCKS = {'TSLA', 'GM', 'F', 'NIO', 'RIVN', 'LCID', 'URI', 'PLUG', 'FCEL', 'BE', 'TROW', 'NVDA'}  # Stocks with problematic news content

# Global flag to disable FinBERT if persistent timeouts occur
_FINBERT_DISABLED = False  # Re-enabled with improved safety mechanisms


class SentimentMetrics:
    """
    Metrics collector for sentiment analysis operations.
    Tracks performance, accuracy, and usage statistics.
    """
    
    def __init__(self):
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all metrics counters."""
        self.total_requests = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_processing_time = 0.0
        self.total_articles_processed = 0
        self.confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}
        self.sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
    def log_request(self, symbol: str, article_count: int):
        """Log a sentiment analysis request."""
        self.total_requests += 1
        self.total_articles_processed += article_count
        logger.info(
            "Sentiment analysis request",
            extra={
                'event_type': 'sentiment_request',
                'symbol': symbol,
                'article_count': article_count,
                'total_requests': self.total_requests
            }
        )
    
    def log_success(self, symbol: str, score: float, confidence: float, processing_time: float):
        """Log successful sentiment analysis."""
        self.successful_analyses += 1
        self.total_processing_time += processing_time
        
        # Categorize confidence
        if confidence >= 0.8:
            confidence_category = 'high'
        elif confidence >= 0.6:
            confidence_category = 'medium'
        else:
            confidence_category = 'low'
        self.confidence_distribution[confidence_category] += 1
        
        # Categorize sentiment
        if score > 0.1:
            sentiment_category = 'positive'
        elif score < -0.1:
            sentiment_category = 'negative'
        else:
            sentiment_category = 'neutral'
        self.sentiment_distribution[sentiment_category] += 1
        
        logger.info(
            "Sentiment analysis completed successfully",
            extra={
                'event_type': 'sentiment_success',
                'symbol': symbol,
                'sentiment_score': score,
                'confidence': confidence,
                'confidence_category': confidence_category,
                'sentiment_category': sentiment_category,
                'processing_time_ms': processing_time * 1000,
                'success_rate': self.successful_analyses / self.total_requests if self.total_requests > 0 else 0
            }
        )
    
    def log_failure(self, symbol: str, error: str, processing_time: float = 0):
        """Log failed sentiment analysis."""
        self.failed_analyses += 1
        self.total_processing_time += processing_time
        
        logger.error(
            "Sentiment analysis failed",
            extra={
                'event_type': 'sentiment_failure',
                'symbol': symbol,
                'error': error,
                'processing_time_ms': processing_time * 1000,
                'failure_rate': self.failed_analyses / self.total_requests if self.total_requests > 0 else 0
            }
        )
    
    def log_cache_hit(self, symbol: str):
        """Log cache hit."""
        self.cache_hits += 1
        logger.debug(
            "Sentiment cache hit",
            extra={
                'event_type': 'sentiment_cache_hit',
                'symbol': symbol,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
        )
    
    def log_cache_miss(self, symbol: str):
        """Log cache miss."""
        self.cache_misses += 1
        logger.debug(
            "Sentiment cache miss",
            extra={
                'event_type': 'sentiment_cache_miss',
                'symbol': symbol,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            }
        )
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_requests': self.total_requests,
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'success_rate': self.successful_analyses / self.total_requests if self.total_requests > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_processing_time_ms': (self.total_processing_time / self.successful_analyses * 1000) if self.successful_analyses > 0 else 0,
            'total_articles_processed': self.total_articles_processed,
            'confidence_distribution': self.confidence_distribution,
            'sentiment_distribution': self.sentiment_distribution
        }


# Global metrics instance
sentiment_metrics = SentimentMetrics()


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT model.
    Provides single and batch analysis with confidence filtering.
    """
    
    # Model configuration
    MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 512
    DEFAULT_BATCH_SIZE = 16
    MIN_BATCH_SIZE = 4
    MAX_BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.6  # 60% minimum confidence
    CACHE_TTL_RECENT = 300  # 5 minutes for recent queries
    CACHE_TTL_HISTORICAL = 86400  # 24 hours for historical data
    
    # Performance monitoring
    MAX_PROCESSING_TIME = 30.0  # Maximum seconds per batch
    ERROR_THRESHOLD = 0.2  # Reduce batch size if error rate > 20%
    
    # Sentiment mapping
    LABEL_MAP = {
        'positive': 'positive',
        'negative': 'negative', 
        'neutral': 'neutral'
    }
    
    def __init__(self):
        """Initialize the sentiment analyzer with FinBERT model."""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialized = False
        
        # Adaptive batch processing
        self.current_batch_size = self.DEFAULT_BATCH_SIZE
        self.recent_processing_times = []
        self.recent_error_count = 0
        self.total_batches_processed = 0
        
    @property
    def is_initialized(self) -> bool:
        """Check if model is loaded and ready."""
        return self._initialized
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to handle problematic characters that cause tokenizer hangs.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text safe for tokenization
        """
        if not text:
            return ""
        
        # Replace smart quotes and special characters
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes 
        text = text.replace('–', '-').replace('—', '-')  # Em/en dashes
        text = text.replace('…', '...')  # Ellipsis
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _has_problematic_content(self, text: str) -> bool:
        """
        Check if text has content that might cause tokenizer hangs.
        
        Args:
            text: Text to check
            
        Returns:
            True if text might be problematic
        """
        if not text:
            return False
        
        # Check for patterns that cause issues
        problematic_patterns = [
            r'[^\x00-\x7F]',  # Non-ASCII characters
            r'[\u2018\u2019\u201C\u201D]',  # Smart quotes
            r'[\u2013\u2014]',  # Em/en dashes
            r'\u2026',  # Ellipsis
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment fallback for problematic text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Basic sentiment result
        """
        if not text:
            return self._neutral_sentiment()
        
        # Simple word-based sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'gain', 'up', 'rise', 'bull', 'buy', 'strong']
        negative_words = ['bad', 'poor', 'negative', 'loss', 'down', 'fall', 'bear', 'sell', 'weak', 'decline']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            score = 0.3  # Mild positive
            label = 'positive'
            confidence = 0.6
        elif neg_count > pos_count:
            score = -0.3  # Mild negative
            label = 'negative'
            confidence = 0.6
        else:
            score = 0.0
            label = 'neutral'
            confidence = 0.5
        
        return {
            'sentimentScore': score,
            'sentimentLabel': label,
            'sentimentConfidence': confidence,
            'analysisTime': 0.001,
            'textLength': len(text),
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }
    
    def _fallback_sentiment_result(self, text: str) -> Dict[str, Any]:
        """
        Generate fallback sentiment result in FinBERT pipeline format.
        
        Args:
            text: Text to analyze
            
        Returns:
            Result in FinBERT pipeline format (label, score)
        """
        if not text:
            return {'label': 'neutral', 'score': 0.5}
        
        # Simple word-based sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'gain', 'up', 'rise', 'bull', 'buy', 'strong']
        negative_words = ['bad', 'poor', 'negative', 'loss', 'down', 'fall', 'bear', 'sell', 'weak', 'decline']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'label': 'positive', 'score': 0.7}
        elif neg_count > pos_count:
            return {'label': 'negative', 'score': 0.7}
        else:
            return {'label': 'neutral', 'score': 0.6}
        
    def _lazy_init(self):
        """Lazy initialization of the model to save memory."""
        global _FINBERT_DISABLED
        
        if _FINBERT_DISABLED:
            logger.info("FinBERT is disabled, skipping model initialization")
            return
            
        if not self._initialized:
            try:
                logger.info("Loading FinBERT model for sentiment analysis...")
                start_time = time.time()
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
                
                # Create pipeline for easier inference
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1  # Use CPU, set to 0 for GPU if available
                )
                
                self._initialized = True
                load_time = time.time() - start_time
                logger.info(f"FinBERT model loaded successfully in {load_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Failed to load FinBERT model, disabling: {str(e)}")
                _FINBERT_DISABLED = True
                raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def analyzeSentimentSingle(
        self, 
        text: str,
        use_cache: bool = True,
        timeout_seconds: int = 15
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Financial text to analyze
            use_cache: Whether to use cache for results
            
        Returns:
            Dictionary with sentiment score, label, and confidence
        """
        global _FINBERT_DISABLED
        
        if not text or not text.strip():
            return self._neutral_sentiment()
        
        # Check if FinBERT is globally disabled or text has problematic content
        if _FINBERT_DISABLED or self._has_problematic_content(text):
            if _FINBERT_DISABLED:
                logger.info("Using fallback sentiment analysis - FinBERT disabled")
            else:
                logger.info("Using fallback sentiment analysis for problematic text")
            return self._fallback_sentiment(text)
        
        # Preprocess text to handle problematic characters
        text = self._preprocess_text(text)
        if not text:
            return self._neutral_sentiment()
            
        # Check cache first
        if use_cache:
            cache_key = f"sentiment:single:{hash(text)}"
            cached_result = cache.get(cache_key)
            if cached_result:
                sentiment_metrics.log_cache_hit("single_text")
                return cached_result
            else:
                sentiment_metrics.log_cache_miss("single_text")
        
        # Ensure model is loaded
        try:
            self._lazy_init()
        except Exception as e:
            logger.warning(f"Model initialization failed, using fallback: {str(e)}")
            return self._fallback_sentiment(text)
        
        if _FINBERT_DISABLED:
            return self._fallback_sentiment(text)
        
        try:
            # Truncate text if too long
            if len(text) > 5000:
                text = text[:5000]
            
            # Run sentiment analysis
            start_time = time.time()
            results = self.pipeline(text, truncation=True, max_length=self.MAX_LENGTH)
            
            if not results:
                return self._neutral_sentiment()
                
            result = results[0]
            
            # Map label and calculate score
            label = self.LABEL_MAP.get(result['label'].lower(), 'neutral')
            confidence = result['score']
            
            # Apply confidence threshold
            if confidence < self.CONFIDENCE_THRESHOLD:
                logger.debug(f"Confidence {confidence:.2f} below threshold, returning neutral")
                return self._neutral_sentiment()
            
            # Calculate sentiment score (-1 to 1)
            if label == 'positive':
                score = confidence
            elif label == 'negative':
                score = -confidence
            else:
                score = 0.0
                
            analysis_time = time.time() - start_time
            
            result_dict = {
                'sentimentScore': round(score, 4),
                'sentimentLabel': label,
                'sentimentConfidence': round(confidence, 4),
                'analysisTime': round(analysis_time, 3),
                'textLength': len(text),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache result
            if use_cache:
                cache.set(cache_key, result_dict, self.CACHE_TTL_RECENT)
            
            # Log successful analysis
            sentiment_metrics.log_success("single_text", score, confidence, analysis_time)
            return result_dict
            
        except Exception as e:
            # Log failure
            sentiment_metrics.log_failure("single_text", str(e), time.time() - start_time)
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self._neutral_sentiment()
    
    def analyzeSentimentBatch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts in batch.
        
        Args:
            texts: List of financial texts to analyze
            use_cache: Whether to use cache for results
            
        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []
            
        # Ensure model is loaded
        self._lazy_init()
        
        results = []
        valid_texts = []
        valid_indices = []
        
        # Check cache and filter valid texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append(self._neutral_sentiment())
            else:
                # Preprocess text to handle problematic characters
                processed_text = self._preprocess_text(text)
                if not processed_text:
                    results.append(self._neutral_sentiment())
                    continue
                    
                if use_cache:
                    cache_key = f"sentiment:single:{hash(processed_text)}"
                    cached_result = cache.get(cache_key)
                    if cached_result:
                        results.append(cached_result)
                        continue
                
                # Truncate if needed
                if len(processed_text) > 5000:
                    processed_text = processed_text[:5000]
                    
                valid_texts.append(processed_text)
                valid_indices.append(i)
                results.append(None)  # Placeholder
        
        # Process valid texts in adaptive batches
        if valid_texts:
            try:
                start_time = time.time()
                
                # Process in adaptive chunks to manage memory and performance
                for batch_start in range(0, len(valid_texts), self.current_batch_size):
                    batch_end = min(batch_start + self.current_batch_size, len(valid_texts))
                    batch_texts = valid_texts[batch_start:batch_end]
                    
                    # Run batch analysis with retry logic
                    batch_results = self._process_batch_with_retry(batch_texts)
                    
                    # Process batch results
                    for j, result in enumerate(batch_results):
                        idx = valid_indices[batch_start + j]
                        
                        if not result:
                            results[idx] = self._neutral_sentiment()
                            continue
                            
                        # Map label and calculate score
                        label = self.LABEL_MAP.get(result['label'].lower(), 'neutral')
                        confidence = result['score']
                        
                        # Apply confidence threshold
                        if confidence < self.CONFIDENCE_THRESHOLD:
                            results[idx] = self._neutral_sentiment()
                            continue
                        
                        # Calculate sentiment score
                        if label == 'positive':
                            score = confidence
                        elif label == 'negative':
                            score = -confidence
                        else:
                            score = 0.0
                        
                        result_dict = {
                            'sentimentScore': round(score, 4),
                            'sentimentLabel': label,
                            'sentimentConfidence': round(confidence, 4),
                            'textLength': len(batch_texts[j]),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results[idx] = result_dict
                        
                        # Cache individual results
                        if use_cache:
                            cache_key = f"sentiment:single:{hash(batch_texts[j])}"
                            cache.set(cache_key, result_dict, self.CACHE_TTL_RECENT)
                
                batch_time = time.time() - start_time
                logger.info(f"Batch sentiment analysis completed: {len(valid_texts)} texts in {batch_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in batch sentiment analysis: {str(e)}")
                # Fill remaining None values with neutral sentiment
                for i in range(len(results)):
                    if results[i] is None:
                        results[i] = self._neutral_sentiment()
        
        return results
    
    def aggregateSentiment(
        self,
        sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple sentiment scores into overall sentiment.
        
        Args:
            sentiments: List of sentiment analysis results
            
        Returns:
            Aggregated sentiment with statistics
        """
        if not sentiments:
            return self._neutral_sentiment()
            
        scores = []
        labels = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = 0
        
        for sentiment in sentiments:
            if sentiment and 'sentimentScore' in sentiment:
                scores.append(sentiment['sentimentScore'])
                label = sentiment.get('sentimentLabel', 'neutral')
                labels[label] = labels.get(label, 0) + 1
                total_confidence += sentiment.get('sentimentConfidence', 0)
        
        if not scores:
            return self._neutral_sentiment()
        
        # Calculate aggregate metrics
        avg_score = sum(scores) / len(scores)
        avg_confidence = total_confidence / len(scores)
        
        # Determine overall label
        if avg_score > 0.1:
            overall_label = 'positive'
        elif avg_score < -0.1:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        return {
            'sentimentScore': round(avg_score, 4),
            'sentimentLabel': overall_label,
            'sentimentConfidence': round(avg_confidence, 4),
            'distribution': labels,
            'sampleCount': len(scores),
            'minScore': round(min(scores), 4) if scores else 0,
            'maxScore': round(max(scores), 4) if scores else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def analyzeNewsArticles(
        self,
        articles: List[Dict[str, str]], 
        aggregate: bool = True,
        symbol: str = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles.
        
        Args:
            articles: List of article dicts with 'title' and 'summary' keys
            aggregate: Whether to return aggregated or individual results
            symbol: Stock symbol (used for problematic stock handling)
            
        Returns:
            Sentiment analysis results
        """
        if not articles:
            return self._neutral_sentiment()
        
        # For problematic stocks, use fallback approach
        if symbol and symbol.upper() in PROBLEMATIC_STOCKS:
            logger.info(f"Using fallback sentiment analysis for problematic stock: {symbol}")
            texts = []
            for article in articles:
                title = article.get('title', '')
                summary = article.get('summary', '')
                text = f"{title}. {summary}".strip()
                if text:
                    texts.append(text)
            
            if not texts:
                return self._neutral_sentiment()
            
            # Use fallback method for each text
            sentiments = [self._fallback_sentiment(text) for text in texts]
            
            if aggregate:
                result = self.aggregateSentiment(sentiments)
                result['newsCount'] = len(articles)
                result['fallback'] = True
                return result
            else:
                return {
                    'articles': sentiments,
                    'newsCount': len(articles),
                    'aggregate': self.aggregateSentiment(sentiments),
                    'fallback': True
                }
        
        # Extract text from articles
        texts = []
        for article in articles:
            # Combine title and summary for analysis
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title}. {summary}".strip()
            if text:
                texts.append(text)
        
        if not texts:
            return self._neutral_sentiment()
        
        # Analyze all texts
        sentiments = self.analyzeSentimentBatch(texts)
        
        if aggregate:
            result = self.aggregateSentiment(sentiments)
            result['newsCount'] = len(articles)
            return result
        else:
            return {
                'articles': sentiments,
                'newsCount': len(articles),
                'aggregate': self.aggregateSentiment(sentiments)
            }
    
    def _neutral_sentiment(self) -> Dict[str, Any]:
        """Return neutral sentiment result."""
        return {
            'sentimentScore': 0.0,
            'sentimentLabel': 'neutral',
            'sentimentConfidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def clearCache(self, pattern: str = "sentiment:*"):
        """Clear sentiment cache entries."""
        try:
            cache.delete_pattern(pattern)
            logger.info(f"Cleared cache entries matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def getCachedSentiment(
        self,
        cache_key: str,
        symbol: str,
        is_recent: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached sentiment with metrics logging.
        
        Args:
            cache_key: Cache key to lookup
            symbol: Stock symbol for logging
            is_recent: Whether this is a recent query (affects TTL)
            
        Returns:
            Cached sentiment result or None
        """
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                sentiment_metrics.log_cache_hit(symbol)
                
                # Add cache metadata
                cached_result['cached'] = True
                cached_result['cache_key'] = cache_key
                cached_result['cache_type'] = 'recent' if is_recent else 'historical'
                
                return cached_result
            else:
                sentiment_metrics.log_cache_miss(symbol)
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cache for {cache_key}: {str(e)}")
            sentiment_metrics.log_cache_miss(symbol)
            return None
    
    def setCachedSentiment(
        self,
        cache_key: str,
        result: Dict[str, Any],
        symbol: str,
        is_recent: bool = True
    ) -> bool:
        """
        Set cached sentiment with appropriate TTL.
        
        Args:
            cache_key: Cache key
            result: Sentiment result to cache
            symbol: Stock symbol for logging  
            is_recent: Whether this is a recent query (affects TTL)
            
        Returns:
            True if cached successfully
        """
        try:
            ttl = self.CACHE_TTL_RECENT if is_recent else self.CACHE_TTL_HISTORICAL
            
            # Add cache metadata
            cache_data = result.copy()
            cache_data['cached_at'] = datetime.now().isoformat()
            cache_data['cache_ttl'] = ttl
            
            cache.set(cache_key, cache_data, ttl)
            
            logger.debug(
                f"Cached sentiment for {symbol}",
                extra={
                    'event_type': 'sentiment_cached',
                    'symbol': symbol,
                    'cache_key': cache_key,
                    'ttl_seconds': ttl,
                    'cache_type': 'recent' if is_recent else 'historical'
                }
            )
            return True
            
        except Exception as e:
            logger.error(f"Error caching sentiment for {cache_key}: {str(e)}")
            return False
    
    def generateCacheKey(
        self,
        symbol: str = None,
        text_hash: str = None,
        days: int = None,
        analysis_type: str = "sentiment"
    ) -> str:
        """
        Generate standardized cache keys for different sentiment operations.
        
        Args:
            symbol: Stock symbol
            text_hash: Hash of text content
            days: Number of days for news analysis
            analysis_type: Type of analysis
            
        Returns:
            Standardized cache key
        """
        if text_hash:
            # For single text analysis
            return f"sentiment:text:{text_hash}"
        elif symbol and days:
            # For stock news analysis
            return f"sentiment:stock:{symbol}:{days}d"
        elif symbol:
            # For general stock sentiment
            return f"sentiment:stock:{symbol}"
        else:
            # Fallback
            return f"sentiment:{analysis_type}:{hash(str(datetime.now()))}"
    
    def _update_batch_performance(self, processing_time: float, had_error: bool = False):
        """
        Update batch processing performance metrics and adapt batch size.
        
        Args:
            processing_time: Time taken to process the batch
            had_error: Whether the batch had an error
        """
        self.total_batches_processed += 1
        self.recent_processing_times.append(processing_time)
        
        if had_error:
            self.recent_error_count += 1
        
        # Keep only last 10 measurements
        if len(self.recent_processing_times) > 10:
            self.recent_processing_times.pop(0)
            
        # Reset error count every 20 batches
        if self.total_batches_processed % 20 == 0:
            self.recent_error_count = 0
        
        # Adapt batch size based on performance
        self._adapt_batch_size()
    
    def _adapt_batch_size(self):
        """
        Adapt batch size based on recent performance metrics.
        """
        if len(self.recent_processing_times) < 3:
            return  # Need at least 3 samples
        
        avg_time = sum(self.recent_processing_times) / len(self.recent_processing_times)
        error_rate = self.recent_error_count / min(20, self.total_batches_processed)
        
        old_batch_size = self.current_batch_size
        
        # Reduce batch size if too slow or too many errors
        if avg_time > self.MAX_PROCESSING_TIME or error_rate > self.ERROR_THRESHOLD:
            self.current_batch_size = max(self.MIN_BATCH_SIZE, int(self.current_batch_size * 0.8))
        # Increase batch size if performing well
        elif avg_time < self.MAX_PROCESSING_TIME / 2 and error_rate < 0.05:
            self.current_batch_size = min(self.MAX_BATCH_SIZE, int(self.current_batch_size * 1.2))
        
        if old_batch_size != self.current_batch_size:
            logger.info(
                f"Adapted batch size from {old_batch_size} to {self.current_batch_size}",
                extra={
                    'event_type': 'batch_size_adapted',
                    'old_batch_size': old_batch_size,
                    'new_batch_size': self.current_batch_size,
                    'avg_processing_time': avg_time,
                    'error_rate': error_rate
                }
            )
    
    def _process_batch_with_retry(
        self,
        texts: List[str],
        max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Process batch with retry logic and error handling.
        
        Args:
            texts: List of texts to process
            max_retries: Maximum number of retries per batch
            
        Returns:
            List of sentiment results
        """
        global _FINBERT_DISABLED
        
        # If FinBERT is disabled or pipeline not available, use fallback
        if _FINBERT_DISABLED or not self.pipeline:
            logger.info("Using fallback for batch sentiment processing")
            return [self._fallback_sentiment_result(text) for text in texts]
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Process batch
                batch_results = self.pipeline(
                    texts,
                    truncation=True,
                    max_length=self.MAX_LENGTH,
                    batch_size=len(texts)
                )
                
                processing_time = time.time() - start_time
                self._update_batch_performance(processing_time, had_error=False)
                
                return batch_results
                
            except Exception as e:
                processing_time = time.time() - start_time
                self._update_batch_performance(processing_time, had_error=True)
                
                if attempt < max_retries:
                    logger.warning(
                        f"Batch processing attempt {attempt + 1} failed, retrying: {str(e)}",
                        extra={
                            'event_type': 'batch_retry',
                            'attempt': attempt + 1,
                            'batch_size': len(texts),
                            'error': str(e)
                        }
                    )
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(
                        f"Batch processing failed after {max_retries + 1} attempts: {str(e)}",
                        extra={
                            'event_type': 'batch_failed',
                            'batch_size': len(texts),
                            'error': str(e)
                        }
                    )
                    raise
        
        return []  # Should never reach here


# Singleton instance
_sentiment_analyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create singleton sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer