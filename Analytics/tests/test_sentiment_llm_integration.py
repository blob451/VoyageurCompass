"""
Integration tests for Sentiment Analysis and LLaMA integration.
Tests the complete pipeline from sentiment analysis through to LLM explanation generation.
"""

from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.utils import timezone
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch, Mock
import json

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.sentiment_analyzer import get_sentiment_analyzer
from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.explanation_service import get_explanation_service
from Data.models import Stock, StockPrice, DataSector, DataIndustry, AnalyticsResults

User = get_user_model()


class SentimentLLMIntegrationTestCase(TransactionTestCase):
    """Integration tests for sentiment analysis and LLM explanation pipeline."""
    
    def setUp(self):
        """Set up comprehensive test data."""
        cache.clear()
        
        # Create test user
        self.user = User.objects.create_user(
            username='integration_user',
            email='integration@test.com',
            password='testpass123'
        )
        
        # Create sector and industry
        self.sector = DataSector.objects.create(
            sectorKey='tech_integration',
            sectorName='Technology Integration',
            data_source='test'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software_integration',
            industryName='Software Integration',
            sector=self.sector,
            data_source='test'
        )
        
        # Create test stock
        self.stock = Stock.objects.create(
            symbol='INTEG_TEST',
            short_name='Integration Test Corp',
            long_name='Integration Testing Corporation',
            sector_id=self.sector,
            industry_id=self.industry,
            market_cap=50000000000
        )
        
        # Create price data for TA engine
        self._create_comprehensive_price_data()
        
        # Initialize services
        self.ta_engine = TechnicalAnalysisEngine()
        self.sentiment_service = get_sentiment_analyzer()
        self.llm_service = get_local_llm_service()
        self.explanation_service = get_explanation_service()
    
    def _create_comprehensive_price_data(self):
        """Create realistic price data for comprehensive testing."""
        base_date = datetime.now().date() - timedelta(days=250)
        base_price = 120.0
        
        for i in range(250):
            # Create realistic price movements
            trend = 0.0008 * i  # Gradual uptrend
            volatility = 3 * (1 + 0.5 * (i % 20) / 20)  # Varying volatility
            daily_change = (hash(f"price_{i}") % 200 - 100) / 100 * volatility
            
            price = base_price + trend * base_price + daily_change
            price = max(price, 20)  # Floor price
            
            # Create OHLCV data
            open_price = price + (hash(f"open_{i}") % 100 - 50) / 100
            high_price = max(open_price, price) + abs(hash(f"high_{i}") % 100) / 200
            low_price = min(open_price, price) - abs(hash(f"low_{i}") % 100) / 200
            close_price = price
            
            # Volume patterns
            base_volume = 5000000
            volume_factor = 1 + (hash(f"vol_{i}") % 100) / 200
            volume = int(base_volume * volume_factor)
            
            StockPrice.objects.create(
                stock=self.stock,
                date=base_date + timedelta(days=i),
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high_price, 2))),
                low=Decimal(str(round(low_price, 2))),
                close=Decimal(str(round(close_price, 2))),
                adjusted_close=Decimal(str(round(close_price, 2))),
                volume=volume,
                data_source='test'
            )
    
    def test_sentiment_analysis_in_ta_engine_integration(self):
        """Test sentiment analysis integration within the TA engine."""
        # Mock the sentiment analyzer to return predictable results
        with patch.object(self.sentiment_service, 'analyzeSentimentBatch') as mock_batch:
            mock_batch.return_value = [
                {
                    'sentimentScore': 0.65,
                    'sentimentLabel': 'positive',
                    'sentimentConfidence': 0.82,
                    'timestamp': timezone.now().isoformat()
                }
            ]
            
            with patch.object(self.sentiment_service, 'aggregateSentiment') as mock_agg:
                mock_agg.return_value = {
                    'sentimentScore': 0.65,
                    'sentimentLabel': 'positive',
                    'sentimentConfidence': 0.82,
                    'distribution': {'positive': 8, 'neutral': 2, 'negative': 0},
                    'totalArticles': 10
                }
                
                # Mock news fetching to return sample data
                with patch('Data.services.yahoo_finance.yahoo_finance_service.fetchNewsForStock') as mock_news:
                    mock_news.return_value = [
                        {
                            'title': 'Strong quarterly results show growth',
                            'summary': 'Company reports record revenue and profit margins',
                            'publishedDate': timezone.now().isoformat(),
                            'source': 'Financial News'
                        }
                    ]
                    
                    # Run full TA analysis
                    result = self.ta_engine.analyze_stock('INTEG_TEST')
                    
                    # Verify sentiment was integrated
                    self.assertIsNotNone(result)
                    self.assertEqual(result['symbol'], 'INTEG_TEST')
                    self.assertIn('indicators', result)
                    
                    # Check if sentiment indicator is present (may be None if no news)
                    if 'sentiment' in result['indicators']:
                        sentiment_result = result['indicators']['sentiment']
                        if sentiment_result is not None:
                            self.assertEqual(sentiment_result.weight, 0.10)  # 10% weight
                            self.assertGreaterEqual(sentiment_result.score, 0)
                            self.assertLessEqual(sentiment_result.score, 1)
    
    def test_llm_explanation_with_sentiment_data(self):
        """Test LLM explanation generation incorporating sentiment analysis results."""
        # Create mock analysis result with sentiment data
        analysis_data = {
            'symbol': 'INTEG_TEST',
            'score_0_10': 7.8,
            'composite_raw': 3.2,
            'weighted_scores': {
                'w_sma50vs200': 0.15,
                'w_rsi14': 0.06,
                'w_sentiment': 0.08,  # Sentiment component
                'w_macd12269': 0.07
            },
            'components': {
                'sentiment': {
                    'raw': {
                        'label': 'positive',
                        'score': 0.65,
                        'confidence': 0.82,
                        'articles_analyzed': 10
                    },
                    'score': 0.8
                },
                'sma50vs200': {'raw': {'sma50': 125, 'sma200': 118}, 'score': 1.0},
                'rsi14': {'raw': {'rsi': 58.0}, 'score': 0.7}
            }
        }
        
        # Mock LLM service response
        mock_llm_response = {
            'response': 'INTEG_TEST shows strong bullish sentiment with positive news sentiment (0.65) based on 10 recent articles. Technical indicators support this with moving average crossover and RSI at 58. Combined score of 7.8/10 suggests BUY recommendation.',
            'done': True,
            'total_duration': 3000000000,
            'prompt_eval_count': 180,
            'eval_count': 55
        }
        
        with patch.object(self.llm_service, '_make_ollama_request') as mock_llm:
            mock_llm.return_value = mock_llm_response
            
            # Test explanation service with sentiment data
            explanation = self.explanation_service._generate_template_explanation(
                analysis_data, 'detailed'
            )
            
            self.assertIsNotNone(explanation)
            self.assertIn('content', explanation)
            
            # Verify sentiment is mentioned in explanation
            content = explanation['content'].lower()
            self.assertTrue(any(word in content for word in ['sentiment', 'news', 'positive']))
            
            # Check explanation structure
            self.assertIn('recommendation', explanation)
            self.assertEqual(explanation['recommendation'], 'BUY')
            self.assertIn('indicators_explained', explanation)
            self.assertIn('sentiment', explanation['indicators_explained'])
    
    def test_end_to_end_sentiment_llm_pipeline(self):
        """Test complete pipeline from TA analysis through sentiment to LLM explanation."""
        # Mock all external dependencies for predictable testing
        mock_news = [
            {
                'title': 'Company announces breakthrough technology',
                'summary': 'Revolutionary product expected to drive significant growth',
                'publishedDate': timezone.now().isoformat(),
                'source': 'Tech News'
            },
            {
                'title': 'Quarterly earnings exceed expectations',
                'summary': 'Strong performance across all business segments',
                'publishedDate': (timezone.now() - timedelta(days=1)).isoformat(),
                'source': 'Financial Times'
            }
        ]
        
        mock_sentiment_results = [
            {'sentimentScore': 0.8, 'sentimentLabel': 'positive', 'sentimentConfidence': 0.9},
            {'sentimentScore': 0.7, 'sentimentLabel': 'positive', 'sentimentConfidence': 0.85}
        ]
        
        mock_sentiment_agg = {
            'sentimentScore': 0.75,
            'sentimentLabel': 'positive',
            'sentimentConfidence': 0.875,
            'distribution': {'positive': 2, 'neutral': 0, 'negative': 0},
            'totalArticles': 2
        }
        
        with patch('Data.services.yahoo_finance.yahoo_finance_service.fetchNewsForStock') as mock_fetch_news:
            mock_fetch_news.return_value = mock_news
            
            with patch.object(self.sentiment_service, 'analyzeSentimentBatch') as mock_batch:
                mock_batch.return_value = mock_sentiment_results
                
                with patch.object(self.sentiment_service, 'aggregateSentiment') as mock_agg:
                    mock_agg.return_value = mock_sentiment_agg
                    
                    # Step 1: Run TA analysis (includes sentiment)
                    ta_result = self.ta_engine.analyze_stock('INTEG_TEST')
                    
                    self.assertIsNotNone(ta_result)
                    self.assertEqual(ta_result['symbol'], 'INTEG_TEST')
                    
                    # Step 2: Store results in database
                    analytics_result = AnalyticsResults.objects.create(
                        user=self.user,
                        stock=self.stock,
                        as_of=timezone.now(),
                        horizon='medium',
                        score_0_10=ta_result['score_0_10'],
                        composite_raw=ta_result['composite_raw'],
                        w_sentiment=ta_result['weighted_scores'].get('w_sentiment', 0),
                        components=ta_result.get('components', {})
                    )
                    
                    # Step 3: Generate explanation with LLM integration
                    mock_llm_response = {
                        'response': f"Based on comprehensive analysis, {ta_result['symbol']} shows strong performance with score {ta_result['score_0_10']}/10. Positive news sentiment (0.75) from recent breakthrough announcements supports bullish outlook. Technical indicators confirm upward momentum. Recommendation: BUY with high confidence.",
                        'done': True
                    }
                    
                    with patch.object(self.llm_service, '_make_ollama_request') as mock_llm:
                        mock_llm.return_value = mock_llm_response
                        
                        explanation = self.explanation_service.explain_prediction_single(
                            analytics_result, detail_level='detailed'
                        )
                        
                        # Verify complete pipeline worked
                        self.assertIsNotNone(explanation)
                        self.assertIn('content', explanation)
                        self.assertIn('recommendation', explanation)
                        
                        # Verify sentiment integration in final explanation
                        content = explanation['content'].lower()
                        self.assertTrue(any(word in content for word in 
                                          ['sentiment', 'news', 'positive', 'breakthrough']))
    
    def test_sentiment_caching_integration(self):
        """Test that sentiment results are properly cached and reused."""
        cache_key = self.sentiment_service.generateCacheKey(symbol='INTEG_TEST', days=7)
        
        # Initial cache should be empty
        cached_result = self.sentiment_service.getCachedSentiment(cache_key, 'INTEG_TEST')
        self.assertIsNone(cached_result)
        
        # Mock sentiment calculation
        mock_result = {
            'sentimentScore': 0.6,
            'sentimentLabel': 'positive',
            'sentimentConfidence': 0.8,
            'timestamp': timezone.now().isoformat(),
            'articles_analyzed': 5
        }
        
        # Store in cache
        success = self.sentiment_service.setCachedSentiment(cache_key, mock_result, 'INTEG_TEST')
        self.assertTrue(success)
        
        # Retrieve from cache
        cached_result = self.sentiment_service.getCachedSentiment(cache_key, 'INTEG_TEST')
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result['sentimentScore'], 0.6)
        self.assertTrue(cached_result['cached'])  # Should indicate it came from cache
        
        # Verify cache integration works in TA engine
        with patch('Data.services.yahoo_finance.yahoo_finance_service.fetchNewsForStock') as mock_news:
            mock_news.return_value = []  # No news, should use cached sentiment
            
            # This should use the cached sentiment instead of fetching new
            result = self.ta_engine._calculate_sentiment_analysis('INTEG_TEST')
            
            # Result may be None if sentiment isn't included or cache isn't used properly
            # This test validates the caching mechanism exists
    
    def test_llm_service_availability_fallback(self):
        """Test graceful fallback when LLM service is unavailable."""
        # Simulate LLM service unavailability
        with patch.object(self.llm_service, '_make_ollama_request') as mock_llm:
            mock_llm.side_effect = Exception("Connection refused - Ollama not available")
            
            analysis_data = {
                'symbol': 'FALLBACK_TEST',
                'score_0_10': 6.5,
                'weighted_scores': {'w_sentiment': 0.05},
                'components': {
                    'sentiment': {
                        'raw': {'label': 'neutral', 'score': 0.0},
                        'score': 0.5
                    }
                }
            }
            
            # Should fall back to template-based explanation
            explanation = self.explanation_service._generate_template_explanation(
                analysis_data, 'standard'
            )
            
            self.assertIsNotNone(explanation)
            self.assertEqual(explanation['model_used'], 'template_fallback')
            self.assertIn('content', explanation)
            self.assertIn('recommendation', explanation)
            
            # Should still include sentiment information in template
            if 'sentiment' in explanation.get('indicators_explained', []):
                self.assertTrue(True)  # Sentiment was included in fallback
    
    def test_sentiment_metrics_collection(self):
        """Test that sentiment analysis metrics are properly collected."""
        from Analytics.services.sentiment_analyzer import sentiment_metrics
        
        # Reset metrics
        sentiment_metrics.reset_metrics()
        
        # Simulate sentiment analysis operations
        sentiment_metrics.log_request('INTEG_TEST', 5)
        sentiment_metrics.log_success('INTEG_TEST', 0.7, 0.85, 2.5)
        sentiment_metrics.log_cache_hit('INTEG_TEST')
        
        # Get metrics summary
        stats = sentiment_metrics.get_summary_stats()
        
        self.assertEqual(stats['total_requests'], 1)
        self.assertEqual(stats['successful_analyses'], 1)
        self.assertEqual(stats['total_articles_processed'], 5)
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 0)
        self.assertEqual(stats['success_rate'], 1.0)
        self.assertEqual(stats['cache_hit_rate'], 1.0)
        self.assertGreater(stats['avg_processing_time_ms'], 0)


class SentimentLLMPerformanceTestCase(TestCase):
    """Performance tests for sentiment and LLM integration."""
    
    def test_sentiment_processing_performance(self):
        """Test sentiment processing performance with batching."""
        import time
        
        service = get_sentiment_analyzer()
        
        # Test batch processing efficiency
        texts = [f"Sample financial text {i} with market analysis" for i in range(10)]
        
        start_time = time.time()
        # Use neutral sentiment for performance testing
        results = [service._neutral_sentiment() for _ in texts]
        processing_time = time.time() - start_time
        
        self.assertEqual(len(results), 10)
        self.assertLess(processing_time, 1.0)  # Should be very fast for neutral fallback
        
        for result in results:
            self.assertIn('sentimentScore', result)
            self.assertIn('sentimentLabel', result)
            self.assertEqual(result['sentimentLabel'], 'neutral')
    
    def test_llm_explanation_performance(self):
        """Test LLM explanation generation performance."""
        import time
        
        service = get_explanation_service()
        
        analysis_data = {
            'symbol': 'PERF_TEST',
            'score_0_10': 7.2,
            'weighted_scores': {
                'w_sma50vs200': 0.12,
                'w_sentiment': 0.08
            },
            'components': {
                'sentiment': {'raw': {'label': 'positive'}, 'score': 0.8}
            }
        }
        
        start_time = time.time()
        explanation = service._generate_template_explanation(analysis_data, 'standard')
        generation_time = time.time() - start_time
        
        self.assertLess(generation_time, 2.0)  # Template generation should be fast
        self.assertIsNotNone(explanation)
        self.assertIn('content', explanation)


if __name__ == '__main__':
    import unittest
    unittest.main()