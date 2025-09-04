"""
Test Utilities for Analytics Testing Infrastructure.
Provides helper functions, fixtures, and utilities for comprehensive testing.
"""

import json
import time
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.core.cache import cache
from pathlib import Path

User = get_user_model()


class AnalyticsTestCase(TestCase):
    """Enhanced base test case for Analytics tests with common utilities."""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_user = None
    
    def setUp(self):
        super().setUp()
        # Clear cache before each test
        cache.clear()
        
        # Create test user if needed
        if not self.test_user:
            self.test_user = User.objects.create_user(
                username='testuser',
                email='test@example.com',
                password='testpass123'
            )
    
    def create_mock_analysis_data(self, symbol: str = 'TEST', score: float = 7.5) -> Dict[str, Any]:
        """Create mock analysis data for testing."""
        return {
            'symbol': symbol,
            'score_0_10': score,
            'weighted_scores': {
                'w_sma50vs200': 0.15,
                'w_rsi14': 0.08,
                'w_macd12269': 0.12,
                'w_bbpos20': 0.06,
                'w_volsurge': 0.10,
                'w_obv20': 0.05,
                'w_rel1y': 0.07,
                'w_candlerev': 0.02
            },
            'components': {
                'sma_50': 150.25,
                'sma_200': 145.80,
                'rsi_14': 65.4,
                'macd_line': 2.35,
                'macd_signal': 1.98,
                'bb_upper': 155.0,
                'bb_lower': 140.0,
                'current_price': 152.30
            },
            'news_articles': [
                {
                    'title': f'{symbol} Reports Strong Q3 Earnings',
                    'summary': f'{symbol} exceeded expectations with revenue growth and positive guidance.',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }
    
    def create_mock_sentiment_data(self, 
                                  sentiment_score: float = 0.7, 
                                  confidence: float = 0.85,
                                  label: str = 'positive') -> Dict[str, Any]:
        """Create mock sentiment data for testing."""
        return {
            'sentimentScore': sentiment_score,
            'sentimentConfidence': confidence,
            'sentimentLabel': label,
            'newsCount': 2,
            'timestamp': datetime.now().isoformat(),
            'articles_analyzed': [
                {'title': 'Positive news article', 'sentiment': 0.8},
                {'title': 'Another positive article', 'sentiment': 0.6}
            ]
        }
    
    def create_mock_llm_response(self, content: str = None) -> Dict[str, Any]:
        """Create mock LLM response for testing."""
        if content is None:
            content = "Based on technical analysis, TEST shows strong BUY signals with RSI indicating bullish momentum."
        
        return {
            'content': content,
            'generation_time': 2.3,
            'model_used': 'llama3.1:8b',
            'word_count': len(content.split()),
            'confidence_score': 0.85,
            'timestamp': time.time()
        }
    
    def assertAnalysisDataValid(self, analysis_data: Dict[str, Any]):
        """Assert that analysis data has required structure."""
        self.assertIn('symbol', analysis_data)
        self.assertIn('score_0_10', analysis_data)
        self.assertIn('weighted_scores', analysis_data)
        
        score = analysis_data['score_0_10']
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 10)
    
    def assertSentimentDataValid(self, sentiment_data: Dict[str, Any]):
        """Assert that sentiment data has required structure."""
        self.assertIn('sentimentScore', sentiment_data)
        self.assertIn('sentimentConfidence', sentiment_data)
        self.assertIn('sentimentLabel', sentiment_data)
        
        score = sentiment_data['sentimentScore']
        confidence = sentiment_data['sentimentConfidence']
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def assertLLMResponseValid(self, llm_response: Dict[str, Any]):
        """Assert that LLM response has required structure."""
        self.assertIn('content', llm_response)
        self.assertIn('generation_time', llm_response)
        
        content = llm_response['content']
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 10)  # Minimum content length
    
    def wait_for_condition(self, condition_func, timeout: float = 5.0, interval: float = 0.1) -> bool:
        """Wait for a condition to become true with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False


class MockServiceManager:
    """Manager for creating and coordinating mock services."""
    
    def __init__(self):
        self.mocks = {}
        self.patches = {}
    
    def mock_llm_service(self, responses: List[Dict[str, Any]] = None):
        """Mock the LLM service with predefined responses."""
        if responses is None:
            responses = [
                {
                    'content': 'Strong BUY recommendation based on technical indicators.',
                    'generation_time': 1.5,
                    'model_used': 'llama3.1:8b'
                }
            ]
        
        mock_service = MagicMock()
        mock_service.generate_explanation.side_effect = responses
        mock_service.generate_sentiment_aware_explanation.side_effect = responses
        mock_service.is_available.return_value = True
        
        self.mocks['llm_service'] = mock_service
        return mock_service
    
    def mock_sentiment_service(self, sentiment_responses: List[Dict[str, Any]] = None):
        """Mock the sentiment analyzer service."""
        if sentiment_responses is None:
            sentiment_responses = [
                {
                    'sentimentScore': 0.6,
                    'sentimentConfidence': 0.8,
                    'sentimentLabel': 'positive',
                    'newsCount': 2
                }
            ]
        
        mock_service = MagicMock()
        mock_service.analyzeNewsArticles.side_effect = sentiment_responses
        mock_service.analyzeSentimentSingle.side_effect = sentiment_responses
        
        self.mocks['sentiment_service'] = mock_service
        return mock_service
    
    def mock_hybrid_coordinator(self, enhanced_responses: List[Dict[str, Any]] = None):
        """Mock the hybrid analysis coordinator."""
        if enhanced_responses is None:
            enhanced_responses = [
                {
                    'content': 'Enhanced explanation with sentiment integration.',
                    'generation_time': 2.1,
                    'sentiment_enhanced': True,
                    'hybrid_coordination': {
                        'sentiment_integration_success': True,
                        'quality_metrics': {'has_recommendation': True}
                    }
                }
            ]
        
        mock_coordinator = MagicMock()
        mock_coordinator.generate_enhanced_explanation.side_effect = enhanced_responses
        
        self.mocks['hybrid_coordinator'] = mock_coordinator
        return mock_coordinator
    
    def start_patches(self):
        """Start all mock patches."""
        if 'llm_service' in self.mocks:
            patch_path = 'Analytics.services.local_llm_service.get_local_llm_service'
            self.patches['llm_service'] = patch(patch_path, return_value=self.mocks['llm_service'])
            self.patches['llm_service'].start()
        
        if 'sentiment_service' in self.mocks:
            patch_path = 'Analytics.services.sentiment_analyzer.get_sentiment_analyzer'
            self.patches['sentiment_service'] = patch(patch_path, return_value=self.mocks['sentiment_service'])
            self.patches['sentiment_service'].start()
        
        if 'hybrid_coordinator' in self.mocks:
            patch_path = 'Analytics.services.hybrid_analysis_coordinator.get_hybrid_analysis_coordinator'
            self.patches['hybrid_coordinator'] = patch(patch_path, return_value=self.mocks['hybrid_coordinator'])
            self.patches['hybrid_coordinator'].start()
    
    def stop_patches(self):
        """Stop all mock patches."""
        for patch_obj in self.patches.values():
            patch_obj.stop()
        self.patches.clear()


class TestDataGenerator:
    """Generates test data for various testing scenarios."""
    
    @staticmethod
    def generate_stock_symbols(count: int = 10) -> List[str]:
        """Generate list of test stock symbols."""
        base_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
        if count <= len(base_symbols):
            return base_symbols[:count]
        
        # Generate additional symbols if needed
        additional = [f'TEST{i:03d}' for i in range(count - len(base_symbols))]
        return base_symbols + additional
    
    @staticmethod
    def generate_analysis_batch(symbols: List[str], score_range: tuple = (3.0, 8.0)) -> List[Dict[str, Any]]:
        """Generate batch of analysis data for multiple symbols."""
        import random
        
        batch_data = []
        for symbol in symbols:
            score = random.uniform(score_range[0], score_range[1])
            analysis_data = {
                'symbol': symbol,
                'score_0_10': round(score, 2),
                'weighted_scores': {
                    'w_sma50vs200': random.uniform(-0.3, 0.3),
                    'w_rsi14': random.uniform(-0.15, 0.15),
                    'w_macd12269': random.uniform(-0.2, 0.2),
                    'w_bbpos20': random.uniform(-0.1, 0.1),
                    'w_volsurge': random.uniform(-0.15, 0.15)
                },
                'components': {
                    'current_price': round(random.uniform(50, 300), 2),
                    'rsi_14': random.uniform(20, 80),
                    'macd_line': random.uniform(-5, 5)
                }
            }
            batch_data.append(analysis_data)
        
        return batch_data
    
    @staticmethod
    def generate_training_samples(count: int = 100, quality_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Generate training samples for fine-tuning tests."""
        import random
        
        samples = []
        symbols = TestDataGenerator.generate_stock_symbols(count // 10 + 1)
        
        for i in range(count):
            symbol = random.choice(symbols)
            score = random.uniform(1.0, 10.0)
            detail_level = random.choice(['summary', 'standard', 'detailed'])
            
            # Generate realistic content based on score
            if score >= 7.5:
                recommendation = 'BUY'
                sentiment = 'bullish'
            elif score <= 3.5:
                recommendation = 'SELL'
                sentiment = 'bearish'
            else:
                recommendation = 'HOLD'
                sentiment = 'neutral'
            
            content = f"Based on technical analysis, {symbol} shows {sentiment} signals with a score of {score:.1f}/10. Recommendation: {recommendation}. Key indicators include RSI and MACD analysis supporting this outlook."
            
            sample = {
                'instruction': f"Provide investment analysis for {symbol}",
                'input': {
                    'symbol': symbol,
                    'score_0_10': score,
                    'detail_level': detail_level
                },
                'output': content,
                'quality_score': random.uniform(quality_threshold, 1.0),
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'sample_id': f"{symbol}_{detail_level}_{i}"
                }
            }
            samples.append(sample)
        
        return samples


class TestFileManager:
    """Manages temporary test files and datasets."""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dir = None
    
    def create_temp_dir(self) -> Path:
        """Create temporary directory for test files."""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='analytics_test_'))
        return self.temp_dir
    
    def create_test_dataset(self, samples: List[Dict[str, Any]], filename: str = None) -> str:
        """Create test dataset file."""
        if filename is None:
            filename = f'test_dataset_{int(time.time())}.json'
        
        temp_dir = self.create_temp_dir()
        dataset_path = temp_dir / filename
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)
        
        self.temp_files.append(dataset_path)
        return str(dataset_path)
    
    def create_test_metadata(self, dataset_path: str, metadata: Dict[str, Any] = None) -> str:
        """Create metadata file for test dataset."""
        if metadata is None:
            metadata = {
                'created_at': datetime.now().isoformat(),
                'total_samples': 100,
                'generation_time': 30.5,
                'quality_threshold': 0.7
            }
        
        metadata_path = Path(dataset_path).with_suffix('.meta.json')
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.temp_files.append(metadata_path)
        return str(metadata_path)
    
    def cleanup(self):
        """Clean up all temporary files."""
        import shutil
        
        for file_path in self.temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
        
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
        
        self.temp_files.clear()
        self.temp_dir = None


class PerformanceTestMixin:
    """Mixin for performance testing utilities."""
    
    def assertResponseTime(self, func, max_time: float = 5.0, *args, **kwargs):
        """Assert that function completes within max_time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        self.assertLessEqual(
            duration, max_time,
            f"Function took {duration:.2f}s, expected <= {max_time}s"
        )
        return result
    
    def measure_performance(self, func, iterations: int = 10, *args, **kwargs) -> Dict[str, float]:
        """Measure performance statistics for function."""
        durations = []
        
        for _ in range(iterations):
            start_time = time.time()
            func(*args, **kwargs)
            duration = time.time() - start_time
            durations.append(duration)
        
        return {
            'min_time': min(durations),
            'max_time': max(durations),
            'avg_time': sum(durations) / len(durations),
            'total_time': sum(durations),
            'iterations': iterations
        }


class IntegrationTestMixin:
    """Mixin for integration testing utilities."""
    
    def setUp_integration(self):
        """Set up integration test environment."""
        # Clear all caches
        cache.clear()
        
        # Reset singleton services if needed
        self.reset_singleton_services()
    
    def reset_singleton_services(self):
        """Reset singleton service instances for clean testing."""
        # Reset monitoring service
        import Analytics.services.advanced_monitoring_service
        Analytics.services.advanced_monitoring_service._monitoring_service = None
        
        # Reset async pipeline
        import Analytics.services.async_processing_pipeline
        Analytics.services.async_processing_pipeline._async_pipeline = None
        
        # Reset fine-tuning manager
        import Analytics.services.enhanced_finetuning_service
        Analytics.services.enhanced_finetuning_service._finetuning_manager = None
    
    def assert_service_integration(self, service1, service2, interaction_func):
        """Assert that two services integrate correctly."""
        # Test service 1 can call service 2
        try:
            result = interaction_func(service1, service2)
            self.assertIsNotNone(result)
            return result
        except Exception as e:
            self.fail(f"Service integration failed: {str(e)}")


# Context managers for testing
class mock_services:
    """Context manager for mocking multiple services."""
    
    def __init__(self, **service_configs):
        self.service_configs = service_configs
        self.manager = MockServiceManager()
        
    def __enter__(self):
        for service_name, config in self.service_configs.items():
            if service_name == 'llm':
                self.manager.mock_llm_service(config.get('responses'))
            elif service_name == 'sentiment':
                self.manager.mock_sentiment_service(config.get('responses'))
            elif service_name == 'hybrid':
                self.manager.mock_hybrid_coordinator(config.get('responses'))
        
        self.manager.start_patches()
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.stop_patches()


class temp_test_files:
    """Context manager for temporary test files."""
    
    def __init__(self):
        self.file_manager = TestFileManager()
    
    def __enter__(self):
        return self.file_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_manager.cleanup()


# Test fixtures and data
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

SAMPLE_ANALYSIS_DATA = {
    'symbol': 'SAMPLE',
    'score_0_10': 7.2,
    'weighted_scores': {
        'w_sma50vs200': 0.18,
        'w_rsi14': 0.09,
        'w_macd12269': 0.14,
        'w_bbpos20': 0.05,
        'w_volsurge': 0.08
    },
    'components': {
        'sma_50': 155.0,
        'sma_200': 150.0,
        'rsi_14': 68.5,
        'macd_line': 2.1,
        'current_price': 157.25
    }
}

SAMPLE_SENTIMENT_DATA = {
    'sentimentScore': 0.65,
    'sentimentConfidence': 0.82,
    'sentimentLabel': 'positive',
    'newsCount': 3,
    'timestamp': datetime.now().isoformat()
}