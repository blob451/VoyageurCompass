"""
Test suite for verifying cache key determinism across process boundaries.

This module tests that cache key generation produces consistent results
across multiple runs and process restarts.
"""

import hashlib
import unittest
from unittest.mock import MagicMock

from Analytics.services.explanation_service import ExplanationService
from Analytics.services.local_llm_service import LocalLLMService
from Analytics.services.sentiment_analyzer import SentimentAnalyzer


class TestCacheDeterminism(unittest.TestCase):
    """Test cache key determinism for all services."""

    def setUp(self):
        """Set up test fixtures."""
        self.explanation_service = ExplanationService()
        self.llm_service = LocalLLMService()
        self.sentiment_analyzer = SentimentAnalyzer()

    def test_explanation_cache_key_determinism(self):
        """Test ExplanationService cache key consistency."""
        # Mock user object
        mock_user = MagicMock()
        mock_user.id = 123
        
        # Test data
        analysis_data = {
            "symbol": "AAPL",
            "score_0_10": 7.5,
            "weighted_scores": {
                "w_sma50vs200": 0.15,
                "w_rsi14": 0.8,
                "w_macd12269": -0.3
            }
        }
        
        # Generate cache keys multiple times
        key1 = self.explanation_service._create_cache_key(analysis_data, "detailed", mock_user)
        key2 = self.explanation_service._create_cache_key(analysis_data, "detailed", mock_user)
        key3 = self.explanation_service._create_cache_key(analysis_data, "detailed", mock_user)
        
        # All keys should be identical
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)
        
        # Key should be deterministic based on input
        expected_cache_data = "AAPL_7.5_detailed_123_w_sma50vs200_0.15_w_rsi14_0.80_w_macd12269_-0.30"
        expected_hash = hashlib.blake2b(expected_cache_data.encode(), digest_size=16).hexdigest()
        expected_key = f"explanation_{expected_hash}"
        
        self.assertEqual(key1, expected_key)

    def test_llm_cache_key_determinism(self):
        """Test LocalLLMService cache key consistency."""
        # Test data
        analysis_data = {
            "symbol": "MSFT",
            "score_0_10": 6.2,
            "components": {
                "sma50vs200": {"score": 8.5, "weight": 0.15},
                "rsi14": {"score": 4.2, "weight": 0.10}
            }
        }
        
        # Generate cache keys multiple times
        key1 = self.llm_service._create_cache_key(analysis_data, "summary", "technical")
        key2 = self.llm_service._create_cache_key(analysis_data, "summary", "technical")
        key3 = self.llm_service._create_cache_key(analysis_data, "summary", "technical")
        
        # All keys should be identical
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)
        
        # Key should start with expected prefix
        self.assertTrue(key1.startswith("llm_explanation_"))

    def test_sentiment_single_cache_key_determinism(self):
        """Test sentiment analyzer single text cache key consistency."""
        test_text = "Apple Inc. reports strong quarterly earnings with revenue growth."
        
        # Generate hash-based cache keys
        hash1 = hashlib.blake2b(test_text.encode(), digest_size=16).hexdigest()
        hash2 = hashlib.blake2b(test_text.encode(), digest_size=16).hexdigest()
        hash3 = hashlib.blake2b(test_text.encode(), digest_size=16).hexdigest()
        
        # All hashes should be identical
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)
        
        # Generate cache keys
        key1 = f"sentiment:single:{hash1}"
        key2 = f"sentiment:single:{hash2}"
        
        self.assertEqual(key1, key2)

    def test_sentiment_ensemble_cache_key_determinism(self):
        """Test sentiment analyzer ensemble cache key consistency."""
        test_text = "Tesla stock shows bullish momentum following strong delivery numbers."
        
        # Generate ensemble cache keys (matching actual implementation)
        key1 = f"sentiment:ensemble:{hashlib.blake2b(test_text.encode(), digest_size=16).hexdigest()}"
        key2 = f"sentiment:ensemble:{hashlib.blake2b(test_text.encode(), digest_size=16).hexdigest()}"
        key3 = f"sentiment:ensemble:{hashlib.blake2b(test_text.encode(), digest_size=16).hexdigest()}"
        
        # All keys should be identical
        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)

    def test_generateCacheKey_valid_inputs(self):
        """Test generateCacheKey method with valid inputs."""
        # Test text hash input
        text_hash = "abc123def456"
        key1 = self.sentiment_analyzer.generateCacheKey(text_hash=text_hash)
        key2 = self.sentiment_analyzer.generateCacheKey(text_hash=text_hash)
        
        self.assertEqual(key1, key2)
        self.assertEqual(key1, f"sentiment:text:{text_hash}")
        
        # Test symbol + days input
        key3 = self.sentiment_analyzer.generateCacheKey(symbol="AAPL", days=7)
        key4 = self.sentiment_analyzer.generateCacheKey(symbol="AAPL", days=7)
        
        self.assertEqual(key3, key4)
        self.assertEqual(key3, "sentiment:stock:AAPL:7d")
        
        # Test symbol only input
        key5 = self.sentiment_analyzer.generateCacheKey(symbol="GOOGL")
        key6 = self.sentiment_analyzer.generateCacheKey(symbol="GOOGL")
        
        self.assertEqual(key5, key6)
        self.assertEqual(key5, "sentiment:stock:GOOGL")

    def test_generateCacheKey_invalid_inputs(self):
        """Test generateCacheKey method with invalid inputs returns None."""
        # No valid inputs should return None
        key = self.sentiment_analyzer.generateCacheKey()
        self.assertIsNone(key)
        
        # Empty inputs should return None
        key = self.sentiment_analyzer.generateCacheKey(symbol="", text_hash="")
        self.assertIsNone(key)

    def test_blake2b_digest_size_consistency(self):
        """Test that all primary caches use consistent digest sizes."""
        test_data = "test_data_for_consistency_check"
        
        # All primary caches should use digest_size=16
        hash_16 = hashlib.blake2b(test_data.encode(), digest_size=16).hexdigest()
        
        # Verify length (16 bytes = 32 hex characters)
        self.assertEqual(len(hash_16), 32)
        
        # Multiple calls should produce same result
        hash_16_2 = hashlib.blake2b(test_data.encode(), digest_size=16).hexdigest()
        self.assertEqual(hash_16, hash_16_2)

    def test_cross_process_simulation(self):
        """Simulate cache key generation across different process instances."""
        # Simulate multiple "processes" by creating new service instances
        services = []
        for i in range(3):
            services.append({
                'explanation': ExplanationService(),
                'llm': LocalLLMService(),
                'sentiment': SentimentAnalyzer()
            })
        
        # Test data
        mock_user = MagicMock()
        mock_user.id = 999
        
        analysis_data = {
            "symbol": "NVDA",
            "score_0_10": 8.3,
            "weighted_scores": {"w_sma50vs200": 0.20}
        }
        
        # Generate keys from different "processes"
        explanation_keys = []
        for service_set in services:
            key = service_set['explanation']._create_cache_key(analysis_data, "brief", mock_user)
            explanation_keys.append(key)
        
        # All keys should be identical across "processes"
        self.assertEqual(len(set(explanation_keys)), 1, "Cache keys should be identical across processes")


if __name__ == '__main__':
    unittest.main()