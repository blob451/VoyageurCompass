"""
Unit tests for Analytics Local LLM Service.
Tests LLaMA 3.1 integration via Ollama, explanation generation, and performance optimization.
"""

import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase
from django.core.cache import cache
from django.test.utils import override_settings
import json

from Analytics.services.local_llm_service import LocalLLMService, get_local_llm_service


class LocalLLMServiceTestCase(TestCase):
    """Test cases for LocalLLMService functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.service = LocalLLMService()
        
        # Mock analysis data
        self.mock_analysis_data = {
            'symbol': 'TEST',
            'score_0_10': 7,
            'components': {
                'rsi14': 0.65,
                'sma50vs200': 0.8,
                'macd12269': 0.55,
                'bollinger_bands': 0.4
            },
            'weighted_scores': {
                'w_rsi14': 0.13,
                'w_sma50vs200': 0.16,
                'w_macd12269': 0.11,
                'w_bollinger_bands': 0.08
            }
        }
        
        # Mock Ollama response
        self.mock_ollama_response = {
            'response': 'TEST shows strong bullish indicators with a score of 7/10. The moving averages and RSI suggest upward momentum. Recommendation: BUY with moderate confidence due to mixed signals from Bollinger Bands.',
            'done': True,
            'total_duration': 2500000000,  # 2.5 seconds in nanoseconds
            'load_duration': 500000000,
            'prompt_eval_count': 150,
            'eval_count': 45
        }
    
    def test_service_initialization(self):
        """Test LocalLLMService initialization."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            with patch.object(LocalLLMService, '_verify_model_availability') as mock_verify:
                mock_verify.return_value = True
                
                service = LocalLLMService()
                
                self.assertEqual(service.primary_model, "llama3.1:8b")
                self.assertEqual(service.detailed_model, "llama3.1:70b")
                self.assertEqual(service.current_model, service.primary_model)
                self.assertTrue(service.performance_mode)
                self.assertEqual(service.generation_timeout, 45)
    
    @patch('Analytics.services.local_llm_service.Client')
    def test_model_availability_check(self, mock_client):
        """Test model availability verification."""
        # Mock Ollama client response
        mock_client_instance = mock_client.return_value
        mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama3.1:8b'},
                {'name': 'llama3.1:70b'},
                {'name': 'other_model'}
            ]
        }
        
        service = LocalLLMService()
        service.client = mock_client_instance
        
        # Test primary model availability
        self.assertTrue(service._verify_model_availability('llama3.1:8b'))
        
        # Test detailed model availability
        self.assertTrue(service._verify_model_availability('llama3.1:70b'))
        
        # Test non-existent model
        self.assertFalse(service._verify_model_availability('non_existent_model'))
    
    def test_optimal_model_selection(self):
        """Test optimal model selection based on detail level."""
        with patch.object(self.service, '_verify_model_availability') as mock_verify:
            # Both models available
            mock_verify.return_value = True
            
            # Should always prefer 8B model for performance
            self.assertEqual(self.service._select_optimal_model('summary'), 'llama3.1:8b')
            self.assertEqual(self.service._select_optimal_model('standard'), 'llama3.1:8b')
            self.assertEqual(self.service._select_optimal_model('detailed'), 'llama3.1:8b')
    
    def test_optimal_model_selection_fallback(self):
        """Test model selection with fallback logic."""
        def mock_verify_side_effect(model):
            if model == 'llama3.1:8b':
                return False  # Primary model unavailable
            elif model == 'llama3.1:70b':
                return True   # Detailed model available
            return False
        
        with patch.object(self.service, '_verify_model_availability', side_effect=mock_verify_side_effect):
            # Should fallback to detailed model when primary unavailable
            self.assertEqual(self.service._select_optimal_model('summary'), 'llama3.1:70b')
    
    @patch('Analytics.services.local_llm_service.Client')
    def test_generate_explanation_success(self, mock_client):
        """Test successful explanation generation."""
        mock_client_instance = mock_client.return_value
        mock_client_instance.generate.return_value = self.mock_ollama_response
        mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
        
        service = LocalLLMService()
        service.client = mock_client_instance
        
        with patch.object(service, '_verify_model_availability', return_value=True):
            result = service.generate_explanation(
                self.mock_analysis_data,
                detail_level='standard'
            )
            
            self.assertIsNotNone(result)
            self.assertIn('content', result)
            self.assertIn('generation_time', result)
            self.assertIn('confidence_score', result)
            self.assertIn('model_used', result)
            self.assertEqual(result['detail_level'], 'standard')
            self.assertEqual(result['model_used'], 'llama3.1:8b')
    
    def test_generate_explanation_different_detail_levels(self):
        """Test explanation generation with different detail levels."""
        detail_levels = ['summary', 'standard', 'detailed']
        
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.generate.return_value = self.mock_ollama_response
            mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
            
            service = LocalLLMService()
            service.client = mock_client_instance
            
            with patch.object(service, '_verify_model_availability', return_value=True):
                for detail_level in detail_levels:
                    result = service.generate_explanation(
                        self.mock_analysis_data,
                        detail_level=detail_level
                    )
                    
                    self.assertIsNotNone(result)
                    self.assertEqual(result['detail_level'], detail_level)
    
    def test_generate_explanation_service_unavailable(self):
        """Test explanation generation when service is unavailable."""
        with patch.object(self.service, 'is_available', return_value=False):
            result = self.service.generate_explanation(self.mock_analysis_data)
            
            self.assertIsNone(result)
    
    def test_generate_explanation_ollama_error(self):
        """Test handling of Ollama generation errors."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.generate.side_effect = Exception("Ollama connection error")
            mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
            
            service = LocalLLMService()
            service.client = mock_client_instance
            
            with patch.object(service, '_verify_model_availability', return_value=True):
                result = service.generate_explanation(self.mock_analysis_data)
                
                self.assertIsNone(result)
    
    def test_prompt_building(self):
        """Test LLaMA prompt construction."""
        prompt = self.service._build_prompt(
            self.mock_analysis_data,
            detail_level='standard',
            explanation_type='technical_analysis'
        )
        
        self.assertIn('TEST', prompt)
        self.assertIn('7/10', prompt)
        self.assertIn('Key Indicators', prompt)
        self.assertIn('Analysis:', prompt)
        
        # Should contain top weighted indicators
        self.assertIn('w_sma50vs200', prompt)
        self.assertIn('w_rsi14', prompt)
    
    def test_prompt_building_different_detail_levels(self):
        """Test prompt construction for different detail levels."""
        detail_levels = ['summary', 'standard', 'detailed']
        
        for detail_level in detail_levels:
            prompt = self.service._build_prompt(
                self.mock_analysis_data,
                detail_level=detail_level,
                explanation_type='technical_analysis'
            )
            
            self.assertIn('TEST', prompt)
            
            if detail_level == 'summary':
                self.assertIn('1-2 sentences', prompt)
            elif detail_level == 'detailed':
                self.assertIn('investment recommendation', prompt)
                self.assertIn('Key supporting indicators', prompt)
    
    def test_generation_options_optimization(self):
        """Test generation options optimization."""
        # Test 8B model options
        options_8b = self.service._get_generation_options('standard', 'llama3.1:8b')
        
        self.assertEqual(options_8b['num_ctx'], 1024)
        self.assertEqual(options_8b['temperature'], 0.6)
        self.assertIn('stop', options_8b)
        
        # Test 70B model options
        options_70b = self.service._get_generation_options('standard', 'llama3.1:70b')
        
        self.assertEqual(options_70b['num_ctx'], 2048)
        self.assertEqual(options_70b['temperature'], 0.4)
    
    def test_max_tokens_by_detail_level(self):
        """Test token limits for different detail levels."""
        tokens_summary = self.service._get_max_tokens('summary')
        tokens_standard = self.service._get_max_tokens('standard')
        tokens_detailed = self.service._get_max_tokens('detailed')
        
        self.assertEqual(tokens_summary, 75)
        self.assertEqual(tokens_standard, 150)
        self.assertEqual(tokens_detailed, 250)
        
        # Verify ordering
        self.assertLess(tokens_summary, tokens_standard)
        self.assertLess(tokens_standard, tokens_detailed)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        # Test good quality content
        good_content = "Strong BUY recommendation based on bullish indicators. RSI shows healthy momentum with score of 7/10. Technical trend analysis suggests upward trajectory with moderate risk."
        confidence_good = self.service._calculate_confidence_score(good_content)
        
        # Test poor quality content
        poor_content = "Buy."
        confidence_poor = self.service._calculate_confidence_score(poor_content)
        
        # Good content should have higher confidence
        self.assertGreater(confidence_good, confidence_poor)
        self.assertGreaterEqual(confidence_good, 0.6)
        self.assertLessEqual(confidence_good, 1.0)
    
    def test_caching_mechanism(self):
        """Test explanation caching."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.generate.return_value = self.mock_ollama_response
            mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
            
            service = LocalLLMService()
            service.client = mock_client_instance
            
            with patch.object(service, '_verify_model_availability', return_value=True):
                # Clear cache
                cache.clear()
                
                # First call - should hit LLM
                result1 = service.generate_explanation(self.mock_analysis_data)
                self.assertIsNotNone(result1)
                
                # Second call - should hit cache
                result2 = service.generate_explanation(self.mock_analysis_data)
                self.assertIsNotNone(result2)
                
                # Should only call generate once due to caching
                self.assertEqual(mock_client_instance.generate.call_count, 1)
    
    def test_cache_key_creation(self):
        """Test cache key creation consistency."""
        cache_key1 = self.service._create_cache_key(
            self.mock_analysis_data,
            'standard',
            'technical_analysis'
        )
        
        cache_key2 = self.service._create_cache_key(
            self.mock_analysis_data,
            'standard',
            'technical_analysis'
        )
        
        # Same inputs should produce same cache key
        self.assertEqual(cache_key1, cache_key2)
        
        # Different detail level should produce different key
        cache_key3 = self.service._create_cache_key(
            self.mock_analysis_data,
            'detailed',
            'technical_analysis'
        )
        
        self.assertNotEqual(cache_key1, cache_key3)
    
    def test_batch_explanation_generation(self):
        """Test batch explanation generation."""
        analysis_batch = [
            self.mock_analysis_data,
            {**self.mock_analysis_data, 'symbol': 'TEST2', 'score_0_10': 6},
            {**self.mock_analysis_data, 'symbol': 'TEST3', 'score_0_10': 8}
        ]
        
        with patch.object(self.service, 'generate_explanation') as mock_generate:
            mock_generate.return_value = {
                'content': 'Batch explanation',
                'confidence_score': 0.8
            }
            
            results = self.service.generate_batch_explanations(
                analysis_batch,
                detail_level='summary'
            )
            
            self.assertEqual(len(results), 3)
            self.assertEqual(mock_generate.call_count, 3)
    
    def test_service_status_reporting(self):
        """Test service status reporting."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.list.return_value = {
                'models': [
                    {'name': 'llama3.1:8b'},
                    {'name': 'other_model'}
                ]
            }
            
            service = LocalLLMService()
            service.client = mock_client_instance
            
            with patch.object(service, '_verify_model_availability') as mock_verify:
                def verify_side_effect(model):
                    return model == 'llama3.1:8b'
                mock_verify.side_effect = verify_side_effect
                
                status = service.get_service_status()
                
                self.assertIn('available', status)
                self.assertIn('primary_model_available', status)
                self.assertIn('detailed_model_available', status)
                self.assertIn('current_model', status)
                self.assertTrue(status['primary_model_available'])
                self.assertFalse(status['detailed_model_available'])
    
    def test_timeout_handling(self):
        """Test generation timeout handling."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            
            def slow_generate(*args, **kwargs):
                time.sleep(2)  # Simulate slow generation
                return self.mock_ollama_response
            
            mock_client_instance.generate.side_effect = slow_generate
            mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
            
            service = LocalLLMService()
            service.client = mock_client_instance
            service.generation_timeout = 1  # Very short timeout for testing
            
            with patch.object(service, '_verify_model_availability', return_value=True):
                # Skip timeout handling on Windows for testing
                with patch('Analytics.services.local_llm_service.hasattr', return_value=False):
                    result = service.generate_explanation(self.mock_analysis_data)
                    
                    # Should still work without timeout (fallback behavior)
                    self.assertIsNotNone(result)
    
    def test_performance_monitoring(self):
        """Test performance monitoring and logging."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            with patch('Analytics.services.local_llm_service.logger') as mock_logger:
                mock_client_instance = mock_client.return_value
                mock_client_instance.generate.return_value = self.mock_ollama_response
                mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
                
                service = LocalLLMService()
                service.client = mock_client_instance
                
                with patch.object(service, '_verify_model_availability', return_value=True):
                    with patch('time.time', side_effect=[0, 5]):  # 5 second generation
                        result = service.generate_explanation(self.mock_analysis_data)
                        
                        self.assertIsNotNone(result)
                        
                        # Should log performance metrics
                        mock_logger.info.assert_called()
                        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                        performance_logs = [log for log in log_calls if 'PERFORMANCE' in log]
                        self.assertGreater(len(performance_logs), 0)


class LocalLLMServiceSingletonTestCase(TestCase):
    """Test cases for singleton pattern and service management."""
    
    def test_singleton_pattern(self):
        """Test that get_local_llm_service returns singleton instance."""
        service1 = get_local_llm_service()
        service2 = get_local_llm_service()
        
        self.assertIs(service1, service2)
    
    def test_singleton_initialization(self):
        """Test singleton initialization behavior."""
        with patch('Analytics.services.local_llm_service.LocalLLMService') as mock_service:
            # Clear singleton
            import Analytics.services.local_llm_service as llm_module
            llm_module._llm_service = None
            
            # First call should initialize
            service1 = get_local_llm_service()
            mock_service.assert_called_once()
            
            # Second call should reuse
            service2 = get_local_llm_service()
            mock_service.assert_called_once()  # Still only one call


class LocalLLMServiceIntegrationTestCase(TestCase):
    """Integration tests for LLM service with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test data."""
        self.realistic_analysis_data = {
            'symbol': 'AAPL',
            'score_0_10': 8,
            'components': {
                'sma50vs200': 0.85,
                'price_vs_sma50': 0.72,
                'rsi14': 0.68,
                'macd12269': 0.75,
                'bollinger_bands_percent_b': 0.45,
                'bollinger_bands_bandwidth': 0.62,
                'volume_surge': 0.55,
                'obv_trend': 0.70,
                'relative_strength_sector': 0.82,
                'relative_strength_market': 0.78
            },
            'weighted_scores': {
                'w_sma50vs200': 0.17,
                'w_price_vs_sma50': 0.072,
                'w_rsi14': 0.068,
                'w_macd12269': 0.075,
                'w_bollinger_bands_percent_b': 0.045,
                'w_bollinger_bands_bandwidth': 0.062,
                'w_volume_surge': 0.055,
                'w_obv_trend': 0.035,
                'w_relative_strength_sector': 0.082,
                'w_relative_strength_market': 0.078
            }
        }
    
    def test_realistic_explanation_generation(self):
        """Test explanation generation with realistic financial data."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.generate.return_value = {
                'response': 'AAPL shows strong bullish momentum with a composite score of 8/10. The 50-day moving average is well above the 200-day average, indicating sustained upward trend. RSI at 68 suggests healthy momentum without being overbought. MACD histogram is positive, confirming bullish sentiment. Recommendation: BUY with high confidence based on technical strength and relative outperformance.',
                'done': True
            }
            mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
            
            service = LocalLLMService()
            service.client = mock_client_instance
            
            with patch.object(service, '_verify_model_availability', return_value=True):
                result = service.generate_explanation(
                    self.realistic_analysis_data,
                    detail_level='detailed'
                )
                
                self.assertIsNotNone(result)
                self.assertIn('AAPL', result['content'])
                self.assertIn('8/10', result['content'])
                self.assertIn('BUY', result['content'])
                self.assertGreater(result['confidence_score'], 0.8)
                self.assertGreater(result['word_count'], 30)
    
    def test_multiple_detail_levels_integration(self):
        """Test generation across all detail levels with realistic data."""
        with patch('Analytics.services.local_llm_service.Client') as mock_client:
            mock_client_instance = mock_client.return_value
            mock_client_instance.list.return_value = {'models': [{'name': 'llama3.1:8b'}]}
            
            service = LocalLLMService()
            service.client = mock_client_instance
            
            responses = {
                'summary': 'AAPL strong BUY (8/10) - bullish momentum across indicators.',
                'standard': 'AAPL shows strong technical strength with 8/10 score. Moving averages, RSI, and MACD all confirm bullish trend. Recommendation: BUY with high confidence.',
                'detailed': 'AAPL demonstrates exceptional technical strength with composite score of 8/10. The 50/200 SMA crossover signals sustained uptrend while RSI indicates healthy momentum. MACD histogram positive, volume patterns supportive. Sector and market relative strength both favorable. Strong BUY recommendation with 85% confidence based on comprehensive technical analysis.'
            }
            
            def mock_generate_side_effect(*args, **kwargs):
                # Extract detail level from options
                options = kwargs.get('options', {})
                num_predict = options.get('num_predict', 150)
                
                if num_predict <= 75:  # summary
                    content = responses['summary']
                elif num_predict <= 150:  # standard
                    content = responses['standard']
                else:  # detailed
                    content = responses['detailed']
                
                return {'response': content, 'done': True}
            
            mock_client_instance.generate.side_effect = mock_generate_side_effect
            
            with patch.object(service, '_verify_model_availability', return_value=True):
                for detail_level in ['summary', 'standard', 'detailed']:
                    result = service.generate_explanation(
                        self.realistic_analysis_data,
                        detail_level=detail_level
                    )
                    
                    self.assertIsNotNone(result)
                    self.assertEqual(result['detail_level'], detail_level)
                    self.assertIn('AAPL', result['content'])
                    
                    # Verify content length increases with detail level
                    if detail_level == 'summary':
                        self.assertLess(result['word_count'], 15)
                    elif detail_level == 'standard':
                        self.assertGreater(result['word_count'], 15)
                        self.assertLess(result['word_count'], 40)
                    else:  # detailed
                        self.assertGreater(result['word_count'], 40)