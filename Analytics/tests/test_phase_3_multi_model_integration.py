"""
Phase 3.1 Multi-Model Integration Testing

Tests the complete multi-model LLM system including:
- Model selection logic for different detail levels
- Fallback mechanisms
- Performance benchmarking
- Quality validation
"""

import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.explanation_service import get_explanation_service
from Data.models import AnalyticsResults, DataIndustry, DataSector, Stock, StockPrice

User = get_user_model()


class Phase3MultiModelIntegrationTestCase(TransactionTestCase):
    """Phase 3.1 multi-model integration tests."""

    def setUp(self):
        """Set up comprehensive test data."""
        cache.clear()

        # Create test user
        self.user = User.objects.create_user(
            username="phase3_user", email="phase3@test.com", password="testpass123"
        )

        # Create sector and industry
        self.sector = DataSector.objects.create(
            sectorKey="tech_phase3", sectorName="Technology Phase 3", data_source="test"
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey="software_phase3", 
            industryName="Software Phase 3",
            sector=self.sector,
            data_source="test"
        )

        # Create test stock
        self.stock = Stock.objects.create(
            symbol="TEST",
            short_name="Test Company Phase 3",
            sector=self.sector.sectorName,
            industry=self.industry.industryName,
            market_cap=1000000000,
            shares_outstanding=100000000,
            is_active=True
        )

        # Create stock prices for technical analysis
        self.create_test_price_data()

        # Get services
        self.llm_service = get_local_llm_service()
        self.explanation_service = get_explanation_service()
        self.ta_engine = TechnicalAnalysisEngine()

    def create_test_price_data(self):
        """Create comprehensive test price data for technical analysis."""
        base_price = Decimal('100.00')
        base_date = datetime.now().date() - timedelta(days=60)
        
        prices = []
        for i in range(60):
            date = base_date + timedelta(days=i)
            # Create realistic price movement
            price_change = Decimal(str(0.5 * (i % 5 - 2)))  # Oscillating pattern
            current_price = base_price + price_change + Decimal(str(i * 0.1))
            
            prices.append(StockPrice(
                stock=self.stock,
                date=date,
                open_price=current_price - Decimal('0.50'),
                high_price=current_price + Decimal('1.00'),
                low_price=current_price - Decimal('1.50'),
                close_price=current_price,
                volume=1000000 + i * 10000,
                data_source="test"
            ))
        
        StockPrice.objects.bulk_create(prices)

    def create_test_analysis_data(self, complexity_level: str = "standard") -> Dict[str, Any]:
        """Create test analysis data with varying complexity."""
        base_data = {
            'symbol': 'TEST',
            'technical_score': 7.2,
            'recommendation': 'BUY',
            'analysis_date': datetime.now().isoformat(),
            'indicators': {
                'rsi': {'value': 45.2, 'signal': 'neutral', 'interpretation': 'RSI indicates neutral momentum'},
                'macd': {'value': 0.15, 'signal': 'bullish', 'interpretation': 'MACD shows bullish crossover'},
                'sma_50': {'value': 98.5, 'signal': 'bullish', 'interpretation': 'Price above 50-day SMA'},
                'sma_200': {'value': 95.0, 'signal': 'bullish', 'interpretation': 'Price above 200-day SMA'}
            }
        }
        
        if complexity_level == "detailed":
            # Add more complex indicators for detailed analysis
            base_data['indicators'].update({
                'bollinger_bands': {
                    'upper': 102.5, 'lower': 96.5, 'position': 0.6,
                    'signal': 'neutral', 'interpretation': 'Price in upper half of Bollinger Bands'
                },
                'volume_trend': {
                    'value': 1.2, 'signal': 'bullish', 
                    'interpretation': 'Volume 20% above average'
                },
                'support_levels': [95.0, 98.0],
                'resistance_levels': [105.0, 108.0],
                'trend_analysis': {
                    'short_term': 'bullish',
                    'medium_term': 'bullish', 
                    'long_term': 'neutral'
                }
            })
            base_data['sentiment'] = {
                'score': 0.65,
                'label': 'positive',
                'news_items': 5
            }
        
        return base_data

    def test_summary_explanation_generation(self):
        """Test summary explanation generation using phi3:3.8b model."""
        analysis_data = self.create_test_analysis_data("summary")
        
        start_time = time.time()
        result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="summary"
        )
        response_time = time.time() - start_time
        
        # Verify response structure
        self.assertIsNotNone(result)
        self.assertIn('explanation', result)
        self.assertIn('model_used', result)
        
        # Verify correct model selection
        expected_model = self.llm_service.summary_model
        self.assertEqual(result['model_used'], expected_model)
        
        # Verify explanation length (summary should be concise)
        explanation = result['explanation']
        word_count = len(explanation.split())
        self.assertGreaterEqual(word_count, 20)  # At least substantial content
        self.assertLessEqual(word_count, 150)    # But concise for summary
        
        # Verify response time performance
        self.assertLess(response_time, 15.0)     # Should complete within 15 seconds
        
        # Verify content quality
        self.assertIn('TEST', explanation)       # Should mention the symbol
        self.assertIn('BUY', explanation.upper())  # Should mention recommendation

    def test_standard_explanation_generation(self):
        """Test standard explanation generation using phi3:3.8b model."""
        analysis_data = self.create_test_analysis_data("standard")
        
        start_time = time.time()
        result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="standard"
        )
        response_time = time.time() - start_time
        
        # Verify response structure
        self.assertIsNotNone(result)
        self.assertIn('explanation', result)
        self.assertIn('model_used', result)
        
        # Verify correct model selection
        expected_model = self.llm_service.standard_model
        self.assertEqual(result['model_used'], expected_model)
        
        # Verify explanation length (standard should be moderate)
        explanation = result['explanation']
        word_count = len(explanation.split())
        self.assertGreaterEqual(word_count, 100)  # More detailed than summary
        self.assertLessEqual(word_count, 400)     # But not excessive
        
        # Verify response time performance
        self.assertLess(response_time, 20.0)      # Should complete within 20 seconds

    def test_detailed_explanation_generation(self):
        """Test detailed explanation generation using llama3.1:8b model."""
        analysis_data = self.create_test_analysis_data("detailed")
        
        start_time = time.time()
        result = self.llm_service.generate_explanation(
            analysis_data=analysis_data,
            detail_level="detailed"
        )
        response_time = time.time() - start_time
        
        # Verify response structure
        self.assertIsNotNone(result)
        self.assertIn('explanation', result)
        self.assertIn('model_used', result)
        
        # Verify correct model selection
        expected_model = self.llm_service.detailed_model
        self.assertEqual(result['model_used'], expected_model)
        
        # Verify explanation length (detailed should be comprehensive)
        explanation = result['explanation']
        word_count = len(explanation.split())
        self.assertGreaterEqual(word_count, 200)  # Comprehensive analysis
        self.assertLessEqual(word_count, 800)     # But not excessive
        
        # Verify response time performance
        self.assertLess(response_time, 30.0)      # Should complete within 30 seconds
        
        # Verify detailed content inclusion
        self.assertIn('Bollinger', explanation.lower())  # Should mention advanced indicators

    def test_model_fallback_mechanism(self):
        """Test model fallback when primary model fails."""
        analysis_data = self.create_test_analysis_data("standard")
        
        # Temporarily modify model configuration to trigger fallback
        original_standard_model = self.llm_service.standard_model
        self.llm_service.standard_model = "nonexistent:model"
        
        try:
            result = self.llm_service.generate_explanation(
                analysis_data=analysis_data,
                detail_level="standard"
            )
            
            # Should still get a result via fallback
            self.assertIsNotNone(result)
            
            # Model used should be fallback model, not the nonexistent one
            if 'model_used' in result:
                self.assertNotEqual(result['model_used'], "nonexistent:model")
                
        finally:
            # Restore original configuration
            self.llm_service.standard_model = original_standard_model

    def test_concurrent_multi_level_generation(self):
        """Test concurrent generation of multiple explanation levels."""
        analysis_data = self.create_test_analysis_data("detailed")
        
        import threading
        results = {}
        errors = {}
        
        def generate_explanation(detail_level: str):
            try:
                start_time = time.time()
                result = self.llm_service.generate_explanation(
                    analysis_data=analysis_data,
                    detail_level=detail_level
                )
                response_time = time.time() - start_time
                results[detail_level] = {
                    'result': result,
                    'response_time': response_time
                }
            except Exception as e:
                errors[detail_level] = str(e)
        
        # Launch concurrent requests
        threads = []
        for level in ['summary', 'standard', 'detailed']:
            thread = threading.Thread(target=generate_explanation, args=(level,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests completed successfully
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)
        
        # Verify each level used appropriate model
        for level, data in results.items():
            result = data['result']
            self.assertIsNotNone(result)
            self.assertIn('explanation', result)
            self.assertIn('model_used', result)

    def test_explanation_quality_consistency(self):
        """Test explanation quality consistency across multiple generations."""
        analysis_data = self.create_test_analysis_data("standard")
        
        explanations = []
        response_times = []
        
        # Generate multiple explanations
        for i in range(3):
            start_time = time.time()
            result = self.llm_service.generate_explanation(
                analysis_data=analysis_data,
                detail_level="standard"
            )
            response_time = time.time() - start_time
            
            self.assertIsNotNone(result)
            explanations.append(result['explanation'])
            response_times.append(response_time)
        
        # Verify all explanations contain key elements
        for explanation in explanations:
            self.assertIn('TEST', explanation)
            word_count = len(explanation.split())
            self.assertGreater(word_count, 50)  # Substantial content
        
        # Verify response time consistency (after warm-up)
        avg_response_time = sum(response_times) / len(response_times)
        for response_time in response_times[1:]:  # Skip first (cold start)
            # Response times should be within 50% of average
            self.assertLess(abs(response_time - avg_response_time) / avg_response_time, 0.5)

    def test_performance_benchmarking(self):
        """Test and benchmark performance across all models."""
        analysis_data = self.create_test_analysis_data("detailed")
        
        performance_results = {}
        
        for detail_level in ['summary', 'standard', 'detailed']:
            times = []
            
            # Run multiple tests for average
            for _ in range(2):  # Limited runs to save time
                start_time = time.time()
                result = self.llm_service.generate_explanation(
                    analysis_data=analysis_data,
                    detail_level=detail_level
                )
                response_time = time.time() - start_time
                
                if result and 'explanation' in result:
                    times.append(response_time)
            
            if times:
                performance_results[detail_level] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'successful_requests': len(times)
                }
        
        # Verify performance benchmarks are met
        if 'summary' in performance_results:
            self.assertLess(performance_results['summary']['avg_time'], 15.0)
        if 'standard' in performance_results:
            self.assertLess(performance_results['standard']['avg_time'], 20.0)
        if 'detailed' in performance_results:
            self.assertLess(performance_results['detailed']['avg_time'], 30.0)

    def test_integration_with_explanation_service(self):
        """Test integration with the ExplanationService."""
        # Create analysis results
        analysis_result = AnalyticsResults.objects.create(
            stock=self.stock,
            user=self.user,
            analysis_date=timezone.now(),
            technical_score=Decimal('7.5'),
            recommendation='BUY',
            indicators_data={'rsi': 45.2, 'macd': 0.15}
        )
        
        # Test explanation generation through service
        result = self.explanation_service.generate_explanation(
            analysis_result_id=analysis_result.id,
            detail_level="standard"
        )
        
        self.assertIsNotNone(result)
        self.assertIn('explanation', result)
        self.assertIn('model_used', result)