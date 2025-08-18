"""
Unit tests for Data analytics_writer module.
Tests AnalyticsWriter repository for storing technical analysis outputs.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from django.test import TestCase
from django.utils import timezone
from django.contrib.auth.models import User
from django.db import transaction

from Data.repo.analytics_writer import AnalyticsWriter
from Data.models import Stock, AnalyticsResults


class AnalyticsWriterTestCase(TestCase):
    """Test cases for AnalyticsWriter repository."""
    
    def setUp(self):
        """Set up test data."""
        self.writer = AnalyticsWriter()
        
        # Create test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        
        # Create test stock
        self.stock = Stock.objects.create(
            symbol='TEST',
            short_name='Test Company',
            exchange='NASDAQ'
        )
        
        # Test analysis data
        self.as_of = timezone.now()
        self.weighted_scores = {
            'w_sma50vs200': Decimal('0.75'),
            'w_pricevs50': Decimal('0.60'),
            'w_rsi14': Decimal('0.45'),
            'w_macd12269': Decimal('0.80'),
            'w_bbpos20': Decimal('0.55'),
            'w_bbwidth20': Decimal('0.70'),
            'w_volsurge': Decimal('0.85'),
            'w_obv20': Decimal('0.50'),
            'w_rel1y': Decimal('0.65'),
            'w_rel2y': Decimal('0.70'),
            'w_candlerev': Decimal('0.40'),
            'w_srcontext': Decimal('0.60')
        }
        self.components = {
            'sma50vs200': {
                'raw': {'sma50': 150.0, 'sma200': 140.0},
                'normalized': 0.75
            },
            'sentiment': {
                'raw': {
                    'sentiment': 0.6,
                    'label': 'positive',
                    'confidence': 0.85,
                    'newsCount': 15,
                    'sources': {'reuters': 5, 'bloomberg': 10}
                },
                'normalized': 0.6
            },
            'prediction': {
                'raw': {
                    'predicted_price': 155.50,
                    'confidence': 0.75,
                    'model_version': 'lstm_v1.0'
                },
                'normalized': 0.7
            }
        }
        self.composite_raw = Decimal('7.55')
        self.score_0_10 = 8
    
    def test_upsert_analytics_result_create(self):
        """Test creating new analytics result."""
        result = self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=self.as_of,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=self.composite_raw,
            score_0_10=self.score_0_10,
            user=self.user
        )
        
        self.assertIsInstance(result, AnalyticsResults)
        self.assertEqual(result.stock, self.stock)
        self.assertEqual(result.user, self.user)
        self.assertEqual(result.as_of, self.as_of)
        self.assertEqual(result.w_sma50vs200, Decimal('0.75'))
        self.assertEqual(result.composite_raw, Decimal('7.55'))
        self.assertEqual(result.score_0_10, 8)
        self.assertEqual(result.horizon, 'blend')
        
        # Check sentiment fields
        self.assertEqual(result.sentimentScore, 0.6)
        self.assertEqual(result.sentimentLabel, 'positive')
        self.assertEqual(result.sentimentConfidence, 0.85)
        self.assertEqual(result.newsCount, 15)
        
        # Check prediction fields
        self.assertEqual(result.prediction_1d, 155.50)
        self.assertEqual(result.prediction_confidence, 0.75)
        self.assertEqual(result.model_version, 'lstm_v1.0')
    
    def test_upsert_analytics_result_update(self):
        """Test updating existing analytics result."""
        # Create initial result
        initial_result = self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=self.as_of,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=self.composite_raw,
            score_0_10=self.score_0_10,
            user=self.user
        )
        
        # Update with new scores
        updated_scores = self.weighted_scores.copy()
        updated_scores['w_sma50vs200'] = Decimal('0.85')
        updated_composite = Decimal('7.65')
        updated_score = 9
        
        updated_result = self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=self.as_of,
            weighted_scores=updated_scores,
            components=self.components,
            composite_raw=updated_composite,
            score_0_10=updated_score,
            user=self.user
        )
        
        # Should be the same instance, updated
        self.assertEqual(initial_result.id, updated_result.id)
        self.assertEqual(updated_result.w_sma50vs200, Decimal('0.85'))
        self.assertEqual(updated_result.composite_raw, Decimal('7.65'))
        self.assertEqual(updated_result.score_0_10, 9)
        
        # Should only have one record in database
        self.assertEqual(AnalyticsResults.objects.filter(stock=self.stock).count(), 1)
    
    def test_upsert_analytics_result_without_user(self):
        """Test creating analytics result without user."""
        result = self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=self.as_of,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=self.composite_raw,
            score_0_10=self.score_0_10
        )
        
        self.assertIsNone(result.user)
        self.assertEqual(result.stock, self.stock)
    
    def test_upsert_analytics_result_missing_required_score(self):
        """Test that missing required weighted scores raise ValueError."""
        incomplete_scores = self.weighted_scores.copy()
        del incomplete_scores['w_sma50vs200']
        
        with self.assertRaises(ValueError) as context:
            self.writer.upsert_analytics_result(
                symbol='TEST',
                as_of=self.as_of,
                weighted_scores=incomplete_scores,
                components=self.components,
                composite_raw=self.composite_raw,
                score_0_10=self.score_0_10
            )
        
        self.assertIn('Missing required weighted score: w_sma50vs200', str(context.exception))
    
    def test_upsert_analytics_result_nonexistent_stock(self):
        """Test that nonexistent stock raises Stock.DoesNotExist."""
        with self.assertRaises(Stock.DoesNotExist):
            self.writer.upsert_analytics_result(
                symbol='NONEXISTENT',
                as_of=self.as_of,
                weighted_scores=self.weighted_scores,
                components=self.components,
                composite_raw=self.composite_raw,
                score_0_10=self.score_0_10
            )
    
    def test_upsert_analytics_result_custom_horizon(self):
        """Test creating analytics result with custom horizon."""
        result = self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=self.as_of,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=self.composite_raw,
            score_0_10=self.score_0_10,
            horizon='short_term'
        )
        
        self.assertEqual(result.horizon, 'short_term')
    
    def test_batch_upsert_analytics_results(self):
        """Test batch upserting multiple analytics results."""
        # Create second stock
        stock2 = Stock.objects.create(
            symbol='TEST2',
            short_name='Test Company 2',
            exchange='NYSE'
        )
        
        results_data = [
            {
                'symbol': 'TEST',
                'as_of': self.as_of,
                'weighted_scores': self.weighted_scores,
                'components': self.components,
                'composite_raw': self.composite_raw,
                'score_0_10': self.score_0_10,
                'horizon': 'short_term'
            },
            {
                'symbol': 'TEST2',
                'as_of': self.as_of + timedelta(minutes=1),
                'weighted_scores': self.weighted_scores,
                'components': self.components,
                'composite_raw': Decimal('6.50'),
                'score_0_10': 7
            }
        ]
        
        results = self.writer.batch_upsert_analytics_results(results_data, user=self.user)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].stock.symbol, 'TEST')
        self.assertEqual(results[0].horizon, 'short_term')
        self.assertEqual(results[1].stock.symbol, 'TEST2')
        self.assertEqual(results[1].horizon, 'blend')  # Default
        
        # Check database
        self.assertEqual(AnalyticsResults.objects.count(), 2)
    
    def test_batch_upsert_transaction_rollback(self):
        """Test that batch upsert rolls back on error."""
        results_data = [
            {
                'symbol': 'TEST',
                'as_of': self.as_of,
                'weighted_scores': self.weighted_scores,
                'components': self.components,
                'composite_raw': self.composite_raw,
                'score_0_10': self.score_0_10
            },
            {
                'symbol': 'NONEXISTENT',  # This will cause an error
                'as_of': self.as_of,
                'weighted_scores': self.weighted_scores,
                'components': self.components,
                'composite_raw': self.composite_raw,
                'score_0_10': self.score_0_10
            }
        ]
        
        with self.assertRaises(Stock.DoesNotExist):
            self.writer.batch_upsert_analytics_results(results_data)
        
        # No results should be saved due to transaction rollback
        self.assertEqual(AnalyticsResults.objects.count(), 0)
    
    def test_get_latest_analytics_result(self):
        """Test getting latest analytics result."""
        # Create multiple results with different timestamps
        older_time = self.as_of - timedelta(hours=1)
        newer_time = self.as_of + timedelta(hours=1)
        
        self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=older_time,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=Decimal('6.00'),
            score_0_10=6
        )
        
        self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=newer_time,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=Decimal('8.00'),
            score_0_10=8
        )
        
        latest = self.writer.get_latest_analytics_result('TEST')
        
        self.assertIsNotNone(latest)
        self.assertEqual(latest.as_of, newer_time)
        self.assertEqual(latest.score_0_10, 8)
    
    def test_get_latest_analytics_result_nonexistent_stock(self):
        """Test getting latest result for nonexistent stock."""
        latest = self.writer.get_latest_analytics_result('NONEXISTENT')
        self.assertIsNone(latest)
    
    def test_get_analytics_results_range(self):
        """Test getting analytics results within date range."""
        # Create results over time
        times = [
            self.as_of - timedelta(days=2),
            self.as_of - timedelta(days=1),
            self.as_of,
            self.as_of + timedelta(days=1),
            self.as_of + timedelta(days=2)
        ]
        
        for i, time in enumerate(times):
            self.writer.upsert_analytics_result(
                symbol='TEST',
                as_of=time,
                weighted_scores=self.weighted_scores,
                components=self.components,
                composite_raw=Decimal(f'{6 + i}.00'),
                score_0_10=6 + i
            )
        
        # Get results for middle 3 days
        start_date = self.as_of - timedelta(days=1, hours=12)
        end_date = self.as_of + timedelta(days=1, hours=12)
        
        results = self.writer.get_analytics_results_range(
            'TEST',
            start_date=start_date,
            end_date=end_date
        )
        
        self.assertEqual(len(results), 3)
        # Should be ordered by as_of descending
        self.assertEqual(results[0].score_0_10, 9)  # Most recent
        self.assertEqual(results[2].score_0_10, 7)  # Oldest in range
    
    def test_get_analytics_results_range_with_limit(self):
        """Test getting analytics results with limit."""
        # Create 5 results
        for i in range(5):
            self.writer.upsert_analytics_result(
                symbol='TEST',
                as_of=self.as_of + timedelta(hours=i),
                weighted_scores=self.weighted_scores,
                components=self.components,
                composite_raw=Decimal(f'{6 + i}.00'),
                score_0_10=6 + i
            )
        
        results = self.writer.get_analytics_results_range('TEST', limit=3)
        
        self.assertEqual(len(results), 3)
        # Should get the 3 most recent
        self.assertEqual(results[0].score_0_10, 10)
        self.assertEqual(results[2].score_0_10, 8)
    
    def test_delete_old_analytics_results(self):
        """Test deleting old analytics results."""
        # Create old and recent results
        old_time = self.as_of - timedelta(days=100)
        recent_time = self.as_of - timedelta(days=30)
        
        self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=old_time,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=Decimal('6.00'),
            score_0_10=6
        )
        
        self.writer.upsert_analytics_result(
            symbol='TEST',
            as_of=recent_time,
            weighted_scores=self.weighted_scores,
            components=self.components,
            composite_raw=Decimal('8.00'),
            score_0_10=8
        )
        
        # Delete results older than 90 days
        deleted_count = self.writer.delete_old_analytics_results('TEST', keep_days=90)
        
        self.assertEqual(deleted_count, 1)
        
        # Only recent result should remain
        remaining = AnalyticsResults.objects.filter(stock=self.stock)
        self.assertEqual(remaining.count(), 1)
        self.assertEqual(remaining.first().score_0_10, 8)
    
    def test_delete_old_analytics_results_nonexistent_stock(self):
        """Test deleting old results for nonexistent stock."""
        deleted_count = self.writer.delete_old_analytics_results('NONEXISTENT')
        self.assertEqual(deleted_count, 0)
    
    def test_get_analytics_summary_with_data(self):
        """Test getting analytics summary with data."""
        # Create multiple results
        scores = [6, 7, 8, 9, 10]
        times = []
        
        for i, score in enumerate(scores):
            time = self.as_of + timedelta(hours=i)
            times.append(time)
            self.writer.upsert_analytics_result(
                symbol='TEST',
                as_of=time,
                weighted_scores=self.weighted_scores,
                components=self.components,
                composite_raw=Decimal(f'{score}.00'),
                score_0_10=score
            )
        
        summary = self.writer.get_analytics_summary('TEST')
        
        self.assertEqual(summary['total_results'], 5)
        self.assertEqual(summary['earliest_analysis'], times[0])
        self.assertEqual(summary['latest_analysis'], times[-1])
        self.assertEqual(summary['avg_score'], 8.0)  # (6+7+8+9+10)/5
        self.assertEqual(summary['min_score'], 6)
        self.assertEqual(summary['max_score'], 10)
    
    def test_get_analytics_summary_no_data(self):
        """Test getting analytics summary with no data."""
        summary = self.writer.get_analytics_summary('TEST')
        
        self.assertEqual(summary['total_results'], 0)
        self.assertIsNone(summary['earliest_analysis'])
        self.assertIsNone(summary['latest_analysis'])
        self.assertIsNone(summary['avg_score'])
        self.assertIsNone(summary['min_score'])
        self.assertIsNone(summary['max_score'])
    
    def test_get_analytics_summary_nonexistent_stock(self):
        """Test getting analytics summary for nonexistent stock."""
        summary = self.writer.get_analytics_summary('NONEXISTENT')
        
        self.assertEqual(summary['total_results'], 0)
        self.assertIsNone(summary['earliest_analysis'])


class AnalyticsWriterIntegrationTestCase(TestCase):
    """Integration tests for AnalyticsWriter with complex scenarios."""
    
    def setUp(self):
        """Set up test data."""
        self.writer = AnalyticsWriter()
        
        # Create multiple users and stocks
        self.users = []
        for i in range(2):
            user = User.objects.create_user(
                username=f'user{i}',
                email=f'user{i}@example.com',
                password='testpass123'
            )
            self.users.append(user)
        
        self.stocks = []
        for i in range(2):
            stock = Stock.objects.create(
                symbol=f'STOCK{i}',
                short_name=f'Stock {i} Company',
                exchange='NASDAQ'
            )
            self.stocks.append(stock)
        
        self.weighted_scores = {
            'w_sma50vs200': Decimal('0.75'),
            'w_pricevs50': Decimal('0.60'),
            'w_rsi14': Decimal('0.45'),
            'w_macd12269': Decimal('0.80'),
            'w_bbpos20': Decimal('0.55'),
            'w_bbwidth20': Decimal('0.70'),
            'w_volsurge': Decimal('0.85'),
            'w_obv20': Decimal('0.50'),
            'w_rel1y': Decimal('0.65'),
            'w_rel2y': Decimal('0.70'),
            'w_candlerev': Decimal('0.40'),
            'w_srcontext': Decimal('0.60')
        }
    
    def test_concurrent_upserts(self):
        """Test concurrent upsert operations."""
        import threading
        results = []
        errors = []
        
        def upsert_result(stock_symbol, user, score):
            try:
                result = self.writer.upsert_analytics_result(
                    symbol=stock_symbol,
                    as_of=timezone.now(),
                    weighted_scores=self.weighted_scores,
                    components={'test': {'raw': {}, 'normalized': 0.5}},
                    composite_raw=Decimal('7.00'),
                    score_0_10=score,
                    user=user
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(10):
            stock = self.stocks[i % 2]
            user = self.users[i % 2]
            thread = threading.Thread(
                target=upsert_result,
                args=(stock.symbol, user, 7 + (i % 3))
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 0)
    
    def test_user_specific_analytics_isolation(self):
        """Test that analytics results are properly isolated by user."""
        as_of = timezone.now()
        
        # Both users analyze the same stock at same time
        result1 = self.writer.upsert_analytics_result(
            symbol='STOCK0',
            as_of=as_of,
            weighted_scores=self.weighted_scores,
            components={'test': {'raw': {}, 'normalized': 0.5}},
            composite_raw=Decimal('7.00'),
            score_0_10=7,
            user=self.users[0]
        )
        
        result2 = self.writer.upsert_analytics_result(
            symbol='STOCK0',
            as_of=as_of,
            weighted_scores=self.weighted_scores,
            components={'test': {'raw': {}, 'normalized': 0.6}},
            composite_raw=Decimal('8.00'),
            score_0_10=8,
            user=self.users[1]
        )
        
        # Should have separate results
        self.assertNotEqual(result1.id, result2.id)
        self.assertEqual(result1.user, self.users[0])
        self.assertEqual(result2.user, self.users[1])
        self.assertEqual(result1.score_0_10, 7)
        self.assertEqual(result2.score_0_10, 8)
        
        # Database should have 2 records
        self.assertEqual(AnalyticsResults.objects.count(), 2)
    
    def test_large_batch_upsert_performance(self):
        """Test performance with large batch upsert."""
        # Create batch data for 100 results
        batch_data = []
        base_time = timezone.now()
        
        for i in range(100):
            batch_data.append({
                'symbol': self.stocks[i % 2].symbol,
                'as_of': base_time + timedelta(seconds=i),
                'weighted_scores': self.weighted_scores,
                'components': {'test': {'raw': {}, 'normalized': 0.5}},
                'composite_raw': Decimal(f'{6 + (i % 5)}.00'),
                'score_0_10': 6 + (i % 5)
            })
        
        # Time the batch operation
        start_time = timezone.now()
        results = self.writer.batch_upsert_analytics_results(batch_data, user=self.users[0])
        end_time = timezone.now()
        
        # Verify results
        self.assertEqual(len(results), 100)
        
        # Should complete reasonably quickly (less than 10 seconds)
        duration = (end_time - start_time).total_seconds()
        self.assertLess(duration, 10)
        
        # Verify database state
        self.assertEqual(AnalyticsResults.objects.count(), 100)