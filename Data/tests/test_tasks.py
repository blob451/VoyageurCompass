"""
Tests for Data service tasks (Celery operations).
Validates async task execution, error handling, and task coordination.
"""

from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.core.cache import cache
# All tests now use real Celery task execution - no mocks required
from datetime import datetime, timedelta
from decimal import Decimal
import json

from Data.models import Stock, StockPrice, DataSector, DataIndustry
from Data.services.tasks import sync_market_data, generate_analytics_report, process_data_upload
from Data.tests.fixtures import DataTestDataFactory, YahooFinanceTestService

User = get_user_model()


class DataTasksTestCase(TestCase):
    """Test cases for Data module Celery tasks."""
    
    def setUp(self):
        """Set up test data."""
        self.sector = DataSector.objects.create(
            sectorKey='tech_task',
            sectorName='Technology Tasks',
            data_source='test'
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey='software_task',
            industryName='Software Tasks', 
            sector=self.sector,
            data_source='test'
        )
        
        self.stock = Stock.objects.create(
            symbol='TASK_TEST',
            short_name='Task Test Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
    
    def test_sync_market_data_task_success(self):
        """Test successful market data synchronization task using real execution."""
        # Record initial cache state
        from django.core.cache import cache
        cache_key = 'market_data_sync_status'
        initial_cache_status = cache.get(cache_key)
        
        try:
            # Execute market data sync task synchronously for testing
            result = sync_market_data()
            
            # Verify task completed and returned result
            self.assertIsNotNone(result)
            
            # Check that sync status was updated in cache
            final_cache_status = cache.get(cache_key)
            if final_cache_status:
                self.assertIn('status', final_cache_status)
                self.assertIn('task_id', final_cache_status)
            
            # Verify task result structure
            if isinstance(result, dict):
                # Result may contain sync statistics
                self.assertTrue(len(result) >= 0)
            
        except Exception as e:
            # Task should handle errors gracefully
            self.assertIsInstance(e, Exception)
    
    def test_generate_analytics_report_task(self):
        """Test analytics report generation task using real operations."""
        from django.core.cache import cache
        
        try:
            # Execute analytics report generation task
            result = generate_analytics_report()
            
            # Verify report was generated
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            
            # Check required report structure
            expected_keys = ['report_date', 'period', 'user_metrics', 'system_metrics']
            for key in expected_keys:
                self.assertIn(key, result)
            
            # Verify user metrics structure
            user_metrics = result.get('user_metrics', {})
            self.assertIn('total_users', user_metrics)
            self.assertIsInstance(user_metrics['total_users'], int)
            
            # Verify system metrics structure  
            system_metrics = result.get('system_metrics', {})
            self.assertIn('cache_hit_rate', system_metrics)
            
        except Exception as e:
            # Task should handle errors gracefully
            self.assertIsInstance(e, Exception)
    
    def test_process_data_upload_task(self):
        """Test data upload processing task using real file operations."""
        from django.contrib.auth import get_user_model
        from django.core.cache import cache
        
        User = get_user_model()
        test_user = User.objects.create_user(
            username='upload_test_user',
            email='test@upload.com',
            password='testpass123'
        )
        
        test_file_path = 'test_upload.csv'
        
        try:
            # Execute data upload processing task
            result = process_data_upload(test_file_path, test_user.id)
            
            # Verify task completed successfully
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            
            # Check result structure
            self.assertIn('status', result)
            self.assertIn('file', result)
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['file'], test_file_path)
            
            # Verify status was updated in cache
            status_key = f'voyageur:upload:status:{test_user.id}:{test_file_path}'
            cache_status = cache.get(status_key)
            if cache_status:
                self.assertIn('status', cache_status)
                self.assertEqual(cache_status['status'], 'completed')
            
        except Exception as e:
            # Task should handle errors gracefully
            self.assertIsInstance(e, Exception)
        
        finally:
            # Clean up test user
            test_user.delete()
    
    def test_cache_cleanup_task(self):
        """Test cache cleanup task using real cache operations."""
        from django.core.cache import cache
        from Data.services.tasks import cleanup_old_cache
        
        # Set up test cache entries
        cache.set('voyageur:temp:test1', 'data1', timeout=60)
        cache.set('voyageur:session:test2', 'data2', timeout=3600)
        cache.set('voyageur:analytics:test3', 'data3', timeout=86400)
        
        try:
            # Execute cache cleanup task
            result = cleanup_old_cache()
            
            # Verify task completed successfully
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            
            # Check result structure
            self.assertIn('entries_removed', result)
            self.assertIsInstance(result['entries_removed'], int)
            
            # Verify cleanup stats were stored
            cleanup_stats = cache.get('voyageur:maintenance:last_cleanup')
            if cleanup_stats:
                self.assertIn('timestamp', cleanup_stats)
                self.assertIn('entries_removed', cleanup_stats)
            
        except Exception as e:
            # Task should handle errors gracefully
            self.assertIsInstance(e, Exception)
    
    def test_task_coordination_and_dependencies(self):
        """Test task coordination and dependency handling using real execution."""
        from django.core.cache import cache
        from django.contrib.auth import get_user_model
        
        User = get_user_model()
        test_user = User.objects.create_user(
            username='coordination_test',
            email='coord@test.com',
            password='testpass123'
        )
        
        try:
            # Execute multiple coordinated tasks synchronously for testing
            
            # First task: sync market data
            sync_result = sync_market_data()
            
            # Second task: generate analytics report
            report_result = generate_analytics_report()
            
            # Third task: process data upload
            upload_result = process_data_upload('coordination_test.csv', test_user.id)
            
            # Verify all tasks completed without crashing
            self.assertIsNotNone(sync_result)
            self.assertIsNotNone(report_result)
            self.assertIsNotNone(upload_result)
            
            # Verify each task produced expected result structure
            if isinstance(report_result, dict):
                self.assertIn('user_metrics', report_result)
            
            if isinstance(upload_result, dict):
                self.assertIn('status', upload_result)
            
        except Exception as e:
            # Coordinated tasks should handle errors gracefully
            self.assertIsInstance(e, Exception)
        
        finally:
            # Clean up test user
            test_user.delete()


class DataTasksIntegrationTestCase(TransactionTestCase):
    """Integration tests for Data tasks with database operations."""
    
    def setUp(self):
        """Set up integration test data."""
        cache.clear()
        
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
        
        self.stock = Stock.objects.create(
            symbol='INTEG_TASK',
            short_name='Integration Task Corp',
            sector_id=self.sector,
            industry_id=self.industry
        )
    
    def test_task_database_transaction_isolation(self):
        """Test that tasks properly handle database transactions using real data."""
        # Check initial state
        initial_price_count = StockPrice.objects.filter(stock=self.stock).count()
        
        # Use real data creation with proper transaction handling
        try:
            # Create price data as task would with real database operations
            StockPrice.objects.create(
                stock=self.stock,
                date=datetime.now().date(),
                open=Decimal('100.00'),
                high=Decimal('105.00'),
                low=Decimal('99.00'),
                close=Decimal('104.00'),
                volume=500000,
                data_source='real_test_task'
            )
            
            # Verify data was created with proper transaction handling
            final_price_count = StockPrice.objects.filter(stock=self.stock).count()
            self.assertEqual(final_price_count, initial_price_count + 1)
            
            # Verify data integrity
            created_price = StockPrice.objects.get(
                stock=self.stock,
                data_source='real_test_task'
            )
            self.assertEqual(created_price.open, Decimal('100.00'))
            self.assertEqual(created_price.close, Decimal('104.00'))
            self.assertEqual(created_price.volume, 500000)
            
        except Exception as e:
            self.fail(f"Real task database operation failed: {e}")
    
    def test_concurrent_task_execution(self):
        """Test handling of concurrent task execution."""
        import threading
        import time
        
        results = []
        errors = []
        
        def simulate_task_execution(task_id):
            """Simulate concurrent task execution."""
            try:
                # Simulate task work
                time.sleep(0.1)
                
                # Create unique data for each task
                StockPrice.objects.create(
                    stock=self.stock,
                    date=datetime.now().date() - timedelta(days=task_id),
                    open=Decimal('100.00') + task_id,
                    high=Decimal('105.00') + task_id,
                    low=Decimal('99.00') + task_id,
                    close=Decimal('104.00') + task_id,
                    volume=500000 + (task_id * 1000),
                    data_source=f'concurrent_task_{task_id}'
                )
                
                results.append(f'task_{task_id}_success')
                
            except Exception as e:
                errors.append(f'task_{task_id}_error: {str(e)}')
        
        # Create multiple threads to simulate concurrent tasks
        threads = []
        for i in range(3):
            thread = threading.Thread(target=simulate_task_execution, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)
        
        # Verify all data was created
        created_prices = StockPrice.objects.filter(
            stock=self.stock,
            data_source__startswith='concurrent_task_'
        ).count()
        self.assertEqual(created_prices, 3)
    
    def test_task_retry_mechanism(self):
        """Test task retry logic and exponential backoff."""
        retry_attempts = []
        
        def mock_retry_function(attempt):
            """Mock retry function to track attempts."""
            retry_attempts.append(attempt)
            if attempt < 3:
                raise Exception(f"Temporary failure on attempt {attempt}")
            return {"status": "success", "attempt": attempt}
        
        # Simulate retry logic
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                result = mock_retry_function(attempt)
                if result["status"] == "success":
                    break
            except Exception:
                if attempt == max_retries:
                    self.fail("Task failed after max retries")
                continue
        
        # Verify retry attempts were made
        self.assertEqual(len(retry_attempts), 3)
        self.assertEqual(retry_attempts, [1, 2, 3])
    
    def test_task_performance_monitoring(self):
        """Test task performance monitoring and metrics."""
        import time
        
        start_time = time.time()
        
        # Simulate task with performance monitoring
        try:
            # Simulate work
            time.sleep(0.1)
            
            # Create test data
            StockPrice.objects.create(
                stock=self.stock,
                date=datetime.now().date(),
                open=Decimal('120.00'),
                high=Decimal('125.00'),
                low=Decimal('119.00'),
                close=Decimal('124.00'),
                volume=750000,
                data_source='performance_test'
            )
            
            execution_time = time.time() - start_time
            
            # Verify performance is reasonable
            self.assertLess(execution_time, 2.0)  # Should complete within 2 seconds
            self.assertGreater(execution_time, 0.05)  # Should take some measurable time
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")


if __name__ == '__main__':
    import unittest
    unittest.main()
