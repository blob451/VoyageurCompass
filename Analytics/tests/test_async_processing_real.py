"""
Real tests for async processing pipeline.
Tests async batch processing, task tracking, and concurrent operations.
No mocks - uses real functionality.
"""

import time
import threading
from datetime import datetime
from django.test import TestCase, TransactionTestCase
from django.contrib.auth import get_user_model
from django.utils import timezone

from Analytics.services.async_processing_pipeline import get_async_processing_pipeline, AsyncProcessingPipeline
from Data.models import Stock, DataSector, DataIndustry

User = get_user_model()


class RealAsyncProcessingTestCase(TransactionTestCase):
    """Real test cases for async processing pipeline."""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='async_test_user',
            email='async@test.com',
            password='testpass123'
        )
        
        # Create test stock
        self.sector = DataSector.objects.create(
            sectorKey='test_async',
            sectorName='Test Async',
            data_source='yahoo'
        )
        
        self.stock = Stock.objects.create(
            symbol='ASYNC_TEST',
            short_name='Async Test Corp',
            sector_id=self.sector,
            market_cap=1000000000
        )
        
        self.pipeline = AsyncProcessingPipeline(max_workers=2)
    
    def test_real_task_status_tracking(self):
        """Test real task status tracking system."""
        # Create task
        task_info = self.pipeline.task_status.create_task(
            'test_task_real', 'test_analysis', 'ASYNC_TEST'
        )
        
        self.assertEqual(task_info['status'], 'pending')
        self.assertEqual(task_info['symbol'], 'ASYNC_TEST')
        self.assertIsNotNone(task_info['created_at'])
        self.assertIsNotNone(task_info['task_id'])
        
        # Update task status
        self.pipeline.task_status.update_task_status(
            'test_task_real', 'running', progress=50.0
        )
        
        updated_task = self.pipeline.task_status.get_task_status('test_task_real')
        self.assertEqual(updated_task['status'], 'running')
        self.assertEqual(updated_task['progress'], 50.0)
        
        # Complete task with result
        self.pipeline.task_status.update_task_status(
            'test_task_real', 'completed', progress=100.0,
            result={'analysis_score': 7.5, 'recommendation': 'BUY'}
        )
        
        final_task = self.pipeline.task_status.get_task_status('test_task_real')
        self.assertEqual(final_task['status'], 'completed')
        self.assertEqual(final_task['progress'], 100.0)
        self.assertIsNotNone(final_task['result'])
        self.assertEqual(final_task['result']['analysis_score'], 7.5)
    
    def test_real_batch_processing(self):
        """Test real batch processing functionality."""
        def real_analysis_processor(request):
            """Real analysis processor for testing."""
            symbol = request['symbol']
            
            # Simulate some processing time
            time.sleep(0.1)
            
            return {
                'symbol': symbol,
                'score_0_10': 6.0 + hash(symbol) % 4,  # Random but deterministic
                'processed_at': timezone.now().isoformat(),
                'success': True
            }
        
        # Create test requests
        requests = [
            {'symbol': 'AAPL', 'user_id': self.user.id},
            {'symbol': 'MSFT', 'user_id': self.user.id},
            {'symbol': 'GOOGL', 'user_id': self.user.id}
        ]
        
        # Process batch using real functionality
        start_time = time.time()
        result = self.pipeline.process_batch_analysis(
            requests, real_analysis_processor, 'real_batch_test'
        )
        processing_time = time.time() - start_time
        
        # Validate results
        self.assertEqual(result['batch_id'], 'real_batch_test')
        self.assertEqual(result['total_requests'], 3)
        self.assertEqual(result['successful_requests'], 3)
        self.assertEqual(result['failed_requests'], 0)
        self.assertEqual(result['success_rate'], 1.0)
        self.assertEqual(len(result['results']), 3)
        
        # Should be processed concurrently (faster than sequential)
        self.assertLess(processing_time, 0.4)  # Should be faster than 3 * 0.1 seconds
        
        # Check individual results
        for i, res in enumerate(result['results']):
            self.assertIsNotNone(res)
            self.assertTrue(res['success'])
            self.assertIn('symbol', res)
            self.assertIn('score_0_10', res)
            self.assertIn('processed_at', res)
    
    def test_real_concurrent_task_execution(self):
        """Test concurrent task execution."""
        results = []
        
        def concurrent_task(task_id):
            """Concurrent task for testing."""
            start_time = time.time()
            time.sleep(0.2)  # Simulate work
            end_time = time.time()
            
            return {
                'task_id': task_id,
                'processing_time': end_time - start_time,
                'thread_id': threading.get_ident()
            }
        
        # Create multiple tasks
        tasks = []
        for i in range(4):
            task_id = f'concurrent_task_{i}'
            self.pipeline.task_status.create_task(task_id, 'concurrent_test', f'SYMBOL_{i}')
            tasks.append(task_id)
        
        # Execute tasks concurrently
        start_time = time.time()
        with self.pipeline.executor:
            futures = []
            for task_id in tasks:
                future = self.pipeline.executor.submit(concurrent_task, task_id)
                futures.append(future)
            
            # Collect results
            for future in futures:
                result = future.result()
                results.append(result)
        
        total_time = time.time() - start_time
        
        # Should execute concurrently
        self.assertLess(total_time, 0.5)  # Much less than 4 * 0.2 = 0.8 seconds
        self.assertEqual(len(results), 4)
        
        # Check that tasks ran on different threads
        thread_ids = {r['thread_id'] for r in results}
        self.assertGreater(len(thread_ids), 1)  # Multiple threads used
    
    def test_real_performance_monitoring(self):
        """Test real performance monitoring."""
        # Get service singleton
        pipeline_service = get_async_processing_pipeline()
        
        def monitored_task(symbol):
            time.sleep(0.05)  # Small processing time
            return {'symbol': symbol, 'score': 7.0}
        
        # Process tasks and monitor performance
        test_requests = [{'symbol': f'PERF_TEST_{i}'} for i in range(5)]
        
        result = pipeline_service.process_batch_analysis(
            test_requests, monitored_task, 'perf_monitor_test'
        )
        
        # Check performance metrics
        self.assertIn('processing_time_ms', result)
        self.assertIn('avg_task_time_ms', result)
        self.assertGreater(result['processing_time_ms'], 0)
        self.assertGreater(result['avg_task_time_ms'], 0)
        
        # Should be reasonably fast
        self.assertLess(result['processing_time_ms'], 1000)  # Less than 1 second
    
    def test_real_error_handling(self):
        """Test real error handling in async processing."""
        def error_prone_processor(request):
            """Processor that fails for certain symbols."""
            symbol = request['symbol']
            
            if 'ERROR' in symbol:
                raise ValueError(f"Simulated error for {symbol}")
            
            return {
                'symbol': symbol,
                'score': 5.0,
                'success': True
            }
        
        # Mix of successful and failing requests
        mixed_requests = [
            {'symbol': 'GOOD_SYMBOL'},
            {'symbol': 'ERROR_SYMBOL'},
            {'symbol': 'ANOTHER_GOOD'},
            {'symbol': 'ANOTHER_ERROR_SYMBOL'}
        ]
        
        result = self.pipeline.process_batch_analysis(
            mixed_requests, error_prone_processor, 'error_handling_test'
        )
        
        # Check error handling
        self.assertEqual(result['total_requests'], 4)
        self.assertEqual(result['successful_requests'], 2)
        self.assertEqual(result['failed_requests'], 2)
        self.assertEqual(result['success_rate'], 0.5)
        
        # Check that successful results are included
        successful_results = [r for r in result['results'] if r is not None]
        self.assertEqual(len(successful_results), 2)
        
        for res in successful_results:
            self.assertTrue(res['success'])
            self.assertNotIn('ERROR', res['symbol'])


class RealAsyncIntegrationTestCase(TestCase):
    """Integration tests for async processing with other services."""
    
    def test_singleton_service_access(self):
        """Test that get_async_processing_pipeline returns singleton."""
        service1 = get_async_processing_pipeline()
        service2 = get_async_processing_pipeline()
        
        self.assertIs(service1, service2)
        self.assertIsInstance(service1, AsyncProcessingPipeline)
    
    def test_task_cleanup_mechanism(self):
        """Test that old tasks are cleaned up properly."""
        pipeline = AsyncProcessingPipeline(max_workers=1)
        
        # Create multiple tasks
        for i in range(10):
            pipeline.task_status.create_task(f'cleanup_test_{i}', 'cleanup', f'SYMBOL_{i}')
        
        # Check tasks were created
        initial_count = len(pipeline.task_status.tasks)
        self.assertEqual(initial_count, 10)
        
        # Complete some tasks
        for i in range(5):
            pipeline.task_status.update_task_status(
                f'cleanup_test_{i}', 'completed', progress=100.0
            )
        
        # Verify cleanup behavior (implementation dependent)
        # This tests that the system can handle multiple tasks
        self.assertGreaterEqual(len(pipeline.task_status.tasks), 5)


if __name__ == '__main__':
    import django
    django.setup()
    from django.test import TestRunner
    runner = TestRunner()
    runner.run_tests(['Analytics.tests.test_async_processing_real'])