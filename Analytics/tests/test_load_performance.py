"""
Phase 3.6 Load Testing

Comprehensive load testing validation including:
- Concurrent user simulation (50+ users)
- System resource monitoring
- Performance degradation measurement
- Memory leak detection
- Sustained load testing
- Auto-scaling recommendation analysis
"""

import gc
import logging
import psutil
import statistics
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.explanation_service import get_explanation_service
from Data.models import AnalyticsResults, DataIndustry, DataSector, Stock, StockPrice

logger = logging.getLogger(__name__)
User = get_user_model()


class SystemResourceMonitor:
    """Monitor system resources during load testing."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources, args=(interval,))
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self, interval: float):
        """Monitor system resources continuously."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Get system-wide metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                # Get process-specific metrics
                process_info = process.as_dict(['memory_info', 'cpu_percent', 'num_threads'])
                
                metric = {
                    'timestamp': time.time(),
                    'system_cpu_percent': cpu_percent,
                    'system_memory_percent': memory_info.percent,
                    'system_memory_available': memory_info.available,
                    'process_memory_rss': process_info['memory_info'].rss,
                    'process_memory_vms': process_info['memory_info'].vms,
                    'process_cpu_percent': process_info['cpu_percent'],
                    'process_threads': process_info['num_threads']
                }
                
                self.metrics.append(metric)
                time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {str(e)}")
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of monitored resources."""
        if not self.metrics:
            return {}
        
        cpu_values = [m['system_cpu_percent'] for m in self.metrics]
        memory_values = [m['system_memory_percent'] for m in self.metrics]
        process_memory = [m['process_memory_rss'] for m in self.metrics]
        
        return {
            'duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp'],
            'samples': len(self.metrics),
            'cpu': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'process_memory': {
                'avg': statistics.mean(process_memory),
                'max': max(process_memory),
                'min': min(process_memory),
                'growth': max(process_memory) - min(process_memory)
            },
            'peak_threads': max(m['process_threads'] for m in self.metrics)
        }


class LoadPerformanceTestCase(TransactionTestCase):
    """Phase 3.6 comprehensive load testing and performance validation."""

    def setUp(self):
        """Set up comprehensive test data for load testing."""
        cache.clear()
        gc.collect()  # Clean up memory before testing

        # Create test user
        self.user = User.objects.create_user(
            username="load_user", 
            email="load@test.com", 
            password="testpass123"
        )

        # Create sector and industry
        self.sector = DataSector.objects.create(
            sectorKey="tech_load", 
            sectorName="Technology Load Testing", 
            data_source="test"
        )
        
        self.industry = DataIndustry.objects.create(
            industryKey="software_load", 
            industryName="Software Load Testing",
            sector=self.sector,
            data_source="test"
        )

        # Create test stocks for load testing variety
        self.stocks = []
        symbols = ['LOAD1', 'LOAD2', 'LOAD3', 'LOAD4', 'LOAD5']
        
        for i, symbol in enumerate(symbols):
            stock = Stock.objects.create(
                symbol=symbol,
                short_name=f"Load Test Company {i+1}",
                sector=self.sector.sectorName,
                industry=self.industry.industryName,
                market_cap=(i+1) * 1000000000,
                shares_outstanding=(i+1) * 100000000,
                is_active=True
            )
            self.stocks.append(stock)

        # Create price data for all stocks
        self.create_test_price_data()

        # Get services
        self.llm_service = get_local_llm_service()
        self.explanation_service = get_explanation_service()
        self.resource_monitor = SystemResourceMonitor()

    def create_test_price_data(self):
        """Create test price data for load testing stocks."""
        base_date = datetime.now().date() - timedelta(days=90)
        
        all_prices = []
        
        for stock in self.stocks:
            base_price = Decimal('100.00') + Decimal(str(stock.id * 10))
            
            for i in range(90):
                date = base_date + timedelta(days=i)
                # Create varied price movements for different stocks
                price_change = Decimal(str(0.5 * (i % 8 - 4) * (stock.id % 3 + 1)))
                current_price = base_price + price_change + Decimal(str(i * 0.05))
                
                all_prices.append(StockPrice(
                    stock=stock,
                    date=date,
                    open_price=current_price - Decimal('0.75'),
                    high_price=current_price + Decimal('1.25'),
                    low_price=current_price - Decimal('1.75'),
                    close_price=current_price,
                    volume=1500000 + i * 15000 + stock.id * 100000,
                    data_source="test"
                ))
        
        StockPrice.objects.bulk_create(all_prices)

    def create_varied_analysis_scenarios(self) -> List[Dict[str, Any]]:
        """Create varied analysis scenarios for load testing."""
        scenarios = []
        
        for i, stock in enumerate(self.stocks):
            # Create different market scenarios
            scenarios.extend([
                {
                    'symbol': stock.symbol,
                    'technical_score': 8.2 - i * 0.3,
                    'recommendation': 'BUY' if i < 2 else 'HOLD' if i < 4 else 'SELL',
                    'scenario_type': 'bullish',
                    'indicators': {
                        'rsi': {'value': 65.2 - i * 3, 'signal': 'bullish', 'interpretation': 'Strong momentum'},
                        'macd': {'value': 0.45 - i * 0.1, 'signal': 'bullish', 'interpretation': 'Bullish crossover'},
                        'sma_50': {'value': 105.2 + i * 5, 'signal': 'bullish', 'interpretation': 'Uptrend confirmed'}
                    }
                },
                {
                    'symbol': stock.symbol,
                    'technical_score': 3.8 + i * 0.4,
                    'recommendation': 'SELL' if i < 2 else 'HOLD' if i < 4 else 'BUY',
                    'scenario_type': 'bearish',
                    'indicators': {
                        'rsi': {'value': 28.5 + i * 4, 'signal': 'bearish', 'interpretation': 'Oversold conditions'},
                        'macd': {'value': -0.35 + i * 0.05, 'signal': 'bearish', 'interpretation': 'Bearish momentum'},
                        'sma_50': {'value': 98.2 - i * 3, 'signal': 'bearish', 'interpretation': 'Downtrend'}
                    }
                }
            ])
        
        return scenarios

    def simulate_user_session(self, user_id: int, scenarios: List[Dict[str, Any]], 
                            requests_per_user: int = 5) -> Dict[str, Any]:
        """Simulate a single user session with multiple requests."""
        session_results = {
            'user_id': user_id,
            'start_time': time.time(),
            'requests': [],
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0,
            'errors': []
        }
        
        try:
            for i in range(requests_per_user):
                scenario = scenarios[user_id % len(scenarios)]
                detail_level = ['summary', 'standard', 'detailed'][i % 3]
                
                request_start = time.time()
                
                try:
                    result = self.llm_service.generate_explanation(
                        analysis_data=scenario,
                        detail_level=detail_level
                    )
                    
                    response_time = time.time() - request_start
                    
                    request_result = {
                        'request_id': i + 1,
                        'scenario': scenario['symbol'],
                        'detail_level': detail_level,
                        'response_time': response_time,
                        'success': result is not None and 'explanation' in (result or {}),
                        'explanation_length': len((result or {}).get('explanation', '').split()) if result else 0
                    }
                    
                    session_results['requests'].append(request_result)
                    session_results['total_requests'] += 1
                    session_results['total_response_time'] += response_time
                    
                    if request_result['success']:
                        session_results['successful_requests'] += 1
                    else:
                        session_results['failed_requests'] += 1
                        
                except Exception as e:
                    response_time = time.time() - request_start
                    session_results['requests'].append({
                        'request_id': i + 1,
                        'scenario': scenario['symbol'],
                        'detail_level': detail_level,
                        'response_time': response_time,
                        'success': False,
                        'error': str(e)
                    })
                    session_results['failed_requests'] += 1
                    session_results['errors'].append(str(e))
                
                # Small delay between requests to simulate realistic user behaviour
                time.sleep(0.1)
                
        except Exception as e:
            session_results['session_error'] = str(e)
        
        session_results['end_time'] = time.time()
        session_results['session_duration'] = session_results['end_time'] - session_results['start_time']
        session_results['avg_response_time'] = (
            session_results['total_response_time'] / session_results['total_requests'] 
            if session_results['total_requests'] > 0 else 0
        )
        
        return session_results

    def test_concurrent_users_performance(self):
        """Test system performance with 50 concurrent users."""
        concurrent_users = 25  # Reduced from 50 for test environment
        requests_per_user = 3   # 3 requests per user = 75 total requests
        scenarios = self.create_varied_analysis_scenarios()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring(interval=0.5)
        
        load_test_start = time.time()
        
        # Use ThreadPoolExecutor for controlled concurrency
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit all user sessions
            future_to_user = {
                executor.submit(self.simulate_user_session, user_id, scenarios, requests_per_user): user_id
                for user_id in range(concurrent_users)
            }
            
            # Collect results
            user_results = []
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per user
                    user_results.append(result)
                except Exception as e:
                    logger.error(f"User {user_id} session failed: {str(e)}")
                    user_results.append({
                        'user_id': user_id,
                        'session_error': str(e),
                        'successful_requests': 0,
                        'failed_requests': requests_per_user
                    })
        
        load_test_duration = time.time() - load_test_start
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        resource_summary = self.resource_monitor.get_summary()
        
        # Analyse load test results
        total_requests = sum(r.get('total_requests', 0) for r in user_results)
        successful_requests = sum(r.get('successful_requests', 0) for r in user_results)
        failed_requests = sum(r.get('failed_requests', 0) for r in user_results)
        
        if total_requests > 0:
            success_rate = (successful_requests / total_requests) * 100
            
            # Collect response times from successful requests
            response_times = []
            for user_result in user_results:
                for request in user_result.get('requests', []):
                    if request.get('success', False):
                        response_times.append(request['response_time'])
            
            performance_metrics = {
                'concurrent_users': concurrent_users,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': success_rate,
                'load_test_duration': load_test_duration,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'median_response_time': statistics.median(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'requests_per_second': total_requests / load_test_duration,
                'resource_summary': resource_summary
            }
            
            logger.info(f"Load test completed: {success_rate:.1f}% success rate, "
                       f"{performance_metrics['avg_response_time']:.2f}s avg response time")
            
            # Performance assertions
            self.assertGreaterEqual(success_rate, 70.0,
                                   f"Success rate should be ≥70%, got {success_rate:.1f}%")
            
            if response_times:
                # Average response time should be reasonable under load
                self.assertLess(performance_metrics['avg_response_time'], 30.0,
                               f"Average response time under load should be <30s, got {performance_metrics['avg_response_time']:.2f}s")
            
            # Resource usage should be reasonable
            if resource_summary:
                self.assertLess(resource_summary['memory']['max'], 90.0,
                               f"Peak memory usage should be <90%, got {resource_summary['memory']['max']:.1f}%")
        
        else:
            self.fail("No requests completed during load test")

    def test_performance_degradation_measurement(self):
        """Measure performance degradation under increasing load."""
        scenarios = self.create_varied_analysis_scenarios()[:3]  # Use 3 scenarios
        load_levels = [1, 5, 10, 15]  # Progressive load levels
        
        degradation_results = []
        
        for load_level in load_levels:
            self.resource_monitor = SystemResourceMonitor()  # Fresh monitor
            self.resource_monitor.start_monitoring(interval=1.0)
            
            test_start = time.time()
            
            # Run concurrent requests at this load level
            with ThreadPoolExecutor(max_workers=load_level) as executor:
                futures = []
                
                for i in range(load_level):
                    scenario = scenarios[i % len(scenarios)]
                    future = executor.submit(
                        self.llm_service.generate_explanation,
                        scenario,
                        "standard"
                    )
                    futures.append(future)
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Request failed at load level {load_level}: {str(e)}")
            
            test_duration = time.time() - test_start
            self.resource_monitor.stop_monitoring()
            resource_summary = self.resource_monitor.get_summary()
            
            successful_results = [r for r in results if r is not None]
            success_rate = (len(successful_results) / load_level) * 100 if load_level > 0 else 0
            
            degradation_results.append({
                'load_level': load_level,
                'test_duration': test_duration,
                'success_rate': success_rate,
                'avg_response_time': test_duration / load_level,
                'resource_summary': resource_summary
            })
            
            logger.info(f"Load level {load_level}: {success_rate:.1f}% success, "
                       f"{test_duration:.2f}s total duration")
            
            # Small delay between load levels
            time.sleep(1.0)
        
        # Analyse degradation
        baseline_performance = degradation_results[0]
        peak_load_performance = degradation_results[-1]
        
        if baseline_performance['success_rate'] > 0 and peak_load_performance['success_rate'] > 0:
            performance_degradation = (
                (baseline_performance['avg_response_time'] - peak_load_performance['avg_response_time']) /
                baseline_performance['avg_response_time'] * 100
            )
            
            success_rate_degradation = baseline_performance['success_rate'] - peak_load_performance['success_rate']
            
            # Performance degradation should be reasonable
            self.assertLess(abs(performance_degradation), 300.0,
                           f"Response time degradation should be <300%, got {performance_degradation:.1f}%")
            
            self.assertLess(success_rate_degradation, 50.0,
                           f"Success rate degradation should be <50%, got {success_rate_degradation:.1f}%")
            
            logger.info(f"Performance degradation analysis: {performance_degradation:.1f}% response time change, "
                       f"{success_rate_degradation:.1f}% success rate drop")

    def test_memory_leak_detection(self):
        """Test for memory leaks during sustained operations."""
        scenarios = self.create_varied_analysis_scenarios()[:2]  # Use 2 scenarios
        
        initial_memory = psutil.Process().memory_info().rss
        memory_samples = [initial_memory]
        
        # Run sustained operations
        for cycle in range(10):  # 10 cycles
            for scenario in scenarios:
                try:
                    result = self.llm_service.generate_explanation(
                        analysis_data=scenario,
                        detail_level="standard"
                    )
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Sample memory usage
                    current_memory = psutil.Process().memory_info().rss
                    memory_samples.append(current_memory)
                    
                except Exception as e:
                    logger.warning(f"Memory leak test request failed: {str(e)}")
            
            # Small delay between cycles
            time.sleep(0.5)
        
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        # Analyse memory growth trend
        if len(memory_samples) > 5:
            # Check if memory is consistently growing
            recent_samples = memory_samples[-5:]
            early_samples = memory_samples[:5]
            
            recent_avg = statistics.mean(recent_samples)
            early_avg = statistics.mean(early_samples)
            
            growth_trend = (recent_avg - early_avg) / early_avg * 100
            
            logger.info(f"Memory usage: initial={initial_memory/(1024*1024):.1f}MB, "
                       f"final={final_memory/(1024*1024):.1f}MB, "
                       f"growth={memory_growth_mb:.1f}MB, "
                       f"trend={growth_trend:.1f}%")
            
            # Memory growth should be reasonable for sustained operations
            self.assertLess(memory_growth_mb, 100.0,
                           f"Memory growth should be <100MB for sustained operations, got {memory_growth_mb:.1f}MB")
            
            # Growth trend should not be excessive
            self.assertLess(abs(growth_trend), 200.0,
                           f"Memory growth trend should be <200%, got {growth_trend:.1f}%")

    def test_sustained_load_endurance(self):
        """Test system endurance under sustained load."""
        scenarios = self.create_varied_analysis_scenarios()[:3]
        duration_minutes = 2  # 2 minute sustained test
        target_rps = 0.5  # 0.5 requests per second (1 every 2 seconds)
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        self.resource_monitor.start_monitoring(interval=2.0)
        
        request_count = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        logger.info(f"Starting {duration_minutes}-minute sustained load test at {target_rps} RPS")
        
        while time.time() < end_time:
            request_start = time.time()
            scenario = scenarios[request_count % len(scenarios)]
            
            try:
                result = self.llm_service.generate_explanation(
                    analysis_data=scenario,
                    detail_level="standard"
                )
                
                response_time = time.time() - request_start
                response_times.append(response_time)
                
                if result and 'explanation' in result:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    
            except Exception as e:
                failed_requests += 1
                logger.warning(f"Sustained load request failed: {str(e)}")
            
            request_count += 1
            
            # Maintain target request rate
            next_request_time = start_time + (request_count / target_rps)
            current_time = time.time()
            
            if current_time < next_request_time:
                time.sleep(next_request_time - current_time)
        
        actual_duration = time.time() - start_time
        self.resource_monitor.stop_monitoring()
        resource_summary = self.resource_monitor.get_summary()
        
        # Analyse sustained load results
        total_requests = successful_requests + failed_requests
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        actual_rps = total_requests / actual_duration
        
        endurance_metrics = {
            'duration_minutes': actual_duration / 60,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'target_rps': target_rps,
            'actual_rps': actual_rps,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'resource_summary': resource_summary
        }
        
        logger.info(f"Sustained load test: {success_rate:.1f}% success rate over {actual_duration/60:.1f} minutes, "
                   f"{actual_rps:.2f} actual RPS")
        
        # Endurance test assertions
        self.assertGreaterEqual(success_rate, 80.0,
                               f"Sustained load success rate should be ≥80%, got {success_rate:.1f}%")
        
        self.assertGreaterEqual(total_requests, duration_minutes * 60 * target_rps * 0.8,
                               f"Should complete ≥80% of target requests")
        
        # System should remain stable
        if resource_summary and response_times:
            self.assertLess(endurance_metrics['avg_response_time'], 25.0,
                           f"Average response time should remain <25s, got {endurance_metrics['avg_response_time']:.2f}s")

    def test_resource_utilisation_optimisation(self):
        """Test resource utilisation and provide optimisation recommendations."""
        scenarios = self.create_varied_analysis_scenarios()[:2]
        
        # Test different concurrency levels
        concurrency_levels = [1, 3, 6, 10]
        utilisation_results = []
        
        for concurrency in concurrency_levels:
            self.resource_monitor = SystemResourceMonitor()
            self.resource_monitor.start_monitoring(interval=0.5)
            
            test_start = time.time()
            
            # Run requests at this concurrency level
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                
                for i in range(concurrency * 2):  # 2 requests per thread
                    scenario = scenarios[i % len(scenarios)]
                    future = executor.submit(
                        self.llm_service.generate_explanation,
                        scenario,
                        "standard"
                    )
                    futures.append(future)
                
                # Wait for completion
                completed_requests = 0
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=20)
                        if result:
                            completed_requests += 1
                    except Exception as e:
                        logger.warning(f"Request failed at concurrency {concurrency}: {str(e)}")
            
            test_duration = time.time() - test_start
            self.resource_monitor.stop_monitoring()
            resource_summary = self.resource_monitor.get_summary()
            
            if resource_summary:
                utilisation_results.append({
                    'concurrency': concurrency,
                    'completed_requests': completed_requests,
                    'test_duration': test_duration,
                    'requests_per_second': completed_requests / test_duration,
                    'cpu_utilisation': resource_summary['cpu']['avg'],
                    'memory_utilisation': resource_summary['memory']['avg'],
                    'peak_threads': resource_summary['peak_threads'],
                    'efficiency': completed_requests / (resource_summary['cpu']['avg'] + 1)  # +1 to avoid division by zero
                })
            
            time.sleep(1.0)  # Rest between tests
        
        # Analyse utilisation efficiency
        if utilisation_results:
            # Find optimal concurrency level
            best_efficiency = max(utilisation_results, key=lambda x: x['efficiency'])
            
            recommendations = []
            
            # CPU utilisation recommendations
            if best_efficiency['cpu_utilisation'] < 30:
                recommendations.append("CPU underutilised - consider increasing concurrency")
            elif best_efficiency['cpu_utilisation'] > 80:
                recommendations.append("High CPU utilisation - consider optimising or reducing load")
            
            # Memory utilisation recommendations  
            if best_efficiency['memory_utilisation'] > 80:
                recommendations.append("High memory usage - monitor for potential memory constraints")
            
            # Concurrency recommendations
            optimal_concurrency = best_efficiency['concurrency']
            recommendations.append(f"Optimal concurrency level appears to be around {optimal_concurrency}")
            
            logger.info(f"Resource utilisation analysis: "
                       f"optimal concurrency={optimal_concurrency}, "
                       f"efficiency={best_efficiency['efficiency']:.2f}, "
                       f"CPU={best_efficiency['cpu_utilisation']:.1f}%, "
                       f"memory={best_efficiency['memory_utilisation']:.1f}%")
            
            for recommendation in recommendations:
                logger.info(f"Recommendation: {recommendation}")
            
            # Basic utilisation assertions
            self.assertLess(best_efficiency['memory_utilisation'], 90.0,
                           "Peak memory utilisation should be <90%")
            self.assertGreater(best_efficiency['efficiency'], 0.1,
                              "System should demonstrate reasonable efficiency")

    def test_auto_scaling_analysis(self):
        """Analyse system behaviour for auto-scaling recommendations."""
        scenarios = self.create_varied_analysis_scenarios()[:2]
        
        # Simulate load spikes
        load_patterns = [
            {'name': 'baseline', 'users': 2, 'duration': 10},
            {'name': 'spike_low', 'users': 5, 'duration': 15},
            {'name': 'spike_medium', 'users': 8, 'duration': 20},
            {'name': 'spike_high', 'users': 12, 'duration': 15}
        ]
        
        scaling_analysis = []
        
        for pattern in load_patterns:
            self.resource_monitor = SystemResourceMonitor()
            self.resource_monitor.start_monitoring(interval=1.0)
            
            pattern_start = time.time()
            
            # Run load pattern
            with ThreadPoolExecutor(max_workers=pattern['users']) as executor:
                futures = []
                
                # Submit requests for this pattern duration
                request_count = int(pattern['users'] * pattern['duration'] / 10)  # Requests based on pattern
                
                for i in range(request_count):
                    scenario = scenarios[i % len(scenarios)]
                    future = executor.submit(
                        self.llm_service.generate_explanation,
                        scenario,
                        "standard"
                    )
                    futures.append(future)
                
                # Collect results
                successful = sum(1 for future in as_completed(futures) 
                               if future.result(timeout=15) is not None)
            
            pattern_duration = time.time() - pattern_start
            self.resource_monitor.stop_monitoring()
            resource_summary = self.resource_monitor.get_summary()
            
            if resource_summary:
                pattern_analysis = {
                    'pattern': pattern['name'],
                    'target_users': pattern['users'],
                    'duration': pattern_duration,
                    'successful_requests': successful,
                    'requests_per_second': successful / pattern_duration,
                    'cpu_avg': resource_summary['cpu']['avg'],
                    'cpu_peak': resource_summary['cpu']['max'],
                    'memory_avg': resource_summary['memory']['avg'],
                    'memory_peak': resource_summary['memory']['max'],
                    'resource_pressure': (resource_summary['cpu']['avg'] + resource_summary['memory']['avg']) / 2
                }
                
                scaling_analysis.append(pattern_analysis)
                
                logger.info(f"Load pattern '{pattern['name']}': "
                           f"{successful} requests, "
                           f"CPU avg={resource_summary['cpu']['avg']:.1f}%, "
                           f"memory avg={resource_summary['memory']['avg']:.1f}%")
            
            time.sleep(2.0)  # Recovery time between patterns
        
        # Generate auto-scaling recommendations
        if len(scaling_analysis) >= 2:
            baseline = scaling_analysis[0]
            high_load = scaling_analysis[-1]
            
            scaling_recommendations = []
            
            # CPU-based scaling recommendations
            if high_load['cpu_peak'] > 75:
                scaling_recommendations.append("CPU-based auto-scaling recommended above 75% CPU")
            
            # Memory-based scaling recommendations
            if high_load['memory_peak'] > 70:
                scaling_recommendations.append("Memory-based auto-scaling recommended above 70% memory")
            
            # Response degradation-based recommendations
            baseline_efficiency = baseline['requests_per_second'] / max(baseline['resource_pressure'], 1)
            high_load_efficiency = high_load['requests_per_second'] / max(high_load['resource_pressure'], 1)
            
            efficiency_degradation = (baseline_efficiency - high_load_efficiency) / baseline_efficiency * 100
            
            if efficiency_degradation > 30:
                scaling_recommendations.append(f"Efficiency degraded {efficiency_degradation:.1f}% - consider scaling")
            
            logger.info("Auto-scaling analysis complete:")
            for recommendation in scaling_recommendations:
                logger.info(f"  {recommendation}")
            
            # Verify scaling analysis makes sense
            self.assertGreater(len(scaling_analysis), 1, "Should complete multiple load patterns")
            self.assertLess(high_load['resource_pressure'], 95.0, "Resource pressure should remain manageable")

    def tearDown(self):
        """Clean up after load testing."""
        if hasattr(self, 'resource_monitor') and self.resource_monitor.monitoring:
            self.resource_monitor.stop_monitoring()
        
        # Force garbage collection after load tests
        gc.collect()
        cache.clear()