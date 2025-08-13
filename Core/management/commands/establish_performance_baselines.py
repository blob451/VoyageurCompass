"""
Django management command to establish API performance baselines.
This command tests critical API endpoints and establishes baseline metrics.
"""

import json
import time
import statistics
from datetime import datetime
from django.core.management.base import BaseCommand
from django.test import Client
from django.contrib.auth.models import User
from django.db import connection
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken


class Command(BaseCommand):
    """
    Management command to establish performance baselines for critical API endpoints.
    """
    
    help = 'Establish performance baselines for critical API endpoints'

    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            '--runs',
            type=int,
            default=10,
            help='Number of test runs per endpoint (default: 10)'
        )
        parser.add_argument(
            '--output',
            type=str,
            default='performance_baselines.json',
            help='Output file for baseline results (default: performance_baselines.json)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )

    def handle(self, *args, **options):
        """Main command handler."""
        self.runs = options['runs']
        self.output_file = options['output']
        self.verbose = options['verbose']
        
        self.stdout.write(
            self.style.SUCCESS(
                f'🚀 Establishing performance baselines with {self.runs} runs per endpoint...'
            )
        )
        
        # Initialize test client
        self.client = APIClient()
        
        # Create test user for authenticated endpoints
        self.test_user = self._create_test_user()
        
        # Define critical endpoints to test
        endpoints = [
            # Authentication endpoints
            {
                'name': 'auth_login',
                'method': 'POST',
                'url': '/api/auth/login/',
                'data': {'username': 'testuser', 'password': 'testpass123'},
                'auth_required': False,
                'budget_ms': 500
            },
            {
                'name': 'auth_refresh',
                'method': 'POST', 
                'url': '/api/auth/refresh/',
                'data': None,  # Will be set dynamically
                'auth_required': False,
                'budget_ms': 300
            },
            
            # Core API endpoints
            {
                'name': 'api_health',
                'method': 'GET',
                'url': '/api/health/',
                'data': None,
                'auth_required': False,
                'budget_ms': 200
            },
            
            # Data endpoints
            {
                'name': 'stocks_list',
                'method': 'GET',
                'url': '/api/data/stocks/',
                'data': None,
                'auth_required': True,
                'budget_ms': 1000
            },
            {
                'name': 'portfolio_list',
                'method': 'GET',
                'url': '/api/data/portfolio/',
                'data': None,
                'auth_required': True,
                'budget_ms': 800
            },
            
            # Analytics endpoints
            {
                'name': 'analytics_indicators',
                'method': 'GET',
                'url': '/api/analytics/indicators/',
                'data': None,
                'auth_required': True,
                'budget_ms': 2000
            }
        ]
        
        # Run performance tests
        results = {}
        for endpoint in endpoints:
            self.stdout.write(f'Testing {endpoint["name"]}...')
            results[endpoint['name']] = self._test_endpoint(endpoint)
            
        # Generate baseline report
        baseline_report = self._generate_baseline_report(results)
        
        # Save results
        self._save_results(baseline_report)
        
        # Display summary
        self._display_summary(baseline_report)
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Performance baselines established and saved to {self.output_file}')
        )

    def _create_test_user(self):
        """Create a test user for authenticated endpoints."""
        try:
            user = User.objects.get(username='testuser')
        except User.DoesNotExist:
            user = User.objects.create_user(
                username='testuser',
                email='test@example.com',
                password='testpass123'
            )
        return user

    def _get_auth_token(self):
        """Get JWT token for authenticated requests."""
        refresh = RefreshToken.for_user(self.test_user)
        return str(refresh.access_token)

    def _test_endpoint(self, endpoint):
        """Test a single endpoint multiple times and collect metrics."""
        response_times = []
        query_counts = []
        query_times = []
        errors = 0
        
        # Set up authentication if required
        if endpoint['auth_required']:
            token = self._get_auth_token()
            self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        else:
            self.client.credentials()
            
        # Handle special cases for data preparation
        data = endpoint['data']
        if endpoint['name'] == 'auth_refresh':
            refresh = RefreshToken.for_user(self.test_user)
            data = {'refresh': str(refresh)}
            
        # Run multiple tests
        for i in range(self.runs):
            # Reset query tracking
            connection.queries_log.clear()
            
            # Record start time
            start_time = time.time()
            
            try:
                # Make request
                if endpoint['method'] == 'GET':
                    response = self.client.get(endpoint['url'])
                elif endpoint['method'] == 'POST':
                    response = self.client.post(endpoint['url'], data=data)
                else:
                    response = self.client.generic(endpoint['method'], endpoint['url'], data=data)
                
                # Record end time
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                # Check if response is successful
                if 200 <= response.status_code < 300:
                    response_times.append(response_time_ms)
                    
                    # Record database metrics
                    query_count = len(connection.queries)
                    total_query_time = sum(float(q['time']) for q in connection.queries)
                    query_counts.append(query_count)
                    query_times.append(total_query_time * 1000)  # Convert to ms
                    
                    if self.verbose:
                        self.stdout.write(
                            f'  Run {i+1}: {response_time_ms:.2f}ms '
                            f'({query_count} queries, {total_query_time*1000:.2f}ms DB)'
                        )
                else:
                    errors += 1
                    if self.verbose:
                        self.stdout.write(
                            self.style.WARNING(f'  Run {i+1}: Error {response.status_code}')
                        )
                        
            except Exception as e:
                errors += 1
                if self.verbose:
                    self.stdout.write(self.style.ERROR(f'  Run {i+1}: Exception - {str(e)}'))
        
        # Calculate statistics
        if response_times:
            return {
                'endpoint': endpoint['name'],
                'url': endpoint['url'],
                'method': endpoint['method'],
                'budget_ms': endpoint['budget_ms'],
                'runs': len(response_times),
                'errors': errors,
                'response_time_ms': {
                    'min': min(response_times),
                    'max': max(response_times),
                    'mean': statistics.mean(response_times),
                    'median': statistics.median(response_times),
                    'p95': self._percentile(response_times, 95),
                    'p99': self._percentile(response_times, 99)
                },
                'database': {
                    'avg_query_count': statistics.mean(query_counts) if query_counts else 0,
                    'max_query_count': max(query_counts) if query_counts else 0,
                    'avg_query_time_ms': statistics.mean(query_times) if query_times else 0,
                    'max_query_time_ms': max(query_times) if query_times else 0
                }
            }
        else:
            return {
                'endpoint': endpoint['name'],
                'url': endpoint['url'],
                'method': endpoint['method'],
                'budget_ms': endpoint['budget_ms'],
                'runs': 0,
                'errors': errors,
                'error': 'All requests failed'
            }

    def _percentile(self, data, percentile):
        """Calculate percentile of a dataset."""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def _generate_baseline_report(self, results):
        """Generate comprehensive baseline report."""
        timestamp = datetime.now().isoformat()
        
        # Calculate overall statistics
        all_response_times = []
        budget_violations = []
        
        for endpoint_name, result in results.items():
            if 'response_time_ms' in result:
                all_response_times.extend([result['response_time_ms']['mean']])
                
                # Check budget violations
                if result['response_time_ms']['p95'] > result['budget_ms']:
                    budget_violations.append({
                        'endpoint': endpoint_name,
                        'p95_ms': result['response_time_ms']['p95'],
                        'budget_ms': result['budget_ms'],
                        'violation_percent': ((result['response_time_ms']['p95'] - result['budget_ms']) / result['budget_ms']) * 100
                    })
        
        return {
            'metadata': {
                'timestamp': timestamp,
                'total_runs_per_endpoint': self.runs,
                'total_endpoints_tested': len(results)
            },
            'summary': {
                'overall_avg_response_time_ms': statistics.mean(all_response_times) if all_response_times else 0,
                'budget_violations': len(budget_violations),
                'total_errors': sum(r.get('errors', 0) for r in results.values())
            },
            'budget_violations': budget_violations,
            'detailed_results': results
        }

    def _save_results(self, baseline_report):
        """Save baseline results to JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(baseline_report, f, indent=2)

    def _display_summary(self, baseline_report):
        """Display summary of baseline results."""
        summary = baseline_report['summary']
        violations = baseline_report['budget_violations']
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('📊 PERFORMANCE BASELINE SUMMARY'))
        self.stdout.write('='*60)
        
        self.stdout.write(f'Overall Average Response Time: {summary["overall_avg_response_time_ms"]:.2f}ms')
        self.stdout.write(f'Total Errors: {summary["total_errors"]}')
        self.stdout.write(f'Budget Violations: {summary["budget_violations"]}')
        
        if violations:
            self.stdout.write('\n⚠️  BUDGET VIOLATIONS:')
            for violation in violations:
                self.stdout.write(
                    self.style.WARNING(
                        f'  {violation["endpoint"]}: {violation["p95_ms"]:.2f}ms '
                        f'(budget: {violation["budget_ms"]}ms, '
                        f'+{violation["violation_percent"]:.1f}%)'
                    )
                )
        
        self.stdout.write('\n📋 ENDPOINT DETAILS:')
        for endpoint_name, result in baseline_report['detailed_results'].items():
            if 'response_time_ms' in result:
                rt = result['response_time_ms']
                db = result['database']
                
                status_icon = '✅' if rt['p95'] <= result['budget_ms'] else '⚠️'
                
                self.stdout.write(
                    f'{status_icon} {endpoint_name}: '
                    f'P95={rt["p95"]:.2f}ms, Mean={rt["mean"]:.2f}ms '
                    f'(DB: {db["avg_query_count"]:.1f} queries, {db["avg_query_time_ms"]:.2f}ms)'
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'❌ {endpoint_name}: {result.get("error", "Failed")}')
                )