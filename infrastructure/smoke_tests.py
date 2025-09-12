#!/usr/bin/env python3
"""
Deployment Smoke Tests

Validates critical system functionality after deployment to ensure services
are operational and endpoints respond correctly.
"""

import os
import sys
import time
import requests
import json
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin


class DeploymentSmokeTests:
    """Comprehensive deployment validation test suite."""
    
    def __init__(self, base_url: str = None, timeout: int = 30):
        """Initialise smoke test suite with target deployment."""
        self.base_url = base_url or os.getenv('SMOKE_TEST_BASE_URL', 'http://localhost:8000')
        self.timeout = timeout
        self.results: List[Dict] = []
        self.session = requests.Session()
        self.session.timeout = self.timeout
    
    def log_test(self, test_name: str, status: str, details: str = "") -> None:
        """Log individual test result."""
        result = {
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
        self.results.append(result)
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"[{status_symbol}] {test_name}: {status}")
        if details:
            print(f"    {details}")
    
    def test_basic_connectivity(self) -> bool:
        """Verify basic HTTP connectivity to deployment."""
        try:
            response = self.session.get(
                urljoin(self.base_url, '/admin/login/'),
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.log_test("Basic Connectivity", "PASS", f"HTTP {response.status_code}")
                return True
            else:
                self.log_test("Basic Connectivity", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Basic Connectivity", "FAIL", str(e))
            return False
    
    def test_database_health(self) -> bool:
        """Verify database connectivity and health."""
        try:
            response = self.session.get(
                urljoin(self.base_url, '/api/health/database/'),
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_test("Database Health", "PASS", "Database responsive")
                    return True
                else:
                    self.log_test("Database Health", "FAIL", f"Status: {data.get('status')}")
                    return False
            else:
                self.log_test("Database Health", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Database Health", "FAIL", str(e))
            return False
    
    def test_redis_connectivity(self) -> bool:
        """Verify Redis cache connectivity."""
        try:
            response = self.session.get(
                urljoin(self.base_url, '/api/health/cache/'),
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_test("Redis Connectivity", "PASS", "Cache responsive")
                    return True
                else:
                    self.log_test("Redis Connectivity", "FAIL", f"Status: {data.get('status')}")
                    return False
            else:
                self.log_test("Redis Connectivity", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Redis Connectivity", "FAIL", str(e))
            return False
    
    def test_api_authentication(self) -> bool:
        """Test API authentication endpoint."""
        try:
            # Test registration endpoint
            response = self.session.post(
                urljoin(self.base_url, '/api/auth/register/'),
                json={
                    'username': f'smoke_test_user_{int(time.time())}',
                    'email': f'smoke_test_{int(time.time())}@example.com',
                    'password': 'SmokeTest123!',
                    'password_confirm': 'SmokeTest123!'
                },
                timeout=self.timeout
            )
            if response.status_code in [201, 400]:  # 400 for validation errors is acceptable
                self.log_test("API Authentication", "PASS", "Auth endpoints responsive")
                return True
            else:
                self.log_test("API Authentication", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Authentication", "FAIL", str(e))
            return False
    
    def test_data_api_endpoints(self) -> bool:
        """Test core data API endpoints."""
        try:
            # Test stocks endpoint
            response = self.session.get(
                urljoin(self.base_url, '/api/data/stocks/'),
                timeout=self.timeout
            )
            if response.status_code in [200, 401]:  # 401 for auth required is acceptable
                self.log_test("Data API Endpoints", "PASS", "Data endpoints responsive")
                return True
            else:
                self.log_test("Data API Endpoints", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Data API Endpoints", "FAIL", str(e))
            return False
    
    def test_analytics_api_endpoints(self) -> bool:
        """Test analytics API endpoints."""
        try:
            # Test analytics health endpoint
            response = self.session.get(
                urljoin(self.base_url, '/api/analytics/health/'),
                timeout=self.timeout
            )
            if response.status_code in [200, 401]:  # 401 for auth required is acceptable
                self.log_test("Analytics API Endpoints", "PASS", "Analytics endpoints responsive")
                return True
            else:
                self.log_test("Analytics API Endpoints", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Analytics API Endpoints", "FAIL", str(e))
            return False
    
    def test_static_files(self) -> bool:
        """Verify static file serving."""
        try:
            response = self.session.get(
                urljoin(self.base_url, '/static/admin/css/base.css'),
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.log_test("Static Files", "PASS", "Static files accessible")
                return True
            else:
                self.log_test("Static Files", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Static Files", "FAIL", str(e))
            return False
    
    def test_performance_baseline(self) -> bool:
        """Basic performance validation."""
        try:
            start_time = time.time()
            response = self.session.get(
                urljoin(self.base_url, '/admin/login/'),
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200 and response_time < 2000:  # Under 2 seconds
                self.log_test("Performance Baseline", "PASS", f"{response_time:.0f}ms response time")
                return True
            else:
                self.log_test("Performance Baseline", "FAIL", f"{response_time:.0f}ms (>2000ms threshold)")
                return False
        except Exception as e:
            self.log_test("Performance Baseline", "FAIL", str(e))
            return False
    
    def run_all_tests(self) -> Tuple[int, int]:
        """Execute complete smoke test suite."""
        print(f"[INFO] Starting deployment smoke tests against: {self.base_url}")
        print(f"[INFO] Test timeout: {self.timeout} seconds")
        print("")
        
        tests = [
            self.test_basic_connectivity,
            self.test_database_health,
            self.test_redis_connectivity,
            self.test_api_authentication,
            self.test_data_api_endpoints,
            self.test_analytics_api_endpoints,
            self.test_static_files,
            self.test_performance_baseline
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            if test():
                passed += 1
            else:
                failed += 1
        
        self.print_summary(passed, failed)
        return passed, failed
    
    def print_summary(self, passed: int, failed: int) -> None:
        """Print comprehensive test results summary."""
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print("")
        print("=" * 50)
        print("DEPLOYMENT SMOKE TEST RESULTS")
        print("=" * 50)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("")
        
        if failed > 0:
            print("FAILED TESTS:")
            for result in self.results:
                if result['status'] == 'FAIL':
                    print(f"  ✗ {result['test']}: {result['details']}")
        
        deployment_status = "HEALTHY" if failed == 0 else "DEGRADED" if passed > failed else "CRITICAL"
        print(f"Deployment Status: {deployment_status}")
        print("=" * 50)
    
    def save_results(self, output_file: str = 'smoke_test_results.json') -> None:
        """Save test results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump({
                'base_url': self.base_url,
                'timestamp': time.time(),
                'results': self.results,
                'summary': {
                    'total': len(self.results),
                    'passed': len([r for r in self.results if r['status'] == 'PASS']),
                    'failed': len([r for r in self.results if r['status'] == 'FAIL'])
                }
            }, f, indent=2)
        print(f"[INFO] Results saved to: {output_file}")


def main():
    """Main execution function for smoke tests."""
    base_url = os.getenv('SMOKE_TEST_BASE_URL', 'http://localhost:8000')
    timeout = int(os.getenv('SMOKE_TEST_TIMEOUT', '30'))
    
    # Allow command line override
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    smoke_tests = DeploymentSmokeTests(base_url, timeout)
    passed, failed = smoke_tests.run_all_tests()
    
    # Save results for CI/CD pipeline
    smoke_tests.save_results()
    
    # Exit with appropriate code for CI/CD
    if failed == 0:
        print("[OK] All smoke tests passed - deployment healthy")
        sys.exit(0)
    elif passed > failed:
        print("[WARNING] Some tests failed - deployment degraded but functional")
        sys.exit(1)
    else:
        print("[ERROR] Critical failures detected - deployment requires attention")
        sys.exit(2)


if __name__ == "__main__":
    main()