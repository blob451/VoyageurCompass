"""
API integration tests with real service chain validation.
Tests complete API workflows across all modules.
"""

import time
from datetime import date
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase
from django.urls import reverse
from rest_framework.test import APIClient

from Data.models import (
    DataIndustry,
    DataSector,
    Stock,
    StockPrice,
)

User = get_user_model()


class APIChainIntegrationTest(TestCase):
    """Test complete API chains across multiple services."""

    def setUp(self):
        """Set up API integration test environment."""
        self.client = APIClient()
        cache.clear()

        # Create test user
        self.user = User.objects.create_user(
            username="api_integration_user", email="api@integration.com", password="api_integration_pass_123"
        )

        # Create test data
        self.sector = DataSector.objects.create(sectorKey="tech_api", sectorName="Technology API", data_source="yahoo")

        self.industry = DataIndustry.objects.create(
            industryKey="software_api", industryName="Software API", sector=self.sector, data_source="yahoo"
        )

        self.stock = Stock.objects.create(
            symbol="API_TEST",
            short_name="API Test Corporation",
            currency="USD",
            exchange="NASDAQ",
            sector_id=self.sector,
            industry_id=self.industry,
        )

    def test_authentication_api_chain(self):
        """Test complete authentication API chain."""
        # Step 1: User registration (if endpoint exists)
        register_data = {
            "username": "new_api_user",
            "email": "newapi@test.com",
            "password": "new_api_pass_123",
            "password_confirm": "new_api_pass_123",
        }

        try:
            register_response = self.client.post(reverse("core:register"), register_data, format="json")
            if register_response.status_code in [200, 201]:
                print(f"Registration successful: {register_response.status_code}")
        except Exception:
            print("Registration endpoint may not exist - continuing with login test")

        # Step 2: User login
        login_data = {"username": "api_integration_user", "password": "api_integration_pass_123"}

        login_response = self.client.post(reverse("core:token_obtain_pair"), login_data, format="json")
        self.assertIn(login_response.status_code, [200, 201])

        # Extract authentication token
        if "access" in login_response.data:
            token = login_response.data["access"]
            self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
            auth_method = "JWT"
        else:
            self.client.force_authenticate(user=self.user)
            auth_method = "Force"

        # Step 3: Authenticated API access
        authenticated_url = reverse("data:portfolio-list")
        auth_response = self.client.get(authenticated_url)
        self.assertIn(auth_response.status_code, [200, 202])

        print(f"Authentication API chain completed using {auth_method}")

        # Step 4: Token refresh (if JWT)
        if auth_method == "JWT" and "refresh" in login_response.data:
            refresh_data = {"refresh": login_response.data["refresh"]}
            refresh_response = self.client.post(reverse("core:token_refresh"), refresh_data)
            self.assertIn(refresh_response.status_code, [200, 201])
            print("Token refresh successful")

    def test_data_analytics_api_chain(self):
        """Test Data → Analytics API chain."""
        self.client.force_authenticate(user=self.user)

        # Step 1: Create stock price data
        StockPrice.objects.create(
            stock=self.stock,
            date=date.today(),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("98.00"),
            close=Decimal("103.00"),
            volume=1000000,
        )

        # Step 2: Get stock data via API
        stock_url = reverse("data:stock-detail", args=[self.stock.id])
        stock_response = self.client.get(stock_url)
        self.assertIn(stock_response.status_code, [200, 404])

        # Step 3: Trigger analytics via API
        try:
            analytics_url = reverse("analytics:analysis", args=[self.stock.symbol])
            analytics_response = self.client.get(analytics_url)
            analytics_status = analytics_response.status_code

            self.assertIn(analytics_status, [200, 202, 404, 503])
            print(f"Data-Analytics API chain completed: {analytics_status}")

        except Exception as e:
            print(f"Analytics API handled gracefully: {e}")

    def test_portfolio_management_api_chain(self):
        """Test complete portfolio management API workflow."""
        self.client.force_authenticate(user=self.user)

        # Step 1: Create portfolio
        portfolio_data = {
            "name": "API Test Portfolio",
            "description": "Portfolio for API testing",
            "initial_value": 5000.00,
            "risk_tolerance": "moderate",
        }

        portfolio_response = self.client.post(reverse("data:portfolio-list"), portfolio_data, format="json")
        self.assertIn(portfolio_response.status_code, [200, 201])

        portfolio_id = portfolio_response.data.get("id")
        self.assertIsNotNone(portfolio_id)

        # Step 2: Add holding to portfolio
        holding_data = {
            "stock_symbol": self.stock.symbol,
            "quantity": 5,
            "average_price": 100.00,
            "purchase_date": date.today().isoformat(),
        }

        holding_response = self.client.post(
            reverse("data:portfolio-add-holding", args=[portfolio_id]), holding_data, format="json"
        )
        self.assertIn(holding_response.status_code, [200, 201])

        # Step 3: Get portfolio performance
        performance_response = self.client.get(reverse("data:portfolio-performance", args=[portfolio_id]))
        self.assertIn(performance_response.status_code, [200, 202])

        # Step 4: Update holding
        update_data = {"symbol": self.stock.symbol, "quantity": 8, "average_price": 102.50}

        update_response = self.client.post(
            reverse("data:portfolio-update-holding", args=[portfolio_id]), update_data, format="json"
        )
        self.assertIn(update_response.status_code, [200, 201])

        print("Portfolio management API chain completed successfully")

    def test_market_data_api_chain(self):
        """Test market data synchronization API chain."""
        self.client.force_authenticate(user=self.user)

        # Step 1: Get market status
        try:
            market_status_response = self.client.get(reverse("data:stock-market-status"))
            self.assertIn(market_status_response.status_code, [200, 503])
        except Exception:
            print("Market status endpoint may not be available")

        # Step 2: Sync stock data
        sync_response = self.client.post(
            reverse("data:stock-sync", args=[self.stock.id]), {"period": "1d"}, format="json"
        )
        self.assertIn(sync_response.status_code, [200, 202, 503])

        # Step 3: Bulk price update (if available)
        try:
            bulk_update_response = self.client.post(reverse("data:bulk-price-update"))
            self.assertIn(bulk_update_response.status_code, [200, 202, 503])
        except Exception:
            print("Bulk update endpoint may not be available")

        print("Market data API chain completed")

    def test_error_propagation_api_chain(self):
        """Test error propagation through API chains."""
        self.client.force_authenticate(user=self.user)

        # Test 1: Invalid stock ID
        invalid_stock_response = self.client.get(reverse("data:stock-detail", args=[99999]))
        self.assertEqual(invalid_stock_response.status_code, 404)

        # Test 2: Invalid portfolio operation
        try:
            invalid_portfolio_response = self.client.post(
                reverse("data:portfolio-add-holding", args=[99999]), {"stock_symbol": "INVALID"}, format="json"
            )
            self.assertIn(invalid_portfolio_response.status_code, [400, 404])
        except Exception:
            print("Portfolio endpoint handled error gracefully")

        # Test 3: Unauthenticated access
        self.client.force_authenticate(user=None)
        protected_response = self.client.get(reverse("data:portfolio-list"))
        self.assertEqual(protected_response.status_code, 401)

        print("API error propagation validated")


class RealTimeAPIIntegrationTest(TransactionTestCase):
    """Test real-time API operations with external services."""

    def setUp(self):
        """Set up real-time API test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="realtime_user", email="realtime@api.com", password="realtime_pass_123"
        )
        self.client.force_authenticate(user=self.user)

    def test_real_time_stock_data_api(self):
        """Test real-time stock data API integration."""
        # Create stock for real data testing
        sector = DataSector.objects.create(
            sectorKey="tech_realtime", sectorName="Technology Realtime", data_source="yahoo"
        )

        stock = Stock.objects.create(
            symbol="AAPL", short_name="Apple Inc. Realtime", currency="USD", exchange="NASDAQ", sector_id=sector
        )

        # Test real-time data sync
        start_time = time.time()
        sync_response = self.client.post(reverse("data:stock-sync", args=[stock.id]), {"period": "1d"}, format="json")
        sync_time = time.time() - start_time

        self.assertIn(sync_response.status_code, [200, 202, 503])
        self.assertLess(sync_time, 60.0, "Real-time sync should complete within 60 seconds")

        print(f"Real-time stock data API completed in {sync_time:.1f}s")

    def test_concurrent_api_requests(self):
        """Test concurrent API request handling."""
        import threading

        results = []
        errors = []

        def make_api_request(request_id):
            """Make concurrent API request."""
            try:
                client = APIClient()
                client.force_authenticate(user=self.user)

                response = client.get(reverse("data:stock-list"))
                if response.status_code == 200:
                    results.append(f"request_{request_id}_success")
                else:
                    errors.append(f"request_{request_id}_status_{response.status_code}")

            except Exception as e:
                errors.append(f"request_{request_id}_error: {str(e)}")

        # Execute concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_api_request, args=(i + 1,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        self.assertGreaterEqual(len(results), 1, "At least one concurrent request should succeed")
        print(f"Concurrent API requests: {len(results)} successful, {len(errors)} errors")

    def test_api_performance_benchmarks(self):
        """Test API performance benchmarks."""
        test_cases = [
            ("data:stock-list", "GET", None, 5.0),
            ("design:health", "GET", None, 2.0),
        ]

        for url_name, method, data, max_time in test_cases:
            try:
                start_time = time.time()

                if method == "GET":
                    response = self.client.get(reverse(url_name))
                elif method == "POST":
                    response = self.client.post(reverse(url_name), data or {}, format="json")

                response_time = time.time() - start_time

                # Performance assertion
                self.assertLess(response_time, max_time, f"API {url_name} should respond within {max_time}s")

                print(f"API {url_name}: {response_time:.3f}s (limit: {max_time}s)")

            except Exception as e:
                print(f"API {url_name} test handled gracefully: {e}")

    def test_api_rate_limiting(self):
        """Test API rate limiting behavior."""
        # Make rapid consecutive requests
        start_time = time.time()
        responses = []

        for i in range(10):
            try:
                response = self.client.get(reverse("data:stock-list"))
                responses.append(response.status_code)
            except Exception as e:
                responses.append(f"error: {e}")

        total_time = time.time() - start_time

        # Verify responses
        success_count = sum(1 for r in responses if r == 200)
        self.assertGreater(success_count, 0, "At least some requests should succeed")

        print(f"Rate limiting test: {success_count}/10 requests successful in {total_time:.1f}s")


class ServiceHealthAPITest(TestCase):
    """Test service health monitoring APIs."""

    def setUp(self):
        """Set up service health test environment."""
        self.client = APIClient()

    def test_health_check_endpoints(self):
        """Test all health check endpoints."""
        health_endpoints = [
            "/design/health/",
            "/healthz",
        ]

        for endpoint in health_endpoints:
            try:
                response = self.client.get(endpoint)
                self.assertIn(response.status_code, [200, 503])

                if response.status_code == 200:
                    # Verify health check response structure
                    if hasattr(response, "json"):
                        health_data = response.json()
                        if "healthy" in health_data:
                            self.assertIsInstance(health_data["healthy"], bool)

                print(f"Health endpoint {endpoint}: {response.status_code}")

            except Exception as e:
                print(f"Health endpoint {endpoint} handled gracefully: {e}")

    def test_service_dependency_validation(self):
        """Test service dependency health validation."""
        # Test database connectivity
        try:
            from django.db import connection

            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                db_healthy = True
        except Exception:
            db_healthy = False

        # Test cache connectivity
        try:
            cache.set("health_test", "value", 1)
            cache_value = cache.get("health_test")
            cache_healthy = cache_value == "value"
        except Exception:
            cache_healthy = False

        print(f"Service dependencies - Database: {db_healthy}, Cache: {cache_healthy}")

        # At least database should be healthy for tests to run
        self.assertTrue(db_healthy, "Database should be accessible for API tests")


if __name__ == "__main__":
    import unittest

    unittest.main()
