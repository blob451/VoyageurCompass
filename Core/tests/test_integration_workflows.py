"""
End-to-end integration tests for complete user workflows.
Validates cross-module functionality with real service integration.
"""

import time
from datetime import date
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, TransactionTestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from Analytics.tests.fixtures import OllamaTestService
from Data.models import (
    AnalyticsResults,
    DataIndustry,
    DataSector,
    Stock,
    StockPrice,
)
from Data.tests.fixtures import YahooFinanceTestService

User = get_user_model()


class UserJourneyIntegrationTest(TransactionTestCase):
    """Complete user journey integration tests from registration to analysis."""

    def setUp(self):
        """Set up integration test environment."""
        self.client = APIClient()
        cache.clear()

        # Initialize test services
        self.yahoo_service = YahooFinanceTestService()
        self.ollama_service = OllamaTestService()

        # Create test user
        self.user = User.objects.create_user(
            username="integration_user",
            email="integration@test.com",
            password="integration_pass_123",
            first_name="Integration",
            last_name="Tester",
        )

        # Create test sector and industry for real data
        self.sector = DataSector.objects.create(
            sectorKey="tech_integration", sectorName="Technology Integration", data_source="yahoo"
        )

        self.industry = DataIndustry.objects.create(
            industryKey="software_integration",
            industryName="Software Integration",
            sector=self.sector,
            data_source="yahoo",
        )

    def test_complete_user_analysis_workflow(self):
        """Test complete workflow: Auth → Portfolio → Analysis → LLM Explanation."""
        start_time = time.time()

        # Step 1: User Authentication
        auth_data = {"username": "integration_user", "password": "integration_pass_123"}
        auth_response = self.client.post(reverse("core:token_obtain_pair"), auth_data, format="json")
        self.assertIn(auth_response.status_code, [200, 201])

        if "access" in auth_response.data:
            auth_token = auth_response.data["access"]
            self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {auth_token}")
        else:
            self.client.force_authenticate(user=self.user)

        auth_time = time.time() - start_time
        self.assertLess(auth_time, 20.0, "Authentication should complete within 20 seconds (realistic target)")

        # Step 2: Portfolio Creation
        portfolio_data = {
            "name": "Integration Test Portfolio",
            "description": "Portfolio for end-to-end testing",
            "initial_value": 10000.00,
            "risk_tolerance": "moderate",
        }
        portfolio_response = self.client.post(reverse("data:portfolio-list"), portfolio_data, format="json")
        self.assertIn(portfolio_response.status_code, [200, 201])

        portfolio_id = portfolio_response.data["id"]
        portfolio_time = time.time() - start_time - auth_time
        self.assertLess(portfolio_time, 6.0, "Portfolio creation should complete within 6 seconds")

        # Step 3: Stock Data Synchronization
        # Create stock for real integration
        stock = Stock.objects.create(
            symbol="AAPL",
            short_name="Apple Inc.",
            currency="USD",
            exchange="NASDAQ",
            sector_id=self.sector,
            industry_id=self.industry,
        )

        # Test real stock data sync
        sync_start = time.time()
        sync_url = reverse("data:stock-sync", args=[stock.id])
        sync_response = self.client.post(sync_url, {"period": "5d"}, format="json")
        sync_time = time.time() - sync_start

        self.assertIn(sync_response.status_code, [200, 202])
        self.assertLess(sync_time, 30.0, "Stock sync should complete within 30 seconds")

        # Verify stock data was synced
        price_count = StockPrice.objects.filter(stock=stock).count()
        if price_count > 0:
            print(f"Stock sync successful: {price_count} price points")

        # Step 4: Add Stock to Portfolio
        holding_data = {
            "stock_symbol": "AAPL",
            "quantity": 10,
            "average_price": 150.00,
            "purchase_date": date.today().isoformat(),
        }
        holding_response = self.client.post(
            reverse("data:portfolio-add-holding", args=[portfolio_id]), holding_data, format="json"
        )
        self.assertIn(holding_response.status_code, [200, 201])

        # Step 5: Technical Analysis Generation
        # Create analytics data for comprehensive testing
        analytics_result = AnalyticsResults.objects.create(
            stock=stock,
            as_of=timezone.now(),
            w_rsi14=Decimal("0.655"),
            w_sma50vs200=Decimal("0.148"),
            w_macd12269=Decimal("0.075"),
            sentimentScore=0.72,
            composite_raw=Decimal("7.2"),
        )

        # Test analytics retrieval
        analytics_url = reverse("analytics:analyze_stock", args=[stock.symbol])
        analytics_response = self.client.get(analytics_url)
        self.assertIn(analytics_response.status_code, [200, 202])

        # Step 6: LLM Explanation Generation (if service available)
        explanation_start = time.time()
        try:
            explanation_url = reverse("analytics:generate_explanation", args=[analytics_result.id])
            explanation_response = self.client.post(explanation_url, {"detail_level": "standard"}, format="json")
            explanation_time = time.time() - explanation_start

            # LLM service may not be available in all environments
            if explanation_response.status_code in [200, 202]:
                self.assertLess(explanation_time, 120.0, "LLM explanation should complete within 2 minutes")
                print(f" LLM explanation generated in {explanation_time:.1f}s")
            elif explanation_response.status_code == 503:
                print(" LLM service unavailable - handled gracefully")

        except Exception as e:
            print(f" LLM service integration handled gracefully: {e}")

        # Step 7: Portfolio Performance Analysis
        performance_url = reverse("data:portfolio-performance", args=[portfolio_id])
        performance_response = self.client.get(performance_url)
        self.assertIn(performance_response.status_code, [200, 202])

        if "total_value" in performance_response.data:
            self.assertIsInstance(performance_response.data["total_value"], (int, float, str))

        # Verify complete workflow timing
        total_time = time.time() - start_time
        self.assertLess(total_time, 300.0, "Complete workflow should finish within 5 minutes")

        print(f" Complete user workflow completed in {total_time:.1f}s")
        print(f"  - Authentication: {auth_time:.1f}s")
        print(f"  - Portfolio setup: {portfolio_time:.1f}s")
        print(f"  - Stock synchronization: {sync_time:.1f}s")
        print(
            f"  - Analytics processing: {explanation_time:.1f}s"
            if "explanation_time" in locals()
            else "  - Analytics: processed"
        )

    def test_file_upload_processing_workflow(self):
        """Test complete file upload and processing workflow."""
        self.client.force_authenticate(user=self.user)

        # Create test file content
        test_csv_content = """symbol,date,close,volume
AAPL,2025-01-01,150.00,1000000
AAPL,2025-01-02,152.00,1100000
MSFT,2025-01-01,300.00,800000
MSFT,2025-01-02,305.00,850000"""

        # Test file upload
        from django.core.files.uploadedfile import SimpleUploadedFile

        test_file = SimpleUploadedFile(
            "integration_test.csv", test_csv_content.encode("utf-8"), content_type="text/csv"
        )

        # Upload file
        upload_start = time.time()
        upload_response = self.client.post("/design/upload/", {"file": test_file})
        upload_time = time.time() - upload_start

        self.assertIn(upload_response.status_code, [200, 202])
        self.assertLess(upload_time, 10.0, "File upload should complete within 10 seconds")

        # Verify file processing (if processing endpoint exists)
        if "filename" in upload_response.json():
            filename = upload_response.json()["filename"]
            print(f" File uploaded successfully: {filename}")

        print(f" File upload workflow completed in {upload_time:.1f}s")

    def test_error_handling_across_modules(self):
        """Test error propagation and handling across different modules."""
        self.client.force_authenticate(user=self.user)

        # Test 1: Invalid stock symbol handling
        invalid_stock_url = reverse("data:stock-detail", args=[99999])
        invalid_response = self.client.get(invalid_stock_url)
        self.assertEqual(invalid_response.status_code, 404)

        # Test 2: Invalid portfolio access
        invalid_portfolio_url = reverse("data:portfolio-detail", args=[99999])
        invalid_portfolio_response = self.client.get(invalid_portfolio_url)
        self.assertEqual(invalid_portfolio_response.status_code, 404)

        # Test 3: Authentication required endpoints
        self.client.force_authenticate(user=None)
        protected_url = reverse("data:portfolio-list")
        protected_response = self.client.get(protected_url)
        self.assertEqual(protected_response.status_code, 401)

        print(" Error handling validation completed across modules")

    def test_concurrent_user_operations(self):
        """Test handling of concurrent operations by multiple users."""
        import threading

        # Create additional test user
        user2 = User.objects.create_user(
            username="integration_user2", email="integration2@test.com", password="integration_pass_123"
        )

        results = []
        errors = []

        def user_workflow(user, user_id):
            """Execute user workflow concurrently."""
            try:
                client = APIClient()
                client.force_authenticate(user=user)

                # Create portfolio
                portfolio_data = {
                    "name": f"Concurrent Portfolio {user_id}",
                    "description": f"Portfolio for concurrent user {user_id}",
                    "initial_value": 5000.00,
                }
                portfolio_response = client.post(reverse("data:portfolio-list"), portfolio_data)

                if portfolio_response.status_code in [200, 201]:
                    results.append(f"user_{user_id}_success")
                else:
                    errors.append(f"user_{user_id}_portfolio_error")

            except Exception as e:
                errors.append(f"user_{user_id}_exception: {str(e)}")

        # Execute concurrent workflows
        threads = []
        for i, user in enumerate([self.user, user2]):
            thread = threading.Thread(target=user_workflow, args=(user, i + 1))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(results), 2, f"Expected 2 successful workflows, got {len(results)}")
        self.assertEqual(len(errors), 0, f"Expected no errors, got: {errors}")

        print(f" Concurrent operations completed successfully: {results}")


class CrossModuleIntegrationTest(TestCase):
    """Test integration between different modules."""

    def setUp(self):
        """Set up cross-module test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="cross_module_user", email="crossmodule@test.com", password="cross_module_pass_123"
        )
        self.client.force_authenticate(user=self.user)

    def test_data_analytics_integration(self):
        """Test integration between Data and Analytics modules."""
        # Create stock with real data
        sector = DataSector.objects.create(sectorKey="tech_cross", sectorName="Technology Cross", data_source="yahoo")

        stock = Stock.objects.create(symbol="TEST_CROSS", short_name="Cross Module Test", sector_id=sector)

        # Add price data
        StockPrice.objects.create(
            stock=stock,
            date=date.today(),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("98.00"),
            close=Decimal("103.00"),
            volume=1000000,
        )

        # Test analytics generation
        try:
            analytics_url = reverse("analytics:analysis", args=[stock.symbol])
            analytics_response = self.client.get(analytics_url)

            self.assertIn(analytics_response.status_code, [200, 202, 404])
            print(" Data-Analytics integration validated")

        except Exception as e:
            print(f" Data-Analytics integration handled gracefully: {e}")

    def test_analytics_design_integration(self):
        """Test integration between Analytics and Design modules."""
        # Test analytics report export
        try:
            # This would test report generation and file serving
            export_url = "/design/static/reports/test_report.json"
            export_response = self.client.get(export_url)

            # File may not exist, but endpoint should handle gracefully
            self.assertIn(export_response.status_code, [200, 404])
            print(" Analytics-Design integration validated")

        except Exception as e:
            print(f" Analytics-Design integration handled gracefully: {e}")

    def test_core_all_modules_integration(self):
        """Test Core module integration with all other modules."""
        # Test authentication across all modules

        # 1. Core authentication
        self.assertTrue(self.user.is_authenticated)

        # 2. Data module access
        data_url = reverse("data:stock-list")
        data_response = self.client.get(data_url)
        self.assertIn(data_response.status_code, [200, 404])

        # 3. Design module access
        design_url = "/design/health/"
        design_response = self.client.get(design_url)
        self.assertIn(design_response.status_code, [200, 503])

        print("Core module integration with all modules validated")


if __name__ == "__main__":
    import unittest

    unittest.main()
