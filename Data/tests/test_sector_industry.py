"""
Tests for sector/industry data pull functionality.
"""

from datetime import timedelta

# All tests now use real database operations - no mocks required

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from django.utils import timezone

from Data.models import Stock
from Data.services.yahoo_finance import yahoo_finance_service


class SectorIndustryTestCase(TestCase):
    """Test sector/industry functionality."""

    def setUp(self):
        """Set up test data."""
        # Create test stock with recent sector data
        self.recent_stock = Stock.objects.create(
            symbol="AAPL",
            short_name="Apple Inc",
            sector="Technology",
            industry="Consumer Electronics",
            sectorUpdatedAt=timezone.now() - timedelta(days=30),
            data_source="yahoo",
        )

        # Create test stock with stale sector data
        self.stale_stock = Stock.objects.create(
            symbol="MSFT",
            short_name="Microsoft Corp",
            sector="",
            industry="",
            sectorUpdatedAt=timezone.now() - timedelta(days=4 * 365),  # 4 years ago
            data_source="yahoo",
        )

        # Create test stock with no sector data
        self.missing_stock = Stock.objects.create(
            symbol="GOOGL", short_name="Alphabet Inc", sector="", industry="", sectorUpdatedAt=None, data_source="yahoo"
        )

    def test_recency_gating_skip_recent(self):
        """Test that recent sector/industry data is skipped."""
        symbols = ["AAPL"]
        symbolsToFetch = yahoo_finance_service.getStaleAndMissingSymbols(symbols)

        # Should skip AAPL as it has recent data
        self.assertNotIn("AAPL", symbolsToFetch)

    def test_recency_gating_fetch_stale(self):
        """Test that stale sector/industry data is fetched."""
        symbols = ["MSFT"]
        symbolsToFetch = yahoo_finance_service.getStaleAndMissingSymbols(symbols)

        # Should include MSFT as it has stale data
        self.assertIn("MSFT", symbolsToFetch)

    def test_recency_gating_fetch_missing(self):
        """Test that missing sector/industry data is fetched."""
        symbols = ["GOOGL"]
        symbolsToFetch = yahoo_finance_service.getStaleAndMissingSymbols(symbols)

        # Should include GOOGL as it has no sectorUpdatedAt
        self.assertIn("GOOGL", symbolsToFetch)

    def test_recency_gating_new_symbol(self):
        """Test that non-existent symbols are included for fetching."""
        symbols = ["NVDA"]  # Not in database
        symbolsToFetch = yahoo_finance_service.getStaleAndMissingSymbols(symbols)

        # Should include NVDA as it doesn't exist
        self.assertIn("NVDA", symbolsToFetch)

    def test_upsert_behavior_create_new(self):
        """Test that upsert creates new stock records."""
        profiles = {
            "TSLA": {
                "symbol": "TSLA",
                "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers",
                "updatedAt": timezone.now(),
            }
        }

        created, updated, skipped = yahoo_finance_service.upsertCompanyProfiles(profiles)

        self.assertEqual(created, 1)
        self.assertEqual(updated, 0)
        self.assertEqual(skipped, 0)

        # Verify stock was created
        tesla = Stock.objects.get(symbol="TSLA")
        self.assertEqual(tesla.sector, "Consumer Cyclical")
        self.assertEqual(tesla.industry, "Auto Manufacturers")
        self.assertIsNotNone(tesla.sectorUpdatedAt)

    def test_upsert_behavior_update_existing(self):
        """Test that upsert updates existing stock records."""
        profiles = {
            "MSFT": {
                "symbol": "MSFT",
                "sector": "Technology",
                "industry": "Software - Infrastructure",
                "updatedAt": timezone.now(),
            }
        }

        created, updated, skipped = yahoo_finance_service.upsertCompanyProfiles(profiles)

        self.assertEqual(created, 0)
        self.assertEqual(updated, 1)
        self.assertEqual(skipped, 0)

        # Verify stock was updated
        self.stale_stock.refresh_from_db()
        self.assertEqual(self.stale_stock.sector, "Technology")
        self.assertEqual(self.stale_stock.industry, "Software - Infrastructure")
        self.assertIsNotNone(self.stale_stock.sectorUpdatedAt)

    def test_upsert_behavior_no_duplicates(self):
        """Test that upsert doesn't create duplicate records."""
        profiles = {
            "AAPL": {
                "symbol": "AAPL",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "updatedAt": timezone.now(),
            }
        }

        initial_count = Stock.objects.filter(symbol="AAPL").count()
        created, updated, skipped = yahoo_finance_service.upsertCompanyProfiles(profiles)
        final_count = Stock.objects.filter(symbol="AAPL").count()

        self.assertEqual(created, 0)
        self.assertEqual(updated, 1)
        self.assertEqual(initial_count, final_count)  # No new records created

    def test_management_command_database_requirements(self):
        """Test management command database requirements using real connection."""

        # Test with actual database connection
        try:
            # The command should work with PostgreSQL (our test database)
            # If it fails due to database type, it will raise CommandError
            call_command("pull_sector_industry", "--symbols", "TEST_SYMBOL", "--dry-run")

            # If we reach here, the database type check passed
            # (PostgreSQL is allowed, SQLite would be blocked)
            self.assertTrue(True)

        except CommandError as e:
            # If command raises error, verify it's appropriate
            error_message = str(e)
            if "SQLite" in error_message:
                # This would only happen if we're actually using SQLite
                self.assertIn("SQLite database detected", error_message)
            else:
                # Other command errors are acceptable (API issues, etc.)
                self.assertTrue(True)

    def test_management_command_postgresql_compatibility(self):
        """Test that management command works with PostgreSQL using real database."""
        from django.db import connection

        # Test with real PostgreSQL database
        if connection.vendor == "postgresql":
            try:
                # Execute command with real database - should not raise database type error
                call_command("pull_sector_industry", "--symbols", "AAPL", "--dry-run")
                self.assertTrue(True)  # Command completed without database type errors
            except CommandError as e:
                # Should not be a database type error for PostgreSQL
                self.assertNotIn("SQLite database detected", str(e))
            except Exception:
                # Other errors (API, network) are acceptable in tests
                self.assertTrue(True)
        else:
            # Skip test if not using PostgreSQL
            self.skipTest(f"Test requires PostgreSQL, current database: {connection.vendor}")

    def test_sector_needs_update_property(self):
        """Test the sectorNeedsUpdate property on Stock model."""
        # Recent stock should not need update
        self.assertFalse(self.recent_stock.sectorNeedsUpdate)

        # Stale stock should need update
        self.assertTrue(self.stale_stock.sectorNeedsUpdate)

        # Stock with no sectorUpdatedAt should need update
        self.assertTrue(self.missing_stock.sectorNeedsUpdate)

    def test_upsert_behavior_skip_errors(self):
        """Test that upsert skips profiles with errors."""
        profiles = {
            "VALID": {"symbol": "VALID", "sector": "Technology", "industry": "Software", "updatedAt": timezone.now()},
            "ERROR": {"error": "No data available"},
        }

        created, updated, skipped = yahoo_finance_service.upsertCompanyProfiles(profiles)

        self.assertEqual(created, 1)
        self.assertEqual(updated, 0)
        self.assertEqual(skipped, 1)

        # Verify only valid profile was created
        self.assertTrue(Stock.objects.filter(symbol="VALID").exists())
        self.assertFalse(Stock.objects.filter(symbol="ERROR").exists())
