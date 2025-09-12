"""
Management command to pull sector/industry data from Yahoo Finance.
Validates database engine and prevents SQLite usage.
"""

import logging

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from Data.models import DataSourceChoices, Stock
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Pull sector/industry data from Yahoo Finance with database engine validation"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            help="Comma-separated list of symbols (e.g., AAPL,MSFT,NVDA). If omitted, uses existing Stock symbols.",
        )

    def handle(self, *args, **options):
        """Execute the sector/industry data pull."""
        try:
            # Validate database engine first
            self.validateDatabaseEngine()

            # Get symbols to process
            symbols = self.getSymbolsToProcess(options.get("symbols"))

            if not symbols:
                self.stdout.write(
                    self.style.ERROR("No symbols to process. Either provide --symbols or ensure Stock table has data.")
                )
                return

            self.stdout.write(f"Processing {len(symbols)} symbols for sector/industry data")
            self.stdout.write(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

            # Perform recency check
            symbolsToFetch = yahoo_finance_service.getStaleAndMissingSymbols(symbols)

            if not symbolsToFetch:
                self.stdout.write(
                    self.style.SUCCESS(
                        "All symbols have recent sector/industry data (within 3 years). No fetch needed."
                    )
                )
                return

            self.stdout.write(f"Fetching data for {len(symbolsToFetch)} symbols that need updates")

            # Fetch sector/industry data
            profiles = yahoo_finance_service.fetchSectorIndustryBatch(symbolsToFetch)

            if not profiles:
                self.stdout.write(self.style.WARNING("No profile data fetched"))
                return

            # Upsert to database
            created, updated, skipped = yahoo_finance_service.upsertCompanyProfiles(profiles)

            # Print summary
            self.stdout.write(
                self.style.SUCCESS(
                    f"Sector/Industry pull complete:\n"
                    f"  - {len(symbols)} total symbols processed\n"
                    f"  - {len(symbolsToFetch)} symbols needed updates\n"
                    f"  - {len(profiles)} profiles fetched from Yahoo Finance\n"
                    f"  - {created} new stock records created\n"
                    f"  - {updated} existing records updated\n"
                    f"  - {skipped} profiles skipped due to errors"
                )
            )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Command failed: {str(e)}"))
            raise CommandError(f"Sector/industry pull failed: {str(e)}")

    def validateDatabaseEngine(self):
        """Validate database engine and abort if SQLite is detected."""
        engine_name = connection.vendor

        if engine_name == "sqlite":
            raise CommandError(
                "SQLite database detected. This command requires PostgreSQL or another production-grade database. "
                "SQLite is not supported for sector/industry data operations due to performance and reliability requirements."
            )

        self.stdout.write(f"Database engine validated: {engine_name}")
        logger.info(f"Database engine validation passed: {engine_name}")

    def getSymbolsToProcess(self, symbols_arg):
        """Get list of symbols to process."""
        if symbols_arg:
            # Parse provided symbols
            raw_symbols = symbols_arg.split(",")
            symbols = []

            for symbol in raw_symbols:
                normalized = symbol.strip().upper()
                if normalized:
                    symbols.append(normalized)

            return symbols
        else:
            # Get symbols from existing Stock records
            symbols = list(
                Stock.objects.filter(is_active=True)
                .exclude(data_source=DataSourceChoices.MOCK)
                .values_list("symbol", flat=True)
                .distinct()
            )

            if not symbols:
                logger.warning("No active real stocks found in database")

            return symbols
