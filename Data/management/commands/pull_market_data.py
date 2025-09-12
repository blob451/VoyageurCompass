"""
Management command to pull market data from Yahoo Finance.
"""

import logging

from django.core.management.base import BaseCommand, CommandError

from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Pull market data from Yahoo Finance"

    def add_arguments(self, parser):
        parser.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA", help="Comma-separated list of tickers")
        parser.add_argument(
            "--range", type=str, default="6mo", help="Date range (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1d",
            help="Bar interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)",
        )
        parser.add_argument(
            "--batch",
            action="store_true",
            default=True,
            help="Enable batch mode for processing multiple tickers efficiently (default: enabled)",
        )
        parser.add_argument(
            "--no-batch", action="store_false", dest="batch", help="Disable batch mode and process tickers individually"
        )

    def handle(self, *args, **options):
        """Execute the data pull."""
        # Parse and validate ticker symbols with proper normalization
        raw_tickers = options["tickers"].split(",")
        tickers = []
        seen_tickers = set()

        for ticker in raw_tickers:
            # Strip whitespace and convert to uppercase
            normalized_ticker = ticker.strip().upper()
            # Filter out empty entries and duplicates
            if normalized_ticker and normalized_ticker not in seen_tickers:
                tickers.append(normalized_ticker)
                seen_tickers.add(normalized_ticker)

        # Check if ticker list is empty after filtering
        if not tickers:
            raise CommandError("No valid tickers provided. Please specify at least one ticker symbol.")

        range_str = options["range"]
        interval = options["interval"]
        batch_mode = options["batch"]

        self.stdout.write(f"Pulling data for {tickers} - {range_str} @ {interval}")
        self.stdout.write(f"Batch mode: {batch_mode}")

        service = yahoo_finance_service

        try:
            if batch_mode and len(tickers) > 1:
                # Batch fetch
                results = service.fetchBatchHistorical(tickers, range_str, interval)

                total_created = 0
                total_skipped = 0

                for ticker, bars in results.items():
                    if bars:
                        created, skipped = service.saveBars(bars)
                        total_created += created
                        total_skipped += skipped
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"{ticker}: {len(bars)} bars fetched, {created} saved, {skipped} skipped"
                            )
                        )
                    else:
                        self.stdout.write(self.style.WARNING(f"{ticker}: No data fetched"))

                self.stdout.write(
                    self.style.SUCCESS(f"Total: {total_created} bars saved, {total_skipped} duplicates skipped")
                )
            else:
                # Individual ticker processing (non-batch mode)
                total_created = 0
                total_skipped = 0

                for ticker in tickers:
                    bars = service.fetchSingleHistorical(ticker, range_str, interval)

                    if bars:
                        created, skipped = service.saveBars(bars)
                        total_created += created
                        total_skipped += skipped
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"{ticker}: {len(bars)} bars fetched, {created} saved, {skipped} skipped"
                            )
                        )
                    else:
                        self.stdout.write(self.style.WARNING(f"{ticker}: No data fetched"))

                # Show total summary if multiple tickers were processed
                if len(tickers) > 1:
                    self.stdout.write(
                        self.style.SUCCESS(f"Total: {total_created} bars saved, {total_skipped} duplicates skipped")
                    )

        except Exception as e:
            # Log the full exception with traceback (no need to format the exception)
            logger.exception("Error pulling market data")

            # Raise a user-friendly error message
            ticker_list = (
                ", ".join(tickers) if len(tickers) <= 5 else f"{', '.join(tickers[:5])}, and {len(tickers)-5} more"
            )
            raise CommandError(
                f"Failed to pull market data for {ticker_list}. "
                f"Please check your internet connection and ticker symbols. "
                f"See logs for detailed error information."
            )
        # Note: We don't close the service here since it's a singleton that might be reused
        # The service's session will be cleaned up when the process ends
