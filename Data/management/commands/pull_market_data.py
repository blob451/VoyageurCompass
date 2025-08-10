"""
Management command to pull market data from Yahoo Finance.
"""

from django.core.management.base import BaseCommand
from Data.services.yahoo_finance import yahoo_finance_service
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Pull market data from Yahoo Finance'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--tickers',
            type=str,
            default='AAPL,MSFT,NVDA',
            help='Comma-separated list of tickers'
        )
        parser.add_argument(
            '--range',
            type=str,
            default='6mo',
            help='Date range (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)'
        )
        parser.add_argument(
            '--interval',
            type=str,
            default='1d',
            help='Bar interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)'
        )
        parser.add_argument(
            '--batch',
            type=str,
            default='yes',
            choices=['yes', 'no'],
            help='Use batch mode for multiple tickers'
        )
    
    def handle(self, *args, **options):
        """Execute the data pull."""
        # Parse and validate ticker symbols with proper normalization
        raw_tickers = options['tickers'].split(',')
        tickers = []
        
        for ticker in raw_tickers:
            # Strip whitespace and convert to uppercase
            normalized_ticker = ticker.strip().upper()
            # Filter out empty entries
            if normalized_ticker:
                tickers.append(normalized_ticker)
        
        # Check if ticker list is empty after filtering
        if not tickers:
            self.stdout.write(
                self.style.ERROR('Error: No valid tickers provided. Please specify at least one ticker symbol.')
            )
            return
        
        range_str = options['range']
        interval = options['interval']
        batch_mode = options['batch'] == 'yes'
        
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
                                f'{ticker}: {len(bars)} bars fetched, {created} saved, {skipped} skipped'
                            )
                        )
                    else:
                        self.stdout.write(self.style.WARNING(f'{ticker}: No data fetched'))
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Total: {total_created} bars saved, {total_skipped} duplicates skipped'
                    )
                )
            else:
                # Single ticker fetch
                ticker = tickers[0]
                bars = service.fetchSingleHistorical(ticker, range_str, interval)
                
                if bars:
                    created, skipped = service.saveBars(bars)
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'{ticker}: {len(bars)} bars fetched, {created} saved, {skipped} skipped'
                        )
                    )
                else:
                    self.stdout.write(self.style.WARNING(f'{ticker}: No data fetched'))
                    
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {str(e)}'))
            raise