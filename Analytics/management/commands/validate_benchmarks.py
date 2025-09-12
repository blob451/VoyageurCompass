"""
Management command to validate benchmark data quality and detect anomalies.
"""

import logging
from datetime import date, timedelta
from django.core.management.base import BaseCommand, CommandError
from django.db.models import F, Q

from Data.models import Stock, StockPrice

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Validate benchmark stock data quality and detect anomalies'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            nargs='+',
            default=['SPY', 'QQQ', 'DIA', 'VTI'],
            help='Benchmark symbols to validate (default: SPY QQQ DIA VTI)'
        )
        
        parser.add_argument(
            '--fix-anomalies',
            action='store_true',
            help='Attempt to fix detected data anomalies'
        )

    def handle(self, *args, **options):
        """Execute benchmark validation."""
        self.stdout.write(self.style.SUCCESS('=== Benchmark Data Validation ==='))
        
        symbols = options['symbols']
        fix_anomalies = options['fix_anomalies']
        
        anomalies_found = []
        
        for symbol in symbols:
            self.stdout.write(f'\nValidating {symbol}...')
            
            try:
                anomalies = self._validate_benchmark(symbol, fix_anomalies)
                if anomalies:
                    anomalies_found.extend(anomalies)
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Validation failed for {symbol}: {str(e)}')
                )
        
        # Summary
        self.stdout.write('\n' + '='*60)
        if anomalies_found:
            self.stdout.write(
                self.style.WARNING(f'Found {len(anomalies_found)} data quality issues')
            )
            for anomaly in anomalies_found:
                self.stdout.write(f'  - {anomaly}')
        else:
            self.stdout.write(
                self.style.SUCCESS('All benchmark data passes validation')
            )

    def _validate_benchmark(self, symbol: str, fix_anomalies: bool = False) -> list:
        """Validate a single benchmark stock."""
        anomalies = []
        
        try:
            stock = Stock.objects.get(symbol=symbol)
            
            # Check if stock has price data
            price_count = StockPrice.objects.filter(stock=stock).count()
            if price_count == 0:
                anomalies.append(f"{symbol}: No price data found")
                self.stdout.write(
                    self.style.ERROR(f'  FAIL: No price data for {symbol}')
                )
                return anomalies
            
            self.stdout.write(f'  INFO: {price_count} price records found')
            
            # Get recent and historical prices
            recent_prices = StockPrice.objects.filter(stock=stock).order_by('-date')[:5]
            old_prices = StockPrice.objects.filter(stock=stock).order_by('date')[:5]
            
            if recent_prices and old_prices:
                latest_price = float(recent_prices[0].close)
                oldest_price = float(old_prices[0].close)
                
                # Check 1-year return
                one_year_ago = date.today() - timedelta(days=365)
                one_year_price = StockPrice.objects.filter(
                    stock=stock, date__gte=one_year_ago
                ).order_by('date').first()
                
                if one_year_price:
                    one_year_return = ((latest_price / float(one_year_price.close)) - 1) * 100
                    self.stdout.write(f'  INFO: 1-year return: {one_year_return:.2f}%')
                    
                    # Realistic range for major ETFs: -40% to +60% annually
                    if one_year_return > 60:
                        anomalies.append(f"{symbol}: Unrealistic 1-year return: {one_year_return:.2f}%")
                        self.stdout.write(
                            self.style.ERROR(f'  FAIL: Unrealistic 1-year return: {one_year_return:.2f}%')
                        )
                    elif one_year_return < -40:
                        anomalies.append(f"{symbol}: Extreme 1-year loss: {one_year_return:.2f}%")
                        self.stdout.write(
                            self.style.WARNING(f'  WARN: Extreme 1-year loss: {one_year_return:.2f}%')
                        )
                    else:
                        self.stdout.write(
                            self.style.SUCCESS(f'  OK: 1-year return within normal range')
                        )
                
                # Check 2-year return
                two_year_ago = date.today() - timedelta(days=730)
                two_year_price = StockPrice.objects.filter(
                    stock=stock, date__gte=two_year_ago
                ).order_by('date').first()
                
                if two_year_price:
                    two_year_return = ((latest_price / float(two_year_price.close)) - 1) * 100
                    self.stdout.write(f'  INFO: 2-year return: {two_year_return:.2f}%')
                    
                    # Realistic range for major ETFs: -60% to +150% over 2 years
                    if two_year_return > 150:
                        anomalies.append(f"{symbol}: Unrealistic 2-year return: {two_year_return:.2f}%")
                        self.stdout.write(
                            self.style.ERROR(f'  FAIL: Unrealistic 2-year return: {two_year_return:.2f}%')
                        )
                    elif two_year_return < -60:
                        anomalies.append(f"{symbol}: Extreme 2-year loss: {two_year_return:.2f}%")
                        self.stdout.write(
                            self.style.WARNING(f'  WARN: Extreme 2-year loss: {two_year_return:.2f}%')
                        )
                    else:
                        self.stdout.write(
                            self.style.SUCCESS(f'  OK: 2-year return within normal range')
                        )
                
                # Check for data gaps in recent period (last 30 days)
                recent_date = date.today() - timedelta(days=30)
                recent_count = StockPrice.objects.filter(
                    stock=stock, date__gte=recent_date
                ).count()
                
                # Should have roughly 22 trading days in 30 calendar days
                if recent_count < 15:
                    anomalies.append(f"{symbol}: Insufficient recent data: {recent_count} records in last 30 days")
                    self.stdout.write(
                        self.style.WARNING(f'  WARN: Only {recent_count} records in last 30 days')
                    )
                else:
                    self.stdout.write(
                        self.style.SUCCESS(f'  OK: {recent_count} recent records (last 30 days)')
                    )
                
                # Check for price anomalies (single-day changes > 10%)
                anomalous_changes = self._check_price_anomalies(stock, symbol)
                if anomalous_changes:
                    anomalies.extend(anomalous_changes)
            
            # If fixing is enabled and anomalies were found, attempt fixes
            if fix_anomalies and anomalies:
                self.stdout.write(f'  INFO: Attempting to fix anomalies for {symbol}...')
                # For now, just log that fixing would happen here
                # In a real implementation, you'd add data correction logic
                self.stdout.write(f'  INFO: Fix logic would be implemented here')
                
        except Stock.DoesNotExist:
            anomalies.append(f"{symbol}: Benchmark stock not found in database")
            self.stdout.write(
                self.style.ERROR(f'  FAIL: {symbol} not found in database')
            )
        
        return anomalies

    def _check_price_anomalies(self, stock, symbol: str) -> list:
        """Check for single-day price anomalies."""
        anomalies = []
        
        # Get prices with day-over-day changes > 10%
        extreme_changes = StockPrice.objects.filter(stock=stock).extra(
            select={
                'prev_close': '''
                    LAG(close) OVER (ORDER BY date)
                ''',
                'pct_change': '''
                    (close / LAG(close) OVER (ORDER BY date) - 1) * 100
                '''
            }
        ).extra(
            where=["ABS((close / LAG(close) OVER (ORDER BY date) - 1) * 100) > 10"]
        )[:10]  # Limit to 10 most extreme
        
        if extreme_changes:
            for change in extreme_changes:
                if hasattr(change, 'pct_change'):
                    anomalies.append(
                        f"{symbol}: Extreme daily change on {change.date}: {change.pct_change:.2f}%"
                    )
            
            self.stdout.write(
                self.style.WARNING(f'  WARN: Found {len(extreme_changes)} extreme daily price changes')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f'  OK: No extreme daily price changes detected')
            )
        
        return anomalies