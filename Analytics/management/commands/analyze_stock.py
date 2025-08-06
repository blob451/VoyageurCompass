"""
Django management command to analyze a stock.
Usage: python manage.py analyze_stock AAPL
"""

from django.core.management.base import BaseCommand
from Analytics.services.engine import analytics_engine
from Data.services.yahoo_finance import yahoo_finance_service
import json


class Command(BaseCommand):
    help = 'Analyze a stock and generate trading signals'

    def add_arguments(self, parser):
        parser.add_argument('symbol', type=str, help='Stock symbol to analyze')
        parser.add_argument(
            '--months',
            type=int,
            default=6,
            help='Number of months to analyze (default: 6)'
        )
        parser.add_argument(
            '--sync',
            action='store_true',
            help='Sync data from Yahoo Finance before analysis'
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output results as JSON'
        )

    def handle(self, *args, **options):
        symbol = options['symbol'].upper()
        months = options['months']
        
        self.stdout.write(f"Analyzing {symbol} for {months} months...")
        
        # Sync data if requested
        if options['sync']:
            self.stdout.write("Syncing data from Yahoo Finance...")
            period = f"{months}mo" if months <= 12 else "2y"
            sync_result = yahoo_finance_service.get_stock_data(symbol, period=period, sync_db=True)
            
            if 'error' in sync_result:
                self.stdout.write(self.style.ERROR(f"Sync failed: {sync_result['error']}"))
                return
        
        # Run analysis
        analysis = analytics_engine.run_full_analysis(symbol, analysis_months=months)
        
        if not analysis.get('success'):
            self.stdout.write(self.style.ERROR(f"Analysis failed: {analysis.get('error')}"))
            return
        
        # Output results
        if options['json']:
            self.stdout.write(json.dumps(analysis, indent=2, default=str))
        else:
            self._display_results(analysis)
    
    def _display_results(self, analysis):
        """Display analysis results in a formatted way."""
        
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("ANALYSIS COMPLETE"))
        self.stdout.write("=" * 60)
        
        self.stdout.write(f"\nCompany: {analysis['company_name']}")
        self.stdout.write(f"Sector: {analysis['sector']}")
        self.stdout.write(f"Analysis Date: {analysis['analysis_date'][:10]}")
        
        self.stdout.write("\n" + "-" * 40)
        self.stdout.write("PRICE INFORMATION")
        self.stdout.write("-" * 40)
        
        if analysis['current_price']:
            self.stdout.write(f"Current Price: ${analysis['current_price']:.2f}")
        if analysis['target_price']:
            self.stdout.write(f"Target Price: ${analysis['target_price']:.2f}")
            if analysis['price_to_target_ratio']:
                self.stdout.write(f"Price/Target Ratio: {analysis['price_to_target_ratio']:.2%}")
        
        self.stdout.write("\n" + "-" * 40)
        self.stdout.write("PERFORMANCE METRICS")
        self.stdout.write("-" * 40)
        
        if analysis['stock_return'] is not None:
            self.stdout.write(f"Stock Return: {analysis['stock_return']:.2f}%")
        if analysis['etf_return'] is not None:
            self.stdout.write(f"Sector ETF Return: {analysis['etf_return']:.2f}%")
        if analysis['outperformance'] is not None:
            self.stdout.write(f"Outperformance: {analysis['outperformance']:.2f}%")
        if analysis['volatility']:
            self.stdout.write(f"Volatility: {analysis['volatility']:.2%} (Threshold: {analysis['volatility_threshold']:.2%})")
        if analysis['sharpe_ratio']:
            self.stdout.write(f"Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
        
        self.stdout.write("\n" + "-" * 40)
        self.stdout.write("TECHNICAL INDICATORS")
        self.stdout.write("-" * 40)
        
        if analysis['rsi']:
            rsi_status = "Oversold" if analysis['rsi_oversold'] else "Overbought" if analysis['rsi_overbought'] else "Normal"
            self.stdout.write(f"RSI: {analysis['rsi']:.2f} ({rsi_status})")
        
        if analysis['ma_20']:
            self.stdout.write(f"MA(20): ${analysis['ma_20']:.2f}")
        if analysis['ma_50']:
            self.stdout.write(f"MA(50): ${analysis['ma_50']:.2f}")
        if analysis['ma_200']:
            self.stdout.write(f"MA(200): ${analysis['ma_200']:.2f}")
        
        if analysis['bollinger_bands']['middle']:
            self.stdout.write(f"\nBollinger Bands:")
            self.stdout.write(f"  Upper: ${analysis['bollinger_bands']['upper']:.2f}")
            self.stdout.write(f"  Middle: ${analysis['bollinger_bands']['middle']:.2f}")
            self.stdout.write(f"  Lower: ${analysis['bollinger_bands']['lower']:.2f}")
            self.stdout.write(f"  Position: {analysis['bb_position']}")
        
        self.stdout.write("\n" + "=" * 60)
        
        # Signal - use color based on signal type
        signal = analysis['signal']
        if signal == 'BUY':
            self.stdout.write(self.style.SUCCESS(f"📈 SIGNAL: {signal}"))
        elif signal == 'SELL':
            self.stdout.write(self.style.ERROR(f"📉 SIGNAL: {signal}"))
        else:
            self.stdout.write(self.style.WARNING(f"⏸️  SIGNAL: {signal}"))
        
        self.stdout.write(f"Reason: {analysis['signal_reason']}")
        
        self.stdout.write("\nCriteria Met:")
        for criterion, met in analysis['criteria_met'].items():
            status = "✓" if met else "✗"
            if met:
                self.stdout.write(self.style.SUCCESS(f"  {status} {criterion}"))
            else:
                self.stdout.write(self.style.ERROR(f"  {status} {criterion}"))
        
        self.stdout.write("=" * 60 + "\n")