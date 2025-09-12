"""
Management command to monitor data quality and alert on issues.
"""

import logging
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Q, F, Min, Max
from django.db import connection

from Data.models import Stock, StockPrice, DataSectorPrice, DataIndustryPrice, AnalyticsResults

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Monitor data quality and alert on issues'

    def add_arguments(self, parser):
        parser.add_argument(
            '--alert-threshold',
            type=int,
            default=80,
            help='Percentage threshold for data coverage alerts (default: 80)'
        )
        
        parser.add_argument(
            '--days-back',
            type=int,
            default=30,
            help='Number of days to check for recent data (default: 30)'
        )
        
        parser.add_argument(
            '--output-format',
            choices=['text', 'json', 'summary'],
            default='text',
            help='Output format for alerts'
        )

    def handle(self, *args, **options):
        """Execute data quality monitoring."""
        self.stdout.write(self.style.SUCCESS('=== Data Quality Monitoring ===\n'))
        
        alert_threshold = options['alert_threshold']
        days_back = options['days_back']
        output_format = options['output_format']
        
        alerts = []
        
        try:
            # 1. Check stock data coverage
            stock_alerts = self._check_stock_data_coverage(alert_threshold, days_back)
            alerts.extend(stock_alerts)
            
            # 2. Check sector/industry data integrity
            sector_alerts = self._check_sector_industry_integrity()
            alerts.extend(sector_alerts)
            
            # 3. Check for anomalous price data
            price_alerts = self._check_anomalous_prices(days_back)
            alerts.extend(price_alerts)
            
            # 4. Check analysis performance
            analysis_alerts = self._check_analysis_performance(days_back)
            alerts.extend(analysis_alerts)
            
            # 5. Check benchmark data availability
            benchmark_alerts = self._check_benchmark_availability()
            alerts.extend(benchmark_alerts)
            
            # Output results
            if output_format == 'json':
                self._output_json(alerts)
            elif output_format == 'summary':
                self._output_summary(alerts)
            else:
                self._output_text(alerts)
                
            if alerts:
                self.stdout.write(
                    self.style.WARNING(f'\nFound {len(alerts)} data quality issues')
                )
            else:
                self.stdout.write(
                    self.style.SUCCESS('\nNo data quality issues detected')
                )
                
        except Exception as e:
            raise CommandError(f'Data quality monitoring failed: {str(e)}')

    def _check_stock_data_coverage(self, threshold: int, days_back: int) -> list:
        """Check stock data coverage and freshness."""
        alerts = []
        cutoff_date = datetime.now().date() - timedelta(days=days_back)
        
        # Check stocks without recent price data
        stocks_without_recent_data = Stock.objects.annotate(
            latest_price_date=Max('prices__date')
        ).filter(
            Q(latest_price_date__lt=cutoff_date) | Q(latest_price_date__isnull=True),
            is_active=True
        ).count()
        
        total_active_stocks = Stock.objects.filter(is_active=True).count()
        
        if total_active_stocks > 0:
            coverage_pct = ((total_active_stocks - stocks_without_recent_data) / total_active_stocks) * 100
            
            if coverage_pct < threshold:
                alerts.append({
                    'type': 'data_coverage',
                    'severity': 'high' if coverage_pct < 50 else 'medium',
                    'message': f'Stock data coverage is {coverage_pct:.1f}% (below {threshold}% threshold)',
                    'details': {
                        'stocks_without_data': stocks_without_recent_data,
                        'total_stocks': total_active_stocks,
                        'coverage_percentage': coverage_pct,
                        'cutoff_date': cutoff_date.isoformat()
                    }
                })
        
        return alerts

    def _check_sector_industry_integrity(self) -> list:
        """Check sector and industry data integrity."""
        alerts = []
        
        # Check stocks without sector/industry mappings
        stocks_without_sector = Stock.objects.filter(
            sector_id__isnull=True,
            is_active=True
        ).count()
        
        stocks_without_industry = Stock.objects.filter(
            industry_id__isnull=True,
            is_active=True
        ).count()
        
        total_stocks = Stock.objects.filter(is_active=True).count()
        
        if stocks_without_sector > 0:
            pct_missing = (stocks_without_sector / total_stocks) * 100
            alerts.append({
                'type': 'sector_mapping',
                'severity': 'medium' if pct_missing > 20 else 'low',
                'message': f'{stocks_without_sector} stocks ({pct_missing:.1f}%) missing sector mapping',
                'details': {
                    'missing_count': stocks_without_sector,
                    'total_stocks': total_stocks,
                    'percentage': pct_missing
                }
            })
            
        if stocks_without_industry > 0:
            pct_missing = (stocks_without_industry / total_stocks) * 100
            alerts.append({
                'type': 'industry_mapping',
                'severity': 'medium' if pct_missing > 20 else 'low',
                'message': f'{stocks_without_industry} stocks ({pct_missing:.1f}%) missing industry mapping',
                'details': {
                    'missing_count': stocks_without_industry,
                    'total_stocks': total_stocks,
                    'percentage': pct_missing
                }
            })
        
        return alerts

    def _check_anomalous_prices(self, days_back: int) -> list:
        """Check for anomalous price data."""
        alerts = []
        cutoff_date = datetime.now().date() - timedelta(days=days_back)
        
        with connection.cursor() as cursor:
            # Check for extreme price changes (>50% in one day)
            cursor.execute("""
                SELECT symbol, date, close, prev_close, change_pct
                FROM (
                    SELECT symbol, date, close, 
                           LAG(close) OVER (PARTITION BY stock_id ORDER BY date) as prev_close,
                           (close / LAG(close) OVER (PARTITION BY stock_id ORDER BY date) - 1) * 100 as change_pct
                    FROM "Data_stockprice" sp
                    JOIN "Data_stock" s ON sp.stock_id = s.id
                    WHERE sp.date >= %s
                ) subq
                WHERE prev_close IS NOT NULL
                  AND ABS(change_pct) > 50
                ORDER BY ABS(change_pct) DESC
                LIMIT 10
            """, [cutoff_date])
            
            extreme_changes = cursor.fetchall()
            
            if extreme_changes:
                alerts.append({
                    'type': 'anomalous_prices',
                    'severity': 'high',
                    'message': f'Found {len(extreme_changes)} stocks with extreme price changes (>50% in one day)',
                    'details': {
                        'extreme_changes': [
                            {
                                'symbol': row[0],
                                'date': row[1].isoformat(),
                                'price': float(row[2]),
                                'prev_price': float(row[3]),
                                'change_pct': float(row[4])
                            }
                            for row in extreme_changes[:5]  # Limit details to top 5
                        ]
                    }
                })
        
        return alerts

    def _check_analysis_performance(self, days_back: int) -> list:
        """Check analysis performance and failure rates."""
        alerts = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        total_analyses = AnalyticsResults.objects.filter(
            as_of__gte=cutoff_date
        ).count()
        
        if total_analyses > 0:
            # Check for very low or very high scores (potential issues)
            extreme_scores = AnalyticsResults.objects.filter(
                as_of__gte=cutoff_date,
                composite_raw__lt=0.1  # Very low scores
            ).count()
            
            if extreme_scores > total_analyses * 0.1:  # More than 10% very low scores
                alerts.append({
                    'type': 'analysis_performance',
                    'severity': 'medium',
                    'message': f'{extreme_scores} analyses ({(extreme_scores/total_analyses)*100:.1f}%) had very low scores',
                    'details': {
                        'low_score_count': extreme_scores,
                        'total_analyses': total_analyses,
                        'percentage': (extreme_scores/total_analyses)*100
                    }
                })
        
        return alerts

    def _check_benchmark_availability(self) -> list:
        """Check availability of benchmark stocks."""
        alerts = []
        
        required_benchmarks = ['SPY', 'QQQ', 'DIA', 'VTI']
        missing_benchmarks = []
        
        for benchmark in required_benchmarks:
            if not Stock.objects.filter(symbol=benchmark).exists():
                missing_benchmarks.append(benchmark)
        
        if missing_benchmarks:
            alerts.append({
                'type': 'missing_benchmarks',
                'severity': 'high',
                'message': f'Missing benchmark stocks: {", ".join(missing_benchmarks)}',
                'details': {
                    'missing_benchmarks': missing_benchmarks,
                    'required_benchmarks': required_benchmarks
                }
            })
        
        return alerts

    def _output_text(self, alerts: list):
        """Output alerts in text format."""
        if not alerts:
            return
            
        self.stdout.write(self.style.HTTP_INFO('Data Quality Alerts:'))
        
        for i, alert in enumerate(alerts, 1):
            severity_style = {
                'high': self.style.ERROR,
                'medium': self.style.WARNING,
                'low': self.style.NOTICE
            }.get(alert['severity'], self.style.NOTICE)
            
            self.stdout.write(f"\n{i}. {severity_style(alert['severity'].upper())}: {alert['message']}")
            self.stdout.write(f"   Type: {alert['type']}")

    def _output_summary(self, alerts: list):
        """Output alerts summary."""
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        type_counts = {}
        
        for alert in alerts:
            severity_counts[alert['severity']] += 1
            type_counts[alert['type']] = type_counts.get(alert['type'], 0) + 1
        
        self.stdout.write("Data Quality Summary:")
        self.stdout.write(f"  Total Issues: {len(alerts)}")
        self.stdout.write(f"  High Severity: {severity_counts['high']}")
        self.stdout.write(f"  Medium Severity: {severity_counts['medium']}")
        self.stdout.write(f"  Low Severity: {severity_counts['low']}")
        
        if type_counts:
            self.stdout.write("\nIssue Types:")
            for issue_type, count in type_counts.items():
                self.stdout.write(f"  {issue_type}: {count}")

    def _output_json(self, alerts: list):
        """Output alerts in JSON format."""
        import json
        self.stdout.write(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(alerts),
            'alerts': alerts
        }, indent=2))