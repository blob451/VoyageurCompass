"""
Data Quality Verification Command

Runs comprehensive verification checks after data quality improvements.
Compares before/after metrics and validates that fixes were successful.
"""

import logging
from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta

from Data.models import Stock, StockPrice, DataSector, DataIndustry
from Data.services.data_quality_monitor import data_quality_monitor

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Verify data quality improvements and generate detailed report'

    def add_arguments(self, parser):
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed breakdown of improvements'
        )
        
        parser.add_argument(
            '--export',
            help='Export results to file (JSON format)'
        )

    def handle(self, *args, **options):
        """Execute data quality verification."""
        detailed = options['detailed']
        export_file = options.get('export')
        
        self.stdout.write(self.style.SUCCESS('=== DATA QUALITY VERIFICATION ==='))
        
        # Run comprehensive quality check
        self.stdout.write('Running comprehensive data quality analysis...')
        quality_results = data_quality_monitor.run_comprehensive_check()
        
        if 'error' in quality_results:
            self.stdout.write(self.style.ERROR(f'Quality check failed: {quality_results["error"]}'))
            return
        
        # Generate detailed statistics
        stats = self._generate_detailed_stats()
        
        # Display results
        self._display_verification_results(quality_results, stats, detailed)
        
        # Check if target metrics are met
        self._check_target_metrics(quality_results, stats)
        
        # Export results if requested
        if export_file:
            self._export_results(quality_results, stats, export_file)
            self.stdout.write(f'Results exported to: {export_file}')

    def _generate_detailed_stats(self):
        """Generate detailed statistics about the current data state."""
        stats = {}
        
        # Stock statistics
        total_stocks = Stock.objects.count()
        real_stocks = self._get_real_stocks_query().count()
        
        stocks_with_prices = Stock.objects.filter(prices__isnull=False).distinct().count()
        real_stocks_with_prices = self._get_real_stocks_query().filter(prices__isnull=False).distinct().count()
        
        # Data coverage by time
        week_ago = timezone.now().date() - timedelta(days=7)
        month_ago = timezone.now().date() - timedelta(days=30)
        
        stocks_with_recent_data = Stock.objects.filter(
            prices__date__gte=week_ago
        ).distinct().count()
        
        stocks_with_monthly_data = Stock.objects.filter(
            prices__date__gte=month_ago
        ).distinct().count()
        
        # Sector/Industry mappings
        stocks_with_sector = Stock.objects.filter(sector_id__isnull=False).count()
        stocks_with_industry = Stock.objects.filter(industry_id__isnull=False).count()
        
        real_stocks_with_sector = self._get_real_stocks_query().filter(sector_id__isnull=False).count()
        real_stocks_with_industry = self._get_real_stocks_query().filter(industry_id__isnull=False).count()
        
        # Price data volume
        total_price_records = StockPrice.objects.count()
        avg_records_per_stock = total_price_records / max(stocks_with_prices, 1)
        
        # Sector/Industry counts
        total_sectors = DataSector.objects.count()
        active_sectors = DataSector.objects.filter(isActive=True).count()
        total_industries = DataIndustry.objects.count()
        active_industries = DataIndustry.objects.filter(isActive=True).count()
        
        # Test vs real stock breakdown
        test_stocks = total_stocks - real_stocks
        
        stats = {
            'stocks': {
                'total_stocks': total_stocks,
                'real_stocks': real_stocks,
                'test_stocks': test_stocks,
                'stocks_with_prices': stocks_with_prices,
                'real_stocks_with_prices': real_stocks_with_prices,
                'stocks_with_recent_data': stocks_with_recent_data,
                'stocks_with_monthly_data': stocks_with_monthly_data,
                'coverage_percentage': (real_stocks_with_prices / max(real_stocks, 1)) * 100,
                'recent_data_percentage': (stocks_with_recent_data / max(total_stocks, 1)) * 100
            },
            'mappings': {
                'stocks_with_sector': stocks_with_sector,
                'stocks_with_industry': stocks_with_industry,
                'real_stocks_with_sector': real_stocks_with_sector,
                'real_stocks_with_industry': real_stocks_with_industry,
                'sector_mapping_percentage': (real_stocks_with_sector / max(real_stocks, 1)) * 100,
                'industry_mapping_percentage': (real_stocks_with_industry / max(real_stocks, 1)) * 100
            },
            'data_volume': {
                'total_price_records': total_price_records,
                'avg_records_per_stock': round(avg_records_per_stock, 1),
                'total_sectors': total_sectors,
                'active_sectors': active_sectors,
                'total_industries': total_industries,
                'active_industries': active_industries
            },
            'timestamp': timezone.now().isoformat()
        }
        
        return stats

    def _get_real_stocks_query(self):
        """Get queryset for real stocks (excluding test data)."""
        test_patterns = [
            'AAPL_', 'CASCADE_TEST', 'CONSTRAINT_TEST', 'NO_PRICES', 
            'PORTFOLIO_TEST', 'TSLA_MOCK', 'TEST_STOCK', 'PERF_', 'ROLLBACK_TEST'
        ]
        
        real_stocks_query = Stock.objects.filter(is_active=True)
        for pattern in test_patterns:
            real_stocks_query = real_stocks_query.exclude(symbol__icontains=pattern)
        
        return real_stocks_query

    def _display_verification_results(self, quality_results, stats, detailed):
        """Display verification results in a formatted way."""
        self.stdout.write('\n' + '='*80)
        self.stdout.write(self.style.SUCCESS('DATA QUALITY VERIFICATION RESULTS'))
        self.stdout.write('='*80)
        
        # Overall Quality Score
        overall_score = quality_results.get('overall_quality_score', 0)
        summary = quality_results.get('summary', {})
        
        self.stdout.write(f'\nOVERALL QUALITY SCORE: {overall_score:.1f}/10')
        
        score_color = self.style.SUCCESS if overall_score >= 7.5 else self.style.WARNING if overall_score >= 5.0 else self.style.ERROR
        status = summary.get('overall_status', 'unknown').upper()
        self.stdout.write(score_color(f'STATUS: {status}'))
        
        # Stock Data Coverage
        self.stdout.write('\n--- STOCK DATA COVERAGE ---')
        self.stdout.write(f'Total Stocks: {stats["stocks"]["total_stocks"]} ({stats["stocks"]["real_stocks"]} real, {stats["stocks"]["test_stocks"]} test)')
        self.stdout.write(f'Stocks with Price Data: {stats["stocks"]["real_stocks_with_prices"]}/{stats["stocks"]["real_stocks"]} ({stats["stocks"]["coverage_percentage"]:.1f}%)')
        self.stdout.write(f'Stocks with Recent Data (7d): {stats["stocks"]["stocks_with_recent_data"]} ({stats["stocks"]["recent_data_percentage"]:.1f}%)')
        self.stdout.write(f'Average Price Records per Stock: {stats["data_volume"]["avg_records_per_stock"]}')
        self.stdout.write(f'Total Price Records: {stats["data_volume"]["total_price_records"]:,}')
        
        # Sector/Industry Mappings
        self.stdout.write('\n--- SECTOR/INDUSTRY MAPPINGS ---')
        self.stdout.write(f'Real Stocks with Sector Mapping: {stats["mappings"]["real_stocks_with_sector"]}/{stats["stocks"]["real_stocks"]} ({stats["mappings"]["sector_mapping_percentage"]:.1f}%)')
        self.stdout.write(f'Real Stocks with Industry Mapping: {stats["mappings"]["real_stocks_with_industry"]}/{stats["stocks"]["real_stocks"]} ({stats["mappings"]["industry_mapping_percentage"]:.1f}%)')
        self.stdout.write(f'Total Sectors: {stats["data_volume"]["total_sectors"]} ({stats["data_volume"]["active_sectors"]} active)')
        self.stdout.write(f'Total Industries: {stats["data_volume"]["total_industries"]} ({stats["data_volume"]["active_industries"]} active)')
        
        # Quality Metrics
        if detailed:
            self._display_detailed_quality_metrics(quality_results)
        
        # Issues and Recommendations
        recommendations = quality_results.get('recommendations', [])
        if recommendations:
            self.stdout.write('\n--- CURRENT ISSUES ---')
            for i, rec in enumerate(recommendations[:10], 1):  # Show top 10
                severity = rec.get('severity', 'medium').upper()
                severity_color = {
                    'HIGH': self.style.ERROR,
                    'MEDIUM': self.style.WARNING,
                    'LOW': self.style.SUCCESS
                }.get(severity, self.style.SUCCESS)
                
                self.stdout.write(f'{i:2d}. {severity_color(f"[{severity}]")} {rec.get("message", "")}')
                if detailed and rec.get('action'):
                    self.stdout.write(f'    Action: {rec["action"]}')

    def _display_detailed_quality_metrics(self, quality_results):
        """Display detailed quality metrics."""
        self.stdout.write('\n--- DETAILED QUALITY METRICS ---')
        
        # Stock data quality
        stock_quality = quality_results.get('stock_data_quality', {})
        if stock_quality:
            self.stdout.write(f'Stock Data Quality Score: {stock_quality.get("quality_score", 0):.1f}/10')
            self.stdout.write(f'  - Stocks with sufficient history (252+ days): {stock_quality.get("stocks_with_sufficient_history", 0)}')
            self.stdout.write(f'  - Stocks with gaps in last 30 days: {stock_quality.get("stocks_with_gaps", 0)}')
        
        # Data freshness
        freshness = quality_results.get('data_freshness', {})
        if freshness:
            self.stdout.write(f'Data Freshness Score: {freshness.get("overall_freshness_score", 0):.1f}/10')
            self.stdout.write(f'  - Stock data staleness: {freshness.get("stock_data_staleness_days", 0)} days')
            self.stdout.write(f'  - Latest stock data: {freshness.get("latest_stock_date", "N/A")}')
        
        # Gap analysis
        gaps = quality_results.get('gap_analysis', {})
        if gaps:
            self.stdout.write(f'Gap Analysis Score: {gaps.get("gap_score", 0):.1f}/10')
            self.stdout.write(f'  - Stocks with significant gaps: {gaps.get("stocks_with_significant_gaps", 0)}')
            self.stdout.write(f'  - Average gap percentage: {gaps.get("average_gap_percentage", 0):.1f}%')

    def _check_target_metrics(self, quality_results, stats):
        """Check if target improvement metrics have been met."""
        self.stdout.write('\n--- TARGET METRICS CHECK ---')
        
        targets = {
            'data_coverage': {'target': 80.0, 'current': stats['stocks']['coverage_percentage'], 'unit': '%'},
            'sector_mapping': {'target': 95.0, 'current': stats['mappings']['sector_mapping_percentage'], 'unit': '%'},
            'industry_mapping': {'target': 90.0, 'current': stats['mappings']['industry_mapping_percentage'], 'unit': '%'},
            'overall_quality_score': {'target': 7.5, 'current': quality_results.get('overall_quality_score', 0), 'unit': '/10'},
            'data_freshness': {'target': 24, 'current': quality_results.get('data_freshness', {}).get('stock_data_staleness_days', 999), 'unit': ' hours', 'reverse': True}
        }
        
        all_targets_met = True
        
        for metric, data in targets.items():
            target = data['target']
            current = data['current']
            unit = data['unit']
            reverse = data.get('reverse', False)  # For metrics where lower is better
            
            if reverse:
                met = current <= target
                comparison = f'<= {target}'
            else:
                met = current >= target
                comparison = f'>= {target}'
            
            status_color = self.style.SUCCESS if met else self.style.ERROR
            status = 'PASS' if met else 'FAIL'
            
            if not met:
                all_targets_met = False
            
            self.stdout.write(f'{metric.replace("_", " ").title()}: {status_color(f"{current:.1f}{unit} ({comparison})")} - {status_color(status)}')
        
        self.stdout.write('\n' + '-'*80)
        if all_targets_met:
            self.stdout.write(self.style.SUCCESS('ALL TARGET METRICS ACHIEVED!'))
        else:
            self.stdout.write(self.style.WARNING('Some target metrics not yet achieved. Continue improvements.'))

    def _export_results(self, quality_results, stats, export_file):
        """Export verification results to JSON file."""
        import json
        
        export_data = {
            'verification_timestamp': timezone.now().isoformat(),
            'quality_results': quality_results,
            'detailed_stats': stats,
            'summary': {
                'overall_score': quality_results.get('overall_quality_score', 0),
                'data_coverage_percentage': stats['stocks']['coverage_percentage'],
                'sector_mapping_percentage': stats['mappings']['sector_mapping_percentage'],
                'industry_mapping_percentage': stats['mappings']['industry_mapping_percentage'],
                'total_price_records': stats['data_volume']['total_price_records'],
                'issues_count': len(quality_results.get('recommendations', []))
            }
        }
        
        try:
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to export results: {str(e)}'))