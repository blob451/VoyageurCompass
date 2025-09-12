"""
Data Quality Monitoring Command

Runs comprehensive data quality checks and reports on system health.
"""

import json
from django.core.management.base import BaseCommand
from Data.services.data_quality_monitor import data_quality_monitor


class Command(BaseCommand):
    help = 'Monitor data quality across stocks, sectors, and industries'

    def add_arguments(self, parser):
        parser.add_argument(
            '--format',
            choices=['text', 'json'],
            default='text',
            help='Output format (default: text)'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output'
        )

    def handle(self, *args, **options):
        """Execute data quality monitoring."""
        self.stdout.write(self.style.SUCCESS('=== DATA QUALITY MONITORING ==='))
        
        # Run comprehensive check
        self.stdout.write('Running comprehensive data quality check...')
        result = data_quality_monitor.run_comprehensive_check()
        
        if 'error' in result:
            self.stdout.write(self.style.ERROR(f'Error: {result["error"]}'))
            return
        
        # Output format
        if options['format'] == 'json':
            self.stdout.write(json.dumps(result, indent=2))
            return
        
        # Text format output
        self._display_text_report(result, options['verbose'])
    
    def _display_text_report(self, result: dict, verbose: bool = False):
        """Display data quality report in text format."""
        # Summary
        summary = result.get('summary', {})
        self.stdout.write('\n' + '='*60)
        self.stdout.write(f"OVERALL DATA QUALITY: {summary.get('overall_status', 'unknown').upper()}")
        self.stdout.write(f"Quality Score: {summary.get('overall_score', 0):.1f}/10")
        self.stdout.write(f"Issues Found: {summary.get('issues_found', 0)}")
        self.stdout.write('='*60)
        
        # Individual metrics
        self.stdout.write('\n📊 DETAILED METRICS:')
        
        # Stock data quality
        stock_quality = result.get('stock_data_quality', {})
        if stock_quality:
            self.stdout.write(f"\n🏢 Stock Data Quality: {stock_quality.get('quality_score', 0):.1f}/10")
            self.stdout.write(f"  • Total stocks: {stock_quality.get('total_stocks', 0)}")
            self.stdout.write(f"  • With recent data: {stock_quality.get('stocks_with_recent_data', 0)}")
            self.stdout.write(f"  • With sufficient history: {stock_quality.get('stocks_with_sufficient_history', 0)}")
            self.stdout.write(f"  • Avg data points/stock: {stock_quality.get('average_data_points_per_stock', 0)}")
        
        # Data freshness
        freshness = result.get('data_freshness', {})
        if freshness:
            self.stdout.write(f"\n📅 Data Freshness: {freshness.get('overall_freshness_score', 0):.1f}/10")
            self.stdout.write(f"  • Stock data staleness: {freshness.get('stock_data_staleness_days', 0)} days")
            self.stdout.write(f"  • Sector data staleness: {freshness.get('sector_data_staleness_days', 0)} days")
            self.stdout.write(f"  • Industry data staleness: {freshness.get('industry_data_staleness_days', 0)} days")
        
        # Gap analysis
        gaps = result.get('gap_analysis', {})
        if gaps:
            self.stdout.write(f"\n🕳️  Data Gaps: {gaps.get('gap_score', 0):.1f}/10")
            self.stdout.write(f"  • Stocks analyzed: {gaps.get('total_stocks_analyzed', 0)}")
            self.stdout.write(f"  • With significant gaps: {gaps.get('stocks_with_significant_gaps', 0)}")
            self.stdout.write(f"  • High severity gaps: {gaps.get('high_severity_gaps', 0)}")
            self.stdout.write(f"  • Average gap percentage: {gaps.get('average_gap_percentage', 0):.1f}%")
        
        # Anomalies
        anomalies = result.get('anomaly_detection', {})
        if anomalies:
            self.stdout.write(f"\n⚠️  Anomaly Detection: {anomalies.get('anomaly_score', 0):.1f}/10")
            self.stdout.write(f"  • Total anomalies: {anomalies.get('total_anomalies_detected', 0)}")
            anomaly_types = anomalies.get('anomaly_types', {})
            if anomaly_types:
                for anomaly_type, count in anomaly_types.items():
                    self.stdout.write(f"  • {anomaly_type.replace('_', ' ').title()}: {count}")
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            self.stdout.write('\n🔧 RECOMMENDATIONS:')
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                severity = rec.get('severity', 'medium')
                severity_color = {
                    'high': self.style.ERROR,
                    'medium': self.style.WARNING,
                    'low': self.style.SUCCESS
                }.get(severity, self.style.SUCCESS)
                
                self.stdout.write(f"\n{i}. {severity_color(f'[{severity.upper()}]')} {rec.get('message', '')}")
                self.stdout.write(f"   Action: {rec.get('action', 'No action specified')}")
        
        # Top gap stocks (if verbose)
        if verbose and gaps.get('top_gap_stocks'):
            self.stdout.write('\n📉 TOP STOCKS WITH DATA GAPS:')
            for stock in gaps['top_gap_stocks'][:10]:
                self.stdout.write(f"  • {stock['symbol']}: {stock['gap_percentage']:.1f}% gap ({stock['actual_days']}/{stock['expected_days']} days)")
        
        # Recent anomalies (if verbose)
        if verbose and anomalies.get('anomalies'):
            self.stdout.write('\n🚨 RECENT ANOMALIES:')
            for anomaly in anomalies['anomalies'][:10]:
                self.stdout.write(f"  • {anomaly['symbol']}: {anomaly['type']} on {anomaly['date']} [{anomaly['severity']}]")
        
        self.stdout.write(f"\n✅ Report generated at: {result.get('timestamp', 'unknown')}")
        self.stdout.write('\n' + '='*60)