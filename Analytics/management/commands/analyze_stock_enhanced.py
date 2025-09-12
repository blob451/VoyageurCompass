"""
Enhanced Analytics Orchestration Command

Uses enhanced TA engine with adaptive indicators, multi-source data fetching,
and comprehensive confidence scoring for improved reliability.

Usage:
    python manage.py analyze_stock_enhanced --symbol AAPL
    python manage.py analyze_stock_enhanced --symbol PSA --detailed
"""

import logging
import os
import uuid
from typing import Any, Dict

from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.utils import timezone

from Analytics.engine.enhanced_ta_engine import enhanced_ta_engine
from Data.models import Stock
from Data.services.data_verification import data_verification_service
from Data.services.enhanced_backfill import enhanced_backfill_service
from Core.utils.cache_utils import sanitize_cache_key

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Enhanced analytics orchestration command with adaptive indicators and multi-source support.
    """

    help = "Run enhanced technical analysis with adaptive indicators and confidence scoring"

    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument("--symbol", type=str, required=True, help="Stock ticker symbol to analyze (required)")
        parser.add_argument(
            "--horizon",
            type=str,
            default="blend", 
            choices=["short", "medium", "long", "blend"],
            help="Analysis time horizon (default: blend)",
        )
        parser.add_argument(
            "--required-years", type=int, default=2, help="Years of historical data required (default: 2)"
        )
        parser.add_argument(
            "--skip-backfill", action="store_true", help="Skip enhanced backfill and proceed with existing data"
        )
        parser.add_argument(
            "--include-sentiment",
            action="store_true",
            default=True,
            help="Include sentiment analysis from news (default: True)",
        )
        parser.add_argument(
            "--skip-sentiment", action="store_true", help="Skip sentiment analysis to speed up processing"
        )
        parser.add_argument(
            "--detailed", action="store_true", help="Show detailed analysis breakdown and recommendations"
        )
        parser.add_argument(
            "--confidence-threshold", type=float, default=0.6, 
            help="Minimum confidence threshold for analysis (default: 0.6)"
        )
        parser.add_argument(
            "--export-report", type=str, help="Export detailed report to JSON file"
        )

    def handle(self, *args, **options):
        """Main command handler implementing enhanced orchestration workflow."""
        symbol = options["symbol"].upper()
        horizon = options["horizon"]
        required_years = options["required_years"]
        skip_backfill = options["skip_backfill"]
        include_sentiment = options["include_sentiment"] and not options["skip_sentiment"]
        detailed = options["detailed"]
        confidence_threshold = options["confidence_threshold"]
        export_report = options.get("export_report")

        # Generate analysis ID
        timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
        analysis_id = f"{symbol}_{timestamp}_enhanced"
        analysis_uuid = str(uuid.uuid4())

        # Set up logging
        log_file_path = self.setup_logging(analysis_id)

        try:
            self.stdout.write(self.style.SUCCESS(f"Starting ENHANCED analytics for {symbol}"))
            self.stdout.write(f"Analysis ID: {analysis_id}")
            self.stdout.write(f"Internal UUID: {analysis_uuid}")
            self.stdout.write(f"Log file: {log_file_path}")
            self.stdout.write(f"Confidence threshold: {confidence_threshold}")

            logger.info(f"=== ENHANCED ANALYTICS START: {symbol} ===")
            logger.info(f"Analysis ID: {analysis_uuid}")
            logger.info(f"Horizon: {horizon}")
            logger.info(f"Confidence threshold: {confidence_threshold}")

            # Step 1: Pre-analysis assessment
            self.stdout.write("Step 1: Assessing data availability and analysis readiness...")
            readiness = data_verification_service.assess_analysis_readiness(symbol)
            
            self.stdout.write(f"Current confidence: {readiness.overall_confidence:.2f}")
            self.stdout.write(f"Available indicators: {len(readiness.available_indicators)}")
            self.stdout.write(f"Missing indicators: {len(readiness.missing_indicators)}")
            
            if detailed:
                self.stdout.write("\nDETAILED READINESS ASSESSMENT:")
                self.stdout.write(f"Data quality: {readiness.data_availability.quality_level.value}")
                self.stdout.write(f"Total days: {readiness.data_availability.total_days}")
                self.stdout.write(f"Gap percentage: {readiness.data_availability.gap_percentage:.1f}%")
                self.stdout.write(f"Available: {', '.join(readiness.available_indicators)}")
                if readiness.missing_indicators:
                    self.stdout.write(f"Missing: {', '.join(readiness.missing_indicators)}")
                if readiness.adaptable_indicators:
                    self.stdout.write(f"Adaptable: {', '.join(readiness.adaptable_indicators)}")

            # Step 2: Enhanced backfill if needed
            if not skip_backfill and readiness.overall_confidence < confidence_threshold:
                self.stdout.write("\nStep 2: Running enhanced multi-source backfill...")
                backfill_result = enhanced_backfill_service.enhanced_backfill_concurrent(
                    symbol=symbol,
                    required_years=required_years,
                    min_acceptable_days=20
                )
                
                if backfill_result['success']:
                    self.stdout.write(self.style.SUCCESS(
                        f"Enhanced backfill successful: {backfill_result['stock_backfilled']} days from {backfill_result['best_source']}"
                    ))
                    self.stdout.write(f"Data quality improved from {backfill_result['data_quality_before']['quality_level']} to {backfill_result['data_quality_after']['quality_level']}")
                else:
                    self.stdout.write(self.style.WARNING(
                        f"⚠ Enhanced backfill had issues: {', '.join(backfill_result['sources_attempted'])}"
                    ))
            else:
                self.stdout.write("Step 2: Skipping backfill (confidence sufficient or explicitly skipped)")

            # Step 3: Run enhanced analysis
            self.stdout.write("\nStep 3: Running enhanced technical analysis...")
            
            analysis_result = enhanced_ta_engine.analyze_with_confidence(
                symbol=symbol,
                horizon=horizon,
                include_sentiment=include_sentiment,
                skip_backfill=skip_backfill
            )

            if 'error' in analysis_result:
                self.stdout.write(self.style.ERROR(f"Analysis failed: {analysis_result['error']}"))
                return

            # Step 4: Display results
            self.stdout.write("\nStep 4: Analysis Results")
            self.display_enhanced_results(analysis_result, detailed)
            
            # Step 5: Export report if requested
            if export_report:
                self.export_analysis_report(analysis_result, export_report)
                self.stdout.write(f"Detailed report exported to: {export_report}")

            # Final status
            final_score = analysis_result['final_score']
            confidence = analysis_result['confidence_score']
            
            if confidence >= confidence_threshold:
                status_style = self.style.SUCCESS
                status = "HIGH CONFIDENCE"
            elif confidence >= 0.4:
                status_style = self.style.WARNING  
                status = "MEDIUM CONFIDENCE"
            else:
                status_style = self.style.ERROR
                status = "LOW CONFIDENCE"

            self.stdout.write("\n" + "="*60)
            self.stdout.write(status_style(f"ENHANCED ANALYSIS COMPLETE: {symbol}"))
            self.stdout.write(f"Final Score: {final_score}/10")
            self.stdout.write(status_style(f"Confidence: {confidence:.2f} ({status})"))
            self.stdout.write(f"Duration: {analysis_result.get('analysis_duration', 0):.2f}s")
            self.stdout.write("="*60)

        except Exception as e:
            error_msg = f"Enhanced analysis failed for {symbol}: {str(e)}"
            logger.error(error_msg)
            self.stdout.write(self.style.ERROR(error_msg))
            raise CommandError(error_msg)

    def display_enhanced_results(self, analysis_result: Dict[str, Any], detailed: bool = False):
        """Display enhanced analysis results."""
        
        # Basic results
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(f"ENHANCED ANALYSIS RESULTS FOR {analysis_result['symbol']}")
        self.stdout.write(f"{'='*60}")
        
        self.stdout.write(f"Final Score: {analysis_result['final_score']}/10")
        self.stdout.write(f"Confidence Score: {analysis_result['confidence_score']:.3f}")
        self.stdout.write(f"Raw Composite: {analysis_result['raw_composite']:.6f}")
        
        # Data quality info
        data_quality = analysis_result['data_quality']
        self.stdout.write(f"\nData Quality:")
        self.stdout.write(f"  Total Days: {data_quality['total_days']}")
        self.stdout.write(f"  Quality Level: {data_quality['quality_level']}")
        self.stdout.write(f"  Overall Confidence: {data_quality['overall_confidence']:.3f}")

        if detailed:
            # Detailed indicator breakdown
            self.stdout.write(f"\nINDICATOR BREAKDOWN:")
            indicators = analysis_result['indicators']
            
            for indicator_name, indicator_data in indicators.items():
                status_flags = []
                if indicator_data['adapted']:
                    status_flags.append('ADAPTED')
                if indicator_data['fallback_used']:
                    status_flags.append('FALLBACK')
                
                status_str = f" ({', '.join(status_flags)})" if status_flags else ""
                
                self.stdout.write(
                    f"  {indicator_name:12s}: "
                    f"score={indicator_data['score']:.3f}, "
                    f"confidence={indicator_data['confidence']:.2f}, "
                    f"data_points={indicator_data['data_points_used']}"
                    f"{status_str}"
                )

            # Recommendations
            if analysis_result.get('recommendations'):
                self.stdout.write(f"\nRECOMMENDATIONS:")
                for i, rec in enumerate(analysis_result['recommendations'], 1):
                    self.stdout.write(f"  {i}. {rec}")

    def export_analysis_report(self, analysis_result: Dict[str, Any], file_path: str):
        """Export detailed analysis report to JSON file."""
        import json
        
        try:
            # Add metadata for export
            export_data = {
                'export_timestamp': timezone.now().isoformat(),
                'analysis_type': 'enhanced_technical_analysis',
                'version': '2.0',
                'analysis_result': analysis_result
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to export report: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Export failed: {str(e)}"))

    def setup_logging(self, analysis_id: str) -> str:
        """Set up detailed logging for the analysis."""
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create sanitized log filename
        safe_analysis_id = sanitize_cache_key(analysis_id)
        log_file_path = os.path.join(log_dir, f"{safe_analysis_id}.txt")

        # Configure file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Add handler to relevant loggers
        for logger_name in [
            "Analytics.management.commands.analyze_stock_enhanced",
            "Analytics.engine.enhanced_ta_engine", 
            "Analytics.engine.adaptive_indicators",
            "Data.services.multi_source_fetcher",
            "Data.services.enhanced_backfill",
            "Data.services.data_verification"
        ]:
            logger_obj = logging.getLogger(logger_name)
            logger_obj.addHandler(file_handler)
            logger_obj.setLevel(logging.INFO)

        return log_file_path