"""
Analytics Orchestration Command
Coordinates data validation, backfill, and technical analysis execution.

Usage:
    python manage.py analyze_stock --symbol AAPL
"""

import logging
import os
import uuid
from typing import Any, Dict

from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.utils import timezone

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Data.models import Stock
from Data.repo.price_reader import PriceReader
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Analytics orchestration command for technical analysis pipeline.
    Implements the complete workflow: validation → backfill → analytics → verification.
    """

    help = "Run technical analysis for a single stock symbol with data validation and backfill"

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
            "--max-backfill-attempts", type=int, default=3, help="Maximum backfill attempts (default: 3)"
        )
        parser.add_argument(
            "--required-years", type=int, default=2, help="Years of historical data required (default: 2)"
        )
        parser.add_argument(
            "--skip-backfill", action="store_true", help="Skip data backfill and proceed with existing data"
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
            "--fast-mode", action="store_true", 
            help="Fast analysis mode: skip backfill, use cached data, 6-month lookback (target <30s)"
        )

    def handle(self, *args, **options):
        """Main command handler implementing the orchestration workflow."""
        symbol = options["symbol"].upper()
        horizon = options["horizon"]
        max_attempts = options["max_backfill_attempts"]
        required_years = options["required_years"]
        skip_backfill = options["skip_backfill"]
        fast_mode = options["fast_mode"]
        
        # Fast mode overrides
        if fast_mode:
            skip_backfill = True
            required_years = 1  # Reduce to 6 months for speed
            self.stdout.write(
                self.style.WARNING('FAST MODE: Skipping backfill, using 1-year lookback for speed')
            )

        # Generate timestamp-based analysis ID for logging
        timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
        analysis_id = f"{symbol}_{timestamp}"
        analysis_uuid = str(uuid.uuid4())  # Keep UUID for internal tracking

        # Set up logging
        log_file_path = self.setup_logging(analysis_id)

        try:
            self.stdout.write(self.style.SUCCESS(f"Starting analytics for {symbol}"))
            self.stdout.write(f"Analysis ID: {analysis_id}")
            self.stdout.write(f"Internal UUID: {analysis_uuid}")
            self.stdout.write(f"Log file: {log_file_path}")

            logger.info(f"=== ANALYTICS START: {symbol} ===")
            logger.info(f"Analysis ID: {analysis_uuid}")
            logger.info(f"Horizon: {horizon}")
            logger.info(f"Required years: {required_years}")
            logger.info(f"Max backfill attempts: {max_attempts}")

            # Step 1: Engine Guard - Ensure PostgreSQL
            self.stdout.write("Step 1: Validating database engine...")
            self.ensure_postgresql_engine()
            logger.info("PostgreSQL engine confirmed")

            # Step 2: Data Coverage Check
            self.stdout.write("Step 2: Checking 2-year data coverage...")
            coverage_result = self.check_data_coverage(symbol, required_years)
            self.report_data_coverage(symbol, coverage_result)

            # Step 3: Backfill if needed (up to 3 attempts)
            if not skip_backfill:
                if self.needs_backfill(coverage_result):
                    self.stdout.write("Step 3: Backfilling missing data...")
                    backfill_result = self.perform_backfill(symbol, required_years, max_attempts)
                    logger.info(f"Backfill result: {backfill_result}")
                else:
                    self.stdout.write("Step 3: Data coverage adequate, skipping backfill")
                    logger.info("Skipping backfill - adequate data coverage")
            else:
                self.stdout.write("Step 3: Skipping backfill per user request")
                logger.info("Skipping backfill per --skip-backfill flag")

            # Step 4: Verify Final Data Coverage
            self.stdout.write("Step 4: Verifying final data coverage...")
            final_coverage = self.check_data_coverage(symbol, required_years)
            self.validate_final_coverage(final_coverage, symbol)

            # Step 5: Run Analytics Engine
            self.stdout.write("Step 5: Running technical analysis...")
            analysis_result = self.run_analytics(symbol, horizon, fast_mode)
            logger.info(f"Analysis complete: {analysis_result['score_0_10']}/10")

            # Step 6: Print Verification Results
            self.stdout.write("Step 6: Generating verification output...")
            self.print_verification_results(symbol, analysis_result)

            # Success summary
            self.stdout.write(self.style.SUCCESS(f"[SUCCESS] Analytics completed successfully for {symbol}"))
            self.stdout.write(f'Final Score: {analysis_result["score_0_10"]}/10')
            self.stdout.write(f"Log file: {log_file_path}")

            logger.info(f"=== ANALYTICS COMPLETE: {symbol} ===")
            logger.info(f"Final score: {analysis_result['score_0_10']}/10")

        except Exception as e:
            error_msg = f"Analytics failed for {symbol}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.stdout.write(self.style.ERROR(f"[ERROR] {error_msg}"))
            self.stdout.write(f"Log file: {log_file_path}")
            raise CommandError(error_msg)

    def setup_logging(self, analysis_id: str) -> str:
        """Set up dedicated log file for this analysis run."""
        # Ensure logs directory exists
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Create log file path with stock_timestamp format
        log_filename = f"{analysis_id}.txt"
        log_file_path = os.path.join(logs_dir, log_filename)

        # Set up file handler for this specific analysis
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler to relevant loggers
        loggers_to_configure = [
            "Analytics.engine.ta_engine",
            "Data.repo.price_reader",
            "Data.repo.analytics_writer",
            "Data.services.yahoo_finance",
            __name__,
        ]

        for logger_name in loggers_to_configure:
            target_logger = logging.getLogger(logger_name)
            target_logger.addHandler(file_handler)
            target_logger.setLevel(logging.INFO)

        return log_file_path

    def ensure_postgresql_engine(self):
        """Ensure PostgreSQL engine is configured."""
        engine = connection.vendor

        if engine == "sqlite":
            raise CommandError(
                "[ERROR] SQLite database detected. This analytics system requires PostgreSQL. "
                "Please configure PostgreSQL database in settings."
            )
        elif engine != "postgresql":
            self.stdout.write(
                self.style.WARNING(f"[WARNING]  Unexpected database engine '{engine}'. Expected 'postgresql'.")
            )

        logger.info(f"Database engine verified: {engine}")

    def report_data_coverage(self, symbol: str, coverage: Dict[str, Any]):
        """Report detailed data coverage statistics."""
        try:
            self.stdout.write("\n" + "=" * 50)
            self.stdout.write("DATA COVERAGE REPORT")
            self.stdout.write("=" * 50)

            # Stock data coverage
            stock_info = coverage["stock"]
            if stock_info["has_data"]:
                days_available = (stock_info["latest_date"] - stock_info["earliest_date"]).days + 1
                expected_trading_days = int(days_available * 0.714)  # ~252/365 ratio for display period

                # Get actual trading day count from database
                try:
                    from Data.models import Stock

                    stock = Stock.objects.get(symbol=symbol)
                    actual_trading_days = stock.prices.filter(
                        date__gte=stock_info["earliest_date"], date__lte=stock_info["latest_date"]
                    ).count()
                except Stock.DoesNotExist:
                    actual_trading_days = 0

                coverage_pct = (actual_trading_days / expected_trading_days * 100) if expected_trading_days > 0 else 0

                self.stdout.write(f"Stock ({symbol}):")
                self.stdout.write(f"  Date Range: {stock_info['earliest_date']} to {stock_info['latest_date']}")
                self.stdout.write(f"  Calendar Days: {days_available}")
                self.stdout.write(f"  Trading Days Available: {actual_trading_days}")
                self.stdout.write(f"  Expected Trading Days: {expected_trading_days}")
                self.stdout.write(f"  Coverage: {coverage_pct:.1f}%")

                logger.info(f"Stock data coverage: {actual_trading_days}/{expected_trading_days} ({coverage_pct:.1f}%)")
            else:
                self.stdout.write(f"Stock ({symbol}): No data available")
                logger.warning(f"No stock data available for {symbol}")

            # Sector data coverage
            sector_info = coverage["sector"]
            if sector_info["has_data"]:
                sector_days = (sector_info["latest_date"] - sector_info["earliest_date"]).days + 1
                sector_expected = int(sector_days * 0.714)

                # Get actual sector trading day count
                try:
                    sector_trading_days = (
                        stock.sector_id.prices.filter(
                            date__gte=sector_info["earliest_date"], date__lte=sector_info["latest_date"]
                        ).count()
                        if stock.sector_id
                        else 0
                    )
                except Exception:
                    sector_trading_days = 0

                sector_coverage_pct = (sector_trading_days / sector_expected * 100) if sector_expected > 0 else 0

                self.stdout.write("\nSector Data:")
                self.stdout.write(f"  Date Range: {sector_info['earliest_date']} to {sector_info['latest_date']}")
                self.stdout.write(f"  Trading Days Available: {sector_trading_days}")
                self.stdout.write(f"  Coverage: {sector_coverage_pct:.1f}%")

                # Compare stock vs sector coverage
                if stock_info["has_data"]:
                    coverage_diff = abs(actual_trading_days - sector_trading_days)
                    coverage_diff_pct = (
                        (coverage_diff / max(actual_trading_days, sector_trading_days) * 100)
                        if max(actual_trading_days, sector_trading_days) > 0
                        else 0
                    )
                    self.stdout.write(f"  Difference from Stock: {coverage_diff} days ({coverage_diff_pct:.1f}%)")
                    logger.info(f"Sector vs Stock coverage difference: {coverage_diff} days ({coverage_diff_pct:.1f}%)")
            else:
                self.stdout.write("\nSector Data: Not available")
                logger.warning("No sector data available")

            # Industry data coverage
            industry_info = coverage["industry"]
            if industry_info["has_data"]:
                industry_days = (industry_info["latest_date"] - industry_info["earliest_date"]).days + 1
                industry_expected = int(industry_days * 0.714)

                # Get actual industry trading day count
                try:
                    industry_trading_days = (
                        stock.industry_id.prices.filter(
                            date__gte=industry_info["earliest_date"], date__lte=industry_info["latest_date"]
                        ).count()
                        if stock.industry_id
                        else 0
                    )
                except Exception:
                    industry_trading_days = 0

                industry_coverage_pct = (
                    (industry_trading_days / industry_expected * 100) if industry_expected > 0 else 0
                )

                self.stdout.write("\nIndustry Data:")
                self.stdout.write(f"  Date Range: {industry_info['earliest_date']} to {industry_info['latest_date']}")
                self.stdout.write(f"  Trading Days Available: {industry_trading_days}")
                self.stdout.write(f"  Coverage: {industry_coverage_pct:.1f}%")

                # Compare stock vs industry coverage
                if stock_info["has_data"]:
                    coverage_diff = abs(actual_trading_days - industry_trading_days)
                    coverage_diff_pct = (
                        (coverage_diff / max(actual_trading_days, industry_trading_days) * 100)
                        if max(actual_trading_days, industry_trading_days) > 0
                        else 0
                    )
                    self.stdout.write(f"  Difference from Stock: {coverage_diff} days ({coverage_diff_pct:.1f}%)")
                    logger.info(
                        f"Industry vs Stock coverage difference: {coverage_diff} days ({coverage_diff_pct:.1f}%)"
                    )
            else:
                self.stdout.write("\nIndustry Data: Not available")
                logger.warning("No industry data available")

            self.stdout.write("=" * 50 + "\n")

        except Exception as e:
            logger.error(f"Error reporting data coverage: {str(e)}")
            self.stdout.write(f"Error generating coverage report: {str(e)}")

    def check_data_coverage(self, symbol: str, required_years: int) -> Dict[str, Any]:
        """Check 3-year data coverage for stock, sector, and industry."""
        try:
            price_reader = PriceReader()
            coverage = price_reader.check_data_coverage(symbol, required_years)

            # Log coverage details
            logger.info(f"Data coverage check for {symbol}:")
            for data_type, info in coverage.items():
                logger.info(f"  {data_type}: has_data={info['has_data']}, gaps={info['gap_count']}")
                if info["earliest_date"]:
                    logger.info(f"    Date range: {info['earliest_date']} to {info['latest_date']}")

            return coverage

        except Exception as e:
            logger.error(f"Error checking data coverage: {str(e)}")
            raise

    def needs_backfill(self, coverage: Dict[str, Any]) -> bool:
        """Determine if backfill is needed based on coverage."""
        stock_needs = not coverage["stock"]["has_data"] or coverage["stock"]["gap_count"] > 50
        sector_needs = coverage["sector"]["gap_count"] > 50 if coverage["sector"]["has_data"] else False
        industry_needs = coverage["industry"]["gap_count"] > 50 if coverage["industry"]["has_data"] else False

        needs_backfill = stock_needs or sector_needs or industry_needs

        logger.info(f"Backfill assessment: stock={stock_needs}, sector={sector_needs}, industry={industry_needs}")
        return needs_backfill

    def perform_backfill(self, symbol: str, required_years: int, max_attempts: int) -> Dict[str, Any]:
        """Perform backfill with retry logic."""
        try:
            backfill_result = yahoo_finance_service.backfill_eod_gaps_concurrent(
                symbol=symbol, required_years=required_years, max_attempts=max_attempts
            )

            if backfill_result["success"]:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"[SUCCESS] Backfill successful after {backfill_result['attempts_used']} attempts"
                    )
                )
                self.stdout.write(f"  Stock records: {backfill_result['stock_backfilled']}")
                self.stdout.write(f"  Sector records: {backfill_result['sector_backfilled']}")
                self.stdout.write(f"  Industry records: {backfill_result['industry_backfilled']}")
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f"[WARNING]  Backfill completed with issues after {backfill_result['attempts_used']} attempts"
                    )
                )
                for error in backfill_result["errors"]:
                    self.stdout.write(f"  Error: {error}")
                    logger.warning(f"Backfill error: {error}")

            return backfill_result

        except Exception as e:
            logger.error(f"Backfill failed: {str(e)}")
            raise

    def validate_final_coverage(self, coverage: Dict[str, Any], symbol: str):
        """Validate that final coverage is adequate for analysis."""
        stock_adequate = coverage["stock"]["has_data"] and coverage["stock"]["gap_count"] <= 800

        if not stock_adequate:
            raise CommandError(
                f"[ERROR] Insufficient stock data for {symbol} after backfill attempts. "
                f"Has data: {coverage['stock']['has_data']}, Gaps: {coverage['stock']['gap_count']}"
            )

        # Warn about sector/industry coverage but don't fail
        if not coverage["sector"]["has_data"]:
            self.stdout.write(self.style.WARNING(f"[WARNING]  No sector data available for {symbol}"))

        if not coverage["industry"]["has_data"]:
            self.stdout.write(self.style.WARNING(f"[WARNING]  No industry data available for {symbol}"))

        logger.info(f"Final coverage validation passed for {symbol}")

    def run_analytics(self, symbol: str, horizon: str, fast_mode: bool = False) -> Dict[str, Any]:
        """Run the technical analysis engine."""
        try:
            engine = TechnicalAnalysisEngine()
            result = engine.analyze_stock(symbol, horizon=horizon, fast_mode=fast_mode)

            logger.info(f"Technical analysis completed for {symbol}")
            logger.info(f"Composite score: {result['score_0_10']}/10")

            # Log individual indicator scores
            for indicator, values in result["components"].items():
                logger.info(f"  {indicator}: raw={values['raw']}, score={values['score']:.3f}")

            return result

        except Exception as e:
            logger.error(f"Analytics engine failed: {str(e)}")
            raise

    def print_verification_results(self, symbol: str, analysis_result: Dict[str, Any]):
        """Print verification output as specified."""
        try:
            self.stdout.write("\n" + "=" * 60)
            self.stdout.write("VERIFICATION RESULTS")
            self.stdout.write("=" * 60)

            # Print first/last 5 rows of ANALYTICS_RESULTS
            self.print_analytics_results_sample(symbol)

            # Print first/last 5 rows of price data
            self.print_price_data_sample(symbol)

            # Print analysis summary
            self.stdout.write("\nANALYSIS SUMMARY:")
            self.stdout.write(f"Symbol: {symbol}")
            self.stdout.write(f"Analysis Date: {analysis_result['analysis_date']}")
            self.stdout.write(f"Horizon: {analysis_result['horizon']}")
            self.stdout.write(f"Composite Raw Score: {analysis_result['composite_raw']:.6f}")
            self.stdout.write(f"Final Score (0-10): {analysis_result['score_0_10']}")

            # Print indicator breakdown
            self.stdout.write("\nINDICATOR BREAKDOWN:")
            for indicator, weighted_score in analysis_result["weighted_scores"].items():
                component = analysis_result["components"][indicator.replace("w_", "")]
                self.stdout.write(f"  {indicator}: {float(weighted_score):.6f} (score: {component['score']:.3f})")

        except Exception as e:
            logger.error(f"Error printing verification results: {str(e)}")
            self.stdout.write(self.style.ERROR(f"Error generating verification output: {str(e)}"))

    def print_analytics_results_sample(self, symbol: str):
        """Print first/last 5 rows of ANALYTICS_RESULTS."""
        try:
            stock = Stock.objects.get(symbol=symbol)
            results = stock.analytics_results.order_by("as_of")

            self.stdout.write(f"\nANALYTICS_RESULTS for {symbol}:")

            if results.exists():
                self.stdout.write("First 5 rows:")
                for result in results[:5]:
                    self.stdout.write(f"  {result.as_of}: Score {result.score_0_10}/10 (Raw: {result.composite_raw})")

                if results.count() > 5:
                    self.stdout.write("Last 5 rows:")
                    for result in results.reverse()[:5]:
                        self.stdout.write(
                            f"  {result.as_of}: Score {result.score_0_10}/10 (Raw: {result.composite_raw})"
                        )
            else:
                self.stdout.write("  No analytics results found")

        except Stock.DoesNotExist:
            self.stdout.write(f"  Stock {symbol} not found")
        except Exception as e:
            logger.error(f"Error printing analytics results: {str(e)}")
            self.stdout.write(f"  Error retrieving analytics results: {str(e)}")

    def print_price_data_sample(self, symbol: str):
        """Print first/last 5 rows of stock price data."""
        try:
            stock = Stock.objects.get(symbol=symbol)
            prices = stock.prices.order_by("date")

            self.stdout.write(f"\nSTOCK PRICE DATA for {symbol}:")

            if prices.exists():
                self.stdout.write("First 5 rows:")
                for price in prices[:5]:
                    self.stdout.write(f"  {price.date}: ${price.close} (Vol: {price.volume:,})")

                if prices.count() > 5:
                    self.stdout.write("Last 5 rows:")
                    for price in prices.reverse()[:5]:
                        self.stdout.write(f"  {price.date}: ${price.close} (Vol: {price.volume:,})")

                # Print sector/industry data if available
                if stock.sector_id:
                    sector_prices = stock.sector_id.prices.order_by("date")
                    if sector_prices.exists():
                        self.stdout.write(f"\nSECTOR COMPOSITE DATA ({stock.sector_id.sectorName}):")
                        self.stdout.write("First 5 rows:")
                        for sp in sector_prices[:5]:
                            self.stdout.write(f"  {sp.date}: Index {sp.close_index} ({sp.constituents_count} stocks)")

                if stock.industry_id:
                    industry_prices = stock.industry_id.prices.order_by("date")
                    if industry_prices.exists():
                        self.stdout.write(f"\nINDUSTRY COMPOSITE DATA ({stock.industry_id.industryName}):")
                        self.stdout.write("First 5 rows:")
                        for ip in industry_prices[:5]:
                            self.stdout.write(f"  {ip.date}: Index {ip.close_index} ({ip.constituents_count} stocks)")
            else:
                self.stdout.write("  No price data found")

        except Stock.DoesNotExist:
            self.stdout.write(f"  Stock {symbol} not found")
        except Exception as e:
            logger.error(f"Error printing price data: {str(e)}")
            self.stdout.write(f"  Error retrieving price data: {str(e)}")
