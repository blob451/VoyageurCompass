"""
Technical analysis engine with 12-indicator framework and normalised scoring.
"""

import logging
import statistics
import time
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from functools import wraps

from django.utils import timezone
from django.core.cache import cache

from Analytics.services.sentiment_analyzer import get_sentiment_analyzer
from Analytics.services.universal_predictor import get_universal_lstm_service
from Data.repo.analytics_writer import AnalyticsWriter
from Data.repo.price_reader import (
    IndustryPriceData,
    PriceData,
    PriceReader,
    SectorPriceData,
)
from Data.services.yahoo_finance import yahoo_finance_service

logger = logging.getLogger(__name__)


def cache_indicator(cache_ttl: int = 300):
    """
    Decorator to cache expensive technical indicator calculations.
    
    Args:
        cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key based on function name, symbol, and price data hash
            symbol = getattr(self, '_current_symbol', 'unknown')
            
            # Price data hash generation for caching
            price_data_str = ""
            if args and hasattr(args[0], '__iter__'):
                # First argument is usually price data
                try:
                    price_data_str = str([(p.date, float(p.close), int(p.volume)) for p in args[0][-50:]])  # Last 50 days
                except (AttributeError, IndexError, TypeError):
                    price_data_str = str(args)[:200]  # Fallback to string representation
            
            cache_key = f"ta_indicator:{func.__name__}:{symbol}:{hashlib.md5(price_data_str.encode()).hexdigest()[:12]}"
            
            # Cache retrieval attempt
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for {func.__name__} on {symbol}")
                return cached_result
            
            # Calculate and cache result
            result = func(self, *args, **kwargs)
            if result:  # Only cache successful calculations
                cache.set(cache_key, result, cache_ttl)
                logger.debug(f"Cached {func.__name__} result for {symbol}")
            
            return result
        return wrapper
    return decorator


class IndicatorResult(NamedTuple):
    """Structured container for individual indicator analysis results."""

    raw: Any
    score: float
    weight: float
    weighted_score: float


class TechnicalAnalysisEngine:
    """Comprehensive technical analysis engine with weighted 12-indicator framework."""

    # Indicator weights with sentiment and LSTM integration
    WEIGHTS = {
        "sma50vs200": 0.12,  # 15% * 0.8
        "pricevs50": 0.08,  # 10% * 0.8
        "rsi14": 0.08,  # 10% * 0.8
        "macd12269": 0.08,  # 10% * 0.8
        "bbpos20": 0.08,  # 10% * 0.8
        "bbwidth20": 0.04,  # 5% * 0.8
        "volsurge": 0.08,  # 10% * 0.8
        "obv20": 0.04,  # 5% * 0.8
        "rel1y": 0.04,  # 5% * 0.8
        "rel2y": 0.04,  # 5% * 0.8
        "candlerev": 0.064,  # 8% * 0.8
        "srcontext": 0.056,  # 7% * 0.8
        "sentiment": 0.10,  # 10% for sentiment
        "prediction": 0.10,  # 10% for LSTM predictions
    }

    # Human-readable indicator display names
    INDICATOR_NAMES = {
        "sma50vs200": "Moving Average Crossover",
        "pricevs50": "Price vs 50-Day Average",
        "rsi14": "Relative Strength Index",
        "macd12269": "MACD Histogram",
        "bbpos20": "Bollinger Position",
        "bbwidth20": "Bollinger Bandwidth",
        "volsurge": "Volume Surge",
        "obv20": "On-Balance Volume",
        "rel1y": "Relative Strength 1Y",
        "rel2y": "Relative Strength 2Y",
        "candlerev": "Candlestick Pattern",  # Special case: will be modified dynamically
        "srcontext": "Support/Resistance",
        "sentiment": "News Sentiment Analysis",
        "prediction": "LSTM Price Prediction",
    }

    def __init__(self):
        """Initialise technical analysis engine with data repositories."""
        self.price_reader = PriceReader()
        self.analytics_writer = AnalyticsWriter()
        self._current_symbol = None  # Track current symbol for caching

    def get_indicator_display_name(self, indicator_code: str, indicator_result=None) -> str:
        """Retrieve human-readable display name for indicator code.

        Args:
            indicator_code: The internal indicator code (e.g., 'sma50vs200')
            indicator_result: The indicator result object (needed for CANDLEREV pattern)

        Returns:
            Human-readable indicator name
        """
        if indicator_code == "candlerev" and indicator_result and hasattr(indicator_result, "raw"):
            # Special case for candlestick patterns
            pattern = indicator_result.raw.get("pattern", "unknown") if indicator_result.raw else "unknown"
            return f"Pattern: {pattern.replace('_', ' ').title()}"

        return self.INDICATOR_NAMES.get(indicator_code, indicator_code.upper())

    def analyze_stock(
        self,
        symbol: str,
        analysis_date: Optional[datetime] = None,
        horizon: str = "blend",
        user=None,
        logger_instance=None,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform complete technical analysis for a stock.

        Args:
            symbol: Stock ticker symbol
            analysis_date: Optional analysis date (defaults to now)
            horizon: Analysis horizon ('short', 'medium', 'long', 'blend')
            user: User instance who initiated this analysis
            logger_instance: AnalysisLogger instance for web-based analysis logging
            fast_mode: Enable fast mode (use cached data, shorter lookbacks)

        Returns:
            Dict containing all analysis results and composite score
        """
        try:
            # Set current symbol for caching
            self._current_symbol = symbol
            
            logger.info(f"[TA ENGINE] Starting technical analysis for {symbol}")
            analysis_start_time = time.time()

            if analysis_date is None:
                analysis_date = timezone.now()

            # Log analysis start if web logger is provided
            if logger_instance:
                logger_instance.log_analysis_start(analysis_date, horizon)

            logger.debug(f"Getting stock data for {symbol}")
            # Required data retrieval (3-year comprehensive analysis)
            stock_prices = self._get_stock_data(symbol, analysis_date, logger_instance)
            logger.debug(f"Stock data retrieved: {len(stock_prices)} records")

            logger.debug(f"Getting sector/industry data for {symbol}")
            sector_prices, industry_prices = self._get_sector_industry_data(symbol, analysis_date)
            logger.debug(
                f"Sector data: {len(sector_prices)} records, Industry data: {len(industry_prices)} records"
            )

            if not stock_prices:
                # Auto-sync failure verification
                auto_sync_attempted = hasattr(self, "_last_auto_sync") and self._last_auto_sync
                if auto_sync_attempted:
                    error_msg = (f"No price data available for {symbol}. Auto-sync failed due to data provider issues. "
                                f"Please try again later or use the sync button to manually refresh data.")
                else:
                    error_msg = (f"No price data available for {symbol}. Stock may not exist or data provider is unavailable. "
                                f"Please verify the symbol and try again.")
                
                logger.error(f"Data unavailable for {symbol} - auto-sync attempted: {auto_sync_attempted}")
                if logger_instance:
                    logger_instance.log_analysis_error(error_msg)
                raise ValueError(error_msg)

            # Log data retrieval
            if logger_instance:
                auto_sync = hasattr(self, "_last_auto_sync") and self._last_auto_sync
                logger_instance.log_data_retrieval(len(stock_prices), auto_sync)
                if hasattr(self, "_last_auto_sync"):
                    delattr(self, "_last_auto_sync")

            logger.debug("Starting calculation of 14 indicators (parallelized)")
            # Calculate all 14 indicators with parallel execution
            indicators = self._calculate_indicators_parallel(
                symbol, stock_prices, sector_prices, industry_prices, logger_instance, fast_mode
            )

            # Calculate weighted scores and composite with dynamic weight reallocation
            weighted_scores = {}
            components = {}
            composite_raw = Decimal("0")

            # CRITICAL FIX: Dynamic weight reallocation when indicators are missing
            available_indicators = {k: v for k, v in indicators.items() if v is not None}
            missing_indicators = [k for k, v in indicators.items() if v is None]

            # Calculate total weight of missing indicators
            missing_weight = sum(self.WEIGHTS[indicator] for indicator in missing_indicators)
            available_weight = sum(self.WEIGHTS[indicator] for indicator in available_indicators.keys())

            logger.debug(
                f"Weight allocation - Available: {len(available_indicators)}/{len(indicators)} indicators"
            )
            if missing_indicators:
                logger.warning(f"Missing indicators: {missing_indicators} (total weight: {missing_weight:.3f})")
                logger.warning(f"Redistributing {missing_weight:.3f} weight to available indicators")

            # Adjusted weight calculation for available indicators
            adjusted_weights = {}
            for indicator_name in indicators.keys():
                if indicator_name in available_indicators:
                    # Redistribute missing weight proportionally to available indicators
                    base_weight = self.WEIGHTS[indicator_name]
                    if available_weight > 0 and missing_weight > 0:
                        weight_boost = (base_weight / available_weight) * missing_weight
                        adjusted_weights[indicator_name] = base_weight + weight_boost
                    else:
                        adjusted_weights[indicator_name] = base_weight
                else:
                    # Missing indicators get zero weight instead of default 0.5 score
                    adjusted_weights[indicator_name] = 0.0

            # Verify total weight is still 1.0 (with some tolerance for floating point)
            total_weight = sum(adjusted_weights.values())
            logger.debug(f"Total adjusted weight: {total_weight:.6f} (should be 1.0)")

            for indicator_name, result in indicators.items():
                if result is None:
                    # Missing indicators contribute nothing to the composite score
                    score = 0.0  # Changed from 0.5 to 0.0
                    raw_value = None
                    weight = 0.0  # No weight for missing indicators
                else:
                    score = result.score
                    raw_value = result.raw
                    weight = adjusted_weights[indicator_name]

                weighted_score = Decimal(str(score)) * Decimal(str(weight))

                weighted_scores[f"w_{indicator_name}"] = weighted_score
                # Display name retrieval for API response
                display_name = self.get_indicator_display_name(indicator_name, result)

                components[indicator_name] = {
                    "raw": raw_value,
                    "score": score,
                    "description": display_name,
                    "weight_used": float(weight),  # Weight transparency inclusion
                    "original_weight": self.WEIGHTS[indicator_name],
                }
                composite_raw += weighted_score

            logger.debug("All indicators calculated successfully")

            # Final composite score (0-10, rounded)
            score_0_10 = round(float(composite_raw) * 10)
            logger.info(f"Composite score calculated: {score_0_10}/10 (raw: {composite_raw})")

            logger.debug("Storing results to database")
            # Store results in database
            analytics_result = self.analytics_writer.upsert_analytics_result(
                symbol=symbol,
                as_of=analysis_date,
                weighted_scores=weighted_scores,
                components=components,
                composite_raw=composite_raw,
                score_0_10=score_0_10,
                horizon=horizon,
                user=user,
            )
            logger.debug(f"Results stored to database successfully (ID: {analytics_result.id})")

            result = {
                "symbol": symbol,
                "analysis_date": analysis_date,
                "horizon": horizon,
                "indicators": indicators,
                "weighted_scores": weighted_scores,
                "components": components,
                "composite_raw": float(composite_raw),
                "score_0_10": score_0_10,
                "analytics_result_id": analytics_result.id,
            }

            logger.debug(f"Finalizing analysis for {symbol}")

            # Log analysis completion
            if logger_instance:
                logger_instance.log_analysis_complete(float(composite_raw), score_0_10)
                logger_instance.finalize()

            analysis_duration = time.time() - analysis_start_time
            logger.info(f"Analysis complete for {symbol}: {score_0_10}/10")
            logger.info(f"[TA ENGINE] Analysis complete for {symbol}: {score_0_10}/10 in {analysis_duration:.2f}s")
            return result

        except Exception as e:
            error_msg = f"Error analyzing {symbol}: {str(e)}"
            logger.error(f"EXCEPTION in analysis: {error_msg}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            logger.error(error_msg)
            if logger_instance:
                logger_instance.log_analysis_error(error_msg)
                logger_instance.finalize()
            raise

    def _get_stock_data(self, symbol: str, analysis_date: datetime, logger_instance=None) -> List[PriceData]:
        """Get 3 years of stock price data ending at analysis date."""
        start_date = (analysis_date - timedelta(days=3 * 365 + 30)).date()
        end_date = analysis_date.date()
        stock_prices = []

        try:
            # First attempt to get existing data
            stock_prices = self.price_reader.get_stock_prices(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            # Stock might not exist, continue to auto-sync attempt

        # If no data found, attempt to auto-sync
        if not stock_prices:
            logger.info(f"No existing data for {symbol}, attempting auto-sync")
            if self._auto_sync_stock_data(symbol):
                self._last_auto_sync = True  # Track that auto-sync was used

                # Retry after sync with multiple attempts
                for attempt in range(3):
                    try:
                        import time

                        if attempt > 0:
                            time.sleep(2)  # Wait 2 seconds between retries

                        stock_prices = self.price_reader.get_stock_prices(symbol, start_date, end_date)
                        if stock_prices:
                            logger.info(
                                f"Successfully retrieved data for {symbol} after auto-sync (attempt {attempt + 1})"
                            )
                            break
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            logger.error(
                                f"Error getting stock data after auto-sync for {symbol} (final attempt): {str(e)}"
                            )
                        else:
                            logger.warning(
                                f"Error getting stock data after auto-sync for {symbol} (attempt {attempt + 1}): {str(e)}"
                            )

        return stock_prices

    def _get_sector_industry_data(
        self, symbol: str, analysis_date: datetime
    ) -> Tuple[List[SectorPriceData], List[IndustryPriceData]]:
        """Get sector and industry composite data with optimized availability checks."""
        try:
            # Step 1: Get sector/industry keys with early validation
            sector_key, industry_key = self.price_reader.get_stock_sector_industry_keys(symbol)

            # Auto-populate missing sector/industry data
            auto_populate_needed = False
            if not sector_key and not industry_key:
                auto_populate_needed = True
                reason = "both sector and industry missing"
            elif not industry_key and sector_key:
                auto_populate_needed = True
                reason = "industry missing"
            
            if auto_populate_needed and symbol not in ["TEST", "INTEGRATION", "EMPTY_RESILIENCE"]:
                logger.info(f"Auto-populating {symbol} ({reason})")
                self._auto_populate_sector_industry(symbol)
                # Retry getting keys after auto-population
                sector_key, industry_key = self.price_reader.get_stock_sector_industry_keys(symbol)
                if not sector_key and not industry_key:
                    logger.debug(f"Auto-population failed for {symbol}, proceeding with stock-only analysis")
                    return [], []

            start_date = (analysis_date - timedelta(days=3 * 365 + 30)).date()
            end_date = analysis_date.date()

            sector_prices = []
            industry_prices = []

            # Step 2: Get sector prices with availability check
            if sector_key:
                try:
                    sector_prices = self.price_reader.get_sector_prices(sector_key, start_date, end_date)
                    if not sector_prices and symbol not in ["TEST", "INTEGRATION", "EMPTY_RESILIENCE"]:
                        logger.debug(f"No sector price data available for {symbol} (sector: {sector_key})")
                except Exception as e:
                    logger.warning(f"Error retrieving sector prices for {symbol}: {str(e)}")

            # Step 3: Get industry prices with availability check
            if industry_key:
                try:
                    industry_prices = self.price_reader.get_industry_prices(industry_key, start_date, end_date)
                    if not industry_prices and symbol not in ["TEST", "INTEGRATION", "EMPTY_RESILIENCE"]:
                        logger.debug(f"No industry price data available for {symbol} (industry: {industry_key})")
                except Exception as e:
                    logger.warning(f"Error retrieving industry prices for {symbol}: {str(e)}")

            # Log summary only for real stocks with missing data
            if symbol not in ["TEST", "INTEGRATION", "EMPTY_RESILIENCE"]:
                if not sector_prices and not industry_prices and (sector_key or industry_key):
                    logger.info(
                        f"No sector/industry comparative data available for {symbol} - using stock-only analysis"
                    )
                elif sector_prices or industry_prices:
                    logger.debug(
                        f"Retrieved sector/industry data for {symbol}: {len(sector_prices)} sector, {len(industry_prices)} industry records"
                    )

            return sector_prices, industry_prices

        except Exception as e:
            logger.warning(f"Error getting sector/industry data for {symbol}: {str(e)}")
            return [], []

    def _auto_sync_stock_data(self, symbol: str) -> bool:
        """
        Automatically sync stock data when it's missing from the database.
        Uses cross-platform timeout handling and direct yfinance fallback for reliability.

        Args:
            symbol: Stock ticker symbol to sync

        Returns:
            True if sync was successful, False otherwise
        """
        import platform
        import threading
        from contextlib import contextmanager
        
        @contextmanager
        def timeout_context(seconds):
            """Cross-platform timeout context manager."""
            if platform.system() == 'Windows' or not hasattr(__import__('signal'), 'SIGALRM'):
                # Threading-based timeout for Windows systems
                timer = None
                timed_out = [False]  # List-based modification for nested functions
                
                def timeout_handler():
                    timed_out[0] = True
                    # Windows operation interruption limitation
                    # Timeout tracking capability
                
                try:
                    if seconds and seconds > 0:
                        timer = threading.Timer(seconds, timeout_handler)
                        timer.start()
                    yield
                    if timed_out[0]:
                        raise TimeoutError(f"Operation timed out after {seconds} seconds")
                finally:
                    if timer:
                        timer.cancel()
            else:
                # Signal-based timeout for Unix systems
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {seconds} seconds")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

        try:
            logger.info(f"Auto-syncing data for {symbol}")

            # Import here to avoid circular imports
            from Data.models import Stock
            from Data.services.yahoo_finance import yahoo_finance_service

            # Service timeout attempt
            try:
                with timeout_context(30):  # 30 second timeout
                    # First, validate that Yahoo Finance has data for this symbol
                    test_result = yahoo_finance_service.get_stock_data(symbol, period="5d", sync_db=False)

                    if "error" in test_result:
                        logger.warning(f"Yahoo Finance service validation failed for {symbol}: {test_result['error']}")
                        raise Exception("Service validation failed")

                    # If validation passes, proceed with full sync
                    sync_result = yahoo_finance_service.get_stock_data(symbol, period="2y", sync_db=True)

                    if "error" not in sync_result:
                        logger.info(f"Successfully auto-synced data for {symbol} via service")

                        # Verify the stock was actually created
                        try:
                            Stock.objects.get(symbol=symbol.upper())
                            return True
                        except Stock.DoesNotExist:
                            logger.error(f"Auto-sync appeared successful but stock {symbol} not found in database")
                            raise Exception("Stock not found after sync")
                    else:
                        logger.warning(f"Service sync failed for {symbol}: {sync_result['error']}")
                        raise Exception("Service sync failed")

            except (TimeoutError, Exception) as e:
                logger.warning(f"Service method failed for {symbol}: {str(e)}, trying direct yfinance")
                
                # Fallback to direct yfinance
                try:
                    import yfinance as yf
                    from decimal import Decimal
                    from django.db import transaction

                    # Direct stock information and history retrieval
                    ticker = yf.Ticker(symbol)
                    
                    # Basic information retrieval with timeout
                    with timeout_context(15):
                        try:
                            info = ticker.info
                        except Exception:
                            # If info fails, use minimal defaults
                            info = {
                                'shortName': symbol,
                                'longName': symbol,
                                'currency': 'USD',
                                'exchange': 'Unknown',
                                'sector': '',
                                'industry': ''
                            }
                    
                    # Historical data retrieval with timeout
                    with timeout_context(15):
                        hist = ticker.history(period="2y")
                    
                    if hist.empty:
                        logger.error(f"No historical data available for {symbol}")
                        return False

                    # Stock and price record creation
                    with transaction.atomic():
                        # Stock creation or update
                        stock, created = Stock.objects.get_or_create(
                            symbol=symbol.upper(),
                            defaults={
                                'short_name': (info.get('shortName', symbol) or symbol)[:100],
                                'long_name': (info.get('longName', symbol) or symbol)[:255],
                                'currency': (info.get('currency', 'USD') or 'USD')[:10],
                                'exchange': (info.get('exchange', 'Unknown') or 'Unknown')[:50],
                                'sector': (info.get('sector', '') or '')[:100],
                                'industry': (info.get('industry', '') or '')[:100],
                                'market_cap': info.get('marketCap', 0) or 0,
                            }
                        )

                        # Update if not created
                        if not created:
                            stock.short_name = (info.get('shortName', stock.short_name) or stock.short_name)[:100]
                            stock.long_name = (info.get('longName', stock.long_name) or stock.long_name)[:255]
                            stock.currency = (info.get('currency', stock.currency) or stock.currency)[:10]
                            stock.exchange = (info.get('exchange', stock.exchange) or stock.exchange)[:50]
                            stock.sector = (info.get('sector', stock.sector) or stock.sector)[:100]
                            stock.industry = (info.get('industry', stock.industry) or stock.industry)[:100]
                            stock.market_cap = info.get('marketCap', stock.market_cap) or stock.market_cap
                            stock.save()

                        # Import StockPrice here to avoid circular imports
                        from Data.models import StockPrice

                        # Price record creation
                        prices_created = 0
                        for date, row in hist.iterrows():
                            try:
                                StockPrice.objects.update_or_create(
                                    stock=stock,
                                    date=date.date(),
                                    defaults={
                                        'open': Decimal(str(row['Open'])),
                                        'high': Decimal(str(row['High'])),
                                        'low': Decimal(str(row['Low'])),
                                        'close': Decimal(str(row['Close'])),
                                        'volume': int(row['Volume']),
                                    }
                                )
                                prices_created += 1
                            except Exception as price_e:
                                logger.warning(f"Error creating price record for {symbol} on {date}: {str(price_e)}")

                    logger.info(f"Successfully auto-synced {symbol} via direct yfinance: {prices_created} prices")
                    return True

                except Exception as direct_e:
                    logger.error(f"Direct yfinance fallback failed for {symbol}: {str(direct_e)}")
                    return False

        except Exception as e:
            logger.error(f"Error during auto-sync for {symbol}: {str(e)}")
            return False

    def _calculate_indicators_parallel(self, symbol: str, stock_prices: List, sector_prices: List, 
                                     industry_prices: List, logger_instance=None, fast_mode: bool = False) -> Dict[str, Any]:
        """
        Calculate all 14 indicators in parallel for improved performance.
        
        Groups indicators by dependency and execution characteristics:
        - Group 1: Independent technical indicators (can run fully parallel)
        - Group 2: Relative strength indicators (need sector/industry data)  
        - Group 3: External data indicators (sentiment, prediction)
        
        Args:
            symbol: Stock symbol
            stock_prices: Stock price data
            sector_prices: Sector price data
            industry_prices: Industry price data
            logger_instance: Optional web logger
            
        Returns:
            Dict of calculated indicators
        """
        indicators = {}
        start_time = time.time()
        
        logger.debug(f"Starting parallel indicator calculation for {symbol} (fast_mode: {fast_mode})")
        
        # Fast mode optimisations
        if fast_mode:
            # Fast mode: Use cached sentiment/prediction if available, otherwise skip
            external_indicators = []
            
            # Try to get cached sentiment (24 hour TTL)
            sentiment_cache_key = f"sentiment_cache_{symbol}_24h"
            cached_sentiment = cache.get(sentiment_cache_key)
            
            # Try to get cached prediction (6 hour TTL)
            prediction_cache_key = f"prediction_cache_{symbol}_6h"  
            cached_prediction = cache.get(prediction_cache_key)
            
            if cached_sentiment or cached_prediction:
                # Add cached indicators to external processing
                if cached_sentiment:
                    external_indicators.append(("sentiment", lambda s: cached_sentiment, (symbol,)))
                    logger.info(f"Fast mode: Using cached sentiment analysis for {symbol}")
                    
                if cached_prediction:
                    external_indicators.append(("prediction", lambda s: cached_prediction, (symbol,)))
                    logger.info(f"Fast mode: Using cached prediction for {symbol}")
                    
                if not cached_sentiment and not cached_prediction:
                    logger.info(f"Fast mode: No cached data available, skipping sentiment and prediction analysis for speed")
            else:
                logger.info(f"Fast mode: No cached sentiment/prediction available, skipping for speed")
        else:
            # Standard mode: Full sentiment and prediction analysis
            external_indicators = [
                ("sentiment", self._calculate_sentiment_analysis, (symbol,)),
                ("prediction", self._calculate_prediction_score, (symbol,)),
            ]
        
        # Group 1: Independent technical indicators (fully parallel)
        technical_indicators = [
            ("sma50vs200", self._calculate_sma_crossover, (stock_prices,)),
            ("pricevs50", self._calculate_price_vs_50d, (stock_prices,)),
            ("rsi14", self._calculate_rsi14, (stock_prices,)),
            ("macd12269", self._calculate_macd_histogram, (stock_prices,)),
            ("bbpos20", self._calculate_bollinger_position, (stock_prices,)),
            ("bbwidth20", self._calculate_bollinger_bandwidth, (stock_prices,)),
            ("volsurge", self._calculate_volume_surge, (stock_prices,)),
            ("obv20", self._calculate_obv_trend, (stock_prices,)),
            ("candlerev", self._calculate_candlestick_reversal, (stock_prices,)),
            ("srcontext", self._calculate_support_resistance, (stock_prices,)),
        ]
        
        # Group 2: Relative strength indicators (parallel within group)
        relative_indicators = [
            ("rel1y", self._calculate_relative_strength_1y, (stock_prices, sector_prices, industry_prices)),
            ("rel2y", self._calculate_relative_strength_2y, (stock_prices, sector_prices, industry_prices)),
        ]
        
        # Execute Group 1: Technical indicators (max parallelism)
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="TechInd") as executor:
            future_to_indicator = {
                executor.submit(self._safe_indicator_calculation, name, func, args): name 
                for name, func, args in technical_indicators
            }
            
            for future in as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per indicator
                    indicators[indicator_name] = result
                    
                    # Log to web interface if available
                    if logger_instance and result is not None:
                        display_name = self.get_indicator_display_name(indicator_name, result)
                        logger_instance.log_indicator_calculation(display_name, result)
                        
                    logger.debug(f"Completed technical indicator: {indicator_name}")
                    
                except Exception as e:
                    logger.warning(f"Technical indicator {indicator_name} failed: {str(e)}")
                    indicators[indicator_name] = None
        
        # Execute Group 2: Relative strength indicators  
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="RelStr") as executor:
            future_to_indicator = {
                executor.submit(self._safe_indicator_calculation, name, func, args): name 
                for name, func, args in relative_indicators
            }
            
            for future in as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                try:
                    result = future.result(timeout=15)  # 15 second timeout
                    indicators[indicator_name] = result
                    
                    if logger_instance and result is not None:
                        display_name = self.get_indicator_display_name(indicator_name, result)
                        logger_instance.log_indicator_calculation(display_name, result)
                        
                    logger.debug(f"Completed relative strength indicator: {indicator_name}")
                    
                except Exception as e:
                    logger.warning(f"Relative strength indicator {indicator_name} failed: {str(e)}")
                    indicators[indicator_name] = None
        
        # Execute Group 3: External data indicators
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ExtData") as executor:
            future_to_indicator = {
                executor.submit(self._safe_indicator_calculation, name, func, args): name 
                for name, func, args in external_indicators
            }
            
            for future in as_completed(future_to_indicator):
                indicator_name = future_to_indicator[future]
                try:
                    result = future.result(timeout=45)  # 45 second timeout for external data
                    indicators[indicator_name] = result
                    
                    if logger_instance and result is not None:
                        display_name = self.get_indicator_display_name(indicator_name, result)
                        logger_instance.log_indicator_calculation(display_name, result)
                        
                    logger.debug(f"Completed external data indicator: {indicator_name}")
                    
                except Exception as e:
                    logger.warning(f"External data indicator {indicator_name} failed: {str(e)}")
                    indicators[indicator_name] = None
        
        duration = time.time() - start_time
        logger.info(f"Parallel indicator calculation completed for {symbol} in {duration:.2f}s")
        
        return indicators
    
    def _safe_indicator_calculation(self, name: str, func, args: tuple):
        """
        Safely execute an indicator calculation with proper error handling.
        
        Args:
            name: Indicator name  
            func: Calculation function
            args: Function arguments
            
        Returns:
            Indicator result or None if failed
        """
        try:
            logger.debug(f"Starting calculation: {name}")
            start_time = time.time()
            
            result = func(*args)
            
            duration = time.time() - start_time  
            logger.debug(f"Completed {name} in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            logger.warning(f"Safe indicator calculation failed for {name}: {str(e)}")
            return None

    @cache_indicator(cache_ttl=600)  # 10 minutes cache for moving averages
    def _calculate_sma_crossover(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        SMA 50/200 Crossover: Score=1 if 50>200; 0 if 50<200. Weight 0.15.
        """
        try:
            if len(prices) < 200:
                return None

            # Vectorized calculation using NumPy
            closes = np.array([float(p.adjusted_close or p.close) for p in prices[-200:]])

            sma50 = np.mean(closes[-50:])
            sma200 = np.mean(closes)

            score = 1.0 if sma50 > sma200 else 0.0

            return IndicatorResult(
                raw={"sma50": float(sma50), "sma200": float(sma200)},
                score=score,
                weight=self.WEIGHTS["sma50vs200"],
                weighted_score=score * self.WEIGHTS["sma50vs200"],
            )

        except Exception as e:
            logger.warning(f"Error calculating SMA crossover: {str(e)}")
            return None

    def _calculate_price_vs_50d(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        Price vs 50d: pct=(close/SMA50-1); map −10%→0, 0→0.5, +10%→1 (clamp). Weight 0.10.
        """
        try:
            if len(prices) < 50:
                return None

            # Vectorized calculation using NumPy
            closes = np.array([float(p.adjusted_close or p.close) for p in prices[-50:]])
            current_price = closes[-1]
            sma50 = np.mean(closes)

            pct_diff = (current_price / sma50) - 1.0

            # Vectorized linear mapping: -10% → 0, 0% → 0.5, +10% → 1
            score = np.clip(0.5 + (pct_diff / 0.20), 0.0, 1.0)

            return IndicatorResult(
                raw={"price": float(current_price), "sma50": float(sma50), "pct_diff": float(pct_diff)},
                score=float(score),
                weight=self.WEIGHTS["pricevs50"],
                weighted_score=float(score) * self.WEIGHTS["pricevs50"],
            )

        except Exception as e:
            logger.warning(f"Error calculating price vs 50d: {str(e)}")
            return None

    @cache_indicator(cache_ttl=300)  # 5 minutes cache for RSI
    def _calculate_rsi14(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        RSI(14): standard formula over adjusted close; Score=RSI/100 (cap 0..1). Weight 0.10.
        """
        try:
            if len(prices) < 15:
                return None

            # Vectorized calculation using NumPy
            closes = np.array([float(p.adjusted_close or p.close) for p in prices[-15:]])

            # Calculate price changes using diff
            changes = np.diff(closes)
            
            # Vectorized gains and losses calculation
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)

            if len(gains) == 0:
                return None

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            score = np.clip(rsi / 100.0, 0.0, 1.0)

            return IndicatorResult(
                raw={"rsi": float(rsi)},
                score=float(score),
                weight=self.WEIGHTS["rsi14"],
                weighted_score=float(score) * self.WEIGHTS["rsi14"],
            )

        except Exception as e:
            logger.warning(f"Error calculating RSI14: {str(e)}")
            return None

    @cache_indicator(cache_ttl=300)  # 5 minutes cache for MACD
    def _calculate_macd_histogram(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        MACD(12,26,9) histogram: Score=0.5+0.5*(hist/(1% of price)); clamp 0..1. Weight 0.10.
        """
        try:
            if len(prices) < 35:  # Need at least 26 + 9 days
                return None

            # Vectorized calculation using NumPy
            closes = np.array([float(p.adjusted_close or p.close) for p in prices[-35:]])
            current_price = closes[-1]

            # Vectorized EMA calculation using pandas-style approach
            def ema_vectorized(data, period):
                alpha = 2 / (period + 1)
                weights = (1 - alpha) ** np.arange(len(data))[::-1]
                weights /= weights.sum()
                return np.convolve(data, weights, mode='valid')[-1]

            # Alternative vectorized EMA using NumPy cumulative approach
            def ema_cumulative(data, period):
                alpha = 2 / (period + 1)
                result = np.zeros_like(data)
                result[0] = data[0]
                for i in range(1, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                return result

            ema12_values = ema_cumulative(closes, 12)
            ema26_values = ema_cumulative(closes, 26)

            # MACD line calculation
            macd_line = ema12_values - ema26_values

            # Signal line (9-day EMA of MACD)
            if len(macd_line) >= 9:
                signal_line_values = ema_cumulative(macd_line[-9:], 9)
                histogram = macd_line[-1] - signal_line_values[-1]
                signal_value = signal_line_values[-1]
            else:
                histogram = 0
                signal_value = 0

            # Vectorized normalization: 0.5 + 0.5 * (hist / 1% of price)
            one_percent = current_price * 0.01
            if one_percent > 0:
                normalized = histogram / one_percent
                score = np.clip(0.5 + 0.5 * normalized, 0.0, 1.0)
            else:
                score = 0.5

            return IndicatorResult(
                raw={
                    "histogram": float(histogram),
                    "macd": float(macd_line[-1]),
                    "signal": float(signal_value),
                },
                score=float(score),
                weight=self.WEIGHTS["macd12269"],
                weighted_score=float(score) * self.WEIGHTS["macd12269"],
            )

        except Exception as e:
            logger.warning(f"Error calculating MACD histogram: {str(e)}")
            return None

    @cache_indicator(cache_ttl=300)  # 5 minutes cache for Bollinger Bands
    def _calculate_bollinger_position(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        Bollinger %B (20,2): %B=(close−lower)/(upper−lower); Score=1−%B; clamp 0..1. Weight 0.10.
        """
        try:
            if len(prices) < 20:
                return None

            # Vectorized calculation using NumPy
            closes = np.array([float(p.adjusted_close or p.close) for p in prices[-20:]])
            current_price = closes[-1]

            mean_price = np.mean(closes)
            std_dev = np.std(closes, ddof=1)  # Sample standard deviation

            upper_band = mean_price + (2 * std_dev)
            lower_band = mean_price - (2 * std_dev)

            if upper_band == lower_band:
                percent_b = 0.5
            else:
                percent_b = (current_price - lower_band) / (upper_band - lower_band)

            score = np.clip(1 - percent_b, 0.0, 1.0)

            return IndicatorResult(
                raw={
                    "percent_b": float(percent_b), 
                    "upper": float(upper_band), 
                    "lower": float(lower_band), 
                    "middle": float(mean_price)
                },
                score=float(score),
                weight=self.WEIGHTS["bbpos20"],
                weighted_score=float(score) * self.WEIGHTS["bbpos20"],
            )

        except Exception as e:
            logger.warning(f"Error calculating Bollinger %B: {str(e)}")
            return None

    def _calculate_bollinger_bandwidth(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        Bollinger Bandwidth (bbWidth20): width=(upper−lower)/mid; map ≤5%→0.6, 12.5%→0.5, ≥20%→0.4 linear. Weight 0.05.
        """
        try:
            if len(prices) < 20:
                return None

            # Vectorized calculation using NumPy
            closes = np.array([float(p.adjusted_close or p.close) for p in prices[-20:]])

            mean_price = np.mean(closes)
            std_dev = np.std(closes, ddof=1)  # Sample standard deviation

            upper_band = mean_price + (2 * std_dev)
            lower_band = mean_price - (2 * std_dev)

            if mean_price == 0:
                bandwidth_pct = 0
            else:
                bandwidth = upper_band - lower_band
                bandwidth_pct = bandwidth / mean_price

            # Vectorized linear mapping using np.select for multiple conditions
            conditions = [
                bandwidth_pct <= 0.05,
                bandwidth_pct >= 0.20,
                bandwidth_pct <= 0.125
            ]
            
            choices = [
                0.6,  # ≤5% → 0.6
                0.4,  # ≥20% → 0.4
                0.6 - ((bandwidth_pct - 0.05) / (0.125 - 0.05)) * 0.1  # 5%-12.5%: 0.6 to 0.5
            ]
            
            # Default case for 12.5%-20%: 0.5 to 0.4
            default = 0.5 - ((bandwidth_pct - 0.125) / (0.20 - 0.125)) * 0.1
            
            score = np.select(conditions, choices, default=default)

            return IndicatorResult(
                raw={"bandwidth_pct": float(bandwidth_pct), "bandwidth": float(upper_band - lower_band)},
                score=float(score),
                weight=self.WEIGHTS["bbwidth20"],
                weighted_score=float(score) * self.WEIGHTS["bbwidth20"],
            )

        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bandwidth: {str(e)}")
            return None

    def _calculate_volume_surge(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        Volume Surge: compare today volume to 10-day mean; complex mapping based on price direction. Weight 0.10.
        """
        try:
            if len(prices) < 11:
                return None

            current_volume = prices[-1].volume
            # Vectorized volume calculation
            recent_volumes = np.array([p.volume for p in prices[-11:-1]])  # Last 10 days excluding today
            avg_volume = np.mean(recent_volumes)

            if avg_volume == 0:
                volume_ratio = 1.0
            else:
                volume_ratio = current_volume / avg_volume

            # Price direction for today
            current_price = prices[-1]
            price_up = current_price.close > current_price.open

            # Vectorized mapping based on direction and volume ratio using np.select
            if price_up:
                conditions = [volume_ratio >= 1.5, volume_ratio >= 1.0]
                choices = [1.0, 0.8]
                score = np.select(conditions, choices, default=0.6)
            else:  # price down
                conditions = [volume_ratio >= 1.5, volume_ratio >= 1.0]
                choices = [0.0, 0.3]
                score = np.select(conditions, choices, default=0.4)

            return IndicatorResult(
                raw={"volume_ratio": float(volume_ratio), "price_up": bool(price_up)},
                score=float(score),
                weight=self.WEIGHTS["volsurge"],
                weighted_score=float(score) * self.WEIGHTS["volsurge"],
            )

        except Exception as e:
            logger.warning(f"Error calculating Volume Surge: {str(e)}")
            return None

    def _calculate_obv_trend(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        OBV 20-day trend: OBV_delta=(OBV_t−OBV_t−20)/sum(volume_20); Score=(OBV_delta+1)/2. Weight 0.05.
        """
        try:
            if len(prices) < 21:
                return None

            # Vectorized OBV calculation using NumPy
            closes = np.array([float(p.close) for p in prices])
            volumes = np.array([p.volume for p in prices])
            
            # Calculate price changes and volume directions
            price_changes = np.diff(closes)
            volume_directions = np.where(price_changes > 0, 1, np.where(price_changes < 0, -1, 0))
            
            # Calculate OBV using cumulative sum
            volume_flow = volumes[1:] * volume_directions
            obv_values = np.concatenate([[0], np.cumsum(volume_flow)])

            # Get current and 20-day ago OBV
            obv_current = obv_values[-1]
            obv_20_ago = obv_values[-21]

            # Volume sum for last 20 days using NumPy
            volume_sum = np.sum(volumes[-20:])

            if volume_sum == 0:
                obv_delta = 0
            else:
                obv_delta = (obv_current - obv_20_ago) / volume_sum

            # Vectorized score calculation and clipping
            score = np.clip((obv_delta + 1) / 2, 0.0, 1.0)

            return IndicatorResult(
                raw={
                    "obv_delta": float(obv_delta), 
                    "obv_current": float(obv_current), 
                    "obv_20_ago": float(obv_20_ago)
                },
                score=float(score),
                weight=self.WEIGHTS["obv20"],
                weighted_score=float(score) * self.WEIGHTS["obv20"],
            )

        except Exception as e:
            logger.warning(f"Error calculating OBV trend: {str(e)}")
            return None

    def _calculate_relative_strength_1y(
        self,
        stock_prices: List[PriceData],
        sector_prices: List[SectorPriceData],
        industry_prices: List[IndustryPriceData],
    ) -> Optional[IndicatorResult]:
        """
        Rel Strength 1Y: stock_1Y minus avg(sector_1Y, industry_1Y); map −20pp→0, 0→0.5, +20pp→1. Weight 0.05.
        """
        try:
            if len(stock_prices) < 252:  # ~1 year trading days
                return None

            # Vectorized stock 1Y return calculation
            stock_prices_array = np.array([float(p.adjusted_close or p.close) for p in stock_prices])
            stock_current = stock_prices_array[-1]
            stock_1y_ago = stock_prices_array[-252]
            stock_1y_return = (stock_current / stock_1y_ago - 1) * 100

            # Enhanced benchmark returns calculation including market benchmark (SPY)
            benchmark_returns = []
            
            # Add market benchmark (SPY) if available
            try:
                from Data.models import Stock
                spy_stock = Stock.objects.get(symbol='SPY')
                spy_prices = self.price_reader.get_stock_prices(
                    symbol='SPY', 
                    start_date=(datetime.now() - timedelta(days=400)).date(),
                    end_date=datetime.now().date()
                )
                
                if spy_prices and len(spy_prices) >= 252:
                    spy_current = float(spy_prices[-1].close)
                    spy_1y_ago = float(spy_prices[-252].close)
                    spy_1y_return = (spy_current / spy_1y_ago - 1) * 100
                    
                    # Validate SPY return (should be reasonable for a market ETF)
                    if -50 <= spy_1y_return <= 60:  # Reasonable market range
                        benchmark_returns.append(spy_1y_return)
                        logger.debug(f"REL1Y: SPY benchmark added: {spy_1y_return:.2f}%")
                    else:
                        logger.warning(f"REL1Y: Extreme SPY return detected ({spy_1y_return:.1f}%), skipping")
                        
            except Exception as e:
                logger.debug(f"REL1Y: Could not get SPY benchmark data: {str(e)}")

            if sector_prices and len(sector_prices) >= 252:
                sector_prices_array = np.array([float(p.close_index) for p in sector_prices])
                sector_current = sector_prices_array[-1]
                sector_1y_ago = sector_prices_array[-252]
                sector_1y_return = (sector_current / sector_1y_ago - 1) * 100
                
                # Validate sector return for anomalies
                if abs(sector_1y_return) > 300:  # >300% return is likely erroneous
                    logger.warning(
                        f"REL1Y: Extreme sector return detected ({sector_1y_return:.1f}%), "
                        f"sector prices: current={sector_current:.2f}, 1y_ago={sector_1y_ago:.2f}. Skipping."
                    )
                else:
                    benchmark_returns.append(sector_1y_return)

            if industry_prices and len(industry_prices) >= 252:
                industry_prices_array = np.array([float(p.close_index) for p in industry_prices])
                industry_current = industry_prices_array[-1]
                industry_1y_ago = industry_prices_array[-252]
                industry_1y_return = (industry_current / industry_1y_ago - 1) * 100
                
                # Validate industry return for anomalies
                if abs(industry_1y_return) > 300:  # >300% return is likely erroneous
                    logger.warning(
                        f"REL1Y: Extreme industry return detected ({industry_1y_return:.1f}%), "
                        f"industry prices: current={industry_current:.2f}, 1y_ago={industry_1y_ago:.2f}. Skipping."
                    )
                else:
                    benchmark_returns.append(industry_1y_return)

            if not benchmark_returns:
                logger.info("REL1Y: No benchmark data available, using neutral baseline (0% relative strength)")
                avg_benchmark = stock_1y_return  # This makes relative_strength = 0
            else:
                # Weight SPY more heavily if available (market benchmark is primary)
                if len(benchmark_returns) > 1 and 'spy_1y_return' in locals():
                    # Weighted average: 60% SPY, 40% sector/industry average
                    other_benchmarks = [b for i, b in enumerate(benchmark_returns) if i > 0]  # Skip SPY (first)
                    if other_benchmarks:
                        spy_weight = 0.6
                        other_weight = 0.4
                        avg_benchmark = spy_weight * spy_1y_return + other_weight * np.mean(other_benchmarks)
                    else:
                        avg_benchmark = spy_1y_return
                else:
                    avg_benchmark = np.mean(benchmark_returns)
            
            relative_strength = stock_1y_return - avg_benchmark

            # Vectorized mapping: -20pp → 0, 0 → 0.5, +20pp → 1
            score = np.clip(0.5 + (relative_strength / 40), 0.0, 1.0)

            return IndicatorResult(
                raw={
                    "relative_strength": float(relative_strength),
                    "stock_return": float(stock_1y_return),
                    "benchmark_return": float(avg_benchmark),
                },
                score=float(score),
                weight=self.WEIGHTS["rel1y"],
                weighted_score=float(score) * self.WEIGHTS["rel1y"],
            )

        except Exception as e:
            logger.warning(f"Error calculating 1Y relative strength: {str(e)}")
            return None

    def _calculate_relative_strength_2y(
        self,
        stock_prices: List[PriceData],
        sector_prices: List[SectorPriceData],
        industry_prices: List[IndustryPriceData],
    ) -> Optional[IndicatorResult]:
        """
        Rel Strength 2Y: stock_2Y minus avg(sector_2Y, industry_2Y); map −100pp→0, 0→0.5, +100pp→1. Weight 0.05.
        """
        try:
            # Require at least 99% of expected 2-year data (~500 trading days minimum)
            min_required_days = int(504 * 0.99)  # ~499 days
            if len(stock_prices) < min_required_days:
                return None

            # Use available data length for baseline calculation
            baseline_index = min(504, len(stock_prices)) - 1

            # Vectorized stock 2Y return calculation
            stock_prices_array = np.array([float(p.adjusted_close or p.close) for p in stock_prices])
            stock_current = stock_prices_array[-1]
            stock_2y_ago = stock_prices_array[-baseline_index - 1]
            stock_2y_return = (stock_current / stock_2y_ago - 1) * 100

            # Enhanced benchmark returns calculation including market benchmark (SPY)
            benchmark_returns = []
            
            # Add market benchmark (SPY) if available
            try:
                from Data.models import Stock
                spy_stock = Stock.objects.get(symbol='SPY')
                spy_prices = self.price_reader.get_stock_prices(
                    symbol='SPY', 
                    start_date=(datetime.now() - timedelta(days=800)).date(),
                    end_date=datetime.now().date()
                )
                
                if spy_prices and len(spy_prices) >= min_required_days:
                    spy_baseline_index = min(504, len(spy_prices)) - 1
                    spy_current = float(spy_prices[-1].close)
                    spy_2y_ago = float(spy_prices[-spy_baseline_index - 1].close)
                    spy_2y_return = (spy_current / spy_2y_ago - 1) * 100
                    
                    # Validate SPY return (should be reasonable for a market ETF)
                    if -60 <= spy_2y_return <= 150:  # Reasonable 2-year market range
                        benchmark_returns.append(spy_2y_return)
                        logger.debug(f"REL2Y: SPY benchmark added: {spy_2y_return:.2f}%")
                    else:
                        logger.warning(f"REL2Y: Extreme SPY return detected ({spy_2y_return:.1f}%), skipping")
                        
            except Exception as e:
                logger.debug(f"REL2Y: Could not get SPY benchmark data: {str(e)}")

            if sector_prices and len(sector_prices) >= min_required_days:
                sector_baseline_index = min(504, len(sector_prices)) - 1
                sector_prices_array = np.array([float(p.close_index) for p in sector_prices])
                sector_current = sector_prices_array[-1]
                sector_2y_ago = sector_prices_array[-sector_baseline_index - 1]
                sector_2y_return = (sector_current / sector_2y_ago - 1) * 100
                
                # Validate sector return for anomalies
                if abs(sector_2y_return) > 500:  # >500% return is likely erroneous
                    logger.warning(
                        f"REL2Y: Extreme sector return detected ({sector_2y_return:.1f}%), "
                        f"sector prices: current={sector_current:.2f}, 2y_ago={sector_2y_ago:.2f}. Skipping."
                    )
                else:
                    benchmark_returns.append(sector_2y_return)

            if industry_prices and len(industry_prices) >= min_required_days:
                industry_baseline_index = min(504, len(industry_prices)) - 1
                industry_prices_array = np.array([float(p.close_index) for p in industry_prices])
                industry_current = industry_prices_array[-1]
                industry_2y_ago = industry_prices_array[-industry_baseline_index - 1]
                industry_2y_return = (industry_current / industry_2y_ago - 1) * 100
                
                # Validate industry return for anomalies
                if abs(industry_2y_return) > 500:  # >500% return is likely erroneous
                    logger.warning(
                        f"REL2Y: Extreme industry return detected ({industry_2y_return:.1f}%), "
                        f"industry prices: current={industry_current:.2f}, 2y_ago={industry_2y_ago:.2f}. Skipping."
                    )
                else:
                    benchmark_returns.append(industry_2y_return)

            if not benchmark_returns:
                logger.info("REL2Y: No benchmark data available, using neutral baseline (0% relative strength)")
                avg_benchmark = stock_2y_return  # This makes relative_strength = 0
            else:
                # Weight SPY more heavily if available (market benchmark is primary)
                if len(benchmark_returns) > 1 and 'spy_2y_return' in locals():
                    # Weighted average: 60% SPY, 40% sector/industry average
                    other_benchmarks = [b for i, b in enumerate(benchmark_returns) if i > 0]  # Skip SPY (first)
                    if other_benchmarks:
                        spy_weight = 0.6
                        other_weight = 0.4
                        avg_benchmark = spy_weight * spy_2y_return + other_weight * np.mean(other_benchmarks)
                    else:
                        avg_benchmark = spy_2y_return
                else:
                    avg_benchmark = np.mean(benchmark_returns)
                
                # Validate for extreme benchmark returns (likely data errors)
                if abs(avg_benchmark) > 500:  # >500% return is likely erroneous
                    logger.warning(
                        f"REL2Y: Extreme benchmark return detected ({avg_benchmark:.1f}%), "
                        f"individual returns: {benchmark_returns}. Using neutral baseline."
                    )
                    avg_benchmark = stock_2y_return  # Use neutral baseline to avoid skewed results
            
            relative_strength = stock_2y_return - avg_benchmark

            # Vectorized mapping: -100pp → 0, 0 → 0.5, +100pp → 1
            score = np.clip(0.5 + (relative_strength / 200), 0.0, 1.0)

            return IndicatorResult(
                raw={
                    "relative_strength": float(relative_strength),
                    "stock_return": float(stock_2y_return),
                    "benchmark_return": float(avg_benchmark),
                },
                score=float(score),
                weight=self.WEIGHTS["rel2y"],
                weighted_score=float(score) * self.WEIGHTS["rel2y"],
            )

        except Exception as e:
            logger.warning(f"Error calculating 2Y relative strength: {str(e)}")
            return None

    def _calculate_candlestick_reversal(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        Candlestick Reversal: detect patterns in last 3-5 days; bullish→1, bearish→0, none→0.5. Weight 0.08.
        """
        try:
            if len(prices) < 5:
                return None

            recent_prices = prices[-5:]
            pattern = self._detect_candlestick_patterns(recent_prices)

            if pattern["type"] == "bullish":
                score = 1.0
            elif pattern["type"] == "bearish":
                score = 0.0
            elif pattern["type"] == "neutral":
                score = 0.5  # Doji indicates indecision, neutral signal
            else:
                score = 0.5

            return IndicatorResult(
                raw=pattern,
                score=score,
                weight=self.WEIGHTS["candlerev"],
                weighted_score=score * self.WEIGHTS["candlerev"],
            )

        except Exception as e:
            logger.warning(f"Error calculating candlestick reversal: {str(e)}")
            return None

    def _detect_candlestick_patterns(self, prices: List[PriceData]) -> Dict[str, Any]:
        """Detect candlestick reversal patterns with expanded pattern library."""
        try:
            if not prices:
                return {"type": "none", "pattern": "insufficient_data"}

            # Check for patterns in all recent candles, prioritizing multi-candle patterns

            # Three candle patterns first (most significant)
            if len(prices) >= 3:
                three_candle_result = self._check_three_candle_patterns(prices)
                if three_candle_result["type"] != "none":
                    return three_candle_result

            # Two candle patterns second
            if len(prices) >= 2:
                two_candle_result = self._check_two_candle_patterns(prices)
                if two_candle_result["type"] != "none":
                    return two_candle_result

            # Single candle patterns last - check all recent candles
            for i in range(len(prices)):  # Check all candles for single patterns
                candle = prices[-(i + 1)]  # Start from most recent
                single_result = self._check_single_candle_patterns(candle)
                if single_result["type"] != "none":
                    return single_result

            return {"type": "none", "pattern": "no_pattern"}

        except Exception as e:
            logger.warning(f"Error detecting candlestick patterns: {str(e)}")
            return {"type": "none", "pattern": "error"}

    def _check_single_candle_patterns(self, price: PriceData) -> Dict[str, Any]:
        """Check single candle patterns."""
        try:
            # Get candle data
            last_open = float(price.open)
            last_high = float(price.high)
            last_low = float(price.low)
            last_close = float(price.close)

            # Calculate body and shadow sizes
            body_size = abs(last_close - last_open)
            total_range = last_high - last_low
            upper_shadow = last_high - max(last_open, last_close)
            lower_shadow = min(last_open, last_close) - last_low

            # Avoid division by zero
            if total_range == 0:
                return {"type": "none", "pattern": "no_range"}

            # Single candle patterns with relaxed thresholds

            # Doji - body is very small relative to range (indecision)
            if body_size <= total_range * 0.1 and total_range > 0:
                return {"type": "neutral", "pattern": "doji"}

            # Hammer pattern (bullish) - relaxed threshold
            if lower_shadow >= 1.5 * body_size and upper_shadow <= body_size * 0.7 and body_size > total_range * 0.05:
                return {"type": "bullish", "pattern": "hammer"}

            # Inverted Hammer (bullish) - long upper shadow at bottom of downtrend
            if upper_shadow >= 1.5 * body_size and lower_shadow <= body_size * 0.7 and body_size > total_range * 0.05:
                return {"type": "bullish", "pattern": "inverted_hammer"}

            # Shooting Star pattern (bearish) - relaxed threshold
            if upper_shadow >= 1.5 * body_size and lower_shadow <= body_size * 0.7 and body_size > total_range * 0.05:
                return {"type": "bearish", "pattern": "shooting_star"}

            # Hanging Man (bearish) - same as hammer but in uptrend context
            if lower_shadow >= 1.5 * body_size and upper_shadow <= body_size * 0.7 and body_size > total_range * 0.05:
                return {"type": "bearish", "pattern": "hanging_man"}

            return {"type": "none", "pattern": "no_pattern"}

        except Exception as e:
            logger.warning(f"Error detecting single candle patterns: {str(e)}")
            return {"type": "none", "pattern": "error"}

    def _check_two_candle_patterns(self, prices: List[PriceData]) -> Dict[str, Any]:
        """Check two candle patterns."""
        try:
            if len(prices) < 2:
                return {"type": "none", "pattern": "insufficient_data"}

            prev = prices[-2]
            last = prices[-1]

            prev_open = float(prev.open)
            prev_close = float(prev.close)
            prev_high = float(prev.high)
            prev_low = float(prev.low)
            prev_body_size = abs(prev_close - prev_open)

            last_open = float(last.open)
            last_close = float(last.close)
            body_size = abs(last_close - last_open)

            # Bullish Engulfing - relaxed requirements
            if (
                prev_close < prev_open  # Previous was bearish
                and last_close > last_open  # Current is bullish
                and last_close > prev_open  # Current close above prev open
                and last_open < prev_close  # Current open below prev close
                and body_size >= prev_body_size * 0.8
            ):  # Current body at least 80% of prev
                return {"type": "bullish", "pattern": "bullish_engulfing"}

            # Bearish Engulfing - relaxed requirements
            if (
                prev_close > prev_open  # Previous was bullish
                and last_close < last_open  # Current is bearish
                and last_close < prev_open  # Current close below prev open
                and last_open > prev_close  # Current open above prev close
                and body_size >= prev_body_size * 0.8
            ):  # Current body at least 80% of prev
                return {"type": "bearish", "pattern": "bearish_engulfing"}

            # Piercing Pattern (bullish)
            if (
                prev_close < prev_open  # Previous bearish
                and last_close > last_open  # Current bullish
                and last_open < prev_low  # Gap down open
                and last_close > (prev_open + prev_close) / 2  # Close above midpoint
                and last_close < prev_open
            ):  # But below prev open
                return {"type": "bullish", "pattern": "piercing"}

            # Dark Cloud Cover (bearish)
            if (
                prev_close > prev_open  # Previous bullish
                and last_close < last_open  # Current bearish
                and last_open > prev_high  # Gap up open
                and last_close < (prev_open + prev_close) / 2  # Close below midpoint
                and last_close > prev_open
            ):  # But above prev open
                return {"type": "bearish", "pattern": "dark_cloud"}

            return {"type": "none", "pattern": "no_pattern"}

        except Exception as e:
            logger.warning(f"Error detecting two candle patterns: {str(e)}")
            return {"type": "none", "pattern": "error"}

    def _check_three_candle_patterns(self, prices: List[PriceData]) -> Dict[str, Any]:
        """Check three candle patterns."""
        try:
            if len(prices) < 3:
                return {"type": "none", "pattern": "insufficient_data"}

            first = prices[-3]
            middle = prices[-2]
            last = prices[-1]

            first_open = float(first.open)
            first_close = float(first.close)
            first_high = float(first.high)
            first_low = float(first.low)

            middle_open = float(middle.open)
            middle_close = float(middle.close)
            middle_high = float(middle.high)
            middle_low = float(middle.low)

            last_open = float(last.open)
            last_close = float(last.close)

            middle_body = abs(middle_close - middle_open)
            middle_range = middle_high - middle_low

            # Morning Star (bullish)
            if (
                first_close < first_open  # First bearish
                and last_close > last_open  # Third bullish
                and middle_body < middle_range * 0.3  # Middle is small body (star)
                and middle_high < first_close  # Gap down from first
                and last_open > middle_high  # Gap up to third
                and last_close > (first_open + first_close) / 2
            ):  # Close above first midpoint
                return {"type": "bullish", "pattern": "morning_star"}

            # Evening Star (bearish)
            if (
                first_close > first_open  # First bullish
                and last_close < last_open  # Third bearish
                and middle_body < middle_range * 0.3  # Middle is small body (star)
                and middle_low > first_close  # Gap up from first
                and last_open < middle_low  # Gap down to third
                and last_close < (first_open + first_close) / 2
            ):  # Close below first midpoint
                return {"type": "bearish", "pattern": "evening_star"}

            return {"type": "none", "pattern": "no_pattern"}

        except Exception as e:
            logger.warning(f"Error detecting three candle patterns: {str(e)}")
            return {"type": "none", "pattern": "error"}

    @cache_indicator(cache_ttl=600)  # 10 minutes cache for support/resistance
    def _calculate_support_resistance(self, prices: List[PriceData]) -> Optional[IndicatorResult]:
        """
        Support/Resistance: break above resistance→1; near resistance (≤2%)→0.3; near support (≤2%)→0.7; break below→0; else 0.5. Weight 0.07.
        """
        try:
            if len(prices) < 50:
                return None

            current_price = float(prices[-1].close)

            # Find support and resistance levels using pivot points
            highs = [float(p.high) for p in prices[-50:]]
            lows = [float(p.low) for p in prices[-50:]]

            resistance_levels = self._find_resistance_levels(highs)
            support_levels = self._find_support_levels(lows)

            # Find nearest levels
            nearest_resistance = None
            nearest_support = None

            for level in resistance_levels:
                if level > current_price:
                    if nearest_resistance is None or level < nearest_resistance:
                        nearest_resistance = level

            for level in support_levels:
                if level < current_price:
                    if nearest_support is None or level > nearest_support:
                        nearest_support = level

            # Determine score based on position relative to levels
            score = 0.5  # Default
            context = "neutral"

            # Check for breaks above resistance
            if nearest_resistance and current_price > nearest_resistance:
                # Already broke above resistance
                score = 1.0
                context = "break_above_resistance"
            # Check if near resistance (within 2%)
            elif nearest_resistance and abs(current_price - nearest_resistance) / nearest_resistance <= 0.02:
                if current_price < nearest_resistance:
                    score = 0.3
                    context = "near_resistance"
            # Check for breaks below support
            elif nearest_support and current_price < nearest_support:
                # Already broke below support
                score = 0.0
                context = "break_below_support"
            # Check if near support (within 2%)
            elif nearest_support and abs(current_price - nearest_support) / nearest_support <= 0.02:
                if current_price > nearest_support:
                    score = 0.7
                    context = "near_support"

            return IndicatorResult(
                raw={
                    "current_price": current_price,
                    "nearest_resistance": nearest_resistance,
                    "nearest_support": nearest_support,
                    "context": context,
                },
                score=score,
                weight=self.WEIGHTS["srcontext"],
                weighted_score=score * self.WEIGHTS["srcontext"],
            )

        except Exception as e:
            logger.warning(f"Error calculating support/resistance: {str(e)}")
            return None

    def _find_resistance_levels(self, highs: List[float]) -> List[float]:
        """Find resistance levels from high prices using vectorized operations."""
        if len(highs) < 10:
            return []

        # Vectorized approach: find local maxima using NumPy
        highs_array = np.array(highs)
        
        # Create sliding windows for comparison
        if len(highs_array) < 5:
            return []
        
        # Find local maxima by comparing each point with its neighbors
        local_maxima_mask = np.zeros(len(highs_array), dtype=bool)
        
        for i in range(2, len(highs_array) - 2):
            if (highs_array[i] > highs_array[i-2:i]).all() and (highs_array[i] > highs_array[i+1:i+3]).all():
                local_maxima_mask[i] = True
        
        levels = highs_array[local_maxima_mask].tolist()

        # Vectorized filtering: remove levels too close to each other (within 1%)
        if not levels:
            return []
        
        levels.sort()
        levels_array = np.array(levels)
        
        # Calculate percentage differences between consecutive levels
        if len(levels_array) > 1:
            diffs = np.diff(levels_array) / levels_array[:-1]
            keep_mask = np.concatenate([[True], diffs > 0.01])
            filtered_levels = levels_array[keep_mask].tolist()
        else:
            filtered_levels = levels

        return filtered_levels[-3:]  # Return top 3 levels

    def _find_support_levels(self, lows: List[float]) -> List[float]:
        """Find support levels from low prices using vectorized operations."""
        if len(lows) < 10:
            return []

        # Vectorized approach: find local minima using NumPy
        lows_array = np.array(lows)
        
        if len(lows_array) < 5:
            return []
        
        # Find local minima by comparing each point with its neighbors
        local_minima_mask = np.zeros(len(lows_array), dtype=bool)
        
        for i in range(2, len(lows_array) - 2):
            if (lows_array[i] < lows_array[i-2:i]).all() and (lows_array[i] < lows_array[i+1:i+3]).all():
                local_minima_mask[i] = True
        
        levels = lows_array[local_minima_mask].tolist()

        # Vectorized filtering: remove levels too close to each other (within 1%)
        if not levels:
            return []
        
        levels.sort(reverse=True)
        levels_array = np.array(levels)
        
        # Calculate percentage differences between consecutive levels
        if len(levels_array) > 1:
            diffs = np.abs(np.diff(levels_array)) / levels_array[:-1]
            keep_mask = np.concatenate([[True], diffs > 0.01])
            filtered_levels = levels_array[keep_mask].tolist()
        else:
            filtered_levels = levels

        return filtered_levels[-3:]  # Return bottom 3 levels

    def _calculate_sentiment_analysis(self, symbol: str) -> Optional[IndicatorResult]:
        """
        Calculate sentiment score from news analysis.
        Score range: -1 (bearish) to 1 (bullish), normalized to 0-1 scale.
        Weight: 0.10 (10% of composite score)
        """
        try:
            from Analytics.services.sentiment_analyzer import sentiment_metrics

            logger.info(f"Calculating sentiment analysis for {symbol}")
            
            # Check for cached result (24 hour TTL)
            sentiment_cache_key = f"sentiment_cache_{symbol}_24h"
            cached_result = cache.get(sentiment_cache_key)
            if cached_result:
                logger.debug(f"Using cached sentiment analysis for {symbol}")
                return cached_result

            # Get sentiment analyzer instance
            sentiment_analyzer = get_sentiment_analyzer()

            # Fetch news for the stock (90 days history)
            news_items = yahoo_finance_service.fetchNewsForStock(symbol, days=90, max_items=50)

            # Log request metrics
            sentiment_metrics.log_request(symbol, len(news_items))

            if not news_items:
                logger.warning(f"No news found for {symbol}, using neutral sentiment")
                return IndicatorResult(
                    raw={"sentiment": 0.0, "label": "neutral", "newsCount": 0},
                    score=0.5,  # Neutral score
                    weight=self.WEIGHTS["sentiment"],
                    weighted_score=0.5 * self.WEIGHTS["sentiment"],
                )

            # Analyze sentiment using the news articles method with symbol
            aggregated = sentiment_analyzer.analyzeNewsArticles(news_items[:30], aggregate=True, symbol=symbol)

            # Convert sentiment score (-1 to 1) to normalized score (0 to 1)
            raw_sentiment = aggregated.get("sentimentScore", 0.0)
            normalized_score = (raw_sentiment + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]

            # Apply confidence-based adjustment
            confidence = aggregated.get("sentimentConfidence", 0.0)
            if confidence < 0.6:  # Below 60% confidence threshold
                # Pull score toward neutral based on low confidence
                normalized_score = 0.5 + (normalized_score - 0.5) * (confidence / 0.6)

            news_count = aggregated.get("newsCount", 0)
            logger.info(
                f"Sentiment for {symbol}: score={raw_sentiment:.3f}, normalized={normalized_score:.3f}, "
                f"label={aggregated.get('sentimentLabel')}, articles={news_count}"
            )

            result = IndicatorResult(
                raw={
                    "sentiment": raw_sentiment,
                    "label": aggregated.get("sentimentLabel", "neutral"),
                    "confidence": confidence,
                    "newsCount": news_count,
                    "distribution": aggregated.get("distribution", {}),
                    "fallback": aggregated.get("fallback", False),
                },
                score=normalized_score,
                weight=self.WEIGHTS["sentiment"],
                weighted_score=normalized_score * self.WEIGHTS["sentiment"],
            )
            
            # Cache the result for 24 hours
            cache.set(sentiment_cache_key, result, 24 * 60 * 60)
            
            return result

        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {str(e)}")
            # Return neutral sentiment on error
            return IndicatorResult(
                raw={"sentiment": 0.0, "label": "neutral", "error": str(e)},
                score=0.5,
                weight=self.WEIGHTS["sentiment"],
                weighted_score=0.5 * self.WEIGHTS["sentiment"],
            )

    def _calculate_prediction_score(self, symbol: str) -> Optional[IndicatorResult]:
        """
        Calculate LSTM price prediction score.
        Score range: 0 (bearish) to 1 (bullish) based on predicted price movement.
        Weight: 0.10 (10% of composite score)
        """
        try:
            # Check for cached result (6 hour TTL - predictions are more time-sensitive)
            prediction_cache_key = f"prediction_cache_{symbol}_6h"
            cached_result = cache.get(prediction_cache_key)
            if cached_result:
                logger.debug(f"Using cached prediction for {symbol}")
                return cached_result
            
            # Use Universal LSTM service instead of stock-specific predictor
            universal_lstm_service = get_universal_lstm_service()

            logger.info(f"Calculating Universal LSTM prediction for {symbol}")

            # Get 1-day prediction using universal model
            prediction_result = universal_lstm_service.predict_stock_price(symbol, horizon="1d")

            if not prediction_result:
                logger.warning(f"No Universal LSTM prediction available for {symbol}, using neutral score")
                return IndicatorResult(
                    raw={"prediction": None, "error": "Universal LSTM model did not produce prediction"},
                    score=0.5,  # Neutral score
                    weight=self.WEIGHTS["prediction"],
                    weighted_score=0.5 * self.WEIGHTS["prediction"],
                )

            # Sanity check: reject unrealistic predictions
            price_change_pct = prediction_result.get("price_change_pct", 0)
            if abs(price_change_pct) > 50:  # More than 50% change is unrealistic for 1-day prediction
                logger.warning(
                    f"Unrealistic LSTM prediction for {symbol}: {price_change_pct:.2f}% change, using neutral score"
                )
                return IndicatorResult(
                    raw={"prediction": None, "error": f"Unrealistic prediction: {price_change_pct:.2f}% change"},
                    score=0.5,  # Neutral score
                    weight=self.WEIGHTS["prediction"],
                    weighted_score=0.5 * self.WEIGHTS["prediction"],
                )

            # Normalize prediction to score using the Universal service's method
            normalized_score = universal_lstm_service.normalize_prediction_score(prediction_result)

            logger.info(
                f"Universal LSTM prediction for {symbol} ({prediction_result.get('sector_name', 'Unknown')}): "
                f"price=${prediction_result['predicted_price']:.2f}, "
                f"change={prediction_result['price_change_pct']:+.2f}%, "
                f"score={normalized_score:.3f}, "
                f"confidence={prediction_result['confidence']:.3f}"
            )

            result = IndicatorResult(
                raw={
                    "predicted_price": prediction_result["predicted_price"],
                    "current_price": prediction_result["current_price"],
                    "price_change": prediction_result["price_change"],
                    "price_change_pct": prediction_result["price_change_pct"],
                    "confidence": prediction_result["confidence"],
                    "model_version": prediction_result["model_version"],
                    "model_type": prediction_result.get("model_type", "UniversalLSTM"),
                    "sector_name": prediction_result.get("sector_name", "Unknown"),
                    "sector_id": prediction_result.get("sector_id", 10),
                    "horizon": prediction_result["horizon"],
                },
                score=normalized_score,
                weight=self.WEIGHTS["prediction"],
                weighted_score=normalized_score * self.WEIGHTS["prediction"],
            )
            
            # Cache the result for 6 hours
            cache.set(prediction_cache_key, result, 6 * 60 * 60)
            
            return result

        except Exception as e:
            logger.error(f"Error calculating LSTM prediction for {symbol}: {str(e)}")
            # Return neutral score on error
            return IndicatorResult(
                raw={"prediction": None, "error": str(e)},
                score=0.5,
                weight=self.WEIGHTS["prediction"],
                weighted_score=0.5 * self.WEIGHTS["prediction"],
            )

    def _auto_populate_sector_industry(self, symbol: str) -> bool:
        """
        Auto-populate missing sector/industry data for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from Data.services.yahoo_finance import yahoo_finance_service
            
            logger.info(f"Fetching sector/industry data for {symbol}")
            
            # Fetch sector/industry data from Yahoo Finance
            profile_data = yahoo_finance_service.fetchSectorIndustrySingle(symbol)
            
            if profile_data and "error" not in profile_data:
                # Upsert the data
                result = yahoo_finance_service.upsertCompanyProfiles({symbol: profile_data})
                if result[0] > 0 or result[1] > 0:  # created_count or updated_count
                    logger.info(f"Successfully populated sector/industry for {symbol}: {profile_data.get('sector', '')}/{profile_data.get('industry', '')}")
                    return True
                else:
                    logger.warning(f"Failed to upsert sector/industry data for {symbol}")
                    return False
            else:
                error_msg = profile_data.get("error", "Unknown error") if profile_data else "No data returned"
                logger.warning(f"Could not fetch sector/industry data for {symbol}: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"Error auto-populating sector/industry for {symbol}: {str(e)}")
            return False
