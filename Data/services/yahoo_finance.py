"""
Yahoo Finance API integration service providing comprehensive financial data retrieval.
Implements caching optimisation and data synchronisation for market data processing.
"""

import hashlib
import logging
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Configure yfinance with proper headers and SSL verification
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import connection pooling
from Core.connection_pool import get_connection_pool, PooledHTTPClient

# Module-level timeout for explicit HTTP call configuration
DEFAULT_TIMEOUT = 30

from django.db import models, transaction
from django.utils import timezone

from Data.models import (
    DataIndustry,
    DataIndustryPrice,
    DataSector,
    DataSectorPrice,
    Stock,
    StockPrice,
)
from Data.services.provider import data_provider
from Data.services.synchronizer import data_synchronizer
from Data.services.yahoo_cache import yahoo_cache

logger = logging.getLogger(__name__)


class CompositeCache:
    """Intelligent caching system for composite data generation optimisation."""

    def __init__(self):
        self.price_data_cache = {}
        self.composite_cache = {}
        self.stock_date_index = {}

    def get_cache_key(self, stocks, start_date, end_date):
        """Generate a cache key for stock price data using BLAKE2b hashing."""
        if hasattr(stocks, "__iter__") and not isinstance(stocks, str):
            stock_ids = sorted([stock.id for stock in stocks])
        else:
            stock_ids = [stocks.id] if hasattr(stocks, "id") else [stocks]

        key_string = f"{stock_ids}_{start_date}_{end_date}"
        # Standardised digest_size=16 for consistency across codebase
        return hashlib.blake2b(key_string.encode(), digest_size=16).hexdigest()

    def get_composite_cache_key(self, entity, date):
        """Generate a cache key for composite calculations."""
        entity_type = entity.__class__.__name__
        entity_id = entity.id if hasattr(entity, "id") else str(entity)
        return f"{entity_type}_{entity_id}_{date}"

    def index_prices_by_stock_and_date(self, all_prices):
        """
        Phase 4: Index price data for fast lookup.
        Creates a nested dictionary: {stock: {date: price_record}}
        """
        if hasattr(self, "_indexed_prices") and self._indexed_prices:
            return self._indexed_prices

        prices_by_stock_date = defaultdict(dict)
        for price in all_prices:
            prices_by_stock_date[price.stock][price.date] = price

        self._indexed_prices = dict(prices_by_stock_date)
        return self._indexed_prices

    def get_cached_composite(self, cache_key):
        """Get cached composite data."""
        return self.composite_cache.get(cache_key)

    def cache_composite(self, cache_key, composite_data):
        """Cache composite calculation results."""
        self.composite_cache[cache_key] = composite_data

    def clear_cache(self):
        """Clear all cached data."""
        self.price_data_cache.clear()
        self.composite_cache.clear()
        self.stock_date_index.clear()
        if hasattr(self, "_indexed_prices"):
            del self._indexed_prices


class YahooFinanceService:
    """
    Main service class for Yahoo Finance API integration.
    Coordinates between provider, synchronizer, and database.
    """

    # Class-level constants for validation
    VALID_PERIODS = frozenset(["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    VALID_INTERVALS = frozenset(["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])

    def __init__(self):
        """Initialize the Yahoo Finance service."""
        self.provider = data_provider
        self.synchronizer = data_synchronizer
        self.cache = yahoo_cache
        self.timeout = 30  # Default timeout
        self.maxRetries = 5
        self.baseDelay = 2
        self.maxBackoff = 60

        # Use connection pooling for HTTP requests
        self.http_client = PooledHTTPClient()
        self.connection_pool = get_connection_pool()

        logger.info("Yahoo Finance Service initialized with connection pooling and multi-level caching")

    def get_connection_metrics(self):
        """Get connection pool metrics for monitoring."""
        return self.connection_pool.get_metrics()

    def close(self):
        """Close connections and cleanup resources."""
        # Connection pool will handle cleanup automatically
        logger.info("Yahoo Finance service closed")

    def __enter__(self):
        """Enter the context manager, returning the service instance."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, ensuring session cleanup."""
        self.close()
        # Don't suppress exceptions
        return False

    def _is_mock_or_test_symbol(self, symbol: str) -> bool:
        """
        Detect mock or test symbols to avoid unnecessary API calls.

        Args:
            symbol: Stock ticker symbol to check

        Returns:
            True if symbol appears to be for testing/mocking, False otherwise
        """
        # Convert to uppercase for consistent comparison
        symbol = symbol.upper()

        # Common test symbol patterns
        test_patterns = ["TEST", "MOCK", "FAKE", "DEMO", "SAMPLE", "INTEGRATION", "EMPTY_", "NULL_", "_RESILIENCE"]

        # Check if symbol matches any test patterns
        for pattern in test_patterns:
            if pattern in symbol:
                return True

        # Check for obviously fake symbols (all same character, too short, etc.)
        if len(symbol) < 1 or len(symbol) > 10:
            return True

        # Check for patterns like 'AAA', 'XXXX', etc.
        if len(set(symbol)) == 1 and len(symbol) > 2:
            return True

        return False

    # =====================================================================
    # Input validation methods (camelCase)
    # =====================================================================

    def validateSymbol(self, symbol: str) -> str:
        """Validate and sanitize stock symbol"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        symbol = symbol.strip().upper()
        # Allow only valid stock symbols
        if not re.match(r"^[A-Z0-9\.\-\^]{1,10}$", symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        return symbol

    def validatePeriod(self, period: str) -> str:
        """Validate period parameter"""
        if period not in self.VALID_PERIODS:
            raise ValueError(f"Invalid period: {period}. Must be one of {sorted(self.VALID_PERIODS)}")
        return period

    def validateInterval(self, interval: str) -> str:
        """Validate interval parameter"""
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {sorted(self.VALID_INTERVALS)}")
        return interval

    def validateDateRange(self, startDate: datetime, endDate: datetime) -> bool:
        """Validate date range parameters"""
        if not isinstance(startDate, datetime) or not isinstance(endDate, datetime):
            raise ValueError("Date parameters must be datetime objects")
        if endDate < startDate:
            raise ValueError("End date must be after start date")
        # Reasonable limit: no more than 10 years
        if (endDate - startDate).days > 3650:
            raise ValueError("Date range cannot exceed 10 years")
        return True

    def validateSymbolList(self, symbols: List[str]) -> List[str]:
        """Validate list of symbols"""
        if not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("Symbols must be a non-empty list")
        if len(symbols) > 100:  # Reasonable limit
            raise ValueError("Cannot process more than 100 symbols at once")

        validatedSymbols = []
        for symbol in symbols:
            validatedSymbols.append(self.validateSymbol(symbol))

        return validatedSymbols



    def get_stock_data(self, symbol: str, period: str = "1mo", sync_db: bool = True) -> Dict:
        """
        Fetch stock data for a given symbol using multi-level caching.

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            sync_db: Whether to sync data to database

        Returns:
            Dictionary containing stock data
        """
        try:
            # Validate inputs
            symbol = self.validateSymbol(symbol)
            period = self.validatePeriod(period)

            logger.info(f"Getting data for {symbol} with period {period}")

            # Use the multi-level caching service for history data
            cached_history = self.cache.get_stock_history(symbol, period)
            cached_info = self.cache.get_stock_info(symbol)
            
            if cached_history:
                stock_data = {
                    "symbol": symbol,
                    "period": period,
                    "info": cached_info or {},
                    "prices": [],
                    "volumes": [],
                    "dates": [],
                    "fetched_at": datetime.now().isoformat(),
                    "from_cache": True,
                }
                
                # Process the cached history data
                if cached_history.get("prices"):
                    for price_record in cached_history["prices"]:
                        if isinstance(price_record, dict):
                            stock_data["prices"].append(float(price_record.get("Close", 0)))
                            stock_data["volumes"].append(int(price_record.get("Volume", 0)))
                            date_val = price_record.get("Date")
                            if isinstance(date_val, str):
                                stock_data["dates"].append(date_val)
                            else:
                                stock_data["dates"].append(str(date_val))

                # Also sync to database if requested
                if sync_db:
                    try:
                        self.synchronizer.sync_stock_data(symbol, period)
                    except Exception as sync_error:
                        logger.warning(f"Database sync failed for {symbol}: {sync_error}")
                        
                return stock_data
            else:
                # Fallback to original database-centric approach
                if sync_db:
                    # Sync to database and return results
                    sync_result = self.synchronizer.sync_stock_data(symbol, period)

                    if sync_result["success"]:
                        # Fetch from database for consistent format
                        stock = Stock.objects.get(symbol=symbol.upper())
                        prices = StockPrice.objects.filter(stock=stock).order_by("-date")[:30]

                        stock_data = {
                            "symbol": stock.symbol,
                            "period": period,
                            "info": {
                                "shortName": stock.short_name,
                                "longName": stock.long_name,
                                "currency": stock.currency,
                                "exchange": stock.exchange,
                                "sector": stock.sector,
                                "industry": stock.industry,
                                "marketCap": stock.market_cap,
                            },
                            "prices": [float(p.close) for p in prices],
                            "volumes": [p.volume for p in prices],
                            "dates": [p.date.isoformat() for p in prices],
                            "fetched_at": datetime.now().isoformat(),
                            "from_cache": False,
                            "fallback_source": "database",
                        }
                    else:
                        stock_data = {"error": sync_result.get("error", "Sync failed")}
                else:
                    # Just fetch without syncing to database
                    stock_data = self.provider.fetch_stock_data(symbol, period)

                return stock_data

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return {"error": str(e)}


    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a stock using multi-level caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing stock information
        """
        try:
            # Validate input
            symbol = self.validateSymbol(symbol)

            logger.info(f"Fetching stock info for {symbol}")

            # Use the multi-level caching service
            cached_info = self.cache.get_stock_info(symbol)
            
            if cached_info:
                # Format the response consistently
                info = {
                    "symbol": symbol,
                    "shortName": cached_info.get("shortName"),
                    "longName": cached_info.get("longName"), 
                    "sector": cached_info.get("sector"),
                    "industry": cached_info.get("industry"),
                    "marketCap": cached_info.get("marketCap"),
                    "currency": cached_info.get("currency"),
                    "exchange": cached_info.get("exchange"),
                    "currentPrice": cached_info.get("currentPrice"),
                    "previousClose": cached_info.get("previousClose"),
                    "from_cache": True,
                    "fetched_at": datetime.now().isoformat(),
                }
                
                # Also try to sync to database for consistency
                try:
                    self.synchronizer.sync_stock_data(symbol)
                except Exception as sync_error:
                    logger.warning(f"Database sync failed for {symbol}: {sync_error}")
                    
                return info
            else:
                # Fallback to database if cache completely fails
                try:
                    stock = Stock.objects.get(symbol=symbol.upper())
                    latest_price = stock.get_latest_price()

                    info = {
                        "symbol": stock.symbol,
                        "shortName": stock.short_name,
                        "longName": stock.long_name,
                        "sector": stock.sector,
                        "industry": stock.industry,
                        "marketCap": stock.market_cap,
                        "currency": stock.currency,
                        "exchange": stock.exchange,
                        "currentPrice": float(latest_price.close) if latest_price else None,
                        "previousClose": float(latest_price.close) if latest_price else None,
                        "from_cache": True,
                        "fallback_source": "database",
                        "fetched_at": datetime.now().isoformat(),
                    }
                    return info

                except Stock.DoesNotExist:
                    logger.error(f"Stock {symbol} not found in cache or database")
                    return {"error": f"Stock {symbol} not found"}

        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {"error": str(e)}

    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """
        Get historical price data for a stock.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            Dictionary containing historical data
        """
        try:
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")

            # Check if we have data in database
            try:
                stock = Stock.objects.get(symbol=symbol.upper())
                prices = StockPrice.objects.filter(
                    stock=stock, date__gte=start_date.date(), date__lte=end_date.date()
                ).order_by("date")

                if prices.exists():
                    historical_data = {
                        "symbol": stock.symbol,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "data": [
                            {
                                "date": p.date.isoformat(),
                                "open": float(p.open),
                                "high": float(p.high),
                                "low": float(p.low),
                                "close": float(p.close),
                                "volume": p.volume,
                                "change_amount": float(p.daily_change) if p.daily_change else None,
                                "change_percent": float(p.daily_change_percent) if p.daily_change_percent else None,
                            }
                            for p in prices
                        ],
                        "from_cache": True,
                        "fetched_at": datetime.now().isoformat(),
                    }

                    return historical_data

            except Stock.DoesNotExist:
                pass

            # If not in database or no data, fetch from Yahoo Finance
            # Calculate period based on date range
            days_diff = (end_date - start_date).days
            if days_diff <= 5:
                period = "5d"
            elif days_diff <= 30:
                period = "1mo"
            elif days_diff <= 90:
                period = "3mo"
            elif days_diff <= 365:
                period = "1y"
            else:
                period = "2y"

            # Sync data and then return
            sync_result = self.synchronizer.sync_stock_data(symbol, period)

            if sync_result["success"]:
                # Fetch from database after sync
                stock = Stock.objects.get(symbol=symbol.upper())
                prices = StockPrice.objects.filter(
                    stock=stock, date__gte=start_date.date(), date__lte=end_date.date()
                ).order_by("date")

                historical_data = {
                    "symbol": stock.symbol,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "data": [
                        {
                            "date": p.date.isoformat(),
                            "open": float(p.open),
                            "high": float(p.high),
                            "low": float(p.low),
                            "close": float(p.close),
                            "volume": p.volume,
                        }
                        for p in prices
                    ],
                    "from_cache": False,
                    "fetched_at": datetime.now().isoformat(),
                }
            else:
                historical_data = {"error": sync_result.get("error", "Failed to fetch data")}

            return historical_data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return {"error": str(e)}

    def get_multiple_stocks(self, symbols: List[str], period: str = "1mo") -> Dict[str, Dict]:
        """
        Fetch data for multiple stocks.

        Args:
            symbols: List of stock ticker symbols
            period: Time period

        Returns:
            Dictionary with symbol as key and stock data as value
        """
        # Use synchronizer for batch operations
        sync_results = self.synchronizer.sync_multiple_stocks(symbols, period)

        results = {}
        for symbol in symbols:
            if symbol in sync_results["results"] and sync_results["results"][symbol]["success"]:
                results[symbol] = self.get_stock_data(symbol, period, sync_db=False)
            else:
                results[symbol] = {"error": "Failed to sync data"}

        return results

    def get_market_status(self) -> Dict:
        """
        Get current market status.

        Returns:
            Dictionary containing market status information
        """
        now = datetime.now()

        # Simple market hours check (9:30 AM - 4:00 PM EST, weekdays)
        is_weekday = now.weekday() < 5
        market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)

        is_open = is_weekday and market_open_time <= now <= market_close_time

        # Get some market indicators if available
        market_indicators = {}
        try:
            # Try to get major indices
            for symbol in ["^GSPC", "^DJI", "^IXIC"]:  # S&P 500, Dow Jones, NASDAQ
                try:
                    price = self.provider.fetch_realtime_price(symbol)
                    if price:
                        market_indicators[symbol] = price
                except Exception as e:
                    logger.warning(f"Failed to fetch price for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error retrieving market indicators: {e}")

        return {
            "is_open": is_open,
            "current_time": now.isoformat(),
            "timezone": "America/New_York",
            "market_hours": {"open": "09:30 EST", "close": "16:00 EST"},
            "indicators": market_indicators,
            "next_open": self._get_next_market_open(now),
        }

    def _get_next_market_open(self, current_time: datetime) -> str:
        """Get the next market open time."""
        next_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)

        # If it's already past market open today, move to next day
        if current_time.hour >= 9 and current_time.minute >= 30:
            next_open += timedelta(days=1)

        # Skip weekends
        while next_open.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_open += timedelta(days=1)

        return next_open.isoformat()

    def search_symbols(self, query: str) -> List[Dict]:
        """
        Search for stock symbols matching a query.

        Args:
            query: Search query

        Returns:
            List of matching symbols with information
        """
        try:
            logger.info(f"Searching for symbols matching: {query}")

            # First search in database
            stocks = Stock.objects.filter(
                models.Q(symbol__icontains=query)
                | models.Q(short_name__icontains=query)
                | models.Q(long_name__icontains=query)
            )[:10]

            results = []
            for stock in stocks:
                results.append(
                    {
                        "symbol": stock.symbol,
                        "name": stock.long_name or stock.short_name,
                        "type": "Stock",
                        "exchange": stock.exchange,
                        "sector": stock.sector,
                        "from_cache": True,
                    }
                )

            # If no results in database, validate the symbol with Yahoo Finance
            if not results and len(query) <= 5 and self.provider.validate_symbol(query):
                # Valid symbol, fetch and sync
                self.synchronizer.sync_stock_data(query)
                try:
                    stock = Stock.objects.get(symbol=query.upper())
                    results.append(
                        {
                            "symbol": stock.symbol,
                            "name": stock.long_name or stock.short_name,
                            "type": "Stock",
                            "exchange": stock.exchange,
                            "from_cache": False,
                        }
                    )
                except Stock.DoesNotExist:
                    pass

            return results

        except Exception as e:
            logger.error(f"Error searching symbols: {str(e)}")
            return []

    def get_realtime_quotes(self, symbols: List[str]) -> Dict:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary with real-time quote data
        """
        return self.synchronizer.update_realtime_prices(symbols)

    def validateSymbolExists(self, symbol: str) -> bool:
        """camelCase wrapper for validate_symbol"""
        return self.validate_symbol(symbol)

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists.

        Args:
            symbol: Stock ticker symbol

        Returns:
            True if valid, False otherwise
        """
        try:
            # First validate format
            symbol = self.validateSymbol(symbol)
            return self.provider.validate_symbol(symbol)
        except ValueError:
            return False

    def _retryWithBackoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff on rate limiting."""
        for attempt in range(self.maxRetries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < self.maxRetries - 1:
                        delay = min(self.baseDelay * (2**attempt) + random.uniform(0, 1), self.maxBackoff)
                        logger.warning(f"Rate limited, attempt {attempt + 1}/{self.maxRetries}, waiting {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        raise
                else:
                    raise
        raise Exception(f"Max retries ({self.maxRetries}) exceeded")

    def fetchBatchHistorical(
        self, tickers: List[str], period: str = "6mo", interval: str = "1d"
    ) -> Dict[str, List[Dict]]:
        """
        Fetch historical data for multiple tickers with rate limit handling.
        Process each ticker individually to avoid overwhelming the API.
        """
        results = {}

        for i, ticker in enumerate(tickers):
            logger.info(f"Fetching {ticker} ({i+1}/{len(tickers)}) - {period} @ {interval}")

            # Add delay between tickers to avoid rate limiting
            if i > 0:
                time.sleep(random.uniform(1, 2))

            try:
                data = self._retryWithBackoff(self._fetchSingleTicker, ticker, period, interval)
                results[ticker] = data
                logger.info(f"Successfully fetched {len(data)} bars for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {str(e)}")
                results[ticker] = []

        return results

    def fetchSingleHistorical(self, ticker: str, period: str = "1wk", interval: str = "1d") -> List[Dict]:
        """Fetch historical data for a single ticker."""
        logger.info(f"Fetching single ticker {ticker} - {period} @ {interval}")

        try:
            data = self._retryWithBackoff(self._fetchSingleTicker, ticker, period, interval)
            logger.info(f"Successfully fetched {len(data)} bars for {ticker}")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {str(e)}")
            return []

    def _validate_period_interval(self, period: str, interval: str) -> None:
        """
        Validate period and interval parameters for yf.download.

        Args:
            period: Time period for data (e.g., '1d', '5d', '1mo', etc.)
            interval: Data interval (e.g., '1m', '5m', '1d', etc.)

        Raises:
            ValueError: If period or interval values are invalid
        """
        # Validate using centralized validation functions
        period = self.validatePeriod(period)
        interval = self.validateInterval(interval)

        # Additional validation for period-interval combinations
        # Intraday intervals (< 1d) have restrictions on period
        intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        max_intraday_periods = {"1d", "5d", "1mo"}

        if interval in intraday_intervals and period not in max_intraday_periods:
            raise ValueError(
                f"Intraday interval '{interval}' can only be used with periods: {', '.join(sorted(max_intraday_periods))}"
            )

        logger.debug(f"Validated period='{period}' and interval='{interval}'")

    def _find_matching_ticker(self, columns: pd.MultiIndex, ticker: str) -> Optional[str]:
        """
        Find the matching ticker symbol in MultiIndex columns, handling case and format differences.

        Args:
            columns: MultiIndex columns from pandas DataFrame
            ticker: Original ticker symbol to match

        Returns:
            Matched ticker symbol from columns, or None if not found
        """
        if not isinstance(columns, pd.MultiIndex) or len(columns.levels) < 2:
            return None

        # Get all ticker symbols from the second level (assuming structure like ('Close', 'AAPL'))
        ticker_level = columns.levels[1]

        # Normalize the input ticker for comparison
        normalized_ticker = ticker.upper().strip()

        # Try exact match first
        if normalized_ticker in ticker_level:
            return normalized_ticker

        # Try case-insensitive matching
        for col_ticker in ticker_level:
            if col_ticker.upper().strip() == normalized_ticker:
                return col_ticker

        # Try partial matching for cases like 'BRK-B' vs 'BRK.B'
        for col_ticker in ticker_level:
            normalized_col = col_ticker.upper().strip().replace("-", ".").replace("_", ".")
            normalized_input = normalized_ticker.replace("-", ".").replace("_", ".")
            if normalized_col == normalized_input:
                return col_ticker

        logger.warning(f"Could not find matching ticker for '{ticker}' in MultiIndex levels: {list(ticker_level)}")
        return None

    def _safe_multiindex_access(self, row: pd.Series, column_type: str, ticker: str):
        """
        Safely access MultiIndex columns with fallback mechanisms.

        Args:
            row: Pandas Series row from DataFrame
            column_type: Type of column ('Close', 'Open', etc.)
            ticker: Ticker symbol

        Returns:
            Column value or None if not accessible
        """
        try:
            # Try different variations of the column access
            variations = [
                (column_type, ticker),
                (column_type, ticker.upper()),
                (column_type, ticker.lower()),
                (column_type.lower(), ticker),
                (column_type.lower(), ticker.upper()),
                (column_type.lower(), ticker.lower()),
            ]

            for col_tuple in variations:
                try:
                    if col_tuple in row.index:
                        return row[col_tuple]
                except (KeyError, TypeError):
                    continue

            # Last resort: try to find any column that contains the column_type
            for idx in row.index:
                if isinstance(idx, tuple) and len(idx) >= 2:
                    if isinstance(idx[0], str) and column_type.lower() in idx[0].lower():
                        return row[idx]

            logger.warning(f"Could not access {column_type} for ticker {ticker} in MultiIndex row")
            return None

        except Exception as e:
            logger.warning(f"Error accessing MultiIndex column {column_type} for {ticker}: {str(e)}")
            return None

    def _fetchSingleTicker(self, ticker: str, period: str, interval: str) -> List[Dict]:
        """Internal method to fetch data for one ticker."""
        try:
            # Validate period and interval parameters before calling yf.download
            self._validate_period_interval(period, interval)

            # Use yf.download without custom session to avoid curl_cffi conflicts
            # Let yfinance handle its own session management
            history = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,  # Preserve unadjusted close prices for consistency
                prepost=False,
                threads=False,
                progress=False,
            )

            if history.empty:
                logger.warning(f"No data returned for {ticker}")
                return []

            bars = []
            for date, row in history.iterrows():
                try:
                    # Handle MultiIndex columns with robust ticker matching
                    if isinstance(history.columns, pd.MultiIndex):
                        # Find the correct ticker symbol in MultiIndex levels
                        matched_ticker = self._find_matching_ticker(history.columns, ticker)

                        if matched_ticker:
                            # Use the matched ticker for column access
                            close_val = row[("Close", matched_ticker)]
                            open_val = row[("Open", matched_ticker)]
                            high_val = row[("High", matched_ticker)]
                            low_val = row[("Low", matched_ticker)]
                            volume_val = row[("Volume", matched_ticker)]
                        else:
                            # Fallback: try to find columns by pattern matching
                            close_val = self._safe_multiindex_access(row, "Close", ticker)
                            open_val = self._safe_multiindex_access(row, "Open", ticker)
                            high_val = self._safe_multiindex_access(row, "High", ticker)
                            low_val = self._safe_multiindex_access(row, "Low", ticker)
                            volume_val = self._safe_multiindex_access(row, "Volume", ticker)
                    else:
                        # Simple column names
                        close_val = row["Close"]
                        open_val = row["Open"]
                        high_val = row["High"]
                        low_val = row["Low"]
                        volume_val = row["Volume"]

                    # Skip if no valid close price
                    if pd.isna(close_val):
                        continue

                    # Convert to appropriate types, handling NaN values
                    open_val = float(open_val) if pd.notna(open_val) else float(close_val)
                    high_val = float(high_val) if pd.notna(high_val) else float(close_val)
                    low_val = float(low_val) if pd.notna(low_val) else float(close_val)
                    close_val = float(close_val)
                    volume_val = int(volume_val) if pd.notna(volume_val) else 0

                    # Handle timezone - check if date has timezone info
                    if hasattr(date, "tzinfo") and date.tzinfo is not None:
                        date_utc = date.tz_convert("UTC")
                    else:
                        date_utc = date.replace(tzinfo=dt_timezone.utc)

                    bars.append(
                        {
                            "symbol": ticker.upper(),
                            "date": date_utc,
                            "open": Decimal(str(round(open_val, 2))),
                            "high": Decimal(str(round(high_val, 2))),
                            "low": Decimal(str(round(low_val, 2))),
                            "close": Decimal(str(round(close_val, 2))),
                            "volume": volume_val,
                            "interval": interval,
                            "data_source": "yahoo",
                        }
                    )
                except Exception as e:
                    logger.warning(f"Skipping row for {ticker} due to: {str(e)}")
                    continue

            return bars
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return []

    def normalizeBars(self, bars: List[Dict]) -> List[Dict]:
        """Normalize bar data for database insertion."""
        # Already normalized in _fetchSingleTicker
        return bars

    # =====================================================================
    # Sector/Industry Data Fetching Methods
    # =====================================================================

    def fetchSectorIndustryBatch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch sector/industry data for multiple symbols with rate limit handling.
        Enhanced for technical analysis with normalized classification data.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary with symbol as key and normalized classification data as value
        """
        try:
            # Validate symbols
            validatedSymbols = self.validateSymbolList(symbols)
            logger.info(f"Fetching sector/industry for {len(validatedSymbols)} symbols")

            # Check for recent data and filter what needs fetching
            symbolsToFetch = self.getStaleAndMissingSymbols(validatedSymbols)

            if not symbolsToFetch:
                logger.info("All symbols have recent sector/industry data, skipping fetch")
                return {}

            logger.info(f"Fetching sector/industry for {len(symbolsToFetch)} symbols: {symbolsToFetch}")

            results = {}

            for i, symbol in enumerate(symbolsToFetch):
                logger.info(f"Fetching {symbol} sector/industry ({i+1}/{len(symbolsToFetch)})")

                # Add delay between symbols to avoid rate limiting
                if i > 0:
                    time.sleep(random.uniform(1, 2))

                try:
                    data = self._retryWithBackoff(self.fetchSectorIndustrySingle, symbol)
                    if data and "error" not in data:
                        # Enhance with normalized keys for classification upsert
                        enhanced_data = {
                            "symbol": data["symbol"],
                            "sectorKey": self._normalizeSectorKey(data.get("sector", "")),
                            "sectorName": data.get("sector", ""),
                            "industryKey": self._normalizeIndustryKey(data.get("industry", "")),
                            "industryName": data.get("industry", ""),
                            "updatedAt": data.get("updatedAt"),
                        }
                        results[symbol] = enhanced_data
                        logger.info(f"Successfully fetched sector/industry for {symbol}")
                    else:
                        logger.warning(f"No sector/industry data for {symbol}")
                        results[symbol] = {"error": "No data available"}

                except Exception as e:
                    logger.error(f"Failed to fetch sector/industry for {symbol}: {str(e)}")
                    results[symbol] = {"error": str(e)}

            return results

        except Exception as e:
            logger.error(f"Error in fetchSectorIndustryBatch: {str(e)}")
            return {}

    def fetchSectorIndustrySingle(self, symbol: str) -> Dict:
        """
        Fetch sector/industry data for a single symbol.
        Enhanced for technical analysis with normalized classification data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing sector/industry data with normalized keys
        """
        try:
            # Validate symbol
            symbol = self.validateSymbol(symbol)

            logger.info(f"Fetching sector/industry data for {symbol}")

            # Create yfinance Ticker object without custom session to avoid curl_cffi conflicts
            # Let yfinance handle its own session management
            ticker = yf.Ticker(symbol)

            # Get company info which contains sector/industry
            info = ticker.info

            if not info or len(info) < 2:  # Minimal check for valid response
                logger.warning(f"No info data returned for {symbol}")
                return {"error": "No company information available"}

            # Extract sector and industry
            sector = info.get("sector", "").strip()
            industry = info.get("industry", "").strip()

            if not sector and not industry:
                logger.warning(f"No sector/industry data found for {symbol}")
                return {"error": "No sector/industry data available"}

            result = {
                "symbol": symbol.upper(),
                "sector": sector or "",
                "industry": industry or "",
                "sectorKey": self._normalizeSectorKey(sector),
                "sectorName": sector,
                "industryKey": self._normalizeIndustryKey(industry),
                "industryName": industry,
                "updatedAt": timezone.now(),
            }

            logger.info(f"Retrieved sector/industry for {symbol}: {sector} / {industry}")
            return result

        except Exception as e:
            logger.error(f"Error fetching sector/industry for {symbol}: {str(e)}")
            return {"error": str(e)}

    def getStaleAndMissingSymbols(self, symbols: List[str]) -> List[str]:
        """
        Identify symbols that need sector/industry data fetching.

        Args:
            symbols: List of symbols to check

        Returns:
            List of symbols that need fetching (stale + missing)
        """
        from datetime import timedelta

        # 3 years threshold
        threshold_date = timezone.now() - timedelta(days=3 * 365)

        # Query existing stocks
        existing_stocks = Stock.objects.filter(symbol__in=symbols)

        staleSymbols = []
        recentSymbols = []

        existing_symbols_set = set()

        for stock in existing_stocks:
            existing_symbols_set.add(stock.symbol)

            if not stock.sectorUpdatedAt or stock.sectorUpdatedAt < threshold_date:
                staleSymbols.append(stock.symbol)
            else:
                recentSymbols.append(stock.symbol)

        # Find missing symbols
        missingSymbols = [s for s in symbols if s not in existing_symbols_set]

        logger.info(
            f"Symbol analysis: {len(recentSymbols)} recent, {len(staleSymbols)} stale, {len(missingSymbols)} missing"
        )

        return staleSymbols + missingSymbols

    def upsertCompanyProfiles(self, profiles: Dict[str, Dict]) -> Tuple[int, int, int]:
        """
        Upsert sector/industry data into Stock model.

        Args:
            profiles: Dictionary with symbol as key and profile data as value

        Returns:
            Tuple of (created_count, updated_count, skipped_count)
        """
        if not profiles:
            return 0, 0, 0

        created_count = 0
        updated_count = 0
        skipped_count = 0

        for symbol, profile_data in profiles.items():
            if "error" in profile_data:
                skipped_count += 1
                continue

            try:
                sector_str = profile_data.get("sector", "")
                industry_str = profile_data.get("industry", "")

                stock, created = Stock.objects.get_or_create(
                    symbol=symbol,
                    defaults={
                        "short_name": symbol,
                        "sector": sector_str,
                        "industry": industry_str,
                        "sectorUpdatedAt": profile_data.get("updatedAt"),
                        "data_source": "yahoo",
                    },
                )

                if created:
                    created_count += 1
                    # Set foreign keys for newly created stock
                    sector_obj, industry_obj = self._mapSectorIndustryToForeignKeys(sector_str, industry_str)
                    if sector_obj and industry_obj:
                        stock.sector_id = sector_obj
                        stock.industry_id = industry_obj
                        stock.save(update_fields=["sector_id", "industry_id"])
                        logger.info(
                            f"Created new stock record for {symbol} with foreign keys: {sector_obj.sectorKey}/{industry_obj.industryKey}"
                        )
                    else:
                        logger.info(f"Created new stock record for {symbol} (foreign key mapping failed)")
                else:
                    # Update existing record with both string fields and foreign keys
                    sector_str = profile_data.get("sector", "")
                    industry_str = profile_data.get("industry", "")

                    stock.sector = sector_str
                    stock.industry = industry_str
                    stock.sectorUpdatedAt = profile_data.get("updatedAt")

                    # Map to foreign keys
                    sector_obj, industry_obj = self._mapSectorIndustryToForeignKeys(sector_str, industry_str)
                    if sector_obj and industry_obj:
                        stock.sector_id = sector_obj
                        stock.industry_id = industry_obj
                        stock.save(update_fields=["sector", "industry", "sector_id", "industry_id", "sectorUpdatedAt"])
                        logger.info(
                            f"Updated sector/industry with foreign keys for {symbol}: {sector_obj.sectorKey}/{industry_obj.industryKey}"
                        )
                    else:
                        stock.save(update_fields=["sector", "industry", "sectorUpdatedAt"])
                        logger.warning(
                            f"Updated sector/industry strings only for {symbol} (foreign key mapping failed)"
                        )

                    updated_count += 1

            except Exception as e:
                logger.error(f"Error upserting profile for {symbol}: {str(e)}")
                skipped_count += 1

        logger.info(f"Upsert complete: {created_count} created, {updated_count} updated, {skipped_count} skipped")
        return created_count, updated_count, skipped_count

    # =====================================================================
    # Technical Analysis Enhancement Methods
    # =====================================================================

    def upsertClassification(self, rows: List[Dict]) -> Tuple[int, int, int]:
        """
        Create/update DataSector and DataIndustry records and establish FK relationships.

        Args:
            rows: List of classification dictionaries with keys:
                  symbol, sectorKey, sectorName, industryKey, industryName, updatedAt

        Returns:
            Tuple of (created_count, updated_count, skipped_count)
        """
        if not rows:
            return 0, 0, 0

        created_count = 0
        updated_count = 0
        skipped_count = 0

        with transaction.atomic():
            for row in rows:
                try:
                    symbol = row.get("symbol", "").upper()
                    sector_key = self._normalizeSectorKey(row.get("sectorKey", ""))
                    sector_name = row.get("sectorName", "").strip()
                    industry_key = self._normalizeIndustryKey(row.get("industryKey", ""))
                    industry_name = row.get("industryName", "").strip()
                    updated_at = row.get("updatedAt")

                    if not symbol or not sector_key or not industry_key:
                        skipped_count += 1
                        continue

                    # Create or get sector
                    sector, sector_created = DataSector.objects.get_or_create(
                        sectorKey=sector_key,
                        defaults={"sectorName": sector_name, "last_sync": updated_at, "data_source": "yahoo"},
                    )

                    if not sector_created and sector.sectorName != sector_name:
                        sector.sectorName = sector_name
                        sector.last_sync = updated_at
                        sector.save(update_fields=["sectorName", "last_sync"])

                    # Create or get industry
                    industry, industry_created = DataIndustry.objects.get_or_create(
                        industryKey=industry_key,
                        defaults={
                            "industryName": industry_name,
                            "sector": sector,
                            "last_sync": updated_at,
                            "data_source": "yahoo",
                        },
                    )

                    if not industry_created and (industry.industryName != industry_name or industry.sector != sector):
                        industry.industryName = industry_name
                        industry.sector = sector
                        industry.last_sync = updated_at
                        industry.save(update_fields=["industryName", "sector", "last_sync"])

                    # Update Stock model with FK relationships
                    try:
                        stock = Stock.objects.get(symbol=symbol)
                        if stock.sector_id != sector or stock.industry_id != industry:
                            stock.sector_id = sector
                            stock.industry_id = industry
                            stock.sectorUpdatedAt = updated_at
                            stock.save(update_fields=["sector_id", "industry_id", "sectorUpdatedAt"])
                            updated_count += 1
                    except Stock.DoesNotExist:
                        # Create stock if it doesn't exist
                        Stock.objects.create(
                            symbol=symbol,
                            short_name=symbol,
                            sector_id=sector,
                            industry_id=industry,
                            sectorUpdatedAt=updated_at,
                            data_source="yahoo",
                        )
                        created_count += 1

                except Exception as e:
                    logger.error(f"Error upserting classification for {row.get('symbol', 'unknown')}: {str(e)}")
                    skipped_count += 1

        logger.info(
            f"Classification upsert complete: {created_count} created, {updated_count} updated, {skipped_count} skipped"
        )
        return created_count, updated_count, skipped_count

    def _normalizeSectorKey(self, sector: str) -> str:
        """Normalize sector name to consistent key format."""
        if not sector:
            return ""
        # Replace non-alphanumeric with underscores, then collapse multiple underscores
        normalized = re.sub(r"[^a-zA-Z0-9]", "_", sector.lower().strip())
        # Collapse multiple underscores into single underscore
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def _normalizeIndustryKey(self, industry: str) -> str:
        """Normalize industry name to consistent key format."""
        if not industry:
            return ""
        # Replace non-alphanumeric with underscores, then collapse multiple underscores
        normalized = re.sub(r"[^a-zA-Z0-9]", "_", industry.lower().strip())
        # Collapse multiple underscores into single underscore
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def _mapSectorIndustryToForeignKeys(
        self, sector_str: str, industry_str: str
    ) -> Tuple[Optional["DataSector"], Optional["DataIndustry"]]:
        """
        Map sector/industry string values to DataSector/DataIndustry foreign key objects.
        Creates new records if they don't exist.

        Args:
            sector_str: Sector name from Yahoo Finance
            industry_str: Industry name from Yahoo Finance

        Returns:
            Tuple of (DataSector object or None, DataIndustry object or None)
        """
        from django.db import transaction

        from Data.models import DataIndustry, DataSector

        if not sector_str or not industry_str:
            return None, None

        # Normalize the keys
        sector_key = self._normalizeSectorKey(sector_str)
        industry_key = self._normalizeIndustryKey(industry_str)

        if not sector_key or not industry_key:
            return None, None

        try:
            with transaction.atomic():
                # Get or create DataSector
                sector_obj, sector_created = DataSector.objects.get_or_create(
                    sectorKey=sector_key,
                    defaults={
                        "sectorName": sector_str.strip(),
                        "isActive": True,
                        "data_source": "yahoo",
                        "last_sync": timezone.now(),
                    },
                )

                if sector_created:
                    logger.info(f"Created new DataSector: {sector_key} ({sector_str})")

                # Get or create DataIndustry
                industry_obj, industry_created = DataIndustry.objects.get_or_create(
                    industryKey=industry_key,
                    defaults={
                        "industryName": industry_str.strip(),
                        "sector": sector_obj,
                        "isActive": True,
                        "data_source": "yahoo",
                        "last_sync": timezone.now(),
                    },
                )

                if industry_created:
                    logger.info(f"Created new DataIndustry: {industry_key} ({industry_str}) under sector {sector_key}")
                elif industry_obj.sector != sector_obj:
                    # Update industry's sector if it has changed
                    industry_obj.sector = sector_obj
                    industry_obj.last_sync = timezone.now()
                    industry_obj.save(update_fields=["sector", "last_sync"])
                    logger.info(f"Updated DataIndustry {industry_key} sector mapping to {sector_key}")

                return sector_obj, industry_obj

        except Exception as e:
            logger.error(f"Error mapping sector/industry to foreign keys: {sector_str}/{industry_str} -> {str(e)}")
            return None, None

    def fetchStockEodHistory(self, symbol: str, startDate: datetime, endDate: datetime) -> List[Dict]:
        """
        Fetch EOD stock price history for the specified date range.

        Args:
            symbol: Stock ticker symbol
            startDate: Start date for historical data
            endDate: End date for historical data

        Returns:
            List of EOD price dictionaries
        """
        try:
            # Validate inputs
            symbol = self.validateSymbol(symbol)
            self.validateDateRange(startDate, endDate)

            logger.info(f"Fetching EOD history for {symbol} from {startDate.date()} to {endDate.date()}")

            # Use existing historical data method but force EOD interval
            historical_data = self.get_historical_data(symbol, startDate, endDate)

            if "error" in historical_data:
                return []

            # Convert to the expected format with adjusted_close
            eod_data = []
            for item in historical_data.get("data", []):
                eod_data.append(
                    {
                        "symbol": symbol.upper(),
                        "date": datetime.fromisoformat(item["date"]).date(),
                        "open": Decimal(str(item["open"])),
                        "high": Decimal(str(item["high"])),
                        "low": Decimal(str(item["low"])),
                        "close": Decimal(str(item["close"])),
                        "adjusted_close": Decimal(str(item.get("adjusted_close", item["close"]))),
                        "volume": item["volume"],
                        "data_source": "yahoo",
                    }
                )

            logger.info(f"Retrieved {len(eod_data)} EOD records for {symbol}")

            # Limit to maximum expected trading days to prevent excessive data
            calendar_days = (endDate - startDate).days
            max_expected_trading_days = int(calendar_days * 0.690420)
            min_threshold = int(max_expected_trading_days * 0.95)

            if len(eod_data) > max_expected_trading_days * 1.05:
                trim_to = int(max_expected_trading_days * 1.05)
                logger.warning(f"Trimming {len(eod_data)} records to {trim_to} expected trading days")
                eod_data = eod_data[-trim_to:]  # Keep most recent data
                logger.info(f"Trimmed to {len(eod_data)} EOD records for {symbol}")
            elif len(eod_data) < min_threshold:
                logger.warning(f"Data count {len(eod_data)} is below minimum threshold of {min_threshold} trading days")

            return eod_data

        except Exception as e:
            logger.error(f"Error fetching EOD history for {symbol}: {str(e)}")
            return []

    def composeSectorIndustryEod(self, date_range: Tuple[datetime, datetime]) -> Dict[str, int]:
        """
        OPTIMIZED: Combined Phases 1, 2, 4 implementation.
        Calculate and persist sector/industry EOD composites for the given date range.

        Optimizations:
        - Phase 2: Unified data fetching (single query for all stocks)
        - Phase 1: Bulk database operations (bulk_create/bulk_update)
        - Phase 4: Smart caching for repeated calculations

        Args:
            date_range: Tuple of (start_date, end_date)

        Returns:
            Dictionary with counts of created sector and industry price records
        """
        start_date, end_date = date_range

        try:
            logger.info(f"Computing sector/industry composites from {start_date.date()} to {end_date.date()}")

            # Phase 4: Initialize cache
            cache = CompositeCache()

            # Phase 2: Get all active sectors and industries with associated stocks
            sectors = DataSector.objects.filter(isActive=True, stocks__isnull=False).distinct()

            industries = DataIndustry.objects.filter(isActive=True, stocks__isnull=False).distinct()

            # Phase 2: Collect all stocks across sectors and industries (unified approach)
            all_stocks_set = set()
            sector_stock_mapping = {}
            industry_stock_mapping = {}

            logger.info(f"Mapping stocks for {len(sectors)} sectors and {len(industries)} industries")

            for sector in sectors:
                sector_stocks = list(
                    Stock.objects.filter(
                        sector_id=sector,
                        is_active=True,
                        prices__date__gte=start_date.date(),
                        prices__date__lte=end_date.date(),
                    ).distinct()
                )

                if sector_stocks:
                    sector_stock_mapping[sector.id] = sector_stocks
                    all_stocks_set.update(sector_stocks)

            for industry in industries:
                industry_stocks = list(
                    Stock.objects.filter(
                        industry_id=industry,
                        is_active=True,
                        prices__date__gte=start_date.date(),
                        prices__date__lte=end_date.date(),
                    ).distinct()
                )

                if industry_stocks:
                    industry_stock_mapping[industry.id] = industry_stocks
                    all_stocks_set.update(industry_stocks)

            # Phase 2: Single unified data fetch for all stocks (major optimization)
            logger.info(f"Fetching unified price data for {len(all_stocks_set)} unique stocks")
            all_prices = (
                StockPrice.objects.filter(
                    stock__in=all_stocks_set, date__gte=start_date.date(), date__lte=end_date.date()
                )
                .select_related("stock")
                .order_by("date", "stock")
            )

            # Phase 4: Cache the unified data for fast lookup
            prices_by_stock_date = cache.index_prices_by_stock_and_date(all_prices)
            price_dates = sorted(set(price.date for price in all_prices))
            logger.info(f"Indexed {len(all_prices)} price records across {len(price_dates)} dates")

            sector_prices_created = 0
            industry_prices_created = 0

            # PHASE 3: Parallel Processing with ThreadPoolExecutor
            # Determine optimal worker count (conservative approach)
            total_entities = len([s for s in sectors if s.id in sector_stock_mapping]) + len(
                [i for i in industries if i.id in industry_stock_mapping]
            )

            max_workers = min(4, max(1, total_entities // 2))  # Conservative threading
            logger.info(f"Using parallel processing with {max_workers} workers for {total_entities} entities")

            # Process sectors in parallel
            if sector_stock_mapping:
                sector_tasks = []
                for sector in sectors:
                    if sector.id in sector_stock_mapping:
                        sector_tasks.append({"entity": sector, "stocks": sector_stock_mapping[sector.id]})

                if sector_tasks:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all sector tasks
                        future_to_entity = {
                            executor.submit(
                                self._processEntityParallel,
                                task_data,
                                "sector",
                                prices_by_stock_date,
                                price_dates,
                                cache,
                            ): task_data["entity"].id
                            for task_data in sector_tasks
                        }

                        # Collect results as they complete
                        for future in as_completed(future_to_entity):
                            entity_name, created = future.result()
                            sector_prices_created += created
                            logger.debug(f"Sector task completed: {entity_name} -> {created} records")

            # Process industries in parallel
            if industry_stock_mapping:
                industry_tasks = []
                for industry in industries:
                    if industry.id in industry_stock_mapping:
                        industry_tasks.append({"entity": industry, "stocks": industry_stock_mapping[industry.id]})

                if industry_tasks:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all industry tasks
                        future_to_entity = {
                            executor.submit(
                                self._processEntityParallel,
                                task_data,
                                "industry",
                                prices_by_stock_date,
                                price_dates,
                                cache,
                            ): task_data["entity"].id
                            for task_data in industry_tasks
                        }

                        # Collect results as they complete
                        for future in as_completed(future_to_entity):
                            entity_name, created = future.result()
                            industry_prices_created += created
                            logger.debug(f"Industry task completed: {entity_name} -> {created} records")

            # Phase 4: Clean up cache
            cache.clear_cache()

            logger.info(
                f"Composite creation complete: {sector_prices_created} sector prices, {industry_prices_created} industry prices"
            )

            return {"sector_prices_created": sector_prices_created, "industry_prices_created": industry_prices_created}

        except Exception as e:
            logger.error(f"Error creating sector/industry composites: {str(e)}")
            return {"sector_prices_created": 0, "industry_prices_created": 0}

    def _createCompositesUnifiedBulk(
        self,
        entity,
        entity_type: str,
        stocks: List,
        prices_by_stock_date: Dict,
        price_dates: List,
        cache: CompositeCache,
    ) -> int:
        """
        COMBINED PHASES 1, 2, 4: Create composite records with unified data and bulk operations.

        Args:
            entity: DataSector or DataIndustry instance
            entity_type: 'sector' or 'industry'
            stocks: List of Stock instances for this entity
            prices_by_stock_date: Pre-fetched price data {stock: {date: price}}
            price_dates: List of dates to process
            cache: CompositeCache instance for optimization

        Returns:
            Number of composite records created/updated
        """
        try:
            # Phase 2: Extract relevant data from unified dataset
            entity_prices_by_date = {}
            for date in price_dates:
                daily_prices = []
                for stock in stocks:
                    if stock in prices_by_stock_date and date in prices_by_stock_date[stock]:
                        daily_prices.append(prices_by_stock_date[stock][date])

                if daily_prices:
                    entity_prices_by_date[date] = daily_prices

            if not entity_prices_by_date:
                return 0

            # Phase 1: Bulk existence check
            model_class = DataSectorPrice if entity_type == "sector" else DataIndustryPrice
            entity_field = "sector" if entity_type == "sector" else "industry"

            # Phase 1: Get existing records for this entity
            existing_records_qs = model_class.objects.filter(
                **{entity_field: entity}, date__in=list(entity_prices_by_date.keys())
            )
            existing_records = {record.date: record for record in existing_records_qs}

            # Phase 1 + 4: Calculate composites with caching and prepare bulk operations
            new_records = []
            update_records = []

            for date in sorted(entity_prices_by_date.keys()):
                daily_prices = entity_prices_by_date[date]

                # Phase 4: Check cache first
                cache_key = cache.get_composite_cache_key(entity, date)
                composite_data = cache.get_cached_composite(cache_key)

                if not composite_data:
                    # Calculate and cache
                    composite_data = self._calculateComposite(daily_prices)
                    if composite_data:
                        cache.cache_composite(cache_key, composite_data)

                if not composite_data:
                    continue

                # Phase 1: Bulk record preparation
                if date in existing_records:
                    # Update existing record if needed
                    record = existing_records[date]
                    if record.constituents_count != composite_data["constituents_count"]:
                        for key, value in composite_data.items():
                            setattr(record, key, value)
                        update_records.append(record)
                else:
                    # Prepare new record
                    record_data = {entity_field: entity, "date": date}
                    record_data.update(composite_data)
                    new_records.append(model_class(**record_data))

            # Phase 1: Bulk database operations
            created_count = 0

            if new_records:
                try:
                    model_class.objects.bulk_create(new_records, batch_size=100, ignore_conflicts=True)
                    created_count += len(new_records)
                    logger.debug(f"Bulk created {len(new_records)} {entity_type} composite records")
                except Exception as e:
                    logger.warning(f"Bulk create failed, falling back to individual creates: {str(e)}")
                    # Fallback to individual creates
                    for record in new_records:
                        try:
                            record.save()
                            created_count += 1
                        except Exception:
                            pass  # Skip conflicts

            if update_records:
                try:
                    updated_fields = ["close_index", "constituents_count"]
                    model_class.objects.bulk_update(update_records, updated_fields, batch_size=100)
                    created_count += len(update_records)
                    logger.debug(f"Bulk updated {len(update_records)} {entity_type} composite records")
                except Exception as e:
                    logger.warning(f"Bulk update failed, falling back to individual updates: {str(e)}")
                    # Fallback to individual updates
                    for record in update_records:
                        try:
                            record.save()
                            created_count += 1
                        except Exception:
                            pass  # Skip errors

            return created_count

        except Exception as e:
            logger.error(f"Error creating {entity_type} composites for {entity}: {str(e)}")
            return 0

    def _processEntityParallel(
        self, entity_data: Dict, entity_type: str, prices_by_stock_date: Dict, price_dates: List, cache: CompositeCache
    ) -> Tuple[str, int]:
        """
        PHASE 3: Thread-safe parallel processing for a single entity.

        Args:
            entity_data: Dict with 'entity' and 'stocks' keys
            entity_type: 'sector' or 'industry'
            prices_by_stock_date: Pre-fetched price data
            price_dates: List of dates to process
            cache: CompositeCache instance

        Returns:
            Tuple of (entity_name, created_count) for result aggregation
        """
        from django.db import connection

        entity_name = "unknown"
        try:
            entity = entity_data["entity"]
            stocks = entity_data["stocks"]
            entity_name = f"{entity_type}:{entity.id}"

            # Ensure fresh database connection for this thread
            connection.ensure_connection()

            created = self._createCompositesUnifiedBulk(
                entity=entity,
                entity_type=entity_type,
                stocks=stocks,
                prices_by_stock_date=prices_by_stock_date,
                price_dates=price_dates,
                cache=cache,
            )

            logger.debug(f"Parallel {entity_type} processing complete: {entity_name} -> {created} records")
            return (entity_name, created)

        except Exception as e:
            logger.error(f"Error in parallel {entity_type} processing for {entity_name}: {str(e)}")
            # Return 0 but don't fail the entire process
            return (entity_name, 0)
        finally:
            # Close connection for this thread to prevent connection leaks
            try:
                connection.close()
            except Exception:
                pass  # Ignore connection close errors

    def _createSectorComposites(self, sector: "DataSector", start_date: datetime, end_date: datetime) -> int:
        """Create composite price records for a sector."""
        try:
            # Get all stocks in this sector with price data
            stocks = Stock.objects.filter(
                sector_id=sector, is_active=True, prices__date__gte=start_date.date(), prices__date__lte=end_date.date()
            ).distinct()

            if not stocks.exists():
                return 0

            # Get unique dates from stock prices in the date range
            price_dates = (
                StockPrice.objects.filter(stock__in=stocks, date__gte=start_date.date(), date__lte=end_date.date())
                .values_list("date", flat=True)
                .distinct()
                .order_by("date")
            )

            prices_created = 0

            # Bulk fetch all price data for all dates at once (optimization fix)
            all_prices = (
                StockPrice.objects.filter(stock__in=stocks, date__in=price_dates)
                .select_related("stock")
                .order_by("date", "stock")
            )

            # Group prices by date
            prices_by_date = {}
            for price in all_prices:
                if price.date not in prices_by_date:
                    prices_by_date[price.date] = []
                prices_by_date[price.date].append(price)

            for date in price_dates:
                daily_prices = prices_by_date.get(date, [])

                if not daily_prices:
                    continue

                # Calculate composite
                composite_data = self._calculateComposite(daily_prices)

                if composite_data:
                    # Create or update sector price record
                    sector_price, created = DataSectorPrice.objects.get_or_create(
                        sector=sector, date=date, defaults=composite_data
                    )

                    if created:
                        prices_created += 1
                    elif sector_price.constituents_count != composite_data["constituents_count"]:
                        # Update if constituent count changed
                        for key, value in composite_data.items():
                            setattr(sector_price, key, value)
                        sector_price.save()

            return prices_created

        except Exception as e:
            logger.error(f"Error creating sector composites for {sector.sectorName}: {str(e)}")
            return 0

    def _createIndustryComposites(self, industry: "DataIndustry", start_date: datetime, end_date: datetime) -> int:
        """Create composite price records for an industry."""
        try:
            # Get all stocks in this industry with price data
            stocks = Stock.objects.filter(
                industry_id=industry,
                is_active=True,
                prices__date__gte=start_date.date(),
                prices__date__lte=end_date.date(),
            ).distinct()

            if not stocks.exists():
                return 0

            # Get unique dates from stock prices in the date range
            price_dates = (
                StockPrice.objects.filter(stock__in=stocks, date__gte=start_date.date(), date__lte=end_date.date())
                .values_list("date", flat=True)
                .distinct()
                .order_by("date")
            )

            prices_created = 0

            # Bulk fetch all price data for all dates at once (optimization fix)
            all_prices = (
                StockPrice.objects.filter(stock__in=stocks, date__in=price_dates)
                .select_related("stock")
                .order_by("date", "stock")
            )

            # Group prices by date
            prices_by_date = {}
            for price in all_prices:
                if price.date not in prices_by_date:
                    prices_by_date[price.date] = []
                prices_by_date[price.date].append(price)

            for date in price_dates:
                daily_prices = prices_by_date.get(date, [])

                if not daily_prices:
                    continue

                # Calculate composite
                composite_data = self._calculateComposite(daily_prices)

                if composite_data:
                    # Create or update industry price record
                    industry_price, created = DataIndustryPrice.objects.get_or_create(
                        industry=industry, date=date, defaults=composite_data
                    )

                    if created:
                        prices_created += 1
                    elif industry_price.constituents_count != composite_data["constituents_count"]:
                        # Update if constituent count changed
                        for key, value in composite_data.items():
                            setattr(industry_price, key, value)
                        industry_price.save()

            return prices_created

        except Exception as e:
            logger.error(f"Error creating industry composites for {industry.industryName}: {str(e)}")
            return 0

    def _calculateComposite(self, daily_prices) -> Optional[Dict]:
        """Calculate cap-weighted composite for a set of daily prices."""
        try:
            total_weighted_price = Decimal("0")
            total_market_cap = Decimal("0")
            total_volume = 0
            constituent_count = 0

            # Try cap-weighted first
            for price in daily_prices:
                market_cap = price.stock.market_cap
                adjusted_close = price.adjusted_close or price.close

                if market_cap and market_cap > 0:
                    weight = Decimal(str(market_cap))
                    total_weighted_price += adjusted_close * weight
                    total_market_cap += weight
                    total_volume += price.volume
                    constituent_count += 1

            # If we have cap-weighted data
            if total_market_cap > 0:
                close_index = total_weighted_price / total_market_cap
                method = "cap_weighted"
            else:
                # Fallback to equal-weighted
                total_price = Decimal("0")
                constituent_count = 0

                for price in daily_prices:
                    adjusted_close = price.adjusted_close or price.close
                    total_price += adjusted_close
                    total_volume += price.volume
                    constituent_count += 1

                if constituent_count > 0:
                    close_index = total_price / Decimal(str(constituent_count))
                    method = "equal_weighted"
                else:
                    return None

            return {
                "close_index": close_index,
                "volume_agg": total_volume,
                "constituents_count": constituent_count,
                "method": method,
                "data_source": "yahoo",
            }

        except Exception as e:
            logger.error(f"Error calculating composite: {str(e)}")
            return None

    # =====================================================================
    # Required Service Functions for AAPL 3Y Import
    # =====================================================================

    def fetchQuoteSingle(self, symbol: str) -> Dict:
        """
        Fetch quote data for a single symbol with all required Stocks fields.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing all required Stocks category fields
        """
        try:
            symbol = self.validateSymbol(symbol)
            logger.info(f"Fetching quote for {symbol}")

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or len(info) < 2:
                return {"error": "No quote data available"}

            # Extract all required Stocks fields
            result = {
                "symbol": symbol.upper(),
                "currentPrice": info.get("currentPrice"),
                "previousClose": info.get("previousClose"),
                "open": info.get("open"),
                "dayLow": info.get("dayLow"),
                "dayHigh": info.get("dayHigh"),
                "regularMarketPrice": info.get("regularMarketPrice"),
                "regularMarketOpen": info.get("regularMarketOpen"),
                "regularMarketDayLow": info.get("regularMarketDayLow"),
                "regularMarketDayHigh": info.get("regularMarketDayHigh"),
                "regularMarketPreviousClose": info.get("regularMarketPreviousClose"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekChange": info.get("52WeekChange"),
                "fiftyDayAverage": info.get("fiftyDayAverage"),
                "twoHundredDayAverage": info.get("twoHundredDayAverage"),
                "beta": info.get("beta"),
                "impliedVolatility": info.get("impliedVolatility"),
                "volume": info.get("volume"),
                "regularMarketVolume": info.get("regularMarketVolume"),
                "averageVolume": info.get("averageVolume"),
                "averageVolume10days": info.get("averageVolume10days"),
                "averageVolume3months": info.get("averageVolume3months"),
                "updatedAt": timezone.now(),
            }

            return result

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return {"error": str(e)}

    def fetchHistory(self, symbol: str, startDate: datetime, endDate: datetime, interval: str = "1d") -> List[Dict]:
        """
        Fetch historical data with history.AdjClose and history.Volume fields.

        Args:
            symbol: Stock ticker symbol
            startDate: Start date
            endDate: End date
            interval: Data interval

        Returns:
            List of dictionaries with adjClose and volume fields
        """
        try:
            symbol = self.validateSymbol(symbol)
            self.validateDateRange(startDate, endDate)
            interval = self.validateInterval(interval)

            logger.info(f"Fetching history for {symbol} from {startDate.date()} to {endDate.date()}")

            ticker = yf.Ticker(symbol)
            history = ticker.history(start=startDate, end=endDate, interval=interval)

            if history.empty:
                return []

            result = []
            for date, row in history.iterrows():
                try:
                    if pd.isna(row["Close"]):
                        continue

                    # Handle timezone
                    if hasattr(date, "tzinfo") and date.tzinfo is not None:
                        date_utc = date.tz_convert("UTC")
                    else:
                        date_utc = date.replace(tzinfo=dt_timezone.utc)

                    result.append(
                        {
                            "symbol": symbol.upper(),
                            "date": date_utc,
                            "adjClose": Decimal(str(round(float(row["Close"]), 2))),
                            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                            "interval": interval,
                            "data_source": "yahoo",
                        }
                    )
                except Exception as e:
                    logger.warning(f"Skipping row for {symbol}: {str(e)}")
                    continue

            return result

        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {str(e)}")
            return []

    def fetchIndustrySectorSingle(self, symbol: str) -> Dict:
        """
        Fetch industry and sector data with recency timestamp.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with sector, industry, and updatedAt fields
        """
        return self.fetchSectorIndustrySingle(symbol)

    def upsertStocksMetrics(self, row: Dict) -> bool:
        """
        Upsert Stocks fields into Stock model.

        Args:
            row: Dictionary with Stocks category field data

        Returns:
            True if successful, False otherwise
        """
        try:
            symbol = row.get("symbol", "").upper()
            if not symbol:
                return False

            # Convert decimal fields safely
            decimal_fields = [
                "currentPrice",
                "previousClose",
                "dayLow",
                "dayHigh",
                "regularMarketPrice",
                "regularMarketOpen",
                "regularMarketDayLow",
                "regularMarketDayHigh",
                "regularMarketPreviousClose",
                "fiftyTwoWeekLow",
                "fiftyTwoWeekHigh",
                "fiftyTwoWeekChange",
                "fiftyDayAverage",
                "twoHundredDayAverage",
                "beta",
                "impliedVolatility",
            ]

            integer_fields = [
                "volume",
                "regularMarketVolume",
                "averageVolume",
                "averageVolume10days",
                "averageVolume3months",
            ]

            update_data = {"last_sync": row.get("updatedAt", timezone.now())}

            for field in decimal_fields:
                value = row.get(field)
                if value is not None:
                    try:
                        update_data[field] = Decimal(str(value))
                    except (ValueError, TypeError):
                        pass

            for field in integer_fields:
                value = row.get(field)
                if value is not None:
                    try:
                        update_data[field] = int(value)
                    except (ValueError, TypeError):
                        pass

            stock, created = Stock.objects.get_or_create(symbol=symbol, defaults=update_data)

            if not created:
                for field, value in update_data.items():
                    setattr(stock, field, value)
                stock.save()

            return True

        except Exception as e:
            logger.error(f"Error upserting stocks metrics: {str(e)}")
            return False

    def upsertIndustrySector(self, row: Dict) -> bool:
        """
        Upsert Industry & Sector non-time-series fields.

        Args:
            row: Dictionary with sector/industry data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use existing upsertCompanyProfiles method
            symbol = row.get("symbol", "").upper()
            if not symbol:
                return False

            result = self.upsertCompanyProfiles({symbol: row})
            return sum(result) > 0  # created + updated > 0

        except Exception as e:
            logger.error(f"Error upserting industry/sector: {str(e)}")
            return False

    def ensure_postgresql_engine(self):
        """
        Ensure PostgreSQL database engine is being used.
        Raises exception if SQLite is detected per requirements.

        Raises:
            RuntimeError: If SQLite engine is detected
        """
        from django.db import connection

        engine = connection.vendor

        if engine == "sqlite":
            raise RuntimeError(
                "SQLite database detected. This analytics system requires PostgreSQL. "
                "Please configure PostgreSQL database in settings."
            )
        elif engine != "postgresql":
            logger.warning(f"Unexpected database engine '{engine}'. Expected 'postgresql'.")

        logger.info(f"Database engine verified: {engine}")

    def backfill_eod_gaps_concurrent(self, symbol: str, required_years: int = 2, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Enhanced concurrent backfill with intelligent caching and pre-validation.
        
        Performance optimizations:
        1. Pre-check stock existence to avoid expensive failed fetches
        2. Concurrent data fetching for stock, sector, industry
        3. Intelligent caching with 24h TTL
        4. Early exit if recent backfill completed
        
        Args:
            symbol: Stock ticker symbol
            required_years: Years of history required
            max_attempts: Maximum retry attempts
            
        Returns:
            Dict with backfill results and performance metrics
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.time()
        
        result = {
            "success": False,
            "stock_backfilled": 0,
            "sector_backfilled": 0,
            "industry_backfilled": 0,
            "attempts_used": 0,
            "errors": [],
            "performance": {
                "total_time": 0,
                "cache_hits": 0,
                "api_calls": 0,
                "concurrent_fetches": 0
            }
        }
        
        try:
            from django.core.cache import cache
            from Data.repo.price_reader import PriceReader
            
            # Step 1: Check backfill cache (24h TTL)
            cache_key = f"backfill_status:{symbol}:{required_years}"
            cached_result = cache.get(cache_key)
            
            if cached_result and cached_result.get("success"):
                hours_old = (time.time() - cached_result.get("cached_timestamp", 0)) / 3600
                if hours_old < 24:
                    logger.info(f"Using cached backfill result for {symbol} (age: {hours_old:.1f}h)")
                    result["performance"]["cache_hits"] = 1
                    result.update(cached_result)
                    return result
            
            # Step 2: Pre-validate stock existence (quick Yahoo Finance check)
            logger.info(f"Pre-validating {symbol} existence")
            if not self._quick_stock_validation(symbol):
                result["errors"].append(f"Stock {symbol} not found on Yahoo Finance")
                return result
            
            result["performance"]["api_calls"] += 1
            
            # Ensure PostgreSQL engine
            self.ensure_postgresql_engine()
            price_reader = PriceReader()
            
            # Calculate optimized date range
            end_date = timezone.now()
            target_trading_days = required_years * 252
            calendar_days_needed = int(target_trading_days / 0.690420)
            start_date = end_date - timezone.timedelta(days=calendar_days_needed)
            
            # Step 3: Concurrent backfill execution
            for attempt in range(1, max_attempts + 1):
                result["attempts_used"] = attempt
                logger.info(f"Concurrent backfill attempt {attempt}/{max_attempts} for {symbol}")
                
                try:
                    # Check current coverage
                    coverage = price_reader.check_data_coverage(symbol, required_years)
                    
                    # Prepare concurrent tasks
                    tasks = []
                    
                    # Task 1: Stock data backfill
                    if not coverage["stock"]["has_data"] or coverage["stock"]["gap_count"] > 50:
                        tasks.append(("stock", self._concurrent_stock_backfill, (symbol, start_date, end_date)))
                    
                    # Task 2: Sector/Industry classification (parallel with stock fetch)
                    tasks.append(("classification", self._concurrent_classification_backfill, (symbol,)))
                    
                    # Execute concurrent tasks
                    if tasks:
                        logger.info(f"Executing {len(tasks)} concurrent backfill tasks for {symbol}")
                        result["performance"]["concurrent_fetches"] = len(tasks)
                        
                        with ThreadPoolExecutor(max_workers=min(3, len(tasks)), thread_name_prefix="Backfill") as executor:
                            future_to_task = {
                                executor.submit(task_func, *args): task_name 
                                for task_name, task_func, args in tasks
                            }
                            
                            for future in as_completed(future_to_task):
                                task_name = future_to_task[future]
                                try:
                                    task_result = future.result(timeout=45)  # 45s timeout per task
                                    
                                    if task_name == "stock" and task_result:
                                        result["stock_backfilled"] = task_result
                                        logger.info(f"Concurrent stock backfill completed: {task_result} records")
                                    elif task_name == "classification" and task_result:
                                        logger.info("Concurrent classification completed successfully")
                                        
                                except Exception as e:
                                    logger.warning(f"Concurrent task {task_name} failed: {str(e)}")
                                    result["errors"].append(f"Task {task_name}: {str(e)}")
                    
                    # Task 3: Sector/Industry composites (after classification completes)
                    if result["stock_backfilled"] > 0:
                        sector_key, industry_key = price_reader.get_stock_sector_industry_keys(symbol)
                        
                        composite_tasks = []
                        if sector_key and not coverage["sector"]["has_data"]:
                            composite_tasks.append(("sector", sector_key))
                        if industry_key and not coverage["industry"]["has_data"]:
                            composite_tasks.append(("industry", industry_key))
                        
                        if composite_tasks:
                            logger.info(f"Executing composite backfill for {len(composite_tasks)} entities")
                            composite_result = self.composeSectorIndustryEod((start_date, end_date))
                            result["sector_backfilled"] += composite_result.get("sector_prices_created", 0)
                            result["industry_backfilled"] += composite_result.get("industry_prices_created", 0)
                    
                    # Validate final coverage
                    final_coverage = price_reader.check_data_coverage(symbol, required_years)
                    stock_adequate = final_coverage["stock"]["has_data"] and final_coverage["stock"]["gap_count"] <= 50
                    
                    if stock_adequate:
                        result["success"] = True
                        
                        # Auto-fit Universal LSTM scalers for this stock after successful backfill
                        try:
                            self._fit_scalers_for_stock(symbol)
                            logger.info(f"Auto-fitted scalers for {symbol} after successful backfill")
                        except Exception as e:
                            logger.warning(f"Could not auto-fit scalers for {symbol}: {str(e)}")
                        
                        # Cache successful result for 24 hours
                        cache_data = dict(result)
                        cache_data["cached_timestamp"] = time.time()
                        cache.set(cache_key, cache_data, 86400)  # 24 hours
                        
                        break
                    else:
                        logger.warning(f"Coverage still inadequate after attempt {attempt}")
                        
                except Exception as e:
                    error_msg = f"Attempt {attempt} failed: {str(e)}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    
                    if attempt == max_attempts:
                        break
                    
                    # Exponential backoff
                    time.sleep(2 ** attempt)
            
            # Final performance metrics
            result["performance"]["total_time"] = time.time() - start_time
            result["performance"]["api_calls"] += result["attempts_used"]
            
            if result["success"]:
                logger.info(f"Concurrent backfill completed for {symbol} in {result['performance']['total_time']:.2f}s")
            else:
                logger.error(f"Concurrent backfill failed for {symbol} after {result['attempts_used']} attempts")
                
            return result
            
        except Exception as e:
            error_msg = f"Concurrent backfill error for {symbol}: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            result["performance"]["total_time"] = time.time() - start_time
            return result

    def _quick_stock_validation(self, symbol: str) -> bool:
        """Quick validation to check if stock exists before expensive backfill."""
        try:
            import yfinance as yf
            
            # Quick info fetch (much faster than full history)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Basic validation - stock should have a market cap or sector
            if info and (info.get('marketCap') or info.get('sector')):
                return True
            
            logger.warning(f"Stock validation failed for {symbol}: no market data found")
            return False
            
        except Exception as e:
            logger.warning(f"Quick stock validation failed for {symbol}: {str(e)}")
            return False

    def _concurrent_stock_backfill(self, symbol: str, start_date, end_date) -> int:
        """Concurrent stock data backfill task."""
        try:
            logger.debug(f"Starting concurrent stock backfill for {symbol}")
            
            stock_data = self.fetchStockEodHistory(symbol, start_date, end_date)
            if not stock_data:
                return 0
            
            # Convert to StockPrice format
            stock_rows = []
            for item in stock_data:
                stock_rows.append({
                    "symbol": symbol,
                    "date": item["date"],
                    "open": item["open"],
                    "high": item["high"],
                    "low": item["low"],
                    "close": item["close"],
                    "adjusted_close": item["adjusted_close"],
                    "volume": item["volume"],
                    "data_source": "yahoo",
                })
            
            backfilled = self._upsert_stock_prices(stock_rows)
            logger.debug(f"Concurrent stock backfill completed for {symbol}: {backfilled} records")
            return backfilled
            
        except Exception as e:
            logger.warning(f"Concurrent stock backfill failed for {symbol}: {str(e)}")
            raise

    def _concurrent_classification_backfill(self, symbol: str) -> bool:
        """Concurrent classification backfill task."""
        try:
            from Data.models import Stock
            
            logger.debug(f"Starting concurrent classification for {symbol}")
            
            stock_record = Stock.objects.get(symbol=symbol)
            
            # Create classification if needed
            if (stock_record.sector and stock_record.industry and 
                not stock_record.sector_id and not stock_record.industry_id):
                
                sector_key = self._normalizeSectorKey(stock_record.sector)
                industry_key = self._normalizeIndustryKey(f"{stock_record.industry}_{stock_record.sector}")
                
                classification_row = {
                    "symbol": symbol,
                    "sectorKey": sector_key,
                    "sectorName": stock_record.sector,
                    "industryKey": industry_key,
                    "industryName": stock_record.industry,
                    "updatedAt": timezone.now(),
                }
                
                self.upsertClassification([classification_row])
                logger.debug(f"Concurrent classification completed for {symbol}")
                return True
                
            return True
            
        except Exception as e:
            logger.warning(f"Concurrent classification failed for {symbol}: {str(e)}")
            raise

    def backfill_eod_gaps(self, symbol: str, required_years: int = 2, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Backfill missing EOD data for stock, sector, and industry with retry logic.

        Args:
            symbol: Stock ticker symbol
            required_years: Years of history required
            max_attempts: Maximum retry attempts for each data type

        Returns:
            Dict with backfill results:
            {
                'success': bool,
                'stock_backfilled': int,
                'sector_backfilled': int,
                'industry_backfilled': int,
                'attempts_used': int,
                'errors': List[str]
            }
        """
        from Data.repo.price_reader import PriceReader

        result = {
            "success": False,
            "stock_backfilled": 0,
            "sector_backfilled": 0,
            "industry_backfilled": 0,
            "attempts_used": 0,
            "errors": [],
        }

        try:
            # Ensure PostgreSQL engine
            self.ensure_postgresql_engine()

            price_reader = PriceReader()

            # Calculate date range with trading day limit
            # Target precise trading day count with minimal excess data
            end_date = timezone.now()

            # Calculate calendar days needed for target trading days
            # Use conservative multiplier to avoid fetching excess data
            target_trading_days = required_years * 252  # Standard trading days per year
            calendar_days_needed = int(target_trading_days / 0.690420)  # Inverse of trading day ratio
            start_date = end_date - timezone.timedelta(days=calendar_days_needed)

            for attempt in range(1, max_attempts + 1):
                result["attempts_used"] = attempt
                logger.info(f"Backfill attempt {attempt}/{max_attempts} for {symbol}")

                try:
                    # Check current coverage
                    coverage = price_reader.check_data_coverage(symbol, required_years)

                    # Backfill stock data if needed
                    if not coverage["stock"]["has_data"] or coverage["stock"]["gap_count"] > 50:
                        logger.info(f"Backfilling stock data for {symbol}")
                        stock_data = self.fetchStockEodHistory(symbol, start_date, end_date)
                        if stock_data:
                            # Convert to StockPrice format and upsert
                            stock_rows = []
                            for item in stock_data:
                                stock_rows.append(
                                    {
                                        "symbol": symbol,
                                        "date": item["date"],
                                        "open": item["open"],
                                        "high": item["high"],
                                        "low": item["low"],
                                        "close": item["close"],
                                        "adjusted_close": item["adjusted_close"],
                                        "volume": item["volume"],
                                        "data_source": "yahoo",
                                    }
                                )

                            backfilled = self._upsert_stock_prices(stock_rows)
                            result["stock_backfilled"] += backfilled
                            logger.info(f"Backfilled {backfilled} stock price records")

                    # Create sector/industry classifications if stock was backfilled successfully
                    if result["stock_backfilled"] > 0:
                        try:
                            from Data.models import Stock

                            stock_record = Stock.objects.get(symbol=symbol)

                            # Create classification record if sector/industry strings exist but FK relationships don't
                            if (
                                stock_record.sector
                                and stock_record.industry
                                and not stock_record.sector_id
                                and not stock_record.industry_id
                            ):

                                # Normalize keys for classification
                                sector_key = self._normalizeSectorKey(stock_record.sector)
                                industry_key = self._normalizeIndustryKey(
                                    f"{stock_record.industry}_{stock_record.sector}"
                                )

                                classification_row = {
                                    "symbol": symbol,
                                    "sectorKey": sector_key,
                                    "sectorName": stock_record.sector,
                                    "industryKey": industry_key,
                                    "industryName": stock_record.industry,
                                    "updatedAt": timezone.now(),
                                }

                                logger.info(f"Creating sector/industry classification for {symbol}")
                                self.upsertClassification([classification_row])

                        except Exception as e:
                            logger.error(f"Error creating classification for {symbol}: {str(e)}")

                    # Get sector/industry keys (should now exist after classification)
                    sector_key, industry_key = price_reader.get_stock_sector_industry_keys(symbol)

                    # Backfill sector and industry composites in one call to avoid duplicate fetches
                    needs_sector_backfill = sector_key and not coverage["sector"]["has_data"]
                    needs_industry_backfill = industry_key and not coverage["industry"]["has_data"]
                    
                    if needs_sector_backfill or needs_industry_backfill:
                        if needs_sector_backfill:
                            logger.info(f"Backfilling sector composite for {sector_key}")
                        if needs_industry_backfill:
                            logger.info(f"Backfilling industry composite for {industry_key}")
                        
                        # Single call to avoid duplicate data fetching
                        composite_result = self.composeSectorIndustryEod((start_date, end_date))
                        result["sector_backfilled"] += composite_result.get("sector_prices_created", 0)
                        result["industry_backfilled"] += composite_result.get("industry_prices_created", 0)

                    # Check if gaps are adequately filled
                    final_coverage = price_reader.check_data_coverage(symbol, required_years)

                    stock_adequate = final_coverage["stock"]["has_data"] and final_coverage["stock"]["gap_count"] <= 50

                    sector_adequate = not sector_key or (
                        final_coverage["sector"]["has_data"] and final_coverage["sector"]["gap_count"] <= 50
                    )

                    industry_adequate = not industry_key or (
                        final_coverage["industry"]["has_data"] and final_coverage["industry"]["gap_count"] <= 50
                    )

                    if stock_adequate and sector_adequate and industry_adequate:
                        result["success"] = True
                        logger.info(f"Backfill complete for {symbol} after {attempt} attempts")
                        break
                    else:
                        logger.info(f"Gaps still exist after attempt {attempt}, retrying...")

                except Exception as e:
                    error_msg = f"Attempt {attempt} failed: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)

                    # Add exponential backoff with jitter
                    if attempt < max_attempts:
                        delay = (2**attempt) + random.uniform(0, 1)
                        logger.info(f"Waiting {delay:.2f}s before retry...")
                        time.sleep(delay)

            if not result["success"]:
                error_msg = f"Failed to adequately backfill {symbol} after {max_attempts} attempts"
                result["errors"].append(error_msg)
                logger.error(error_msg)

            return result

        except Exception as e:
            error_msg = f"Critical error in backfill_eod_gaps: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
            return result

    def _upsert_stock_prices(self, stock_rows: List[Dict]) -> int:
        """Helper method to upsert stock price records."""
        try:
            if not stock_rows:
                return 0

            upserted_count = 0

            with transaction.atomic():
                for row in stock_rows:
                    try:
                        stock = Stock.objects.get(symbol=row["symbol"].upper())

                        price, created = StockPrice.objects.get_or_create(
                            stock=stock,
                            date=row["date"],
                            defaults={
                                "open": row["open"],
                                "high": row["high"],
                                "low": row["low"],
                                "close": row["close"],
                                "adjusted_close": row["adjusted_close"],
                                "volume": row["volume"],
                                "data_source": row["data_source"],
                            },
                        )

                        if not created:
                            # Update existing record if data is different
                            price.open = row["open"]
                            price.high = row["high"]
                            price.low = row["low"]
                            price.close = row["close"]
                            price.adjusted_close = row["adjusted_close"]
                            price.volume = row["volume"]
                            price.data_source = row["data_source"]
                            price.save()

                        upserted_count += 1

                    except Stock.DoesNotExist:
                        logger.warning(f"Stock {row['symbol']} not found for price upsert")
                        continue

            return upserted_count

        except Exception as e:
            logger.error(f"Error upserting stock prices: {str(e)}")
            return 0

    def upsertHistoryBars(self, rows: List[Dict]) -> int:
        """
        Upsert history.AdjClose and history.Volume time-series data.

        Args:
            rows: List of history dictionaries

        Returns:
            Number of rows upserted
        """
        try:
            if not rows:
                return 0

            upserted_count = 0

            with transaction.atomic():
                for row in rows:
                    symbol = row.get("symbol", "").upper()
                    date = row.get("date")

                    if not symbol or not date:
                        continue

                    try:
                        stock = Stock.objects.get(symbol=symbol)

                        # Convert datetime to date if needed
                        if isinstance(date, datetime):
                            date = date.date()

                        price_data = {
                            "adjusted_close": row.get("adjClose"),
                            "volume": row.get("volume", 0),
                            "data_source": row.get("data_source", "yahoo"),
                        }

                        # Use existing close for required fields if not provided
                        if "adjClose" in row:
                            adj_close = row["adjClose"]
                            price_data.update(
                                {"open": adj_close, "high": adj_close, "low": adj_close, "close": adj_close}
                            )

                        price, created = StockPrice.objects.get_or_create(stock=stock, date=date, defaults=price_data)

                        if not created:
                            # Update existing record
                            for field, value in price_data.items():
                                if value is not None:
                                    setattr(price, field, value)
                            price.save()

                        upserted_count += 1

                    except Stock.DoesNotExist:
                        logger.warning(f"Stock {symbol} not found for history upsert")
                        continue

            return upserted_count

        except Exception as e:
            logger.error(f"Error upserting history bars: {str(e)}")
            return 0

    def _fetch_news_with_fallbacks(self, symbol: str, days: int = 90, max_items: int = 100) -> Optional[List[Dict]]:
        """
        Fetch news with multiple fallback mechanisms for improved reliability.
        
        Fallback order:
        1. Standard yfinance ticker.news
        2. yfinance with custom DNS configuration
        3. Direct requests with alternative DNS
        4. Cached news from previous successful fetch (if available)
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of history
            max_items: Maximum news items to return
            
        Returns:
            List of raw news items or None if all fallbacks fail
        """
        import socket
        import urllib3
        
        # Attempt 1: Standard yfinance approach
        try:
            logger.info(f"Attempt 1/4: Standard yfinance news fetch for {symbol}")
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news
            
            if raw_news:
                logger.info(f"Standard fetch succeeded for {symbol}: {len(raw_news)} items")
                # Cache successful result
                self._cache_successful_news(symbol, raw_news)
                return raw_news
                
        except Exception as e:
            logger.warning(f"Standard news fetch failed for {symbol}: {str(e)}")
        
        # Attempt 2: yfinance with custom DNS settings
        try:
            logger.info(f"Attempt 2/4: yfinance with DNS fallback for {symbol}")
            
            # Configure custom DNS servers
            self._configure_dns_fallback()
            
            # Retry with DNS configuration
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news
            
            if raw_news:
                logger.info(f"DNS fallback succeeded for {symbol}: {len(raw_news)} items")
                self._cache_successful_news(symbol, raw_news)
                return raw_news
                
        except Exception as e:
            logger.warning(f"DNS fallback failed for {symbol}: {str(e)}")
        
        # Attempt 3: Direct requests with alternative approach
        try:
            logger.info(f"Attempt 3/4: Direct HTTP request for {symbol}")
            raw_news = self._direct_news_fetch(symbol, days, max_items)
            
            if raw_news:
                logger.info(f"Direct HTTP fetch succeeded for {symbol}: {len(raw_news)} items")
                self._cache_successful_news(symbol, raw_news)
                return raw_news
                
        except Exception as e:
            logger.warning(f"Direct HTTP fetch failed for {symbol}: {str(e)}")
        
        # Attempt 4: Use cached news from previous successful fetch
        try:
            logger.info(f"Attempt 4/4: Using cached news for {symbol}")
            cached_news = self._get_cached_news(symbol, days)
            
            if cached_news:
                logger.info(f"Using cached news for {symbol}: {len(cached_news)} items")
                return cached_news
                
        except Exception as e:
            logger.warning(f"Cached news fetch failed for {symbol}: {str(e)}")
        
        logger.error(f"All news fetch attempts failed for {symbol}")
        return None

    def _configure_dns_fallback(self):
        """Configure DNS fallback servers for improved reliability."""
        try:
            # Set alternative DNS servers
            import dns.resolver
            resolver = dns.resolver.Resolver()
            resolver.nameservers = ['8.8.8.8', '1.1.1.1', '208.67.222.222']  # Google, Cloudflare, OpenDNS
            dns.resolver.default_resolver = resolver
            logger.debug("DNS fallback configured successfully")
        except ImportError:
            logger.debug("dnspython not available, skipping DNS configuration")
        except Exception as e:
            logger.debug(f"DNS fallback configuration failed: {str(e)}")

    def _direct_news_fetch(self, symbol: str, days: int = 90, max_items: int = 100) -> Optional[List[Dict]]:
        """
        Direct HTTP request to Yahoo Finance news API with custom headers and retry logic.
        """
        try:
            import urllib3
            from urllib3.util.retry import Retry
            
            # Create session with retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            http = urllib3.PoolManager(retries=retry_strategy)
            
            # Yahoo Finance news endpoint
            url = f"https://query2.finance.yahoo.com/v1/finance/search"
            params = {
                'q': symbol,
                'type': 'news',
                'count': max_items,
                'region': 'US',
                'lang': 'en-US'
            }
            
            # Custom headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
            }
            
            # Make request
            response = http.request('GET', url, fields=params, headers=headers, timeout=30)
            
            if response.status == 200:
                import json
                data = json.loads(response.data.decode('utf-8'))
                
                # Extract news items from response
                if 'news' in data and isinstance(data['news'], list):
                    return data['news'][:max_items]
            
            logger.warning(f"Direct HTTP request returned status {response.status} for {symbol}")
            return None
            
        except Exception as e:
            logger.warning(f"Direct HTTP news fetch error for {symbol}: {str(e)}")
            return None

    def _cache_successful_news(self, symbol: str, news_data: List[Dict]):
        """Cache successful news fetch for emergency fallback."""
        try:
            from django.core.cache import cache
            
            cache_key = f"emergency_news:{symbol}"
            cache_data = {
                'news': news_data,
                'cached_at': datetime.now(dt_timezone.utc).isoformat(),
                'symbol': symbol
            }
            
            # Cache for 24 hours
            cache.set(cache_key, cache_data, 86400)
            logger.debug(f"Cached emergency news for {symbol}: {len(news_data)} items")
            
        except Exception as e:
            logger.debug(f"Failed to cache emergency news for {symbol}: {str(e)}")

    def _get_cached_news(self, symbol: str, days: int = 90) -> Optional[List[Dict]]:
        """Retrieve cached news for emergency fallback."""
        try:
            from django.core.cache import cache
            
            cache_key = f"emergency_news:{symbol}"
            cache_data = cache.get(cache_key)
            
            if cache_data and 'news' in cache_data:
                # Check if cache is not too old
                cached_at = datetime.fromisoformat(cache_data['cached_at'])
                age_hours = (datetime.now(dt_timezone.utc) - cached_at).total_seconds() / 3600
                
                if age_hours < 48:  # Use cache if less than 48 hours old
                    logger.info(f"Using emergency cached news for {symbol} (age: {age_hours:.1f}h)")
                    return cache_data['news']
                else:
                    logger.debug(f"Emergency cache too old for {symbol}: {age_hours:.1f}h")
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to retrieve cached news for {symbol}: {str(e)}")
            return None

    def fetchNewsForStock(self, symbol: str, days: int = 90, max_items: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a stock symbol for sentiment analysis.

        Args:
            symbol: Stock ticker symbol
            days: Number of days of history to fetch (default 90)
            max_items: Maximum number of news items to return

        Returns:
            List of news articles with title, summary, published date, and source
        """
        try:
            # Mock data detection - avoid API calls for test symbols
            if self._is_mock_or_test_symbol(symbol):
                logger.debug(f"Mock/test symbol detected for {symbol} - returning empty news list")
                return []

            logger.info(f"Fetching news for {symbol} (last {days} days)")

            # Fetch news with multiple fallback mechanisms
            news_items = []
            raw_news = self._fetch_news_with_fallbacks(symbol, days, max_items)

            logger.info(
                f"Raw news result for {symbol}: {type(raw_news)}, length: {len(raw_news) if raw_news else 'None'}"
            )

            if not raw_news:
                logger.warning(f"No news found for {symbol} - ticker.news returned: {raw_news}")
                return []

            # Calculate date threshold
            cutoff_date = datetime.now(dt_timezone.utc) - timedelta(days=days)

            for item in raw_news[:max_items]:
                try:
                    # Handle new yfinance structure where data is nested under 'content'
                    content = item.get("content", item)  # Fallback to item if no content key

                    # Extract and validate required fields
                    title = content.get("title", "")
                    summary = content.get("summary", content.get("description", ""))

                    # Skip if no meaningful content
                    if not title and not summary:
                        continue

                    # Parse published date - try multiple possible field names
                    published_timestamp = content.get("providerPublishTime", 0)
                    if not published_timestamp:
                        # Try alternative timestamp fields
                        pub_date = content.get("pubDate")
                        if pub_date:
                            try:
                                # Handle different date formats
                                if isinstance(pub_date, str):
                                    from dateutil import parser

                                    published_date = parser.parse(pub_date)
                                    if published_date.tzinfo is None:
                                        published_date = published_date.replace(tzinfo=dt_timezone.utc)
                                else:
                                    published_timestamp = pub_date
                            except Exception:
                                published_date = datetime.now(dt_timezone.utc)
                        else:
                            published_date = datetime.now(dt_timezone.utc)

                    if published_timestamp and not hasattr(published_date, "isoformat"):
                        published_date = datetime.fromtimestamp(published_timestamp, tz=dt_timezone.utc)

                        # Skip if older than cutoff
                        if published_date < cutoff_date:
                            continue
                    elif not hasattr(published_date, "isoformat"):
                        published_date = datetime.now(dt_timezone.utc)

                    # Build clean news item
                    news_item = {
                        "title": title,
                        "summary": summary or "",
                        "publishedDate": published_date.isoformat(),
                        "source": content.get("publisher", content.get("provider", {}).get("displayName", "Unknown")),
                        "link": content.get("link", content.get("canonicalUrl", "")),
                        "uuid": content.get("uuid", content.get("id", "")),
                        "type": content.get("type", content.get("contentType", "STORY")),
                        "relatedTickers": content.get("relatedTickers", []),
                    }

                    news_items.append(news_item)

                except Exception as e:
                    logger.debug(f"Error processing news item: {str(e)}")
                    continue

            # Sort by date (newest first)
            news_items.sort(key=lambda x: x["publishedDate"], reverse=True)

            logger.info(f"Fetched {len(news_items)} news items for {symbol}")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return []

    def fetchBatchNews(
        self, symbols: List[str], days: int = 90, max_items_per_symbol: int = 50
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch news for multiple stock symbols with mock data detection.

        Args:
            symbols: List of stock ticker symbols
            days: Number of days of history to fetch
            max_items_per_symbol: Maximum news items per symbol

        Returns:
            Dictionary mapping symbols to their news articles
        """
        results = {}

        # Filter out mock/test symbols to avoid unnecessary API calls
        real_symbols = [symbol for symbol in symbols if not self._is_mock_or_test_symbol(symbol)]
        mock_symbols = [symbol for symbol in symbols if self._is_mock_or_test_symbol(symbol)]

        if mock_symbols:
            logger.debug(f"Skipping news fetch for mock/test symbols: {mock_symbols}")

        for symbol in real_symbols:
            try:
                news = self.fetchNewsForStock(symbol, days, max_items_per_symbol)
                results[symbol] = news

                # Add small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {str(e)}")
                results[symbol] = []

        # Add empty results for mock symbols to maintain expected dictionary structure
        for symbol in mock_symbols:
            results[symbol] = []

        return results

    def preprocessNewsText(self, article: Dict[str, Any]) -> str:
        """
        Preprocess news article text for sentiment analysis.

        Args:
            article: News article dictionary

        Returns:
            Cleaned and combined text ready for analysis
        """
        try:
            title = article.get("title", "")
            summary = article.get("summary", "")

            # Combine title and summary
            text = f"{title}. {summary}".strip()

            # Basic text cleaning
            # Remove excessive whitespace
            text = " ".join(text.split())

            # Remove HTML tags if any
            text = re.sub(r"<[^>]+>", "", text)

            # Remove URLs
            text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)

            # Limit length for processing efficiency
            if len(text) > 5000:
                text = text[:5000]

            return text

        except Exception as e:
            logger.error(f"Error preprocessing news text: {str(e)}")
            return ""

    def aggregateNewsBySentiment(
        self, news_items: List[Dict[str, Any]], sentiments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate news articles by sentiment scores.

        Args:
            news_items: List of news articles
            sentiments: List of sentiment analysis results

        Returns:
            Aggregated sentiment statistics and breakdown
        """
        try:
            if not news_items or not sentiments:
                return {
                    "totalArticles": 0,
                    "averageSentiment": 0.0,
                    "sentimentBreakdown": {"positive": 0, "negative": 0, "neutral": 0},
                    "sourceBreakdown": {},
                }

            # Track sentiment by source
            source_sentiments = defaultdict(list)
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            total_score = 0.0

            for article, sentiment in zip(news_items, sentiments):
                if sentiment:
                    score = sentiment.get("sentimentScore", 0)
                    label = sentiment.get("sentimentLabel", "neutral")
                    source = article.get("source", "Unknown")

                    total_score += score
                    sentiment_counts[label] += 1
                    source_sentiments[source].append(score)

            # Calculate source averages
            source_breakdown = {}
            for source, scores in source_sentiments.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    source_breakdown[source] = {"count": len(scores), "avg_score": round(avg_score, 4)}

            return {
                "totalArticles": len(news_items),
                "averageSentiment": round(total_score / len(sentiments), 4) if sentiments else 0.0,
                "sentimentBreakdown": sentiment_counts,
                "sourceBreakdown": source_breakdown,
                "lastNewsDate": news_items[0]["publishedDate"] if news_items else None,
            }

        except Exception as e:
            logger.error(f"Error aggregating news sentiment: {str(e)}")
            return {
                "totalArticles": 0,
                "averageSentiment": 0.0,
                "sentimentBreakdown": {"positive": 0, "negative": 0, "neutral": 0},
                "sourceBreakdown": {},
            }

    def sync_stock_data(self, symbol: str, years: int = 1) -> bool:
        """
        Wrapper method for backward compatibility.

        Args:
            symbol: Stock ticker symbol
            years: Number of years of data to sync (1-5, or use max for >5)

        Returns:
            True if sync successful, False otherwise
        """
        try:
            # Convert years to period string
            if years <= 5:
                period = f"{years}y"
            else:
                period = "max"

            # Use existing get_stock_data method with sync_db=True
            result = self.get_stock_data(symbol, period=period, sync_db=True)

            # Return True if successful and no errors
            return result and "error" not in result

        except Exception as e:
            logger.error(f"Error syncing stock data for {symbol}: {str(e)}")
            return False


    def _fit_scalers_for_stock(self, symbol: str):
        """Fit Universal LSTM scalers for the stock after successful backfill."""
        try:
            from Data.models import Stock, StockPrice
            from Analytics.management.commands.fit_universal_scalers import Command as ScalerCommand
            
            # Get stock and ensure we have enough data
            stock = Stock.objects.get(symbol=symbol)
            prices = StockPrice.objects.filter(
                stock=stock
            ).order_by('-date')[:252]  # 1 year of data
            
            if prices.count() < 60:
                logger.debug(f"Not enough data to fit scalers for {symbol}: {prices.count()} records")
                return
            
            # Create instance of scaler fitting command
            scaler_cmd = ScalerCommand()
            
            # Build features for scaler fitting
            features = []
            for i in range(len(prices) - 1):
                price_data = prices[i]
                prev_price = prices[i + 1] if i + 1 < len(prices) else None
                
                if prev_price:
                    feature_dict = {
                        'price': float(price_data.close),
                        'volume': float(price_data.volume),
                        'price_change': float(price_data.close - prev_price.close),
                        'volume_ratio': float(price_data.volume / prev_price.volume) if prev_price.volume > 0 else 1.0,
                        'high_low_ratio': float(price_data.high / price_data.low) if price_data.low > 0 else 1.0,
                        'close_open_ratio': float(price_data.close / price_data.open) if price_data.open > 0 else 1.0,
                    }
                    features.append(feature_dict)
            
            if len(features) >= 30:
                # Fit and save scalers
                scaler_data = scaler_cmd._fit_scalers_for_stock(symbol, features)
                if scaler_data:
                    scaler_cmd._save_scalers_to_file(symbol, scaler_data)
                    logger.debug(f'Auto-fitted and saved scalers for {symbol}')
                    
        except Exception as e:
            logger.warning(f'Could not auto-fit scalers for {symbol}: {str(e)}')


# Singleton instance
yahoo_finance_service = YahooFinanceService()


def create_yahoo_finance_service():
    """
    Create a new YahooFinanceService instance for use with context managers.

    Use this when you want guaranteed resource cleanup:

    Example usage in a management command:

        from Data.services.yahoo_finance import create_yahoo_finance_service

        def handle(self, *args, **options):
            with create_yahoo_finance_service() as service:
                data = service.fetchSingleHistorical('AAPL', '1mo', '1d')
                # Process data as needed
                # Session is automatically closed after this block

    For general use throughout the application, prefer the singleton yahoo_finance_service.
    """
    return YahooFinanceService()
