"""
Yahoo Finance Integration Module
Handles all interactions with Yahoo Finance API for VoyageurCompass.
This module acts as the main interface for Yahoo Finance operations.
"""

import logging
import re
import time
import random
import os
import hashlib
import threading
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone as dt_timezone
from decimal import Decimal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure yfinance with proper headers to handle consent pages
# SSL verification remains enabled for security
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Note: requests.Session doesn't support a default `timeout` attribute.
# Keep a module-level default to pass explicitly to HTTP calls.
DEFAULT_TIMEOUT = 30

from Data.services.provider import data_provider
from Data.services.synchronizer import data_synchronizer
from django.db import models, transaction
from django.utils import timezone
from Data.models import Stock, StockPrice, DataSector, DataIndustry, DataSectorPrice, DataIndustryPrice

logger = logging.getLogger(__name__)


class CompositeCache:
    """
    Phase 4: Smart Caching for composite generation optimization.
    Caches frequently accessed stock price data and composite calculations.
    """
    
    def __init__(self):
        self.price_data_cache = {}
        self.composite_cache = {}
        self.stock_date_index = {}
    
    def get_cache_key(self, stocks, start_date, end_date):
        """Generate a cache key for stock price data."""
        if hasattr(stocks, '__iter__') and not isinstance(stocks, str):
            stock_ids = sorted([stock.id for stock in stocks])
        else:
            stock_ids = [stocks.id] if hasattr(stocks, 'id') else [stocks]
        
        key_string = f"{stock_ids}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    def get_composite_cache_key(self, entity, date):
        """Generate a cache key for composite calculations."""
        entity_type = entity.__class__.__name__
        entity_id = entity.id if hasattr(entity, 'id') else str(entity)
        return f"{entity_type}_{entity_id}_{date}"
    
    def index_prices_by_stock_and_date(self, all_prices):
        """
        Phase 4: Index price data for fast lookup.
        Creates a nested dictionary: {stock: {date: price_record}}
        """
        if hasattr(self, '_indexed_prices') and self._indexed_prices:
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
        if hasattr(self, '_indexed_prices'):
            del self._indexed_prices


class YahooFinanceService:
    """
    Main service class for Yahoo Finance API integration.
    Coordinates between provider, synchronizer, and database.
    """
    
    # Class-level constants for validation
    VALID_PERIODS = frozenset(['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
    VALID_INTERVALS = frozenset(['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'])
    
    def __init__(self):
        """Initialize the Yahoo Finance service."""
        self.provider = data_provider
        self.synchronizer = data_synchronizer
        self.timeout = 30  # Default timeout
        self.maxRetries = 5
        self.baseDelay = 2
        self.maxBackoff = 60
        
        # Create instance-specific session for thread safety
        self._create_session()
        
        logger.info("Yahoo Finance Service initialized with yfinance integration")
    
    def _create_session(self):
        """Create a resilient requests session with retries, pooling, and modern headers."""
        self.session = requests.Session()
        
        # Enhanced headers with modern encoding and refined accept headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,image/webp,*/*;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,*;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',  # Added brotli support
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache'
        })
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=1.0,  # Backoff factor (1 second, then 2, then 4)
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Only retry safe methods
            raise_on_status=False  # Don't raise exception on final failure
        )
        
        # Create HTTP adapter with retry strategy and connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools to cache
            pool_maxsize=20,     # Maximum connections in each pool
            pool_block=False     # Don't block if pool is full
        )
        
        # Mount adapters for both HTTP and HTTPS
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def close(self):
        """Close the requests session and cleanup resources."""
        if hasattr(self, 'session') and self.session:
            self.session.close()
            self.session = None
    
    def _ensure_session(self):
        """Ensure session exists, recreating if necessary."""
        if not hasattr(self, 'session') or self.session is None:
            self._create_session()
    
    def __enter__(self):
        """Enter the context manager, returning the service instance."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, ensuring session cleanup."""
        self.close()
        # Don't suppress exceptions
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
        if not re.match(r'^[A-Z0-9\.\-\^]{1,10}$', symbol):
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
        
    # =====================================================================
    # camelCase wrapper methods
    # =====================================================================
    
    def getStockData(self, symbol: str, period: str = "1mo", syncDb: bool = True) -> Dict:
        """camelCase wrapper for get_stock_data"""
        return self.get_stock_data(symbol, period, syncDb)
    
    def get_stock_data(self, symbol: str, period: str = "1mo", sync_db: bool = True) -> Dict:
        """
        Fetch stock data for a given symbol.
        
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
            
            if sync_db:
                # Sync to database and return results
                sync_result = self.synchronizer.sync_stock_data(symbol, period)
                
                if sync_result['success']:
                    # Fetch from database for consistent format
                    stock = Stock.objects.get(symbol=symbol.upper())
                    prices = StockPrice.objects.filter(stock=stock).order_by('-date')[:30]
                    
                    stock_data = {
                        'symbol': stock.symbol,
                        'period': period,
                        'info': {
                            'shortName': stock.short_name,
                            'longName': stock.long_name,
                            'currency': stock.currency,
                            'exchange': stock.exchange,
                            'sector': stock.sector,
                            'industry': stock.industry,
                            'marketCap': stock.market_cap,
                        },
                        'prices': [float(p.close) for p in prices],
                        'volumes': [p.volume for p in prices],
                        'dates': [p.date.isoformat() for p in prices],
                        'fetched_at': datetime.now().isoformat(),
                        'from_cache': False
                    }
                else:
                    stock_data = {'error': sync_result.get('error', 'Sync failed')}
            else:
                # Just fetch without syncing to database
                stock_data = self.provider.fetch_stock_data(symbol, period)
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def getStockInfo(self, symbol: str) -> Dict:
        """camelCase wrapper for get_stock_info"""
        return self.get_stock_info(symbol)
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a stock.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary containing stock information
        """
        try:
            # Validate input
            symbol = self.validateSymbol(symbol)
            
            logger.info(f"Fetching stock info for {symbol}")
            
            # Try to get from database first
            try:
                stock = Stock.objects.get(symbol=symbol.upper())
                latest_price = stock.get_latest_price()
                
                info = {
                    'symbol': stock.symbol,
                    'shortName': stock.short_name,
                    'longName': stock.long_name,
                    'sector': stock.sector,
                    'industry': stock.industry,
                    'marketCap': stock.market_cap,
                    'currency': stock.currency,
                    'exchange': stock.exchange,
                    'lastSync': stock.last_sync.isoformat() if stock.last_sync else None,
                    'currentPrice': float(latest_price.close) if latest_price else None,
                    'previousClose': float(latest_price.close) if latest_price else None,
                    'from_cache': True
                }
                
                # If data is stale, refresh from Yahoo Finance
                if stock.needs_sync:
                    logger.info(f"Stock {symbol} needs sync, fetching fresh data")
                    company_info = self.provider.fetch_company_info(symbol)
                    if 'error' not in company_info:
                        info.update(company_info)
                        info['from_cache'] = False
                        # Trigger background sync
                        self.synchronizer.sync_stock_data(symbol)
                
                return info
                
            except Stock.DoesNotExist:
                # Not in database, fetch from Yahoo Finance
                company_info = self.provider.fetch_company_info(symbol)
                
                if 'error' not in company_info:
                    # Sync to database for future use
                    self.synchronizer.sync_stock_data(symbol)
                
                return company_info
                
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {'error': str(e)}
    
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
                    stock=stock,
                    date__gte=start_date.date(),
                    date__lte=end_date.date()
                ).order_by('date')
                
                if prices.exists():
                    historical_data = {
                        'symbol': stock.symbol,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'data': [
                            {
                                'date': p.date.isoformat(),
                                'open': float(p.open),
                                'high': float(p.high),
                                'low': float(p.low),
                                'close': float(p.close),
                                'volume': p.volume,
                                'change_amount': float(p.daily_change) if p.daily_change else None,
                                'change_percent': float(p.daily_change_percent) if p.daily_change_percent else None,
                            }
                            for p in prices
                        ],
                        'from_cache': True,
                        'fetched_at': datetime.now().isoformat()
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
            
            if sync_result['success']:
                # Fetch from database after sync
                stock = Stock.objects.get(symbol=symbol.upper())
                prices = StockPrice.objects.filter(
                    stock=stock,
                    date__gte=start_date.date(),
                    date__lte=end_date.date()
                ).order_by('date')
                
                historical_data = {
                    'symbol': stock.symbol,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'data': [
                        {
                            'date': p.date.isoformat(),
                            'open': float(p.open),
                            'high': float(p.high),
                            'low': float(p.low),
                            'close': float(p.close),
                            'volume': p.volume,
                        }
                        for p in prices
                    ],
                    'from_cache': False,
                    'fetched_at': datetime.now().isoformat()
                }
            else:
                historical_data = {'error': sync_result.get('error', 'Failed to fetch data')}
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return {'error': str(e)}
    
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
            if symbol in sync_results['results'] and sync_results['results'][symbol]['success']:
                results[symbol] = self.get_stock_data(symbol, period, sync_db=False)
            else:
                results[symbol] = {'error': 'Failed to sync data'}
        
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
            for symbol in ['^GSPC', '^DJI', '^IXIC']:  # S&P 500, Dow Jones, NASDAQ
                try:
                    price = self.provider.fetch_realtime_price(symbol)
                    if price:
                        market_indicators[symbol] = price
                except Exception as e:
                    logger.warning(f"Failed to fetch price for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error retrieving market indicators: {e}")
        
        return {
            'is_open': is_open,
            'current_time': now.isoformat(),
            'market_hours': {
                'open': '09:30 EST',
                'close': '16:00 EST'
            },
            'indicators': market_indicators,
            'next_open': self._get_next_market_open(now)
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
                models.Q(symbol__icontains=query) |
                models.Q(short_name__icontains=query) |
                models.Q(long_name__icontains=query)
            )[:10]
            
            results = []
            for stock in stocks:
                results.append({
                    'symbol': stock.symbol,
                    'name': stock.long_name or stock.short_name,
                    'type': 'Stock',
                    'exchange': stock.exchange,
                    'sector': stock.sector,
                    'from_cache': True
                })
            
            # If no results in database, validate the symbol with Yahoo Finance
            if not results and len(query) <= 5 and self.provider.validate_symbol(query):
                    # Valid symbol, fetch and sync
                    self.synchronizer.sync_stock_data(query)
                    try:
                        stock = Stock.objects.get(symbol=query.upper())
                        results.append({
                            'symbol': stock.symbol,
                            'name': stock.long_name or stock.short_name,
                            'type': 'Stock',
                            'exchange': stock.exchange,
                            'from_cache': False
                        })
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
                if '429' in str(e) or 'Too Many Requests' in str(e):
                    if attempt < self.maxRetries - 1:
                        delay = min(
                            self.baseDelay * (2 ** attempt) + random.uniform(0, 1),
                            self.maxBackoff
                        )
                        logger.warning(f"Rate limited, attempt {attempt + 1}/{self.maxRetries}, waiting {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        raise
                else:
                    raise
        raise Exception(f"Max retries ({self.maxRetries}) exceeded")
    
    def fetchBatchHistorical(self, tickers: List[str], period: str = '6mo', interval: str = '1d') -> Dict[str, List[Dict]]:
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
    
    def fetchSingleHistorical(self, ticker: str, period: str = '1wk', interval: str = '1d') -> List[Dict]:
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
        intraday_intervals = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'}
        max_intraday_periods = {'1d', '5d', '1mo'}
        
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
            normalized_col = col_ticker.upper().strip().replace('-', '.').replace('_', '.')
            normalized_input = normalized_ticker.replace('-', '.').replace('_', '.')
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
                (column_type.lower(), ticker.lower())
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
                    if (isinstance(idx[0], str) and column_type.lower() in idx[0].lower()):
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
                progress=False
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
                            close_val = row[('Close', matched_ticker)]
                            open_val = row[('Open', matched_ticker)]
                            high_val = row[('High', matched_ticker)]
                            low_val = row[('Low', matched_ticker)]
                            volume_val = row[('Volume', matched_ticker)]
                        else:
                            # Fallback: try to find columns by pattern matching
                            close_val = self._safe_multiindex_access(row, 'Close', ticker)
                            open_val = self._safe_multiindex_access(row, 'Open', ticker)
                            high_val = self._safe_multiindex_access(row, 'High', ticker)
                            low_val = self._safe_multiindex_access(row, 'Low', ticker)
                            volume_val = self._safe_multiindex_access(row, 'Volume', ticker)
                    else:
                        # Simple column names
                        close_val = row['Close']
                        open_val = row['Open'] 
                        high_val = row['High']
                        low_val = row['Low']
                        volume_val = row['Volume']
                    
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
                    if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                        date_utc = date.tz_convert('UTC')
                    else:
                        date_utc = date.replace(tzinfo=dt_timezone.utc)
                    
                    bars.append({
                        'symbol': ticker.upper(),
                        'date': date_utc,
                        'open': Decimal(str(round(open_val, 2))),
                        'high': Decimal(str(round(high_val, 2))),
                        'low': Decimal(str(round(low_val, 2))),
                        'close': Decimal(str(round(close_val, 2))),
                        'volume': volume_val,
                        'interval': interval,
                        'data_source': 'yahoo'
                    })
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
                    if data and 'error' not in data:
                        # Enhance with normalized keys for classification upsert
                        enhanced_data = {
                            'symbol': data['symbol'],
                            'sectorKey': self._normalizeSectorKey(data.get('sector', '')),
                            'sectorName': data.get('sector', ''),
                            'industryKey': self._normalizeIndustryKey(data.get('industry', '')),
                            'industryName': data.get('industry', ''),
                            'updatedAt': data.get('updatedAt')
                        }
                        results[symbol] = enhanced_data
                        logger.info(f"Successfully fetched sector/industry for {symbol}")
                    else:
                        logger.warning(f"No sector/industry data for {symbol}")
                        results[symbol] = {'error': 'No data available'}
                        
                except Exception as e:
                    logger.error(f"Failed to fetch sector/industry for {symbol}: {str(e)}")
                    results[symbol] = {'error': str(e)}
            
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
                return {'error': 'No company information available'}
            
            # Extract sector and industry
            sector = info.get('sector', '').strip()
            industry = info.get('industry', '').strip()
            
            if not sector and not industry:
                logger.warning(f"No sector/industry data found for {symbol}")
                return {'error': 'No sector/industry data available'}
            
            result = {
                'symbol': symbol.upper(),
                'sector': sector or '',
                'industry': industry or '',
                'sectorKey': self._normalizeSectorKey(sector),
                'sectorName': sector,
                'industryKey': self._normalizeIndustryKey(industry),
                'industryName': industry,
                'updatedAt': timezone.now()
            }
            
            logger.info(f"Retrieved sector/industry for {symbol}: {sector} / {industry}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching sector/industry for {symbol}: {str(e)}")
            return {'error': str(e)}
    
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
        threshold_date = timezone.now() - timedelta(days=3*365)
        
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
        
        logger.info(f"Symbol analysis: {len(recentSymbols)} recent, {len(staleSymbols)} stale, {len(missingSymbols)} missing")
        
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
            if 'error' in profile_data:
                skipped_count += 1
                continue
                
            try:
                stock, created = Stock.objects.get_or_create(
                    symbol=symbol,
                    defaults={
                        'short_name': symbol,
                        'sector': profile_data.get('sector', ''),
                        'industry': profile_data.get('industry', ''),
                        'sectorUpdatedAt': profile_data.get('updatedAt'),
                        'data_source': 'yahoo'
                    }
                )
                
                if created:
                    created_count += 1
                    logger.info(f"Created new stock record for {symbol}")
                else:
                    # Update existing record
                    stock.sector = profile_data.get('sector', '')
                    stock.industry = profile_data.get('industry', '')
                    stock.sectorUpdatedAt = profile_data.get('updatedAt')
                    stock.save(update_fields=['sector', 'industry', 'sectorUpdatedAt'])
                    updated_count += 1
                    logger.info(f"Updated sector/industry for {symbol}")
                    
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
                    symbol = row.get('symbol', '').upper()
                    sector_key = self._normalizeSectorKey(row.get('sectorKey', ''))
                    sector_name = row.get('sectorName', '').strip()
                    industry_key = self._normalizeIndustryKey(row.get('industryKey', ''))
                    industry_name = row.get('industryName', '').strip()
                    updated_at = row.get('updatedAt')
                    
                    if not symbol or not sector_key or not industry_key:
                        skipped_count += 1
                        continue
                    
                    # Create or get sector
                    sector, sector_created = DataSector.objects.get_or_create(
                        sectorKey=sector_key,
                        defaults={
                            'sectorName': sector_name,
                            'last_sync': updated_at,
                            'data_source': 'yahoo'
                        }
                    )
                    
                    if not sector_created and sector.sectorName != sector_name:
                        sector.sectorName = sector_name
                        sector.last_sync = updated_at
                        sector.save(update_fields=['sectorName', 'last_sync'])
                    
                    # Create or get industry
                    industry, industry_created = DataIndustry.objects.get_or_create(
                        industryKey=industry_key,
                        defaults={
                            'industryName': industry_name,
                            'sector': sector,
                            'last_sync': updated_at,
                            'data_source': 'yahoo'
                        }
                    )
                    
                    if not industry_created and (industry.industryName != industry_name or industry.sector != sector):
                        industry.industryName = industry_name
                        industry.sector = sector
                        industry.last_sync = updated_at
                        industry.save(update_fields=['industryName', 'sector', 'last_sync'])
                    
                    # Update Stock model with FK relationships
                    try:
                        stock = Stock.objects.get(symbol=symbol)
                        if stock.sector_id != sector or stock.industry_id != industry:
                            stock.sector_id = sector
                            stock.industry_id = industry
                            stock.sectorUpdatedAt = updated_at
                            stock.save(update_fields=['sector_id', 'industry_id', 'sectorUpdatedAt'])
                            updated_count += 1
                    except Stock.DoesNotExist:
                        # Create stock if it doesn't exist
                        Stock.objects.create(
                            symbol=symbol,
                            short_name=symbol,
                            sector_id=sector,
                            industry_id=industry,
                            sectorUpdatedAt=updated_at,
                            data_source='yahoo'
                        )
                        created_count += 1
                        
                except Exception as e:
                    logger.error(f"Error upserting classification for {row.get('symbol', 'unknown')}: {str(e)}")
                    skipped_count += 1
        
        logger.info(f"Classification upsert complete: {created_count} created, {updated_count} updated, {skipped_count} skipped")
        return created_count, updated_count, skipped_count
    
    def _normalizeSectorKey(self, sector: str) -> str:
        """Normalize sector name to consistent key format."""
        if not sector:
            return ''
        # Replace non-alphanumeric with underscores, then collapse multiple underscores
        normalized = re.sub(r'[^a-zA-Z0-9]', '_', sector.lower().strip())
        # Collapse multiple underscores into single underscore
        normalized = re.sub(r'_+', '_', normalized)
        return normalized.strip('_')
    
    def _normalizeIndustryKey(self, industry: str) -> str:
        """Normalize industry name to consistent key format."""
        if not industry:
            return ''
        # Replace non-alphanumeric with underscores, then collapse multiple underscores
        normalized = re.sub(r'[^a-zA-Z0-9]', '_', industry.lower().strip())
        # Collapse multiple underscores into single underscore
        normalized = re.sub(r'_+', '_', normalized)
        return normalized.strip('_')
    
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
            
            if 'error' in historical_data:
                return []
                
            # Convert to the expected format with adjusted_close
            eod_data = []
            for item in historical_data.get('data', []):
                eod_data.append({
                    'symbol': symbol.upper(),
                    'date': datetime.fromisoformat(item['date']).date(),
                    'open': Decimal(str(item['open'])),
                    'high': Decimal(str(item['high'])), 
                    'low': Decimal(str(item['low'])),
                    'close': Decimal(str(item['close'])),
                    'adjusted_close': Decimal(str(item.get('adjusted_close', item['close']))),
                    'volume': item['volume'],
                    'data_source': 'yahoo'
                })
            
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
            sectors = DataSector.objects.filter(
                isActive=True,
                stocks__isnull=False
            ).distinct()
            
            industries = DataIndustry.objects.filter(
                isActive=True,
                stocks__isnull=False
            ).distinct()
            
            # Phase 2: Collect all stocks across sectors and industries (unified approach)
            all_stocks_set = set()
            sector_stock_mapping = {}
            industry_stock_mapping = {}
            
            logger.info(f"Mapping stocks for {len(sectors)} sectors and {len(industries)} industries")
            
            for sector in sectors:
                sector_stocks = list(Stock.objects.filter(
                    sector_id=sector, 
                    is_active=True,
                    prices__date__gte=start_date.date(),
                    prices__date__lte=end_date.date()
                ).distinct())
                
                if sector_stocks:
                    sector_stock_mapping[sector.id] = sector_stocks
                    all_stocks_set.update(sector_stocks)
            
            for industry in industries:
                industry_stocks = list(Stock.objects.filter(
                    industry_id=industry, 
                    is_active=True,
                    prices__date__gte=start_date.date(),
                    prices__date__lte=end_date.date()
                ).distinct())
                
                if industry_stocks:
                    industry_stock_mapping[industry.id] = industry_stocks
                    all_stocks_set.update(industry_stocks)
            
            # Phase 2: Single unified data fetch for all stocks (major optimization)
            logger.info(f"Fetching unified price data for {len(all_stocks_set)} unique stocks")
            all_prices = StockPrice.objects.filter(
                stock__in=all_stocks_set,
                date__gte=start_date.date(),
                date__lte=end_date.date()
            ).select_related('stock').order_by('date', 'stock')
            
            # Phase 4: Cache the unified data for fast lookup
            prices_by_stock_date = cache.index_prices_by_stock_and_date(all_prices)
            price_dates = sorted(set(price.date for price in all_prices))
            logger.info(f"Indexed {len(all_prices)} price records across {len(price_dates)} dates")
            
            sector_prices_created = 0
            industry_prices_created = 0
            
            # PHASE 3: Parallel Processing with ThreadPoolExecutor
            # Determine optimal worker count (conservative approach)
            total_entities = len([s for s in sectors if s.id in sector_stock_mapping]) + \
                           len([i for i in industries if i.id in industry_stock_mapping])
            
            max_workers = min(4, max(1, total_entities // 2))  # Conservative threading
            logger.info(f"Using parallel processing with {max_workers} workers for {total_entities} entities")
            
            # Process sectors in parallel
            if sector_stock_mapping:
                sector_tasks = []
                for sector in sectors:
                    if sector.id in sector_stock_mapping:
                        sector_tasks.append({
                            'entity': sector,
                            'stocks': sector_stock_mapping[sector.id]
                        })
                
                if sector_tasks:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all sector tasks
                        future_to_entity = {
                            executor.submit(
                                self._processEntityParallel,
                                task_data,
                                'sector',
                                prices_by_stock_date,
                                price_dates,
                                cache
                            ): task_data['entity'].id for task_data in sector_tasks
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
                        industry_tasks.append({
                            'entity': industry,
                            'stocks': industry_stock_mapping[industry.id]
                        })
                
                if industry_tasks:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all industry tasks
                        future_to_entity = {
                            executor.submit(
                                self._processEntityParallel,
                                task_data,
                                'industry',
                                prices_by_stock_date,
                                price_dates,
                                cache
                            ): task_data['entity'].id for task_data in industry_tasks
                        }
                        
                        # Collect results as they complete
                        for future in as_completed(future_to_entity):
                            entity_name, created = future.result()
                            industry_prices_created += created
                            logger.debug(f"Industry task completed: {entity_name} -> {created} records")
            
            # Phase 4: Clean up cache
            cache.clear_cache()
            
            logger.info(f"Composite creation complete: {sector_prices_created} sector prices, {industry_prices_created} industry prices")
            
            return {
                'sector_prices_created': sector_prices_created,
                'industry_prices_created': industry_prices_created
            }
            
        except Exception as e:
            logger.error(f"Error creating sector/industry composites: {str(e)}")
            return {'sector_prices_created': 0, 'industry_prices_created': 0}
    
    def _createCompositesUnifiedBulk(self, entity, entity_type: str, stocks: List, 
                                   prices_by_stock_date: Dict, price_dates: List, 
                                   cache: CompositeCache) -> int:
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
            model_class = DataSectorPrice if entity_type == 'sector' else DataIndustryPrice
            entity_field = 'sector' if entity_type == 'sector' else 'industry'
            
            # Phase 1: Get existing records for this entity
            existing_records_qs = model_class.objects.filter(
                **{entity_field: entity},
                date__in=list(entity_prices_by_date.keys())
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
                    if record.constituents_count != composite_data['constituents_count']:
                        for key, value in composite_data.items():
                            setattr(record, key, value)
                        update_records.append(record)
                else:
                    # Prepare new record
                    record_data = {entity_field: entity, 'date': date}
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
                    updated_fields = ['close_index', 'constituents_count']
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

    def _processEntityParallel(self, entity_data: Dict, entity_type: str, 
                             prices_by_stock_date: Dict, price_dates: List, 
                             cache: CompositeCache) -> Tuple[str, int]:
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
            entity = entity_data['entity']
            stocks = entity_data['stocks']
            entity_name = f"{entity_type}:{entity.id}"
            
            # Ensure fresh database connection for this thread
            connection.ensure_connection()
            
            created = self._createCompositesUnifiedBulk(
                entity=entity,
                entity_type=entity_type,
                stocks=stocks,
                prices_by_stock_date=prices_by_stock_date,
                price_dates=price_dates,
                cache=cache
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

    def _createSectorComposites(self, sector: 'DataSector', start_date: datetime, end_date: datetime) -> int:
        """Create composite price records for a sector."""
        try:
            # Get all stocks in this sector with price data
            stocks = Stock.objects.filter(
                sector_id=sector,
                is_active=True,
                prices__date__gte=start_date.date(),
                prices__date__lte=end_date.date()
            ).distinct()
            
            if not stocks.exists():
                return 0
            
            # Get unique dates from stock prices in the date range
            price_dates = StockPrice.objects.filter(
                stock__in=stocks,
                date__gte=start_date.date(),
                date__lte=end_date.date()
            ).values_list('date', flat=True).distinct().order_by('date')
            
            prices_created = 0
            
            # Bulk fetch all price data for all dates at once (optimization fix)
            all_prices = StockPrice.objects.filter(
                stock__in=stocks,
                date__in=price_dates
            ).select_related('stock').order_by('date', 'stock')
            
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
                        sector=sector,
                        date=date,
                        defaults=composite_data
                    )
                    
                    if created:
                        prices_created += 1
                    elif sector_price.constituents_count != composite_data['constituents_count']:
                        # Update if constituent count changed
                        for key, value in composite_data.items():
                            setattr(sector_price, key, value)
                        sector_price.save()
            
            return prices_created
            
        except Exception as e:
            logger.error(f"Error creating sector composites for {sector.sectorName}: {str(e)}")
            return 0
    
    def _createIndustryComposites(self, industry: 'DataIndustry', start_date: datetime, end_date: datetime) -> int:
        """Create composite price records for an industry."""
        try:
            # Get all stocks in this industry with price data
            stocks = Stock.objects.filter(
                industry_id=industry,
                is_active=True,
                prices__date__gte=start_date.date(),
                prices__date__lte=end_date.date()
            ).distinct()
            
            if not stocks.exists():
                return 0
            
            # Get unique dates from stock prices in the date range
            price_dates = StockPrice.objects.filter(
                stock__in=stocks,
                date__gte=start_date.date(),
                date__lte=end_date.date()
            ).values_list('date', flat=True).distinct().order_by('date')
            
            prices_created = 0
            
            # Bulk fetch all price data for all dates at once (optimization fix)
            all_prices = StockPrice.objects.filter(
                stock__in=stocks,
                date__in=price_dates
            ).select_related('stock').order_by('date', 'stock')
            
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
                        industry=industry,
                        date=date,
                        defaults=composite_data
                    )
                    
                    if created:
                        prices_created += 1
                    elif industry_price.constituents_count != composite_data['constituents_count']:
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
            total_weighted_price = Decimal('0')
            total_market_cap = Decimal('0')
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
                method = 'cap_weighted'
            else:
                # Fallback to equal-weighted
                total_price = Decimal('0')
                constituent_count = 0
                
                for price in daily_prices:
                    adjusted_close = price.adjusted_close or price.close
                    total_price += adjusted_close
                    total_volume += price.volume
                    constituent_count += 1
                
                if constituent_count > 0:
                    close_index = total_price / Decimal(str(constituent_count))
                    method = 'equal_weighted'
                else:
                    return None
            
            return {
                'close_index': close_index,
                'volume_agg': total_volume,
                'constituents_count': constituent_count,
                'method': method,
                'data_source': 'yahoo'
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
                return {'error': 'No quote data available'}
            
            # Extract all required Stocks fields
            result = {
                'symbol': symbol.upper(),
                'currentPrice': info.get('currentPrice'),
                'previousClose': info.get('previousClose'),
                'open': info.get('open'),
                'dayLow': info.get('dayLow'),
                'dayHigh': info.get('dayHigh'),
                'regularMarketPrice': info.get('regularMarketPrice'),
                'regularMarketOpen': info.get('regularMarketOpen'),
                'regularMarketDayLow': info.get('regularMarketDayLow'),
                'regularMarketDayHigh': info.get('regularMarketDayHigh'),
                'regularMarketPreviousClose': info.get('regularMarketPreviousClose'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekChange': info.get('52WeekChange'),
                'fiftyDayAverage': info.get('fiftyDayAverage'),
                'twoHundredDayAverage': info.get('twoHundredDayAverage'),
                'beta': info.get('beta'),
                'impliedVolatility': info.get('impliedVolatility'),
                'volume': info.get('volume'),
                'regularMarketVolume': info.get('regularMarketVolume'),
                'averageVolume': info.get('averageVolume'),
                'averageVolume10days': info.get('averageVolume10days'),
                'averageVolume3months': info.get('averageVolume3months'),
                'updatedAt': timezone.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def fetchHistory(self, symbol: str, startDate: datetime, endDate: datetime, interval: str = '1d') -> List[Dict]:
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
            history = ticker.history(
                start=startDate,
                end=endDate,
                interval=interval
            )
            
            if history.empty:
                return []
            
            result = []
            for date, row in history.iterrows():
                try:
                    if pd.isna(row['Close']):
                        continue
                        
                    # Handle timezone
                    if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                        date_utc = date.tz_convert('UTC')
                    else:
                        date_utc = date.replace(tzinfo=dt_timezone.utc)
                    
                    result.append({
                        'symbol': symbol.upper(),
                        'date': date_utc,
                        'adjClose': Decimal(str(round(float(row['Close']), 2))),
                        'volume': int(row['Volume']) if pd.notna(row['Volume']) else 0,
                        'interval': interval,
                        'data_source': 'yahoo'
                    })
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
            symbol = row.get('symbol', '').upper()
            if not symbol:
                return False
            
            # Convert decimal fields safely
            decimal_fields = [
                'currentPrice', 'previousClose', 'dayLow', 'dayHigh',
                'regularMarketPrice', 'regularMarketOpen', 'regularMarketDayLow',
                'regularMarketDayHigh', 'regularMarketPreviousClose', 'fiftyTwoWeekLow',
                'fiftyTwoWeekHigh', 'fiftyTwoWeekChange', 'fiftyDayAverage',
                'twoHundredDayAverage', 'beta', 'impliedVolatility'
            ]
            
            integer_fields = [
                'volume', 'regularMarketVolume', 'averageVolume',
                'averageVolume10days', 'averageVolume3months'
            ]
            
            update_data = {'last_sync': row.get('updatedAt', timezone.now())}
            
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
            
            stock, created = Stock.objects.get_or_create(
                symbol=symbol,
                defaults=update_data
            )
            
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
            symbol = row.get('symbol', '').upper()
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
        
        if engine == 'sqlite':
            raise RuntimeError(
                "SQLite database detected. This analytics system requires PostgreSQL. "
                "Please configure PostgreSQL database in settings."
            )
        elif engine != 'postgresql':
            logger.warning(f"Unexpected database engine '{engine}'. Expected 'postgresql'.")
        
        logger.info(f"Database engine verified: {engine}")
    
    def backfill_eod_gaps(
        self,
        symbol: str,
        required_years: int = 2,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
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
            'success': False,
            'stock_backfilled': 0,
            'sector_backfilled': 0,
            'industry_backfilled': 0,
            'attempts_used': 0,
            'errors': []
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
                result['attempts_used'] = attempt
                logger.info(f"Backfill attempt {attempt}/{max_attempts} for {symbol}")
                
                try:
                    # Check current coverage
                    coverage = price_reader.check_data_coverage(symbol, required_years)
                    
                    # Backfill stock data if needed
                    if not coverage['stock']['has_data'] or coverage['stock']['gap_count'] > 50:
                        logger.info(f"Backfilling stock data for {symbol}")
                        stock_data = self.fetchStockEodHistory(symbol, start_date, end_date)
                        if stock_data:
                            # Convert to StockPrice format and upsert
                            stock_rows = []
                            for item in stock_data:
                                stock_rows.append({
                                    'symbol': symbol,
                                    'date': item['date'],
                                    'open': item['open'],
                                    'high': item['high'],
                                    'low': item['low'],
                                    'close': item['close'],
                                    'adjusted_close': item['adjusted_close'],
                                    'volume': item['volume'],
                                    'data_source': 'yahoo'
                                })
                            
                            backfilled = self._upsert_stock_prices(stock_rows)
                            result['stock_backfilled'] += backfilled
                            logger.info(f"Backfilled {backfilled} stock price records")
                    
                    # Create sector/industry classifications if stock was backfilled successfully
                    if result['stock_backfilled'] > 0:
                        try:
                            from Data.models import Stock
                            stock_record = Stock.objects.get(symbol=symbol)
                            
                            # Create classification record if sector/industry strings exist but FK relationships don't
                            if (stock_record.sector and stock_record.industry and 
                                not stock_record.sector_id and not stock_record.industry_id):
                                
                                # Normalize keys for classification
                                sector_key = self._normalizeSectorKey(stock_record.sector)
                                industry_key = self._normalizeIndustryKey(f"{stock_record.industry}_{stock_record.sector}")
                                
                                classification_row = {
                                    'symbol': symbol,
                                    'sectorKey': sector_key,
                                    'sectorName': stock_record.sector,
                                    'industryKey': industry_key,
                                    'industryName': stock_record.industry,
                                    'updatedAt': timezone.now()
                                }
                                
                                logger.info(f"Creating sector/industry classification for {symbol}")
                                self.upsertClassification([classification_row])
                                
                        except Exception as e:
                            logger.error(f"Error creating classification for {symbol}: {str(e)}")
                    
                    # Get sector/industry keys (should now exist after classification)
                    sector_key, industry_key = price_reader.get_stock_sector_industry_keys(symbol)
                    
                    # Backfill sector composite if needed and available
                    if sector_key and not coverage['sector']['has_data']:
                        logger.info(f"Backfilling sector composite for {sector_key}")
                        sector_result = self.composeSectorIndustryEod((start_date, end_date))
                        result['sector_backfilled'] += sector_result.get('sector_prices_created', 0)
                    
                    # Backfill industry composite if needed and available  
                    if industry_key and not coverage['industry']['has_data']:
                        logger.info(f"Backfilling industry composite for {industry_key}")
                        industry_result = self.composeSectorIndustryEod((start_date, end_date))
                        result['industry_backfilled'] += industry_result.get('industry_prices_created', 0)
                    
                    # Check if gaps are adequately filled
                    final_coverage = price_reader.check_data_coverage(symbol, required_years)
                    
                    stock_adequate = (final_coverage['stock']['has_data'] and 
                                    final_coverage['stock']['gap_count'] <= 50)
                    
                    sector_adequate = (not sector_key or 
                                     (final_coverage['sector']['has_data'] and 
                                      final_coverage['sector']['gap_count'] <= 50))
                    
                    industry_adequate = (not industry_key or 
                                       (final_coverage['industry']['has_data'] and 
                                        final_coverage['industry']['gap_count'] <= 50))
                    
                    if stock_adequate and sector_adequate and industry_adequate:
                        result['success'] = True
                        logger.info(f"Backfill complete for {symbol} after {attempt} attempts")
                        break
                    else:
                        logger.info(f"Gaps still exist after attempt {attempt}, retrying...")
                        
                except Exception as e:
                    error_msg = f"Attempt {attempt} failed: {str(e)}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
                    
                    # Add exponential backoff with jitter
                    if attempt < max_attempts:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        logger.info(f"Waiting {delay:.2f}s before retry...")
                        time.sleep(delay)
            
            if not result['success']:
                error_msg = f"Failed to adequately backfill {symbol} after {max_attempts} attempts"
                result['errors'].append(error_msg)
                logger.error(error_msg)
            
            return result
            
        except Exception as e:
            error_msg = f"Critical error in backfill_eod_gaps: {str(e)}"
            result['errors'].append(error_msg)
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
                        stock = Stock.objects.get(symbol=row['symbol'].upper())
                        
                        price, created = StockPrice.objects.get_or_create(
                            stock=stock,
                            date=row['date'],
                            defaults={
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'adjusted_close': row['adjusted_close'],
                                'volume': row['volume'],
                                'data_source': row['data_source']
                            }
                        )
                        
                        if not created:
                            # Update existing record if data is different
                            price.open = row['open']
                            price.high = row['high']
                            price.low = row['low']
                            price.close = row['close']
                            price.adjusted_close = row['adjusted_close']
                            price.volume = row['volume']
                            price.data_source = row['data_source']
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
                    symbol = row.get('symbol', '').upper()
                    date = row.get('date')
                    
                    if not symbol or not date:
                        continue
                    
                    try:
                        stock = Stock.objects.get(symbol=symbol)
                        
                        # Convert datetime to date if needed
                        if isinstance(date, datetime):
                            date = date.date()
                        
                        price_data = {
                            'adjusted_close': row.get('adjClose'),
                            'volume': row.get('volume', 0),
                            'data_source': row.get('data_source', 'yahoo')
                        }
                        
                        # Use existing close for required fields if not provided
                        if 'adjClose' in row:
                            adj_close = row['adjClose']
                            price_data.update({
                                'open': adj_close,
                                'high': adj_close,
                                'low': adj_close,
                                'close': adj_close
                            })
                        
                        price, created = StockPrice.objects.get_or_create(
                            stock=stock,
                            date=date,
                            defaults=price_data
                        )
                        
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