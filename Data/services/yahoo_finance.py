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
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from decimal import Decimal

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
from Data.models import Stock, StockPrice, PriceBar

logger = logging.getLogger(__name__)


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
            
            # Try using yf.download with custom session for better connection handling
            try:
                # Ensure session is available
                self._ensure_session()
                # Try with session parameter (newer yfinance versions)
                history = yf.download(
                    ticker, 
                    period=period, 
                    interval=interval, 
                    auto_adjust=False,  # Preserve unadjusted close prices for consistency
                    prepost=False,
                    threads=False,
                    progress=False,
                    session=self.session  # Use our custom session with headers
                )
            except (TypeError, AttributeError):
                # Fallback for older yfinance versions that don't support session parameter
                logger.debug(f"yfinance session parameter not supported, using default session for {ticker}")
                history = yf.download(
                    ticker, 
                    period=period, 
                    interval=interval, 
                    auto_adjust=False,
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
                        date_utc = date.replace(tzinfo=timezone.utc)
                    
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
    
    def saveBars(self, bars: List[Dict]) -> Tuple[int, int]:
        """
        Save bars to database using bulk operations.
        Returns (created_count, skipped_count).
        """
        from Data.models import Stock, PriceBar
        
        if not bars:
            return 0, 0
        
        # Ensure stocks exist using bulk operations for efficiency
        symbols = list(set(bar['symbol'] for bar in bars))
        
        # Query existing stocks in one database call
        existing_stocks = Stock.objects.filter(symbol__in=symbols)
        existing_symbols = set(stock.symbol for stock in existing_stocks)
        
        # Identify missing symbols and bulk create them
        missing_symbols = set(symbols) - existing_symbols
        if missing_symbols:
            stocks_to_create = [
                Stock(symbol=symbol, short_name=symbol, data_source='yahoo')
                for symbol in missing_symbols
            ]
            
            try:
                # Attempt bulk create without ignoring conflicts first
                created_stocks = Stock.objects.bulk_create(stocks_to_create, ignore_conflicts=False)
                logger.info(f"Successfully created {len(created_stocks)} new Stock records: {list(missing_symbols)}")
            except Exception as e:
                # Handle conflicts explicitly with detailed logging
                logger.warning(f"Conflicts detected during Stock creation for symbols {list(missing_symbols)}: {str(e)}")
                
                # Retry with ignore_conflicts to handle race conditions
                try:
                    created_stocks = Stock.objects.bulk_create(stocks_to_create, ignore_conflicts=True)
                    actual_created = len(created_stocks)
                    skipped = len(stocks_to_create) - actual_created
                    
                    if skipped > 0:
                        logger.info(f"Stock creation: {actual_created} created, {skipped} skipped due to conflicts")
                        # Log which specific symbols may have been skipped
                        if actual_created == 0:
                            logger.debug(f"All Stock symbols already existed: {list(missing_symbols)}")
                    else:
                        logger.info(f"Successfully created {actual_created} Stock records after retry")
                        
                except Exception as retry_error:
                    logger.error(f"Failed to create Stock records even with ignore_conflicts: {str(retry_error)}")
                    raise
        
        # Create a stock lookup dictionary for efficient access
        all_stocks = Stock.objects.filter(symbol__in=symbols)
        stock_lookup = {stock.symbol: stock for stock in all_stocks}
        
        # Bulk create with conflict handling
        price_bars = []
        skipped_bars = 0
        missing_stocks = {}  # Track missing stocks with sample dates for diagnosis
        
        for bar in bars:
            # Safely check if stock exists in lookup to avoid KeyError
            stock = stock_lookup.get(bar['symbol'])
            if stock is None:
                # Aggregate missing stock info instead of logging per bar
                symbol = bar['symbol']
                if symbol not in missing_stocks:
                    missing_stocks[symbol] = []
                # Store sample dates for diagnosis (limit to 3 samples per symbol)
                if len(missing_stocks[symbol]) < 3:
                    missing_stocks[symbol].append(bar.get('date', 'unknown date'))
                skipped_bars += 1
                continue
                
            price_bars.append(PriceBar(
                stock=stock,
                date=bar['date'],
                interval=bar['interval'],
                open=bar['open'],
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                volume=bar['volume'],
                data_source=bar['data_source']
            ))
        
        try:
            # Use ignore_conflicts from the start to handle duplicates gracefully
            with transaction.atomic():
                created = PriceBar.objects.bulk_create(
                    price_bars,
                    ignore_conflicts=True,
                    batch_size=500
                )
                
            created_count = len(created)
            skipped_count = len(price_bars) - created_count
            
            if skipped_count > 0:
                logger.info(f"PriceBar creation: {created_count} created, {skipped_count} skipped due to duplicates")
            else:
                logger.info(f"Successfully created all {created_count} PriceBar records")
                
        except Exception as e:
            logger.error(f"Failed to create PriceBar records: {str(e)}")
            raise
            
        # Include skipped bars due to missing stocks in the final count
        total_skipped = skipped_count + skipped_bars
        
        # Log aggregated missing stock information to avoid log noise
        if missing_stocks:
            missing_count = len(missing_stocks)
            sample_symbols = list(missing_stocks.keys())[:5]  # Show up to 5 symbols
            sample_info = []
            
            for symbol in sample_symbols:
                dates = missing_stocks[symbol]
                date_str = ', '.join(str(d) for d in dates)
                if len(missing_stocks[symbol]) == 3:
                    date_str += "..."  # Indicate more dates exist
                sample_info.append(f"{symbol} (dates: {date_str})")
            
            logger.warning(f"Skipped {skipped_bars} bars due to {missing_count} missing stocks. "
                          f"This may indicate race conditions or symbol normalization issues. "
                          f"Sample symbols: {'; '.join(sample_info)}")
            
            if missing_count > 5:
                remaining = missing_count - 5
                logger.debug(f"Additional {remaining} missing symbols not shown in sample")
        
        if skipped_bars > 0:
            logger.info(f"Final result: {created_count} bars saved, {skipped_count} duplicates skipped, {skipped_bars} skipped due to missing stocks")
        else:
            logger.info(f"Final result: {created_count} bars saved, {skipped_count} duplicates skipped")
            
        return created_count, total_skipped

    # =====================================================================
    # Sector/Industry Data Fetching Methods
    # =====================================================================
    
    def fetchSectorIndustryBatch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch sector/industry data for multiple symbols with rate limit handling.
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            Dictionary with symbol as key and sector/industry data as value
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
                        results[symbol] = data
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
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing sector/industry data
        """
        try:
            # Validate symbol
            symbol = self.validateSymbol(symbol)
            
            logger.info(f"Fetching sector/industry data for {symbol}")
            
            # Create yfinance Ticker object with custom session
            try:
                # Ensure session is available
                self._ensure_session()
                # Try with session parameter (newer yfinance versions)
                ticker = yf.Ticker(symbol, session=self.session)
            except (TypeError, AttributeError):
                # Fallback for older yfinance versions that don't support session parameter
                logger.debug(f"yfinance Ticker session parameter not supported, using default session for {symbol}")
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
        from django.utils import timezone
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
                created, skipped = service.saveBars(data)
                # Session is automatically closed after this block
    
    For general use throughout the application, prefer the singleton yahoo_finance_service.
    """
    return YahooFinanceService()