"""
Yahoo Finance Integration Module
Handles all interactions with Yahoo Finance API for VoyageurCompass.
This module acts as the main interface for Yahoo Finance operations.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

from Data.services.provider import data_provider
from Data.services.synchronizer import data_synchronizer
from django.db import models
from Data.models import Stock, StockPrice

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """
    Main service class for Yahoo Finance API integration.
    Coordinates between provider, synchronizer, and database.
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance service."""
        self.provider = data_provider
        self.synchronizer = data_synchronizer
        self.timeout = 30  # Default timeout
        logger.info("Yahoo Finance Service initialized with yfinance integration")
    
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
        validPeriods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in validPeriods:
            raise ValueError(f"Invalid period: {period}. Must be one of {validPeriods}")
        return period
    
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
                                'change_amount': float(p.change_amount) if p.change_amount else None,
                                'change_percent': float(p.change_percent) if p.change_percent else None,
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
            if not results and len(query) <= 5:
                if self.provider.validate_symbol(query):
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


# Singleton instance
yahoo_finance_service = YahooFinanceService()