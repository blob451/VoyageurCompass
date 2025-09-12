"""
Multi-Source Data Fetcher Service

Provides robust stock data fetching with multiple fallback data sources.
Handles API failures gracefully and maximizes data coverage.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from django.conf import settings
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Single price data point"""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str


@dataclass
class FetchResult:
    """Result of data fetching operation"""
    success: bool
    data: List[DataPoint]
    source: str
    error: Optional[str] = None
    days_fetched: int = 0
    api_calls: int = 0
    cache_hits: int = 0


class MultiSourceDataFetcher:
    """
    Multi-source stock data fetcher with intelligent fallback strategy.
    Attempts multiple data sources to maximize historical data coverage.
    """
    
    def __init__(self):
        self.sources = [
            'yahoo',
            'alphavantage', 
            'iex',
            'polygon'
        ]
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
        
    def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        min_days: int = 20
    ) -> FetchResult:
        """
        Fetch historical data using multiple sources with intelligent fallback.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data  
            min_days: Minimum acceptable days of data
            
        Returns:
            FetchResult with best available data
        """
        logger.info(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        best_result = FetchResult(success=False, data=[], source='none', error="No sources succeeded")
        
        # Try each source in priority order
        for source in self.sources:
            try:
                logger.info(f"Attempting {source} for {symbol}")
                result = self._fetch_from_source(source, symbol, start_date, end_date)
                
                if result.success and result.days_fetched >= min_days:
                    logger.info(f"Success with {source}: {result.days_fetched} days")
                    return result
                elif result.success and result.days_fetched > best_result.days_fetched:
                    logger.info(f"Partial success with {source}: {result.days_fetched} days (best so far)")
                    best_result = result
                else:
                    logger.warning(f"Failed with {source}: {result.error}")
                    
            except Exception as e:
                logger.error(f"Exception with {source} for {symbol}: {str(e)}")
                continue
                
        # Return best result even if it doesn't meet min_days
        if best_result.days_fetched > 0:
            logger.info(f"Returning best available: {best_result.days_fetched} days from {best_result.source}")
            best_result.success = True
        
        return best_result
        
    def _fetch_from_source(
        self, 
        source: str, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> FetchResult:
        """Fetch data from a specific source."""
        
        if source == 'yahoo':
            return self._fetch_yahoo(symbol, start_date, end_date)
        elif source == 'alphavantage':
            return self._fetch_alphavantage(symbol, start_date, end_date)
        elif source == 'iex':
            return self._fetch_iex(symbol, start_date, end_date)
        elif source == 'polygon':
            return self._fetch_polygon(symbol, start_date, end_date)
        else:
            return FetchResult(success=False, data=[], source=source, error=f"Unknown source: {source}")
            
    def _fetch_yahoo(self, symbol: str, start_date: datetime, end_date: datetime) -> FetchResult:
        """Fetch data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try different periods to maximize data
            periods_to_try = ['2y', '5y', 'max']
            best_data = []
            
            for period in periods_to_try:
                try:
                    data = ticker.history(period=period)
                    if len(data) > len(best_data):
                        best_data = data
                        if len(data) >= 500:  # Good enough for most analyses
                            break
                except:
                    continue
            
            if best_data.empty:
                # Fallback to date range
                data = ticker.history(start=start_date, end=end_date)
                best_data = data
                
            if best_data.empty:
                return FetchResult(success=False, data=[], source='yahoo', 
                                 error="No data returned from Yahoo Finance")
            
            # Convert to DataPoint objects
            data_points = []
            for date, row in best_data.iterrows():
                data_points.append(DataPoint(
                    date=date.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    source='yahoo'
                ))
            
            return FetchResult(
                success=True,
                data=data_points,
                source='yahoo',
                days_fetched=len(data_points),
                api_calls=1
            )
            
        except Exception as e:
            return FetchResult(success=False, data=[], source='yahoo', 
                             error=f"Yahoo Finance error: {str(e)}")
            
    def _fetch_alphavantage(self, symbol: str, start_date: datetime, end_date: datetime) -> FetchResult:
        """Fetch data from Alpha Vantage (requires API key)."""
        api_key = getattr(settings, 'ALPHAVANTAGE_API_KEY', None)
        if not api_key:
            return FetchResult(success=False, data=[], source='alphavantage', 
                             error="Alpha Vantage API key not configured")
        
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Error Message' in data:
                return FetchResult(success=False, data=[], source='alphavantage', 
                                 error=data['Error Message'])
            
            if 'Time Series (Daily)' not in data:
                return FetchResult(success=False, data=[], source='alphavantage', 
                                 error="No time series data in response")
            
            time_series = data['Time Series (Daily)']
            data_points = []
            
            for date_str, values in time_series.items():
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                if start_date <= date_obj <= end_date:
                    data_points.append(DataPoint(
                        date=date_obj,
                        open=float(values['1. open']),
                        high=float(values['2. high']),
                        low=float(values['3. low']),
                        close=float(values['5. adjusted close']),
                        volume=int(values['6. volume']),
                        source='alphavantage'
                    ))
            
            return FetchResult(
                success=True,
                data=sorted(data_points, key=lambda x: x.date),
                source='alphavantage',
                days_fetched=len(data_points),
                api_calls=1
            )
            
        except Exception as e:
            return FetchResult(success=False, data=[], source='alphavantage', 
                             error=f"Alpha Vantage error: {str(e)}")
            
    def _fetch_iex(self, symbol: str, start_date: datetime, end_date: datetime) -> FetchResult:
        """Fetch data from IEX Cloud (requires API key)."""
        api_key = getattr(settings, 'IEX_API_KEY', None)
        if not api_key:
            return FetchResult(success=False, data=[], source='iex', 
                             error="IEX API key not configured")
        
        try:
            # IEX Cloud historical data endpoint
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/2y"
            params = {'token': api_key}
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                return FetchResult(success=False, data=[], source='iex', 
                                 error=f"IEX API error: {response.status_code}")
            
            data = response.json()
            data_points = []
            
            for item in data:
                date_obj = datetime.strptime(item['date'], '%Y-%m-%d')
                if start_date <= date_obj <= end_date:
                    data_points.append(DataPoint(
                        date=date_obj,
                        open=float(item['open']),
                        high=float(item['high']),
                        low=float(item['low']),
                        close=float(item['close']),
                        volume=int(item['volume']),
                        source='iex'
                    ))
            
            return FetchResult(
                success=True,
                data=data_points,
                source='iex',
                days_fetched=len(data_points),
                api_calls=1
            )
            
        except Exception as e:
            return FetchResult(success=False, data=[], source='iex', 
                             error=f"IEX error: {str(e)}")
            
    def _fetch_polygon(self, symbol: str, start_date: datetime, end_date: datetime) -> FetchResult:
        """Fetch data from Polygon.io (requires API key)."""
        api_key = getattr(settings, 'POLYGON_API_KEY', None)
        if not api_key:
            return FetchResult(success=False, data=[], source='polygon', 
                             error="Polygon API key not configured")
        
        try:
            # Polygon aggregates endpoint
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_str}/{end_str}"
            params = {'apikey': api_key}
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                return FetchResult(success=False, data=[], source='polygon', 
                                 error=f"Polygon API error: {response.status_code}")
            
            data = response.json()
            
            if 'results' not in data:
                return FetchResult(success=False, data=[], source='polygon', 
                                 error="No results in Polygon response")
            
            data_points = []
            
            for item in data['results']:
                # Polygon timestamps are in milliseconds
                date_obj = datetime.fromtimestamp(item['t'] / 1000)
                data_points.append(DataPoint(
                    date=date_obj,
                    open=float(item['o']),
                    high=float(item['h']),
                    low=float(item['l']),
                    close=float(item['c']),
                    volume=int(item['v']),
                    source='polygon'
                ))
            
            return FetchResult(
                success=True,
                data=data_points,
                source='polygon',
                days_fetched=len(data_points),
                api_calls=1
            )
            
        except Exception as e:
            return FetchResult(success=False, data=[], source='polygon', 
                             error=f"Polygon error: {str(e)}")
            
    def get_available_data_range(self, symbol: str) -> Dict[str, Any]:
        """
        Check data availability across all sources without fetching full datasets.
        
        Returns:
            Dictionary with availability information per source
        """
        availability = {}
        
        for source in self.sources:
            try:
                if source == 'yahoo':
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    availability[source] = {
                        'available': bool(info),
                        'estimated_range': 'Up to 10+ years' if info else 'Unknown'
                    }
                else:
                    # For other sources, we'd need to make test calls
                    # For now, mark as potentially available if API keys exist
                    api_key = getattr(settings, f'{source.upper()}_API_KEY', None)
                    availability[source] = {
                        'available': api_key is not None,
                        'estimated_range': '2+ years' if api_key else 'API key not configured'
                    }
                    
            except Exception as e:
                availability[source] = {
                    'available': False,
                    'error': str(e)
                }
                
        return availability


# Global service instance
multi_source_fetcher = MultiSourceDataFetcher()