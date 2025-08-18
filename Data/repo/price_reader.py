"""
Price data reader for EOD (End of Day) stock, sector, and industry data.
Provides typed interfaces for accessing historical price data required by analytics engine.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple, NamedTuple, Any
from django.db.models import QuerySet, Min, Max, Count
from django.utils import timezone

from Data.models import Stock, StockPrice, DataSector, DataSectorPrice, DataIndustry, DataIndustryPrice


class PriceData(NamedTuple):
    """Structure for OHLCV price data."""
    date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    adjusted_close: Optional[Decimal]
    volume: int


class SectorPriceData(NamedTuple):
    """Structure for sector composite price data."""
    date: date
    close_index: Decimal
    fifty_day_average: Optional[Decimal]
    two_hundred_day_average: Optional[Decimal]
    volume_agg: int
    constituents_count: int


class IndustryPriceData(NamedTuple):
    """Structure for industry composite price data."""
    date: date
    close_index: Decimal
    fifty_day_average: Optional[Decimal]
    two_hundred_day_average: Optional[Decimal]
    volume_agg: int
    constituents_count: int


class PriceReader:
    """
    Repository for reading EOD price data with PostgreSQL optimization.
    """
    
    def __init__(self):
        """Initialize the price reader."""
        pass
    
    def get_stock_prices(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: Optional[date] = None
    ) -> List[PriceData]:
        """
        Get stock price history for specified date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for price data
            end_date: End date for price data (defaults to today)
            
        Returns:
            List of PriceData tuples ordered by date ascending
            
        Raises:
            Stock.DoesNotExist: If stock symbol not found after retries
        """
        if end_date is None:
            end_date = timezone.now().date()
        
        symbol_upper = symbol.upper()
        
        # Retry logic to handle race conditions and transaction issues
        for attempt in range(3):
            try:
                # Robust stock lookup with retry logic
                stock = Stock.objects.get(symbol=symbol_upper)
                
                prices_qs = stock.prices.filter(
                    date__gte=start_date,
                    date__lte=end_date
                ).order_by('date').select_related('stock')
                
                return [
                    PriceData(
                        date=price.date,
                        open=price.open,
                        high=price.high,
                        low=price.low,
                        close=price.close,
                        adjusted_close=price.adjusted_close,
                        volume=price.volume
                    )
                    for price in prices_qs
                ]
                
            except Stock.DoesNotExist:
                if attempt < 2:  # Retry for first 2 attempts
                    import time
                    time.sleep(0.5)  # Brief pause before retry
                    # Try again - might be a transient database issue
                    continue
                else:
                    # Final attempt failed, re-raise the exception
                    raise
            except Exception as e:
                if attempt < 2:
                    import time
                    time.sleep(0.5)
                    continue
                else:
                    raise
    
    def get_sector_prices(
        self,
        sector_key: str,
        start_date: date,
        end_date: Optional[date] = None
    ) -> List[SectorPriceData]:
        """
        Get sector composite price history for specified date range.
        
        Args:
            sector_key: Normalized sector key
            start_date: Start date for price data
            end_date: End date for price data (defaults to today)
            
        Returns:
            List of SectorPriceData tuples ordered by date ascending
            
        Raises:
            DataSector.DoesNotExist: If sector not found
        """
        if end_date is None:
            end_date = timezone.now().date()
            
        sector = DataSector.objects.get(sectorKey=sector_key)
        
        prices_qs = sector.prices.filter(
            date__gte=start_date,
            date__lte=end_date
        ).order_by('date').select_related('sector')
        
        return [
            SectorPriceData(
                date=price.date,
                close_index=price.close_index,
                fifty_day_average=price.fiftyDayAverage,
                two_hundred_day_average=price.twoHundredDayAverage,
                volume_agg=price.volume_agg,
                constituents_count=price.constituents_count
            )
            for price in prices_qs
        ]
    
    def get_industry_prices(
        self,
        industry_key: str,
        start_date: date,
        end_date: Optional[date] = None
    ) -> List[IndustryPriceData]:
        """
        Get industry composite price history for specified date range.
        
        Args:
            industry_key: Normalized industry key
            start_date: Start date for price data
            end_date: End date for price data (defaults to today)
            
        Returns:
            List of IndustryPriceData tuples ordered by date ascending
            
        Raises:
            DataIndustry.DoesNotExist: If industry not found
        """
        if end_date is None:
            end_date = timezone.now().date()
            
        industry = DataIndustry.objects.get(industryKey=industry_key)
        
        prices_qs = industry.prices.filter(
            date__gte=start_date,
            date__lte=end_date
        ).order_by('date').select_related('industry')
        
        return [
            IndustryPriceData(
                date=price.date,
                close_index=price.close_index,
                fifty_day_average=price.fiftyDayAverage,
                two_hundred_day_average=price.twoHundredDayAverage,
                volume_agg=price.volume_agg,
                constituents_count=price.constituents_count
            )
            for price in prices_qs
        ]
    
    def get_stock_sector_industry_keys(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get normalized sector and industry keys for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Tuple of (sector_key, industry_key), either may be None
            
        Raises:
            Stock.DoesNotExist: If stock symbol not found
        """
        stock = Stock.objects.select_related('sector_id', 'industry_id').get(symbol=symbol.upper())
        
        sector_key = stock.sector_id.sectorKey if stock.sector_id else None
        industry_key = stock.industry_id.industryKey if stock.industry_id else None
        
        return sector_key, industry_key
    
    def check_data_coverage(
        self,
        symbol: str,
        required_years: int = 2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Check 2-year EOD data coverage for stock, sector, and industry.
        
        Args:
            symbol: Stock ticker symbol
            required_years: Number of years of history required
            
        Returns:
            Dict with coverage info:
            {
                'stock': {'has_data': bool, 'earliest_date': date, 'latest_date': date, 'gap_count': int},
                'sector': {'has_data': bool, 'earliest_date': date, 'latest_date': date, 'gap_count': int},
                'industry': {'has_data': bool, 'earliest_date': date, 'latest_date': date, 'gap_count': int}
            }
        """
        result = {
            'stock': {'has_data': False, 'earliest_date': None, 'latest_date': None, 'gap_count': 0},
            'sector': {'has_data': False, 'earliest_date': None, 'latest_date': None, 'gap_count': 0},
            'industry': {'has_data': False, 'earliest_date': None, 'latest_date': None, 'gap_count': 0}
        }
        
        # Calculate required start date
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=required_years * 365 + 30)  # Add buffer for leap years
        
        try:
            stock = Stock.objects.select_related('sector_id', 'industry_id').get(symbol=symbol.upper())
        except Stock.DoesNotExist:
            return result
        
        # Check stock data coverage
        stock_prices = stock.prices.filter(date__gte=start_date).aggregate(
            earliest=Min('date'),
            latest=Max('date'),
            count=Count('id')
        )
        
        if stock_prices['count'] > 0:
            result['stock']['has_data'] = True
            result['stock']['earliest_date'] = stock_prices['earliest']
            result['stock']['latest_date'] = stock_prices['latest']
            
            # Estimate gaps (rough calculation)
            expected_trading_days = self._estimate_trading_days(start_date, end_date)
            result['stock']['gap_count'] = max(0, expected_trading_days - stock_prices['count'])
        
        # Check sector data coverage
        if stock.sector_id:
            sector_prices = stock.sector_id.prices.filter(date__gte=start_date).aggregate(
                earliest=Min('date'),
                latest=Max('date'),
                count=Count('id')
            )
            
            if sector_prices['count'] > 0:
                result['sector']['has_data'] = True
                result['sector']['earliest_date'] = sector_prices['earliest']
                result['sector']['latest_date'] = sector_prices['latest']
                
                expected_trading_days = self._estimate_trading_days(start_date, end_date)
                result['sector']['gap_count'] = max(0, expected_trading_days - sector_prices['count'])
        
        # Check industry data coverage
        if stock.industry_id:
            industry_prices = stock.industry_id.prices.filter(date__gte=start_date).aggregate(
                earliest=Min('date'),
                latest=Max('date'),
                count=Count('id')
            )
            
            if industry_prices['count'] > 0:
                result['industry']['has_data'] = True
                result['industry']['earliest_date'] = industry_prices['earliest']
                result['industry']['latest_date'] = industry_prices['latest']
                
                expected_trading_days = self._estimate_trading_days(start_date, end_date)
                result['industry']['gap_count'] = max(0, expected_trading_days - industry_prices['count'])
        
        return result
    
    def _estimate_trading_days(self, start_date: date, end_date: date) -> int:
        """
        Estimate number of trading days between two dates.
        Rough approximation: 252 trading days per year.
        """
        total_days = (end_date - start_date).days
        return int(total_days * 252 / 365)
    
    def get_latest_price_date(self, symbol: str) -> Optional[date]:
        """
        Get the latest available price date for a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Latest date with price data, or None if no data
        """
        try:
            stock = Stock.objects.get(symbol=symbol.upper())
            latest_price = stock.prices.order_by('-date').first()
            return latest_price.date if latest_price else None
        except Stock.DoesNotExist:
            return None