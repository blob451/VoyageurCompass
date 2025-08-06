"""
Data Synchronizer Service Module
Manages data synchronization between Yahoo Finance and the database for VoyageurCompass.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from django.db import transaction
from django.utils import timezone

from Data.models import Stock, StockPrice
from Data.services.provider import data_provider

logger = logging.getLogger(__name__)


class DataSynchronizer:
    """
    Service class for synchronizing market data with the database.
    """
    
    def __init__(self):
        """Initialize the Data Synchronizer."""
        self.provider = data_provider
        logger.info("Data Synchronizer Service initialized")
    
    @transaction.atomic
    def sync_stock_data(self, symbol: str, period: str = "1mo") -> Dict:
        """
        Synchronize stock data from Yahoo Finance to the database.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period for historical data
        
        Returns:
            Dictionary with sync results
        """
        try:
            logger.info(f"Starting sync for {symbol}")
            
            # Fetch data from Yahoo Finance
            stock_data = self.provider.fetch_stock_data(symbol, period)
            
            if not stock_data.get('success', False):
                error_msg = stock_data.get('error', 'Unknown error')
                logger.error(f"Failed to fetch data for {symbol}: {error_msg}")
                return {
                    'success': False,
                    'symbol': symbol,
                    'error': error_msg
                }
            
            # Create or update Stock record
            stock, created = self._update_stock_record(symbol, stock_data['info'])
            
            # Sync historical price data
            prices_synced = self._sync_price_history(stock, stock_data['history'])
            
            # Update last sync timestamp
            stock.last_sync = timezone.now()
            stock.save(update_fields=['last_sync'])
            
            result = {
                'success': True,
                'symbol': symbol,
                'stock_created': created,
                'prices_synced': prices_synced,
                'total_prices': StockPrice.objects.filter(stock=stock).count(),
                'sync_time': datetime.now().isoformat()
            }
            
            logger.info(f"Sync completed for {symbol}: {prices_synced} prices synced")
            return result
            
        except Exception as e:
            logger.error(f"Error syncing stock data for {symbol}: {str(e)}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }
    
    def _update_stock_record(self, symbol: str, info: Dict) -> Tuple[Stock, bool]:
        """
        Create or update a Stock record in the database.
        
        Args:
            symbol: Stock ticker symbol
            info: Stock information dictionary
        
        Returns:
            Tuple of (Stock instance, created_flag)
        """
        stock, created = Stock.objects.get_or_create(
            symbol=symbol.upper(),
            defaults={
                'short_name': info.get('shortName', '')[:100],
                'long_name': info.get('longName', '')[:255],
                'currency': info.get('currency', 'USD')[:10],
                'exchange': info.get('exchange', '')[:50],
                'sector': info.get('sector', '')[:100],
                'industry': info.get('industry', '')[:100],
                'market_cap': info.get('marketCap', 0) or 0,
            }
        )
        
        if not created:
            # Update existing stock record
            stock.short_name = info.get('shortName', stock.short_name)[:100]
            stock.long_name = info.get('longName', stock.long_name)[:255]
            stock.currency = info.get('currency', stock.currency)[:10]
            stock.exchange = info.get('exchange', stock.exchange)[:50]
            stock.sector = info.get('sector', stock.sector)[:100]
            stock.industry = info.get('industry', stock.industry)[:100]
            stock.market_cap = info.get('marketCap', stock.market_cap) or 0
            stock.save()
        
        action = "created" if created else "updated"
        logger.info(f"Stock record {action} for {symbol}")
        
        return stock, created
    
    def _sync_price_history(self, stock: Stock, history: List[Dict]) -> int:
        """
        Synchronize historical price data for a stock.
        
        Args:
            stock: Stock instance
            history: List of historical price dictionaries
        
        Returns:
            Number of price records created/updated
        """
        prices_synced = 0
        
        for price_data in history:
            try:
                # Parse date
                date_str = price_data['date']
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                # Create or update price record
                price, created = StockPrice.objects.update_or_create(
                    stock=stock,
                    date=date_obj,
                    defaults={
                        'open': Decimal(str(price_data['open'])),
                        'high': Decimal(str(price_data['high'])),
                        'low': Decimal(str(price_data['low'])),
                        'close': Decimal(str(price_data['close'])),
                        'volume': price_data['volume'],
                    }
                )
                
                if created or price:
                    prices_synced += 1
                    
            except Exception as e:
                logger.warning(f"Error syncing price for {stock.symbol} on {price_data.get('date')}: {str(e)}")
                continue
        
        return prices_synced
    
    def sync_multiple_stocks(self, symbols: List[str], period: str = "1mo") -> Dict:
        """
        Synchronize data for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period for historical data
        
        Returns:
            Dictionary with sync results for each symbol
        """
        results = {}
        total_success = 0
        total_failed = 0
        
        for symbol in symbols:
            logger.info(f"Syncing {symbol} ({symbols.index(symbol) + 1}/{len(symbols)})")
            result = self.sync_stock_data(symbol, period)
            results[symbol] = result
            
            if result['success']:
                total_success += 1
            else:
                total_failed += 1
        
        summary = {
            'total_symbols': len(symbols),
            'successful': total_success,
            'failed': total_failed,
            'results': results,
            'sync_time': datetime.now().isoformat()
        }
        
        logger.info(f"Batch sync completed: {total_success} successful, {total_failed} failed")
        return summary
    
    def sync_portfolio_stocks(self, user_id: int) -> Dict:
        """
        Sync all stocks in a user's portfolios.
        
        Args:
            user_id: User ID
        
        Returns:
            Dictionary with sync results
        """
        try:
            from Data.models import Portfolio
            
            # Get all unique stocks in user's portfolios
            portfolios = Portfolio.objects.filter(user_id=user_id, is_active=True)
            stock_symbols = set()
            
            for portfolio in portfolios:
                for holding in portfolio.holdings.all():
                    stock_symbols.add(holding.stock.symbol)
            
            if not stock_symbols:
                return {
                    'success': True,
                    'message': 'No stocks to sync',
                    'stocks_synced': 0
                }
            
            # Sync all stocks
            results = self.sync_multiple_stocks(list(stock_symbols))
            
            return {
                'success': True,
                'stocks_synced': len(stock_symbols),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error syncing portfolio stocks for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_realtime_prices(self, symbols: List[str]) -> Dict:
        """
        Update real-time prices for a list of stocks.
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            Dictionary with update results
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Get current price from Yahoo Finance
                current_price = self.provider.fetch_realtime_price(symbol)
                
                if current_price:
                    # Update today's price record
                    stock = Stock.objects.get(symbol=symbol.upper())
                    today = datetime.now().date()
                    
                    StockPrice.objects.update_or_create(
                        stock=stock,
                        date=today,
                        defaults={
                            'close': Decimal(str(current_price)),
                            # For real-time updates, we might not have OHLC data
                            'open': Decimal(str(current_price)),
                            'high': Decimal(str(current_price)),
                            'low': Decimal(str(current_price)),
                        }
                    )
                    
                    results[symbol] = {
                        'success': True,
                        'price': current_price
                    }
                else:
                    results[symbol] = {
                        'success': False,
                        'error': 'Could not fetch price'
                    }
                    
            except Stock.DoesNotExist:
                results[symbol] = {
                    'success': False,
                    'error': 'Stock not found in database'
                }
            except Exception as e:
                results[symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def cleanup_old_prices(self, days_to_keep: int = 365) -> int:
        """
        Remove old price records to manage database size.
        
        Args:
            days_to_keep: Number of days of history to keep
        
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)
            deleted_count, _ = StockPrice.objects.filter(date__lt=cutoff_date).delete()
            
            logger.info(f"Cleaned up {deleted_count} old price records older than {cutoff_date}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old prices: {str(e)}")
            return 0
    
    def validate_and_sync(self, symbol: str) -> Dict:
        """
        Validate a stock symbol and sync if valid.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with validation and sync results
        """
        # First validate the symbol
        is_valid = self.provider.validate_symbol(symbol)
        
        if not is_valid:
            return {
                'success': False,
                'symbol': symbol,
                'error': 'Invalid stock symbol'
            }
        
        # If valid, proceed with sync
        return self.sync_stock_data(symbol)


# Singleton instance
data_synchronizer = DataSynchronizer()