"""
Data Synchronizer Service Module
Manages data synchronization between Yahoo Finance and the database for VoyageurCompass.
Enhanced with Celery task integration and caching support.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple

from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

from Data.models import Stock, StockPrice
from Data.services.provider import data_provider
from Data.services.yahoo_cache import yahoo_cache

logger = logging.getLogger(__name__)


class DataSynchronizer:
    """
    Service class for synchronizing market data with the database.
    Now integrated with Celery for background processing and Redis for caching.
    """

    def __init__(self):
        """Initialize the Data Synchronizer."""
        self.provider = data_provider
        self.cache_timeout = 3600  # 1 hour default cache
        self.batch_size = 100
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

            # Check cache for recent sync
            cache_key = f"voyageur:sync:stock:{symbol.upper()}"
            cached_result = cache.get(cache_key)

            if cached_result and self._is_recently_synced(cached_result):
                logger.info(f"Using cached data for {symbol}")
                return cached_result

            # Fetch data using enhanced caching service (handles partial failures)
            info_data = yahoo_cache.get_stock_info(symbol)
            history_data = yahoo_cache.get_stock_history(symbol, period)
            
            # Handle partial success - continue with whatever data we have
            partial_success = False
            info_source = "api"
            history_source = "api"
            
            if not info_data:
                # Try to get existing stock data
                existing_stock = Stock.objects.filter(symbol=symbol).first()
                if existing_stock:
                    logger.warning(f"Using existing stock info for {symbol} - Yahoo info API failed")
                    info_source = "existing"
                else:
                    logger.warning(f"No info data available for {symbol}, creating minimal stock record with defaults")
                    # Create minimal info data with defaults when API fails but we have history data
                    if history_data and history_data.get('prices'):
                        info_data = {
                            'shortName': symbol,
                            'longName': symbol,
                            'currency': 'USD',
                            'exchange': 'Unknown',
                            'sector': 'Unknown',
                            'industry': 'Unknown',
                            'marketCap': None
                        }
                        info_source = "minimal_defaults"
                        partial_success = True
                        logger.info(f"Created minimal stock info for {symbol} to enable price data loading")
                    else:
                        logger.error(f"No info data or history data available for {symbol}")
                        return {"success": False, "symbol": symbol, "error": "No stock info or price data available"}
            
            if not history_data:
                logger.warning(f"No history data available for {symbol} - will create stock without prices")
                partial_success = True
                history_source = "none"

            # Create or update Stock record (even with partial data)
            if info_data:
                stock, created = self._update_stock_record(symbol, info_data)
            else:
                stock = existing_stock
                created = False

            # Sync historical price data if available
            prices_synced = 0
            if history_data and history_data.get('prices'):
                prices_synced = self._sync_price_history(stock, history_data['prices'])

            # Update last sync timestamp
            stock.last_sync = timezone.now()
            stock.save(update_fields=["last_sync"])

            result = {
                "success": True,
                "symbol": symbol,
                "stock_created": created,
                "prices_synced": prices_synced,
                "total_prices": StockPrice.objects.filter(stock=stock).count(),
                "sync_time": datetime.now().isoformat(),
                "partial_success": partial_success,
                "info_source": info_source,
                "history_source": history_source,
                "warnings": []
            }
            
            if partial_success:
                result["warnings"].append("Partial data sync - some API endpoints failed")

            # Cache the successful result
            cache.set(cache_key, result, timeout=self.cache_timeout)

            logger.info(f"Sync completed for {symbol}: {prices_synced} prices synced")
            return result

        except Exception as e:
            logger.error(f"Error syncing stock data for {symbol}: {str(e)}")
            return {"success": False, "symbol": symbol, "error": str(e)}

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
                "short_name": info.get("shortName", "")[:100],
                "long_name": info.get("longName", "")[:255],
                "currency": info.get("currency", "USD")[:10],
                "exchange": info.get("exchange", "")[:50],
                "sector": info.get("sector", "")[:100],
                "industry": info.get("industry", "")[:100],
                "market_cap": info.get("marketCap", 0) or 0,
            },
        )

        if not created:
            # Update existing stock record
            stock.short_name = info.get("shortName", stock.short_name)[:100]
            stock.long_name = info.get("longName", stock.long_name)[:255]
            stock.currency = info.get("currency", stock.currency)[:10]
            stock.exchange = info.get("exchange", stock.exchange)[:50]
            stock.sector = info.get("sector", stock.sector)[:100]
            stock.industry = info.get("industry", stock.industry)[:100]
            stock.market_cap = info.get("marketCap", stock.market_cap) or 0
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
                # Parse date (handle both "Date" and "date" keys from Yahoo Finance)
                date_str = price_data.get("Date") or price_data.get("date")
                
                # Handle different date formats from Yahoo Finance
                if 'T' in str(date_str):
                    # ISO format with timezone: "2025-08-11T00:00:00-04:00"
                    date_obj = datetime.fromisoformat(str(date_str)).date()
                else:
                    # Simple date format: "2025-08-11"
                    date_obj = datetime.strptime(str(date_str), "%Y-%m-%d").date()

                # Create or update price record (handle both uppercase and lowercase field names)
                price, created = StockPrice.objects.update_or_create(
                    stock=stock,
                    date=date_obj,
                    defaults={
                        "open": Decimal(str(price_data.get("Open") or price_data.get("open"))),
                        "high": Decimal(str(price_data.get("High") or price_data.get("high"))),
                        "low": Decimal(str(price_data.get("Low") or price_data.get("low"))),
                        "close": Decimal(str(price_data.get("Close") or price_data.get("close"))),
                        "volume": price_data.get("Volume") or price_data.get("volume"),
                    },
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

            if result["success"]:
                total_success += 1
            else:
                total_failed += 1

        summary = {
            "total_symbols": len(symbols),
            "successful": total_success,
            "failed": total_failed,
            "results": results,
            "sync_time": datetime.now().isoformat(),
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
                return {"success": True, "message": "No stocks to sync", "stocks_synced": 0}

            # Sync all stocks
            results = self.sync_multiple_stocks(list(stock_symbols))

            return {"success": True, "stocks_synced": len(stock_symbols), "results": results}

        except Exception as e:
            logger.error(f"Error syncing portfolio stocks for user {user_id}: {str(e)}")
            return {"success": False, "error": str(e)}

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
                # Check cache first
                cache_key = f"voyageur:price:realtime:{symbol.upper()}"
                cached_price = cache.get(cache_key)

                if cached_price:
                    results[symbol] = {"success": True, "price": cached_price, "cached": True}
                    continue

                # Get current price from Yahoo Finance
                current_price = self.provider.fetch_realtime_price(symbol)

                if current_price:
                    # Cache the price for 60 seconds
                    cache.set(cache_key, current_price, timeout=60)

                    # Update today's price record
                    stock = Stock.objects.get(symbol=symbol.upper())
                    today = datetime.now().date()

                    StockPrice.objects.update_or_create(
                        stock=stock,
                        date=today,
                        defaults={
                            "close": Decimal(str(current_price)),
                            # For real-time updates, we might not have OHLC data
                            "open": Decimal(str(current_price)),
                            "high": Decimal(str(current_price)),
                            "low": Decimal(str(current_price)),
                        },
                    )

                    results[symbol] = {"success": True, "price": current_price, "cached": False}
                else:
                    results[symbol] = {"success": False, "error": "Could not fetch price"}

            except Stock.DoesNotExist:
                results[symbol] = {"success": False, "error": "Stock not found in database"}
            except Exception as e:
                results[symbol] = {"success": False, "error": str(e)}

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
            return {"success": False, "symbol": symbol, "error": "Invalid stock symbol"}

        # If valid, proceed with sync
        return self.sync_stock_data(symbol)

    # New methods for Celery integration

    def sync_all_data(self) -> Dict[str, any]:
        """
        Main synchronization method called by Celery task.
        Synchronizes all active stocks in the database.

        Returns:
            Dictionary with sync results
        """
        start_time = timezone.now()
        results = {"started_at": start_time.isoformat(), "stocks_synced": [], "total_records": 0, "errors": []}

        try:
            # Get all active stocks that need syncing
            stocks_to_sync = Stock.objects.filter(is_active=True).exclude(
                last_sync__gte=timezone.now() - timedelta(hours=1)
            )

            logger.info(f"Starting sync for {stocks_to_sync.count()} stocks")

            # Sync each stock
            for stock in stocks_to_sync:
                try:
                    result = self.sync_stock_data(stock.symbol)
                    if result["success"]:
                        results["stocks_synced"].append(stock.symbol)
                        results["total_records"] += result.get("prices_synced", 0)
                    else:
                        results["errors"].append(
                            {"symbol": stock.symbol, "error": result.get("error", "Unknown error")}
                        )
                except Exception as e:
                    error_msg = f"Error syncing {stock.symbol}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append({"symbol": stock.symbol, "error": str(e)})

            # Clear market data cache after sync
            cache.delete_pattern("voyageur:market:*")

            results["completed_at"] = timezone.now().isoformat()
            results["duration"] = (timezone.now() - start_time).total_seconds()

            # Cache the results
            cache.set("voyageur:sync:last_result", results, timeout=86400)

            return results

        except Exception as e:
            logger.error(f"Critical error in sync_all_data: {str(e)}")
            results["errors"].append(f"Critical error: {str(e)}")
            results["status"] = "failed"
            return results

    def get_sync_status(self) -> Dict[str, any]:
        """
        Get the current synchronization status.
        Used by API endpoints to check sync progress.

        Returns:
            Dictionary with sync status
        """
        # Check for active sync task
        status = cache.get("market_data_sync_status")

        if not status:
            # Check last result if no current sync
            last_result = cache.get("voyageur:sync:last_result")
            if last_result:
                return {"status": "idle", "last_sync": last_result}
            else:
                return {"status": "never_synced", "message": "No synchronization has been performed yet"}

        return status

    def _is_recently_synced(self, cached_data: Dict) -> bool:
        """
        Check if cached data is still fresh.

        Args:
            cached_data: Cached sync result

        Returns:
            True if data is fresh, False otherwise
        """
        if not cached_data or "sync_time" not in cached_data:
            return False

        try:
            # Use timezone-aware datetime
            sync_time = datetime.fromisoformat(cached_data["sync_time"])
            if timezone.is_naive(sync_time):
                sync_time = timezone.make_aware(sync_time)
            time_since_sync = timezone.now() - sync_time
            return time_since_sync < timedelta(hours=1)
        except Exception:
            return False


# Singleton instance
data_synchronizer = DataSynchronizer()
