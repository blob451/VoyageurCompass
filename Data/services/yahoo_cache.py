"""
Yahoo Finance caching service implementing multi-level cache strategy.
Provides graduated fallback through Redis, database, and direct API access patterns.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import yfinance as yf
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone

from Data.models import YahooFinanceCache

logger = logging.getLogger(__name__)


class YahooFinanceCacheService:
    """
    Multi-level caching service implementing graduated Yahoo Finance data retrieval.
    
    Architecture:
    1. Redis cache - Memory-resident storage with optimised TTL configuration
    2. Database persistence - Long-term storage with expiration management
    3. Yahoo Finance API - Primary data source with exponential backoff strategy
    """

    # Cache TTL settings (in seconds)
    REDIS_TTL_REALTIME = 3600  # 1 hour for real-time data
    REDIS_TTL_HISTORICAL = 86400  # 24 hours for historical data
    DB_TTL_REALTIME = 1  # 1 hour for real-time data
    DB_TTL_HISTORICAL = 24  # 24 hours for historical data

    def __init__(self):
        """Initialise caching service with retry configuration parameters."""        
        self.max_retries = 3
        self.base_delay = 2.0
        self.max_backoff = 60.0
        logger.info("Yahoo Finance Cache Service initialised")

    def get_cache_key(self, symbol: str, data_type: str, period: str = "1y") -> str:
        """Generate standardised Redis cache key for data identification."""        
        return f"yahoo_finance:{symbol.upper()}:{data_type}:{period}"

    def get_stock_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve stock information through graduated caching mechanism.
        
        Args:
            symbol: Stock symbol identifier
            use_cache: Cache utilisation preference
        
        Returns:
            Stock information dictionary or None upon failure
        """
        symbol = symbol.upper()
        cache_key = self.get_cache_key(symbol, "info")
        
        if not use_cache:
            return self._fetch_from_yahoo_with_retry(symbol, "info")

        # Level 1: Redis cache
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.debug(f"Redis cache hit for {symbol} info")
            return cached_data

        # Level 2: Database cache
        db_cached = self._get_from_db_cache(symbol, "info")
        if db_cached:
            logger.debug(f"Database cache hit for {symbol} info")
            # Refresh Redis cache
            cache.set(cache_key, db_cached, self.REDIS_TTL_REALTIME)
            return db_cached

        # Level 3: Yahoo Finance API
        yahoo_data = self._fetch_from_yahoo_with_retry(symbol, "info")
        if yahoo_data:
            # Cache in both Redis and Database
            cache.set(cache_key, yahoo_data, self.REDIS_TTL_REALTIME)
            self._save_to_db_cache(symbol, "info", "", yahoo_data, self.DB_TTL_REALTIME)
            return yahoo_data

        # Return stale database cache if available
        stale_data = self._get_from_db_cache(symbol, "info", allow_expired=True)
        if stale_data:
            logger.warning(f"Using stale cache for {symbol} info - Yahoo API unavailable")
            return stale_data

        logger.error(f"All cache levels failed for {symbol} info")
        return None

    def get_stock_history(self, symbol: str, period: str = "1y", use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve historical price data through multi-level caching architecture.
        
        Args:
            symbol: Stock symbol identifier
            period: Historical data timeframe specification
            use_cache: Cache utilisation preference
        
        Returns:
            Historical price data structure or None upon failure
        """
        symbol = symbol.upper()
        cache_key = self.get_cache_key(symbol, "history", period)
        
        if not use_cache:
            return self._fetch_from_yahoo_with_retry(symbol, "history", period)

        # Level 1: Redis cache
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.debug(f"Redis cache hit for {symbol} history ({period})")
            return cached_data

        # Level 2: Database cache
        db_cached = self._get_from_db_cache(symbol, "history", period)
        if db_cached:
            logger.debug(f"Database cache hit for {symbol} history ({period})")
            # Choose appropriate TTL based on period
            redis_ttl = self.REDIS_TTL_HISTORICAL if period in ['1y', '2y', '5y', '10y', 'max'] else self.REDIS_TTL_REALTIME
            cache.set(cache_key, db_cached, redis_ttl)
            return db_cached

        # Level 3: Yahoo Finance API
        yahoo_data = self._fetch_from_yahoo_with_retry(symbol, "history", period)
        if yahoo_data:
            # Cache in both Redis and Database
            is_historical = period in ['1y', '2y', '5y', '10y', 'max']
            redis_ttl = self.REDIS_TTL_HISTORICAL if is_historical else self.REDIS_TTL_REALTIME
            db_ttl = self.DB_TTL_HISTORICAL if is_historical else self.DB_TTL_REALTIME
            
            cache.set(cache_key, yahoo_data, redis_ttl)
            self._save_to_db_cache(symbol, "history", period, yahoo_data, db_ttl)
            return yahoo_data

        # Return stale database cache if available
        stale_data = self._get_from_db_cache(symbol, "history", period, allow_expired=True)
        if stale_data:
            logger.warning(f"Using stale cache for {symbol} history ({period}) - Yahoo API unavailable")
            return stale_data

        logger.error(f"All cache levels failed for {symbol} history ({period})")
        return None

    def get_stock_financials(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve financial statements through graduated caching system.
        
        Args:
            symbol: Stock symbol identifier
            use_cache: Cache utilisation preference
        
        Returns:
            Comprehensive financial data structure or None upon failure
        """
        symbol = symbol.upper()
        cache_key = self.get_cache_key(symbol, "financials")
        
        if not use_cache:
            return self._fetch_from_yahoo_with_retry(symbol, "financials")

        # Level 1: Redis cache
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.debug(f"Redis cache hit for {symbol} financials")
            return cached_data

        # Level 2: Database cache
        db_cached = self._get_from_db_cache(symbol, "financials")
        if db_cached:
            logger.debug(f"Database cache hit for {symbol} financials")
            cache.set(cache_key, db_cached, self.REDIS_TTL_HISTORICAL)
            return db_cached

        # Level 3: Yahoo Finance API
        yahoo_data = self._fetch_from_yahoo_with_retry(symbol, "financials")
        if yahoo_data:
            # Cache in both Redis and Database (financials are historical, use longer TTL)
            cache.set(cache_key, yahoo_data, self.REDIS_TTL_HISTORICAL)
            self._save_to_db_cache(symbol, "financials", "", yahoo_data, self.DB_TTL_HISTORICAL)
            return yahoo_data

        # Return stale database cache if available
        stale_data = self._get_from_db_cache(symbol, "financials", allow_expired=True)
        if stale_data:
            logger.warning(f"Using stale cache for {symbol} financials - Yahoo API unavailable")
            return stale_data

        logger.error(f"All cache levels failed for {symbol} financials")
        return None

    def _get_from_db_cache(self, symbol: str, data_type: str, period: str = "", allow_expired: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieve data from persistent database cache with expiration validation."""        
        try:
            cache_entry = YahooFinanceCache.objects.filter(
                symbol=symbol,
                data_type=data_type,
                period=period
            ).first()

            if not cache_entry:
                return None

            if not cache_entry.fetch_success:
                logger.debug(f"Database cache entry for {symbol} {data_type} marked as failed")
                return None

            if not allow_expired and cache_entry.is_expired():
                logger.debug(f"Database cache entry for {symbol} {data_type} has expired")
                return None

            return cache_entry.data

        except Exception as e:
            logger.error(f"Error reading from database cache: {e}")
            return None

    def _save_to_db_cache(self, symbol: str, data_type: str, period: str, data: Dict[str, Any], ttl_hours: int) -> None:
        """Persist data to database cache with configured expiration parameters."""        
        try:
            expires_at = timezone.now() + timedelta(hours=ttl_hours)
            
            with transaction.atomic():
                cache_entry, created = YahooFinanceCache.objects.update_or_create(
                    symbol=symbol,
                    data_type=data_type,
                    period=period,
                    defaults={
                        'data': data,
                        'expires_at': expires_at,
                        'fetch_success': True,
                        'error_message': None,
                    }
                )
                
                action = "Created" if created else "Updated"
                logger.debug(f"{action} database cache for {symbol} {data_type} (expires: {expires_at})")

        except Exception as e:
            logger.error(f"Error saving to database cache: {e}")

    def _fetch_from_yahoo_with_retry(self, symbol: str, data_type: str, period: str = "") -> Optional[Dict[str, Any]]:
        """Execute Yahoo Finance API request with exponential backoff retry mechanism."""        
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = None

                if data_type == "info":
                    data = ticker.info
                elif data_type == "history":
                    history_df = ticker.history(period=period)
                    if not history_df.empty:
                        # Convert DataFrame to dict format with proper JSON serialization
                        df_records = history_df.reset_index().to_dict('records')
                        # Convert Timestamp objects to strings
                        for record in df_records:
                            for key, value in record.items():
                                if hasattr(value, 'isoformat'):  # Timestamp or datetime
                                    record[key] = value.isoformat()
                                elif hasattr(value, 'item'):  # numpy types
                                    record[key] = value.item()
                        
                        data = {
                            'prices': df_records,
                            'period': period,
                            'symbol': symbol
                        }
                elif data_type == "financials":
                    # Get multiple financial statements
                    data = {
                        'quarterly_financials': ticker.quarterly_financials.to_dict() if not ticker.quarterly_financials.empty else {},
                        'financials': ticker.financials.to_dict() if not ticker.financials.empty else {},
                        'quarterly_balance_sheet': ticker.quarterly_balance_sheet.to_dict() if not ticker.quarterly_balance_sheet.empty else {},
                        'balance_sheet': ticker.balance_sheet.to_dict() if not ticker.balance_sheet.empty else {},
                        'quarterly_cashflow': ticker.quarterly_cashflow.to_dict() if not ticker.quarterly_cashflow.empty else {},
                        'cashflow': ticker.cashflow.to_dict() if not ticker.cashflow.empty else {},
                    }

                if data:
                    logger.info(f"Successfully fetched {symbol} {data_type} from Yahoo Finance")
                    return data
                else:
                    logger.warning(f"Yahoo Finance returned empty data for {symbol} {data_type}")
                    return None

            except Exception as e:
                delay = min(self.base_delay * (2 ** attempt), self.max_backoff)
                logger.warning(f"Yahoo Finance API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying after {delay} seconds...")
                    time.sleep(delay)
                else:
                    # Save failed attempt to database cache to avoid repeated API calls
                    self._save_failed_attempt(symbol, data_type, period, str(e))

        logger.error(f"All {self.max_retries} Yahoo Finance API attempts failed for {symbol} {data_type}")
        return None

    def _save_failed_attempt(self, symbol: str, data_type: str, period: str, error_message: str) -> None:
        """Persist failed API attempt data to optimise subsequent request handling."""        
        try:
            expires_at = timezone.now() + timedelta(minutes=30)  # Short cache for failed attempts
            
            with transaction.atomic():
                YahooFinanceCache.objects.update_or_create(
                    symbol=symbol,
                    data_type=data_type,
                    period=period,
                    defaults={
                        'data': {},
                        'expires_at': expires_at,
                        'fetch_success': False,
                        'error_message': error_message,
                    }
                )
                
            logger.debug(f"Saved failed attempt for {symbol} {data_type}")

        except Exception as e:
            logger.error(f"Error saving failed attempt: {e}")

    def clear_cache(self, symbol: str = None, data_type: str = None) -> Dict[str, int]:
        """
        Execute cache clearing operations across storage layers.
        
        Args:
            symbol: Target symbol for selective clearing
            data_type: Target data type for selective clearing
        
        Returns:
            Clearing operation statistics dictionary
        """
        redis_cleared = 0
        db_cleared = 0

        try:
            # Clear Redis cache
            if symbol and data_type:
                for period in ["", "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]:
                    cache_key = self.get_cache_key(symbol, data_type, period)
                    if cache.delete(cache_key):
                        redis_cleared += 1
            else:
                # Clear all Yahoo Finance Redis cache keys
                pattern = "yahoo_finance:*"
                cache.delete_pattern(pattern)
                redis_cleared = 1  # Pattern deletion count not available

            # Clear database cache
            queryset = YahooFinanceCache.objects.all()
            if symbol:
                queryset = queryset.filter(symbol=symbol.upper())
            if data_type:
                queryset = queryset.filter(data_type=data_type)
            
            db_cleared = queryset.delete()[0]

            logger.info(f"Cache cleared - Redis: {redis_cleared}, Database: {db_cleared}")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

        return {"redis_cleared": redis_cleared, "db_cleared": db_cleared}

    def get_comprehensive_stock_data(self, symbol: str, period: str = "1y", use_cache: bool = True) -> Dict[str, Any]:
        """
        Retrieve comprehensive stock data with mixed result handling.
        
        Args:
            symbol: Stock symbol identifier
            period: Historical data timeframe
            use_cache: Cache utilisation preference
        
        Returns:
            Comprehensive data structure with success indicators for each component
        """
        symbol = symbol.upper()
        
        result = {
            "symbol": symbol,
            "info": None,
            "history": None,
            "financials": None,
            "info_success": False,
            "history_success": False,
            "financials_success": False,
            "partial_success": False,
            "errors": []
        }
        
        # Get stock info
        try:
            info_data = self.get_stock_info(symbol, use_cache)
            if info_data:
                result["info"] = info_data
                result["info_success"] = True
        except Exception as e:
            result["errors"].append(f"Info fetch failed: {str(e)}")
            
        # Get history data  
        try:
            history_data = self.get_stock_history(symbol, period, use_cache)
            if history_data:
                result["history"] = history_data
                result["history_success"] = True
        except Exception as e:
            result["errors"].append(f"History fetch failed: {str(e)}")
            
        # Get financials (optional)
        try:
            financials_data = self.get_stock_financials(symbol, use_cache)
            if financials_data:
                result["financials"] = financials_data
                result["financials_success"] = True
        except Exception as e:
            result["errors"].append(f"Financials fetch failed: {str(e)}")
            
        # Determine success status
        success_count = sum([result["info_success"], result["history_success"], result["financials_success"]])
        
        if success_count == 0:
            result["success"] = False
        elif success_count < 3:
            result["success"] = True
            result["partial_success"] = True
        else:
            result["success"] = True
            
        logger.info(f"Comprehensive data fetch for {symbol}: {success_count}/3 components successful")
        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Generate comprehensive cache performance statistics."""        
        try:
            total_entries = YahooFinanceCache.objects.count()
            expired_entries = YahooFinanceCache.objects.filter(expires_at__lt=timezone.now()).count()
            failed_entries = YahooFinanceCache.objects.filter(fetch_success=False).count()
            successful_entries = total_entries - failed_entries

            stats = {
                "total_entries": total_entries,
                "successful_entries": successful_entries,
                "failed_entries": failed_entries,
                "expired_entries": expired_entries,
                "cache_hit_rate": (successful_entries / total_entries * 100) if total_entries > 0 else 0
            }

            # Get breakdown by data type
            for data_type in ['info', 'history', 'financials']:
                count = YahooFinanceCache.objects.filter(data_type=data_type, fetch_success=True).count()
                stats[f"{data_type}_entries"] = count

            return stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}


# Global instance for easy import
yahoo_cache = YahooFinanceCacheService()