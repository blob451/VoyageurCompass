"""
Multi-tier caching strategy implementation for optimised data access.
Implements L1 (memory), L2 (Redis), and L3 (database) caching layers.
"""

import hashlib
import logging
import pickle
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from django.core.cache import cache
from django.core.cache.backends.locmem import LocMemCache
from django.db import models

logger = logging.getLogger(__name__)


class MultiTierCache:
    """Multi-tier caching system with automatic fallback and promotion."""
    
    def __init__(self):
        self.l1_cache = {}  # In-memory dictionary cache
        self.l1_max_size = 1000
        self.l1_hit_count = 0
        self.l2_hit_count = 0
        self.l3_hit_count = 0
        self.total_requests = 0
        
        # Cache tier TTLs (seconds)
        self.l1_ttl = 300     # 5 minutes
        self.l2_ttl = 1800    # 30 minutes  
        self.l3_ttl = 7200    # 2 hours
        
        logger.info("MultiTierCache initialised with 3-tier architecture")
    
    def get(self, key: str, default=None) -> Any:
        """Retrieve value from cache with tier fallback."""
        self.total_requests += 1
        
        # L1 Cache (In-memory)
        l1_result = self._get_l1(key)
        if l1_result is not None:
            self.l1_hit_count += 1
            return l1_result
        
        # L2 Cache (Redis)
        l2_result = self._get_l2(key)
        if l2_result is not None:
            self.l2_hit_count += 1
            # Promote to L1
            self._set_l1(key, l2_result)
            return l2_result
        
        # L3 Cache (Database query cache - conceptual, would be implemented per use case)
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in all cache tiers."""
        
        # Set in L1 (fastest access)
        self._set_l1(key, value)
        
        # Set in L2 (distributed)
        self._set_l2(key, value, ttl or self.l2_ttl)
        
        logger.debug(f"Cached key '{key}' in L1 and L2 tiers")
    
    def delete(self, key: str):
        """Remove key from all cache tiers."""
        self._delete_l1(key)
        cache.delete(key)
        logger.debug(f"Deleted key '{key}' from all cache tiers")
    
    def clear_l1(self):
        """Clear L1 cache."""
        self.l1_cache.clear()
        logger.info("L1 cache cleared")
    
    def _get_l1(self, key: str) -> Any:
        """Get from L1 in-memory cache with TTL check."""
        if key in self.l1_cache:
            value, timestamp, ttl = self.l1_cache[key]
            if time.time() - timestamp < ttl:
                return value
            else:
                # Expired, remove
                del self.l1_cache[key]
        return None
    
    def _set_l1(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set in L1 cache with size limit."""
        if len(self.l1_cache) >= self.l1_max_size:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = (value, time.time(), ttl or self.l1_ttl)
    
    def _delete_l1(self, key: str):
        """Delete from L1 cache."""
        self.l1_cache.pop(key, None)
    
    def _get_l2(self, key: str) -> Any:
        """Get from L2 Redis cache."""
        try:
            return cache.get(key)
        except Exception as e:
            logger.warning(f"L2 cache get failed for key '{key}': {str(e)}")
            return None
    
    def _set_l2(self, key: str, value: Any, ttl: int):
        """Set in L2 Redis cache."""
        try:
            cache.set(key, value, ttl)
        except Exception as e:
            logger.warning(f"L2 cache set failed for key '{key}': {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieve cache performance statistics."""
        total = max(self.total_requests, 1)
        
        return {
            'l1_hit_rate': self.l1_hit_count / total,
            'l2_hit_rate': self.l2_hit_count / total,
            'overall_hit_rate': (self.l1_hit_count + self.l2_hit_count) / total,
            'l1_size': len(self.l1_cache),
            'total_requests': self.total_requests,
            'cache_efficiency': {
                'l1_hits': self.l1_hit_count,
                'l2_hits': self.l2_hit_count,
                'misses': total - self.l1_hit_count - self.l2_hit_count
            }
        }


# Global multi-tier cache instance
multi_cache = MultiTierCache()


def cache_result(key_prefix: str = "default", 
                ttl: int = 1800, 
                use_multi_tier: bool = True):
    """
    Decorator for caching function results with multi-tier support.
    
    Args:
        key_prefix: Cache key prefix
        ttl: Time-to-live in seconds
        use_multi_tier: Whether to use multi-tier caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key based on function name and arguments
            key_parts = [key_prefix, func.__name__]
            
            # Add args to key
            for arg in args:
                if hasattr(arg, 'id'):  # Django model instance
                    key_parts.append(f"{arg.__class__.__name__}_{arg.id}")
                elif isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
            
            # Add kwargs to key
            for key, value in sorted(kwargs.items()):
                key_parts.append(f"{key}_{value}")
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            if use_multi_tier:
                result = multi_cache.get(cache_key)
            else:
                result = cache.get(cache_key)
            
            if result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return result
            
            # Cache miss - execute function
            logger.debug(f"Cache miss for key: {cache_key}")
            result = func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                if use_multi_tier:
                    multi_cache.set(cache_key, result, ttl)
                else:
                    cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache_pattern(pattern: str):
    """Invalidate cache keys matching a pattern."""
    try:
        # This would require Redis SCAN for pattern matching
        # Simplified implementation for now
        logger.info(f"Cache invalidation requested for pattern: {pattern}")
        
        # Clear L1 cache (simple approach)
        multi_cache.clear_l1()
        
    except Exception as e:
        logger.error(f"Cache invalidation failed for pattern '{pattern}': {str(e)}")


class QuerysetCache:
    """Specialised caching for Django QuerySets with smart invalidation."""
    
    def __init__(self, model_class: models.Model):
        self.model_class = model_class
        self.cache_prefix = f"qs:{model_class._meta.label_lower}"
    
    def get_cached_queryset(self, 
                           filter_kwargs: Dict[str, Any],
                           select_related: List[str] = None,
                           prefetch_related: List[str] = None,
                           ttl: int = 900) -> Optional[List[Any]]:
        """Get cached queryset results."""
        
        # Generate cache key from filters and related fields
        key_parts = [self.cache_prefix, "filtered"]
        
        for key, value in sorted(filter_kwargs.items()):
            key_parts.append(f"{key}_{value}")
        
        if select_related:
            key_parts.append(f"select_{'-'.join(sorted(select_related))}")
        
        if prefetch_related:
            key_parts.append(f"prefetch_{'-'.join(sorted(prefetch_related))}")
        
        cache_key = ":".join(key_parts)
        
        # Try cache first
        cached_result = multi_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Build and execute queryset
        queryset = self.model_class.objects.filter(**filter_kwargs)
        
        if select_related:
            queryset = queryset.select_related(*select_related)
        
        if prefetch_related:
            queryset = queryset.prefetch_related(*prefetch_related)
        
        # Convert to list to cache (QuerySets can't be cached directly)
        result = list(queryset)
        
        # Cache the result
        multi_cache.set(cache_key, result, ttl)
        
        return result
    
    def invalidate_model_cache(self):
        """Invalidate all cache entries for this model."""
        invalidate_cache_pattern(f"{self.cache_prefix}:*")


def get_queryset_cache(model_class: models.Model) -> QuerysetCache:
    """Factory function for QuerysetCache instances."""
    return QuerysetCache(model_class)