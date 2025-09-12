"""
Cache Utilities

Provides utilities for proper cache key handling and sanitization.
Fixes memcached compatibility issues with special characters.
"""

import hashlib
import re
from typing import Any, Dict, Optional, Union


def sanitize_cache_key(key: str, max_length: int = 200) -> str:
    """
    Sanitize cache key to be compatible with memcached.
    
    Args:
        key: Original cache key
        max_length: Maximum key length (memcached limit is 250)
        
    Returns:
        Sanitized cache key safe for memcached
    """
    # Remove or replace problematic characters
    # Memcached doesn't allow: spaces, control chars, and some special chars
    sanitized = re.sub(r'[^\w\-_\.:]', '_', key)
    
    # Replace multiple underscores with single
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # If key is too long, create hash suffix
    if len(sanitized) > max_length:
        # Keep meaningful prefix and add hash suffix
        hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        max_prefix = max_length - len(hash_suffix) - 1
        sanitized = sanitized[:max_prefix] + '_' + hash_suffix
    
    return sanitized


def create_composite_cache_key(
    prefix: str,
    sector_or_industry: str,
    start_date: str,
    end_date: str,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a properly formatted cache key for composite data.
    
    Args:
        prefix: Cache key prefix (e.g., 'sector_composite', 'industry_composite')
        sector_or_industry: Sector or industry name
        start_date: Start date string
        end_date: End date string
        additional_params: Additional parameters to include in key
        
    Returns:
        Sanitized cache key
    """
    # Start with basic key components
    key_parts = [prefix, sector_or_industry, start_date, end_date]
    
    # Add additional parameters if provided
    if additional_params:
        for key, value in sorted(additional_params.items()):
            key_parts.append(f"{key}={value}")
    
    # Join with colons and sanitize
    raw_key = ':'.join(str(part) for part in key_parts)
    return sanitize_cache_key(raw_key)


def create_analysis_cache_key(
    symbol: str,
    analysis_type: str,
    horizon: str,
    data_hash: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a cache key for analysis results.
    
    Args:
        symbol: Stock symbol
        analysis_type: Type of analysis (e.g., 'ta', 'sentiment', 'prediction')
        horizon: Analysis horizon
        data_hash: Hash of underlying data for invalidation
        additional_params: Additional parameters
        
    Returns:
        Sanitized cache key
    """
    key_parts = ['analysis', analysis_type, symbol, horizon]
    
    if data_hash:
        key_parts.append(data_hash[:8])  # Use short hash
        
    if additional_params:
        for key, value in sorted(additional_params.items()):
            key_parts.append(f"{key}={value}")
    
    raw_key = ':'.join(str(part) for part in key_parts)
    return sanitize_cache_key(raw_key)


def create_data_cache_key(
    symbol: str,
    data_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None
) -> str:
    """
    Create a cache key for data retrieval.
    
    Args:
        symbol: Stock symbol
        data_type: Type of data (e.g., 'prices', 'info', 'news')
        start_date: Start date string
        end_date: End date string
        period: Period string (alternative to date range)
        
    Returns:
        Sanitized cache key
    """
    key_parts = ['data', data_type, symbol]
    
    if period:
        key_parts.append(period)
    elif start_date and end_date:
        key_parts.extend([start_date, end_date])
    
    raw_key = ':'.join(str(part) for part in key_parts)
    return sanitize_cache_key(raw_key)


class CacheKeyBuilder:
    """
    Helper class for building consistent cache keys across the application.
    """
    
    def __init__(self, prefix: str = "voyageur"):
        self.prefix = prefix
        
    def build_key(self, *components: Any, **params: Any) -> str:
        """
        Build a cache key from components and parameters.
        
        Args:
            *components: Key components to join
            **params: Additional parameters to include
            
        Returns:
            Sanitized cache key with prefix
        """
        # Start with prefix
        key_parts = [self.prefix]
        
        # Add components
        key_parts.extend(str(comp) for comp in components)
        
        # Add parameters in sorted order for consistency
        for key, value in sorted(params.items()):
            key_parts.append(f"{key}={value}")
        
        raw_key = ':'.join(key_parts)
        return sanitize_cache_key(raw_key)
        
    def stock_analysis_key(
        self,
        symbol: str,
        analysis_type: str,
        horizon: str = "blend",
        **kwargs: Any
    ) -> str:
        """Build cache key for stock analysis."""
        return self.build_key(
            'analysis', analysis_type, symbol, horizon, **kwargs
        )
        
    def composite_data_key(
        self,
        composite_type: str,
        name: str,
        start_date: str,
        end_date: str,
        **kwargs: Any
    ) -> str:
        """Build cache key for composite data."""
        return self.build_key(
            'composite', composite_type, name, start_date, end_date, **kwargs
        )
        
    def stock_data_key(
        self,
        symbol: str,
        data_type: str,
        period: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Build cache key for stock data."""
        components = ['data', data_type, symbol]
        if period:
            components.append(period)
        return self.build_key(*components, **kwargs)


# Global cache key builder instance
cache_key_builder = CacheKeyBuilder()


def get_memcached_safe_key(original_key: str) -> str:
    """
    Simple wrapper to get memcached-safe key.
    
    Args:
        original_key: Original cache key
        
    Returns:
        Memcached-safe cache key
    """
    return sanitize_cache_key(original_key)


def validate_cache_key(key: str) -> Dict[str, Any]:
    """
    Validate a cache key for memcached compatibility.
    
    Args:
        key: Cache key to validate
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check length
    if len(key) > 250:
        issues.append(f"Key too long: {len(key)} > 250 characters")
    
    # Check for problematic characters
    problematic_chars = re.findall(r'[^\w\-_\.:]', key)
    if problematic_chars:
        issues.append(f"Problematic characters found: {set(problematic_chars)}")
    
    # Check for spaces
    if ' ' in key:
        issues.append("Spaces not allowed in memcached keys")
    
    # Check for control characters
    if any(ord(char) < 32 for char in key):
        issues.append("Control characters found")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'length': len(key),
        'sanitized_key': sanitize_cache_key(key) if issues else key
    }


# Example usage and testing functions
def test_cache_key_sanitization():
    """Test cache key sanitization functionality."""
    test_keys = [
        "sector_composite:Consumer Cyclical:2025-08-12:2025-09-11",
        "industry_composite:Software - Application:2023-01-01:2024-01-01",
        "analysis:ta:AAPL:blend:include_sentiment=true",
        "data:prices:MSFT:2y:force_refresh=false"
    ]
    
    print("Cache Key Sanitization Tests:")
    print("=" * 50)
    
    for original_key in test_keys:
        sanitized = sanitize_cache_key(original_key)
        validation = validate_cache_key(original_key)
        
        print(f"Original:  {original_key}")
        print(f"Sanitized: {sanitized}")
        print(f"Valid:     {validation['valid']}")
        if validation['issues']:
            print(f"Issues:    {', '.join(validation['issues'])}")
        print("-" * 50)


if __name__ == "__main__":
    test_cache_key_sanitization()