"""
Core utility functions module providing financial calculations and data processing utilities.
Implements standardised formatting, calculation, and validation functions for financial applications.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def safe_decimal(value: Any) -> Optional[Decimal]:
    """Safely convert value to Decimal with error handling."""
    try:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        logger.warning(f"Could not convert {value} to Decimal")
        return None


def format_currency(amount: Union[float, Decimal], currency: str = "USD") -> str:
    """Format numeric amount as currency string."""
    try:
        if currency == "USD":
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    except (ValueError, TypeError):
        return f"0.00 {currency}"


def format_percentage(value: Union[float, Decimal], decimal_places: int = 2) -> str:
    """Format numeric value as percentage string with specified precision."""
    try:
        return f"{value:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "0.00%"


def calculate_percentage_change(old_value: float, new_value: float) -> Optional[float]:
    """Calculate percentage change between two numeric values."""
    try:
        if old_value == 0:
            return None
        return ((new_value - old_value) / old_value) * 100
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def date_range(start_date: datetime, end_date: datetime, delta: timedelta = timedelta(days=1)) -> List[datetime]:
    """
    Generate a list of dates between start and end date.

    Args:
        start_date: Start date
        end_date: End date
        delta: Time delta between dates

    Returns:
        List of dates
    """
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += delta
    return dates


def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get list of trading days (weekdays) between two dates.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of trading days
    """
    all_days = date_range(start_date, end_date)
    # Filter out weekends (Saturday=5, Sunday=6)
    trading_days = [day for day in all_days if day.weekday() < 5]
    return trading_days


def generate_cache_key(*args) -> str:
    """
    Generate a cache key from arguments using BLAKE2b hashing.

    Args:
        *args: Arguments to include in cache key

    Returns:
        Cache key string
    """
    key_parts = []
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))

    key_string = ":".join(key_parts)
    return hashlib.blake2b(key_string.encode(), digest_size=16).hexdigest()


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = "", separator: str = ".") -> Dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        separator: Separator for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))
    return dict(items)


def sanitize_filename(filename: str) -> str:
    """Sanitise filename by removing invalid filesystem characters for cross-platform compatibility."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename.strip()


def is_market_open() -> bool:
    """Check US stock market operating status."""
    now = datetime.now()

    # Check if weekend
    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    # This is a simplified check - doesn't account for holidays
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def validate_stock_symbol(symbol: str) -> bool:
    """Validate stock ticker symbol format according to market standards."""
    if not symbol:
        return False

    # Standard validation: 1-5 uppercase alphabetic characters
    if not symbol.isupper():
        return False

    if not 1 <= len(symbol) <= 5:
        return False

    if not symbol.isalpha():
        return False

    return True


def round_to_cents(value: Union[float, Decimal]) -> Decimal:
    """Round numeric value to two decimal places (cents precision)."""
    try:
        decimal_value = Decimal(str(value))
        return decimal_value.quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0.00")


# Export commonly used utilities
__all__ = [
    "safe_decimal",
    "format_currency",
    "format_percentage",
    "calculate_percentage_change",
    "date_range",
    "get_trading_days",
    "generate_cache_key",
    "chunk_list",
    "flatten_dict",
    "sanitize_filename",
    "is_market_open",
    "validate_stock_symbol",
    "round_to_cents",
]
