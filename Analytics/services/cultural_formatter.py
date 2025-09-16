"""
Cultural formatting service for financial data display across different locales.
Handles number, currency, date, and percentage formatting according to regional preferences.
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional, Union
from decimal import Decimal

from django.conf import settings


class CulturalFormatter:
    """Service for applying cultural formatting to financial data."""

    def __init__(self):
        """Initialize cultural formatter with settings-based configuration."""
        self.enabled = getattr(settings, "CULTURAL_FORMATTING_ENABLED", True)
        self.formatting_configs = getattr(settings, "FINANCIAL_FORMATTING", {})
        self.default_language = getattr(settings, "DEFAULT_USER_LANGUAGE", "en")

    def format_currency(
        self,
        amount: Union[int, float, Decimal, str],
        currency_code: str = "USD",
        language: str = "en",
    ) -> str:
        """
        Format currency amount according to locale preferences.

        Args:
            amount: The currency amount to format
            currency_code: ISO currency code (USD, EUR, etc.)
            language: Target language/locale code

        Returns:
            Formatted currency string
        """
        if not self.enabled or language not in self.formatting_configs:
            return self._format_currency_fallback(amount, currency_code)

        try:
            # Convert to float for processing
            if isinstance(amount, str):
                amount = float(amount.replace(',', '').replace(' ', ''))
            amount = float(amount)

            config = self.formatting_configs[language]

            # Format the number part
            formatted_number = self._format_number_part(amount, config)

            # Get currency symbol
            currency_symbol = self._get_currency_symbol(currency_code, config)

            # Apply currency position
            if config.get("currency_position") == "before":
                return f"{currency_symbol}{formatted_number}"
            else:
                return f"{formatted_number} {currency_symbol}"

        except (ValueError, TypeError) as e:
            return self._format_currency_fallback(amount, currency_code)

    def format_number(
        self,
        number: Union[int, float, Decimal, str],
        language: str = "en",
        decimal_places: int = 2,
    ) -> str:
        """
        Format numbers according to locale preferences.

        Args:
            number: Number to format
            language: Target language/locale code
            decimal_places: Number of decimal places to show

        Returns:
            Formatted number string
        """
        if not self.enabled or language not in self.formatting_configs:
            return self._format_number_fallback(number, decimal_places)

        try:
            # Convert to float for processing
            if isinstance(number, str):
                number = float(number.replace(',', '').replace(' ', ''))
            number = float(number)

            config = self.formatting_configs[language]
            return self._format_number_part(number, config, decimal_places)

        except (ValueError, TypeError):
            return self._format_number_fallback(number, decimal_places)

    def format_percentage(
        self,
        percentage: Union[int, float, Decimal, str],
        language: str = "en",
        decimal_places: int = 2,
    ) -> str:
        """
        Format percentage according to locale preferences.

        Args:
            percentage: Percentage value (e.g., 0.15 for 15%)
            language: Target language/locale code
            decimal_places: Number of decimal places to show

        Returns:
            Formatted percentage string
        """
        try:
            # Convert to float and multiply by 100 for percentage display
            if isinstance(percentage, str):
                percentage = float(percentage.replace(',', '').replace(' ', '').replace('%', ''))
            percentage = float(percentage) * 100

            formatted_number = self.format_number(percentage, language, decimal_places)
            return f"{formatted_number}%"

        except (ValueError, TypeError):
            return f"{percentage}%"

    def format_date(
        self,
        date: Union[datetime, str],
        language: str = "en",
        format_type: str = "short",
    ) -> str:
        """
        Format dates according to locale preferences.

        Args:
            date: Date to format (datetime object or ISO string)
            language: Target language/locale code
            format_type: 'short', 'medium', 'long'

        Returns:
            Formatted date string
        """
        if not self.enabled or language not in self.formatting_configs:
            return self._format_date_fallback(date, format_type)

        try:
            # Convert string to datetime if needed
            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))

            config = self.formatting_configs[language]
            date_format = config.get("date_format", "M/d/Y")

            # Convert Django-style format to Python strftime format
            python_format = self._convert_date_format(date_format, format_type)

            return date.strftime(python_format)

        except (ValueError, TypeError):
            return self._format_date_fallback(date, format_type)

    def format_financial_text(
        self,
        text: str,
        language: str = "en",
        preserve_original: bool = False,
    ) -> str:
        """
        Apply cultural formatting to financial text content.

        Args:
            text: Text containing financial data
            language: Target language/locale code
            preserve_original: Whether to preserve original formatting

        Returns:
            Text with culturally appropriate formatting applied
        """
        if not self.enabled or language not in self.formatting_configs:
            return text

        try:
            formatted_text = text

            # Apply currency formatting in text
            formatted_text = self._format_currencies_in_text(formatted_text, language)

            # Apply number formatting in text
            formatted_text = self._format_numbers_in_text(formatted_text, language)

            # Apply percentage formatting in text
            formatted_text = self._format_percentages_in_text(formatted_text, language)

            return formatted_text

        except Exception:
            return text

    def _format_number_part(
        self,
        number: float,
        config: Dict[str, Any],
        decimal_places: int = 2,
    ) -> str:
        """Format the numeric part according to locale configuration."""
        # Round to specified decimal places
        number = round(number, decimal_places)

        # Split into integer and decimal parts
        integer_part = int(abs(number))
        decimal_part = abs(number) - integer_part

        # Format integer part with thousands separator
        thousands_sep = config.get("thousands_separator", ",")
        formatted_integer = f"{integer_part:,}".replace(",", thousands_sep)

        # Handle negative numbers
        if number < 0:
            formatted_integer = f"-{formatted_integer}"

        # Add decimal part if needed
        if decimal_places > 0 and decimal_part > 0:
            decimal_sep = config.get("decimal_separator", ".")
            decimal_str = f"{decimal_part:.{decimal_places}f}"[2:]  # Remove "0."
            return f"{formatted_integer}{decimal_sep}{decimal_str}"

        return formatted_integer

    def _get_currency_symbol(self, currency_code: str, config: Dict[str, Any]) -> str:
        """Get appropriate currency symbol for the locale."""
        # Currency mapping
        currency_symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CAD": "C$",
            "AUD": "A$",
        }

        # Use config-specific symbol if available
        default_symbol = config.get("currency_symbol", currency_symbols.get(currency_code, currency_code))

        return default_symbol

    def _format_currencies_in_text(self, text: str, language: str) -> str:
        """Apply currency formatting within text content."""
        config = self.formatting_configs[language]

        # Pattern to match currency amounts like $1,234.56
        pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'

        def replace_currency(match):
            amount_str = match.group(1)
            try:
                amount = float(amount_str.replace(',', ''))
                return self.format_currency(amount, "USD", language)
            except ValueError:
                return match.group(0)

        return re.sub(pattern, replace_currency, text)

    def _format_numbers_in_text(self, text: str, language: str) -> str:
        """Apply number formatting within text content."""
        # Pattern to match standalone numbers with commas/decimals
        pattern = r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b'

        def replace_number(match):
            number_str = match.group(1)
            # Skip if it looks like it's already part of a currency or percentage
            if '$' in match.string[max(0, match.start()-1):match.end()+1] or \
               '%' in match.string[max(0, match.start()-1):match.end()+1]:
                return match.group(0)

            try:
                number = float(number_str.replace(',', ''))
                # Only format if it's a significant number (> 999)
                if abs(number) > 999:
                    return self.format_number(number, language)
                return match.group(0)
            except ValueError:
                return match.group(0)

        return re.sub(pattern, replace_number, text)

    def _format_percentages_in_text(self, text: str, language: str) -> str:
        """Apply percentage formatting within text content."""
        # Pattern to match percentages like 15.5%
        pattern = r'(\d+(?:\.\d+)?)%'

        def replace_percentage(match):
            percent_str = match.group(1)
            try:
                percent = float(percent_str) / 100  # Convert to decimal
                return self.format_percentage(percent, language)
            except ValueError:
                return match.group(0)

        return re.sub(pattern, replace_percentage, text)

    def _convert_date_format(self, django_format: str, format_type: str) -> str:
        """Convert Django date format to Python strftime format."""
        format_mapping = {
            "M": "%m",
            "d": "%d",
            "Y": "%Y",
            "y": "%y",
            "j": "%d",
            "n": "%m",
        }

        python_format = django_format
        for django_code, python_code in format_mapping.items():
            python_format = python_format.replace(django_code, python_code)

        return python_format

    # Fallback methods for when cultural formatting is disabled or unavailable

    def _format_currency_fallback(self, amount: Union[int, float, str], currency_code: str) -> str:
        """Fallback currency formatting (US style)."""
        try:
            if isinstance(amount, str):
                amount = float(amount.replace(',', '').replace(' ', ''))
            return f"${amount:,.2f}"
        except (ValueError, TypeError):
            return str(amount)

    def _format_number_fallback(self, number: Union[int, float, str], decimal_places: int) -> str:
        """Fallback number formatting (US style)."""
        try:
            if isinstance(number, str):
                number = float(number.replace(',', '').replace(' ', ''))
            if decimal_places == 0:
                return f"{number:,.0f}"
            return f"{number:,.{decimal_places}f}"
        except (ValueError, TypeError):
            return str(number)

    def _format_date_fallback(self, date: Union[datetime, str], format_type: str) -> str:
        """Fallback date formatting (US style)."""
        try:
            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))

            if format_type == "short":
                return date.strftime("%m/%d/%Y")
            elif format_type == "medium":
                return date.strftime("%b %d, %Y")
            else:  # long
                return date.strftime("%B %d, %Y")
        except (ValueError, TypeError):
            return str(date)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages for cultural formatting."""
        return list(self.formatting_configs.keys())

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported for cultural formatting."""
        return language in self.formatting_configs


# Singleton instance
_cultural_formatter = None


def get_cultural_formatter() -> CulturalFormatter:
    """Get singleton instance of CulturalFormatter."""
    global _cultural_formatter
    if _cultural_formatter is None:
        _cultural_formatter = CulturalFormatter()
    return _cultural_formatter