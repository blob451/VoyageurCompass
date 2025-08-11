"""
Custom managers for Data models to handle mock vs real data separation.
"""

from django.db import models


# Data source constants to avoid circular imports
# These match the values in DataSourceChoices
DATA_SOURCE_YAHOO = 'yahoo'
DATA_SOURCE_MOCK = 'mock'


class StockQuerySet(models.QuerySet):
    """Custom QuerySet with real/mock data filtering methods."""
    
    def real(self):
        """Return queryset excluding mock data."""
        return self.exclude(data_source=DATA_SOURCE_MOCK)
    
    def mock(self):
        """Return queryset with only mock data."""
        return self.filter(data_source=DATA_SOURCE_MOCK)
    
    def active(self):
        """Return queryset with only active stocks."""
        return self.filter(is_active=True)


class RealDataManager(models.Manager.from_queryset(StockQuerySet)):
    """
    Manager that excludes mock data by default while providing all QuerySet methods.
    
    This manager always returns real data (non-mock) by default, but still provides
    access to all StockQuerySet methods like .active(), .mock(), etc. for chaining.
    
    Examples:
        # Get active real stocks (default behavior)
        Stock.real_objects.active()
        
        # Override to get mock data if needed
        Stock.real_objects.mock()  # Returns mock data despite manager name
        
        # Chain multiple filters
        Stock.real_objects.active().filter(sector='Technology')
    """
    
    def get_queryset(self):
        """Return queryset with real data only (excludes mock data by default)."""
        return super().get_queryset().real()


class StockManager(models.Manager.from_queryset(StockQuerySet)):
    """Custom manager for Stock model with enhanced filtering capabilities."""
    
    def real_data(self):
        """Return queryset excluding mock data."""
        return self.get_queryset().real()
    
    def mock_data(self):
        """Return queryset with only mock data."""
        return self.get_queryset().mock()
    
    def active_real_stocks(self):
        """Return active stocks with real data only."""
        return self.get_queryset().active().real()