"""
Database managers for mock and production data source filtering.
"""

from django.db import models


# Data source constants matching DataSourceChoices values
DATA_SOURCE_YAHOO = 'yahoo'
DATA_SOURCE_MOCK = 'mock'


class StockQuerySet(models.QuerySet):
    """Enhanced queryset with data source filtering capabilities."""
    
    def real(self):
        """Filter queryset to exclude mock data sources."""
        return self.exclude(data_source=DATA_SOURCE_MOCK)
    
    def mock(self):
        """Filter queryset to include only mock data sources."""
        return self.filter(data_source=DATA_SOURCE_MOCK)
    
    def active(self):
        """Filter queryset to include only active records."""
        return self.filter(is_active=True)


class RealDataManager(models.Manager.from_queryset(StockQuerySet)):
    """Database manager defaulting to production data with full queryset capabilities."""
    
    def get_queryset(self):
        """Generate base queryset excluding mock data sources."""
        return super().get_queryset().real()


class StockManager(models.Manager.from_queryset(StockQuerySet)):
    """Enhanced stock manager with data source filtering methods."""
    
    def real_data(self):
        """Retrieve production data sources only."""
        return self.get_queryset().real()
    
    def mock_data(self):
        """Retrieve mock data sources only."""
        return self.get_queryset().mock()
    
    def active_real_stocks(self):
        """Retrieve active records from production data sources."""
        return self.get_queryset().active().real()
