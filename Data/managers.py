"""
Custom managers for Data models to handle mock vs real data separation.
"""

from django.db import models


class RealDataManager(models.Manager):
    """Manager that excludes mock data by default."""
    
    def get_queryset(self):
        return super().get_queryset().exclude(dataSource='mock')


class StockManager(models.Manager):
    """Custom manager for Stock model."""
    
    def real_data(self):
        """Return queryset excluding mock data."""
        return self.exclude(dataSource='mock')
    
    def mock_data(self):
        """Return queryset with only mock data."""
        return self.filter(dataSource='mock')
    
    def active_real_stocks(self):
        """Return active stocks with real data only."""
        return self.filter(is_active=True).exclude(dataSource='mock')