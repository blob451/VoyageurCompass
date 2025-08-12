"""
Management command to show evidence with first and last 5 rows for DATA tables.
"""

from django.core.management.base import BaseCommand
from django.apps import apps
from django.db import models as djm


class Command(BaseCommand):
    help = 'Show first 5 and last 5 rows for DATA tables containing strict field names'

    def handle(self, *args, **options):
        """Display evidence of data import."""
        
        strictStocks = {
            'currentPrice', 'previousClose', 'open', 'dayLow', 'dayHigh',
            'regularMarketPrice', 'regularMarketOpen', 'regularMarketDayLow',
            'regularMarketDayHigh', 'regularMarketPreviousClose', 'fiftyTwoWeekLow',
            'fiftyTwoWeekHigh', 'fiftyTwoWeekChange', 'fiftyDayAverage',
            'twoHundredDayAverage', 'beta', 'impliedVolatility', 'volume',
            'regularMarketVolume', 'averageVolume', 'averageVolume10days',
            'averageVolume3months'
        }
        
        strictIndSec = {
            'fiftyTwoWeekChange', 'fiftyDayAverage', 'twoHundredDayAverage',
            'averageVolume', 'averageVolume3months', 'adjusted_close', 'volume'
        }
        
        all_strict_fields = strictStocks | strictIndSec
        
        # Get all Data app models
        data_models = [m for m in apps.get_models() if m.__module__.startswith('Data.')]
        
        self.stdout.write("=== DATA TABLES EVIDENCE ===")
        self.stdout.write("")
        
        for model in data_models:
            # Get field names for this model
            field_names = {f.name for f in model._meta.get_fields() if isinstance(f, djm.Field)}
            
            # Check if this model has any of the strict fields
            if field_names & all_strict_fields:
                try:
                    # Get queryset ordered by id
                    qs = model.objects.all().order_by('id')
                    
                    # Get first 5 rows
                    head = list(qs.values()[:5])
                    
                    # Get last 5 rows  
                    tail = list(qs.values().order_by('-id')[:5])
                    
                    self.stdout.write(f"=== {model.__name__} HEAD (5) ===")
                    for row in head:
                        self.stdout.write(str(row))
                    
                    self.stdout.write(f"=== {model.__name__} TAIL (5) ===")
                    for row in tail:
                        self.stdout.write(str(row))
                    
                    self.stdout.write("")
                    
                except Exception as e:
                    self.stdout.write(f"Error processing {model.__name__}: {str(e)}")
                    self.stdout.write("")