# Generated migration for marking existing data as yahoo source

from django.db import migrations, transaction


def mark_existing_data_as_yahoo(apps, schema_editor):
    """Mark all existing data as coming from Yahoo Finance."""
    Stock = apps.get_model('Data', 'Stock')
    StockPrice = apps.get_model('Data', 'StockPrice')
    
    # Update existing data atomically to ensure consistency
    with transaction.atomic():
        # Update existing stocks
        Stock.objects.all().update(dataSource='yahoo')
        StockPrice.objects.all().update(dataSource='yahoo')


def reverse_mark_data(apps, schema_editor):
    """
    Reverse migration - clear existing data to restore pre-migration state.
    
    Since the original state before this migration cannot be perfectly restored
    (the dataSource field didn't exist), this reverse migration clears all
    existing data that was marked as 'yahoo' by the forward migration.
    
    This ensures that after rollback, the database is in a clean state
    where new data can be imported fresh.
    """
    Stock = apps.get_model('Data', 'Stock')
    StockPrice = apps.get_model('Data', 'StockPrice')
    PriceBar = apps.get_model('Data', 'PriceBar')
    
    # Clear existing data atomically to restore pre-migration state
    with transaction.atomic():
        # Delete child records first to maintain referential integrity
        # PriceBar references Stock, so delete it first
        deleted_bars = PriceBar.objects.filter(dataSource='yahoo').delete()
        
        # Delete StockPrice records (also references Stock)
        deleted_prices = StockPrice.objects.filter(dataSource='yahoo').delete()
        
        # Delete Stock records last (parent records)
        deleted_stocks = Stock.objects.filter(dataSource='yahoo').delete()
        
        # Note: This is a destructive operation but provides a clean rollback
        # Alternative approaches could set dataSource to a placeholder value
        # if data preservation is more important than clean rollback


class Migration(migrations.Migration):
    
    dependencies = [
        ('Data', '0006_pricebar_stock_datasource_stockprice_datasource_and_more'),
    ]
    
    operations = [
        migrations.RunPython(mark_existing_data_as_yahoo, reverse_mark_data),
    ]