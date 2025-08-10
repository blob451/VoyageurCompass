# Generated migration for marking existing data as yahoo source

from django.db import migrations


def mark_existing_data_as_yahoo(apps, schema_editor):
    """Mark all existing data as coming from Yahoo Finance."""
    Stock = apps.get_model('Data', 'Stock')
    StockPrice = apps.get_model('Data', 'StockPrice')
    
    # Update existing stocks
    Stock.objects.all().update(dataSource='yahoo')
    StockPrice.objects.all().update(dataSource='yahoo')


def reverse_mark_data(apps, schema_editor):
    """Reverse migration - no action needed."""
    pass


class Migration(migrations.Migration):
    
    dependencies = [
        ('Data', '0006_pricebar_stock_datasource_stockprice_datasource_and_more'),
    ]
    
    operations = [
        migrations.RunPython(mark_existing_data_as_yahoo, reverse_mark_data),
    ]