# Generated migration to rename dataSource fields to data_source for snake_case consistency

from django.db import migrations


class Migration(migrations.Migration):
    
    dependencies = [
        ('Data', '0007_mark_existing_data_as_yahoo'),
    ]
    
    operations = [
        # Rename dataSource to data_source in Stock model
        migrations.RenameField(
            model_name='stock',
            old_name='dataSource',
            new_name='data_source',
        ),
        # Rename dataSource to data_source in StockPrice model
        migrations.RenameField(
            model_name='stockprice',
            old_name='dataSource',
            new_name='data_source',
        ),
        # Rename dataSource to data_source in PriceBar model
        migrations.RenameField(
            model_name='pricebar',
            old_name='dataSource',
            new_name='data_source',
        ),
    ]