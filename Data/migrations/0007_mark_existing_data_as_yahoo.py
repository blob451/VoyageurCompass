# Generated migration for marking existing data as yahoo source

import logging

from django.db import migrations, transaction

logger = logging.getLogger(__name__)


def mark_existing_data_as_yahoo(apps, schema_editor):
    """Mark all existing data as coming from Yahoo Finance."""
    Stock = apps.get_model("Data", "Stock")
    StockPrice = apps.get_model("Data", "StockPrice")

    # Update existing data atomically to ensure consistency
    with transaction.atomic():
        # Update existing stocks
        Stock.objects.all().update(dataSource="yahoo")
        StockPrice.objects.all().update(dataSource="yahoo")


def reverse_mark_data(apps, schema_editor):
    """
    Safe reverse migration - no-op to prevent data loss.

    This reverse migration does not delete any data to prevent irreversible
    data loss in production environments. The original state before this
    migration cannot be perfectly restored since the dataSource field
    didn't exist previously.

    Manual cleanup may be necessary if a true rollback is required.
    """
    logger.warning(
        "Reverse migration for 0007_mark_existing_data_as_yahoo: "
        "No data will be deleted during rollback to prevent irreversible data loss. "
        "Data marked as 'yahoo' source will remain in the database. "
        "Manual cleanup may be necessary if a true rollback is required."
    )


class Migration(migrations.Migration):

    dependencies = [
        ("Data", "0006_pricebar_stock_datasource_stockprice_datasource_and_more"),
    ]

    operations = [
        migrations.RunPython(mark_existing_data_as_yahoo, reverse_mark_data),
    ]
