# Generated migration for Yahoo Finance cache model

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('Data', '0020_extend_stock_symbol_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='YahooFinanceCache',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(db_index=True, max_length=20)),
                ('period', models.CharField(max_length=10)),
                ('data_type', models.CharField(choices=[('info', 'Stock Info'), ('history', 'Price History'), ('financials', 'Financials')], max_length=20)),
                ('data', models.JSONField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('expires_at', models.DateTimeField()),
                ('fetch_success', models.BooleanField(default=True)),
                ('error_message', models.TextField(blank=True, null=True)),
            ],
            options={
                'verbose_name': 'Yahoo Finance Cache',
                'verbose_name_plural': 'Yahoo Finance Cache Entries',
                'db_table': 'data_yahoo_cache',
                'indexes': [
                    models.Index(fields=['symbol', 'period', 'data_type'], name='data_yahoo_sym_per_typ_idx'),
                    models.Index(fields=['expires_at'], name='data_yahoo_expires_idx'),
                ],
            },
        ),
    ]