"""
Database performance optimization through strategic index creation.
Adds indexes on frequently queried fields to improve query performance.
"""

from django.db import migrations


class Migration(migrations.Migration):
    
    dependencies = [
        ('Data', '0021_add_yahoo_cache_model'),
    ]
    
    operations = [
        # Stock indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_stock_symbol_active ON data_stock(symbol, is_active) WHERE is_active = true;",
            reverse_sql="DROP INDEX IF EXISTS idx_stock_symbol_active;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_stock_data_source ON data_stock(data_source) WHERE data_source != 'mock';",
            reverse_sql="DROP INDEX IF EXISTS idx_stock_data_source;"
        ),
        
        # StockPrice indexes for faster time-series queries
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_stockprice_symbol_date ON data_stockprice(stock_id, date DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_stockprice_symbol_date;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_stockprice_date_range ON data_stockprice(date) WHERE date >= CURRENT_DATE - INTERVAL '2 years';",
            reverse_sql="DROP INDEX IF EXISTS idx_stockprice_date_range;"
        ),
        
        # Portfolio and PortfolioHolding indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_user_active ON data_portfolio(user_id, is_active) WHERE is_active = true;",
            reverse_sql="DROP INDEX IF EXISTS idx_portfolio_user_active;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_holding_portfolio_stock ON data_portfolioholding(portfolio_id, stock_id);",
            reverse_sql="DROP INDEX IF EXISTS idx_holding_portfolio_stock;"
        ),
        
        # AnalyticsResults indexes for quick lookups
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_analytics_stock_date ON data_analyticsresults(stock_id, analysis_date DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_analytics_stock_date;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_analytics_user_date ON data_analyticsresults(user_id, analysis_date DESC) WHERE user_id IS NOT NULL;",
            reverse_sql="DROP INDEX IF EXISTS idx_analytics_user_date;"
        ),
        
        # Sector and Industry composite indexes
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_sector_composite_date ON data_sectorcomposite(sector_id, date DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_sector_composite_date;"
        ),
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_industry_composite_date ON data_industrycomposite(industry_id, date DESC);",
            reverse_sql="DROP INDEX IF EXISTS idx_industry_composite_date;"
        ),
        
        # StockClassification for sector/industry lookups
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_classification_sector_industry ON data_stockclassification(sector, industry);",
            reverse_sql="DROP INDEX IF EXISTS idx_classification_sector_industry;"
        ),
        
        # Partial index for recent data (most frequently accessed)
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS idx_stockprice_recent ON data_stockprice(stock_id, date DESC) WHERE date >= CURRENT_DATE - INTERVAL '90 days';",
            reverse_sql="DROP INDEX IF EXISTS idx_stockprice_recent;"
        ),
    ]