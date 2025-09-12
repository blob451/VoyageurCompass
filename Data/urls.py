"""
URL configuration for Data app.
"""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from Data.market_views import (
    bulk_price_update,
    compare_stocks,
    economic_calendar,
    market_overview,
    sector_performance,
    sync_watchlist,
)
from Data.views import (
    PortfolioHoldingViewSet,
    PortfolioViewSet,
    StockPriceViewSet,
    StockViewSet,
)
from Data.quality_views import (
    data_quality_alerts,
    data_quality_dashboard,
    data_quality_trends,
    run_data_quality_check,
)

app_name = "data"

# Create router for ViewSets
router = DefaultRouter()
router.register(r"stocks", StockViewSet, basename="stock")
router.register(r"prices", StockPriceViewSet, basename="price")
router.register(r"portfolios", PortfolioViewSet, basename="portfolio")
router.register(r"holdings", PortfolioHoldingViewSet, basename="holding")

# Additional URL patterns for custom views
urlpatterns = [
    path("", include(router.urls)),
    # Market data endpoints
    path("market/overview/", market_overview, name="market-overview"),
    path("market/sectors/", sector_performance, name="sector-performance"),
    path("market/calendar/", economic_calendar, name="economic-calendar"),
    # Synchronization endpoints
    path("sync/watchlist/", sync_watchlist, name="sync-watchlist"),
    path("sync/bulk-update/", bulk_price_update, name="bulk-price-update"),
    # Comparison endpoint
    path("compare/", compare_stocks, name="compare-stocks"),
    # Data Quality Monitoring endpoints
    path("quality/dashboard/", data_quality_dashboard, name="data-quality-dashboard"),
    path("quality/check/", run_data_quality_check, name="run-data-quality-check"),
    path("quality/trends/", data_quality_trends, name="data-quality-trends"),
    path("quality/alerts/", data_quality_alerts, name="data-quality-alerts"),
]
