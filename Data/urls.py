"""
URL configuration for Data app.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from Data.views import (
    StockViewSet,
    StockPriceViewSet,
    PortfolioViewSet,
    PortfolioHoldingViewSet
)
from Data.market_views import (
    market_overview,
    sync_watchlist,
    sector_performance,
    compare_stocks,
    economic_calendar,
    bulk_price_update
)

app_name = 'data'

# Create router for ViewSets
router = DefaultRouter()
router.register(r'stocks', StockViewSet, basename='stock')
router.register(r'prices', StockPriceViewSet, basename='price')
router.register(r'portfolios', PortfolioViewSet, basename='portfolio')
router.register(r'holdings', PortfolioHoldingViewSet, basename='holding')

# Additional URL patterns for custom views
urlpatterns = [
    path('', include(router.urls)),
    
    # Market data endpoints
    path('market/overview/', market_overview, name='market-overview'),
    path('market/sectors/', sector_performance, name='sector-performance'),
    path('market/calendar/', economic_calendar, name='economic-calendar'),
    
    # Synchronization endpoints
    path('sync/watchlist/', sync_watchlist, name='sync-watchlist'),
    path('sync/bulk-update/', bulk_price_update, name='bulk-price-update'),
    
    # Comparison endpoint
    path('compare/', compare_stocks, name='compare-stocks'),
]
