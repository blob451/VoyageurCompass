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

app_name = 'data'

# Create router for ViewSets
router = DefaultRouter()
router.register(r'stocks', StockViewSet, basename='stock')
router.register(r'prices', StockPriceViewSet, basename='price')
router.register(r'portfolios', PortfolioViewSet, basename='portfolio')
router.register(r'holdings', PortfolioHoldingViewSet, basename='holding')

urlpatterns = [
    path('', include(router.urls)),
]