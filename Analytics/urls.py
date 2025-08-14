"""
URL configuration for Analytics app.
"""

from django.urls import path
from Analytics.views import (
    analyze_stock,
    analyze_portfolio,
    batch_analysis
)

app_name = 'analytics'

urlpatterns = [
    path('analyze/<str:symbol>/', analyze_stock, name='analyze_stock'),
    path('analyze-portfolio/<int:portfolio_id>/', analyze_portfolio, name='analyze_portfolio'),
    path('batch-analysis/', batch_analysis, name='batch_analysis'),
]