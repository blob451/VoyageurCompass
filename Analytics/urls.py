"""
URL configuration for Analytics app.
"""

from django.urls import path
from Analytics.views import (
    analyze_stock,
    analyze_portfolio,
    batch_analysis,
    get_user_analysis_history,
    get_user_latest_analysis,
    get_analysis_by_id
)

app_name = 'analytics'

urlpatterns = [
    path('analyze/<str:symbol>/', analyze_stock, name='analyze_stock'),
    path('analyze-portfolio/<int:portfolio_id>/', analyze_portfolio, name='analyze_portfolio'),
    path('batch-analysis/', batch_analysis, name='batch_analysis'),
    path('user/history/', get_user_analysis_history, name='user_analysis_history'),
    path('user/<str:symbol>/latest/', get_user_latest_analysis, name='user_latest_analysis'),
    path('analysis/<int:analysis_id>/', get_analysis_by_id, name='get_analysis_by_id'),
]