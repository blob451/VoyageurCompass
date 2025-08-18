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
from Analytics.sentiment_views import stock_sentiment
from Analytics.explanation_views import (
    generate_explanation,
    explanation_service_status,
    get_explanation
)

app_name = 'analytics'

urlpatterns = [
    path('analyze/<str:symbol>/', analyze_stock, name='analyze_stock'),
    path('analyze-portfolio/<int:portfolio_id>/', analyze_portfolio, name='analyze_portfolio'),
    path('batch-analysis/', batch_analysis, name='batch_analysis'),
    path('user/history/', get_user_analysis_history, name='user_analysis_history'),
    path('user/<str:symbol>/latest/', get_user_latest_analysis, name='user_latest_analysis'),
    path('analysis/<int:analysis_id>/', get_analysis_by_id, name='get_analysis_by_id'),
    path('sentiment/<str:symbol>/', stock_sentiment, name='stock_sentiment'),
    
    # Explanation endpoints
    path('explain/<int:analysis_id>/', generate_explanation, name='generate_explanation'),
    path('explanation/<int:analysis_id>/', get_explanation, name='get_explanation'),
    path('explanation-status/', explanation_service_status, name='explanation_status'),
]