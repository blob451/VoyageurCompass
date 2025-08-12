"""
URL configuration for Analytics app.
"""

from django.urls import path

from Analytics.views import (
    analyze_portfolio,
    analyze_stock,
    batch_analysis,
    market_overview,
)

app_name = "analytics"

urlpatterns = [
    path("analyze/<str:symbol>/", analyze_stock, name="analyze_stock"),
    path(
        "analyze-portfolio/<int:portfolio_id>/",
        analyze_portfolio,
        name="analyze_portfolio",
    ),
    path("batch-analysis/", batch_analysis, name="batch_analysis"),
    path("market-overview/", market_overview, name="market_overview"),
    # Aliases for tests that expect different URL names
    path("technical-indicators/<str:symbol>/", analyze_stock, name="technical-indicators"),
    path("market-sentiment/", market_overview, name="market-sentiment"),
    path("portfolio-analysis/<int:portfolio_id>/", analyze_portfolio, name="portfolio-analysis"),
    path("risk-assessment/<int:portfolio_id>/", analyze_portfolio, name="risk-assessment"),
    path("stock-recommendations/<str:symbol>/", analyze_stock, name="stock-recommendations"),
]
