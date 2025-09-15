"""
Batch API endpoints for high-performance concurrent stock analysis.
Implements batch processing with async capabilities and intelligent caching.
"""

import asyncio
import logging
from typing import List

from django.core.cache import cache
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

from Analytics.services.batch_analysis_service import get_batch_analysis_service
from Core.caching import cache_result, multi_cache
from Core.pagination import paginate_queryset, OptimisedPageNumberPagination
from Data.models import Portfolio, Stock

logger = logging.getLogger(__name__)


class BatchAnalysisThrottle(UserRateThrottle):
    """Throttling for batch analysis endpoints."""
    rate = "100/hour"  # More restrictive for batch operations


@extend_schema(
    summary="Batch analyze multiple stocks",
    description="Analyze multiple stocks concurrently with optimised performance and caching",
    parameters=[
        OpenApiParameter(
            name="symbols",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=True,
            description="Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)",
        ),
        OpenApiParameter(
            name="use_cache",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Use cached results if available (default: true)",
        ),
        OpenApiParameter(
            name="cache_ttl",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Cache time-to-live in seconds (default: 1800)",
        ),
        OpenApiParameter(
            name="max_symbols",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Maximum number of symbols to process (default: 20, max: 50)",
        ),
    ],
    responses={
        200: {
            "description": "Batch analysis completed",
            "example": {
                "batch_id": "batch_1234567890_5",
                "results": {
                    "AAPL": {"symbol": "AAPL", "score": 7, "analysis": {}},
                    "MSFT": {"symbol": "MSFT", "score": 6, "analysis": {}}
                },
                "stats": {
                    "total_symbols": 5,
                    "new_analyses": 3,
                    "cache_hits": 2,
                    "processing_time": 12.34,
                    "success_rate": 1.0
                }
            }
        },
        400: {"description": "Invalid parameters"},
        429: {"description": "Rate limit exceeded"},
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([BatchAnalysisThrottle])
def batch_analyze_stocks(request):
    """
    Analyze multiple stocks concurrently.
    
    Query Parameters:
        symbols: Comma-separated list of stock symbols
        use_cache: Whether to use cached results (default: true)
        cache_ttl: Cache time-to-live in seconds (default: 1800)
        max_symbols: Maximum symbols to process (default: 20, max: 50)
    
    Returns:
        Batch analysis results with performance statistics
    """
    
    # Parse and validate parameters
    symbols_param = request.data.get('symbols') or request.query_params.get('symbols', '')
    if not symbols_param:
        return Response(
            {"error": "symbols parameter is required"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    symbols = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]
    
    # Validate symbol count
    max_symbols = min(
        int(request.query_params.get('max_symbols', 20)),
        50  # Hard limit for performance
    )
    
    if len(symbols) > max_symbols:
        return Response(
            {
                "error": f"Too many symbols requested. Maximum: {max_symbols}, requested: {len(symbols)}",
                "max_allowed": max_symbols
            },
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Parse other parameters
    use_cache = request.query_params.get('use_cache', 'true').lower() == 'true'
    cache_ttl = int(request.query_params.get('cache_ttl', 1800))
    
    try:
        # Get batch analysis service
        batch_service = get_batch_analysis_service()
        
        # Execute batch analysis
        result = batch_service.analyze_stock_batch(
            symbols=symbols,
            user=request.user,
            use_cache=use_cache,
            cache_ttl=cache_ttl
        )
        
        # Add API-specific metadata
        result['api_info'] = {
            'user_id': request.user.id,
            'request_time': result.get('batch_id', '').split('_')[1] if '_' in result.get('batch_id', '') else None,
            'cache_enabled': use_cache,
            'throttle_info': {
                'rate': BatchAnalysisThrottle.rate,
                'scope': 'user'
            }
        }
        
        logger.info(f"Batch analysis completed for user {request.user.id}: {result.get('batch_id')}")
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        return Response(
            {"error": f"Batch analysis failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Analyze user portfolio",
    description="Analyze all stocks in user's active portfolio using batch processing",
    parameters=[
        OpenApiParameter(
            name="portfolio_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=False,
            description="Portfolio ID (optional, uses default portfolio if not specified)",
        ),
    ],
    responses={
        200: {"description": "Portfolio analysis completed"},
        404: {"description": "Portfolio not found"},
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([BatchAnalysisThrottle])
def analyze_portfolio_batch(request, portfolio_id=None):
    """
    Analyze user's portfolio using async batch processing.
    
    Path Parameters:
        portfolio_id: Portfolio ID (optional)
    
    Returns:
        Portfolio analysis results with holdings breakdown
    """
    
    try:
        # Get user's portfolio
        if portfolio_id:
            try:
                portfolio = Portfolio.objects.get(
                    id=portfolio_id, 
                    user=request.user, 
                    is_active=True
                )
            except Portfolio.DoesNotExist:
                return Response(
                    {"error": "Portfolio not found"},
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            # Get user's default portfolio
            portfolio = Portfolio.objects.filter(
                user=request.user, 
                is_active=True
            ).first()
            
            if not portfolio:
                return Response(
                    {"error": "No active portfolio found"},
                    status=status.HTTP_404_NOT_FOUND
                )
        
        # Get portfolio holdings
        holdings = portfolio.holdings.select_related('stock').filter(
            is_active=True,
            quantity__gt=0
        )
        
        if not holdings.exists():
            return Response(
                {
                    "portfolio_id": portfolio.id,
                    "portfolio_name": portfolio.name,
                    "message": "Portfolio has no active holdings",
                    "results": {}
                },
                status=status.HTTP_200_OK
            )
        
        # Extract symbols from holdings
        symbols = [holding.stock.symbol for holding in holdings]
        
        # Get batch analysis service
        batch_service = get_batch_analysis_service()
        
        # Execute portfolio analysis
        result = batch_service.analyze_stock_batch(
            symbols=symbols,
            user=request.user,
            use_cache=True,
            cache_ttl=1800
        )
        
        # Enhance result with portfolio information
        portfolio_result = {
            "portfolio_id": portfolio.id,
            "portfolio_name": portfolio.name,
            "total_holdings": len(symbols),
            "batch_analysis": result,
            "holdings_breakdown": []
        }
        
        # Add holdings details with analysis results
        for holding in holdings:
            symbol = holding.stock.symbol
            analysis_result = result.get('results', {}).get(symbol)
            
            holding_info = {
                "symbol": symbol,
                "quantity": float(holding.quantity),
                "average_cost": float(holding.average_cost) if holding.average_cost else None,
                "current_value": float(holding.current_value) if holding.current_value else None,
                "analysis": analysis_result
            }
            
            portfolio_result["holdings_breakdown"].append(holding_info)
        
        logger.info(f"Portfolio analysis completed for user {request.user.id}, portfolio {portfolio.id}")
        
        return Response(portfolio_result, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Portfolio batch analysis failed: {str(e)}")
        return Response(
            {"error": f"Portfolio analysis failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get batch analysis performance stats",
    description="Retrieve performance statistics for batch analysis service",
    responses={
        200: {"description": "Performance statistics retrieved"},
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
@cache_result(key_prefix="batch_stats", ttl=60)  # Cache for 1 minute
def get_batch_performance_stats(request):
    """
    Get batch analysis service performance statistics.
    
    Returns:
        Performance metrics and cache statistics
    """
    
    try:
        # Get service statistics
        batch_service = get_batch_analysis_service()
        batch_stats = batch_service.get_batch_performance_stats()
        
        # Get multi-tier cache statistics
        cache_stats = multi_cache.get_stats()
        
        # Combine statistics
        performance_stats = {
            "batch_processing": batch_stats,
            "caching": cache_stats,
            "system_info": {
                "max_workers": batch_service.max_workers,
                "cache_ttl_default": 1800,
                "throttle_rate": BatchAnalysisThrottle.rate
            }
        }
        
        return Response(performance_stats, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Failed to get batch performance stats: {str(e)}")
        return Response(
            {"error": f"Failed to retrieve statistics: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Warm cache for popular stocks",
    description="Pre-warm cache for frequently analysed stocks to improve performance",
    parameters=[
        OpenApiParameter(
            name="symbol_list",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Predefined symbol list (sp500, nasdaq100, portfolio) or custom comma-separated symbols",
        ),
    ],
    responses={
        200: {"description": "Cache warming initiated"},
        400: {"description": "Invalid parameters"},
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def warm_analysis_cache(request):
    """
    Warm cache for popular or user-specific stocks.
    
    Query Parameters:
        symbol_list: Predefined list (sp500, nasdaq100, portfolio) or custom symbols
    
    Returns:
        Cache warming status
    """
    
    symbol_list_param = request.query_params.get('symbol_list', 'portfolio')
    
    try:
        symbols = []
        
        if symbol_list_param == 'portfolio':
            # Get user's portfolio stocks
            user_portfolios = Portfolio.objects.filter(
                user=request.user, 
                is_active=True
            )
            
            for portfolio in user_portfolios:
                portfolio_symbols = list(
                    portfolio.holdings.select_related('stock')
                    .filter(is_active=True, quantity__gt=0)
                    .values_list('stock__symbol', flat=True)
                )
                symbols.extend(portfolio_symbols)
            
            symbols = list(set(symbols))  # Remove duplicates
            
        elif symbol_list_param == 'sp500':
            # Top 50 S&P 500 stocks by market cap (simplified list)
            symbols = [
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'GOOG',
                'BRK-B', 'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'CVX',
                'HD', 'LLY', 'ABBV', 'PFE', 'KO', 'AVGO', 'MRK', 'PEP', 'COST',
                'WMT', 'BAC', 'TMO', 'DIS', 'ABT', 'ACN', 'CRM', 'NFLX', 'VZ',
                'ADBE', 'DHR', 'NKE', 'QCOM', 'TXN', 'BMY', 'PM', 'RTX', 'INTC',
                'WFC', 'T', 'COP', 'UPS', 'NEE', 'ORCL'
            ]
            
        elif symbol_list_param == 'nasdaq100':
            # Top NASDAQ 100 tech stocks
            symbols = [
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'GOOG',
                'AVGO', 'NFLX', 'ORCL', 'CRM', 'ADBE', 'AMD', 'INTC', 'QCOM',
                'INTU', 'CSCO', 'TXN', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC'
            ]
            
        else:
            # Custom symbol list
            symbols = [s.strip().upper() for s in symbol_list_param.split(',') if s.strip()]
        
        if not symbols:
            return Response(
                {"error": "No symbols found for cache warming"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Limit number of symbols for performance
        if len(symbols) > 100:
            symbols = symbols[:100]
        
        # Initiate cache warming (async operation)
        batch_service = get_batch_analysis_service()
        
        # Run cache warming in background (non-blocking)
        import threading
        warming_thread = threading.Thread(
            target=batch_service.warm_cache_for_symbols,
            args=(symbols,),
            daemon=True
        )
        warming_thread.start()
        
        return Response(
            {
                "message": "Cache warming initiated",
                "symbol_count": len(symbols),
                "symbol_list": symbol_list_param,
                "status": "in_progress"
            },
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Cache warming failed: {str(e)}")
        return Response(
            {"error": f"Cache warming failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )