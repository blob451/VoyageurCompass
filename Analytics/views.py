from django.shortcuts import render

# Create your views here.
"""
API Views for Analytics app.
Provides endpoints for stock analysis and trading signals.
"""

from datetime import datetime
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Data.services.yahoo_finance import yahoo_finance_service
from Data.models import Portfolio


class AnalysisThrottle(UserRateThrottle):
    """Custom throttle for analysis endpoints."""
    rate = '100/hour'


@extend_schema(
    summary="Analyze a stock",
    description="Run comprehensive analysis on a stock including technical indicators and trading signals",
    parameters=[
        OpenApiParameter(
            name='symbol',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description='Stock ticker symbol (e.g., AAPL, MSFT)'
        ),
        OpenApiParameter(
            name='months',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Number of months to analyze (default: 6)'
        ),
        OpenApiParameter(
            name='sync',
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Sync latest data before analysis (default: false)'
        ),
    ],
    responses={
        200: {
            'description': 'Analysis completed successfully',
            'example': {
                'success': True,
                'symbol': 'AAPL',
                'signal': 'BUY',
                'analysis': {}
            }
        },
        400: {'description': 'Invalid parameters'},
        404: {'description': 'Stock not found'},
        429: {'description': 'Rate limit exceeded'}
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
@throttle_classes([AnalysisThrottle])
def analyze_stock(request, symbol):
    """
    Analyze a stock and generate trading signals.
    
    Path Parameters:
        symbol: Stock ticker symbol
    
    Query Parameters:
        months: Number of months to analyze (default: 6)
        sync: Whether to sync data before analysis (default: false)
    
    Returns:
        Comprehensive analysis with trading signals
    """
    # Get parameters
    symbol = symbol.upper()
    months = int(request.query_params.get('months', 6))
    sync = request.query_params.get('sync', 'false').lower() == 'true'
    
    # Validate months parameter
    if months < 1 or months > 24:
        return Response(
            {'error': 'Months parameter must be between 1 and 24'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Sync data if requested
    if sync:
        period = f"{months}mo" if months <= 12 else "2y"
        sync_result = yahoo_finance_service.get_stock_data(
            symbol, 
            period=period, 
            sync_db=True
        )
        
        if 'error' in sync_result:
            return Response(
                {'error': f'Failed to sync data: {sync_result["error"]}'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    # Run analysis
    try:
        engine = TechnicalAnalysisEngine()
        analysis = engine.analyze_stock(symbol)
        
        # New engine always returns successful analysis or raises exception
        
        # Format response for new TA engine
        response_data = {
            'success': True,
            'symbol': analysis['symbol'],
            'analysis_date': analysis['analysis_date'].isoformat(),
            'horizon': analysis['horizon'],
            'composite_score': analysis['score_0_10'],
            'composite_raw': analysis['composite_raw'],
            'indicators': analysis['components'],
            'weighted_scores': {k: float(v) for k, v in analysis['weighted_scores'].items()},
            'analytics_result_id': analysis['analytics_result_id']
        }
        
        return Response(response_data)
        
    except Exception as e:
        return Response(
            {'error': f'Analysis failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Analyze portfolio",
    description="Run analysis on all stocks in a portfolio",
    parameters=[
        OpenApiParameter(
            name='portfolio_id',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description='Portfolio ID'
        ),
    ],
    responses={
        200: {'description': 'Portfolio analysis completed'},
        403: {'description': 'Not authorized to access this portfolio'},
        404: {'description': 'Portfolio not found'}
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analyze_portfolio(request, portfolio_id):
    """
    Analyze all stocks in a portfolio.
    
    Path Parameters:
        portfolio_id: Portfolio ID
    
    Returns:
        Analysis results for all holdings
    """
    # Check portfolio ownership
    try:
        portfolio = Portfolio.objects.get(id=portfolio_id)
        
        if portfolio.user != request.user:
            return Response(
                {'error': 'You are not authorized to analyze this portfolio'},
                status=status.HTTP_403_FORBIDDEN
            )
    except Portfolio.DoesNotExist:
        return Response(
            {'error': 'Portfolio not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Run portfolio analysis (simplified implementation)
    try:
        # For now, disable portfolio analysis - could be implemented later
        # by running individual stock analyses for each holding
        return Response(
            {'error': 'Portfolio analysis not yet implemented with new TA engine'},
            status=status.HTTP_501_NOT_IMPLEMENTED
        )
        
        if not analysis.get('success', True):
            return Response(
                {'error': analysis.get('error', 'Analysis failed')},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response(analysis)
        
    except Exception as e:
        return Response(
            {'error': f'Portfolio analysis failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Batch stock analysis",
    description="Analyze multiple stocks at once",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'symbols': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of stock symbols',
                    'example': ['AAPL', 'MSFT', 'GOOGL']
                },
                'months': {
                    'type': 'integer',
                    'description': 'Analysis period in months',
                    'default': 6
                }
            },
            'required': ['symbols']
        }
    },
    responses={
        200: {'description': 'Batch analysis completed'},
        400: {'description': 'Invalid request data'}
    }
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def batch_analysis(request):
    """
    Analyze multiple stocks in a single request.
    
    Request Body:
        symbols: List of stock symbols (max 10)
        months: Analysis period (default: 6)
    
    Returns:
        Analysis results for all requested stocks
    """
    symbols = request.data.get('symbols', [])
    months = int(request.data.get('months', 6))
    
    # Validate input
    if not symbols:
        return Response(
            {'error': 'Symbols list is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if len(symbols) > 10:
        return Response(
            {'error': 'Maximum 10 symbols allowed per batch'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if months < 1 or months > 24:
        return Response(
            {'error': 'Months parameter must be between 1 and 24'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Run analysis for each symbol with new TA engine
    results = {}
    engine = TechnicalAnalysisEngine()
    
    for symbol in symbols:
        symbol = symbol.upper()
        try:
            analysis = engine.analyze_stock(symbol)
            
            results[symbol] = {
                'success': True,
                'composite_score': analysis['score_0_10'],
                'composite_raw': analysis['composite_raw'],
                'analysis_date': analysis['analysis_date'].isoformat(),
                'horizon': analysis['horizon']
            }
        except Exception as e:
            results[symbol] = {
                'success': False,
                'error': str(e)
            }
    
    return Response({
        'success': True,
        'results': results,
        'total_analyzed': len(results),
        'successful': sum(1 for r in results.values() if r.get('success'))
    })


@extend_schema(
    summary="Get market overview",
    description="Get analysis of major market indices",
    responses={
        200: {'description': 'Market overview data'}
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def market_overview(request):
    """
    Get market overview with major indices analysis.
    
    Returns:
        Analysis of S&P 500, Dow Jones, and NASDAQ
    """
    indices = {
        'SPY': 'S&P 500 ETF',
        'DIA': 'Dow Jones ETF',
        'QQQ': 'NASDAQ ETF',
    }
    
    results = {}
    engine = TechnicalAnalysisEngine()
    
    for symbol, name in indices.items():
        try:
            analysis = engine.analyze_stock(symbol, horizon='short')
            
            results[symbol] = {
                'name': name,
                'composite_score': analysis['score_0_10'],
                'analysis_date': analysis['analysis_date'].isoformat(),
                'horizon': analysis['horizon']
            }
        except:
            results[symbol] = {
                'name': name,
                'error': 'Data unavailable'
            }
    
    # Get market status
    market_status = yahoo_finance_service.get_market_status()
    
    return Response({
        'market_status': market_status,
        'indices': results,
        'timestamp': datetime.now().isoformat()
    })