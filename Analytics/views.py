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

from Analytics.services.engine import analytics_engine
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
        analysis = analytics_engine.run_full_analysis(symbol, analysis_months=months)
        
        if not analysis.get('success'):
            error_msg = analysis.get('error', 'Analysis failed')
            if 'No data found' in error_msg:
                return Response(
                    {'error': f'Stock {symbol} not found or no data available'},
                    status=status.HTTP_404_NOT_FOUND
                )
            return Response(
                {'error': error_msg},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Format response
        response_data = {
            'success': True,
            'symbol': symbol,
            'signal': analysis['signal'],
            'signal_reason': analysis['signal_reason'],
            'analysis': {
                'company': analysis.get('company_name'),
                'sector': analysis.get('sector'),
                'current_price': analysis.get('current_price'),
                'target_price': analysis.get('target_price'),
                'metrics': {
                    'stock_return': analysis.get('stock_return'),
                    'etf_return': analysis.get('etf_return'),
                    'outperformance': analysis.get('outperformance'),
                    'volatility': analysis.get('volatility'),
                    'sharpe_ratio': analysis.get('sharpe_ratio'),
                },
                'technical_indicators': {
                    'rsi': analysis.get('rsi'),
                    'macd': analysis.get('macd'),
                    'bollinger_bands': analysis.get('bollinger_bands'),
                    'ma_20': analysis.get('ma_20'),
                    'ma_50': analysis.get('ma_50'),
                    'ma_200': analysis.get('ma_200'),
                },
                'signals': {
                    'criteria_met': analysis.get('criteria_met'),
                    'rsi_oversold': analysis.get('rsi_oversold'),
                    'rsi_overbought': analysis.get('rsi_overbought'),
                    'price_above_ma20': analysis.get('price_above_ma20'),
                    'price_above_ma50': analysis.get('price_above_ma50'),
                    'price_above_ma200': analysis.get('price_above_ma200'),
                },
                'analysis_date': analysis.get('analysis_date'),
                'analysis_period_months': analysis.get('analysis_period_months'),
            }
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
    
    # Run portfolio analysis
    try:
        analysis = analytics_engine.analyze_portfolio(portfolio_id)
        
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
    
    # Run analysis for each symbol
    results = {}
    for symbol in symbols:
        symbol = symbol.upper()
        try:
            analysis = analytics_engine.run_full_analysis(symbol, analysis_months=months)
            
            if analysis.get('success'):
                results[symbol] = {
                    'success': True,
                    'signal': analysis['signal'],
                    'signal_reason': analysis['signal_reason'],
                    'current_price': analysis.get('current_price'),
                    'stock_return': analysis.get('stock_return'),
                    'volatility': analysis.get('volatility'),
                }
            else:
                results[symbol] = {
                    'success': False,
                    'error': analysis.get('error', 'Analysis failed')
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
    for symbol, name in indices.items():
        try:
            analysis = analytics_engine.run_full_analysis(symbol, analysis_months=3)
            
            if analysis.get('success'):
                results[symbol] = {
                    'name': name,
                    'current_price': analysis.get('current_price'),
                    'return_3m': analysis.get('stock_return'),
                    'volatility': analysis.get('volatility'),
                    'signal': analysis.get('signal'),
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