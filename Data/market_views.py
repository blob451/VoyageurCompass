"""
Additional API views for market data and synchronization.
"""

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from django.core.cache import cache
from datetime import datetime, timedelta
import logging

from Data.models import Stock, StockPrice
from Data.services.yahoo_finance import yahoo_finance_service
from Data.serializers import StockSerializer, StockPriceSerializer

logger = logging.getLogger(__name__)


@api_view(['GET'])
@permission_classes([AllowAny])
def market_overview(request):
    """
    Get a comprehensive market overview including indices, sectors, and trending stocks.
    """
    cache_key = 'market_overview'
    cached_data = cache.get(cache_key)
    
    if cached_data:
        return Response(cached_data)
    
    try:
        # Get major indices
        indices = [
            {'^GSPC': 'S&P 500'},
            {'^DJI': 'Dow Jones'},
            {'^IXIC': 'NASDAQ'},
            {'^VIX': 'Volatility Index'},
            {'^TNX': '10-Year Treasury'}
        ]
        
        index_data = []
        for index_dict in indices:
            for symbol, name in index_dict.items():
                data = yahoo_finance_service.get_stock_data(symbol, period='1d', sync_db=False)
                if 'error' not in data:
                    index_data.append({
                        'symbol': symbol,
                        'name': name,
                        'price': data.get('prices', [None])[0] if data.get('prices') else None,
                        'change': None  # Calculate if historical data available
                    })
        
        # Get top gainers and losers from database
        active_stocks = Stock.objects.filter(is_active=True)
        gainers = []
        losers = []
        
        for stock in active_stocks[:50]:  # Limit to avoid performance issues
            latest_price = stock.get_latest_price()
            if latest_price and latest_price.daily_change_percent:
                stock_info = {
                    'symbol': stock.symbol,
                    'name': stock.short_name or stock.long_name,
                    'price': float(latest_price.close),
                    'change_percent': float(latest_price.daily_change_percent)
                }
                
                if latest_price.daily_change_percent > 0:
                    gainers.append(stock_info)
                else:
                    losers.append(stock_info)
        
        # Sort and limit
        gainers = sorted(gainers, key=lambda x: x['change_percent'], reverse=True)[:5]
        losers = sorted(losers, key=lambda x: x['change_percent'])[:5]
        
        # Get market status
        market_status = yahoo_finance_service.get_market_status()
        
        overview = {
            'market_status': market_status,
            'indices': index_data,
            'top_gainers': gainers,
            'top_losers': losers,
            'last_updated': datetime.now().isoformat()
        }
        
        # Cache for 5 minutes
        cache.set(cache_key, overview, 300)
        
        return Response(overview)
        
    except Exception as e:
        logger.error(f"Error generating market overview: {str(e)}")
        return Response(
            {'error': 'Failed to generate market overview'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def sync_watchlist(request):
    """
    Synchronize a watchlist of stocks.
    Body: { "symbols": ["AAPL", "MSFT", ...], "period": "1mo" }
    """
    symbols = request.data.get('symbols', [])
    period = request.data.get('period', '1mo')
    
    if not symbols:
        return Response(
            {'error': 'Symbols list is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Validate symbols
        validated_symbols = []
        for symbol in symbols[:50]:  # Limit to 50 symbols
            try:
                validated_symbol = yahoo_finance_service.validateSymbol(symbol)
                validated_symbols.append(validated_symbol)
            except ValueError as e:
                logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        
        # Sync all symbols
        results = yahoo_finance_service.get_multiple_stocks(validated_symbols, period)
        
        # Count successes and failures
        success_count = sum(1 for r in results.values() if 'error' not in r)
        failure_count = len(results) - success_count
        
        return Response({
            'message': f'Synchronized {success_count} stocks',
            'success_count': success_count,
            'failure_count': failure_count,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error syncing watchlist: {str(e)}")
        return Response(
            {'error': 'Failed to sync watchlist'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def sector_performance(request):
    """
    Get performance metrics grouped by sector.
    """
    cache_key = 'sector_performance'
    cached_data = cache.get(cache_key)
    
    if cached_data:
        return Response(cached_data)
    
    try:
        # Get all active stocks grouped by sector
        sectors = {}
        stocks = Stock.objects.filter(is_active=True, sector__isnull=False)
        
        for stock in stocks:
            sector = stock.sector
            if sector not in sectors:
                sectors[sector] = {
                    'name': sector,
                    'stocks': [],
                    'avg_change': 0,
                    'total_market_cap': 0
                }
            
            latest_price = stock.get_latest_price()
            if latest_price:
                sectors[sector]['stocks'].append({
                    'symbol': stock.symbol,
                    'name': stock.short_name or stock.long_name,
                    'change_percent': float(latest_price.daily_change_percent)
                })
                sectors[sector]['total_market_cap'] += stock.market_cap
        
        # Calculate average change per sector
        for sector_data in sectors.values():
            if sector_data['stocks']:
                total_change = sum(s['change_percent'] for s in sector_data['stocks'])
                sector_data['avg_change'] = total_change / len(sector_data['stocks'])
                # Keep only top 5 stocks per sector
                sector_data['stocks'] = sorted(
                    sector_data['stocks'], 
                    key=lambda x: abs(x['change_percent']), 
                    reverse=True
                )[:5]
        
        # Sort sectors by average change
        sorted_sectors = sorted(
            sectors.values(), 
            key=lambda x: x['avg_change'], 
            reverse=True
        )
        
        result = {
            'sectors': sorted_sectors,
            'last_updated': datetime.now().isoformat()
        }
        
        # Cache for 10 minutes
        cache.set(cache_key, result, 600)
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"Error getting sector performance: {str(e)}")
        return Response(
            {'error': 'Failed to get sector performance'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def compare_stocks(request):
    """
    Compare multiple stocks side by side.
    Body: { "symbols": ["AAPL", "MSFT", "GOOGL"], "metrics": ["price", "volume", "market_cap"] }
    """
    symbols = request.data.get('symbols', [])
    metrics = request.data.get('metrics', ['price', 'change_percent', 'volume', 'market_cap'])
    
    if not symbols or len(symbols) < 2:
        return Response(
            {'error': 'At least 2 symbols are required for comparison'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if len(symbols) > 10:
        return Response(
            {'error': 'Maximum 10 symbols allowed for comparison'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        comparison = {}
        
        for symbol in symbols:
            symbol = symbol.upper()
            
            # Try to get from database first
            try:
                stock = Stock.objects.get(symbol=symbol)
                latest_price = stock.get_latest_price()
                
                comparison[symbol] = {
                    'symbol': symbol,
                    'name': stock.short_name or stock.long_name,
                    'sector': stock.sector,
                    'industry': stock.industry
                }
                
                # Add requested metrics
                if 'price' in metrics and latest_price:
                    comparison[symbol]['price'] = float(latest_price.close)
                
                if 'change_percent' in metrics and latest_price:
                    comparison[symbol]['change_percent'] = float(latest_price.daily_change_percent)
                
                if 'volume' in metrics and latest_price:
                    comparison[symbol]['volume'] = latest_price.volume
                
                if 'market_cap' in metrics:
                    comparison[symbol]['market_cap'] = stock.market_cap
                
                if 'pe_ratio' in metrics:
                    # Would need to add PE ratio to model or fetch from Yahoo
                    comparison[symbol]['pe_ratio'] = None
                
            except Stock.DoesNotExist:
                # Fetch from Yahoo Finance if not in database
                stock_info = yahoo_finance_service.get_stock_info(symbol)
                if 'error' not in stock_info:
                    comparison[symbol] = {
                        'symbol': symbol,
                        'name': stock_info.get('longName', stock_info.get('shortName')),
                        'price': stock_info.get('currentPrice'),
                        'from_cache': False
                    }
        
        return Response({
            'comparison': list(comparison.values()),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error comparing stocks: {str(e)}")
        return Response(
            {'error': 'Failed to compare stocks'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([AllowAny])
def economic_calendar(request):
    """
    Get upcoming economic events and earnings releases.
    Query params: days (default 7)
    """
    days = int(request.query_params.get('days', 7))
    
    # This would typically integrate with an economic calendar API
    # For now, return a placeholder structure
    
    return Response({
        'message': 'Economic calendar integration pending',
        'events': [],
        'earnings': [],
        'start_date': datetime.now().date().isoformat(),
        'end_date': (datetime.now().date() + timedelta(days=days)).isoformat()
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def bulk_price_update(request):
    """
    Update prices for all active stocks in the database.
    This should typically be run as a scheduled task.
    """
    try:
        # Get all active stocks
        active_stocks = Stock.objects.filter(is_active=True)
        
        updated_count = 0
        failed_count = 0
        
        for stock in active_stocks:
            try:
                # Sync latest data
                result = yahoo_finance_service.get_stock_data(
                    stock.symbol, 
                    period='1d', 
                    sync_db=True
                )
                
                if 'error' not in result:
                    updated_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to update {stock.symbol}: {str(e)}")
                failed_count += 1
        
        return Response({
            'message': 'Bulk price update completed',
            'updated': updated_count,
            'failed': failed_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in bulk price update: {str(e)}")
        return Response(
            {'error': 'Failed to perform bulk update'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )