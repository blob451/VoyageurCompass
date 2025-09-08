"""
Additional API views for market data and synchronization.
"""

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.response import Response
from django.core.cache import cache
from django.utils import timezone
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
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'Volatility Index',
            '^TNX': '10-Year Treasury'
        }
        
        index_data = []
        for symbol, name in indices.items():
            data = yahoo_finance_service.get_stock_data(symbol, period='1d', sync_db=False)
            if 'error' not in data:
                index_data.append({
                    'symbol': symbol,
                    'name': name,
                    'price': data.get('prices', [None])[0] if data.get('prices') else None,
                    'change': None  # Calculate if historical data available
                })
        
        # Get top gainers and losers from database
        # Use iterator to avoid loading all stocks into memory, limit to 200 for performance
        active_stocks_iterator = Stock.objects.filter(is_active=True).iterator(chunk_size=50)
        gainers = []
        losers = []
        processed_count = 0
        
        for stock in active_stocks_iterator:
            if processed_count >= 200:  # Reasonable limit to avoid performance issues
                break
                
            latest_price = stock.get_latest_price()
            if latest_price and latest_price.daily_change_percent is not None:
                stock_info = {
                    'symbol': stock.symbol,
                    'name': stock.short_name or stock.long_name,
                    'price': float(latest_price.close),
                    'change_percent': float(latest_price.daily_change_percent)
                }
                
                if latest_price.daily_change_percent > 0:
                    gainers.append(stock_info)
                elif latest_price.daily_change_percent < 0:
                    losers.append(stock_info)
                # Note: stocks with 0% change are not included in gainers/losers lists
                # but are still processed and counted
            
            processed_count += 1
        
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
            'last_updated': timezone.now().isoformat()
        }
        
        # Cache for 5 minutes
        cache.set(cache_key, overview, 300)
        
        return Response(overview)
        
    except Exception as e:
        logger.exception("Error generating market overview")
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
                if yahoo_finance_service.validate_symbol(symbol):
                    validated_symbols.append(symbol.upper())
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
        logger.error(f"Error syncing watchlist: {str(e)}", exc_info=True)
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
        # Get all active stocks grouped by sector using iterator to avoid memory issues
        sectors = {}
        stocks_iterator = Stock.objects.filter(
            is_active=True, 
            sector__isnull=False
        ).iterator(chunk_size=100)
        
        for stock in stocks_iterator:
            sector = stock.sector
            if sector not in sectors:
                sectors[sector] = {
                    'name': sector,
                    'stocks': [],
                    'avg_change': 0,
                    'total_market_cap': 0
                }
            
            latest_price = stock.get_latest_price()
            if latest_price and latest_price.daily_change_percent is not None:
                sectors[sector]['stocks'].append({
                    'symbol': stock.symbol,
                    'name': stock.short_name or stock.long_name,
                    'change_percent': float(latest_price.daily_change_percent)
                })
                # Only add market cap if it's not None
                if stock.market_cap is not None:
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
            'last_updated': timezone.now().isoformat()
        }
        
        # Cache for 10 minutes
        cache.set(cache_key, result, 600)
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"Error getting sector performance: {str(e)}", exc_info=True)
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
        # Initialize comparison dictionary with all requested symbols to preserve order and count
        comparison = {}
        for symbol in symbols:
            comparison[symbol.upper()] = {'symbol': symbol.upper()}
        
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
                else:
                    # Preserve symbol with error message when Yahoo Finance lookup fails
                    comparison[symbol] = {
                        'symbol': symbol,
                        'error': f'Stock data not found for {symbol}'
                    }
        
        return Response({
            'comparison': list(comparison.values()),
            'metrics': metrics,
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error comparing stocks: {str(e)}", exc_info=True)
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
    
    # TODO: Replace this placeholder implementation with actual economic calendar API integration
    # Suggested APIs: Alpha Vantage Economic Calendar, Trading Economics API, or FMP Calendar API
    # Implementation should:
    # 1. Fetch real economic events (GDP, unemployment, interest rates, etc.)
    # 2. Retrieve earnings announcements for the specified date range
    # 3. Parse and format the data consistently
    # 4. Handle API errors and rate limiting appropriately
    # 5. Cache results to improve performance
    
    return Response({
        'message': 'Economic calendar integration pending',
        'events': [],
        'earnings': [],
        'start_date': timezone.now().date().isoformat(),
        'end_date': (timezone.now().date() + timedelta(days=days)).isoformat()
    })


@api_view(['POST'])
@permission_classes([IsAdminUser])  # Restrict to admin users only
def bulk_price_update(request):
    """
    Update prices for all active stocks in the database.
    This should typically be run as a scheduled task.
    Restricted to admin users only due to resource intensity.
    """
    try:
        # Gather all active stock symbols for batch processing
        active_stocks = Stock.objects.filter(is_active=True)
        stock_symbols = list(active_stocks.values_list('symbol', flat=True))
        
        if not stock_symbols:
            return Response({
                'message': 'No active stocks found to update',
                'updated': 0,
                'failed': 0,
                'timestamp': timezone.now().isoformat()
            })
        
        logger.info(f"Starting bulk price update for {len(stock_symbols)} stocks")
        
        # Process stocks in batches to avoid overwhelming the API
        batch_size = 100
        total_updated = 0
        total_failed = 0
        
        for i in range(0, len(stock_symbols), batch_size):
            batch_symbols = stock_symbols[i:i + batch_size]
            
            try:
                # Make a single batch API call for this chunk
                batch_results = yahoo_finance_service.get_multiple_stocks(
                    batch_symbols, 
                    period='1d'
                )
                
                # Count successes and failures in the batch
                batch_updated = 0
                batch_failed = 0
                
                for symbol, result in batch_results.items():
                    if result and 'error' not in result:
                        batch_updated += 1
                    else:
                        batch_failed += 1
                        logger.warning(f"Failed to update {symbol}: {result.get('error', 'Unknown error')}")
                
                total_updated += batch_updated
                total_failed += batch_failed
                
                # Log progress
                progress_pct = ((i + len(batch_symbols)) / len(stock_symbols)) * 100
                logger.info(f"Batch {i//batch_size + 1} completed: {batch_updated} updated, {batch_failed} failed. Progress: {progress_pct:.1f}%")
                
            except Exception as e:
                logger.error(f"Batch processing failed for symbols {batch_symbols[:5]}...: {str(e)}")
                total_failed += len(batch_symbols)
        
        logger.info(f"Bulk price update completed: {total_updated} updated, {total_failed} failed")
        
        return Response({
            'message': 'Bulk price update completed',
            'updated': total_updated,
            'failed': total_failed,
            'total_stocks': len(stock_symbols),
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in bulk price update: {str(e)}", exc_info=True)
        return Response(
            {'error': 'Failed to perform bulk update'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )