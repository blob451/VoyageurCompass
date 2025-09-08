from django.shortcuts import render

"""
REST API views for financial data and portfolio management.
"""

from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from django.db import models  
from django.db.models import Q
from datetime import datetime, timedelta
from decimal import Decimal
from django.utils import timezone

from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
from Data.serializers import (
    StockSerializer, StockDetailSerializer, StockPriceSerializer,
    PortfolioSerializer, PortfolioDetailSerializer, PortfolioHoldingSerializer,
    StockSearchSerializer, MarketStatusSerializer
)
from Data.services.yahoo_finance import yahoo_finance_service


class StockViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset for stock data and market information."""
    
    queryset = Stock.objects.filter(is_active=True)
    serializer_class = StockSerializer
    permission_classes = [AllowAny]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['symbol', 'short_name', 'long_name', 'sector', 'industry']
    ordering_fields = ['symbol', 'market_cap', 'last_sync']
    ordering = ['symbol']
    
    def get_serializer_class(self):
        """Select appropriate serialiser based on action type."""
        if self.action == 'retrieve':
            return StockDetailSerializer
        return StockSerializer
    
    @action(detail=True, methods=['get'])
    def prices(self, request, pk=None):
        """Retrieve historical price data with optional day range filter."""
        stock = self.get_object()
        days = int(request.query_params.get('days', 30))
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        prices = StockPrice.objects.filter(
            stock=stock,
            date__gte=start_date,
            date__lte=end_date
        ).order_by('-date')
        
        serializer = StockPriceSerializer(prices, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def sync(self, request, pk=None):
        """Synchronise stock data from external financial data provider."""
        stock = self.get_object()
        period = request.data.get('period', '1mo')
        
        result = yahoo_finance_service.get_stock_data(
            stock.symbol, 
            period=period, 
            sync_db=True
        )
        
        if 'error' in result:
            return Response(
                {'error': result['error']},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        stock.refresh_from_db()
        serializer = self.get_serializer(stock)
        
        return Response({
            'message': 'Stock data synchronized successfully',
            'stock': serializer.data
        })
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Search stocks by symbol or company name."""
        query = request.query_params.get('q', '').strip()
        
        if not query:
            return Response(
                {'error': 'Search query is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = yahoo_finance_service.search_symbols(query)
        serializer = StockSearchSerializer(results, many=True)
        
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def market_status(self, request):
        """Retrieve current market operating status."""
        status_data = yahoo_finance_service.get_market_status()
        serializer = MarketStatusSerializer(status_data)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def historical(self, request, pk=None):
        """Retrieve historical stock data within specified date range."""
        stock = self.get_object()
        
        start_date_str = request.query_params.get('start_date')
        end_date_str = request.query_params.get('end_date')
        
        if not start_date_str or not end_date_str:
            end_date = timezone.now()
            start_date = end_date - timedelta(days=90)
        else:
            try:
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)
                
                if timezone.is_naive(start_date):
                    start_date = timezone.make_aware(start_date)
                if timezone.is_naive(end_date):
                    end_date = timezone.make_aware(end_date)
                
                if start_date > end_date:
                    return Response(
                        {'error': 'start_date must be before end_date'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                    
            except (ValueError, TypeError) as e:
                return Response(
                    {'error': 'Invalid date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        historical_data = yahoo_finance_service.get_historical_data(
            stock.symbol, start_date, end_date
        )
        
        return Response(historical_data)
    
    @action(detail=True, methods=['get'])
    def info(self, request, pk=None):
        """Retrieve comprehensive company information."""
        stock = self.get_object()
        
        try:
            info = yahoo_finance_service.get_stock_info(stock.symbol)
            
            if 'error' in info:
                return Response(
                    {'error': 'Failed to retrieve stock information. Please try again later.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            return Response(info)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Stock info retrieval failed for {stock.symbol}: {str(e)}")
            
            return Response(
                {'error': 'Failed to retrieve stock information. Please try again later.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def batch_sync(self, request):
        """
        Synchronize multiple stocks at once.
        Body: { "symbols": ["AAPL", "MSFT", ...], "period": "1mo" }
        """
        symbols = request.data.get('symbols', [])
        period = request.data.get('period', '1mo')
        
        if not symbols:
            return Response(
                {'error': 'Symbols list is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = yahoo_finance_service.get_multiple_stocks(symbols, period)
        return Response(results)
    
    @action(detail=False, methods=['get'])
    def trending(self, request):
        """
        Get trending stocks from configurable list.
        Symbols can be configured via TRENDING_STOCKS setting or environment variable.
        """
        from django.conf import settings
        from django.core.cache import cache
        
        # Try to get from cache first (cache for 1 hour)
        cache_key = 'trending_stocks_data'
        cached_data = cache.get(cache_key)
        
        if cached_data is not None:
            return Response(cached_data)
        
        # Get trending symbols from settings (configurable via environment)
        trending_symbols = getattr(settings, 'TRENDING_STOCKS', [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM'
        ])
        
        # Filter active stocks and order by symbol for consistency
        stocks = Stock.objects.filter(
            symbol__in=trending_symbols, 
            is_active=True
        ).order_by('symbol')
        
        serializer = self.get_serializer(stocks, many=True)
        response_data = serializer.data
        
        # Cache the result for 1 hour to improve performance
        cache.set(cache_key, response_data, 3600)
        
        return Response(response_data)
    
    @action(detail=False, methods=['get'])
    def indices(self, request):
        """
        Get major market indices.
        Query parameters:
        - include_full_data: Set to 'true' to include the full data payload
        """
        # Check if full data is explicitly requested
        include_full_data = request.query_params.get('include_full_data', 'false').lower() == 'true'
        
        indices = [
            {'symbol': '^GSPC', 'name': 'S&P 500'},
            {'symbol': '^DJI', 'name': 'Dow Jones Industrial Average'},
            {'symbol': '^IXIC', 'name': 'NASDAQ Composite'},
            {'symbol': '^VIX', 'name': 'CBOE Volatility Index'},
            {'symbol': '^TNX', 'name': '10-Year Treasury Yield'}
        ]
        
        result = []
        for index in indices:
            try:
                price_data = yahoo_finance_service.get_stock_data(index['symbol'], period='1d', sync_db=False)
                
                if 'error' not in price_data:
                    # Safely extract essential fields with proper null checks
                    price = None
                    change = None
                    change_percent = None
                    
                    prices_list = price_data.get('prices', [])
                    if prices_list and len(prices_list) > 0 and prices_list[0] is not None:
                        latest_data = prices_list[0]
                        price = latest_data.get('close')
                        change = latest_data.get('change_amount')
                        change_percent = latest_data.get('change_percent')
                    
                    index_summary = {
                        'symbol': index['symbol'],
                        'name': index['name'],
                        'price': price,
                        'change': change,
                        'change_percent': change_percent
                    }
                    
                    # Include full data payload if explicitly requested
                    if include_full_data:
                        index_summary['data'] = price_data
                    
                    result.append(index_summary)
            except Exception as e:
                # Log error but continue with other indices
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to fetch data for index {index['symbol']}: {str(e)}")
                
                # Add entry with null values to maintain consistency
                result.append({
                    'symbol': index['symbol'],
                    'name': index['name'],
                    'price': None,
                    'change': None,
                    'change_percent': None
                })
        
        return Response(result)
    
    @action(detail=False, methods=['post'])
    def realtime_quotes(self, request):
        """
        Get real-time quotes for multiple symbols.
        Body: { "symbols": ["AAPL", "MSFT", ...] }
        """
        symbols = request.data.get('symbols', [])
        
        if not symbols:
            return Response(
                {'error': 'Symbols list is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate symbols list
        if not isinstance(symbols, list):
            return Response(
                {'error': 'Symbols must be provided as a list'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Limit number of symbols to prevent abuse
        if len(symbols) > 50:  # Reasonable limit
            return Response(
                {'error': 'Maximum 50 symbols allowed per request'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            quotes = yahoo_finance_service.get_realtime_quotes(symbols)
            
            # Check if the service returned an error response
            if isinstance(quotes, dict) and 'error' in quotes:
                return Response(
                    {'error': 'Failed to retrieve real-time quotes. Please try again later.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            return Response(quotes)
            
        except Exception as e:
            # Log detailed error for debugging but return generic message
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Real-time quotes retrieval failed for symbols {symbols}: {str(e)}")
            
            return Response(
                {'error': 'Failed to retrieve real-time quotes. Please try again later.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class StockPriceViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset for historical stock price data."""
    
    queryset = StockPrice.objects.all()
    serializer_class = StockPriceSerializer
    permission_classes = [AllowAny]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['date', 'close', 'volume']
    ordering = ['-date']
    
    def get_queryset(self):
        """Apply stock and date range filters to queryset."""
        queryset = super().get_queryset()
        
        symbol = self.request.query_params.get('symbol')
        if symbol:
            queryset = queryset.filter(stock__symbol=symbol.upper())
        
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        
        if start_date:
            queryset = queryset.filter(date__gte=start_date)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)
        
        return queryset


class PortfolioViewSet(viewsets.ModelViewSet):
    """Full CRUD viewset for user portfolio management."""
    
    serializer_class = PortfolioSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['name', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    def get_queryset(self):
        """Filter portfolios to current authenticated user."""
        return Portfolio.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        """Select appropriate serialiser based on action type."""
        if self.action == 'retrieve':
            return PortfolioDetailSerializer
        return PortfolioSerializer
    
    def perform_create(self, serializer):
        """Associate portfolio with authenticated user during creation."""
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def add_holding(self, request, pk=None):
        """Add stock holding to portfolio with validation and synchronisation."""
        portfolio = self.get_object()
        
        stock_symbol = request.data.get('stock_symbol', '').upper()
        
        try:
            stock = Stock.objects.get(symbol=stock_symbol)
        except Stock.DoesNotExist:
            if yahoo_finance_service.validate_symbol(stock_symbol):
                sync_result = yahoo_finance_service.get_stock_data(
                    stock_symbol, 
                    sync_db=True
                )
                if 'error' not in sync_result:
                    stock = Stock.objects.get(symbol=stock_symbol)
                else:
                    return Response(
                        {'error': f'Stock {stock_symbol} not found'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            else:
                return Response(
                    {'error': f'Invalid stock symbol: {stock_symbol}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        data = request.data.copy()
        data['portfolio'] = portfolio.id
        data['stock'] = stock.id
        
        serializer = PortfolioHoldingSerializer(data=data)
        if serializer.is_valid():
            serializer.save(portfolio=portfolio, stock=stock)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def performance(self, request, pk=None):
        """Calculate comprehensive portfolio performance metrics."""
        portfolio = self.get_object()
        holdings = portfolio.holdings.filter(is_active=True)
        
        total_cost = sum(h.cost_basis for h in holdings)
        total_value = sum(h.current_value for h in holdings)
        total_gain_loss = total_value - total_cost
        
        performance = {
            'portfolio_id': portfolio.id,
            'portfolio_name': portfolio.name,
            'total_holdings': holdings.count(),
            'total_cost': float(total_cost),
            'total_value': float(total_value),
            'total_gain_loss': float(total_gain_loss),
            'total_gain_loss_percent': (
                float((total_gain_loss / total_cost) * 100) if total_cost > 0 else 0
            ),
            'holdings': []
        }
        
        for holding in holdings:
            performance['holdings'].append({
                'symbol': holding.stock.symbol,
                'name': holding.stock.short_name or holding.stock.long_name,
                'quantity': float(holding.quantity),
                'cost': float(holding.cost_basis),
                'value': float(holding.current_value),
                'gain_loss': float(holding.unrealized_gain_loss),
                'gain_loss_percent': float(holding.unrealized_gain_loss_percent)
            })
        
        return Response(performance)
    
    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def update_prices(self, request, pk=None):
        """Synchronise current market prices for all portfolio holdings."""
        portfolio = self.get_object()
        holdings = portfolio.holdings.filter(is_active=True)
        
        if not holdings.exists():
            return Response({
                'message': 'No active holdings found in portfolio',
                'updated_holdings': [],
                'new_total_value': float(portfolio.current_value)
            })
        
        stock_symbols = list(holdings.values_list('stock__symbol', flat=True).distinct())
        batch_stock_data = yahoo_finance_service.get_multiple_stocks(stock_symbols, period='1d')
        
        updated_holdings = []
        holdings_to_update = []
        failed_updates = []
        
        for holding in holdings:
            symbol = holding.stock.symbol
            stock_data = batch_stock_data.get(symbol, {})
            
            if stock_data and 'error' not in stock_data:
                latest_price = holding.stock.get_latest_price()
                if latest_price:
                    holding.current_price = latest_price.close
                    holdings_to_update.append(holding)
                    updated_holdings.append({
                        'symbol': symbol,
                        'updated_price': float(holding.current_price)
                    })
                else:
                    failed_updates.append({
                        'symbol': symbol,
                        'error': 'No latest price data available after sync'
                    })
            else:
                error_msg = stock_data.get('error', 'Unknown error') if stock_data else 'No data returned'
                failed_updates.append({
                    'symbol': symbol,
                    'error': error_msg
                })
        
        if holdings_to_update:
            PortfolioHolding.objects.bulk_update(holdings_to_update, ['current_price'], batch_size=100)
        
        if failed_updates:
            import logging
            logger = logging.getLogger(__name__)
            for failure in failed_updates:
                logger.warning(f"Failed to update price for {failure['symbol']}: {failure['error']}")
        
        portfolio.update_value()
        
        return Response({
            'message': 'Portfolio prices updated',
            'updated_holdings': updated_holdings,
            'new_total_value': float(portfolio.current_value)
        })
    
    @action(detail=True, methods=['post'])
    def remove_holding(self, request, pk=None):
        """Deactivate holding within portfolio by stock symbol."""
        portfolio = self.get_object()
        symbol = request.data.get('symbol', '').upper()
        
        if not symbol:
            return Response(
                {'error': 'Stock symbol is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            holding = portfolio.holdings.get(stock__symbol=symbol)
            holding.is_active = False
            holding.save()
            
            portfolio.update_value()
            
            return Response({
                'message': f'Holding {symbol} removed from portfolio',
                'portfolio_value': float(portfolio.current_value)
            })
        except PortfolioHolding.DoesNotExist:
            return Response(
                {'error': f'Holding {symbol} not found in portfolio'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=True, methods=['post'])
    def update_holding(self, request, pk=None):
        """Modify existing portfolio holding attributes with validation."""
        portfolio = self.get_object()
        symbol = request.data.get('symbol', '').upper()
        
        if not symbol:
            return Response(
                {'error': 'Stock symbol is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            holding = portfolio.holdings.get(stock__symbol=symbol)
            
            if 'quantity' in request.data:
                try:
                    quantity = float(request.data['quantity'])
                    if quantity <= 0:
                        return Response(
                            {'error': 'Quantity must be a positive number'},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    holding.quantity = Decimal(str(quantity))
                except (ValueError, TypeError):
                    return Response(
                        {'error': 'Quantity must be a valid number'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                    
            if 'average_price' in request.data:
                try:
                    average_price = float(request.data['average_price'])
                    if average_price <= 0:
                        return Response(
                            {'error': 'Average price must be a positive number'},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    holding.average_price = Decimal(str(average_price))
                except (ValueError, TypeError):
                    return Response(
                        {'error': 'Average price must be a valid number'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                    
            if 'purchase_date' in request.data:
                try:
                    purchase_date_str = str(request.data['purchase_date'])
                    purchase_date = datetime.strptime(purchase_date_str, '%Y-%m-%d').date()
                    
                    from datetime import date
                    if purchase_date > date.today():
                        return Response(
                            {'error': 'Purchase date cannot be in the future'},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    
                    holding.purchase_date = purchase_date
                except (ValueError, TypeError):
                    return Response(
                        {'error': 'Purchase date must be a valid date in YYYY-MM-DD format'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            holding.save()
            
            serializer = PortfolioHoldingSerializer(holding)
            return Response(serializer.data)
            
        except PortfolioHolding.DoesNotExist:
            return Response(
                {'error': f'Holding {symbol} not found in portfolio'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=True, methods=['get'])
    def allocation(self, request, pk=None):
        """Generate portfolio allocation breakdown by stock, sector, and industry."""
        portfolio = self.get_object()
        holdings = portfolio.holdings.filter(is_active=True)
        
        total_value = sum(h.current_value for h in holdings)
        
        allocation = {
            'portfolio_id': portfolio.id,
            'portfolio_name': portfolio.name,
            'total_value': float(total_value),
            'by_stock': [],
            'by_sector': {},
            'by_industry': {}
        }
        
        total_value_float = float(total_value)
        for holding in holdings:
            percentage = float((float(holding.current_value) / total_value_float * 100) if total_value_float > 0 else 0)
            
            allocation['by_stock'].append({
                'symbol': holding.stock.symbol,
                'name': holding.stock.short_name or holding.stock.long_name,
                'value': float(holding.current_value),
                'percentage': percentage
            })
            
            sector = holding.stock.sector or 'Unknown'
            if sector not in allocation['by_sector']:
                allocation['by_sector'][sector] = {'value': 0}
            allocation['by_sector'][sector]['value'] += float(holding.current_value)
            
            industry = holding.stock.industry or 'Unknown'
            if industry not in allocation['by_industry']:
                allocation['by_industry'][industry] = {'value': 0}
            allocation['by_industry'][industry]['value'] += float(holding.current_value)
        
        for sector_data in allocation['by_sector'].values():
            sector_data['percentage'] = float((sector_data['value'] / total_value_float * 100) if total_value_float > 0 else 0)
        
        for industry_data in allocation['by_industry'].values():
            industry_data['percentage'] = float((industry_data['value'] / total_value_float * 100) if total_value_float > 0 else 0)
        
        allocation['by_stock'].sort(key=lambda x: x['value'], reverse=True)
        
        return Response(allocation)


class PortfolioHoldingViewSet(viewsets.ModelViewSet):
    """Full CRUD viewset for individual portfolio holdings."""
    
    serializer_class = PortfolioHoldingSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter holdings to current authenticated user's portfolios."""
        return PortfolioHolding.objects.filter(
            portfolio__user=self.request.user
        )
    
    def perform_create(self, serializer):
        """Validate portfolio ownership before creating holding."""
        portfolio = serializer.validated_data.get('portfolio')
        if portfolio and portfolio.user != self.request.user:
            raise PermissionError("You can only add holdings to your own portfolios")
        serializer.save()