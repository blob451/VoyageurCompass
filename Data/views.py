from django.shortcuts import render

# Create your views here.
"""
API Views for Data app.
Provides REST endpoints for stock data and portfolio management.
"""

from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from django.db import models  
from django.db.models import Q
from datetime import datetime, timedelta

from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding
from Data.serializers import (
    StockSerializer, StockDetailSerializer, StockPriceSerializer,
    PortfolioSerializer, PortfolioDetailSerializer, PortfolioHoldingSerializer,
    StockSearchSerializer, MarketStatusSerializer
)
from Data.services.yahoo_finance import yahoo_finance_service


class StockViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for Stock model.
    Provides list, retrieve, and custom actions for stock data.
    """
    
    queryset = Stock.objects.filter(is_active=True)
    serializer_class = StockSerializer
    permission_classes = [AllowAny]  # Public data
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['symbol', 'short_name', 'long_name', 'sector', 'industry']
    ordering_fields = ['symbol', 'market_cap', 'last_sync']
    ordering = ['symbol']
    
    def get_serializer_class(self):
        """Use detailed serializer for retrieve action."""
        if self.action == 'retrieve':
            return StockDetailSerializer
        return StockSerializer
    
    @action(detail=True, methods=['get'])
    def prices(self, request, pk=None):
        """
        Get price history for a specific stock.
        Optional query params: days (default 30)
        """
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
    
    @action(detail=True, methods=['post'])
    def sync(self, request, pk=None):
        """
        Sync stock data from Yahoo Finance.
        Requires authentication.
        """
        self.permission_classes = [IsAuthenticated]
        self.check_permissions(request)
        
        stock = self.get_object()
        period = request.data.get('period', '1mo')
        
        # Call Yahoo Finance service to sync
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
        
        # Refresh stock from database
        stock.refresh_from_db()
        serializer = self.get_serializer(stock)
        
        return Response({
            'message': 'Stock data synchronized successfully',
            'stock': serializer.data
        })
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """
        Search for stocks by symbol or name.
        Query param: q (search query)
        """
        query = request.query_params.get('q', '').strip()
        
        if not query:
            return Response(
                {'error': 'Search query is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Search using Yahoo Finance service
        results = yahoo_finance_service.search_symbols(query)
        serializer = StockSearchSerializer(results, many=True)
        
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def market_status(self, request):
        """Get current market status."""
        status_data = yahoo_finance_service.get_market_status()
        serializer = MarketStatusSerializer(status_data)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def historical(self, request, pk=None):
        """
        Get historical data for a specific stock.
        Query params: start_date, end_date (ISO format)
        """
        stock = self.get_object()
        
        from datetime import datetime
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')
        
        if not start_date or not end_date:
            # Default to last 3 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
        else:
            start_date = datetime.fromisoformat(start_date)
            end_date = datetime.fromisoformat(end_date)
        
        historical_data = yahoo_finance_service.get_historical_data(
            stock.symbol, start_date, end_date
        )
        
        return Response(historical_data)
    
    @action(detail=True, methods=['get'])
    def info(self, request, pk=None):
        """
        Get detailed company information for a stock.
        """
        stock = self.get_object()
        info = yahoo_finance_service.get_stock_info(stock.symbol)
        return Response(info)
    
    @action(detail=False, methods=['post'])
    def batch_sync(self, request):
        """
        Synchronize multiple stocks at once.
        Body: { "symbols": ["AAPL", "MSFT", ...], "period": "1mo" }
        """
        self.permission_classes = [IsAuthenticated]
        self.check_permissions(request)
        
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
        Get trending stocks.
        """
        # For now, return some popular stocks
        trending_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
        stocks = Stock.objects.filter(symbol__in=trending_symbols, is_active=True)
        serializer = self.get_serializer(stocks, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def indices(self, request):
        """
        Get major market indices.
        """
        indices = [
            {'symbol': '^GSPC', 'name': 'S&P 500'},
            {'symbol': '^DJI', 'name': 'Dow Jones Industrial Average'},
            {'symbol': '^IXIC', 'name': 'NASDAQ Composite'},
            {'symbol': '^VIX', 'name': 'CBOE Volatility Index'},
            {'symbol': '^TNX', 'name': '10-Year Treasury Yield'}
        ]
        
        result = []
        for index in indices:
            price_data = yahoo_finance_service.get_stock_data(index['symbol'], period='1d', sync_db=False)
            if 'error' not in price_data:
                result.append({
                    'symbol': index['symbol'],
                    'name': index['name'],
                    'price': price_data.get('prices', [None])[0] if price_data.get('prices') else None,
                    'data': price_data
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
        
        quotes = yahoo_finance_service.get_realtime_quotes(symbols)
        return Response(quotes)


class StockPriceViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for StockPrice model.
    Provides historical price data access.
    """
    
    queryset = StockPrice.objects.all()
    serializer_class = StockPriceSerializer
    permission_classes = [AllowAny]  # Public data
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['date', 'close', 'volume']
    ordering = ['-date']
    
    def get_queryset(self):
        """Filter by stock if provided."""
        queryset = super().get_queryset()
        
        # Filter by stock symbol
        symbol = self.request.query_params.get('symbol')
        if symbol:
            queryset = queryset.filter(stock__symbol=symbol.upper())
        
        # Filter by date range
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        
        if start_date:
            queryset = queryset.filter(date__gte=start_date)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)
        
        return queryset


class PortfolioViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Portfolio model.
    Provides CRUD operations for user portfolios.
    """
    
    serializer_class = PortfolioSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['name', 'created_at', 'updated_at']
    ordering = ['-created_at']
    
    def get_queryset(self):
        """Return portfolios for the current user only."""
        return Portfolio.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        """Use detailed serializer for retrieve action."""
        if self.action == 'retrieve':
            return PortfolioDetailSerializer
        return PortfolioSerializer
    
    def perform_create(self, serializer):
        """Set the user when creating a portfolio."""
        serializer.save(user=self.request.user)
    
    @action(detail=True, methods=['post'])
    def add_holding(self, request, pk=None):
        """
        Add a holding to the portfolio.
        Required: stock_symbol, quantity, purchase_price, purchase_date
        """
        portfolio = self.get_object()
        
        # Validate stock exists
        stock_symbol = request.data.get('stock_symbol', '').upper()
        
        try:
            stock = Stock.objects.get(symbol=stock_symbol)
        except Stock.DoesNotExist:
            # Try to sync from Yahoo Finance
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
        
        # Create holding
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
        """
        Get portfolio performance metrics.
        """
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
    
    @action(detail=True, methods=['post'])
    def update_prices(self, request, pk=None):
        """
        Update current prices for all holdings in the portfolio.
        """
        self.permission_classes = [IsAuthenticated]
        self.check_permissions(request)
        
        portfolio = self.get_object()
        holdings = portfolio.holdings.filter(is_active=True)
        
        updated_holdings = []
        for holding in holdings:
            # Get latest price from Yahoo Finance
            stock_data = yahoo_finance_service.get_stock_data(
                holding.stock.symbol, period='1d', sync_db=True
            )
            
            if 'error' not in stock_data:
                # Update holding with latest price
                latest_price = holding.stock.get_latest_price()
                if latest_price:
                    holding.current_price = latest_price.close
                    holding.save()
                    updated_holdings.append({
                        'symbol': holding.stock.symbol,
                        'updated_price': float(holding.current_price)
                    })
        
        # Update portfolio total value
        portfolio.update_value()
        
        return Response({
            'message': 'Portfolio prices updated',
            'updated_holdings': updated_holdings,
            'new_total_value': float(portfolio.current_value)
        })
    
    @action(detail=True, methods=['post'])
    def remove_holding(self, request, pk=None):
        """
        Remove a holding from the portfolio.
        Body: { "symbol": "AAPL" }
        """
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
            
            # Update portfolio value
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
        """
        Update a holding in the portfolio.
        Body: { "symbol": "AAPL", "quantity": 100, "average_price": 150.00 }
        """
        portfolio = self.get_object()
        symbol = request.data.get('symbol', '').upper()
        
        if not symbol:
            return Response(
                {'error': 'Stock symbol is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            holding = portfolio.holdings.get(stock__symbol=symbol)
            
            # Update fields if provided
            if 'quantity' in request.data:
                holding.quantity = request.data['quantity']
            if 'average_price' in request.data:
                holding.average_price = request.data['average_price']
            if 'purchase_date' in request.data:
                holding.purchase_date = request.data['purchase_date']
            
            holding.save()  # This will trigger recalculation of derived fields
            
            serializer = PortfolioHoldingSerializer(holding)
            return Response(serializer.data)
            
        except PortfolioHolding.DoesNotExist:
            return Response(
                {'error': f'Holding {symbol} not found in portfolio'},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=True, methods=['get'])
    def allocation(self, request, pk=None):
        """
        Get portfolio allocation breakdown.
        """
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
        
        for holding in holdings:
            percentage = float((holding.current_value / total_value * 100) if total_value > 0 else 0)
            
            # Stock allocation
            allocation['by_stock'].append({
                'symbol': holding.stock.symbol,
                'name': holding.stock.short_name or holding.stock.long_name,
                'value': float(holding.current_value),
                'percentage': percentage
            })
            
            # Sector allocation
            sector = holding.stock.sector or 'Unknown'
            if sector not in allocation['by_sector']:
                allocation['by_sector'][sector] = {'value': 0, 'percentage': 0}
            allocation['by_sector'][sector]['value'] += float(holding.current_value)
            allocation['by_sector'][sector]['percentage'] += percentage
            
            # Industry allocation
            industry = holding.stock.industry or 'Unknown'
            if industry not in allocation['by_industry']:
                allocation['by_industry'][industry] = {'value': 0, 'percentage': 0}
            allocation['by_industry'][industry]['value'] += float(holding.current_value)
            allocation['by_industry'][industry]['percentage'] += percentage
        
        # Sort by value
        allocation['by_stock'].sort(key=lambda x: x['value'], reverse=True)
        
        return Response(allocation)


class PortfolioHoldingViewSet(viewsets.ModelViewSet):
    """
    ViewSet for PortfolioHolding model.
    Manages individual holdings within portfolios.
    """
    
    serializer_class = PortfolioHoldingSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Return holdings for the current user's portfolios only."""
        return PortfolioHolding.objects.filter(
            portfolio__user=self.request.user
        )
    
    def perform_create(self, serializer):
        """Ensure the portfolio belongs to the current user."""
        portfolio = serializer.validated_data.get('portfolio')
        if portfolio and portfolio.user != self.request.user:
            raise PermissionError("You can only add holdings to your own portfolios")
        serializer.save()