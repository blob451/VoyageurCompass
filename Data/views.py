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
        holdings = portfolio.holdings.all()
        
        total_cost = sum(h.total_cost for h in holdings)
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
                'cost': float(holding.total_cost),
                'value': float(holding.current_value),
                'gain_loss': float(holding.gain_loss),
                'gain_loss_percent': float(holding.gain_loss_percent)
            })
        
        return Response(performance)


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