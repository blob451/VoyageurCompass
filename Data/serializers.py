"""
Data model serialisers for REST API endpoints.
"""

from rest_framework import serializers
from django.contrib.auth.models import User
from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding


class StockPriceSerializer(serializers.ModelSerializer):
    """Historical stock price data serialisation."""
    
    daily_range = serializers.ReadOnlyField()
    is_gain = serializers.ReadOnlyField()
    
    class Meta:
        model = StockPrice
        fields = [
            'id', 'date', 'open', 'high', 'low', 'close',
            'adjusted_close', 'volume', 'daily_range', 'is_gain'
        ]
        read_only_fields = ['id', 'daily_range', 'is_gain']


class StockSerializer(serializers.ModelSerializer):
    """Stock metadata and company information serialisation."""
    
    latest_price = StockPriceSerializer(source='get_latest_price', read_only=True)
    needs_sync = serializers.ReadOnlyField()
    
    class Meta:
        model = Stock
        fields = [
            'id', 'symbol', 'short_name', 'long_name', 'currency',
            'exchange', 'sector', 'industry', 'country', 'website',
            'description', 'market_cap', 'shares_outstanding',
            'is_active', 'last_sync', 'latest_price', 'needs_sync',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'latest_price', 'needs_sync']


class StockDetailSerializer(StockSerializer):
    """Extended stock serialiser with embedded price history."""
    
    price_history = serializers.SerializerMethodField()
    
    class Meta(StockSerializer.Meta):
        fields = StockSerializer.Meta.fields + ['price_history']
    
    def get_price_history(self, obj):
        """Retrieve configurable price history for stock."""
        days = self.context.get('price_history_days', 30)
        prices = obj.get_price_history(days)
        return StockPriceSerializer(prices, many=True).data


class PortfolioHoldingSerializer(serializers.ModelSerializer):
    """Individual portfolio stock holding serialisation."""
    
    stock = StockSerializer(read_only=True)
    stock_symbol = serializers.CharField(write_only=True, required=False)
    cost_basis = serializers.ReadOnlyField()
    current_value = serializers.ReadOnlyField()
    unrealized_gain_loss = serializers.ReadOnlyField()
    unrealized_gain_loss_percent = serializers.ReadOnlyField()
    
    class Meta:
        model = PortfolioHolding
        fields = [
            'id', 'stock', 'stock_symbol', 'quantity', 'average_price',
            'purchase_date', 'cost_basis', 'current_value',
            'unrealized_gain_loss', 'unrealized_gain_loss_percent', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'stock', 'cost_basis', 'current_value',
            'unrealized_gain_loss', 'unrealized_gain_loss_percent', 'created_at', 'updated_at'
        ]
    
    def create(self, validated_data):
        """Create holding with automatic stock symbol resolution."""
        stock_symbol = validated_data.pop('stock_symbol', None)
        
        if stock_symbol:
            try:
                stock = Stock.objects.get(symbol=stock_symbol.upper())
                validated_data['stock'] = stock
            except Stock.DoesNotExist:
                raise serializers.ValidationError(
                    {'stock_symbol': f'Stock with symbol {stock_symbol} not found'}
                )
        
        return super().create(validated_data)


class PortfolioSerializer(serializers.ModelSerializer):
    """Investment portfolio serialisation with summary metrics."""
    
    user = serializers.ReadOnlyField(source='user.username')
    holdings_count = serializers.SerializerMethodField()
    total_value = serializers.SerializerMethodField()
    
    class Meta:
        model = Portfolio
        fields = [
            'id', 'name', 'description', 'user', 'is_active',
            'holdings_count', 'total_value', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']
    
    def get_holdings_count(self, obj):
        """Calculate total number of holdings in portfolio."""
        return obj.holdings.count()
    
    def get_total_value(self, obj):
        """Calculate aggregate portfolio market value."""
        total = sum(holding.current_value for holding in obj.holdings.all())
        return float(total)


class PortfolioDetailSerializer(PortfolioSerializer):
    """Extended portfolio serialiser with embedded holdings."""
    
    holdings = PortfolioHoldingSerializer(many=True, read_only=True)
    
    class Meta(PortfolioSerializer.Meta):
        fields = PortfolioSerializer.Meta.fields + ['holdings']


class StockSearchSerializer(serializers.Serializer):
    """Stock search result serialisation."""
    
    symbol = serializers.CharField()
    name = serializers.CharField()
    type = serializers.CharField(default='Stock')
    exchange = serializers.CharField(required=False)
    sector = serializers.CharField(required=False)
    from_cache = serializers.BooleanField(default=False)


class MarketStatusSerializer(serializers.Serializer):
    """Market operating status serialisation."""
    
    is_open = serializers.BooleanField()
    current_time = serializers.DateTimeField()
    timezone = serializers.CharField()
    market_hours = serializers.DictField()
    indicators = serializers.DictField()
    next_open = serializers.DateTimeField()
