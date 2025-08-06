from django.db import models

# Create your models here.
"""
Data Models Module
Database models for storing financial market data in VoyageurCompass.
"""

from django.db import models
from django.core.validators import MinValueValidator
from django.utils import timezone


class Stock(models.Model):
    """
    Model to store stock metadata and company information.
    """
    
    # Basic identification
    symbol = models.CharField(
        max_length=10, 
        unique=True, 
        db_index=True,
        help_text="Stock ticker symbol (e.g., AAPL, MSFT)"
    )
    short_name = models.CharField(
        max_length=100, 
        blank=True,
        help_text="Short company name"
    )
    long_name = models.CharField(
        max_length=255, 
        blank=True,
        help_text="Full company name"
    )
    
    # Trading information
    currency = models.CharField(
        max_length=10, 
        default='USD',
        help_text="Trading currency"
    )
    exchange = models.CharField(
        max_length=50, 
        blank=True,
        help_text="Stock exchange (e.g., NASDAQ, NYSE)"
    )
    
    # Company details
    sector = models.CharField(
        max_length=100, 
        blank=True,
        help_text="Business sector"
    )
    industry = models.CharField(
        max_length=100, 
        blank=True,
        help_text="Industry classification"
    )
    country = models.CharField(
        max_length=50, 
        blank=True,
        help_text="Country of headquarters"
    )
    website = models.URLField(
        blank=True,
        help_text="Company website"
    )
    description = models.TextField(
        blank=True,
        help_text="Company description/summary"
    )
    
    # Financial metrics
    market_cap = models.BigIntegerField(
        default=0,
        help_text="Market capitalization"
    )
    shares_outstanding = models.BigIntegerField(
        default=0,
        help_text="Number of shares outstanding"
    )
    
    # Tracking fields
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this stock is actively tracked"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_sync = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Last time data was synced from Yahoo Finance"
    )
    
    class Meta:
        ordering = ['symbol']
        verbose_name = 'Stock'
        verbose_name_plural = 'Stocks'
        indexes = [
            models.Index(fields=['symbol']),
            models.Index(fields=['sector', 'industry']),
            models.Index(fields=['is_active', 'symbol']),
        ]
    
    def __str__(self):
        return f"{self.symbol} - {self.short_name or self.long_name}"
    
    def get_latest_price(self):
        """Get the most recent stock price."""
        return self.prices.order_by('-date').first()
    
    def get_price_history(self, days=30):
        """Get price history for specified number of days."""
        return self.prices.order_by('-date')[:days]
    
    @property
    def needs_sync(self):
        """Check if stock data needs to be synced (older than 1 day)."""
        if not self.last_sync:
            return True
        time_since_sync = timezone.now() - self.last_sync
        return time_since_sync.total_seconds() > 86400  # 24 hours


class StockPrice(models.Model):
    """
    Model to store daily stock price data.
    """
    
    # Relationship to Stock
    stock = models.ForeignKey(
        Stock, 
        on_delete=models.CASCADE, 
        related_name='prices',
        help_text="Related stock"
    )
    
    # Date and price data
    date = models.DateField(
        db_index=True,
        help_text="Trading date"
    )
    open = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(0)],
        help_text="Opening price"
    )
    high = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(0)],
        help_text="Daily high price"
    )
    low = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(0)],
        help_text="Daily low price"
    )
    close = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(0)],
        help_text="Closing price"
    )
    adjusted_close = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(0)],
        null=True,
        blank=True,
        help_text="Adjusted closing price (for splits/dividends)"
    )
    
    # Volume data
    volume = models.BigIntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Trading volume"
    )
    
    # Calculated fields
    change_amount = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Price change from previous close"
    )
    change_percent = models.DecimalField(
        max_digits=5, 
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Percentage change from previous close"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date']
        verbose_name = 'Stock Price'
        verbose_name_plural = 'Stock Prices'
        unique_together = [['stock', 'date']]
        indexes = [
            models.Index(fields=['stock', '-date']),
            models.Index(fields=['-date']),
        ]
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.date}: ${self.close}"
    
    def save(self, *args, **kwargs):
        """Calculate change fields before saving."""
        if self.pk is None:  # New record
            # Try to get previous day's closing price
            previous_price = StockPrice.objects.filter(
                stock=self.stock,
                date__lt=self.date
            ).order_by('-date').first()
            
            if previous_price:
                self.change_amount = self.close - previous_price.close
                if previous_price.close > 0:
                    self.change_percent = (self.change_amount / previous_price.close) * 100
        
        super().save(*args, **kwargs)
    
    @property
    def daily_range(self):
        """Get the daily price range as a string."""
        return f"${self.low} - ${self.high}"
    
    @property
    def is_gain(self):
        """Check if this day was a gain."""
        return self.change_amount and self.change_amount > 0


class Portfolio(models.Model):
    """
    Model to store user portfolios.
    """
    
    name = models.CharField(
        max_length=100,
        help_text="Portfolio name"
    )
    description = models.TextField(
        blank=True,
        help_text="Portfolio description"
    )
    user = models.ForeignKey(
        'auth.User',
        on_delete=models.CASCADE,
        related_name='portfolios',
        help_text="Portfolio owner"
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this portfolio is active"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Portfolio'
        verbose_name_plural = 'Portfolios'
    
    def __str__(self):
        return f"{self.name} - {self.user.username}"


class PortfolioHolding(models.Model):
    """
    Model to store individual stock holdings in a portfolio.
    """
    
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.CASCADE,
        related_name='holdings',
        help_text="Related portfolio"
    )
    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name='portfolio_holdings',
        help_text="Stock being held"
    )
    quantity = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[MinValueValidator(0)],
        help_text="Number of shares"
    )
    purchase_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        help_text="Average purchase price per share"
    )
    purchase_date = models.DateField(
        help_text="Date of purchase"
    )
    notes = models.TextField(
        blank=True,
        help_text="Additional notes about this holding"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['stock__symbol']
        verbose_name = 'Portfolio Holding'
        verbose_name_plural = 'Portfolio Holdings'
        unique_together = [['portfolio', 'stock']]
    
    def __str__(self):
        return f"{self.portfolio.name} - {self.stock.symbol}: {self.quantity} shares"
    
    @property
    def total_cost(self):
        """Calculate total cost of this holding."""
        return self.quantity * self.purchase_price
    
    @property
    def current_value(self):
        """Calculate current value based on latest price."""
        latest_price = self.stock.get_latest_price()
        if latest_price:
            return self.quantity * latest_price.close
        return 0
    
    @property
    def gain_loss(self):
        """Calculate gain/loss for this holding."""
        return self.current_value - self.total_cost
    
    @property
    def gain_loss_percent(self):
        """Calculate gain/loss percentage."""
        if self.total_cost > 0:
            return (self.gain_loss / self.total_cost) * 100
        return 0