from django.db import models
from django.core.validators import MinValueValidator
from django.utils import timezone
from django.contrib.auth.models import User
from datetime import timedelta
from decimal import Decimal


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
        """Get price history for the specified number of days."""
        cutoff_date = timezone.now().date() - timedelta(days=days)
        return self.prices.filter(date__gte=cutoff_date).order_by('-date')
    
    @property
    def needs_sync(self):
        """Check if the stock data needs synchronization."""
        if not self.last_sync:
            return True
        # Consider data stale after 1 hour
        return (timezone.now() - self.last_sync).total_seconds() > 3600


class StockPrice(models.Model):
    """
    Model to store historical stock price data.
    """
    
    stock = models.ForeignKey(
        Stock, 
        on_delete=models.CASCADE, 
        related_name='prices',
        help_text="Related stock"
    )
    date = models.DateField(
        db_index=True,
        help_text="Trading date"
    )
    
    # Price data
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
        help_text="Highest price of the day"
    )
    low = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(0)],
        help_text="Lowest price of the day"
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
        help_text="Adjusted closing price (accounts for splits, dividends)"
    )
    
    # Volume data
    volume = models.BigIntegerField(
        default=0,
        help_text="Trading volume"
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
            models.Index(fields=['date']),
        ]
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.date}: ${self.close}"
    
    @property
    def daily_change(self):
        """Calculate the daily price change using Decimal precision."""
        return self.close - self.open
    
    @property
    def daily_change_percent(self):
        """Calculate the daily price change percentage using Decimal precision."""
        if self.open and self.open != 0:
            return (self.daily_change / self.open) * Decimal('100')
        return Decimal('0')
    
    @property
    def daily_range(self):
        """Get the daily price range."""
        return f"{self.low} - {self.high}"
    
    @property
    def is_gain(self):
        """Check if the day was a gain."""
        return self.close > self.open
    
    @property
    def change_amount(self):
        """Calculate the daily price change amount."""
        return self.daily_change
    
    @property
    def change_percent(self):
        """Calculate the daily price change percentage."""
        return self.daily_change_percent


class Portfolio(models.Model):
    """
    Model to store user portfolio information.
    """
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='portfolios',
        null=True,  # Temporarily nullable for migration
        blank=True,
        help_text="Portfolio owner"
    )
    name = models.CharField(
        max_length=100,
        help_text="Portfolio name"
    )
    description = models.TextField(
        blank=True,
        help_text="Portfolio description"
    )
    
    # Portfolio value tracking
    initial_value = models.DecimalField(
        max_digits=12, 
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Initial investment amount"
    )
    current_value = models.DecimalField(
        max_digits=12, 
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Current portfolio value"
    )
    
    # Risk parameters
    risk_tolerance = models.CharField(
        max_length=20,
        choices=[
            ('conservative', 'Conservative'),
            ('moderate', 'Moderate'),
            ('aggressive', 'Aggressive'),
        ],
        default='moderate',
        help_text="Risk tolerance level"
    )
    
    # Metadata
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
        return self.name
    
    def calculate_returns(self):
        """Calculate portfolio returns."""
        if self.initial_value and self.initial_value != 0:
            return ((self.current_value - self.initial_value) / self.initial_value) * Decimal('100')
        return Decimal('0')
    
    def update_value(self):
        """Update the current portfolio value based on holdings."""
        total_value = sum(
            (holding.current_value for holding in self.holdings.filter(is_active=True)),
            start=Decimal('0')
        )
        self.current_value = total_value
        self.save(update_fields=['current_value', 'updated_at'])


class PortfolioHolding(models.Model):
    """
    Model to store individual holdings within a portfolio.
    """
    
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.CASCADE,
        related_name='holdings',
        help_text="Parent portfolio"
    )
    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name='portfolio_holdings',
        help_text="Stock being held"
    )
    
    # Position details
    quantity = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[MinValueValidator(0)],
        help_text="Number of shares"
    )
    average_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(0)],
        help_text="Average purchase price per share"
    )
    current_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Current market price per share"
    )
    
    # Value calculations
    cost_basis = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Total investment cost"
    )
    current_value = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Current market value"
    )
    
    # Performance metrics
    unrealized_gain_loss = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        help_text="Unrealized profit/loss"
    )
    unrealized_gain_loss_percent = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        default=0,
        help_text="Unrealized profit/loss percentage"
    )
    
    # Metadata
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this holding is active"
    )
    purchase_date = models.DateField(
        help_text="Date of initial purchase"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-current_value']
        verbose_name = 'Portfolio Holding'
        verbose_name_plural = 'Portfolio Holdings'
        unique_together = [['portfolio', 'stock']]
    
    def __str__(self):
        return f"{self.portfolio.name} - {self.stock.symbol}: {self.quantity} shares"
    
    def save(self, *args, **kwargs):
        """Override save to calculate derived fields and update portfolio."""
        # Calculate cost basis
        self.cost_basis = self.quantity * self.average_price
        
        # Calculate current value
        self.current_value = self.quantity * self.current_price
        
        # Calculate unrealized gains/losses
        self.unrealized_gain_loss = self.current_value - self.cost_basis
        
        # Calculate percentage
        if self.cost_basis > 0:
            self.unrealized_gain_loss_percent = (self.unrealized_gain_loss / self.cost_basis) * Decimal('100')
        
        super().save(*args, **kwargs)
        
        # Update portfolio total value after saving
        self.portfolio.update_value()