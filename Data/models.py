from django.db import models
from django.core.validators import MinValueValidator
from django.utils import timezone
from django.conf import settings
from datetime import timedelta
from datetime import datetime
from decimal import Decimal
from dateutil.relativedelta import relativedelta

from .managers import StockManager, RealDataManager


class DataSourceChoices(models.TextChoices):
    """Data source enumeration for tracking data origin."""
    YAHOO = 'yahoo', 'Yahoo Finance'
    MOCK = 'mock', 'Mock Data'


class DataSector(models.Model):
    """Normalised sector classification model."""
    
    sectorKey = models.CharField(
        max_length=50, 
        unique=True,
        db_index=True,
        help_text="Normalized sector key (e.g., 'technology', 'healthcare')"
    )
    sectorName = models.CharField(
        max_length=200,
        help_text="Human-readable sector name"
    )
    isActive = models.BooleanField(
        default=True,
        help_text="Whether this sector is actively tracked"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_sync = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Last time sector data was synced"
    )
    data_source = models.CharField(
        max_length=20,
        choices=DataSourceChoices.choices,
        default=DataSourceChoices.YAHOO,
        help_text="Source of the sector data"
    )
    
    class Meta:
        db_table = 'data_sector'
        ordering = ['sectorName']
        verbose_name = 'Data Sector'
        verbose_name_plural = 'Data Sectors'
        indexes = [
            models.Index(fields=['sectorKey']),
            models.Index(fields=['isActive']),
        ]
    
    def __str__(self):
        return self.sectorName


class DataIndustry(models.Model):
    """Normalised industry classification model."""
    
    industryKey = models.CharField(
        max_length=100, 
        unique=True,
        db_index=True,
        help_text="Normalized industry key"
    )
    industryName = models.CharField(
        max_length=200,
        help_text="Human-readable industry name"
    )
    sector = models.ForeignKey(
        DataSector,
        on_delete=models.CASCADE,
        related_name='industries',
        help_text="Parent sector"
    )
    isActive = models.BooleanField(
        default=True,
        help_text="Whether this industry is actively tracked"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_sync = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Last time industry data was synced"
    )
    data_source = models.CharField(
        max_length=20,
        choices=DataSourceChoices.choices,
        default=DataSourceChoices.YAHOO,
        help_text="Source of the industry data"
    )
    
    class Meta:
        db_table = 'data_industry'
        ordering = ['industryName']
        verbose_name = 'Data Industry'
        verbose_name_plural = 'Data Industries'
        indexes = [
            models.Index(fields=['industryKey']),
            models.Index(fields=['sector']),
            models.Index(fields=['isActive']),
        ]
    
    def __str__(self):
        return f"{self.industryName} ({self.sector.sectorName})"


class DataSectorPrice(models.Model):
    """Sector composite price index model."""
    
    sector = models.ForeignKey(
        DataSector,
        on_delete=models.CASCADE,
        related_name='prices',
        help_text="Related sector"
    )
    date = models.DateField(
        db_index=True,
        help_text="Trading date"
    )
    close_index = models.DecimalField(
        max_digits=20, 
        decimal_places=6,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Sector composite close index"
    )
    fiftyTwoWeekChange = models.DecimalField(
        max_digits=10, 
        decimal_places=6,
        null=True,
        blank=True,
        help_text="52-week change percentage"
    )
    fiftyDayAverage = models.DecimalField(
        max_digits=20, 
        decimal_places=6,
        null=True,
        blank=True,
        help_text="50-day moving average"
    )
    twoHundredDayAverage = models.DecimalField(
        max_digits=20, 
        decimal_places=6,
        null=True,
        blank=True,
        help_text="200-day moving average"
    )
    averageVolume = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="Average daily volume"
    )
    averageVolume3months = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="3-month average daily volume"
    )
    volume_agg = models.BigIntegerField(
        default=0,
        help_text="Aggregated volume for sector constituents"
    )
    constituents_count = models.IntegerField(
        help_text="Number of stocks included in composite"
    )
    method = models.CharField(
        max_length=20,
        default='cap_weighted',
        choices=[
            ('cap_weighted', 'Cap Weighted'),
            ('equal_weighted', 'Equal Weighted'),
        ],
        help_text="Composite calculation method"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    data_source = models.CharField(
        max_length=20,
        choices=DataSourceChoices.choices,
        default=DataSourceChoices.YAHOO,
        help_text="Source of the composite data"
    )
    
    class Meta:
        db_table = 'data_sectorprice'
        ordering = ['-date']
        verbose_name = 'Data Sector Price'
        verbose_name_plural = 'Data Sector Prices'
        unique_together = [['sector', 'date']]
        indexes = [
            models.Index(fields=['date']),
            models.Index(fields=['sector', 'date']),
            models.Index(fields=['sector', '-date']),
        ]
    
    def __str__(self):
        return f"{self.sector.sectorName} - {self.date}: {self.close_index}"


class DataIndustryPrice(models.Model):
    """Industry composite price index model."""
    
    industry = models.ForeignKey(
        DataIndustry,
        on_delete=models.CASCADE,
        related_name='prices',
        help_text="Related industry"
    )
    date = models.DateField(
        db_index=True,
        help_text="Trading date"
    )
    close_index = models.DecimalField(
        max_digits=20, 
        decimal_places=6,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Industry composite close index"
    )
    fiftyTwoWeekChange = models.DecimalField(
        max_digits=10, 
        decimal_places=6,
        null=True,
        blank=True,
        help_text="52-week change percentage"
    )
    fiftyDayAverage = models.DecimalField(
        max_digits=20, 
        decimal_places=6,
        null=True,
        blank=True,
        help_text="50-day moving average"
    )
    twoHundredDayAverage = models.DecimalField(
        max_digits=20, 
        decimal_places=6,
        null=True,
        blank=True,
        help_text="200-day moving average"
    )
    averageVolume = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="Average daily volume"
    )
    averageVolume3months = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="3-month average daily volume"
    )
    volume_agg = models.BigIntegerField(
        default=0,
        help_text="Aggregated volume for industry constituents"
    )
    constituents_count = models.IntegerField(
        help_text="Number of stocks included in composite"
    )
    method = models.CharField(
        max_length=20,
        default='cap_weighted',
        choices=[
            ('cap_weighted', 'Cap Weighted'),
            ('equal_weighted', 'Equal Weighted'),
        ],
        help_text="Composite calculation method"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    data_source = models.CharField(
        max_length=20,
        choices=DataSourceChoices.choices,
        default=DataSourceChoices.YAHOO,
        help_text="Source of the composite data"
    )
    
    class Meta:
        db_table = 'data_industryprice'
        ordering = ['-date']
        verbose_name = 'Data Industry Price'
        verbose_name_plural = 'Data Industry Prices'
        unique_together = [['industry', 'date']]
        indexes = [
            models.Index(fields=['date']),
            models.Index(fields=['industry', 'date']),
            models.Index(fields=['industry', '-date']),
        ]
    
    def __str__(self):
        return f"{self.industry.industryName} - {self.date}: {self.close_index}"


class Stock(models.Model):
    """Stock metadata and company information model."""
    
    # Basic identification
    symbol = models.CharField(
        max_length=20, 
        unique=True, 
        db_index=True,
        help_text="Stock ticker symbol (e.g., AAPL, MSFT). Extended length supports test symbols."
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
    # Required Stocks category fields
    currentPrice = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Current trading price"
    )
    previousClose = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Previous day's closing price"
    )
    dayLow = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Day's low price"
    )
    dayHigh = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Day's high price"
    )
    regularMarketPrice = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Regular market price"
    )
    regularMarketOpen = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Regular market open price"
    )
    regularMarketDayLow = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Regular market day low"
    )
    regularMarketDayHigh = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Regular market day high"
    )
    regularMarketPreviousClose = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Regular market previous close"
    )
    fiftyTwoWeekLow = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="52-week low price"
    )
    fiftyTwoWeekHigh = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="52-week high price"
    )
    fiftyTwoWeekChange = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="52-week change percentage"
    )
    fiftyDayAverage = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="50-day moving average"
    )
    twoHundredDayAverage = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="200-day moving average"
    )
    beta = models.DecimalField(
        max_digits=6,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Beta coefficient"
    )
    impliedVolatility = models.DecimalField(
        max_digits=6,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Implied volatility"
    )
    regularMarketVolume = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="Regular market volume"
    )
    averageVolume = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="Average daily volume"
    )
    averageVolume10days = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="10-day average volume"
    )
    averageVolume3months = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="3-month average volume"
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
    
    # Sector/Industry data tracking
    sectorUpdatedAt = models.DateTimeField(
        null=True, 
        blank=True,
        help_text="Last time sector/industry data was updated"
    )
    
    # Foreign key relationships to normalized classification tables
    sector_id = models.ForeignKey(
        DataSector,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='stocks',
        help_text="Related normalized sector"
    )
    industry_id = models.ForeignKey(
        DataIndustry,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='stocks',
        help_text="Related normalized industry"
    )
    
    # Data source tracking
    data_source = models.CharField(
        max_length=10,
        choices=DataSourceChoices.choices,
        default=DataSourceChoices.YAHOO,
        help_text="Source of the stock data"
    )
    
    # Custom managers
    objects = StockManager()
    real_data = RealDataManager()
    
    class Meta:
        ordering = ['symbol']
        verbose_name = 'Stock'
        verbose_name_plural = 'Stocks'
        indexes = [
            models.Index(fields=['symbol']),
            models.Index(fields=['sector', 'industry']),
            models.Index(fields=['is_active', 'symbol']),
            models.Index(fields=['data_source', 'symbol']),
            models.Index(fields=['sectorUpdatedAt']),
            models.Index(fields=['sector_id']),
            models.Index(fields=['industry_id']),
            models.Index(fields=['sector_id', 'industry_id']),
        ]
    
    def __str__(self):
        return f"{self.symbol} - {self.short_name or self.long_name}"
    
    def get_latest_price(self):
        """Retrieve most recent stock price entry."""
        return self.prices.order_by('-date').first()
    
    def get_price_history(self, days=30):
        """Retrieve price history for specified time period."""
        cutoff_date = timezone.now().date() - timedelta(days=days)
        return self.prices.filter(date__gte=cutoff_date).order_by('-date')
    
    @property
    def needs_sync(self):
        """Evaluate data synchronisation requirement."""
        if not self.last_sync:
            return True
        from django.conf import settings
        threshold = getattr(settings, 'STOCK_DATA_SYNC_THRESHOLD_SECONDS', 3600)
        return (timezone.now() - self.last_sync).total_seconds() > threshold

    @property
    def sectorNeedsUpdate(self):
        """Evaluate sector classification update requirement."""
        if not self.sectorUpdatedAt:
            return True
        threshold_date = timezone.now() - relativedelta(years=3)
        return self.sectorUpdatedAt < threshold_date


class StockPrice(models.Model):
    """Historical stock price data model."""
    
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
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Opening price"
    )
    high = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Highest price of the day"
    )
    low = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Lowest price of the day"
    )
    close = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Closing price"
    )
    adjusted_close = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0'))],
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
    
    # Data source tracking
    data_source = models.CharField(
        max_length=10,
        choices=DataSourceChoices.choices,
        default=DataSourceChoices.YAHOO,
        help_text="Source of the price data"
    )
    
    # Custom managers
    objects = models.Manager()
    real_data = RealDataManager()
    
    class Meta:
        ordering = ['-date']
        verbose_name = 'Stock Price'
        verbose_name_plural = 'Stock Prices'
        unique_together = [['stock', 'date']]
        indexes = [
            models.Index(fields=['stock', '-date']),
            models.Index(fields=['date']),
            models.Index(fields=['data_source', 'stock', '-date']),
        ]
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.date}: ${self.close}"
    
    @property
    def daily_change(self):
        """Calculate daily price change with decimal precision."""
        return self.close - self.open
    
    @property
    def daily_change_percent(self):
        """Calculate daily price change percentage with decimal precision."""
        if self.open and self.open != Decimal('0'):
            percentage = (self.daily_change / self.open) * Decimal('100')
            return percentage.quantize(Decimal('0.01'))
        return Decimal('0').quantize(Decimal('0.01'))
    
    @property
    def daily_range(self):
        """Format daily price range as string."""
        return f"{self.low} - {self.high}"
    
    @property
    def is_gain(self):
        """Determine if daily price movement was positive."""
        return self.close > self.open
    
    @property
    def change_amount(self):
        """Daily price change amount (deprecated - use daily_change)."""
        import warnings
        warnings.warn(
            "change_amount is deprecated, use daily_change instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.daily_change
    
    @property
    def change_percent(self):
        """Daily price change percentage (deprecated - use daily_change_percent)."""
        import warnings
        warnings.warn(
            "change_percent is deprecated, use daily_change_percent instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.daily_change_percent



class Portfolio(models.Model):
    """User investment portfolio model."""
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='portfolios',
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
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Initial investment amount"
    )
    current_value = models.DecimalField(
        max_digits=12, 
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(Decimal('0'))],
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
        """Calculate portfolio return percentage."""
        if self.initial_value and self.initial_value != 0:
            return ((self.current_value - self.initial_value) / self.initial_value) * Decimal('100')
        return Decimal('0')
    
    def update_value(self):
        """Recalculate portfolio value from active holdings."""
        total_value = sum(
            (holding.current_value for holding in self.holdings.filter(is_active=True)),
            start=Decimal('0')
        )
        self.current_value = total_value
        self.save(update_fields=['current_value', 'updated_at'])


class PortfolioHolding(models.Model):
    """Individual stock position within portfolio."""
    
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
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Number of shares"
    )
    average_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Average purchase price per share"
    )
    current_price = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Current market price per share"
    )
    
    # Value calculations
    cost_basis = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(Decimal('0'))],
        help_text="Total investment cost"
    )
    current_value = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(Decimal('0'))],
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
        """Calculate derived fields and update portfolio value."""
        # Ensure consistent Decimal type for all calculations
        quantity_decimal = Decimal(str(self.quantity))
        current_price_decimal = Decimal(str(self.current_price))
        
        self.cost_basis = quantity_decimal * self.average_price
        self.current_value = quantity_decimal * current_price_decimal
        self.unrealized_gain_loss = self.current_value - self.cost_basis
        
        if self.cost_basis > 0:
            self.unrealized_gain_loss_percent = (self.unrealized_gain_loss / self.cost_basis) * Decimal('100')
        
        super().save(*args, **kwargs)
        self.portfolio.update_value()


class AnalyticsResults(models.Model):
    """Technical analysis results storage model."""
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='analysis_results',
        null=True,
        blank=True,
        help_text="User who initiated this analysis"
    )
    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
        related_name='analytics_results',
        help_text="Related stock"
    )
    as_of = models.DateTimeField(
        help_text="Analysis timestamp"
    )
    horizon = models.CharField(
        max_length=16,
        choices=[
            ('short', 'Short'),
            ('medium', 'Medium'), 
            ('long', 'Long'),
            ('blend', 'Blend'),
        ],
        default='blend',
        help_text="Analysis time horizon"
    )
    
    # Weighted indicator results (12 indicators)
    w_sma50vs200 = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted SMA 50/200 crossover score"
    )
    w_pricevs50 = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted price vs 50-day SMA score"
    )
    w_rsi14 = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted RSI(14) score"
    )
    w_macd12269 = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted MACD(12,26,9) histogram score"
    )
    w_bbpos20 = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted Bollinger %B (20,2) score"
    )
    w_bbwidth20 = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted Bollinger Bandwidth (20,2) score"
    )
    w_volsurge = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted Volume Surge score"
    )
    w_obv20 = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted OBV 20-day trend score"
    )
    w_rel1y = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted Relative Strength 1Y score"
    )
    w_rel2y = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted Relative Strength 2Y score"
    )
    w_candlerev = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted Candlestick Reversal score"
    )
    w_srcontext = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Weighted Support/Resistance Context score"
    )
    
    # Sentiment analysis fields
    sentimentScore = models.FloatField(
        null=True,
        blank=True,
        help_text="Sentiment score (-1 to 1, where -1 is most negative, 1 is most positive)"
    )
    sentimentLabel = models.CharField(
        max_length=20,
        choices=[
            ('positive', 'Positive'),
            ('negative', 'Negative'),
            ('neutral', 'Neutral'),
        ],
        null=True,
        blank=True,
        help_text="Sentiment classification label"
    )
    sentimentConfidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Sentiment analysis confidence score (0 to 1)"
    )
    newsCount = models.IntegerField(
        null=True,
        blank=True,
        default=0,
        help_text="Number of news articles analyzed for sentiment"
    )
    lastNewsDate = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Date of most recent news article analyzed"
    )
    sentimentSources = models.JSONField(
        default=dict,
        blank=True,
        help_text="Breakdown of sentiment by source: {source: {count: n, avg_score: float}}"
    )
    
    # Composite results
    components = models.JSONField(
        default=dict,
        help_text="Raw and normalized values per indicator: {name: {raw: value, score: normalized_value}}"
    )
    composite_raw = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        null=True,
        blank=True,
        help_text="Sum of all weighted scores"
    )
    score_0_10 = models.IntegerField(
        null=True,
        blank=True,
        help_text="Final composite score (0-10, rounded)"
    )
    
    # LSTM Price Predictions
    prediction_1d = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="1-day price prediction from LSTM model"
    )
    prediction_7d = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="7-day price prediction from LSTM model"
    )
    prediction_30d = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="30-day price prediction from LSTM model"
    )
    prediction_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence score for LSTM predictions (0.0-1.0)"
    )
    model_version = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text="Version of LSTM model used for predictions"
    )
    prediction_timestamp = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when predictions were generated"
    )
    
    # Explanation and Narrative Fields
    explanations_json = models.JSONField(
        default=dict,
        blank=True,
        help_text="Structured explanation data: {indicators: {}, risks: [], recommendations: {}}"
    )
    explanation_method = models.CharField(
        max_length=20,
        choices=[
            ('llm', 'LLM Generated'),
            ('template', 'Template Based'),
            ('hybrid', 'Hybrid Approach'),
        ],
        null=True,
        blank=True,
        help_text="Method used to generate explanations"
    )
    explanation_version = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text="Version of explanation system used"
    )
    narrative_text = models.TextField(
        null=True,
        blank=True,
        help_text="Natural language explanation text"
    )
    narrative_language = models.CharField(
        max_length=5,
        default='en',
        help_text="Language code for narrative text"
    )
    explained_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Timestamp when explanation was generated"
    )
    explanation_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence score for generated explanation (0.0-1.0)"
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'analytics_results'
        ordering = ['-as_of']
        verbose_name = 'Analytics Result'
        verbose_name_plural = 'Analytics Results'
        unique_together = [['user', 'stock', 'as_of']]
        indexes = [
            models.Index(fields=['user', '-as_of']),
            models.Index(fields=['stock', '-as_of']),
            models.Index(fields=['as_of']),
            models.Index(fields=['user', 'stock', '-as_of']),
            models.Index(fields=['stock', 'horizon', '-as_of']),
            models.Index(fields=['explained_at']),
            models.Index(fields=['explanation_method']),
        ]
    
    def __str__(self):
        return f"{self.stock.symbol} - {self.as_of}: {self.score_0_10}/10"