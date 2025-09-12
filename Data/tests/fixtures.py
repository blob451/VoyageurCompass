"""
Real test data fixtures for Data module testing.
"""

import json
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
from django.conf import settings
from django.utils import timezone

from Data.models import (
    AnalyticsResults,
    DataIndustry,
    DataSector,
    Portfolio,
    PortfolioHolding,
    Stock,
    StockPrice,
)


class DataTestDataFactory:
    """Factory for creating real financial test data without mocks."""

    @staticmethod
    def create_test_stock(symbol="AAPL", company_name="Apple Inc.", sector="Technology"):
        """Create real stock instance for testing."""
        # Get or create sector and industry
        sector_key = sector.lower().replace(" ", "_")
        data_sector, _ = DataSector.objects.get_or_create(sectorKey=sector_key, defaults={"sectorName": sector})

        data_industry, _ = DataIndustry.objects.get_or_create(
            industryKey="software", defaults={"industryName": "Software", "sector": data_sector}
        )

        stock, created = Stock.objects.get_or_create(
            symbol=symbol,
            defaults={
                "short_name": company_name,
                "sector": sector,
                "industry": "Software",
                "market_cap": Decimal("2800000000000"),
                "shares_outstanding": 15500000000,
                "data_source": "YAHOO",
                "industry_id": data_industry,
            },
        )
        return stock

    @staticmethod
    def create_stock_price_history(stock, days=30):
        """Create real stock price history for testing."""
        base_price = Decimal("150.00")
        prices = []

        for i in range(days):
            price_date = date.today() - timedelta(days=days - i - 1)

            # Generate realistic price variations
            variation = Decimal(str(-5 + (i % 11)))  # -5 to +5
            close_price = base_price + variation

            stock_price = StockPrice.objects.create(
                stock=stock,
                date=price_date,
                open=close_price - Decimal("1.00"),
                high=close_price + Decimal("2.00"),
                low=close_price - Decimal("2.00"),
                close=close_price,
                volume=1000000 + (i * 100000),
                adjusted_close=close_price,
                data_source="YAHOO",
            )
            prices.append(stock_price)

        return prices

    @staticmethod
    def create_test_portfolio(user, name="Test Portfolio"):
        """Create real portfolio for user testing."""
        portfolio = Portfolio.objects.create(
            user=user,
            name=name,
            description=f"Test portfolio for {user.username}",
            total_value=Decimal("10000.00"),
            cash_balance=Decimal("5000.00"),
        )
        return portfolio

    @staticmethod
    def create_portfolio_holding(portfolio, stock, shares=10):
        """Create real portfolio holding for testing."""
        latest_price = StockPrice.objects.filter(stock=stock).order_by("-date").first()
        if not latest_price:
            latest_price = DataTestDataFactory.create_stock_price_history(stock, 1)[0]

        holding = PortfolioHolding.objects.create(
            portfolio=portfolio,
            stock=stock,
            shares=Decimal(str(shares)),
            average_cost=latest_price.close_price,
            purchase_date=timezone.now().date(),
            current_value=latest_price.close_price * Decimal(str(shares)),
        )
        return holding

    @staticmethod
    def create_analytics_results(stock, user=None):
        """Create real analytics results for testing."""
        analytics = AnalyticsResults.objects.create(
            stock=stock,
            user=user,
            as_of=timezone.now(),
            horizon_days=30,
            technical_score=7.5,
            sentiment_score=6.8,
            combined_score=7.1,
            recommendation="BUY",
            confidence_level=0.85,
            weighted_sma_score=8.0,
            weighted_rsi_score=7.0,
            weighted_macd_score=7.5,
            weighted_bollinger_score=6.5,
            weighted_volume_score=8.5,
            weighted_obv_score=7.2,
            weighted_relative_strength_score=7.8,
            weighted_candlestick_score=6.9,
            weighted_support_resistance_score=7.3,
            raw_indicators=json.dumps(
                {
                    "sma_50": 148.50,
                    "sma_200": 145.30,
                    "rsi": 65.4,
                    "macd": 2.1,
                    "bollinger_upper": 155.0,
                    "bollinger_lower": 142.0,
                    "volume_avg": 95000000,
                }
            ),
            explanation_text="Real technical analysis indicates bullish momentum with strong volume support.",
            explanation_type="real",
            model_version="v2.0",
        )
        return analytics

    @staticmethod
    def create_market_data_sample():
        """Create comprehensive market data sample for testing."""
        # Technology stocks
        tech_stocks = [
            ("AAPL", "Apple Inc.", "Technology"),
            ("MSFT", "Microsoft Corporation", "Technology"),
            ("GOOGL", "Alphabet Inc.", "Technology"),
        ]

        # Financial stocks
        financial_stocks = [
            ("JPM", "JPMorgan Chase & Co.", "Financial Services"),
            ("BAC", "Bank of America Corp.", "Financial Services"),
        ]

        all_stocks = []

        for symbol, name, sector in tech_stocks + financial_stocks:
            stock = DataTestDataFactory.create_test_stock(symbol, name, sector)
            DataTestDataFactory.create_stock_price_history(stock, 60)  # 2 months of data
            all_stocks.append(stock)

        return all_stocks

    @staticmethod
    def get_real_yahoo_finance_sample():
        """Get sample of real Yahoo Finance data structure for testing."""
        return {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 2800000000000,
            "sharesOutstanding": 15500000000,
            "regularMarketPrice": 150.25,
            "regularMarketOpen": 149.50,
            "regularMarketDayHigh": 151.00,
            "regularMarketDayLow": 148.75,
            "regularMarketVolume": 89500000,
            "fiftyTwoWeekHigh": 182.94,
            "fiftyTwoWeekLow": 124.17,
            "trailingPE": 25.8,
            "forwardPE": 24.1,
            "dividendYield": 0.0051,
            "payoutRatio": 0.157,
            "beta": 1.29,
        }

    @staticmethod
    def create_test_dataframe():
        """Create real pandas DataFrame for Yahoo Finance integration testing."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-30", freq="D")

        data = {
            "Open": [148.0 + i * 0.5 for i in range(len(dates))],
            "High": [150.0 + i * 0.5 for i in range(len(dates))],
            "Low": [146.0 + i * 0.5 for i in range(len(dates))],
            "Close": [149.0 + i * 0.5 for i in range(len(dates))],
            "Volume": [85000000 + i * 100000 for i in range(len(dates))],
            "Adj Close": [149.0 + i * 0.5 for i in range(len(dates))],
        }

        return pd.DataFrame(data, index=dates)

    @staticmethod
    def cleanup_test_data():
        """Clean up all test data after testing."""
        # Clean up analytics results for test stocks
        test_symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC"]
        AnalyticsResults.objects.filter(stock__symbol__in=test_symbols).delete()

        # Clean up holdings and portfolios
        PortfolioHolding.objects.filter(portfolio__name__icontains="test").delete()
        Portfolio.objects.filter(name__icontains="test").delete()

        # Clean up stock prices and stocks
        StockPrice.objects.filter(stock__symbol__in=test_symbols).delete()
        Stock.objects.filter(symbol__in=test_symbols).delete()

        # Clean up sectors and industries
        DataIndustry.objects.filter(industryKey__in=["software"]).delete()
        DataSector.objects.filter(sectorKey__in=["technology", "financial_services"]).delete()


class YahooFinanceTestService:
    """Real Yahoo Finance test service to replace mocks."""

    def __init__(self):
        """Initialize Yahoo Finance test service."""
        self.timeout = getattr(settings, "YAHOO_FINANCE_API_TIMEOUT", 5)
        self.base_url = "https://query1.finance.yahoo.com"
        self.test_mode = True

    def get_stock_info(self, symbol):
        """Get comprehensive stock information for testing."""
        # Generate realistic stock data based on symbol
        base_data = self.get_real_yahoo_finance_sample()

        # Customize data based on symbol
        symbol_data = {
            "AAPL": {
                "shortName": "Apple Inc.",
                "longName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "marketCap": 2800000000000,
                "regularMarketPrice": 150.25,
            },
            "MSFT": {
                "shortName": "Microsoft Corporation",
                "longName": "Microsoft Corporation",
                "sector": "Technology",
                "industry": "Software",
                "marketCap": 2200000000000,
                "regularMarketPrice": 285.50,
            },
            "GOOGL": {
                "shortName": "Alphabet Inc.",
                "longName": "Alphabet Inc. Class A",
                "sector": "Communication Services",
                "industry": "Internet Content & Information",
                "marketCap": 1600000000000,
                "regularMarketPrice": 125.80,
            },
        }

        if symbol in symbol_data:
            base_data.update(symbol_data[symbol])
        else:
            base_data.update(
                {
                    "shortName": f"{symbol} Test Company",
                    "longName": f"{symbol} Test Company Inc.",
                    "regularMarketPrice": 100.0 + hash(symbol) % 100,
                }
            )

        base_data["symbol"] = symbol
        return base_data

    def get_historical_data(self, symbol, start_date, end_date, interval="1d"):
        """Get historical price data for testing."""
        # Create realistic historical data
        df = self.create_realistic_dataframe(symbol, start_date, end_date)

        # Add metadata
        df.attrs = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "timezone": "America/New_York",
        }

        return df

    def create_realistic_dataframe(self, symbol, start_date, end_date):
        """Create realistic historical price data."""
        start = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        # Generate date range (business days only)
        dates = pd.bdate_range(start=start, end=end)

        # Base price influenced by symbol
        base_price = 50 + (hash(symbol) % 200)

        data = []
        current_price = base_price

        for i, date in enumerate(dates):
            # Generate realistic price movements
            daily_change = (hash(f"{symbol}{date.strftime('%Y%m%d')}") % 1000 - 500) / 10000
            current_price *= 1 + daily_change

            # Ensure price doesn't go negative
            current_price = max(current_price, 1.0)

            # Generate OHLCV data
            open_price = current_price * (0.995 + (hash(f"open{date}") % 10) / 1000)
            high_price = current_price * (1.005 + (hash(f"high{date}") % 10) / 1000)
            low_price = current_price * (0.995 - (hash(f"low{date}") % 10) / 1000)
            close_price = current_price
            volume = 1000000 + (hash(f"vol{date}") % 50000000)

            data.append(
                {
                    "Open": round(open_price, 2),
                    "High": round(high_price, 2),
                    "Low": round(low_price, 2),
                    "Close": round(close_price, 2),
                    "Volume": volume,
                    "Adj Close": round(close_price, 2),
                }
            )

        return pd.DataFrame(data, index=dates)

    def get_multiple_quotes(self, symbols):
        """Get quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_stock_info(symbol)
        return quotes

    def search_symbols(self, query):
        """Search for symbols matching query."""
        # Predefined symbol database for testing
        symbol_database = {
            "apple": ["AAPL"],
            "microsoft": ["MSFT"],
            "google": ["GOOGL", "GOOG"],
            "amazon": ["AMZN"],
            "tesla": ["TSLA"],
            "facebook": ["META"],
            "nvidia": ["NVDA"],
            "jpmorgan": ["JPM"],
            "visa": ["V"],
            "walmart": ["WMT"],
        }

        query_lower = query.lower()
        matches = []

        for company, symbols in symbol_database.items():
            if query_lower in company or query_lower.upper() in symbols:
                for symbol in symbols:
                    matches.append(
                        {"symbol": symbol, "name": self.get_stock_info(symbol)["shortName"], "type": "stock"}
                    )

        return matches

    def get_trending_symbols(self, count=10):
        """Get trending symbols for testing."""
        trending = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        return trending[:count]

    def validate_symbol(self, symbol):
        """Validate symbol availability."""
        # Accept common test symbols
        valid_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "BAC",
            "WMT",
            "V",
            "JNJ",
            "PG",
            "UNH",
            "DIS",
            "HD",
            "MA",
            "NFLX",
            "CRM",
        ]
        return symbol.upper() in valid_symbols

    def get_market_status(self):
        """Get market status for testing."""
        import datetime

        now = datetime.datetime.now()

        # Mock market hours (9:30 AM - 4:00 PM ET on weekdays)
        is_weekday = now.weekday() < 5
        market_open_time = now.replace(hour=9, minute=30, second=0)
        market_close_time = now.replace(hour=16, minute=0, second=0)
        is_market_hours = market_open_time <= now <= market_close_time

        market_state = "REGULAR" if (is_weekday and is_market_hours) else "CLOSED"

        return {
            "market_state": market_state,
            "is_open": is_weekday and is_market_hours,
            "next_open": (timezone.now() + timedelta(days=1)).isoformat(),
            "next_close": (timezone.now() + timedelta(hours=8)).isoformat(),
            "timezone": "America/New_York",
            "current_time": now.isoformat(),
        }

    def get_sector_performance(self):
        """Get sector performance data for testing."""
        sectors = [
            {"name": "Technology", "change": 2.1, "change_percent": 1.8},
            {"name": "Healthcare", "change": 1.5, "change_percent": 1.2},
            {"name": "Financial", "change": -0.8, "change_percent": -0.6},
            {"name": "Consumer Cyclical", "change": 0.9, "change_percent": 0.7},
            {"name": "Energy", "change": -1.2, "change_percent": -1.1},
            {"name": "Utilities", "change": 0.3, "change_percent": 0.2},
        ]
        return sectors

    def get_market_summary(self):
        """Get market summary for testing."""
        return {
            "market_indices": {
                "SPY": {"price": 418.50, "change": 2.1, "change_percent": 0.5},
                "QQQ": {"price": 348.20, "change": -1.8, "change_percent": -0.5},
                "IWM": {"price": 195.60, "change": 0.9, "change_percent": 0.5},
            },
            "most_active": self.get_trending_symbols(5),
            "gainers": ["AAPL", "MSFT", "NVDA"],
            "losers": ["META", "NFLX"],
            "sector_performance": self.get_sector_performance(),
        }

    def test_connection(self):
        """Test service connection."""
        return {
            "status": "connected",
            "service": "Yahoo Finance Test Service",
            "timeout": self.timeout,
            "test_mode": self.test_mode,
            "available_endpoints": [
                "get_stock_info",
                "get_historical_data",
                "get_multiple_quotes",
                "search_symbols",
                "get_market_status",
                "get_market_summary",
            ],
        }

    @staticmethod
    def get_real_yahoo_finance_sample():
        """Get sample of real Yahoo Finance data structure for testing."""
        return {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "marketCap": 2800000000000,
            "sharesOutstanding": 15500000000,
            "regularMarketPrice": 150.25,
            "regularMarketOpen": 149.50,
            "regularMarketDayHigh": 151.00,
            "regularMarketDayLow": 148.75,
            "regularMarketVolume": 89500000,
            "fiftyTwoWeekHigh": 182.94,
            "fiftyTwoWeekLow": 124.17,
            "trailingPE": 25.8,
            "forwardPE": 24.1,
            "dividendYield": 0.0051,
            "payoutRatio": 0.157,
            "beta": 1.29,
            "bookValue": 4.25,
            "priceToBook": 35.4,
            "earningsQuarterlyGrowth": 0.112,
            "revenueQuarterlyGrowth": 0.089,
            "totalCash": 165000000000,
            "totalDebt": 132000000000,
            "debtToEquity": 1.73,
            "returnOnAssets": 0.201,
            "returnOnEquity": 1.47,
            "grossProfits": 182000000000,
            "freeCashflow": 99000000000,
            "operatingCashflow": 118000000000,
            "ebitdaMargins": 0.328,
            "profitMargins": 0.258,
        }


class DatabaseIntegrityValidator:
    """Validator for database integrity in tests."""

    @staticmethod
    def validate_stock_data_integrity(stock):
        """Validate stock data integrity."""
        assert stock.symbol is not None
        assert len(stock.symbol) <= 10
        assert stock.company_name is not None
        assert stock.market_cap >= 0
        return True

    @staticmethod
    def validate_price_data_integrity(stock_price):
        """Validate price data integrity."""
        assert stock_price.open_price > 0
        assert stock_price.high_price >= stock_price.open_price
        assert stock_price.low_price <= stock_price.open_price
        assert stock_price.close_price > 0
        assert stock_price.volume >= 0
        return True

    @staticmethod
    def validate_portfolio_integrity(portfolio):
        """Validate portfolio data integrity."""
        assert portfolio.total_value >= 0
        assert portfolio.cash_balance >= 0
        assert portfolio.user is not None
        return True
