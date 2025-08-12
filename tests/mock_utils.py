"""
Centralized mock utilities for consistent test mocking across the project.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch


class YahooFinanceMocks:
    """Centralized Yahoo Finance service mocking utilities."""
    
    @staticmethod
    def get_stock_data_success(symbol="AAPL", prices=None, volumes=None):
        """Standard successful stock data response."""
        if prices is None:
            prices = [155.00, 154.50, 156.20]
        if volumes is None:
            volumes = [50000000, 48000000, 52000000]
            
        return {
            "symbol": symbol,
            "prices": prices,
            "volumes": volumes,
            "dates": [
                date.today().isoformat(),
                (date.today() - timedelta(days=1)).isoformat(),
                (date.today() - timedelta(days=2)).isoformat(),
            ],
            "error": None
        }
    
    @staticmethod
    def get_stock_data_failure(symbol="AAPL", error="Network error"):
        """Standard failed stock data response."""
        return {
            "symbol": symbol,
            "prices": [],
            "volumes": [],
            "dates": [],
            "error": error
        }
    
    @staticmethod
    def get_market_status_success():
        """Standard successful market status response."""
        return {
            "is_open": True,
            "current_time": datetime.now().isoformat(),
            "market_hours": {"open": "09:30 EST", "close": "16:00 EST"},
            "indicators": {
                "spy": 450.25,
                "vix": 18.5
            },
            "next_open": datetime.now().isoformat(),
        }
    
    @staticmethod
    def get_historical_data_success(symbol="AAPL", days=30):
        """Standard successful historical data response."""
        data = []
        for i in range(days):
            data.append({
                "date": (date.today() - timedelta(days=i)).isoformat(),
                "open": 150.00 + i * 0.5,
                "high": 155.00 + i * 0.5,
                "low": 149.00 + i * 0.5,
                "close": 152.00 + i * 0.5,
                "volume": 50000000 + (i * 1000000)
            })
        
        return {
            "symbol": symbol,
            "data": data
        }
    
    @staticmethod
    def validate_symbol_success(symbol="AAPL"):
        """Standard symbol validation success."""
        return True
    
    @staticmethod
    def validate_symbol_failure(symbol="INVALID"):
        """Standard symbol validation failure."""
        return False
    
    @staticmethod
    def get_multiple_stocks_success(symbols=None):
        """Standard multiple stocks data response."""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL"]
        
        result = {}
        for symbol in symbols:
            result[symbol] = YahooFinanceMocks.get_stock_data_success(symbol)
        
        return result


class AuthenticationMocks:
    """Centralized authentication mocking utilities."""
    
    @staticmethod
    def get_user_data():
        """Standard test user data."""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123",
            "first_name": "Test",
            "last_name": "User",
        }
    
    @staticmethod
    def get_login_success_response():
        """Standard successful login response."""
        return {
            "user": {
                "id": 1,
                "username": "testuser",
                "email": "test@example.com",
                "first_name": "Test",
                "last_name": "User"
            },
            "access": "fake-access-token",
            "refresh": "fake-refresh-token"
        }
    
    @staticmethod
    def get_registration_success_response():
        """Standard successful registration response."""
        return {
            "user": {
                "id": 1,
                "username": "newuser",
                "email": "newuser@example.com",
                "first_name": "New",
                "last_name": "User",
                "date_joined": datetime.now().isoformat()
            },
            "tokens": {
                "access": "fake-access-token",
                "refresh": "fake-refresh-token"
            }
        }


class AnalyticsMocks:
    """Centralized analytics mocking utilities."""
    
    @staticmethod
    def get_stock_analysis_success(symbol="AAPL"):
        """Standard successful stock analysis response."""
        return {
            "success": True,
            "symbol": symbol,
            "analysis_date": datetime.now().isoformat(),
            "horizon": "blend",
            "composite_score": 7,
            "composite_raw": 0.65,
            "technical_indicators": {
                "sma_50": 150.25,
                "sma_200": 145.80,
                "rsi_14": 58.3,
                "macd": {
                    "line": 1.25,
                    "signal": 0.95,
                    "histogram": 0.30
                }
            },
            "indicators": {
                "sma50vs200": {"raw": {"crossover": True}, "score": 0.8},
                "rsi14": {"raw": {"rsi": 58.3}, "score": 0.6},
            },
            "score_0_10": 7,
            "analytics_result_id": 1
        }
    
    @staticmethod
    def get_portfolio_analysis_success(portfolio_id=1):
        """Standard successful portfolio analysis response."""
        return {
            "portfolio_id": portfolio_id,
            "diversification": {
                "score": 0.75,
                "by_sector": {"Technology": 0.6, "Healthcare": 0.4},
                "by_industry": {"Software": 0.4, "Hardware": 0.2, "Pharma": 0.4},
                "concentration_risk": "moderate"
            },
            "technical_strength": 0.65,
            "risk_score": 6.5,
            "risk_level": "Moderate-High"
        }


# Patch decorators for common mocking patterns
def mock_yahoo_finance_success(func):
    """Decorator to mock Yahoo Finance service with success responses."""
    return patch("Data.services.yahoo_finance.yahoo_finance_service.get_stock_data", 
                 return_value=YahooFinanceMocks.get_stock_data_success())(func)


def mock_yahoo_finance_market_status(func):
    """Decorator to mock Yahoo Finance market status."""
    return patch("Data.services.yahoo_finance.yahoo_finance_service.get_market_status",
                 return_value=YahooFinanceMocks.get_market_status_success())(func)


def mock_yahoo_finance_validation(func):
    """Decorator to mock Yahoo Finance symbol validation."""
    return patch("Data.services.yahoo_finance.yahoo_finance_service.validate_symbol",
                 return_value=True)(func)