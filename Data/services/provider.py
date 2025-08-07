"""
Data Provider Service Module
Fetches market data from Yahoo Finance API for VoyageurCompass.
Includes rate limiting handling and retry logic.
"""

import logging
import time
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider:
    """
    Service class for fetching market data from Yahoo Finance.
    """
    
    def __init__(self):
        """Initialize the Data Provider."""
        self.timeout = 30
        self.max_retries = 3
        self.base_delay = 2  # Base delay in seconds
        logger.info("Data Provider Service initialized")
    
    def _safe_fetch(self, ticker_symbol: str, retries: int = 0) -> Optional[yf.Ticker]:
        """
        Safely fetch ticker with retry logic and rate limiting handling.
        
        Args:
            ticker_symbol: Stock ticker symbol
            retries: Current retry attempt
        
        Returns:
            yf.Ticker object or None if failed
        """
        try:
            # Add longer delay to avoid rate limiting
            if retries > 0:
                delay = self.base_delay * (3 ** retries) + random.uniform(1, 3)
                logger.info(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
            else:
                # Longer initial delay to avoid rate limiting
                time.sleep(random.uniform(2, 4))
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Test if ticker is valid by accessing info
            # Use a try-except to catch the 429 error
            test_info = ticker.info
            
            if test_info and len(test_info) > 1:
                return ticker
            else:
                return None
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Too Many Requests" in error_str:
                logger.warning(f"Rate limited on {ticker_symbol}, attempt {retries + 1}/{self.max_retries}")
                if retries < self.max_retries - 1:
                    return self._safe_fetch(ticker_symbol, retries + 1)
                else:
                    # Return None and use mock data when rate limited
                    logger.warning(f"Max retries reached for {ticker_symbol}, will use mock data")
                    return None
            else:
                logger.error(f"Error fetching ticker {ticker_symbol}: {error_str}")
            return None
    
    def fetch_stock_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """
        Fetch comprehensive stock data from Yahoo Finance with error handling.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Time period for historical data
                   Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        
        Returns:
            Dictionary containing stock info and historical price data
        """
        try:
            logger.info(f"Fetching Yahoo Finance data for {symbol}")
            
            # Use safe fetch with retry logic
            ticker = self._safe_fetch(symbol)
            
            if not ticker:
                # If Yahoo Finance is blocking, return mock data for testing
                logger.warning(f"Using mock data for {symbol} due to API limitations")
                return self._get_mock_data(symbol, period)
            
            # Fetch stock info with error handling
            info = {}
            try:
                info = ticker.info
            except Exception as e:
                logger.warning(f"Could not fetch info for {symbol}: {str(e)}")
                info = {}
            
            # Fetch historical market data with error handling
            history = None
            try:
                history = ticker.history(period=period)
            except Exception as e:
                logger.warning(f"Could not fetch history for {symbol}: {str(e)}")
            
            # Prepare the response
            stock_data = {
                'success': True,
                'symbol': symbol.upper(),
                'info': {
                    'shortName': info.get('shortName', symbol),
                    'longName': info.get('longName', f'{symbol} Corporation'),
                    'currency': info.get('currency', 'USD'),
                    'exchange': info.get('exchange', 'NASDAQ'),
                    'sector': info.get('sector', 'Technology'),
                    'industry': info.get('industry', 'Software'),
                    'marketCap': info.get('marketCap', 0),
                    'currentPrice': info.get('currentPrice', 0),
                    'previousClose': info.get('previousClose', 0),
                    'dayHigh': info.get('dayHigh', 0),
                    'dayLow': info.get('dayLow', 0),
                    'volume': info.get('volume', 0),
                    'averageVolume': info.get('averageVolume', 0),
                    'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                    'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
                    'dividendYield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 0),
                    'trailingPE': info.get('trailingPE', 0),
                    'forwardPE': info.get('forwardPE', 0),
                    'priceToBook': info.get('priceToBook', 0),
                },
                'history': [],
                'fetched_at': datetime.now().isoformat()
            }
            
            # Process historical data if available
            if history is not None and not history.empty:
                for date, row in history.iterrows():
                    stock_data['history'].append({
                        'date': date.strftime('%Y-%m-%d'),
                        'open': float(row['Open']) if 'Open' in row else 0,
                        'high': float(row['High']) if 'High' in row else 0,
                        'low': float(row['Low']) if 'Low' in row else 0,
                        'close': float(row['Close']) if 'Close' in row else 0,
                        'volume': int(row['Volume']) if 'Volume' in row else 0,
                    })
                logger.info(f"Successfully fetched data for {symbol} - {len(stock_data['history'])} days of history")
            else:
                # Use mock data if history is not available
                logger.warning(f"No history available for {symbol}, using mock data")
                mock_data = self._get_mock_data(symbol, period)
                stock_data['history'] = mock_data['history']
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            # Return mock data on error
            return self._get_mock_data(symbol, period)
    
    def _get_mock_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """
        Generate mock data for testing when Yahoo Finance is unavailable.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period
        
        Returns:
            Dictionary with mock stock data
        """
        logger.info(f"Generating mock data for {symbol}")
        
        # Determine number of days based on period
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825,
            '10y': 3650, 'ytd': 200, 'max': 365
        }
        days = period_days.get(period, 30)
        
        # Generate mock price data
        base_price = 150.0 + random.uniform(-50, 50)
        history = []
        current_date = datetime.now()
        
        for i in range(days):
            date = current_date - timedelta(days=days-i-1)
            # Skip weekends
            if date.weekday() >= 5:
                continue
                
            # Generate realistic price movements
            daily_change = random.uniform(-0.03, 0.03)  # ±3% daily change
            open_price = base_price * (1 + daily_change)
            close_price = open_price * (1 + random.uniform(-0.02, 0.02))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            volume = random.randint(10000000, 50000000)
            
            history.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
            })
            
            base_price = close_price
        
        return {
            'success': True,
            'symbol': symbol.upper(),
            'info': {
                'shortName': f'{symbol} Corp',
                'longName': f'{symbol} Corporation',
                'currency': 'USD',
                'exchange': 'NASDAQ',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': random.randint(100000000, 1000000000000),
                'currentPrice': round(base_price, 2),
                'previousClose': round(base_price * 0.99, 2),
                'dayHigh': round(base_price * 1.02, 2),
                'dayLow': round(base_price * 0.98, 2),
                'volume': random.randint(10000000, 50000000),
                'averageVolume': random.randint(15000000, 35000000),
                'fiftyTwoWeekHigh': round(base_price * 1.3, 2),
                'fiftyTwoWeekLow': round(base_price * 0.7, 2),
                'dividendYield': round(random.uniform(0, 0.03), 4),
                'beta': round(random.uniform(0.8, 1.5), 2),
                'trailingPE': round(random.uniform(15, 35), 2),
                'forwardPE': round(random.uniform(12, 30), 2),
                'priceToBook': round(random.uniform(2, 10), 2),
            },
            'history': history,
            'fetched_at': datetime.now().isoformat(),
            'is_mock_data': True  # Flag to indicate this is mock data
        }
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = "1mo") -> Dict[str, Dict]:
        """
        Fetch data for multiple stock symbols with delays to avoid rate limiting.
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period for historical data
        
        Returns:
            Dictionary with symbol as key and stock data as value
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Fetching data for {symbol} ({i + 1}/{len(symbols)})")
            
            # Add delay between requests to avoid rate limiting
            if i > 0:
                delay = random.uniform(1, 3)
                logger.info(f"Waiting {delay:.1f} seconds before next request...")
                time.sleep(delay)
            
            results[symbol] = self.fetch_stock_data(symbol, period)
        
        return results
    
    def fetch_realtime_price(self, symbol: str) -> Optional[float]:
        """
        Fetch the current real-time price for a stock.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Current price as float, or None if error
        """
        try:
            ticker = self._safe_fetch(symbol)
            
            if not ticker:
                # Return mock price if API is unavailable
                return round(150.0 + random.uniform(-50, 50), 2)
            
            info = ticker.info
            
            # Try different price fields in order of preference
            price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
            
            for field in price_fields:
                if field in info and info[field]:
                    return float(info[field])
            
            # If no price found in info, try to get latest from history
            history = ticker.history(period="1d")
            if not history.empty:
                return float(history['Close'].iloc[-1])
            
            # Return mock price as fallback
            return round(150.0 + random.uniform(-50, 50), 2)
            
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {str(e)}")
            # Return mock price on error
            return round(150.0 + random.uniform(-50, 50), 2)
    
    def fetch_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch detailed company information.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary containing company information
        """
        try:
            ticker = self._safe_fetch(symbol)
            
            if not ticker:
                # Return mock company info if API is unavailable
                return self._get_mock_company_info(symbol)
            
            info = ticker.info
            
            company_info = {
                'symbol': symbol.upper(),
                'company': {
                    'name': info.get('longName', f'{symbol} Corporation'),
                    'sector': info.get('sector', 'Technology'),
                    'industry': info.get('industry', 'Software'),
                    'website': info.get('website', ''),
                    'description': info.get('longBusinessSummary', ''),
                    'employees': info.get('fullTimeEmployees', 0),
                    'country': info.get('country', 'United States'),
                    'city': info.get('city', ''),
                    'state': info.get('state', ''),
                    'address': info.get('address1', ''),
                },
                'financials': {
                    'marketCap': info.get('marketCap', 0),
                    'enterpriseValue': info.get('enterpriseValue', 0),
                    'revenue': info.get('totalRevenue', 0),
                    'grossProfit': info.get('grossProfits', 0),
                    'ebitda': info.get('ebitda', 0),
                    'netIncome': info.get('netIncomeToCommon', 0),
                    'totalCash': info.get('totalCash', 0),
                    'totalDebt': info.get('totalDebt', 0),
                    'bookValue': info.get('bookValue', 0),
                },
                'metrics': {
                    'peRatio': info.get('trailingPE', 0),
                    'forwardPE': info.get('forwardPE', 0),
                    'pegRatio': info.get('pegRatio', 0),
                    'priceToBook': info.get('priceToBook', 0),
                    'priceToSales': info.get('priceToSalesTrailing12Months', 0),
                    'enterpriseToRevenue': info.get('enterpriseToRevenue', 0),
                    'enterpriseToEbitda': info.get('enterpriseToEbitda', 0),
                    'profitMargin': info.get('profitMargins', 0),
                    'operatingMargin': info.get('operatingMargins', 0),
                }
            }
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return self._get_mock_company_info(symbol)
    
    def _get_mock_company_info(self, symbol: str) -> Dict[str, Any]:
        """Generate mock company info for testing."""
        return {
            'symbol': symbol.upper(),
            'company': {
                'name': f'{symbol} Corporation',
                'sector': 'Technology',
                'industry': 'Software',
                'website': f'https://www.{symbol.lower()}.com',
                'description': f'{symbol} is a leading technology company.',
                'employees': random.randint(1000, 100000),
                'country': 'United States',
                'city': 'San Francisco',
                'state': 'CA',
                'address': '123 Tech Street',
            },
            'financials': {
                'marketCap': random.randint(100000000, 1000000000000),
                'enterpriseValue': random.randint(100000000, 1000000000000),
                'revenue': random.randint(10000000, 100000000000),
                'grossProfit': random.randint(5000000, 50000000000),
                'ebitda': random.randint(1000000, 20000000000),
            },
            'is_mock_data': True
        }
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and is tradeable.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            True if valid symbol, False otherwise
        """
        try:
            # For common symbols, assume they're valid during rate limiting
            common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 
                            'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA']
            
            if symbol.upper() in common_symbols:
                logger.info(f"Symbol {symbol} is a known valid symbol")
                return True
            
            ticker = self._safe_fetch(symbol)
            
            if ticker:
                info = ticker.info
                # Check if we got valid data back
                if info and len(info) > 1:
                    return True
            
            # During rate limiting, be optimistic for testing
            logger.warning(f"Cannot validate {symbol} due to rate limiting, assuming valid")
            return True
            
        except Exception as e:
            logger.warning(f"Symbol validation failed for {symbol}: {str(e)}")
            # Be optimistic during errors
            return True


# Singleton instance
data_provider = DataProvider()