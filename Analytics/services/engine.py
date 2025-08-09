"""
Analytics Engine Module
Handles all financial calculations and analysis for VoyageurCompass.
Based on the Maple Trade prototype logic with additional technical indicators.
"""

import logging
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from decimal import Decimal
from django.db.models import Q

# Import models from Data app
from Data.models import Stock, StockPrice, Portfolio, PortfolioHolding

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Core analytics engine for financial calculations and signal generation.
    Implements the Maple Trade prototype logic plus additional technical analysis.
    """
    
    # Sector-specific volatility thresholds from prototype
    VOLATILITY_THRESHOLDS = {
        'Technology': 0.50,
        'Financials': 0.38,
        'Healthcare': 0.38,
        'Energy': 0.46,
        'Consumer Discretionary': 0.46,
        'Industrials': 0.42,
        'Utilities': 0.32,
        'Materials': 0.42,
        'Real Estate': 0.38,
        'Communication Services': 0.42,
        'Consumer Staples': 0.38,
        'Default': 0.42
    }
    
    # Sector ETF mapping from prototype
    SECTOR_ETF_MAP = {
        'Technology': 'XLK',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Energy': 'XLE',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Utilities': 'XLU',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC',
        'Consumer Staples': 'XLP',
    }
    
    def __init__(self):
        """Initialize the analytics engine."""
        logger.info("Analytics Engine initialized with Maple Trade logic")
        
    # =====================================================================
    # Input Validation (camelCase)
    # =====================================================================
    
    def validateSymbol(self, symbol: str) -> str:
        """Validate and sanitize stock symbol input"""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        symbol = symbol.strip().upper()
        # Allow only alphanumeric and common stock suffixes
        if not re.match(r'^[A-Z0-9\.\-]{1,10}$', symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        return symbol
    
    def validatePeriod(self, period: int) -> int:
        """Validate period input for calculations"""
        if not isinstance(period, (int, float)) or period <= 0 or period > 3650:
            raise ValueError(f"Period must be between 1 and 3650 days: {period}")
        return int(period)
    
    def validateNumericInput(self, value: Any, minVal: float = None, maxVal: float = None) -> bool:
        """Validate numeric input parameters"""
        try:
            numValue = float(value)
            if minVal is not None and numValue < minVal:
                return False
            if maxVal is not None and numValue > maxVal:
                return False
            return True
        except (TypeError, ValueError):
            return False
    
    def sanitizePriceList(self, prices: List[float]) -> List[float]:
        """Validate and sanitize price list"""
        if not isinstance(prices, list) or len(prices) == 0:
            raise ValueError("Prices must be a non-empty list")
        
        sanitized = []
        for price in prices:
            if not self.validateNumericInput(price, minVal=0, maxVal=1000000):
                raise ValueError(f"Invalid price value: {price}")
            sanitized.append(float(price))
        
        return sanitized
    
    # =====================================================================
    # Basic Calculations (from prototype)
    # =====================================================================
    
    # camelCase wrapper methods
    def calculateReturns(self, prices: List[float], period: str = 'daily') -> List[float]:
        """camelCase wrapper for calculate_returns"""
        return self.calculate_returns(prices, period)
    
    def calculate_returns(self, prices: List[float], period: str = 'daily') -> List[float]:
        """
        Calculate returns from a price series.
        
        Args:
            prices: List of prices (newest first)
            period: Time period for returns ('daily', 'weekly', 'monthly')
        
        Returns:
            List of calculated returns as percentages
        """
        # Input validation
        prices = self.sanitizePriceList(prices)
        valid_periods = ['daily', 'weekly', 'monthly']
        if period not in valid_periods:
            raise ValueError(f"Period must be one of {valid_periods}")
        
        if len(prices) < 2:
            return []
        
        # Reverse to get chronological order
        prices = list(reversed(prices))
        returns = []
        
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                return_value = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                returns.append(return_value)
        
        return returns
    
    def calculatePeriodReturn(self, prices: List[float]) -> Optional[float]:
        """camelCase wrapper for calculate_period_return"""
        return self.calculate_period_return(prices)
        
    def calculate_period_return(self, prices: List[float]) -> Optional[float]:
        """
        Calculate total return over the entire period (from prototype logic).
        
        Args:
            prices: List of prices (newest first)
        
        Returns:
            Total return as percentage or None
        """
        # Input validation
        prices = self.sanitizePriceList(prices)
        
        if len(prices) < 2:
            return None
        
        # Get first and last prices
        start_price = prices[-1]  # Oldest price
        end_price = prices[0]      # Newest price
        
        if start_price == 0:
            return None
        
        return ((end_price - start_price) / start_price) * 100
    
    def calculate_annualized_volatility(self, prices: List[float]) -> Optional[float]:
        """
        Calculate annualized volatility (from prototype).
        
        Args:
            prices: List of prices (newest first)
        
        Returns:
            Annualized volatility (standard deviation * sqrt(252))
        """
        if len(prices) < 2:
            return None
        
        # Calculate daily returns
        returns = self.calculate_returns(prices)
        
        if not returns:
            return None
        
        # Calculate standard deviation and annualize
        returns_array = np.array(returns) / 100  # Convert from percentage
        std_dev = np.std(returns_array)
        annualized_vol = std_dev * np.sqrt(252)  # 252 trading days
        
        return annualized_vol
    
    def calculate_moving_average(self, prices: List[float], window: int) -> List[float]:
        """
        Calculate simple moving average.
        
        Args:
            prices: List of prices (newest first)
            window: Window size for moving average
        
        Returns:
            List of moving average values
        """
        if len(prices) < window:
            return []
        
        # Reverse to get chronological order
        prices_chrono = list(reversed(prices))
        ma_values = []
        
        for i in range(window - 1, len(prices_chrono)):
            window_slice = prices_chrono[i - window + 1:i + 1]
            ma_values.append(sum(window_slice) / window)
        
        # Reverse back to newest first
        return list(reversed(ma_values))
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of return values (as percentages)
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0
        
        # Convert to decimal
        returns_decimal = [r / 100 for r in returns]
        
        mean_return = np.mean(returns_decimal)
        std_return = np.std(returns_decimal)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_std
        return sharpe
    
    # =====================================================================
    # Technical Indicators
    # =====================================================================
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: List of prices (newest first)
            period: Period for RSI calculation (default 14)
        
        Returns:
            RSI value (0-100) or None
        """
        if len(prices) < period + 1:
            return None
        
        # Get price changes
        prices_chrono = list(reversed(prices))[:period + 1]
        changes = []
        
        for i in range(1, len(prices_chrono)):
            changes.append(prices_chrono[i] - prices_chrono[i-1])
        
        # Separate gains and losses
        gains = [c if c > 0 else 0 for c in changes]
        losses = [-c if c < 0 else 0 for c in changes]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0  # Maximum RSI
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: List[float], 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Dict[str, Optional[float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of prices (newest first)
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
        
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if len(prices) < slow_period:
            return {'macd': None, 'signal': None, 'histogram': None}
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        if ema_fast is None or ema_slow is None:
            return {'macd': None, 'signal': None, 'histogram': None}
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need historical MACD values
        # Simplified: return current MACD only
        return {
            'macd': macd_line,
            'signal': None,  # Would need historical MACD values
            'histogram': None
        }
    
    def calculate_bollinger_bands(self, prices: List[float], 
                                 period: int = 20, 
                                 num_std: float = 2) -> Dict[str, Optional[float]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: List of prices (newest first)
            period: Period for moving average (default 20)
            num_std: Number of standard deviations (default 2)
        
        Returns:
            Dictionary with upper band, middle band (SMA), and lower band
        """
        if len(prices) < period:
            return {'upper': None, 'middle': None, 'lower': None}
        
        # Calculate SMA
        ma_values = self.calculate_moving_average(prices, period)
        if not ma_values:
            return {'upper': None, 'middle': None, 'lower': None}
        
        middle_band = ma_values[0]  # Most recent MA
        
        # Calculate standard deviation
        recent_prices = list(reversed(prices))[:period]
        std_dev = np.std(recent_prices)
        
        upper_band = middle_band + (num_std * std_dev)
        lower_band = middle_band - (num_std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of prices (newest first)
            period: EMA period
        
        Returns:
            EMA value or None
        """
        if len(prices) < period:
            return None
        
        # Reverse to chronological order
        prices_chrono = list(reversed(prices))
        
        # Start with SMA
        sma = sum(prices_chrono[:period]) / period
        multiplier = 2 / (period + 1)
        
        # Calculate EMA
        ema = sma
        for price in prices_chrono[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    # =====================================================================
    # Signal Generation (from prototype)
    # =====================================================================
    
    def generate_signal(self, stock_return: Optional[float],
                       etf_return: Optional[float],
                       volatility: Optional[float],
                       target_price: Optional[float],
                       current_price: Optional[float],
                       sector: str = 'Default') -> Dict[str, any]:
        """
        Generate trading signal based on Maple Trade prototype logic.
        
        Args:
            stock_return: Stock's period return (%)
            etf_return: Sector ETF's period return (%)
            volatility: Annualized volatility
            target_price: Analyst target price
            current_price: Current stock price
            sector: Stock's sector for volatility threshold
        
        Returns:
            Dictionary with signal, reason, and criteria met
        """
        # Get volatility threshold for sector
        vol_threshold = self.VOLATILITY_THRESHOLDS.get(sector, self.VOLATILITY_THRESHOLDS['Default'])
        
        # Initialize criteria
        met = {
            'volatility_ok': False,
            'outperformed': False,
            'target_above': False
        }
        
        # Check criteria
        if volatility is not None:
            met['volatility_ok'] = volatility <= vol_threshold
        
        if stock_return is not None and etf_return is not None:
            met['outperformed'] = stock_return > etf_return
        
        if target_price is not None and current_price is not None:
            met['target_above'] = target_price > current_price
        
        # Decision logic from prototype
        positive_count = sum([met['outperformed'], met['target_above']])
        
        # Both outperformance and target are positive
        if met['outperformed'] and met['target_above']:
            signal = "BUY"
            reason = "Both outperformance and target are positive. Volatility does not prevent buy in this case."
        # Exactly one positive parameter
        elif positive_count == 1:
            if met['volatility_ok']:
                signal = "BUY"
                reason = "Exactly one positive parameter and volatility is low."
            else:
                signal = "HOLD"
                reason = "Exactly one positive parameter, but volatility is high—staying cautious."
        # Both are negative
        elif not met['outperformed'] and not met['target_above']:
            if met['volatility_ok']:
                signal = "HOLD"
                reason = "Both outperformance and target are negative, but volatility is low—no action."
            else:
                signal = "SELL"
                reason = "Both outperformance and target are negative, and volatility is high—risk-off."
        else:
            signal = "HOLD"
            reason = "Insufficient or mixed data—default to hold."
        
        return {
            'signal': signal,
            'reason': reason,
            'criteria_met': met,
            'volatility_threshold': vol_threshold
        }
    
    # =====================================================================
    # Database Integration
    # =====================================================================
    
    def getStockData(self, symbol: str, days: int = 180) -> Optional[Dict]:
        """camelCase wrapper for get_stock_data"""
        return self.get_stock_data(symbol, days)
    
    def get_stock_data(self, symbol: str, days: int = 180) -> Optional[Dict]:
        """
        Fetch stock data from database.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of history to fetch
        
        Returns:
            Dictionary with stock info and price history
        """
        try:
            # Input validation
            symbol = self.validateSymbol(symbol)
            days = self.validatePeriod(days)
            # Get stock record
            stock = Stock.objects.get(symbol=symbol.upper())
            
            # Get price history
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            prices = StockPrice.objects.filter(
                stock=stock,
                date__gte=start_date,
                date__lte=end_date
            ).order_by('-date')
            
            if not prices.exists():
                logger.warning(f"No price data found for {symbol}")
                return None
            
            # Extract price lists
            close_prices = [float(p.close) for p in prices]
            open_prices = [float(p.open) for p in prices]
            high_prices = [float(p.high) for p in prices]
            low_prices = [float(p.low) for p in prices]
            volumes = [p.volume for p in prices]
            dates = [p.date for p in prices]
            
            return {
                'stock': stock,
                'close_prices': close_prices,
                'open_prices': open_prices,
                'high_prices': high_prices,
                'low_prices': low_prices,
                'volumes': volumes,
                'dates': dates,
                'latest_price': close_prices[0] if close_prices else None,
                'sector': stock.sector or 'Default'
            }
            
        except Stock.DoesNotExist:
            logger.error(f"Stock {symbol} not found in database")
            return None
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return None
    
    def get_sector_etf_data(self, sector: str, days: int = 180) -> Optional[Dict]:
        """
        Fetch sector ETF data from database.
        
        Args:
            sector: Sector name
            days: Number of days of history
        
        Returns:
            Dictionary with ETF data or None
        """
        etf_symbol = self.SECTOR_ETF_MAP.get(sector)
        
        if not etf_symbol:
            logger.warning(f"No ETF mapping for sector {sector}")
            etf_symbol = 'SPY'  # Default to S&P 500
        
        return self.get_stock_data(etf_symbol, days)
    
    def get_analyst_target(self, stock: Stock) -> Optional[float]:
        """
        Get analyst target price for a stock.
        For now, returns a mock value. In production, this would
        fetch from a real data source.
        
        Args:
            stock: Stock model instance
        
        Returns:
            Target price or None
        """
        # In production, this would fetch from Yahoo Finance or another source
        # For now, return a mock value based on current price
        latest_price = stock.get_latest_price()
        if latest_price:
            # Mock: 15% above current price
            return float(latest_price.close) * 1.15
        return None
    
    # =====================================================================
    # Main Analysis Pipeline
    # =====================================================================
    
    def runFullAnalysis(self, symbol: str, analysisMonths: int = 6) -> Dict:
        """camelCase wrapper for run_full_analysis"""
        return self.run_full_analysis(symbol, analysisMonths)
    
    def run_full_analysis(self, symbol: str, analysis_months: int = 6) -> Dict:
        """
        Run complete analysis pipeline for a stock (main entry point).
        
        Args:
            symbol: Stock ticker symbol
            analysis_months: Number of months to analyze (default 6)
        
        Returns:
            Dictionary with all analysis results
        """
        try:
            # Input validation
            symbol = self.validateSymbol(symbol)
            if not self.validateNumericInput(analysis_months, minVal=1, maxVal=120):  # Max 10 years
                raise ValueError(f"Analysis months must be between 1 and 120: {analysis_months}")
            
            logger.info(f"Running full analysis for {symbol}")
            
            # Convert months to days
            days = int(analysis_months * 30.5)
            
            # Fetch stock data
            stock_data = self.get_stock_data(symbol, days)
            if not stock_data:
                return {
                    'success': False,
                    'error': f'No data found for {symbol}',
                    'symbol': symbol
                }
            
            stock = stock_data['stock']
            close_prices = stock_data['close_prices']
            
            # Fetch sector ETF data
            etf_data = self.get_sector_etf_data(stock_data['sector'], days)
            etf_prices = etf_data['close_prices'] if etf_data else None
            
            # Calculate basic metrics
            stock_return = self.calculate_period_return(close_prices)
            etf_return = self.calculate_period_return(etf_prices) if etf_prices else None
            volatility = self.calculate_annualized_volatility(close_prices)
            
            # Calculate technical indicators
            rsi = self.calculate_rsi(close_prices)
            macd = self.calculate_macd(close_prices)
            bollinger = self.calculate_bollinger_bands(close_prices)
            
            # Calculate moving averages
            ma_20 = self.calculate_moving_average(close_prices, 20)
            ma_50 = self.calculate_moving_average(close_prices, 50)
            ma_200 = self.calculate_moving_average(close_prices, 200)
            
            # Calculate returns for Sharpe ratio
            daily_returns = self.calculate_returns(close_prices)
            sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)
            
            # Get analyst target
            target_price = self.get_analyst_target(stock)
            current_price = stock_data['latest_price']
            
            # Generate trading signal
            signal_data = self.generate_signal(
                stock_return=stock_return,
                etf_return=etf_return,
                volatility=volatility,
                target_price=target_price,
                current_price=current_price,
                sector=stock_data['sector']
            )
            
            # Compile results
            results = {
                'success': True,
                'symbol': symbol,
                'company_name': stock.long_name or stock.short_name,
                'sector': stock_data['sector'],
                'analysis_date': datetime.now().isoformat(),
                'analysis_period_months': analysis_months,
                
                # Price data
                'current_price': current_price,
                'target_price': target_price,
                'price_to_target_ratio': (current_price / target_price) if target_price else None,
                
                # Returns and volatility
                'stock_return': stock_return,
                'etf_return': etf_return,
                'outperformance': (stock_return - etf_return) if (stock_return and etf_return) else None,
                'volatility': volatility,
                'volatility_threshold': signal_data['volatility_threshold'],
                'sharpe_ratio': sharpe_ratio,
                
                # Technical indicators
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bollinger,
                
                # Moving averages
                'ma_20': ma_20[0] if ma_20 else None,
                'ma_50': ma_50[0] if ma_50 else None,
                'ma_200': ma_200[0] if ma_200 else None,
                
                # Signal
                'signal': signal_data['signal'],
                'signal_reason': signal_data['reason'],
                'criteria_met': signal_data['criteria_met'],
                
                # Additional metrics
                'price_above_ma20': (current_price > ma_20[0]) if ma_20 else None,
                'price_above_ma50': (current_price > ma_50[0]) if ma_50 else None,
                'price_above_ma200': (current_price > ma_200[0]) if ma_200 else None,
                
                # RSI signals
                'rsi_oversold': (rsi < 30) if rsi else None,
                'rsi_overbought': (rsi > 70) if rsi else None,
                
                # Bollinger Band position
                'bb_position': self._get_bb_position(current_price, bollinger) if bollinger['middle'] else None
            }
            
            logger.info(f"Analysis complete for {symbol}: Signal = {results['signal']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in full analysis for {symbol}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    def _get_bb_position(self, price: float, bollinger: Dict) -> str:
        """
        Determine price position relative to Bollinger Bands.
        
        Args:
            price: Current price
            bollinger: Bollinger band values
        
        Returns:
            Position description
        """
        if not bollinger['middle']:
            return 'unknown'
        
        if price > bollinger['upper']:
            return 'above_upper'
        elif price < bollinger['lower']:
            return 'below_lower'
        else:
            return 'within_bands'
    
    def analyzePortfolio(self, portfolioId: int) -> Dict:
        """camelCase wrapper for analyze_portfolio"""
        return self.analyze_portfolio(portfolioId)
    
    def analyze_portfolio(self, portfolio_id: int) -> Dict:
        """
        Analyze all stocks in a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
        
        Returns:
            Dictionary with analysis results for all holdings
        """
        try:
            # Input validation
            if not isinstance(portfolio_id, int) or portfolio_id <= 0:
                raise ValueError("Invalid portfolio ID")
            portfolio = Portfolio.objects.get(id=portfolio_id)
            holdings = PortfolioHolding.objects.filter(portfolio=portfolio)
            
            results = {
                'portfolio_name': portfolio.name,
                'analysis_date': datetime.now().isoformat(),
                'holdings': []
            }
            
            for holding in holdings:
                analysis = self.run_full_analysis(holding.stock.symbol)
                
                # Add holding-specific data
                analysis['quantity'] = float(holding.quantity)
                analysis['purchase_price'] = float(holding.average_price)
                analysis['current_value'] = analysis['current_price'] * float(holding.quantity) if analysis.get('current_price') else None
                analysis['purchase_value'] = float(holding.cost_basis)
                analysis['gain_loss'] = float(holding.unrealized_gain_loss)
                analysis['gain_loss_percent'] = float(holding.unrealized_gain_loss_percent)
                
                results['holdings'].append(analysis)
            
            # Calculate portfolio summary
            total_value = sum(h['current_value'] for h in results['holdings'] if h.get('current_value'))
            total_cost = sum(h['purchase_value'] for h in results['holdings'])
            
            results['summary'] = {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_gain_loss': total_value - total_cost,
                'total_gain_loss_percent': ((total_value - total_cost) / total_cost * 100) if total_cost else 0,
                'buy_signals': sum(1 for h in results['holdings'] if h.get('signal') == 'BUY'),
                'sell_signals': sum(1 for h in results['holdings'] if h.get('signal') == 'SELL'),
                'hold_signals': sum(1 for h in results['holdings'] if h.get('signal') == 'HOLD')
            }
            
            return results
            
        except Portfolio.DoesNotExist:
            logger.error(f"Portfolio {portfolio_id} not found")
            return {
                'success': False,
                'error': f'Portfolio {portfolio_id} not found'
            }
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instance
analytics_engine = AnalyticsEngine()