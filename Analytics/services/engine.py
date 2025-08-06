"""
Analytics Engine Module
Handles all financial calculations and analysis for VoyageurCompass.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Core analytics engine for financial calculations.
    """
    
    def __init__(self):
        """Initialize the analytics engine."""
        logger.info("Analytics Engine initialized")
    
    def calculate_returns(self, prices: List[float], period: str = 'daily') -> List[float]:
        """
        Calculate returns from a price series.
        
        Args:
            prices: List of prices
            period: Time period for returns ('daily', 'weekly', 'monthly')
        
        Returns:
            List of calculated returns
        """
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                return_value = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                returns.append(return_value)
        
        return returns
    
    def calculate_moving_average(self, prices: List[float], window: int) -> List[float]:
        """
        Calculate simple moving average.
        
        Args:
            prices: List of prices
            window: Window size for moving average
        
        Returns:
            List of moving average values
        """
        if len(prices) < window:
            return []
        
        ma_values = []
        for i in range(window - 1, len(prices)):
            window_slice = prices[i - window + 1:i + 1]
            ma_values.append(sum(window_slice) / window)
        
        return ma_values
    
    def calculate_volatility(self, returns: List[float]) -> float:
        """
        Calculate volatility (standard deviation) of returns.
        
        Args:
            returns: List of return values
        
        Returns:
            Volatility value
        """
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        squared_diffs = [(r - mean_return) ** 2 for r in returns]
        variance = sum(squared_diffs) / len(returns)
        volatility = variance ** 0.5
        
        return volatility
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of return values
            risk_free_rate: Risk-free rate (annual)
        
        Returns:
            Sharpe ratio
        """
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        volatility = self.calculate_volatility(returns)
        
        if volatility == 0:
            return 0.0
        
        # Convert annual risk-free rate to period rate
        period_rf_rate = risk_free_rate / 252  # Assuming daily returns
        
        sharpe = (mean_return - period_rf_rate) / volatility
        return sharpe
    
    def analyze_stock(self, stock_data: Dict) -> Dict:
        """
        Perform comprehensive analysis on stock data.
        
        Args:
            stock_data: Dictionary containing stock information
        
        Returns:
            Dictionary with analysis results
        """
        try:
            prices = stock_data.get('prices', [])
            
            if not prices:
                return {'error': 'No price data available'}
            
            returns = self.calculate_returns(prices)
            ma_20 = self.calculate_moving_average(prices, 20)
            ma_50 = self.calculate_moving_average(prices, 50)
            volatility = self.calculate_volatility(returns)
            sharpe = self.calculate_sharpe_ratio(returns)
            
            analysis = {
                'symbol': stock_data.get('symbol', 'N/A'),
                'latest_price': prices[-1] if prices else None,
                'returns': returns[-10:] if returns else [],  # Last 10 returns
                'ma_20': ma_20[-1] if ma_20 else None,
                'ma_50': ma_50[-1] if ma_50 else None,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'analysis_date': datetime.now().isoformat()
            }
            
            logger.info(f"Analysis completed for {stock_data.get('symbol', 'Unknown')}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock: {str(e)}")
            return {'error': str(e)}


# Singleton instance
analytics_engine = AnalyticsEngine()