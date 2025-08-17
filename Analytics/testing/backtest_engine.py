"""
Backtesting Engine for Universal LSTM and TA-Weighted Predictions
Validates historical performance and calculates key metrics
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    best_trade: float
    worst_trade: float
    volatility: float
    calmar_ratio: float  # Return / Max Drawdown
    


class BacktestEngine:
    """
    Engine for backtesting prediction models on historical data.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        position_size: float = 0.1,  # 10% of capital per position
        max_positions: int = 10,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005  # 0.05% slippage
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            position_size: Fraction of capital per position
            max_positions: Maximum concurrent positions
            commission: Trading commission rate
            slippage: Slippage rate
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.commission = commission
        self.slippage = slippage
        
        # State tracking
        self.reset()
        
    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.positions = {}  # symbol -> position dict
        self.trades = []  # completed trades
        self.equity_curve = [self.initial_capital]
        self.timestamps = []
        
    def run_backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        strategy: str = 'threshold',
        threshold: float = 0.01,  # 1% threshold for trading
        confidence_min: float = 0.6  # Minimum confidence to trade
    ) -> BacktestResult:
        """
        Run backtest on historical predictions.
        
        Args:
            predictions: DataFrame with columns [date, symbol, predicted_price, confidence]
            prices: DataFrame with columns [date, symbol, open, high, low, close]
            strategy: Trading strategy ('threshold', 'confidence_weighted', 'ta_weighted')
            threshold: Minimum price change to trigger trade
            confidence_min: Minimum confidence to execute trade
            
        Returns:
            BacktestResult with performance metrics
        """
        self.reset()
        
        # Sort by date
        predictions = predictions.sort_values('date')
        dates = predictions['date'].unique()
        
        logger.info(f"Running backtest on {len(dates)} trading days")
        
        for date in dates:
            # Get predictions for this date
            daily_predictions = predictions[predictions['date'] == date]
            
            # Process each prediction
            for _, pred in daily_predictions.iterrows():
                symbol = pred['symbol']
                predicted_price = pred.get('predicted_price', 0)
                confidence = pred.get('confidence', 0.5)
                
                # Get current price
                price_data = prices[(prices['date'] == date) & (prices['symbol'] == symbol)]
                if price_data.empty:
                    continue
                    
                current_price = price_data.iloc[0]['close']
                
                # Calculate expected return
                expected_return = (predicted_price - current_price) / current_price
                
                # Apply trading strategy
                if strategy == 'threshold':
                    should_trade = abs(expected_return) > threshold and confidence >= confidence_min
                elif strategy == 'confidence_weighted':
                    should_trade = abs(expected_return * confidence) > threshold
                elif strategy == 'ta_weighted':
                    ta_weight = pred.get('ta_weight', 1.0)
                    should_trade = abs(expected_return * confidence * ta_weight) > threshold
                else:
                    should_trade = False
                
                if should_trade:
                    if expected_return > 0:
                        # Buy signal
                        self._execute_trade(symbol, 'buy', current_price, confidence, date)
                    else:
                        # Sell signal (or short if allowed)
                        if symbol in self.positions:
                            self._execute_trade(symbol, 'sell', current_price, confidence, date)
            
            # Update equity
            self._update_equity(date, prices)
            
        # Close all remaining positions at last price
        self._close_all_positions(dates[-1], prices)
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _execute_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        confidence: float,
        date: datetime
    ):
        """Execute a trade."""
        # Apply slippage
        if action == 'buy':
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)
        
        # Calculate position size based on confidence
        position_value = self.capital * self.position_size * min(confidence, 1.0)
        shares = int(position_value / execution_price)
        
        if shares == 0:
            return
        
        # Apply commission
        commission_cost = position_value * self.commission
        
        if action == 'buy' and symbol not in self.positions:
            # Open new position
            if len(self.positions) < self.max_positions:
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': execution_price,
                    'entry_date': date,
                    'confidence': confidence
                }
                self.capital -= (shares * execution_price + commission_cost)
                
        elif action == 'sell' and symbol in self.positions:
            # Close position
            position = self.positions[symbol]
            exit_value = shares * execution_price - commission_cost
            
            # Calculate return
            entry_value = position['shares'] * position['entry_price']
            trade_return = (exit_value - entry_value) / entry_value
            
            self.trades.append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': date,
                'entry_price': position['entry_price'],
                'exit_price': execution_price,
                'shares': position['shares'],
                'return': trade_return,
                'confidence': position['confidence']
            })
            
            self.capital += exit_value
            del self.positions[symbol]
    
    def _update_equity(self, date: datetime, prices: pd.DataFrame):
        """Update equity curve with current positions."""
        total_value = self.capital
        
        # Add value of open positions
        for symbol, position in self.positions.items():
            price_data = prices[(prices['date'] == date) & (prices['symbol'] == symbol)]
            if not price_data.empty:
                current_price = price_data.iloc[0]['close']
                total_value += position['shares'] * current_price
        
        self.equity_curve.append(total_value)
        self.timestamps.append(date)
    
    def _close_all_positions(self, date: datetime, prices: pd.DataFrame):
        """Close all remaining positions."""
        for symbol in list(self.positions.keys()):
            price_data = prices[(prices['date'] == date) & (prices['symbol'] == symbol)]
            if not price_data.empty:
                current_price = price_data.iloc[0]['close']
                self._execute_trade(symbol, 'sell', current_price, 1.0, date)
    
    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        # Convert equity curve to returns
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Total return
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # Sharpe ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        if self.trades:
            trade_returns = [t['return'] for t in self.trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r <= 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum(winning_trades)
            gross_loss = abs(sum(losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            best_trade = max(trade_returns)
            worst_trade = min(trade_returns)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            best_trade = 0
            worst_trade = 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len([t for t in self.trades if t['return'] > 0]),
            losing_trades=len([t for t in self.trades if t['return'] <= 0]),
            best_trade=best_trade,
            worst_trade=worst_trade,
            volatility=volatility,
            calmar_ratio=calmar_ratio
        )
    
    def plot_results(self) -> Dict[str, Any]:
        """
        Generate plot data for visualization.
        
        Returns:
            Dictionary with plot data
        """
        return {
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps,
            'trades': self.trades,
            'positions': self.positions
        }


def run_model_comparison_backtest(
    predictions_base: pd.DataFrame,
    predictions_enhanced: pd.DataFrame,
    prices: pd.DataFrame,
    **kwargs
) -> Tuple[BacktestResult, BacktestResult]:
    """
    Compare base model vs enhanced model performance.
    
    Args:
        predictions_base: Base model predictions
        predictions_enhanced: Enhanced model with TA weighting
        prices: Historical prices
        **kwargs: Additional backtest parameters
        
    Returns:
        Tuple of (base_results, enhanced_results)
    """
    engine = BacktestEngine(**kwargs)
    
    # Run base model backtest
    logger.info("Running backtest for base model...")
    base_results = engine.run_backtest(predictions_base, prices)
    
    # Run enhanced model backtest
    logger.info("Running backtest for enhanced model...")
    enhanced_results = engine.run_backtest(
        predictions_enhanced, 
        prices,
        strategy='ta_weighted'  # Use TA-weighted strategy
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("BACKTEST COMPARISON RESULTS")
    print("=" * 60)
    
    print("\n📊 Performance Metrics:")
    print(f"{'Metric':<20} {'Base Model':>15} {'Enhanced Model':>15}")
    print("-" * 50)
    print(f"{'Total Return':<20} {base_results.total_return:>14.2%} {enhanced_results.total_return:>14.2%}")
    print(f"{'Sharpe Ratio':<20} {base_results.sharpe_ratio:>15.3f} {enhanced_results.sharpe_ratio:>15.3f}")
    print(f"{'Max Drawdown':<20} {base_results.max_drawdown:>14.2%} {enhanced_results.max_drawdown:>14.2%}")
    print(f"{'Win Rate':<20} {base_results.win_rate:>14.2%} {enhanced_results.win_rate:>14.2%}")
    print(f"{'Profit Factor':<20} {base_results.profit_factor:>15.2f} {enhanced_results.profit_factor:>15.2f}")
    print(f"{'Calmar Ratio':<20} {base_results.calmar_ratio:>15.3f} {enhanced_results.calmar_ratio:>15.3f}")
    print(f"{'Total Trades':<20} {base_results.total_trades:>15d} {enhanced_results.total_trades:>15d}")
    
    # Determine winner
    print("\n🏆 Winner:")
    base_score = (base_results.sharpe_ratio + base_results.calmar_ratio + base_results.profit_factor)
    enhanced_score = (enhanced_results.sharpe_ratio + enhanced_results.calmar_ratio + enhanced_results.profit_factor)
    
    if enhanced_score > base_score:
        improvement = ((enhanced_score - base_score) / base_score) * 100
        print(f"✅ Enhanced Model (+{improvement:.1f}% improvement)")
    else:
        print("❌ Base Model (Enhanced model needs tuning)")
    
    return base_results, enhanced_results