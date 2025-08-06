"""
Test script for the Analytics Engine.
Run this to verify the engine is working correctly.
"""

import os
import sys
import django

# Setup Django environment
# Get the project root directory (where manage.py is located)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VoyageurCompass.settings')
django.setup()

from Analytics.services.engine import analytics_engine
from Data.services.yahoo_finance import yahoo_finance_service
from Data.models import Stock


def test_analytics_engine():
    """Test the analytics engine with a sample stock."""
    
    print("=" * 60)
    print("Testing VoyageurCompass Analytics Engine")
    print("=" * 60)
    
    # Test stock symbol
    test_symbol = 'AAPL'
    
    print(f"\n1. Ensuring {test_symbol} data is in database...")
    
    # First, sync some data for testing
    sync_result = yahoo_finance_service.get_stock_data(test_symbol, period='6mo', sync_db=True)
    
    if 'error' in sync_result:
        print(f"Error syncing data: {sync_result['error']}")
        return
    
    print(f"   ✓ Data synced successfully")
    
    # Run full analysis
    print(f"\n2. Running full analysis for {test_symbol}...")
    
    analysis = analytics_engine.run_full_analysis(test_symbol)
    
    if not analysis.get('success'):
        print(f"   ✗ Analysis failed: {analysis.get('error')}")
        return
    
    print(f"   ✓ Analysis completed successfully")
    
    # Display results
    print("\n3. Analysis Results:")
    print("-" * 40)
    
    print(f"   Company: {analysis['company_name']}")
    print(f"   Sector: {analysis['sector']}")
    print(f"   Current Price: ${analysis['current_price']:.2f}" if analysis['current_price'] else "   Current Price: N/A")
    print(f"   Target Price: ${analysis['target_price']:.2f}" if analysis['target_price'] else "   Target Price: N/A")
    
    print(f"\n   Performance Metrics:")
    print(f"   - Stock Return: {analysis['stock_return']:.2f}%" if analysis['stock_return'] else "   - Stock Return: N/A")
    print(f"   - ETF Return: {analysis['etf_return']:.2f}%" if analysis['etf_return'] else "   - ETF Return: N/A")
    print(f"   - Volatility: {analysis['volatility']:.2%}" if analysis['volatility'] else "   - Volatility: N/A")
    print(f"   - Sharpe Ratio: {analysis['sharpe_ratio']:.2f}" if analysis['sharpe_ratio'] else "   - Sharpe Ratio: N/A")
    
    print(f"\n   Technical Indicators:")
    print(f"   - RSI: {analysis['rsi']:.2f}" if analysis['rsi'] else "   - RSI: N/A")
    print(f"   - MA(20): ${analysis['ma_20']:.2f}" if analysis['ma_20'] else "   - MA(20): N/A")
    print(f"   - MA(50): ${analysis['ma_50']:.2f}" if analysis['ma_50'] else "   - MA(50): N/A")
    
    if analysis['bollinger_bands']['middle']:
        print(f"   - Bollinger Bands:")
        print(f"     Upper: ${analysis['bollinger_bands']['upper']:.2f}")
        print(f"     Middle: ${analysis['bollinger_bands']['middle']:.2f}")
        print(f"     Lower: ${analysis['bollinger_bands']['lower']:.2f}")
    
    print(f"\n   📊 TRADING SIGNAL: {analysis['signal']}")
    print(f"   Reason: {analysis['signal_reason']}")
    
    print(f"\n   Criteria Met:")
    for criterion, met in analysis['criteria_met'].items():
        status = "✓" if met else "✗"
        print(f"   {status} {criterion}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_analytics_engine()