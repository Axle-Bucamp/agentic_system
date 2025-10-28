"""
Test DEX Simulator integration with agents.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dex_simulator_client import dex_simulator_client
from core.forecasting_client import forecasting_client
from core.config import settings
from core.logging import log


async def test_dex_connection():
    """Test connection to DEX simulator."""
    print("\n" + "="*60)
    print("Testing DEX Simulator Connection")
    print("="*60)
    
    try:
        await dex_simulator_client.connect()
        print("✓ DEX Simulator connected successfully")
        
        # Test getting portfolio
        portfolio = await dex_simulator_client.get_portfolio_status()
        print(f"\nPortfolio Status:")
        print(f"  Balance: ${portfolio['balance_usdc']:.2f} USDC")
        print(f"  Total Value: ${portfolio['total_value_usdc']:.2f} USDC")
        print(f"  Holdings: {len(portfolio['holdings'])} assets")
        print(f"  Daily PnL: ${portfolio['daily_pnl']:.2f}")
        
        # Test getting ticker prices
        prices = await dex_simulator_client.get_ticker_prices()
        if prices:
            print(f"\nTicker Prices:")
            for ticker, price in list(prices.items())[:5]:
                print(f"  {ticker}: ${price:.2f}")
        
        await dex_simulator_client.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ DEX Simulator connection failed: {e}")
        return False


async def test_forecasting_api():
    """Test connection to Forecasting API."""
    print("\n" + "="*60)
    print("Testing Forecasting API Connection")
    print("="*60)
    
    try:
        await forecasting_client.connect()
        print("✓ Forecasting API connected successfully")
        
        # Test getting available tickers
        tickers = await forecasting_client.get_available_tickers()
        print(f"\nAvailable Tickers: {len(tickers)}")
        if tickers:
            print(f"  Sample: {tickers[0] if isinstance(tickers[0], dict) else tickers[0]}")
        
        # Test getting action recommendation
        recommendation = await forecasting_client.get_action_recommendation("BTC-USD", "hours")
        print(f"\nBTC Recommendation:")
        print(f"  Action: {recommendation.get('action_name', 'N/A')}")
        print(f"  Confidence: {recommendation.get('action_confidence', 0):.2%}")
        
        await forecasting_client.disconnect()
        return True
        
    except Exception as e:
        print(f"✗ Forecasting API connection failed: {e}")
        return False


async def test_mock_trade():
    """Test executing a mock trade."""
    print("\n" + "="*60)
    print("Testing Mock Trade Execution")
    print("="*60)
    
    try:
        await dex_simulator_client.connect()
        
        # Get initial state
        initial_portfolio = await dex_simulator_client.get_portfolio_status()
        print(f"\nInitial State:")
        print(f"  Balance: ${initial_portfolio['balance_usdc']:.2f}")
        print(f"  Total Value: ${initial_portfolio['total_value_usdc']:.2f}")
        
        # Execute mock buy
        test_amount = 100  # Buy $100 of BTC
        result = await dex_simulator_client.buy_asset("BTC", test_amount)
        
        if result.get("success"):
            print(f"\n✓ Trade executed successfully")
            
            # Get updated state
            updated_portfolio = await dex_simulator_client.get_portfolio_status()
            print(f"\nUpdated State:")
            print(f"  Balance: ${updated_portfolio['balance_usdc']:.2f}")
            print(f"  Total Value: ${updated_portfolio['total_value_usdc']:.2f}")
            print(f"  Holdings: {updated_portfolio['holdings']}")
            
            return True
        else:
            print(f"✗ Trade failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Mock trade test failed: {e}")
        return False
    finally:
        await dex_simulator_client.disconnect()


async def test_full_workflow():
    """Test complete workflow integration."""
    print("\n" + "="*60)
    print("Testing Complete Workflow")
    print("="*60)
    
    try:
        # 1. Connect to services
        await dex_simulator_client.connect()
        await forecasting_client.connect()
        print("✓ Services connected")
        
        # 2. Get portfolio
        portfolio = await dex_simulator_client.get_portfolio_status()
        balance = portfolio['balance_usdc']
        print(f"✓ Portfolio retrieved: ${balance:.2f} USDC")
        
        # 3. Get DQN recommendation
        recommendation = await forecasting_client.get_action_recommendation("BTC-USD", "hours")
        action = recommendation.get('action_name', 'HOLD')
        confidence = recommendation.get('action_confidence', 0.5)
        print(f"✓ DQN recommendation: {action} (confidence: {confidence:.2%})")
        
        # 4. Execute if confident enough
        if confidence > 0.6 and action in ['BUY', 'SELL']:
            if action == 'BUY' and balance > 100:
                trade_amount = min(balance * 0.1, 100)  # 10% of balance, max $100
                result = await dex_simulator_client.buy_asset("BTC", trade_amount)
                print(f"✓ Trade executed: {result.get('success', False)}")
            elif action == 'SELL':
                holdings = await dex_simulator_client.get_current_holdings()
                if 'BTC' in holdings and holdings['BTC'] > 0:
                    sell_amount = holdings['BTC'] * 0.5  # Sell 50%
                    result = await dex_simulator_client.sell_asset("BTC", sell_amount)
                    print(f"✓ Trade executed: {result.get('success', False)}")
        
        # 5. Get updated portfolio
        updated = await dex_simulator_client.get_portfolio_status()
        print(f"✓ Final portfolio: ${updated['total_value_usdc']:.2f} USDC")
        
        await dex_simulator_client.disconnect()
        await forecasting_client.disconnect()
        
        return True
        
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DEX Simulator Integration Tests")
    print("="*60)
    
    results = {}
    
    # Test 1: DEX Connection
    results['dex'] = await test_dex_connection()
    
    # Test 2: Forecasting API
    results['forecasting'] = await test_forecasting_api()
    
    # Test 3: Mock Trade (only if both connections work)
    if results['dex'] and results['forecasting']:
        results['trade'] = await test_mock_trade()
        
        # Test 4: Full Workflow
        results['workflow'] = await test_full_workflow()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"DEX Connection:       {'✓ PASS' if results.get('dex') else '✗ FAIL'}")
    print(f"Forecasting API:      {'✓ PASS' if results.get('forecasting') else '✗ FAIL'}")
    print(f"Mock Trade:           {'✓ PASS' if results.get('trade') else '✗ FAIL'}")
    print(f"Full Workflow:        {'✓ PASS' if results.get('workflow') else '✗ FAIL'}")
    print("="*60)
    
    success_count = sum(1 for v in results.values() if v)
    print(f"\nTotal: {success_count}/{len(results)} tests passed")
    
    if success_count == len(results):
        print("\n✓ All tests passed! DEX integration is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())

