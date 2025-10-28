"""Quick test to verify DEX simulator is working."""
import asyncio
import httpx

async def test_simulator():
    """Test if DEX simulator is running."""
    print("Testing DEX Simulator...")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test 1: Get wallet balance
            print("\n1. Testing /wallet_usdc endpoint...")
            response = await client.get("http://localhost:8001/wallet_usdc")
            if response.status_code == 200:
                print(f"   ✓ Success! Balance: {response.json()}")
            else:
                print(f"   ✗ Failed with status {response.status_code}")
            
            # Test 2: Get holdings
            print("\n2. Testing /wallet_holding endpoint...")
            response = await client.get("http://localhost:8001/wallet_holding")
            if response.status_code == 200:
                print(f"   ✓ Success! Holdings: {response.json()}")
            else:
                print(f"   ✗ Failed with status {response.status_code}")
            
            # Test 3: Get portfolio value
            print("\n3. Testing /ticker_sum_usdc endpoint...")
            response = await client.get("http://localhost:8001/ticker_sum_usdc")
            if response.status_code == 200:
                print(f"   ✓ Success! Total value: {response.json()}")
            else:
                print(f"   ✗ Failed with status {response.status_code}")
            
            print("\n✓ DEX Simulator is running correctly!")
            return True
            
    except httpx.ConnectError:
        print("\n✗ DEX Simulator is not running!")
        print("   Please start it with: cd dex-simulator && python main.py")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_simulator())

