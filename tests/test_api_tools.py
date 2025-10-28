"""
Test script for API endpoints using tool calls with mocked AI responses.

This demonstrates how to call the forecasting API endpoints with proper
authentication and handle responses with tool calls.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# API Configuration
API_BASE_URL = "https://forecasting.guidry-cloud.com"
API_KEY = "sk_EvUybnfnyK3MImCECBhB0Jhhks4FsTd9H9AF3d5F32o"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


class APIToolCaller:
    """Mock AI tool caller for API endpoints."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
    
    async def _call_api(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        body: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an API call and return the response."""
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            kwargs = {"headers": self.headers}
            if params:
                kwargs["params"] = params
            if body:
                kwargs["json"] = body
            
            async with session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        "error": f"HTTP {response.status}",
                        "message": error_text
                    }
    
    async def get_action_recommendation(
        self, 
        ticker: str, 
        interval: str = "hours"
    ) -> Dict[str, Any]:
        """
        Tool: Get action recommendation for a ticker.
        
        This tool calls the /api/json/action/{ticker}/{interval} endpoint
        to get DQN action recommendations.
        
        Args:
            ticker: Stock ticker (e.g., 'BTC-USD')
            interval: Time interval (days, hours, minutes, thirty)
        
        Returns:
            Mock tool call response with action, confidence, and forecast data
        """
        endpoint = f"/api/json/action/{ticker}/{interval}"
        result = await self._call_api("GET", endpoint)
        
        # Mock AI response with tool call structure
        return {
            "tool_call": {
                "name": "get_action_recommendation",
                "arguments": {
                    "ticker": ticker,
                    "interval": interval
                }
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": result.get("confidence", 0.5),
            "recommendation": result.get("action_name", "HOLD")
        }
    
    async def get_ticker_metrics(
        self, 
        ticker: str, 
        interval: str = "hours"
    ) -> Dict[str, Any]:
        """
        Tool: Get metrics for a ticker.
        
        This tool calls the /api/json/metrics/{ticker}/{interval} endpoint
        to get model performance metrics.
        
        Args:
            ticker: Stock ticker (e.g., 'BTC-USD')
            interval: Time interval (days, hours, minutes, thirty)
        
        Returns:
            Mock tool call response with performance metrics
        """
        endpoint = f"/api/json/metrics/{ticker}/{interval}"
        result = await self._call_api("GET", endpoint)
        
        # Mock AI response with tool call structure
        return {
            "tool_call": {
                "name": "get_ticker_metrics",
                "arguments": {
                    "ticker": ticker,
                    "interval": interval
                }
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "accuracy": result.get("accuracy", 0.75),
            "performance": result.get("performance", {})
        }
    
    async def get_ticker_data(
        self, 
        ticker: str, 
        seq: str = "hours"
    ) -> Dict[str, Any]:
        """
        Tool: Get historical data for a ticker.
        
        This tool calls the /api/json/stock/{seq}/{ticker} endpoint
        to get historical price data.
        
        Args:
            ticker: Stock ticker (e.g., 'BTC-USD')
            seq: Sequence request (days, hours, minutes, thirty)
        
        Returns:
            Mock tool call response with historical data
        """
        endpoint = f"/api/json/stock/{seq}/{ticker}"
        result = await self._call_api("GET", endpoint)
        
        # Mock AI response with tool call structure
        return {
            "tool_call": {
                "name": "get_ticker_data",
                "arguments": {
                    "ticker": ticker,
                    "seq": seq
                }
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "data_points": len(result) if isinstance(result, list) else 1
        }
    
    async def get_stock_at_date(
        self, 
        ticker: str, 
        date: str
    ) -> Dict[str, Any]:
        """
        Tool: Get stock data at a specific date.
        
        This tool calls the /api/json/stockatdate/{ticker}/{date} endpoint.
        
        Args:
            ticker: Stock ticker (e.g., 'BTC-USD')
            date: Date in YYYY-MM-DD format
        
        Returns:
            Mock tool call response with price data at specific date
        """
        endpoint = f"/api/json/stockatdate/{ticker}/{date}"
        result = await self._call_api("GET", endpoint)
        
        return {
            "tool_call": {
                "name": "get_stock_at_date",
                "arguments": {
                    "ticker": ticker,
                    "date": date
                }
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_stock_in_date_range(
        self, 
        ticker: str, 
        date_begin: str,
        date_end: str
    ) -> Dict[str, Any]:
        """
        Tool: Get stock data within a date range.
        
        This tool calls the /api/json/stockindaterange/{ticker}/{date_begin}/{date_end} endpoint.
        
        Args:
            ticker: Stock ticker (e.g., 'BTC-USD')
            date_begin: Start date in YYYY-MM-DD format
            date_end: End date in YYYY-MM-DD format
        
        Returns:
            Mock tool call response with price data in date range
        """
        endpoint = f"/api/json/stockindaterange/{ticker}/{date_begin}/{date_end}"
        result = await self._call_api("GET", endpoint)
        
        return {
            "tool_call": {
                "name": "get_stock_in_date_range",
                "arguments": {
                    "ticker": ticker,
                    "date_begin": date_begin,
                    "date_end": date_end
                }
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "data_points": len(result) if isinstance(result, list) else 1
        }
    
    async def list_available_tickers(self) -> Dict[str, Any]:
        """
        Tool: List available tickers.
        
        This tool calls the /api/tickers/available endpoint
        to get all available tickers.
        
        Returns:
            Mock tool call response with list of tickers
        """
        endpoint = "/api/tickers/available"
        result = await self._call_api("GET", endpoint)
        
        return {
            "tool_call": {
                "name": "list_available_tickers",
                "arguments": {}
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(result) if isinstance(result, list) else 0
        }
    
    async def get_all_actions(self, interval: str = "hours") -> Dict[str, Any]:
        """
        Tool: Get all actions for all tickers.
        
        This tool calls the /api/json/actions/all/{interval} endpoint.
        
        Args:
            interval: Time interval (days, hours, minutes, thirty)
        
        Returns:
            Mock tool call response with all actions
        """
        endpoint = f"/api/json/actions/all/{interval}"
        result = await self._call_api("GET", endpoint)
        
        return {
            "tool_call": {
                "name": "get_all_actions",
                "arguments": {
                    "interval": interval
                }
            },
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(result) if isinstance(result, list) else 0
        }


async def mock_ai_workflow():
    """
    Simulate an AI agent workflow using tool calls to the API.
    
    This demonstrates how an AI agent would:
    1. List available tickers
    2. Get recommendations for specific tickers
    3. Analyze metrics
    4. Retrieve historical data
    5. Make trading decisions based on the data
    """
    tool_caller = APIToolCaller(API_BASE_URL, API_KEY)
    
    print("=" * 80)
    print("Mock AI Agent Workflow - Trading Decision Making")
    print("=" * 80)
    print()
    
    # Step 1: List available tickers
    print("Step 1: Listing available tickers...")
    tickers_result = await tool_caller.list_available_tickers()
    print(f"Tool Call: {tickers_result['tool_call']['name']}")
    print(f"Available Tickers: {tickers_result.get('count', 0)}")
    print()
    
    # Get a sample ticker from the list
    tickers = tickers_result.get('result', [])
    if isinstance(tickers, list) and len(tickers) > 0:
        # Extract ticker names if it's a list of dicts
        if isinstance(tickers[0], dict):
            sample_ticker = tickers[0].get('symbol', 'BTC-USD')
        else:
            sample_ticker = tickers[0]
    else:
        sample_ticker = "BTC-USD"
    
    # Step 2: Get action recommendation for the sample ticker
    print(f"Step 2: Getting action recommendation for {sample_ticker}...")
    action_result = await tool_caller.get_action_recommendation(sample_ticker, "hours")
    print(f"Tool Call: {action_result['tool_call']['name']}")
    print(f"Recommendation: {action_result.get('recommendation', 'N/A')}")
    print(f"Confidence: {action_result.get('confidence', 0):.2%}")
    
    # Extract and display action details
    result = action_result.get('result', {})
    if isinstance(result, dict):
        print(f"Action: {result.get('action_name', 'N/A')}")
        print(f"Q-Values: {result.get('q_values', {})}")
        print(f"Forecast: {result.get('forecast', 'N/A')}")
    print()
    
    # Step 3: Get metrics for the ticker
    print(f"Step 3: Getting metrics for {sample_ticker}...")
    metrics_result = await tool_caller.get_ticker_metrics(sample_ticker, "hours")
    print(f"Tool Call: {metrics_result['tool_call']['name']}")
    print(f"Accuracy: {metrics_result.get('accuracy', 0):.2%}")
    
    result = metrics_result.get('result', {})
    if isinstance(result, dict):
        print(f"Performance Metrics: {result.get('performance', {})}")
    print()
    
    # Step 4: Get historical data for analysis
    print(f"Step 4: Getting historical data for {sample_ticker}...")
    data_result = await tool_caller.get_ticker_data(sample_ticker, "hours")
    print(f"Tool Call: {data_result['tool_call']['name']}")
    print(f"Data Points: {data_result.get('data_points', 0)}")
    print()
    
    # Step 5: Mock AI reasoning and decision
    print("Step 5: AI Agent Reasoning...")
    print("-" * 80)
    
    recommendation = action_result.get('recommendation', 'HOLD')
    confidence = action_result.get('confidence', 0)
    accuracy = metrics_result.get('accuracy', 0)
    
    # Mock AI reasoning output
    reasoning = {
        "analysis": "Based on the data collected:",
        "signals": [
            f"DQN recommends: {recommendation} with {confidence:.1%} confidence",
            f"Model accuracy is {accuracy:.1%}",
            "Technical indicators support the recommendation",
            "Risk metrics are within acceptable limits"
        ],
        "decision": {
            "action": recommendation if confidence > 0.6 else "HOLD",
            "confidence": confidence,
            "reasoning": "High confidence signal with good model accuracy"
        },
        "risk_assessment": {
            "level": "LOW" if confidence > 0.7 else "MEDIUM" if confidence > 0.5 else "HIGH",
            "max_position_size": confidence * 1000,
            "stop_loss": 0.95 if confidence > 0.7 else 0.90
        }
    }
    
    print(json.dumps(reasoning, indent=2))
    print()
    
    # Step 6: Final mock AI tool call summary
    print("Step 6: Tool Call Summary")
    print("-" * 80)
    tool_calls_made = [
        tickers_result['tool_call'],
        action_result['tool_call'],
        metrics_result['tool_call'],
        data_result['tool_call']
    ]
    
    for i, tool_call in enumerate(tool_calls_made, 1):
        print(f"{i}. {tool_call['name']}({', '.join(f'{k}={v}' for k, v in tool_call['arguments'].items())})")
    
    print()
    print("=" * 80)
    print("Mock AI Agent Workflow Complete")
    print("=" * 80)


async def test_specific_endpoints():
    """Test specific endpoints with detailed output."""
    tool_caller = APIToolCaller(API_BASE_URL, API_KEY)
    
    print("\n" + "=" * 80)
    print("Testing Specific API Endpoints")
    print("=" * 80)
    
    # Test 1: Get action recommendation
    print("\n1. Testing /api/json/action/{ticker}/{interval}")
    print("-" * 80)
    result = await tool_caller.get_action_recommendation("BTC-USD", "hours")
    print(json.dumps(result, indent=2))
    
    # Test 2: Get metrics
    print("\n2. Testing /api/json/metrics/{ticker}/{interval}")
    print("-" * 80)
    result = await tool_caller.get_ticker_metrics("BTC-USD", "hours")
    print(json.dumps(result, indent=2))
    
    # Test 3: Get data
    print("\n3. Testing /api/json/stock/{seq}/{ticker}")
    print("-" * 80)
    result = await tool_caller.get_ticker_data("BTC-USD", "hours")
    print(json.dumps(result, indent=2))
    
    # Test 4: List tickers
    print("\n4. Testing /api/tickers/available")
    print("-" * 80)
    result = await tool_caller.list_available_tickers()
    print(f"Found {result.get('count', 0)} tickers")
    if isinstance(result.get('result'), list) and len(result.get('result', [])) > 0:
        print(f"Sample ticker: {result['result'][0]}")
    
    # Test 5: Get all actions
    print("\n5. Testing /api/json/actions/all/{interval}")
    print("-" * 80)
    result = await tool_caller.get_all_actions("hours")
    print(f"Found {result.get('count', 0)} actions")
    
    print("\n" + "=" * 80)
    print("All Endpoint Tests Complete")
    print("=" * 80)


if __name__ == "__main__":
    print("API Tool Caller - Testing with Mock AI Responses")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print()
    
    # Run the mock AI workflow
    asyncio.run(mock_ai_workflow())
    
    # Run specific endpoint tests
    asyncio.run(test_specific_endpoints())

