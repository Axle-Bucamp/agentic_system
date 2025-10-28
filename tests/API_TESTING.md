# API Testing Documentation

This document explains how to test the API endpoints with mocked AI responses using tool calls.

## Overview

The API testing framework (`test_api_tools.py`) demonstrates how to:
1. Call the forecasting API endpoints with proper authentication
2. Mock AI responses with tool call structures
3. Make trading decisions based on collected data
4. Simulate an AI agent workflow

## Configuration

### API Key Setup

Update your environment variables or config file with the API key:

```bash
# In your .env file or environment
MCP_API_KEY=sk_EvUybnfnyK3MImCECBhB0Jhhks4FsTd9H9AF3d5F32o
MCP_API_URL=https://forecasting.guidry-cloud.com
USE_MOCK_SERVICES=false  # Set to false to use real API
```

### Supported Endpoints

The following endpoints are tested in the API test suite:

1. **`/api/json/action/{ticker}/{interval}`** - Get DQN action recommendations
2. **`/api/json/metrics/{ticker}/{interval}`** - Get model performance metrics
3. **`/api/json/stock/{seq}/{ticker}`** - Get historical price data
4. **`/api/json/stockatdate/{ticker}/{date}`** - Get stock data at specific date
5. **`/api/json/stockindaterange/{ticker}/{date_begin}/{date_end}`** - Get data in date range
6. **`/api/tickers/available`** - List all available tickers
7. **`/api/json/actions/all/{interval}`** - Get actions for all tickers

## Running the Tests

### Basic Test Run

```bash
# Run the API tests with real API calls
python tests/test_api_tools.py
```

### Test Output

The test script will output:

1. **Available Tickers** - Lists all supported tickers
2. **Action Recommendations** - Shows DQN buy/sell/hold recommendations
3. **Model Metrics** - Displays model accuracy and performance
4. **Historical Data** - Shows price history analysis
5. **AI Reasoning** - Mock agent reasoning and decision making
6. **Tool Call Summary** - Lists all API calls made

### Example Output

```
Step 1: Listing available tickers...
Tool Call: list_available_tickers
Available Tickers: 15

Step 2: Getting action recommendation for BTC-USD...
Tool Call: get_action_recommendation
Recommendation: BUY
Confidence: 85.00%
Action: BUY
Q-Values: [0.10, 0.25, 0.65]
Forecast: 47000.0

Step 5: AI Agent Reasoning...
{
  "analysis": "Based on the data collected:",
  "signals": [
    "DQN recommends: BUY with 85.0% confidence",
    "Model accuracy is 82.0%",
    "Technical indicators support the recommendation",
    "Risk metrics are within acceptable limits"
  ],
  "decision": {
    "action": "BUY",
    "confidence": 0.85,
    "reasoning": "High confidence signal with good model accuracy"
  },
  "risk_assessment": {
    "level": "LOW",
    "max_position_size": 850.0,
    "stop_loss": 0.95
  }
}
```

## Tool Call Structure

Each API call is wrapped in a mock tool call structure:

```python
{
    "tool_call": {
        "name": "get_action_recommendation",
        "arguments": {
            "ticker": "BTC-USD",
            "interval": "hours"
        }
    },
    "result": {...actual API response...},
    "timestamp": "2024-01-01T12:00:00",
    "confidence": 0.85,
    "recommendation": "BUY"
}
```

This structure allows for:
- Tracking which tools were called
- Monitoring arguments passed to tools
- Storing results with metadata
- Logging timestamps for debugging

## Mock AI Workflow

The test demonstrates a complete AI agent workflow:

1. **Discovery** - List available tickers
2. **Analysis** - Get recommendations for specific tickers
3. **Evaluation** - Retrieve model performance metrics
4. **Research** - Gather historical price data
5. **Decision** - Make trading decision based on all data
6. **Reporting** - Generate summary of actions taken

## Using in Your Code

### Basic Usage

```python
from tests.test_api_tools import APIToolCaller

# Initialize the tool caller
tool_caller = APIToolCaller(
    base_url="https://forecasting.guidry-cloud.com",
    api_key="sk_EvUybnfnyK3MImCECBhB0Jhhks4FsTd9H9AF3d5F32o"
)

# Get action recommendation
result = await tool_caller.get_action_recommendation("BTC-USD", "hours")
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Advanced Usage - Complete Workflow

```python
import asyncio

async def trading_decision_workflow(ticker: str):
    """Complete trading decision workflow."""
    tool_caller = APIToolCaller(API_BASE_URL, API_KEY)
    
    # Get all necessary data
    tickers = await tool_caller.list_available_tickers()
    recommendation = await tool_caller.get_action_recommendation(ticker, "hours")
    metrics = await tool_caller.get_ticker_metrics(ticker, "hours")
    data = await tool_caller.get_ticker_data(ticker, "hours")
    
    # Make decision based on data
    if recommendation['confidence'] > 0.7 and metrics['accuracy'] > 0.75:
        action = recommendation['recommendation']
        print(f"Decision: {action} {ticker}")
    
    return {
        "ticker": ticker,
        "action": recommendation['recommendation'],
        "confidence": recommendation['confidence'],
        "accuracy": metrics['accuracy']
    }

# Run the workflow
result = asyncio.run(trading_decision_workflow("BTC-USD"))
```

## Integration with Trading System

The forecasting client in `core/forecasting_client.py` has been updated to use the `X-API-Key` header. Set the API key in your environment:

```bash
export MCP_API_KEY=sk_EvUybnfnyK3MImCECBhB0Jhhks4FsTd9H9AF3d5F32o
```

The trading system will automatically use the API key for all forecasting API calls.

## Troubleshooting

### Authentication Errors

If you get authentication errors:
1. Check that `MCP_API_KEY` is set correctly
2. Verify the API key is valid and not expired
3. Ensure you're using `X-API-Key` header (not `Authorization`)

### Connection Errors

If you can't connect:
1. Check your internet connection
2. Verify the API URL is correct
3. Check if the API service is running

### Mock Mode

If tests are using mock data:
1. Set `USE_MOCK_SERVICES=false` in your environment
2. Ensure `mock_mode` is set to `False` in client config

## Next Steps

1. Run the API tests to verify connectivity
2. Integrate the API calls into your trading agents
3. Monitor API usage and rate limits
4. Set up alerts for API failures

## Related Files

- `tests/test_api_tools.py` - Main test script
- `core/forecasting_client.py` - Forecasting API client
- `core/config.py` - Configuration management
- `core/exchange_manager.py` - Exchange manager
- `api/main.py` - API endpoints

