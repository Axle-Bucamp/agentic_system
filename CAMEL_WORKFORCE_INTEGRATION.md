# CAMEL Workforce Integration

This document describes the CAMEL Workforce integration for the Agentic Trading System.

## Overview

The system has been upgraded to use CAMEL Workforce for advanced multi-agent orchestration, replacing the traditional OrchestratorAgent with a more sophisticated task decomposition and coordination system.

## Architecture

### Components

1. **Workforce Orchestrator** (`agents/workforce_orchestrator.py`)
   - Main orchestrator using CAMEL Workforce
   - Task decomposition agent
   - Coordinator agent for task assignment
   - Integrates with Redis for backward compatibility

2. **Worker Agents** (`agents/workforce_workers/`)
   - **DQN Worker**: Handles DQN predictions and forecasts
   - **Chart Analysis Worker**: Technical analysis (unique capability)
   - **Risk Assessment Worker**: Risk evaluation (unique capability)
   - **Market Research Worker**: News and sentiment analysis
   - **Trade Execution Worker**: Trade execution (unique capability)

3. **CAMEL Tools** (`core/camel_tools/`)
   - **MCPForecastingToolkit**: MCP forecasting API tools
   - **DEXTradingToolkit**: DEX simulator trading tools
   - **MarketDataToolkit**: Market data retrieval tools
   - **CryptoTools**: Crypto-specific trend and sentiment analysis

4. **Memory System** (`core/memory/`)
   - **CamelMemoryManager**: Long-term memory with Qdrant
   - **QdrantStorageFactory**: Qdrant vector database integration
   - **EmbeddingFactory**: Embedding model configuration

5. **Model Configuration** (`core/models/camel_models.py`)
   - Model factory supporting OpenAI, Gemini, and local models

## Setup

### Dependencies

All dependencies are included in `requirements.txt`:
- `camel-ai>=0.1.0` (includes Workforce and Societies)
- `qdrant-client>=1.7.0`
- `sentence-transformers>=2.2.0`

### Docker Services

The system includes:
- **Qdrant**: Vector database for memory storage (port 6333)
- **Workforce Orchestrator**: New CAMEL-based orchestrator service

### Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: For OpenAI models and embeddings
- `GEMINI_API_KEY`: Optional, for Gemini models
- `MCP_API_KEY`: For MCP forecasting API
- `QDRANT_HOST`: Qdrant host (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)

## Usage

### Starting the Workforce Orchestrator

```bash
# Using Docker Compose
docker-compose up workforce-orchestrator

# Or directly
AGENT_NAME=workforce python -m agents.runner
```

### Worker Capabilities

| Worker | Can Handle Multiple Tasks | Unique Capabilities |
|--------|--------------------------|-------------------|
| DQN Worker | Yes | DQN prediction analysis |
| Chart Analysis Worker | No | RSI, MACD, Bollinger Bands |
| Risk Assessment Worker | No | Position sizing, drawdown analysis |
| Market Research Worker | Yes | News, sentiment analysis |
| Trade Execution Worker | No | DEX integration, order management |

## Memory System

The memory system uses:
- **ChatHistoryBlock**: Recent conversation context (last 100 messages)
- **VectorDBBlock**: All trading decisions and outcomes for semantic search
- **Storage**: Qdrant for vectors, in-memory for chat history (Redis integration can be added)

## Testing

Run tests with:
```bash
# Unit tests
pytest tests/unit/test_camel_tools.py
pytest tests/unit/test_memory.py

# Integration tests
pytest tests/integration/test_workforce.py
pytest tests/integration/test_workers.py
```

## Migration

The system supports both orchestrators:
- **Legacy Orchestrator**: `AGENT_NAME=orchestrator`
- **Workforce Orchestrator**: `AGENT_NAME=workforce`

Both can run simultaneously for gradual migration.

## Configuration

CAMEL and memory settings in `core/config.py`:
- `camel_default_model`: Default model for agents
- `camel_coordinator_model`: Model for coordinator
- `camel_task_model`: Model for task decomposition
- `camel_worker_model`: Model for workers
- `qdrant_host`, `qdrant_port`: Qdrant connection
- `memory_token_limit`: Token limit for context
- `memory_retrieve_limit`: Number of records to retrieve

