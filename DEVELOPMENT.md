# Development Guide

This guide covers setting up the development environment, running tests, and contributing to the Agentic Trading System.

## Prerequisites

- Python 3.11+
- UV package manager
- Docker and Docker Compose
- Git

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agentic-trading-system
```

### 2. Install UV

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Set Up Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Start Development Environment

```bash
# Windows
scripts\dev.bat

# Linux/Mac
chmod +x scripts/dev.sh
./scripts/dev.sh
```

## Project Structure

```
agentic-trading-system/
├── api/                    # FastAPI application
│   └── main.py            # Main API entry point
├── agents/                # Trading agents
│   ├── orchestrator.py    # Main orchestrator
│   ├── dqn_agent.py       # DQN prediction agent
│   ├── chart_agent.py     # Technical analysis agent
│   ├── risk_agent.py      # Risk management agent
│   ├── memory_agent.py    # Memory and performance agent
│   ├── news_agent.py      # News sentiment agent
│   └── copytrade_agent.py # Copy trading agent
├── core/                  # Core functionality
│   ├── config.py          # Configuration management
│   ├── logging.py         # Logging setup
│   ├── redis_client.py    # Redis client
│   ├── exchange_interface.py # Exchange abstraction
│   ├── exchange_manager.py # Exchange management
│   ├── forecasting_client.py # Forecasting API client
│   ├── observability.py   # Monitoring and metrics
│   ├── security.py        # Security utilities
│   └── performance.py     # Performance optimizations
├── core/exchanges/        # Exchange implementations
│   ├── dex_exchange.py    # DEX exchange
│   └── mexc_exchange.py   # MEXC exchange
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── functional/        # Functional tests
│   └── mocks/             # Mock implementations
├── scripts/               # Development scripts
├── config/                # Configuration files
├── docs/                  # Documentation
└── docker/                # Docker configurations
```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest tests/functional/

# Run with coverage
uv run pytest --cov=api --cov=agents --cov=core

# Run specific test file
uv run pytest tests/unit/test_config.py
```

### Linting and Formatting

```bash
# Windows
scripts\lint.bat

# Linux/Mac
chmod +x scripts/lint.sh
./scripts/lint.sh
```

### Building Docker Images

```bash
# Windows
scripts\build.bat

# Linux/Mac
chmod +x scripts/build.sh
./scripts/build.sh
```

## Code Style

### Python

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use `black` for formatting
- Use `isort` for import sorting
- Use `flake8` for linting

### Configuration

- Use Pydantic for settings
- Environment variables for configuration
- No hardcoded secrets
- Use `.env` files for local development

### Error Handling

- Use custom exceptions
- Log errors with context
- Return structured error responses
- Handle async errors properly

## Testing Guidelines

### Unit Tests

- Test individual functions and methods
- Mock external dependencies
- Use `pytest-asyncio` for async tests
- Aim for 90%+ code coverage

### Integration Tests

- Test component interactions
- Use test databases
- Mock external APIs
- Test error scenarios

### Functional Tests

- Test complete workflows
- Use real data when possible
- Test performance characteristics
- Test concurrent operations

## Adding New Features

### 1. Create Feature Branch

```bash
git checkout -b feature/new-feature
```

### 2. Implement Feature

- Follow existing patterns
- Add type hints
- Write tests
- Update documentation

### 3. Test Changes

```bash
uv run pytest
uv run black .
uv run isort .
uv run flake8
```

### 4. Submit Pull Request

- Include description of changes
- Reference any issues
- Ensure CI passes

## Agent Development

### Creating a New Agent

1. **Create agent file** in `agents/` directory
2. **Inherit from base agent class** (if available)
3. **Implement required methods**:
   - `initialize()`
   - `run()`
   - `stop()`
4. **Add to docker-compose.yml**
5. **Write tests**

### Agent Communication

Agents communicate via Redis Pub/Sub:

```python
# Publish message
await redis_client.publish("agent:target", message.dict())

# Subscribe to messages
async for message in redis_client.subscribe("agent:source"):
    # Process message
    pass
```

### Agent Lifecycle

1. **Initialize**: Set up connections and state
2. **Run**: Main processing loop
3. **Stop**: Cleanup and shutdown

## API Development

### Adding New Endpoints

1. **Define route** in `api/main.py`
2. **Add rate limiting** decorator
3. **Add authentication** if needed
4. **Write tests**
5. **Update documentation**

### Request/Response Models

Use Pydantic models for validation:

```python
from pydantic import BaseModel

class TradeRequest(BaseModel):
    ticker: str
    action: str
    quantity: float
    
class TradeResponse(BaseModel):
    trade_id: str
    status: str
    message: str
```

## Database Development

### Migrations

- Use Alembic for database migrations
- Test migrations on sample data
- Backup before applying migrations

### Queries

- Use async database clients
- Implement connection pooling
- Use prepared statements
- Monitor query performance

## Monitoring and Observability

### Logging

Use structured logging with Loguru:

```python
from core.logging import log

log.info("Processing trade", trade_id=trade_id, ticker=ticker)
log.error("Trade failed", error=str(e), trade_id=trade_id)
```

### Metrics

Use Prometheus metrics:

```python
from core.observability import metrics

metrics.trade_counter.labels(action="buy").inc()
metrics.trade_duration.observe(duration)
```

### Tracing

Use OpenTelemetry for distributed tracing:

```python
from core.observability import tracer

with tracer.start_as_current_span("process_trade"):
    # Trade processing code
    pass
```

## Performance Optimization

### Async Programming

- Use `async/await` for I/O operations
- Avoid blocking operations
- Use connection pooling
- Implement proper error handling

### Caching

- Use Redis for caching
- Implement cache invalidation
- Monitor cache hit rates
- Use appropriate TTLs

### Database Optimization

- Use indexes appropriately
- Optimize queries
- Use connection pooling
- Monitor slow queries

## Security Considerations

### Input Validation

- Validate all inputs
- Use Pydantic models
- Sanitize user data
- Implement rate limiting

### Authentication

- Use JWT tokens
- Implement proper session management
- Use secure password hashing
- Implement 2FA when needed

### API Security

- Use HTTPS in production
- Implement CORS properly
- Use security headers
- Monitor for attacks

## Troubleshooting

### Common Issues

1. **Import errors**: Check Python path and dependencies
2. **Database connection**: Verify credentials and network
3. **Redis connection**: Check Redis server status
4. **Agent not starting**: Check logs and dependencies

### Debugging

1. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use debugger**:
   ```python
   import pdb; pdb.set_trace()
   ```

3. **Check logs**:
   ```bash
   docker-compose logs -f service-name
   ```

### Performance Issues

1. **Profile code**:
   ```python
   import cProfile
   cProfile.run('your_function()')
   ```

2. **Monitor resources**:
   ```bash
   docker stats
   ```

3. **Check database queries**:
   ```sql
   EXPLAIN ANALYZE SELECT * FROM trades;
   ```

## Contributing

### Code Review Process

1. **Self-review** your code
2. **Run all tests** and ensure they pass
3. **Check code style** and formatting
4. **Update documentation** if needed
5. **Request review** from team members

### Commit Messages

Use conventional commits:

```
feat: add new trading endpoint
fix: resolve database connection issue
docs: update API documentation
test: add unit tests for risk agent
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Redis Documentation](https://redis.io/documentation)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Pytest Documentation](https://docs.pytest.org/)
