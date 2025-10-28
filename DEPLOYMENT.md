# Production Deployment Guide

This guide covers deploying the Agentic Trading System to production with all observability, security, and performance features enabled.

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- 20GB free disk space
- Valid API keys for external services

## Quick Start

1. **Copy environment template:**
   ```bash
   cp config/production.env.example .env.prod
   ```

2. **Update environment variables:**
   Edit `.env.prod` with your actual values:
   - Database passwords
   - API keys
   - JWT secret key
   - External service URLs

3. **Deploy:**
   ```bash
   # Windows
   scripts/deploy.bat
   
   # Linux/Mac
   chmod +x scripts/deploy.sh
   ./scripts/deploy.sh
   ```

## Architecture

The production deployment includes:

- **Trading API**: Main FastAPI application with 2 replicas
- **Agents**: 7 specialized trading agents
- **Redis**: Caching and pub/sub messaging
- **PostgreSQL**: Persistent data storage
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Jaeger**: Distributed tracing
- **Nginx**: Reverse proxy and load balancing

## Services

### Trading API
- **URL**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Replicas**: 2
- **Resources**: 1GB RAM, 1 CPU per replica

### Monitoring Stack
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: Set in `GRAFANA_ADMIN_PASSWORD`
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

### Databases
- **Redis**: Port 6379 (password protected)
- **PostgreSQL**: Port 5432 (password protected)

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `REDIS_PASSWORD` | Redis password | Yes |
| `MCP_API_KEY` | Forecasting API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `JWT_SECRET_KEY` | JWT signing key (32+ chars) | Yes |
| `SENTRY_DSN` | Sentry error tracking | No |
| `INITIAL_CAPITAL` | Starting capital | No (default: 10000) |
| `MAX_POSITION_SIZE` | Max position size | No (default: 0.20) |

### Resource Limits

Each service has resource limits configured:
- **API**: 1GB RAM, 1 CPU
- **Agents**: 512MB-1GB RAM, 0.5-1 CPU
- **Databases**: 512MB-1GB RAM
- **Monitoring**: 256MB-1GB RAM

## Security

### Network Security
- All services run in isolated Docker network
- No external ports exposed except for API and monitoring
- Internal communication uses service names

### Authentication
- JWT tokens for API authentication
- Password-protected databases
- Rate limiting on all endpoints

### Data Protection
- All sensitive data in environment variables
- No hardcoded secrets
- Encrypted connections between services

## Monitoring

### Metrics
- **Prometheus** collects metrics from all services
- **Grafana** provides pre-configured dashboards
- Custom metrics for trading performance

### Logging
- Structured JSON logs
- Centralized log collection
- Log rotation configured

### Tracing
- **Jaeger** for distributed tracing
- Request flow visualization
- Performance bottleneck identification

## Maintenance

### Viewing Logs
```bash
# All services
docker-compose -f docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker-compose.prod.yml logs -f trading-api
```

### Scaling
```bash
# Scale API replicas
docker-compose -f docker-compose.prod.yml up -d --scale trading-api=3
```

### Updates
```bash
# Pull latest images and restart
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### Backup
```bash
# Backup PostgreSQL
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U trading_user trading_system > backup.sql

# Backup Redis
docker-compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD --rdb /data/dump.rdb
```

## Troubleshooting

### Common Issues

1. **Services not starting**
   - Check resource availability
   - Verify environment variables
   - Check logs: `docker-compose -f docker-compose.prod.yml logs`

2. **API not responding**
   - Check health endpoint: `curl http://localhost:8000/health`
   - Verify all dependencies are healthy
   - Check Nginx configuration

3. **High memory usage**
   - Monitor with Grafana
   - Adjust resource limits
   - Check for memory leaks

### Health Checks

All services have health checks configured:
- **API**: HTTP health endpoint
- **Redis**: `redis-cli ping`
- **PostgreSQL**: `pg_isready`
- **Agents**: Heartbeat monitoring

## Performance Tuning

### Database Optimization
- Connection pooling enabled
- Query optimization
- Index optimization

### Caching
- Redis for session storage
- API response caching
- Database query caching

### Load Balancing
- Nginx load balancing
- Multiple API replicas
- Health check-based routing

## Security Hardening

### Production Checklist
- [ ] Change all default passwords
- [ ] Enable HTTPS with valid certificates
- [ ] Configure firewall rules
- [ ] Set up log monitoring
- [ ] Enable intrusion detection
- [ ] Regular security updates
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan

### SSL/TLS
To enable HTTPS:
1. Obtain SSL certificates
2. Place in `config/ssl/`
3. Uncomment HTTPS server block in `config/nginx.conf`
4. Update environment variables

## Support

For issues and questions:
1. Check logs first
2. Review monitoring dashboards
3. Check service health status
4. Review configuration
5. Contact support team
