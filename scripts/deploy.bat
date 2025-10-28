@echo off
REM Production deployment script for Windows

echo Starting production deployment...

REM Check if .env.prod exists
if not exist .env.prod (
    echo Error: .env.prod file not found!
    echo Please copy config/production.env.example to .env.prod and update with your values.
    exit /b 1
)

REM Load environment variables
for /f "usebackq tokens=1,2 delims==" %%a in (.env.prod) do set %%a=%%b

REM Build and start services
echo Building and starting production services...
docker-compose -f docker-compose.prod.yml up -d --build

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 30 /nobreak > nul

REM Check service health
echo Checking service health...
docker-compose -f docker-compose.prod.yml ps

echo Production deployment completed!
echo.
echo Services available at:
echo - Trading API: http://localhost:8000
echo - Grafana: http://localhost:3000 (admin/%GRAFANA_ADMIN_PASSWORD%)
echo - Prometheus: http://localhost:9090
echo - Jaeger: http://localhost:16686
echo.
echo To view logs: docker-compose -f docker-compose.prod.yml logs -f
echo To stop services: docker-compose -f docker-compose.prod.yml down
