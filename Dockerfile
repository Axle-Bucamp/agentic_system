# Use uv base image with Python pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

# Set working directory
WORKDIR /app

# Environment for reliable installs
ENV UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (minimal, build tools are preinstalled in uv image)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements-base.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application using uv run for better dependency management
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

