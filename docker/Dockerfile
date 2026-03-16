FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir \
    langgraph>=1.1.0 \
    langchain-core>=1.2.0 \
    langchain-anthropic>=1.3.0 \
    anthropic>=0.85.0 \
    websockets>=16.0 \
    pydantic>=2.12.0 \
    pydantic-settings>=2.13.0 \
    prometheus-client>=0.24.0 \
    httpx>=0.28.0 \
    structlog>=25.0.0

ENV PYTHONPATH=/app/src

# Expose Prometheus metrics port
EXPOSE 8000

CMD ["python", "-m", "alpha_pipeline.main"]
