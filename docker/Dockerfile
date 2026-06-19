# ==============================================================================
# Autonomous Code Review Agent - Production Dockerfile
# Multi-stage build for minimal image size and security
# ==============================================================================

# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files for installation
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package and dependencies
RUN pip install --no-cache-dir --prefix=/install .


# --- Stage 2: Production image ---
FROM python:3.11-slim AS production

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Create data directories
RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command: run the API server
CMD ["python", "-m", "uvicorn", "code_review_agent.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
