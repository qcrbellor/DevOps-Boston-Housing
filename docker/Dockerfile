# Multi-stage Docker build for Boston Housing API

# Stage 1: Build stage
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data logs && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Add local Python packages to PATH
ENV PATH=/home/mluser/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]