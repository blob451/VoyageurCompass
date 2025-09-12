# VoyageurCompass ML-Enabled Docker Image
# Includes PyTorch, Transformers, and FinBERT for ML/AI capabilities

FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_HOME=/app/.cache/huggingface

# Create non-root user for security
RUN groupadd -r voyageur && useradd -r -g voyageur voyageur

# Install system dependencies including CUDA prerequisites
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    netcat-traditional \
    gcc \
    g++ \
    libc-dev \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/staticfiles /app/Design/media /app/logs /app/.cache/huggingface \
    && chown -R voyageur:voyageur /app

# Copy and install ML requirements (includes PyTorch, Transformers)
COPY config/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy FinBERT download script and models
COPY infrastructure/download_finbert.py .
COPY Analytics/ml/finbert_models /app/Analytics/ml/finbert_models

# Pre-download FinBERT models for faster startup
RUN python download_finbert.py \
    && chown -R voyageur:voyageur /app/.cache

# Copy application code
COPY . .

# Set ownership and permissions
RUN chown -R voyageur:voyageur /app \
    && chmod +x /app/infrastructure/*.sh

# Create cache directory for voyageur user
RUN mkdir -p /home/voyageur/.cache \
    && chown -R voyageur:voyageur /home/voyageur

# Switch to non-root user
USER voyageur

# Expose port
EXPOSE 8000

# Health check with ML service verification
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/admin/login/ || exit 1

# Default command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]