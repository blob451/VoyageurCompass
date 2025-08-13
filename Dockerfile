# ================================
# Stage 1: Dependencies Builder
# ================================
FROM python:3.11-slim as dependencies

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libxml2-dev \
    libxslt-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt && \
    find /root/.local -name "*.pyc" -delete && \
    find /root/.local -type d -name "__pycache__" -exec rm -rf {} +

# ================================
# Stage 2: Runtime Base
# ================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    netcat-traditional \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies from builder stage
COPY --from=dependencies /root/.local /root/.local

# Set work directory
WORKDIR /app

# ================================
# Stage 3: Development
# ================================
FROM base as development

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/staticfiles /app/logs /app/Design/media /app/Design/static

# Set temporary build-time environment variables for collectstatic
ENV SECRET_KEY=build-time-secret-key-for-static-collection-only
ENV DEBUG=false

# Collect static files for WhiteNoise
RUN python manage.py collectstatic --noinput --clear || echo "Static collection skipped in build"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz -o /dev/null || exit 1

# Development server command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# ================================
# Stage 4: Production
# ================================
FROM base as production

# Create non-root user for security
RUN useradd -m -u 1001 voyageur

# Copy project files with proper ownership
COPY --chown=voyageur:voyageur . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/staticfiles /app/logs /app/Design/media /app/Design/static && \
    chown -R voyageur:voyageur /app

# Switch to non-root user
USER voyageur

# Set production environment variables for static collection
ENV SECRET_KEY=build-time-secret-key-for-static-collection-only
ENV DEBUG=false

# Collect static files for production
RUN python manage.py collectstatic --noinput --clear || echo "Static collection skipped in build"

# Expose port
EXPOSE 8000

# Production health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz -o /dev/null || exit 1

# Production server command (can be overridden)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "VoyageurCompass.wsgi:application"]