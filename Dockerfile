# Use official Python runtime as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    netcat-traditional \
    libxml2-dev \
    libxslt-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories for static files and logs
RUN mkdir -p /app/staticfiles /app/logs /app/Design/media /app/Design/static

# Create a non-root user to run the application
RUN useradd -m -u 1001 voyageur && \
    chown -R voyageur:voyageur /app

# Switch to non-root user
USER voyageur

# Collect static files for WhiteNoise to serve
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]