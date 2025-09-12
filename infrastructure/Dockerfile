FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    postgresql-client \
    netcat-traditional \
    libxml2-dev \
    libxslt-dev \
    python3-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY config/requirements*.txt ./

RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org && \
    pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

COPY . .

RUN useradd -m -u 1001 voyageur

RUN mkdir -p /app/staticfiles /app/logs /app/Design/media /app/Design/static && \
    chown -R voyageur:voyageur /app

RUN python manage.py collectstatic --noinput

# Create HuggingFace cache directory for model downloads
RUN mkdir -p /home/voyageur/.cache/huggingface && \
    chown -R voyageur:voyageur /home/voyageur/.cache

USER voyageur

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz -o /dev/null

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]