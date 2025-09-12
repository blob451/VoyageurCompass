# VoyageurCompass Docker Infrastructure

This directory contains all Docker-related configuration files for running VoyageurCompass in containerized environments.

## Quick Start

### 1. Prerequisites
- Docker Desktop (or Docker Engine + Docker Compose)
- At least 8GB RAM available for containers
- 20GB free disk space

### 2. Environment Setup
```bash
# Copy environment template (from project root)
cp .env.example .env

# Edit .env file with your configuration
# Key settings:
# - Database credentials
# - Redis configuration  
# - Security keys
```

### 3. Start Services

The system supports different profiles for CPU-only or GPU-enabled deployments:

#### CPU-Only Deployment (Default)
```bash
cd infrastructure

# Option 1: Use convenience script (recommended)
./docker-start.sh

# Option 2: Direct docker-compose (CPU profile)
docker-compose --profile cpu up -d
```

#### GPU-Enabled Deployment
For systems with NVIDIA GPU support:
```bash
cd infrastructure

# GPU profile includes Ollama LLM and GPU-accelerated backend
docker-compose --profile gpu up -d
```

**GPU Prerequisites:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime (`nvidia-docker2`)
- At least 8GB GPU VRAM recommended

### 4. Access Services
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Django Admin**: http://localhost:8000/admin/
- **Ollama LLM**: http://localhost:11434

## Management Scripts

### `docker-start.sh`
- Builds and starts all services
- Shows service status and URLs
- Includes health checks

### `docker-stop.sh`
- Gracefully stops all services
- Preserves data volumes
- Shows cleanup status

### `docker-rebuild.sh`
- Full rebuild with no cache
- Useful after code changes
- Force recreates all containers

## Services Overview

| Service | Purpose | Port | Health Check |
|---------|---------|------|--------------|
| **backend** | Django API server | 8000 | HTTP /admin/login/ |
| **frontend** | React web interface | 3000 | - |
| **db** | PostgreSQL database | 5433 | pg_isready |
| **redis** | Cache & message broker | 6379 | redis-cli ping |
| **celery_worker** | Background tasks | - | celery inspect |
| **celery_beat** | Scheduled tasks | - | Redis connection |
| **ollama** | Local LLM service | 11434 | ollama list |
| **nginx** | Reverse proxy/SSL | 80/443 | - |

## File Structure

```
infrastructure/
├── docker-compose.yml     # Main orchestration file
├── Dockerfile            # Backend application container
├── nginx/               # Nginx reverse proxy configuration
│   ├── Dockerfile
│   └── nginx.conf
├── docker-start.sh      # Convenience startup script
├── docker-stop.sh       # Convenience shutdown script
├── docker-rebuild.sh    # Full rebuild script
└── README.md           # This file
```

## Volumes

Persistent data is stored in Docker volumes:

- `voyageur_postgres_data`: Database files
- `voyageur_redis_data`: Redis persistence  
- `voyageur_static`: Django static files
- `voyageur_media`: User uploaded files
- `voyageur_celerybeat_schedule`: Celery schedules
- `voyageur_ollama_data`: LLM models and cache

## Environment Variables

Key environment variables (set in `../.env`):

### Database
- `DB_NAME`: PostgreSQL database name
- `DB_USER`: Database username  
- `DB_PASSWORD`: Database password
- `DB_HOST`: Database host (default: db)
- `DB_PORT`: Database port (default: 5432)

### Redis
- `REDIS_HOST`: Redis host (default: redis)
- `REDIS_PORT`: Redis port (default: 6379)

### Security
- `SECRET_KEY`: Django secret key
- `DEBUG`: Enable debug mode (False for production)
- `ALLOWED_HOSTS`: Comma-separated allowed hosts

### CORS
- `CORS_ALLOWED_ORIGINS`: Frontend URLs
- `CSRF_TRUSTED_ORIGINS`: Trusted origins for CSRF

### Ollama LLM
- `OLLAMA_HOST`: LLM service host (default: ollama)
- `OLLAMA_PORT`: LLM service port (default: 11434)
- `OLLAMA_MODEL`: Default model name

## Troubleshooting

### Service won't start
```bash
# Check service logs
docker-compose logs [service_name]

# Check service status
docker-compose ps

# Restart specific service
docker-compose restart [service_name]
```

### Database connection issues
```bash
# Check database health
docker-compose exec db pg_isready -U voyageur_user

# Connect to database directly
docker-compose exec db psql -U voyageur_user -d voyageur_compass_db
```

### Build failures
```bash
# Clean rebuild
docker-compose build --no-cache --force-rm

# Remove old images
docker image prune -a
```

### Volume issues
```bash
# List volumes
docker volume ls | grep voyageur

# Remove all data (⚠️  destructive)
docker-compose down -v
```

## Production Deployment

For production use:

1. **Security**: Update all passwords and keys in `.env`
2. **SSL**: Configure nginx certificates
3. **Resources**: Adjust resource limits in docker-compose.yml
4. **Monitoring**: Enable health checks and logging
5. **Backups**: Set up automated database backups

## Network Architecture

```
Internet
    ↓
nginx:80/443
    ↓
backend:8000 ← → redis:6379
    ↓
db:5432

frontend:3000 → backend:8000
celery_worker → redis:6379
celery_beat → redis:6379
ollama:11434 ← backend:8000
```

All services run on the `voyageur-network` bridge network for secure inter-service communication.