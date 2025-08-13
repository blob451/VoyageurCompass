# 🚀 VoyageurCompass Development Setup Guide

## Quick Start (5 minutes)

```bash
# 1. Clone and navigate
git clone <repository-url>
cd VoyageurCompass

# 2. Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Setup environment variables
cp .env.example .env
# Edit .env with your database credentials

# 4. Setup database
python manage.py migrate
python manage.py createsuperuser

# 5. Install pre-commit hooks
pre-commit install

# 6. Run development server
python manage.py runserver
```

## 📋 Prerequisites

### Required Software
- **Python 3.11+** (exactly 3.11 recommended)
- **PostgreSQL 15+** (database)
- **Redis 7+** (caching and Celery)
- **Node.js 18+** (frontend development)
- **Git** (version control)

### Optional Tools
- **Docker & Docker Compose** (containerized development)
- **VS Code** (recommended IDE with extensions)

## 🛠️ Detailed Setup Instructions

### 1. Environment Setup

#### Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Environment Variables
Create a `.env` file in the project root:

```env
# Django Settings
SECRET_KEY=your-very-secure-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
DB_NAME=voyageur_compass_db
DB_USER=voyageur_user
DB_PASSWORD=your_password_here
DB_HOST=localhost
DB_PORT=5432

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# CORS Settings
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# API Settings
YAHOO_FINANCE_API_TIMEOUT=30
DATA_REFRESH_INTERVAL=3600
```

### 2. Database Setup

#### PostgreSQL Installation & Configuration
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# macOS (with Homebrew)
brew install postgresql
brew services start postgresql

# Windows
# Download and install from: https://www.postgresql.org/download/windows/
```

#### Database Creation
```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE voyageur_compass_db;
CREATE USER voyageur_user WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE voyageur_compass_db TO voyageur_user;
ALTER USER voyageur_user CREATEDB;  # For running tests
\q
```

#### Django Database Setup
```bash
# Apply migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Load initial data (optional)
python manage.py loaddata fixtures/initial_data.json
```

### 3. Redis Setup

#### Redis Installation
```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS (with Homebrew)
brew install redis
brew services start redis

# Windows
# Download from: https://github.com/microsoftarchive/redis/releases
```

#### Verify Redis
```bash
redis-cli ping
# Should return: PONG
```

### 4. Frontend Setup

```bash
# Navigate to frontend directory
cd Design/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# In another terminal, run tests
npm run test
```

### 5. Development Tools Setup

#### Pre-commit Hooks
```bash
# Install pre-commit (if not installed with pip)
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks (optional)
pre-commit run --all-files
```

#### VS Code Extensions (Recommended)
- Python (Microsoft)
- Pylance (Microsoft)
- Black Formatter
- isort
- ESLint
- Prettier
- Docker
- GitLens

#### VS Code Settings
Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

## 🐳 Docker Development (Alternative)

### Quick Docker Setup
```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Commands
```bash
# Execute commands in containers
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py createsuperuser
docker-compose exec backend python manage.py shell

# Rebuild specific service
docker-compose build backend

# Access container shell
docker-compose exec backend bash
```

## 📝 Development Workflow

### Daily Development
1. **Start Development Environment**
   ```bash
   # Terminal 1: Backend
   python manage.py runserver
   
   # Terminal 2: Frontend
   cd Design/frontend && npm run dev
   
   # Terminal 3: Celery (if needed)
   celery -A VoyageurCompass worker -l info
   ```

2. **Code Changes**
   - Write code following project conventions
   - Pre-commit hooks will run automatically
   - Fix any issues before committing

3. **Testing**
   ```bash
   # Backend tests
   pytest
   
   # Frontend tests
   cd Design/frontend && npm run test
   
   # Coverage reports
   pytest --cov
   cd Design/frontend && npm run test:coverage
   ```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/your-feature-name
```

## 🧪 Testing

### Backend Testing
```bash
# Run all tests
pytest

# Run specific app tests
pytest Analytics/tests
pytest Data/tests
pytest Core/tests

# Run with coverage
pytest --cov=Analytics --cov=Data --cov=Core

# Run specific test
pytest Data/tests/test_models.py::StockTestCase::test_stock_creation
```

### Frontend Testing
```bash
cd Design/frontend

# Run tests
npm run test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

### Integration Testing
```bash
# Run integration tests
pytest tests/test_integration.py

# Run with specific markers
pytest -m integration
pytest -m "not slow"
```

## 🔧 Common Commands

### Django Management Commands
```bash
# Create new migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic

# Django shell
python manage.py shell

# Check system
python manage.py check

# Custom commands
python manage.py analyze_stock AAPL
python manage.py pull_market_data
```

### Database Commands
```bash
# Reset database
python manage.py flush

# Backup database
pg_dump voyageur_compass_db > backup.sql

# Restore database
psql voyageur_compass_db < backup.sql

# Database shell
python manage.py dbshell
```

### Celery Commands
```bash
# Start worker
celery -A VoyageurCompass worker -l info

# Start beat scheduler
celery -A VoyageurCompass beat -l info

# Monitor tasks
celery -A VoyageurCompass flower

# Purge all tasks
celery -A VoyageurCompass purge
```

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Error
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Check connection
psql -h localhost -U voyageur_user -d voyageur_compass_db
```

#### Redis Connection Error
```bash
# Check Redis service
redis-cli ping

# Check Redis logs
sudo journalctl -u redis
```

#### Python Module Not Found
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

#### Frontend Build Issues
```bash
cd Design/frontend

# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
npm run build -- --clean
```

#### Pre-commit Hook Failures
```bash
# Fix formatting issues
black .
isort .

# Skip hooks (emergency only)
git commit --no-verify
```

### Performance Issues

#### Slow Database Queries
```bash
# Enable query logging in Django settings
LOGGING['loggers']['django.db.backends'] = {
    'level': 'DEBUG',
    'handlers': ['console'],
}
```

#### Memory Usage
```bash
# Monitor memory usage
htop
ps aux | grep python

# Django debug toolbar (add to INSTALLED_APPS in development)
pip install django-debug-toolbar
```

## 📚 Additional Resources

### Documentation
- [Django Documentation](https://docs.djangoproject.com/)
- [React Documentation](https://react.dev/)
- [Redux Toolkit](https://redux-toolkit.js.org/)
- [Celery Documentation](https://docs.celeryproject.org/)

### Project-Specific Docs
- [API Documentation](./API_DOCUMENTATION.md)
- [Database Schema](./DATABASE_SCHEMA.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Contributing Guidelines](./CONTRIBUTING.md)

## 🆘 Getting Help

1. **Check existing documentation** in the `docs/` directory
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Clear problem description
   - Steps to reproduce
   - Environment details
   - Error logs

## 🔄 Keeping Development Environment Updated

### Weekly Maintenance
```bash
# Update dependencies
pip list --outdated
pip install -r requirements.txt --upgrade

cd Design/frontend
npm audit
npm update

# Update pre-commit hooks
pre-commit autoupdate
```

### Monthly Maintenance
```bash
# Update system packages
sudo apt update && sudo apt upgrade  # Linux
brew update && brew upgrade           # macOS

# Review and update .env variables
# Check for new Django security settings
# Review database performance
```