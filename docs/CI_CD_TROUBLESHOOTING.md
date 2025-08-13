# 🚨 CI/CD Pipeline Troubleshooting Guide

## 🎯 Quick Diagnosis

### Pipeline Status Dashboard
Check the comprehensive status in GitHub Actions:
- **Quick Checks**: Basic validation and change detection
- **Backend Tests**: Django test suite with coverage
- **Frontend Tests**: React/Vitest test suite with ESLint
- **Code Quality**: Black, isort, flake8 validation
- **Security Scan**: Bandit and vulnerability checks
- **Integration Tests**: End-to-end API testing
- **Build Test**: Docker container validation

### Common Pipeline Failures

#### ❌ Code Quality Failures
**Symptoms**: Black, isort, or flake8 jobs failing
```bash
# Local diagnosis
python -m black --check --diff .
python -m isort --check-only --diff .
python -m flake8 . --count --statistics
```

**Solutions**:
```bash
# Fix automatically
python -m black .
python -m isort .

# Manual fixes for flake8 issues
python -m flake8 . --show-source
```

#### ❌ Frontend Test Failures
**Symptoms**: Frontend tests timing out or failing
```bash
# Local diagnosis
cd Design/frontend
npm run test
npm run lint
```

**Common Issues**:
- **RTK Query Mocking**: Ensure proper mock setup for API calls
- **Component Props**: Verify all required props are provided in tests
- **State Management**: Check Redux store configuration in tests

#### ❌ Backend Test Failures
**Symptoms**: Django tests failing or timing out
```bash
# Local diagnosis
python -m pytest -v --tb=short
python manage.py check
```

**Common Issues**:
- **Database Connections**: Ensure test database is accessible
- **Missing Migrations**: Run `python manage.py makemigrations`
- **Environment Variables**: Check test environment configuration

## 🔍 Detailed Troubleshooting

### Pipeline Job Analysis

#### 1. Quick-Checks Job
**Purpose**: Fast validation to prevent unnecessary resource usage
**Timing**: <2 minutes

**Failure Scenarios**:
```yaml
# Change detection not working
- name: Check for relevant changes
  # Fix: Ensure proper file patterns in workflow
```

**Debug Steps**:
1. Check change detection logic
2. Verify Python syntax validation
3. Ensure Django imports work correctly

#### 2. Code Quality Matrix Jobs
**Purpose**: Parallel validation of code formatting and style
**Timing**: <3 minutes each

**Black Formatting Issues**:
```bash
# Identify problematic files
python -m black --check --diff . 2>&1 | grep "would reformat"

# Fix specific files
python -m black path/to/file.py

# Verify configuration
cat pyproject.toml | grep -A 10 "\[tool\.black\]"
```

**isort Import Issues**:
```bash
# Check import order
python -m isort --check-only --diff .

# Fix automatically
python -m isort .

# Custom configuration check
cat pyproject.toml | grep -A 15 "\[tool\.isort\]"
```

**flake8 Linting Issues**:
```bash
# Get detailed error report
python -m flake8 . --show-source --statistics

# Check configuration
cat pyproject.toml | grep -A 10 "\[tool\.flake8\]"
```

#### 3. Backend Tests Job
**Purpose**: Comprehensive Django application testing
**Timing**: <5 minutes

**Database Issues**:
```yaml
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: postgres
      POSTGRES_USER: postgres
      POSTGRES_DB: test_voyageur_compass
```

**Debug Steps**:
```bash
# Test database connection locally
python manage.py check --database default

# Run specific failing test
python -m pytest path/to/test.py::TestClass::test_method -v

# Check test settings
python manage.py check --settings=VoyageurCompass.settings --verbosity=2
```

**Coverage Issues**:
```bash
# Generate coverage report locally
python -m pytest --cov=Data --cov=Analytics --cov=Core --cov-report=html

# Check coverage configuration
cat pyproject.toml | grep -A 10 "\[tool\.coverage\]"
```

#### 4. Frontend Tests Job
**Purpose**: React application testing and linting
**Timing**: <4 minutes

**Node.js/npm Issues**:
```yaml
- name: Set up Node.js 18
  uses: actions/setup-node@v4
  with:
    node-version: '18'
    cache: 'npm'
    cache-dependency-path: Design/frontend/package-lock.json
```

**Test Failures**:
```bash
cd Design/frontend

# Run tests with detailed output
npm run test -- --reporter=verbose

# Check for async issues
npm run test -- --timeout=10000

# Validate test setup
cat vitest.config.js
```

**ESLint Issues**:
```bash
cd Design/frontend

# Run linter with fix
npm run lint -- --fix

# Check configuration
cat eslint.config.js
```

#### 5. Security Scan Job
**Purpose**: Security vulnerability detection
**Timing**: <3 minutes

**Bandit Issues**:
```bash
# Run security scan locally
python -m bandit -r . -x tests/,Design/frontend/

# Check for hardcoded secrets
python -m bandit -r . -ll -i

# Skip false positives
# Use # nosec comment for known false positives
```

**Safety Issues**:
```bash
# Check for vulnerable dependencies
pip install safety
safety check

# Update vulnerable packages
pip install --upgrade package_name
```

#### 6. Integration Tests Job
**Purpose**: End-to-end API testing
**Timing**: <3 minutes

**API Endpoint Issues**:
```bash
# Test API endpoints locally
python manage.py runserver &
curl -X GET http://localhost:8000/api/health/

# Check URL routing
python manage.py show_urls
```

**Authentication Issues**:
```bash
# Test JWT token generation
python manage.py shell
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
user = User.objects.first()
token = RefreshToken.for_user(user)
print(token.access_token)
```

#### 7. Build Test Job
**Purpose**: Docker container validation
**Timing**: <5 minutes

**Docker Build Issues**:
```bash
# Test Docker build locally
docker build -t voyageur-test .

# Check for common issues
docker run --rm voyageur-test python manage.py check

# Verify health checks
docker run -d --name test-container voyageur-test
docker exec test-container curl -f http://localhost:8000/healthz
docker rm -f test-container
```

## ⚡ Performance Optimization

### Pipeline Speed Improvements

#### 1. Caching Strategy
```yaml
# Effective caching configuration
- name: Cache pip dependencies
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

#### 2. Matrix Parallelization
```yaml
# Run tests in parallel
strategy:
  matrix:
    test-type: [lint, test]
  fail-fast: true
```

#### 3. Conditional Execution
```yaml
# Skip unnecessary jobs
if: needs.quick-checks.outputs.should-run-tests == 'true'
```

### Resource Usage Monitoring
```bash
# Check runner resource usage in logs
grep "CPU\|Memory\|Disk" /home/runner/work/_temp/_runner_file_commands/
```

## 🛠️ Local Development Debugging

### Pre-commit Hook Issues
```bash
# Install pre-commit hooks
pre-commit install

# Test all hooks
pre-commit run --all-files

# Skip specific hooks for testing
SKIP=bandit git commit -m "test commit"

# Update hook versions
pre-commit autoupdate
```

### Environment Consistency
```bash
# Match CI Python version
pyenv install 3.11.0
pyenv local 3.11.0

# Match CI Node version
nvm install 18
nvm use 18

# Verify tool versions
python --version
node --version
npm --version
```

### Database Testing
```bash
# Use same PostgreSQL version as CI
docker run -d \
  --name postgres-test \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=test_voyageur_compass \
  -p 5432:5432 \
  postgres:15

# Test connection
python manage.py check --database default
```

## 📊 Monitoring & Alerts

### Pipeline Health Metrics
Monitor these key indicators:
- **Success Rate**: Target >95% over 30 days
- **Duration**: Target <12 minutes total pipeline time
- **Cache Hit Rate**: Target >80% for dependencies
- **Test Coverage**: Maintain >38% backend coverage

### Performance Baselines
| Job | Target Time | Current Baseline |
|-----|-------------|------------------|
| Quick Checks | <2 min | ~1.5 min |
| Backend Tests | <5 min | ~4 min |
| Frontend Tests | <4 min | ~3 min |
| Code Quality | <3 min each | ~2 min each |
| Security Scan | <3 min | ~2 min |
| Integration Tests | <3 min | ~2.5 min |
| Build Test | <5 min | ~4 min |

### Status Notifications
```yaml
# Comprehensive status reporting
- name: Generate comprehensive status report
  run: |
    echo "📊 Job Results:"
    echo "  Backend Tests: ${{ needs.backend-tests.result }}"
    echo "  Frontend Tests: ${{ needs.frontend-tests.result }}"
    echo "  Code Quality: ${{ needs.code-quality.result }}"
```

## 🚑 Emergency Procedures

### Critical Pipeline Failure
1. **Immediate Response**:
   ```bash
   # Check status report for root cause
   # Review failed job logs
   # Identify if it's a code issue or infrastructure problem
   ```

2. **Temporary Workarounds**:
   ```bash
   # Skip specific checks if needed (emergency only)
   git commit -m "fix: emergency fix" --no-verify
   
   # Bypass specific CI jobs (modify workflow temporarily)
   if: false  # Add to failing job
   ```

3. **Rollback Procedures**:
   ```bash
   # Revert to last known good commit
   git revert <commit-hash>
   
   # Cherry-pick specific fixes
   git cherry-pick <fix-commit-hash>
   ```

### Infrastructure Issues
1. **GitHub Actions Outage**:
   - Monitor [GitHub Status](https://www.githubstatus.com/)
   - Use local testing workflows
   - Implement manual deployment procedures

2. **Dependency Issues**:
   ```bash
   # Pin exact versions in requirements.txt
   pip freeze > requirements.txt
   
   # Lock frontend dependencies
   cd Design/frontend && npm ci
   ```

## 📝 Best Practices

### Commit Message Standards
```bash
# Good commit messages
feat: add user authentication endpoint
fix: resolve database connection timeout
docs: update API documentation
test: add integration tests for portfolio API

# Avoid
fixed bug
update
changes
```

### Branch Naming
```bash
# Feature branches
feature/user-authentication
feature/portfolio-analytics

# Bug fixes
fix/database-connection-error
fix/frontend-login-issue

# Documentation
docs/api-documentation
docs/deployment-guide
```

### Testing Strategies
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete user workflows
4. **Performance Tests**: Monitor response times and resource usage

### Security Considerations
- Never commit secrets or API keys
- Use environment variables for sensitive data
- Regularly update dependencies for security patches
- Monitor for security vulnerabilities in CI/CD

## 📞 Getting Help

### Internal Resources
1. **Documentation**: Check `docs/` directory
2. **Issue Tracking**: Create GitHub issues with labels
3. **Code Reviews**: Request help in PR reviews

### External Resources
1. **GitHub Actions Documentation**: https://docs.github.com/en/actions
2. **Django Testing**: https://docs.djangoproject.com/en/stable/topics/testing/
3. **React Testing Library**: https://testing-library.com/docs/react-testing-library/intro/

### Escalation Path
1. **Level 1**: Check documentation and logs
2. **Level 2**: Create GitHub issue with details
3. **Level 3**: Request code review or pair programming
4. **Level 4**: Architecture review for complex issues