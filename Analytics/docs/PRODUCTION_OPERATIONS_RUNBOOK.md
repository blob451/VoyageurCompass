# VoyageurCompass Multilingual LLM System - Production Operations Runbook

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Production Monitoring](#production-monitoring)
4. [Health Checks](#health-checks)
5. [Feature Flags Management](#feature-flags-management)
6. [Circuit Breaker Operations](#circuit-breaker-operations)
7. [Emergency Procedures](#emergency-procedures)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Performance Optimization](#performance-optimization)
10. [Maintenance Procedures](#maintenance-procedures)
11. [Logging and Monitoring](#logging-and-monitoring)
12. [Scaling Operations](#scaling-operations)

## System Overview

The VoyageurCompass Multilingual LLM System provides financial analysis and explanations in multiple languages (English, French, Spanish) with production-ready features including:

- **Feature Flags**: Gradual rollout and emergency controls
- **Circuit Breakers**: Automatic failure protection
- **Production Monitoring**: Real-time health and performance tracking
- **Health Check Endpoints**: Load balancer and monitoring integration
- **Comprehensive Testing**: Production readiness validation

### Key Components

1. **Local LLM Service**: Core multilingual explanation generation
2. **Feature Flags Service**: Runtime configuration and rollout control
3. **Circuit Breaker Service**: Failure protection and recovery
4. **Production Monitoring Service**: Health monitoring and alerting
5. **Multilingual Metrics Service**: Performance and quality tracking

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                Django Application                           │
├─────────────────────────────────────────────────────────────┤
│  Feature Flags │ Circuit Breaker │ Production Monitoring   │
├─────────────────────────────────────────────────────────────┤
│           Multilingual LLM Service Layer                   │
├─────────────────────────────────────────────────────────────┤
│     Local LLM │ Translation │ Cache │ Metrics Collection   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│              External Dependencies                          │
│    Redis Cache │ Database │ LLM Models │ Monitoring APIs   │
└─────────────────────────────────────────────────────────────┘
```

## Production Monitoring

### Starting Monitoring

```bash
# Start production monitoring service
python manage.py start_production_monitoring

# Start in daemon mode (keeps running)
python manage.py start_production_monitoring --daemon

# Check monitoring status
python manage.py start_production_monitoring --status
```

### Monitoring Endpoints

| Endpoint | Purpose | Authentication |
|----------|---------|----------------|
| `/analytics/health/ping/` | Load balancer health check | None |
| `/analytics/health/multilingual/` | Comprehensive health status | None |
| `/analytics/monitoring/status/` | Monitoring service status | Required |
| `/analytics/monitoring/alerts/` | Recent alerts | Required |

### Alert Thresholds

| Metric | Warning | Error | Critical |
|--------|---------|-------|----------|
| CPU Usage | 80% | 90% | 95% |
| Memory Usage | 85% | 90% | 95% |
| Disk Usage | 90% | 95% | 98% |
| Response Time | 5s | 10s | 15s |
| Error Rate | 5% | 10% | 20% |

## Health Checks

### Comprehensive Health Check

```bash
# Run full health check
python manage.py multilingual_health_check

# Verbose output with details
python manage.py multilingual_health_check --verbose

# JSON output for automation
python manage.py multilingual_health_check --json

# Attempt automatic fixes
python manage.py multilingual_health_check --fix
```

### Health Check Components

1. **Feature Flags Health**
   - Emergency fallback status
   - Core flag states
   - Rollout percentages

2. **Circuit Breaker Health**
   - Circuit states per language
   - Failure counts
   - Recovery status

3. **Cache System Health**
   - Connectivity test
   - Performance timing
   - Read/write validation

4. **Multilingual Metrics Health**
   - Error rates
   - Response times
   - Quality scores

5. **Production Monitoring Health**
   - Service status
   - Background thread status
   - Recent alert counts

### Quick Health Check Commands

```bash
# Quick ping check
curl http://localhost:8000/analytics/health/ping/

# Detailed multilingual health
curl http://localhost:8000/analytics/health/multilingual/

# Feature flags status
curl http://localhost:8000/analytics/health/feature-flags/
```

## Feature Flags Management

### Listing Feature Flags

```bash
# List all feature flags
python manage.py manage_feature_flags --list

# List with JSON output
python manage.py manage_feature_flags --list --json

# List for specific user
python manage.py manage_feature_flags --list --user username
```

### Managing Individual Flags

```bash
# Enable a feature flag
python manage.py manage_feature_flags --enable multilingual_llm_enabled

# Disable a feature flag
python manage.py manage_feature_flags --disable french_generation_enabled

# Set rollout percentage (gradual deployment)
python manage.py manage_feature_flags --rollout french_generation_enabled 50
```

### Emergency Operations

```bash
# Emergency disable all multilingual features
python manage.py manage_feature_flags --emergency-disable

# Clear feature flags cache
python manage.py manage_feature_flags --clear-cache
```

### Core Feature Flags

| Flag Name | Purpose | Safe to Disable |
|-----------|---------|------------------|
| `multilingual_llm_enabled` | Master multilingual switch | Yes |
| `french_generation_enabled` | French language support | Yes |
| `spanish_generation_enabled` | Spanish language support | Yes |
| `direct_generation_enabled` | Direct LLM generation | No* |
| `translation_pipeline_enabled` | Translation fallback | No* |
| `emergency_fallback_enabled` | Emergency mode | System managed |

*Disabling both direct generation and translation pipeline will break multilingual functionality.

## Circuit Breaker Operations

### Checking Circuit Status

```bash
# Show all circuit breaker status
python manage.py manage_circuit_breakers --status

# Detailed statistics
python manage.py manage_circuit_breakers --stats

# JSON output
python manage.py manage_circuit_breakers --stats --json
```

### Manual Circuit Operations

```bash
# Manually open a circuit
python manage.py manage_circuit_breakers --open multilingual fr

# Manually close a circuit
python manage.py manage_circuit_breakers --close multilingual fr

# Reset specific breaker
python manage.py manage_circuit_breakers --reset multilingual

# Reset all breakers
python manage.py manage_circuit_breakers --reset-all
```

### Circuit States

| State | Description | Action Required |
|-------|-------------|-----------------|
| **CLOSED** | Normal operation | None |
| **HALF_OPEN** | Testing recovery | Monitor closely |
| **OPEN** | Service protection active | Investigate root cause |

### Circuit Breaker Thresholds

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Failure Threshold | 5 | Failures before opening |
| Recovery Timeout | 60s | Wait before half-open |
| Success Threshold | 2 | Successes to close from half-open |
| Half-Open Max Calls | 3 | Max calls in half-open state |

## Emergency Procedures

### System-Wide Emergency

1. **Immediate Response**
   ```bash
   # Emergency disable all multilingual features
   python manage.py manage_feature_flags --emergency-disable

   # Reset all circuit breakers
   python manage.py manage_circuit_breakers --reset-all

   # Force health check
   python manage.py multilingual_health_check --fix
   ```

2. **Verify Emergency State**
   ```bash
   # Check emergency status
   curl http://localhost:8000/analytics/health/multilingual/
   ```

3. **Recovery Steps**
   - Identify root cause
   - Fix underlying issue
   - Re-enable features gradually
   - Monitor system health

### High Error Rate Response

1. **Check Circuit Breakers**
   ```bash
   python manage.py manage_circuit_breakers --status
   ```

2. **Review Recent Alerts**
   ```bash
   python manage.py start_production_monitoring --alerts 10
   ```

3. **Check System Resources**
   ```bash
   python manage.py multilingual_health_check --verbose
   ```

4. **Gradual Recovery**
   ```bash
   # Start with 10% traffic
   python manage.py manage_feature_flags --rollout french_generation_enabled 10

   # Monitor and increase gradually
   python manage.py manage_feature_flags --rollout french_generation_enabled 25
   python manage.py manage_feature_flags --rollout french_generation_enabled 50
   python manage.py manage_feature_flags --rollout french_generation_enabled 100
   ```

### Memory/Resource Issues

1. **Check System Resources**
   ```bash
   # Check memory and CPU usage
   htop

   # Check disk space
   df -h

   # Check Django processes
   ps aux | grep python
   ```

2. **Restart Services if Needed**
   ```bash
   # Restart web server
   systemctl restart gunicorn

   # Restart monitoring
   python manage.py start_production_monitoring --restart
   ```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. High Response Times

**Symptoms:**
- Slow API responses
- High average response time in metrics
- User complaints about performance

**Diagnosis:**
```bash
# Check current metrics
curl http://localhost:8000/analytics/health/multilingual-metrics/

# Check circuit breaker status
python manage.py manage_circuit_breakers --status

# Check system resources
python manage.py multilingual_health_check --verbose
```

**Solutions:**
1. Check CPU/memory usage
2. Verify cache hit rates
3. Consider reducing LLM batch sizes
4. Scale up resources if needed

#### 2. Circuit Breakers Opening Frequently

**Symptoms:**
- Multiple circuits in OPEN state
- Fallback responses being returned
- Service degradation alerts

**Diagnosis:**
```bash
# Check detailed circuit statistics
python manage.py manage_circuit_breakers --stats

# Review recent failures
python manage.py start_production_monitoring --alerts 20
```

**Solutions:**
1. Identify failing language/service
2. Check underlying service health
3. Adjust circuit breaker thresholds if needed
4. Investigate model performance issues

#### 3. Cache Performance Issues

**Symptoms:**
- High cache miss rates
- Slow response times
- Cache connectivity errors

**Diagnosis:**
```bash
# Test cache connectivity
python manage.py shell
>>> from django.core.cache import cache
>>> cache.set('test', 'value', 30)
>>> cache.get('test')

# Check Redis status
redis-cli ping
```

**Solutions:**
1. Restart Redis if needed
2. Check Redis memory usage
3. Clear cache if corrupted
4. Verify Redis configuration

#### 4. Feature Flag Issues

**Symptoms:**
- Features not enabling/disabling as expected
- Emergency fallback stuck enabled
- Rollout percentages not working

**Diagnosis:**
```bash
# Check current flag status
python manage.py manage_feature_flags --list

# Clear cache and retry
python manage.py manage_feature_flags --clear-cache
```

**Solutions:**
1. Clear feature flags cache
2. Verify database connectivity
3. Check for conflicting settings
4. Restart application if needed

### Log Analysis

#### Important Log Locations

```bash
# Django application logs
tail -f /var/log/django/voyageurcompass.log

# Production monitoring logs
tail -f /var/log/django/monitoring.log

# Circuit breaker logs
grep "Circuit breaker" /var/log/django/voyageurcompass.log

# Feature flag logs
grep "Feature flag" /var/log/django/voyageurcompass.log
```

#### Key Log Patterns

| Pattern | Severity | Action |
|---------|----------|--------|
| `Circuit breaker OPENED` | ERROR | Investigate service failure |
| `Emergency fallback enabled` | CRITICAL | Emergency response |
| `High response time detected` | WARNING | Performance investigation |
| `Cache system failure` | ERROR | Check Redis/cache |
| `Model generation failed` | ERROR | Check LLM service |

## Performance Optimization

### Monitoring Performance

```bash
# Get current performance metrics
curl http://localhost:8000/analytics/health/multilingual-metrics/

# Check cache hit rates
curl http://localhost:8000/analytics/health/metrics/
```

### Performance Tuning

#### 1. Cache Optimization

```python
# In Django settings.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            }
        },
        'KEY_PREFIX': 'voyageur',
        'TIMEOUT': 300,  # 5 minutes default
    }
}
```

#### 2. Thread Pool Optimization

```python
# In Django settings.py
MULTILINGUAL_POOL_SIZE_FR = 3
MULTILINGUAL_POOL_SIZE_ES = 3
MULTILINGUAL_POOL_SIZE_EN = 2
```

#### 3. Circuit Breaker Tuning

```python
# In Django settings.py
MULTILINGUAL_FAILURE_THRESHOLD = 5
MULTILINGUAL_RECOVERY_TIMEOUT = 60
MULTILINGUAL_SUCCESS_THRESHOLD = 2
```

### Performance Targets

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Response Time (avg) | < 2s | < 5s | > 10s |
| Error Rate | < 1% | < 5% | > 10% |
| Cache Hit Rate | > 80% | > 60% | < 40% |
| CPU Usage | < 70% | < 85% | > 90% |
| Memory Usage | < 80% | < 85% | > 90% |

## Maintenance Procedures

### Scheduled Maintenance

#### 1. Weekly Health Check

```bash
#!/bin/bash
# weekly_maintenance.sh

echo "=== Weekly VoyageurCompass Maintenance ==="
echo "Date: $(date)"

# Full health check
python manage.py multilingual_health_check --verbose

# Feature flags status
python manage.py manage_feature_flags --list

# Circuit breaker status
python manage.py manage_circuit_breakers --stats

# Recent alerts review
python manage.py start_production_monitoring --alerts 50

echo "=== Maintenance Complete ==="
```

#### 2. Cache Maintenance

```bash
# Clear expired cache entries (if needed)
python manage.py shell -c "from django.core.cache import cache; cache.clear()"

# Restart Redis (if needed)
systemctl restart redis
```

#### 3. Log Rotation

```bash
# Rotate Django logs
logrotate /etc/logrotate.d/django

# Clean old monitoring data
python manage.py shell -c "
from Analytics.services.multilingual_metrics import get_multilingual_metrics
metrics = get_multilingual_metrics()
metrics.cleanup_old_metrics()
"
```

### Database Maintenance

```bash
# Django migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput

# Check for missing migrations
python manage.py makemigrations --dry-run
```

### Model Updates

When updating LLM models:

1. **Test in development first**
2. **Enable feature flag for gradual rollout**
3. **Monitor error rates and quality scores**
4. **Have rollback plan ready**

```bash
# Example gradual model deployment
python manage.py manage_feature_flags --rollout french_generation_enabled 10
# Monitor for 1 hour
python manage.py manage_feature_flags --rollout french_generation_enabled 50
# Monitor for 2 hours
python manage.py manage_feature_flags --rollout french_generation_enabled 100
```

## Logging and Monitoring

### Log Levels

| Level | When to Use | Retention |
|-------|-------------|-----------|
| DEBUG | Development only | N/A |
| INFO | Normal operations | 30 days |
| WARNING | Potential issues | 60 days |
| ERROR | Errors that need attention | 90 days |
| CRITICAL | System-wide emergencies | 1 year |

### Monitoring Metrics

#### System Metrics
- CPU usage
- Memory usage
- Disk usage
- Network I/O

#### Application Metrics
- Response times
- Error rates
- Request volumes
- Cache hit rates

#### Business Metrics
- Multilingual usage by language
- Quality scores
- User satisfaction metrics
- Feature adoption rates

### Alert Configuration

```python
# In Django settings.py
PRODUCTION_MONITORING_ENABLED = True
EMAIL_ALERTS_ENABLED = True
ALERT_EMAIL_RECIPIENTS = ['ops@company.com', 'dev@company.com']

# Slack webhook (optional)
WEBHOOK_ALERTS_ENABLED = True
ALERT_WEBHOOK_URL = 'https://hooks.slack.com/services/...'

# Alert thresholds
CPU_ALERT_THRESHOLD = 80
MEMORY_ALERT_THRESHOLD = 85
DISK_ALERT_THRESHOLD = 90
RESPONSE_TIME_THRESHOLD = 10
ERROR_RATE_THRESHOLD = 0.1
```

## Scaling Operations

### Horizontal Scaling

1. **Load Balancer Configuration**
   - Add new application instances
   - Update health check endpoints
   - Configure session affinity if needed

2. **Database Considerations**
   - Connection pooling
   - Read replicas for metrics
   - Caching strategy

3. **Cache Scaling**
   - Redis clustering
   - Cache partitioning
   - Consistent hashing

### Vertical Scaling

1. **CPU Scaling**
   - Monitor CPU usage patterns
   - Increase core count
   - Adjust thread pool sizes

2. **Memory Scaling**
   - Monitor memory usage
   - Increase RAM
   - Optimize cache sizes

3. **Storage Scaling**
   - Monitor disk usage
   - Increase storage capacity
   - Implement log rotation

### Auto-Scaling Triggers

| Metric | Scale Up | Scale Down |
|--------|----------|------------|
| CPU Usage | > 70% for 5 min | < 30% for 15 min |
| Memory Usage | > 80% for 5 min | < 40% for 15 min |
| Response Time | > 5s for 3 min | < 2s for 10 min |
| Error Rate | > 5% for 2 min | < 1% for 10 min |

---

## Quick Reference

### Emergency Commands
```bash
# Emergency stop all multilingual features
python manage.py manage_feature_flags --emergency-disable

# Reset all circuit breakers
python manage.py manage_circuit_breakers --reset-all

# Force health check with fixes
python manage.py multilingual_health_check --fix
```

### Status Commands
```bash
# System health
python manage.py multilingual_health_check

# Feature flags
python manage.py manage_feature_flags --list

# Circuit breakers
python manage.py manage_circuit_breakers --status

# Monitoring
python manage.py start_production_monitoring --status
```

### Important Endpoints
- Health: `http://localhost:8000/analytics/health/ping/`
- Metrics: `http://localhost:8000/analytics/metrics/`
- Admin: `http://localhost:8000/admin/`

### Support Contacts
- Development Team: dev@company.com
- Operations Team: ops@company.com
- Emergency Hotline: +1-XXX-XXX-XXXX

---

*Last Updated: January 2025*
*Version: 1.0*