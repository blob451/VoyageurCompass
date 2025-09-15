# LLM System Troubleshooting Guide

## Quick Diagnosis Checklist

When experiencing LLM system issues, follow this checklist:

- [ ] Check Ollama container status: `docker ps | grep ollama`
- [ ] Verify model availability: `docker exec voyageur-ollama ollama list`
- [ ] Test connectivity: `curl http://localhost:11434/api/tags`
- [ ] Check monitoring dashboard: `python manage.py llm_monitor_dashboard`
- [ ] Review recent logs: `tail -f logs/llm_operations.log`

## Common Issues and Solutions

### 1. Service Unavailable Errors

**Symptoms:**
- "LLM service not available" errors
- `generate_explanation()` returns None
- Connection refused errors

**Diagnosis:**
```bash
# Check Ollama container
docker ps | grep ollama
STATUS: Exited or not running

# Check port accessibility
telnet localhost 11434
CONNECTION: Refused
```

**Solutions:**

**A. Container Not Running:**
```bash
# Restart Ollama container
docker restart voyageur-ollama

# If container doesn't exist, recreate
docker-compose up -d ollama
```

**B. Port Issues:**
```bash
# Check port binding
docker port voyageur-ollama
# Should show: 11434/tcp -> 0.0.0.0:11434

# Check firewall (Linux)
sudo ufw status | grep 11434

# Check firewall (Windows)
netstat -an | findstr :11434
```

**C. Resource Issues:**
```bash
# Check system resources
docker stats voyageur-ollama

# Check disk space
df -h | grep docker

# Check memory usage
free -h
```

### 2. Model Not Found Errors

**Symptoms:**
- "Model not available" errors
- Empty model list
- Specific model failures

**Diagnosis:**
```bash
# List installed models
docker exec voyageur-ollama ollama list

# Expected models:
# phi3:3.8b
# llama3.1:8b
# qwen2
```

**Solutions:**

**A. Missing Models:**
```bash
# Install missing models
docker exec voyageur-ollama ollama pull phi3:3.8b
docker exec voyageur-ollama ollama pull llama3.1:8b
docker exec voyageur-ollama ollama pull qwen2

# Verify installation
docker exec voyageur-ollama ollama list
```

**B. Model Loading Issues:**
```bash
# Test model individually
docker exec voyageur-ollama ollama run phi3:3.8b "Hello"

# Check model status
docker exec voyageur-ollama ollama ps
```

**C. Storage Issues:**
```bash
# Check Ollama storage
docker exec voyageur-ollama du -sh ~/.ollama

# Clean unused models
docker exec voyageur-ollama ollama rm unused_model:tag
```

### 3. Slow Response Times

**Symptoms:**
- Response times >30 seconds
- Request timeouts
- Performance degradation

**Diagnosis:**
```bash
# Run benchmark test
python manage.py benchmark_llm --iterations 3

# Check current performance
python manage.py llm_monitor_dashboard | grep -A10 "Response Time"

# Monitor system resources
htop  # or Task Manager on Windows
```

**Solutions:**

**A. Cold Start Issues:**
```bash
# Warm up models
python manage.py warm_cache --llm-only

# Verify models are loaded
docker exec voyageur-ollama ollama ps
```

**B. Resource Constraints:**
```bash
# Check GPU usage (if available)
nvidia-smi

# Adjust GPU memory fraction
# In .env: OLLAMA_GPU_MEMORY_FRACTION=0.6

# Reduce concurrent models
# In .env: OLLAMA_MAX_LOADED_MODELS=2
```

**C. System Overload:**
```bash
# Check system load
uptime

# Identify resource-heavy processes
ps aux --sort=-%cpu | head -10

# Restart services if needed
docker-compose restart
```

### 4. Quality Issues

**Symptoms:**
- Incoherent explanations
- Incomplete responses
- Incorrect recommendations

**Diagnosis:**
```bash
# Check quality metrics
python manage.py llm_monitor_dashboard | grep -A5 "Quality"

# Run financial accuracy tests
python -m pytest Analytics/tests/test_financial_accuracy.py::FinancialAccuracyValidationTestCase::test_technical_score_recommendation_alignment -v
```

**Solutions:**

**A. Model Issues:**
```bash
# Test individual models
python manage.py benchmark_llm --detail-levels summary --iterations 2
python manage.py benchmark_llm --detail-levels standard --iterations 2
python manage.py benchmark_llm --detail-levels detailed --iterations 2

# Check for corrupted models
docker exec voyageur-ollama ollama list
# Look for incomplete downloads (size mismatches)
```

**B. Prompt Issues:**
```bash
# Check prompt templates
python manage.py shell
>>> from Analytics.services.local_llm_service import LocalLLMService
>>> service = LocalLLMService()
>>> print(service._create_explanation_prompt({'symbol': 'TEST', 'technical_score': 7.0}, 'standard'))
```

**C. Input Data Issues:**
```python
# Validate analysis data structure
def validate_analysis_data(data):
    required_fields = ['symbol', 'technical_score', 'recommendation']
    for field in required_fields:
        if field not in data:
            print(f"Missing required field: {field}")
    
    if not isinstance(data.get('indicators'), dict):
        print("Indicators should be a dictionary")
```

### 5. Translation Failures

**Symptoms:**
- Translation service returns errors
- Missing translated explanations
- Incorrect language output

**Diagnosis:**
```bash
# Test translation model
docker exec voyageur-ollama ollama run qwen2 "Translate to French: Hello world"

# Check terminology mappings
python manage.py shell
>>> from Analytics.services.translation_service import FinancialTerminologyMapper
>>> mapper = FinancialTerminologyMapper()
>>> print(len(mapper.en_to_fr))  # Should be >20
```

**Solutions:**

**A. Translation Model Issues:**
```bash
# Verify qwen2 model
docker exec voyageur-ollama ollama list | grep qwen2

# Reinstall if corrupted
docker exec voyageur-ollama ollama rm qwen2
docker exec voyageur-ollama ollama pull qwen2
```

**B. Service Configuration:**
```python
# Test translation service
from Analytics.services.translation_service import TranslationService
service = TranslationService()
result = service.translate_explanation("Buy recommendation", "fr", {})
print(result)
```

### 6. High Error Rates

**Symptoms:**
- Error rate >10%
- Frequent service failures
- Circuit breaker activation

**Diagnosis:**
```bash
# Check error patterns
python manage.py llm_monitor_dashboard | grep -A10 "Error"

# Review error logs
tail -n 100 logs/llm_operations.log | grep ERROR

# Run error handling tests
python -m pytest Analytics/tests/test_error_handling.py -v
```

**Solutions:**

**A. Circuit Breaker Issues:**
```bash
# Check circuit breaker status
python manage.py shell
>>> from Analytics.services.local_llm_service import get_local_llm_service
>>> service = get_local_llm_service()
>>> print(service.circuit_breaker.state)  # Should be "CLOSED"
```

**B. Network Issues:**
```bash
# Test network connectivity
curl -I http://localhost:11434/api/version

# Check network latency
ping localhost

# Test with different timeout
export OLLAMA_REQUEST_TIMEOUT=60
```

**C. Resource Exhaustion:**
```bash
# Check memory usage
docker stats --no-stream

# Check disk space
df -h

# Clear cache if needed
python manage.py shell
>>> from django.core.cache import cache
>>> cache.clear()
```

### 7. Authentication/Access Issues

**Symptoms:**
- Permission denied errors
- User access failures
- Security-related errors

**Diagnosis:**
```bash
# Check user permissions
python manage.py shell
>>> from django.contrib.auth.models import User
>>> user = User.objects.get(username='test_user')
>>> print(user.is_active, user.is_staff)

# Test access control
python -m pytest Analytics/tests/test_security.py::SecurityValidationTestCase::test_access_control_verification -v
```

**Solutions:**

**A. User Issues:**
```python
# Fix inactive user
from django.contrib.auth.models import User
user = User.objects.get(username='problem_user')
user.is_active = True
user.save()
```

**B. Permission Issues:**
```python
# Check and assign permissions
from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType

# List available permissions
permissions = Permission.objects.all()
for p in permissions:
    print(f"{p.codename}: {p.name}")
```

## Performance Troubleshooting

### Slow Response Analysis

**Step 1: Identify Bottleneck**
```bash
# Run detailed benchmark
python manage.py benchmark_llm --percentiles --iterations 5 --output-file benchmark.json

# Check 95th percentile times
cat benchmark.json | jq '.tests.warm_start[] | select(.p95 > 15)'
```

**Step 2: Resource Analysis**
```bash
# Monitor during load
python -m pytest Analytics/tests/test_load_performance.py::LoadPerformanceTestCase::test_concurrent_users_performance -s

# Watch resources during test
watch -n 1 'docker stats --no-stream voyageur-ollama'
```

**Step 3: Optimisation**
```bash
# Increase model concurrency if CPU/memory allows
# In .env: OLLAMA_MAX_LOADED_MODELS=4

# Adjust GPU allocation if available  
# In .env: OLLAMA_GPU_MEMORY_FRACTION=0.9

# Warm models more frequently
echo "0 */4 * * * python manage.py warm_cache --llm-only" | crontab -
```

### Memory Issues

**Symptoms:**
- Out of memory errors
- Container restarts
- Degraded performance

**Diagnosis:**
```bash
# Check memory usage
docker exec voyageur-ollama free -h

# Check model memory usage
docker exec voyageur-ollama ps aux | grep ollama

# Monitor memory trends
python manage.py llm_monitor_dashboard --watch | grep -i memory
```

**Solutions:**
```bash
# Reduce loaded models
docker exec voyageur-ollama ollama ps
docker exec voyageur-ollama ollama stop model:tag

# Increase container memory
# In docker-compose.yml:
services:
  ollama:
    mem_limit: 8g
    memswap_limit: 8g

# Restart with new limits
docker-compose down && docker-compose up -d
```

## Monitoring and Alerts

### Setting Up Continuous Monitoring

```bash
# Start monitoring daemon (background)
nohup python manage.py llm_monitor_dashboard --watch > monitor.log 2>&1 &

# Monitor specific metrics
watch -n 30 'python manage.py llm_monitor_dashboard --format summary'

# Set up cron for regular checks
echo "*/5 * * * * python manage.py llm_monitor_dashboard --alerts-only | mail -s 'LLM Alerts' admin@company.com" | crontab -
```

### Alert Configuration

Create monitoring script `scripts/monitor_llm.sh`:
```bash
#!/bin/bash
ALERT_FILE="/tmp/llm_alerts.log"
python manage.py llm_monitor_dashboard --alerts-only > "$ALERT_FILE"

if [ -s "$ALERT_FILE" ]; then
    echo "LLM System Alerts Detected:"
    cat "$ALERT_FILE"
    
    # Send notifications (customize as needed)
    # curl -X POST -H 'Content-type: application/json' \
    #     --data "{\"text\":\"$(cat $ALERT_FILE)\"}" \
    #     "$SLACK_WEBHOOK_URL"
fi
```

## Recovery Procedures

### Full Service Recovery

**1. Complete Restart:**
```bash
# Stop all services
docker-compose down

# Clean up containers
docker container prune -f

# Restart services
docker-compose up -d

# Verify models
docker exec voyageur-ollama ollama list

# Warm up models
python manage.py warm_cache --llm-only
```

**2. Model Corruption Recovery:**
```bash
# Remove corrupted models
docker exec voyageur-ollama ollama rm corrupted_model:tag

# Clear Ollama cache
docker exec voyageur-ollama rm -rf ~/.ollama/models/corrupted_model

# Reinstall models
docker exec voyageur-ollama ollama pull phi3:3.8b
docker exec voyageur-ollama ollama pull llama3.1:8b
docker exec voyageur-ollama ollama pull qwen2

# Verify installation
python manage.py benchmark_llm --iterations 3
```

**3. Configuration Reset:**
```bash
# Backup current config
cp .env .env.backup

# Reset to defaults
cat > .env << EOF
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_SUMMARY_MODEL=phi3:3.8b
OLLAMA_STANDARD_MODEL=phi3:3.8b
OLLAMA_DETAILED_MODEL=llama3.1:8b
OLLAMA_TRANSLATION_MODEL=qwen2
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_GPU_MEMORY_FRACTION=0.8
OLLAMA_HEALTH_CHECK_INTERVAL=30
OLLAMA_CIRCUIT_BREAKER_THRESHOLD=3
EOF

# Restart with new config
docker-compose down && docker-compose up -d
```

## Testing After Changes

### Verification Checklist

After any troubleshooting changes:

```bash
# 1. Basic connectivity
curl http://localhost:11434/api/version

# 2. Model availability  
docker exec voyageur-ollama ollama list

# 3. Service health
python manage.py llm_monitor_dashboard --format summary

# 4. Basic functionality
python -c "
from Analytics.services.local_llm_service import get_local_llm_service
service = get_local_llm_service()
result = service.generate_explanation({
    'symbol': 'TEST',
    'technical_score': 7.0,
    'recommendation': 'BUY',
    'analysis_date': '2025-01-15T10:00:00Z'
}, 'summary')
print('✓ Basic test passed' if result else '✗ Basic test failed')
"

# 5. Performance test
python manage.py benchmark_llm --iterations 3

# 6. Run critical tests
python -m pytest Analytics/tests/test_phase_3_multi_model_integration.py::Phase3MultiModelIntegrationTestCase::test_summary_explanation_generation -v
```

## Escalation Procedures

### When to Escalate

Escalate to development team when:
- Service unavailable for >15 minutes
- Error rate >25% for >5 minutes  
- Complete model failure
- Security-related issues
- Data corruption suspected

### Information to Include

When escalating, provide:

1. **Current Status:**
   ```bash
   python manage.py llm_monitor_dashboard --format json > status.json
   ```

2. **Error Logs:**
   ```bash
   tail -n 500 logs/llm_operations.log > error_context.log
   docker logs voyageur-ollama --since 1h > ollama_logs.log
   ```

3. **System Info:**
   ```bash
   docker version > system_info.txt
   docker-compose version >> system_info.txt
   df -h >> system_info.txt
   free -h >> system_info.txt
   ```

4. **Configuration:**
   ```bash
   # Remove sensitive data before sharing
   grep -v PASSWORD .env > config_sanitized.env
   ```

5. **Recent Changes:**
   - Configuration changes
   - Model updates
   - System updates
   - Code deployments

### Emergency Contacts

- **Development Team**: [Contact information]
- **Infrastructure Team**: [Contact information]  
- **Security Team**: [Contact information]
- **On-call Rotation**: [Contact information]

---

*Keep this guide updated with new issues and solutions as they are discovered.*