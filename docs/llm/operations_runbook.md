# LLM System Operations Runbook

## Daily Operations

### Morning Health Check (5 minutes)

**Checklist:**
```bash
# 1. Service status
python manage.py llm_monitor_dashboard --format summary

# 2. Model availability
docker exec voyageur-ollama ollama list | wc -l
# Expected: 3+ models

# 3. Performance check
python manage.py benchmark_llm --iterations 1 --detail-levels summary

# 4. Alert status
python manage.py llm_monitor_dashboard --alerts-only
```

**Expected Results:**
- Status: HEALTHY
- 3+ models available
- Summary response time <5 seconds
- No active alerts

### Evening Performance Review (10 minutes)

**Daily Metrics Collection:**
```bash
# Generate daily report
DATE=$(date +%Y-%m-%d)
python manage.py llm_monitor_dashboard --format json > "reports/llm_metrics_$DATE.json"

# Performance summary
echo "Daily Performance Summary - $DATE" > "reports/daily_summary_$DATE.txt"
python manage.py llm_monitor_dashboard --format summary >> "reports/daily_summary_$DATE.txt"

# Benchmark comparison
python manage.py benchmark_llm --iterations 5 --output-file "reports/benchmark_$DATE.json"
```

**Key Metrics to Track:**
- Total requests processed
- Average response time
- Success rate (target: >95%)
- Error rate (target: <5%)
- Cache hit rate
- Model usage distribution

## Weekly Operations

### Monday: System Health Assessment (30 minutes)

**Comprehensive Health Check:**
```bash
# 1. Resource utilisation analysis
python -m pytest Analytics/tests/test_load_performance.py::LoadPerformanceTestCase::test_resource_utilisation_optimisation -s

# 2. Model performance verification
for level in summary standard detailed; do
    echo "Testing $level explanations..."
    python manage.py benchmark_llm --detail-levels $level --iterations 5
done

# 3. Security validation
python -m pytest Analytics/tests/test_security.py::SecurityValidationTestCase::test_input_sanitisation_validation -v

# 4. Multilingual functionality
python -m pytest Analytics/tests/test_multilingual_generation.py::MultilingualGenerationTestCase::test_french_translation_quality -v
```

### Wednesday: Performance Optimisation (45 minutes)

**Cache Analysis:**
```bash
# 1. Cache performance review
python manage.py llm_monitor_dashboard | grep -A10 "Cache Performance"

# 2. Cache optimisation
# If hit rate <30%, consider adjusting TTL
# Check current cache usage:
python manage.py shell -c "
from django.core.cache import cache
from django.core.cache.backends.base import DEFAULT_TIMEOUT
print(f'Cache backend: {cache.__class__.__name__}')
print(f'Default timeout: {DEFAULT_TIMEOUT}')
"

# 3. Model usage analysis
python manage.py llm_monitor_dashboard --format json | jq '.metrics.models'
# Analyse for load balancing opportunities
```

**Model Warm-up Optimisation:**
```bash
# 1. Test current warm-up effectiveness
python manage.py benchmark_llm --warm-up --iterations 5

# 2. Optimise warm-up schedule if needed
# Consider adding to cron for peak hours:
# 0 8,12,17 * * * python manage.py warm_cache --llm-only
```

### Friday: Quality Assurance Review (60 minutes)

**Quality Validation:**
```bash
# 1. Financial accuracy testing
python -m pytest Analytics/tests/test_financial_accuracy.py -v

# 2. Response quality analysis
python manage.py llm_monitor_dashboard | grep -A5 "Quality Metrics"

# 3. Translation quality check (if multilingual enabled)
python -m pytest Analytics/tests/test_multilingual_generation.py::MultilingualGenerationTestCase::test_translation_consistency_across_detail_levels -v

# 4. Error pattern analysis
tail -n 1000 logs/llm_operations.log | grep ERROR | sort | uniq -c | sort -nr
```

## Monthly Operations

### First Monday: Model Updates and Maintenance (2 hours)

**Model Update Process:**

**1. Current Model Audit:**
```bash
# List current models with sizes
docker exec voyageur-ollama ollama list

# Check for available updates
docker exec voyageur-ollama ollama list | while read line; do
    model=$(echo $line | awk '{print $1}')
    if [ "$model" != "NAME" ]; then
        echo "Checking updates for $model..."
        docker exec voyageur-ollama ollama pull $model 2>&1 | grep -i "up to date\|downloading"
    fi
done
```

**2. Backup Current Models:**
```bash
# Create model backup
DATE=$(date +%Y%m%d)
mkdir -p backups/models_$DATE

# Document current configuration
docker exec voyageur-ollama ollama list > backups/models_$DATE/model_list.txt
cp .env backups/models_$DATE/config.env

# Export metrics baseline
python manage.py llm_monitor_dashboard --format json > backups/models_$DATE/baseline_metrics.json
```

**3. Update Process (if updates available):**
```bash
# For each model needing updates:
MODEL="phi3:3.8b"  # Example

# 1. Test current performance
python manage.py benchmark_llm --detail-levels summary --iterations 5 --output-file "pre_update_$MODEL.json"

# 2. Update model
docker exec voyageur-ollama ollama pull $MODEL

# 3. Test new performance
python manage.py benchmark_llm --detail-levels summary --iterations 5 --output-file "post_update_$MODEL.json"

# 4. Compare performance
python -c "
import json
with open('pre_update_$MODEL.json') as f:
    pre = json.load(f)
with open('post_update_$MODEL.json') as f:
    post = json.load(f)

pre_avg = pre['tests']['warm_start']['summary']['avg_time']
post_avg = post['tests']['warm_start']['summary']['avg_time']
change = ((post_avg - pre_avg) / pre_avg) * 100

print(f'Performance change: {change:.1f}%')
print('Acceptable' if abs(change) < 20 else 'Review needed')
"
```

### Second Monday: Security Review (90 minutes)

**Security Audit Checklist:**
```bash
# 1. Access control verification
python -m pytest Analytics/tests/test_security.py::SecurityValidationTestCase::test_access_control_verification -v

# 2. Input sanitisation testing
python -m pytest Analytics/tests/test_security.py::SecurityValidationTestCase::test_input_sanitisation_validation -v

# 3. Rate limiting validation
python -m pytest Analytics/tests/test_security.py::SecurityValidationTestCase::test_rate_limiting_protection -v

# 4. Audit log review
grep -i "security\|auth\|error" logs/llm_operations.log | tail -100

# 5. Configuration security check
echo "Checking configuration security..."
grep -i "password\|key\|secret" .env && echo "WARNING: Sensitive data in .env" || echo "✓ No plain text secrets found"
```

**Security Updates:**
```bash
# Update security configurations if needed
# Review and update rate limiting thresholds
# Update access control policies
# Rotate any API keys or tokens (if applicable)
```

### Third Monday: Capacity Planning (60 minutes)

**Usage Analysis:**
```bash
# 1. Generate monthly usage report
START_DATE=$(date -d "1 month ago" +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)

echo "LLM Usage Report: $START_DATE to $END_DATE" > monthly_report.txt
echo "============================================" >> monthly_report.txt

# Collect metrics from all daily reports
cat reports/daily_summary_*.txt | grep -E "requests|success|response" >> monthly_report.txt

# 2. Performance trend analysis
python -c "
import json, glob
from statistics import mean

files = glob.glob('reports/llm_metrics_*.json')
response_times = []
success_rates = []

for file in files[-30:]:  # Last 30 days
    try:
        with open(file) as f:
            data = json.load(f)
            if 'response_times' in data.get('metrics', {}):
                avg_time = data['metrics']['response_times'].get('avg', 0)
                if avg_time > 0:
                    response_times.append(avg_time)
            
            if 'requests' in data.get('metrics', {}):
                success_rate = data['metrics']['requests'].get('success_rate', 0)
                success_rates.append(success_rate)
    except:
        continue

if response_times:
    print(f'Average response time (30 days): {mean(response_times):.2f}s')
if success_rates:
    print(f'Average success rate (30 days): {mean(success_rates):.1f}%')
"

# 3. Capacity recommendations
python -m pytest Analytics/tests/test_load_performance.py::LoadPerformanceTestCase::test_auto_scaling_analysis -s
```

### Fourth Monday: Disaster Recovery Testing (45 minutes)

**DR Test Procedure:**
```bash
# 1. Backup current state
docker-compose down
cp -r $(docker volume inspect ollama_data | jq -r '.[0].Mountpoint') ollama_backup/

# 2. Simulate various failure scenarios
echo "Testing recovery scenarios..."

# Scenario A: Container failure
docker stop voyageur-ollama
sleep 30
docker start voyageur-ollama
python manage.py llm_monitor_dashboard --format summary

# Scenario B: Model corruption simulation
docker exec voyageur-ollama mv ~/.ollama/models/phi3 ~/.ollama/models/phi3.backup
python manage.py benchmark_llm --detail-levels summary --iterations 1
docker exec voyageur-ollama mv ~/.ollama/models/phi3.backup ~/.ollama/models/phi3

# 3. Test recovery procedures
python manage.py warm_cache --llm-only
python manage.py benchmark_llm --iterations 3

# 4. Validate full recovery
python -m pytest Analytics/tests/test_phase_3_multi_model_integration.py::Phase3MultiModelIntegrationTestCase::test_summary_explanation_generation -v
```

## Incident Response

### Severity Levels

**P0 - Critical (Response: Immediate)**
- Complete service outage
- Security breach
- Data corruption

**P1 - High (Response: <15 minutes)**
- Partial service outage
- Error rate >25%
- Performance degradation >50%

**P2 - Medium (Response: <1 hour)**
- Individual model failures
- Error rate 10-25%
- Translation service issues

**P3 - Low (Response: <24 hours)**
- Minor performance issues
- Cache inefficiencies
- Documentation updates

### Incident Response Procedures

**P0/P1 Incident Response:**

**1. Immediate Assessment (2 minutes):**
```bash
# Quick status check
python manage.py llm_monitor_dashboard --alerts-only

# Service availability
curl -f http://localhost:11434/api/version || echo "CRITICAL: Ollama not responding"

# Container status
docker ps | grep ollama || echo "CRITICAL: Ollama container down"
```

**2. Initial Mitigation (5 minutes):**
```bash
# Restart services if needed
if ! docker ps | grep -q ollama; then
    echo "Restarting Ollama container..."
    docker-compose up -d ollama
fi

# Clear cache if high error rates
if python manage.py llm_monitor_dashboard | grep -q "Error Rate.*[2-9][0-9]%"; then
    echo "Clearing cache due to high error rate..."
    python manage.py shell -c "from django.core.cache import cache; cache.clear()"
fi
```

**3. Detailed Investigation (10 minutes):**
```bash
# Collect diagnostic data
mkdir -p incident_$(date +%Y%m%d_%H%M)
cd incident_$(date +%Y%m%d_%H%M)

# System state
python manage.py llm_monitor_dashboard --format json > system_state.json
docker logs voyageur-ollama --since 1h > ollama_logs.txt
tail -n 500 ../logs/llm_operations.log > application_logs.txt
docker stats --no-stream > resource_usage.txt

# Test individual components
python manage.py benchmark_llm --iterations 1 > benchmark_test.txt 2>&1
```

**4. Escalation Communication:**
```bash
# Generate incident report
cat > incident_report.txt << EOF
INCIDENT REPORT - $(date)
Severity: P0/P1
Status: INVESTIGATING

Current System State:
$(python manage.py llm_monitor_dashboard --format summary)

Actions Taken:
1. Service restart attempted
2. Cache cleared
3. Diagnostic data collected

Next Steps:
- Detailed analysis in progress
- ETA for resolution: [estimate]

Contact: [your contact info]
EOF

# Send notification (customize as needed)
echo "Incident reported: $(cat incident_report.txt)"
```

### Post-Incident Review

**Within 24 hours of resolution:**

**1. Root Cause Analysis:**
```bash
# Document timeline
cat > post_incident_analysis.md << EOF
# Post-Incident Analysis

## Timeline
- [Time]: Issue detected
- [Time]: Response initiated  
- [Time]: Mitigation applied
- [Time]: Service restored

## Root Cause
[Detailed analysis of what caused the issue]

## Contributing Factors
- [Factor 1]
- [Factor 2]

## Resolution Steps
- [Step 1]
- [Step 2]

## Prevention Measures
- [Action 1: Monitoring improvement]
- [Action 2: Process change]
- [Action 3: Technical fix]
EOF
```

**2. System Improvements:**
```bash
# Update monitoring thresholds if needed
# Add new alerts for detected failure modes
# Update documentation based on lessons learned
# Schedule preventive maintenance if applicable
```

## Maintenance Windows

### Monthly Maintenance Window (2 hours)

**Scheduled for first Saturday of each month, 2:00-4:00 AM local time**

**Pre-Maintenance Checklist:**
```bash
# 1. Notify users of maintenance window
# 2. Create full system backup
docker-compose down
docker run --rm -v ollama_data:/source -v $(pwd)/backups:/backup alpine tar czf /backup/pre_maintenance_$(date +%Y%m%d).tar.gz -C /source .

# 3. Document current state
python manage.py llm_monitor_dashboard --format json > backups/pre_maintenance_state.json
docker exec voyageur-ollama ollama list > backups/pre_maintenance_models.txt
```

**Maintenance Tasks:**
```bash
# 1. System updates
docker-compose pull
docker system prune -f

# 2. Model updates (if available)
# [Follow monthly model update procedure]

# 3. Performance optimisation
# [Review and implement optimisations identified during month]

# 4. Security updates
# [Apply any security patches or configuration updates]

# 5. Cleanup operations
# Remove old logs older than 90 days
find logs/ -name "*.log" -mtime +90 -delete

# Remove old benchmark files
find reports/ -name "benchmark_*.json" -mtime +30 -delete
```

**Post-Maintenance Validation:**
```bash
# 1. Service verification
python manage.py llm_monitor_dashboard --format summary

# 2. Performance validation
python manage.py benchmark_llm --iterations 5

# 3. Functionality testing
python -m pytest Analytics/tests/test_phase_3_multi_model_integration.py::Phase3MultiModelIntegrationTestCase::test_summary_explanation_generation -v

# 4. Document results
echo "Maintenance completed successfully at $(date)" > maintenance_log.txt
python manage.py llm_monitor_dashboard --format summary >> maintenance_log.txt
```

## Monitoring and Alerting Setup

### Continuous Monitoring Script

Create `/usr/local/bin/monitor_llm.sh`:
```bash
#!/bin/bash
set -e

LOG_FILE="/var/log/llm_monitor.log"
ALERT_THRESHOLD_ERROR_RATE=15
ALERT_THRESHOLD_RESPONSE_TIME=25

# Get current metrics
METRICS=$(python manage.py llm_monitor_dashboard --format json)

# Extract key metrics
ERROR_RATE=$(echo "$METRICS" | jq -r '.metrics.errors.error_rate // 0')
AVG_RESPONSE_TIME=$(echo "$METRICS" | jq -r '.metrics.response_times.avg // 0')
SUCCESS_RATE=$(echo "$METRICS" | jq -r '.metrics.requests.success_rate // 100')

# Check thresholds
ALERTS=()
if (( $(echo "$ERROR_RATE > $ALERT_THRESHOLD_ERROR_RATE" | bc -l) )); then
    ALERTS+=("High error rate: ${ERROR_RATE}%")
fi

if (( $(echo "$AVG_RESPONSE_TIME > $ALERT_THRESHOLD_RESPONSE_TIME" | bc -l) )); then
    ALERTS+=("Slow response time: ${AVG_RESPONSE_TIME}s")
fi

if (( $(echo "$SUCCESS_RATE < 90" | bc -l) )); then
    ALERTS+=("Low success rate: ${SUCCESS_RATE}%")
fi

# Log status
echo "$(date): Error=${ERROR_RATE}%, Response=${AVG_RESPONSE_TIME}s, Success=${SUCCESS_RATE}%" >> "$LOG_FILE"

# Send alerts if needed
if [ ${#ALERTS[@]} -gt 0 ]; then
    ALERT_MSG="LLM System Alerts:\n$(printf '%s\n' "${ALERTS[@]}")"
    echo -e "$ALERT_MSG" >> "$LOG_FILE"
    
    # Send notification (customize as needed)
    echo -e "$ALERT_MSG" | mail -s "LLM System Alert" ops@company.com
fi
```

### Cron Configuration

Add to crontab:
```bash
# LLM system monitoring
*/5 * * * * /usr/local/bin/monitor_llm.sh

# Daily warm-up during business hours
0 8 * * 1-5 python manage.py warm_cache --llm-only

# Weekly comprehensive test
0 6 * * 1 python -m pytest Analytics/tests/test_phase_3_multi_model_integration.py -v > /var/log/llm_weekly_test.log 2>&1

# Monthly report generation
0 1 1 * * python manage.py llm_monitor_dashboard --format json > /var/reports/llm_monthly_$(date +\%Y\%m).json
```

---

*This runbook should be reviewed and updated quarterly to reflect operational learnings and system changes.*