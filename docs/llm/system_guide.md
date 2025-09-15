# VoyageurCompass LLM System Guide

## Overview

The VoyageurCompass LLM (Large Language Model) system provides intelligent financial analysis explanations using locally-hosted models through Ollama. This guide covers system architecture, operation, troubleshooting, and best practices.

## System Architecture

### Multi-Model Configuration

The system uses three specialised models for different explanation types:

- **phi3:3.8b** - Summary and standard explanations (fast, efficient)
- **llama3.1:8b** - Detailed explanations (comprehensive, analytical)  
- **qwen2** - Translation services (multilingual support)

### Core Components

1. **LocalLLMService** - Main service handling model selection and generation
2. **ModelHealthService** - Circuit breaker and health monitoring
3. **TranslationService** - Multilingual explanation support
4. **LLM Monitoring System** - Performance and quality tracking

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Ollama service running (port 11434)
- Required models installed (see Installation section)

### Basic Usage

```python
from Analytics.services.local_llm_service import get_local_llm_service

# Get LLM service instance
llm_service = get_local_llm_service()

# Generate explanation
analysis_data = {
    'symbol': 'AAPL',
    'technical_score': 7.5,
    'recommendation': 'BUY',
    'analysis_date': '2025-01-15T10:30:00Z',
    'indicators': {
        'rsi': {'value': 65.2, 'signal': 'bullish'},
        'macd': {'value': 0.45, 'signal': 'bullish'}
    }
}

result = llm_service.generate_explanation(
    analysis_data=analysis_data,
    detail_level="standard"
)

if result:
    print(result['explanation'])
```

## Installation and Setup

### 1. Model Installation

Install required Ollama models:

```bash
# Install summary/standard model
docker exec voyageur-ollama ollama pull phi3:3.8b

# Install detailed analysis model  
docker exec voyageur-ollama ollama pull llama3.1:8b

# Install translation model
docker exec voyageur-ollama ollama pull qwen2
```

### 2. Environment Configuration

Configure environment variables in `.env`:

```env
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_SUMMARY_MODEL=phi3:3.8b
OLLAMA_STANDARD_MODEL=phi3:3.8b
OLLAMA_DETAILED_MODEL=llama3.1:8b
OLLAMA_TRANSLATION_MODEL=qwen2
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_GPU_MEMORY_FRACTION=0.8

# Circuit Breaker Settings
OLLAMA_HEALTH_CHECK_INTERVAL=30
OLLAMA_CIRCUIT_BREAKER_THRESHOLD=3
```

### 3. Docker Configuration

Ensure docker-compose.yml includes:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_MAX_LOADED_MODELS=3
      - OLLAMA_GPU_MEMORY_FRACTION=0.8
```

## Operation Guide

### Model Warm-up

Pre-warm models for optimal performance:

```bash
# Warm-up all models
python manage.py warm_cache --llm-only

# Warm-up with comprehensive caching
python manage.py warm_cache --warm-up
```

### Performance Monitoring

View real-time monitoring dashboard:

```bash
# Display monitoring dashboard
python manage.py llm_monitor_dashboard

# Watch mode (refresh every 30 seconds)
python manage.py llm_monitor_dashboard --watch

# Show only alerts
python manage.py llm_monitor_dashboard --alerts-only
```

### Performance Benchmarking

Run comprehensive performance tests:

```bash
# Basic benchmark
python manage.py benchmark_llm --iterations 5

# Advanced benchmark with percentiles
python manage.py benchmark_llm --percentiles --iterations 10

# Regression testing
python manage.py benchmark_llm --regression-test baseline_results.json
```

## API Reference

### LocalLLMService Methods

#### `generate_explanation(analysis_data, detail_level, explanation_type)`

Generate financial analysis explanation.

**Parameters:**
- `analysis_data` (dict): Analysis results containing symbol, score, recommendation, indicators
- `detail_level` (str): 'summary', 'standard', or 'detailed'
- `explanation_type` (str): Type of explanation (default: 'technical_analysis')

**Returns:**
- dict: Contains 'explanation', 'model_used', optional 'quality_score'
- None: If generation failed

**Example:**
```python
result = llm_service.generate_explanation(
    analysis_data={
        'symbol': 'MSFT',
        'technical_score': 8.2,
        'recommendation': 'BUY',
        'indicators': {...}
    },
    detail_level="detailed"
)
```

#### `get_service_status()`

Get comprehensive service status information.

**Returns:**
- dict: Service status including availability, models, health metrics

#### `warm_up_models()`

Warm up all configured models with test prompts.

**Returns:**
- dict: Warm-up results including timing and success metrics

### TranslationService Methods

#### `translate_explanation(explanation, target_language, analysis_data)`

Translate explanation to target language.

**Parameters:**
- `explanation` (str): English explanation to translate
- `target_language` (str): 'fr' for French, 'es' for Spanish
- `analysis_data` (dict): Original analysis data for context

**Returns:**
- dict: Contains 'translated_explanation', 'target_language', optional 'quality_score'

## Detail Level Guidelines

### Summary (phi3:3.8b)
- **Purpose**: Quick overview for dashboards
- **Length**: 50-150 words
- **Response Time**: < 4 seconds (95th percentile)
- **Content**: Core recommendation with primary indicators

### Standard (phi3:3.8b)  
- **Purpose**: Balanced explanation for general users
- **Length**: 200-400 words
- **Response Time**: < 6 seconds (95th percentile)
- **Content**: Recommendation with supporting analysis

### Detailed (llama3.1:8b)
- **Purpose**: Comprehensive analysis for experienced users
- **Length**: 500-800 words
- **Response Time**: < 15 seconds (95th percentile)
- **Content**: Full technical analysis with market context

## Troubleshooting

### Common Issues

#### 1. "LLM service not available"

**Symptoms**: Service returns None or raises connection errors

**Solutions**:
```bash
# Check Ollama status
docker ps | grep ollama

# Restart Ollama container
docker restart voyageur-ollama

# Check model availability
docker exec voyageur-ollama ollama list

# Test connectivity
curl http://localhost:11434/api/tags
```

#### 2. Slow Response Times

**Symptoms**: Requests taking longer than expected

**Solutions**:
```bash
# Check model loading
python manage.py warm_cache --llm-only

# Monitor resource usage
python manage.py llm_monitor_dashboard --watch

# Run performance benchmark
python manage.py benchmark_llm --iterations 5
```

#### 3. High Error Rates

**Symptoms**: Frequent failures or poor quality responses

**Solutions**:
```bash
# Check service health
python manage.py llm_monitor_dashboard --alerts-only

# Review error logs
tail -f logs/llm_operations.log

# Test individual models
python manage.py benchmark_llm --detail-levels summary
```

#### 4. Translation Failures

**Symptoms**: Translation service returns errors

**Solutions**:
```bash
# Verify translation model
docker exec voyageur-ollama ollama list | grep qwen2

# Test translation model
docker exec voyageur-ollama ollama run qwen2 "Translate to French: Hello"

# Check terminology mappings
python manage.py shell
>>> from Analytics.services.translation_service import FinancialTerminologyMapper
>>> mapper = FinancialTerminologyMapper()
>>> print(mapper.get_terminology_context('fr'))
```

### Performance Optimisation

#### Model Selection Optimisation

Monitor model usage patterns:
```bash
python manage.py llm_monitor_dashboard --format json | jq '.metrics.models'
```

Adjust model assignments based on usage:
- High summary usage → Consider lighter model
- High detailed usage → Ensure sufficient resources

#### Cache Optimisation

Monitor cache performance:
```bash
python manage.py llm_monitor_dashboard | grep -A5 "Cache Performance"
```

Optimise cache TTL based on hit rates:
- Low hit rate → Increase TTL
- High memory usage → Decrease TTL

#### Resource Management

Monitor system resources during peak usage:
```bash
# Run load test
python -m pytest Analytics/tests/test_load_performance.py -v

# Monitor during test
python manage.py llm_monitor_dashboard --watch
```

## Security Best Practices

### Input Validation

- Always validate analysis_data structure
- Sanitise user inputs before LLM processing  
- Implement rate limiting for API endpoints
- Log security events for auditing

### Output Filtering

- Filter sensitive information from explanations
- Implement content safety checks
- Monitor for prompt injection attempts
- Validate explanation quality and relevance

### Access Control

- Implement proper user authentication
- Use role-based access control for features
- Monitor unusual access patterns
- Implement session management security

## Monitoring and Alerting

### Key Metrics

Monitor these critical metrics:

1. **Response Time**: 95th percentile should be within SLA
2. **Success Rate**: Should maintain >95% success rate
3. **Error Rate**: Should remain <5% total requests
4. **Quality Score**: Average should be >0.8
5. **Cache Hit Rate**: Should exceed 30% for efficiency

### Alert Thresholds

Configure alerts for:
- Error rate >10%
- Average response time >20 seconds
- Success rate <90%
- Quality score <0.7

### Dashboard Monitoring

Use the monitoring dashboard for:
```bash
# Real-time monitoring
python manage.py llm_monitor_dashboard --watch

# Performance trends
python manage.py llm_monitor_dashboard --format json | jq '.trends'

# Current alerts
python manage.py llm_monitor_dashboard --alerts-only
```

## Testing and Validation

### Running Test Suites

```bash
# Multi-model integration tests
python -m pytest Analytics/tests/test_phase_3_multi_model_integration.py -v

# Multilingual generation tests  
python -m pytest Analytics/tests/test_multilingual_generation.py -v

# Financial accuracy validation
python -m pytest Analytics/tests/test_financial_accuracy.py -v

# Error handling tests
python -m pytest Analytics/tests/test_error_handling.py -v

# Load testing
python -m pytest Analytics/tests/test_load_performance.py -v

# Security validation
python -m pytest Analytics/tests/test_security.py -v
```

### Performance Testing

```bash
# Comprehensive benchmark
python manage.py benchmark_llm --iterations 10 --percentiles

# Concurrent testing
python manage.py benchmark_llm --concurrent-requests 5 --iterations 8

# Regression testing
python manage.py benchmark_llm --regression-test previous_results.json
```

## Development Guidelines

### Adding New Models

1. **Install Model**:
```bash
docker exec voyageur-ollama ollama pull new_model:version
```

2. **Update Configuration**:
```env
OLLAMA_NEW_MODEL=new_model:version
```

3. **Modify Service**:
```python
# In LocalLLMService.__init__()
self.new_model = getattr(settings, 'OLLAMA_NEW_MODEL', 'default_model')

# Add to model selection logic
def _select_model_for_detail_level(self, detail_level, analysis_data):
    if detail_level == "new_type":
        return self.new_model
    # ... existing logic
```

4. **Add Tests**:
```python
def test_new_model_integration(self):
    result = self.llm_service.generate_explanation(
        analysis_data=test_data,
        detail_level="new_type"
    )
    self.assertEqual(result['model_used'], 'new_model:version')
```

### Custom Explanation Types

1. **Define Type**:
```python
# In generate_explanation method
if explanation_type == "custom_analysis":
    prompt = self._create_custom_prompt(analysis_data)
```

2. **Create Prompt Template**:
```python
def _create_custom_prompt(self, analysis_data):
    return f"""
    Custom analysis for {analysis_data['symbol']}:
    [Custom prompt logic here]
    """
```

3. **Add Validation**:
```python
def test_custom_explanation_type(self):
    result = self.llm_service.generate_explanation(
        analysis_data=test_data,
        explanation_type="custom_analysis"
    )
    self.assertIn('custom', result['explanation'].lower())
```

## Maintenance Procedures

### Regular Maintenance

**Daily**:
- Monitor dashboard for alerts
- Check error rates and response times
- Verify model availability

**Weekly**:
- Run comprehensive benchmark tests
- Review performance trends
- Update model cache if needed

**Monthly**:
- Review and update models
- Analyse usage patterns for optimisation
- Update security configurations
- Run full test suite

### Model Updates

1. **Download New Model**:
```bash
docker exec voyageur-ollama ollama pull model:new_version
```

2. **Test New Model**:
```bash
python manage.py benchmark_llm --detail-levels standard --iterations 3
```

3. **Update Configuration**:
```env
OLLAMA_STANDARD_MODEL=model:new_version
```

4. **Validate Performance**:
```bash
python manage.py benchmark_llm --regression-test baseline.json
```

5. **Monitor Post-Update**:
```bash
python manage.py llm_monitor_dashboard --watch
```

### Backup and Recovery

**Model Backup**:
```bash
# Backup Ollama data
docker run --rm -v ollama_data:/source -v $(pwd):/backup alpine tar czf /backup/ollama_backup.tar.gz -C /source .
```

**Configuration Backup**:
```bash
# Backup configuration
cp .env .env.backup
cp infrastructure/.env infrastructure/.env.backup
```

**Recovery Process**:
```bash
# Restore Ollama data
docker run --rm -v ollama_data:/target -v $(pwd):/backup alpine tar xzf /backup/ollama_backup.tar.gz -C /target

# Restart services
docker-compose down && docker-compose up -d

# Verify models
docker exec voyageur-ollama ollama list
```

## Support and Contact

### Internal Support

- **Development Team**: For technical issues and enhancements
- **Operations Team**: For deployment and infrastructure concerns
- **Security Team**: For security-related issues

### External Resources

- **Ollama Documentation**: https://ollama.com/docs
- **Model Documentation**: Check individual model pages
- **Community Forums**: Ollama GitHub discussions

### Reporting Issues

When reporting issues, include:

1. **Error Details**: Full error messages and stack traces
2. **Context**: Analysis data and configuration used
3. **Monitoring Data**: Dashboard screenshots or metrics
4. **Reproduction Steps**: Clear steps to reproduce the issue
5. **Environment Info**: Version, configuration, system specs

### Emergency Procedures

**Service Outage**:
1. Check Ollama container status
2. Restart services if needed
3. Verify model availability
4. Check system resources
5. Escalate if unresolved within 15 minutes

**Performance Degradation**:
1. Run immediate benchmark
2. Check resource utilisation
3. Review recent changes
4. Implement temporary mitigations
5. Schedule detailed investigation

**Security Incident**:
1. Isolate affected systems immediately
2. Preserve logs and evidence
3. Notify security team within 1 hour
4. Implement containment measures
5. Document incident timeline

---

*This guide is maintained by the VoyageurCompass development team. Last updated: January 2025*