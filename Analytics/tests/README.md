# Analytics Testing Infrastructure

This document describes the testing infrastructure for the VoyageurCompass Analytics platform, including cache backend behaviour, expected test patterns, and troubleshooting guidance.

## Cache Backend Testing

### DummyCache Limitations

The Analytics platform uses Django's cache framework extensively for performance optimisation. During testing, two cache backends are commonly used:

1. **Redis Cache** (Production/CI): Full caching functionality
2. **DummyCache** (Local Development): No-op cache for isolated testing

#### Expected DummyCache Behaviour

When using `DummyCache`, certain tests will fail by design because cache operations return `None`:

```python
# This will fail with DummyCache
def test_cache_hit_rate(self):
    # DummyCache.get() always returns None
    cached_result = cache.get('test_key')  # Returns None
    assert cached_result is not None  # FAILS
```

#### Affected Test Categories

1. **Sentiment Analysis Cache Tests**
   - `test_sentiment_cache_integration`
   - `test_batch_sentiment_caching`
   - Cache hit rate validations

2. **LLM Service Cache Tests**
   - `test_explanation_caching`
   - `test_llm_response_cache`
   - Cache key determinism tests

3. **Hybrid Analysis Cache Tests**
   - `test_coordinated_caching`
   - `test_cache_invalidation`

### Running Tests with Different Cache Backends

#### Local Development (DummyCache)
```bash
# Set in local settings
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}

# Run tests (some cache tests will fail)
pytest Analytics/tests/
```

#### CI/Integration Testing (Redis)
```bash
# Redis backend configured in test_settings.py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://localhost:6379/0',
    }
}

# All cache tests should pass
pytest Analytics/tests/ --cov
```

### Test Markers and Skipping

Cache-dependent tests are marked with `@pytest.mark.cache_required`:

```python
import pytest
from django.core.cache import cache
from django.core.cache.backends.dummy import DummyCache

@pytest.mark.cache_required
def test_sentiment_caching():
    """Test sentiment analysis caching - requires real cache backend."""
    if isinstance(cache, DummyCache):
        pytest.skip("Test requires functional cache backend")
    
    # Test implementation here
```

### Cache Testing Best Practices

#### 1. Cache Backend Detection
```python
def is_dummy_cache():
    """Check if current cache backend is DummyCache."""
    from django.core.cache import cache
    from django.core.cache.backends.dummy import DummyCache
    return isinstance(cache, DummyCache)
```

#### 2. Conditional Test Execution
```python
@pytest.mark.skipif(is_dummy_cache(), reason="Requires functional cache")
def test_cache_functionality():
    # Cache-dependent test logic
    pass
```

#### 3. Cache State Management
```python
def setup_method(self):
    """Clear cache before each test."""
    if not is_dummy_cache():
        cache.clear()

def teardown_method(self):
    """Clean up cache after test."""
    if not is_dummy_cache():
        cache.clear()
```

## Test Categories and Patterns

### Unit Tests
- **Service Layer**: Individual service method testing
- **Cache Logic**: Deterministic cache key generation
- **Model Predictions**: LSTM and sentiment model outputs
- **Data Processing**: TA indicator calculations

### Integration Tests
- **End-to-End Workflows**: Complete analysis pipelines
- **Service Coordination**: Multi-service interactions
- **Cache Integration**: Real cache hit/miss patterns
- **Database Operations**: Analytics result persistence

### Performance Tests
- **Cache Hit Rates**: Measuring cache effectiveness
- **Response Times**: API endpoint performance
- **Memory Usage**: Service resource consumption
- **Throughput**: Batch processing capabilities

## Common Test Failures and Solutions

### 1. Cache Key Determinism Failures
```
AssertionError: Cache keys not deterministic across runs
```
**Solution**: Ensure blake2b digest_size is consistent (16 bytes)

### 2. DummyCache Test Failures
```
AssertionError: Expected cached value, got None
```
**Solution**: Run tests with Redis backend or skip cache-dependent tests

### 3. Model Loading Failures
```
FileNotFoundError: No such file or directory: 'models/finbert'
```
**Solution**: Ensure ML models are downloaded or use mock implementations

### 4. Database Connection Issues
```
django.db.utils.OperationalError: could not connect to server
```
**Solution**: Verify test database configuration in `test_settings.py`

## Running Specific Test Suites

### Cache Tests Only
```bash
pytest Analytics/tests/ -k "cache" -v
```

### Non-Cache Tests Only
```bash
pytest Analytics/tests/ -k "not cache" -v
```

### Model Tests (ML Components)
```bash
pytest Analytics/tests/test_lstm_predictor.py -v
pytest Analytics/tests/test_sentiment_analyzer.py -v
```

### Integration Tests
```bash
pytest Analytics/tests/test_hybrid_integration.py -v
pytest Analytics/tests/test_async_processing_real.py -v
```

## Test Environment Configuration

### Required Environment Variables
```bash
# Database
DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_db

# Cache
REDIS_URL=redis://localhost:6379/0

# Django
DJANGO_SETTINGS_MODULE=VoyageurCompass.test_settings
SECRET_KEY=test-secret-key
DEBUG=False
```

### Docker Test Environment
```bash
# Start test services
docker-compose -f infrastructure/docker-compose.yml up -d db redis

# Run tests in container
docker-compose exec backend pytest Analytics/tests/ --cov
```

## Debugging Test Failures

### 1. Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Cache Inspection
```python
from django.core.cache import cache
print(f"Cache backend: {cache.__class__.__name__}")
print(f"Cache location: {getattr(cache, '_servers', 'N/A')}")
```

### 3. Test Data Examination
```python
def test_debug_cache_behaviour():
    cache.set('test_key', 'test_value', 300)
    result = cache.get('test_key')
    print(f"Set: test_value, Got: {result}")
    assert result == 'test_value'
```

## Contributing to Tests

### Adding New Cache Tests
1. Mark with `@pytest.mark.cache_required`
2. Include DummyCache skip logic
3. Clear cache state in setup/teardown
4. Use deterministic test data

### Test Data Management
- Use factories for consistent test data
- Avoid hardcoded timestamps
- Clean up created database records
- Mock external API calls

### Performance Test Guidelines
- Use `pytest-benchmark` for timing tests
- Set reasonable performance thresholds
- Test with representative data volumes
- Monitor resource usage patterns

This documentation ensures consistent testing practices and helps developers understand expected behaviour across different cache backends.