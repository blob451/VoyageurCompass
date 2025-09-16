# LLM Explanation Service Architecture Documentation

## Overview

The LLM explanation service in VoyageurCompass is a sophisticated, multi-layered system that generates natural language explanations for financial analysis results. It leverages Large Language Models (LLaMA 3.1) via Ollama integration to provide professional-grade investment analysis explanations in multiple languages.

## Architecture Overview

### Core Services

1. **ExplanationService** (`explanation_service.py`) - Main orchestrator
2. **LocalLLMService** (`local_llm_service.py`) - LLM integration layer
3. **MultilingualExplanationPipeline** (`multilingual_pipeline.py`) - Multilingual support
4. **CulturalFormatter** (`cultural_formatter.py`) - Locale-specific formatting
5. **TranslationService** (`translation_service.py`) - Translation capabilities

### Supporting Components

- Security validators
- Circuit breakers for reliability
- Performance monitoring systems
- Quality validators
- Intelligent caching mechanisms

## Service Layer Architecture

### 1. ExplanationService

**Purpose**: Main entry point for generating financial analysis explanations

**Key Features:**
- **Dual-mode operation**: LLM-based generation with template-based fallback
- **Security validation**: Input sanitization and output filtering
- **Intelligent caching**: TTL management with volatility-based adjustments
- **Multiple detail levels**: Summary, standard, and detailed explanations
- **Batch processing**: Handle multiple analyses efficiently

**Core Methods:**
```python
explain_prediction_single(analysis_result, detail_level="standard", user=None, force_regenerate=False)
explain_prediction_batch(analysis_results, detail_level="standard", user=None)
build_indicator_explanation(indicator_name, indicator_result, weighted_score, context=None)
```

**Supported Detail Levels:**
- **Summary**: Brief overview with key points (Standard tier)
- **Standard**: Comprehensive analysis with technical details (Enhanced tier)
- **Detailed**: In-depth analysis with all indicators (Premium tier)

### 2. LocalLLMService

**Purpose**: Direct interface with Ollama-hosted LLaMA models

**Multi-Model Configuration:**
- **Summary Model**: `phi3:3.8b` - Fast, concise explanations
- **Standard Model**: `phi3:3.8b` - Balanced performance
- **Detailed Model**: `llama3.1:8b` - Comprehensive analysis
- **Translation Model**: `qwen2:3b` - Efficient multilingual processing

**Advanced Features:**
- **Sentiment-enhanced prompts**: Integration with FinBERT sentiment analysis
- **Circuit breaker pattern**: Fault tolerance and graceful degradation
- **Performance monitoring**: Real-time metrics and health checks
- **Dynamic model selection**: Complexity-based model assignment
- **GPU acceleration**: Optimized for hardware acceleration

**Key Methods:**
```python
generate_explanation(analysis_data, detail_level="standard", explanation_type="technical_analysis")
generate_multilingual_explanation(analysis_data, target_language="en", detail_level="standard")
is_available() -> bool
get_service_status() -> Dict[str, Any]
```

### 3. MultilingualExplanationPipeline

**Purpose**: Orchestrate multilingual explanation generation with quality assurance

**Supported Languages:**
- **English (en)**: Native LLaMA 3.1 generation
- **French (fr)**: Specialized multilingual model with financial terminology
- **Spanish (es)**: Localized generation with cultural formatting

**Process Flow:**
1. **Base Generation**: Native language generation or translation
2. **Cultural Formatting**: Locale-appropriate number, currency, date formatting
3. **Quality Validation**: Multi-dimensional quality scoring
4. **Caching**: Store results if quality threshold met (>80%)

**Quality Metrics:**
- **Terminology Score**: Financial term accuracy (40% weight)
- **Completeness Score**: Content coverage and structure (30% weight)
- **Cultural Appropriateness**: Locale-specific formatting (30% weight)

### 4. CulturalFormatter

**Purpose**: Apply locale-specific formatting for international users

**Formatting Capabilities:**

| Locale | Currency | Thousands Sep | Decimal Sep | Date Format |
|--------|----------|---------------|-------------|-------------|
| English | $1,234.56 | , | . | MM/DD/YYYY |
| French | 1 234,56 € | space | , | DD/MM/YYYY |
| Spanish | 1.234,56 € | . | , | DD/MM/YYYY |

**Methods:**
```python
format_currency(amount, currency_code="USD", language="en")
format_number(number, language="en", decimal_places=2)
format_percentage(percentage, language="en", decimal_places=2)
format_financial_text(text, language="en")
```

## Data Flow Architecture

### Request Processing Pipeline

```
API Request → ExplanationService → Cache Check → Generation Path
                                                      ↓
                                              LLM Available?
                                                 ├─ Yes → LocalLLMService → Ollama → LLaMA
                                                 └─ No → Template Fallback
                                                      ↓
                                              Multilingual Required?
                                                 ├─ Yes → MultilingualPipeline
                                                 └─ No → Direct Response
                                                      ↓
                                              Security Validation → Cache → Response
```

### Multilingual Processing Flow

```
Non-English Request → Determine Generation Method
                           ├─ Native Generation (French/Spanish prompts)
                           └─ Translation (English → Target Language)
                                      ↓
                              Apply Cultural Formatting
                                      ↓
                              Quality Validation (80% threshold)
                                      ↓
                              Cache (if quality sufficient) → Response
```

## Caching Strategy

### Multi-Level Cache Architecture

1. **L1 Cache**: In-memory (5-minute TTL)
2. **L2 Cache**: Redis persistent (configurable TTL)
3. **Multilingual Cache**: Language-specific keys with extended TTL

### Dynamic TTL Calculation

```python
base_ttl = 300  # 5 minutes
dynamic_ttl = base_ttl * volatility_factor * sentiment_confidence_factor
```

**Cache Key Structure:**
```
explanation_{symbol}_{price}_{detail_level}_{language}_{user_hash}
```

## Security Architecture

### Input Validation
- SQL injection prevention
- XSS protection
- Financial data sanitization
- User input limits

### Output Filtering
- Content validation against security patterns
- PII detection and removal
- Financial accuracy verification

### Rate Limiting
```python
THROTTLE_RATES = {
    'anon': '100/hour',
    'user': '1000/hour'
}
```

## Quality Assurance Framework

### Quality Scoring Algorithm

```python
overall_score = (
    terminology_score * 0.4 +
    completeness_score * 0.3 +
    cultural_appropriateness * 0.3
)
```

### Quality Thresholds
- **Minimum for caching**: 0.8
- **Minimum for display**: 0.6
- **Confidence threshold**: 0.7

### Validation Components

1. **Financial Terminology Validation**
   - Check for correct financial term usage
   - Validate translations against terminology mappings

2. **Completeness Validation**
   - Ensure all key analysis points covered
   - Verify recommendation clarity

3. **Cultural Appropriateness**
   - Validate locale-specific formatting
   - Check cultural context sensitivity

## Performance Optimizations

### Model Management
```python
# Pre-loaded models with health checking
models = {
    'summary': 'phi3:3.8b',
    'standard': 'phi3:3.8b',
    'detailed': 'llama3.1:8b',
    'translation': 'qwen2:3b'
}
```

### Request Optimization
- **Async processing** for batch requests
- **Parallel generation** for multiple languages
- **Request deduplication** to prevent duplicate work

### Resource Management
- Thread pool executor (max 3 workers)
- Memory-efficient prompt construction
- GPU utilization when available

## API Integration

### Main Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analytics/explain/{analysis_id}/` | POST | Generate new explanation |
| `/api/analytics/explanation/{analysis_id}/` | GET | Retrieve cached explanation |
| `/api/analytics/multilingual/status/` | GET | Pipeline status check |
| `/api/analytics/explain/status/` | GET | Service health check |

### Request Parameters

```json
{
  "detail_level": "summary|standard|detailed",
  "language": "en|fr|es",
  "force_regenerate": false,
  "format": "standard"
}
```

### Response Format

```json
{
  "success": true,
  "analysis_id": 123,
  "symbol": "AAPL",
  "explanation": {
    "content": "AAPL receives a 7.5/10 analysis score...",
    "language": "en",
    "confidence_score": 0.85,
    "detail_level": "standard",
    "generation_method": "native",
    "model_used": "llama3.1:8b",
    "generation_time": 3.2,
    "word_count": 245,
    "cultural_formatting_applied": true,
    "quality_metrics": {
      "overall_score": 0.87,
      "terminology_score": 0.90,
      "completeness_score": 0.85,
      "cultural_appropriateness": 0.85
    }
  },
  "multilingual": {
    "requested_language": "en",
    "pipeline_available": true,
    "supported_languages": ["en", "fr", "es"],
    "pipeline_enabled": true
  }
}
```

## Configuration

### Django Settings

```python
# Core Settings
MULTILINGUAL_LLM_ENABLED = True
EXPLAINABILITY_ENABLED = True
CULTURAL_FORMATTING_ENABLED = True

# Model Configuration
LLM_MODELS_BY_LANGUAGE = {
    'en': 'llama3.1:8b',
    'fr': 'qwen2:3b',
    'es': 'qwen2:3b'
}

# Quality Thresholds
TRANSLATION_QUALITY_THRESHOLD = 0.8
EXPLANATION_CACHE_TTL = 300

# Financial Formatting
FINANCIAL_FORMATTING = {
    'en': {
        'currency_symbol': '$',
        'currency_position': 'before',
        'decimal_separator': '.',
        'thousands_separator': ','
    },
    'fr': {
        'currency_symbol': '€',
        'currency_position': 'after',
        'decimal_separator': ',',
        'thousands_separator': ' '
    }
}
```

### Ollama Configuration

```bash
# Required models
ollama pull llama3.1:8b
ollama pull llama3.1:70b  # Optional for premium tier
ollama pull phi3:3.8b
ollama pull qwen2:3b
```

## Error Handling & Resilience

### Fallback Hierarchy

1. **Primary**: LLM generation with target model
2. **Secondary**: Alternative model if primary unavailable
3. **Tertiary**: Translation service for non-English
4. **Quaternary**: Template-based generation
5. **Final**: Minimal safe response

### Circuit Breaker Implementation

```python
class LLMCircuitBreaker:
    states = ["CLOSED", "OPEN", "HALF_OPEN"]

    def call_llm(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is open")
        # ... implementation
```

### Monitoring & Alerting

**Tracked Metrics:**
- Generation times per model
- Cache hit/miss ratios
- Quality score distributions
- Language usage statistics
- Model availability status
- Error rates by category

## Future Enhancements

### Planned Features
1. **Fine-tuning Pipeline**: Custom model training on financial data
2. **Real-time Adaptation**: Dynamic prompt optimization
3. **Advanced Sentiment Integration**: Market news correlation
4. **Additional Languages**: German, Italian, Japanese support
5. **Explanation Personalization**: User preference learning

### Performance Improvements
1. **Model Quantization**: Reduced memory footprint
2. **Streaming Responses**: Progressive explanation delivery
3. **Advanced Caching**: Semantic similarity-based cache hits
4. **Edge Deployment**: Regional model distribution

## Maintenance & Operations

### Health Checks
```bash
# Service status
curl /api/analytics/explain/status/

# Multilingual pipeline status
curl /api/analytics/multilingual/status/
```

### Log Analysis
```python
# Key log patterns
"[EXPLAIN] Processing explanation request"
"[MULTILINGUAL] Generation successful"
"[QUALITY] Validation failed"
"[CACHE] Retrieved cached explanation"
```

### Performance Monitoring
- Response time percentiles (p50, p95, p99)
- Model-specific generation times
- Quality score trends
- Language usage patterns

This architecture provides a robust, scalable, and maintainable foundation for AI-powered financial explanation generation with comprehensive multilingual support.