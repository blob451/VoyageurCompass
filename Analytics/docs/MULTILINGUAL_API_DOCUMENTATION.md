# VoyageurCompass Multilingual LLM API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Parameters](#common-parameters)
4. [Core Analysis Endpoints](#core-analysis-endpoints)
5. [Multilingual Explanation Endpoints](#multilingual-explanation-endpoints)
6. [Health Check Endpoints](#health-check-endpoints)
7. [Production Monitoring Endpoints](#production-monitoring-endpoints)
8. [Feature Flags Endpoints](#feature-flags-endpoints)
9. [Circuit Breaker Endpoints](#circuit-breaker-endpoints)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)
12. [Examples](#examples)

## Overview

The VoyageurCompass Multilingual LLM API provides financial analysis and explanations in multiple languages. The system supports English, French, and Spanish with production-ready features including feature flags, circuit breakers, and comprehensive monitoring.

**Base URL:** `https://api.voyageurcompass.com/analytics/`

**Supported Languages:**
- `en` - English (default)
- `fr` - French
- `es` - Spanish

## Authentication

Most endpoints require authentication using Django's built-in authentication system.

### Authentication Methods

1. **Session Authentication** (for web applications)
2. **Token Authentication** (for API clients)

### Headers
```
Authorization: Token your_api_token_here
Content-Type: application/json
```

## Common Parameters

### Language Parameter
Most endpoints support an optional `language` parameter:

```json
{
  "language": "fr"  // Optional: "en", "fr", "es"
}
```

### Detail Level Parameter
Explanation endpoints support a `detail_level` parameter:

```json
{
  "detail_level": "summary"  // "summary", "detailed", "comprehensive"
}
```

### Explanation Type Parameter
```json
{
  "explanation_type": "technical_analysis"  // "technical_analysis", "fundamental_analysis", "market_overview"
}
```

## Core Analysis Endpoints

### Analyze Stock

Analyze a specific stock symbol with optional multilingual explanations.

**Endpoint:** `POST /analyze/{symbol}/`

**Parameters:**
- `symbol` (path): Stock symbol (e.g., "AAPL")
- `language` (optional): Target language for explanation
- `detail_level` (optional): Level of detail for explanation
- `explanation_type` (optional): Type of analysis explanation

**Request Example:**
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/analyze/AAPL/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "fr",
    "detail_level": "detailed",
    "explanation_type": "technical_analysis"
  }'
```

**Response Example:**
```json
{
  "symbol": "AAPL",
  "analysis_id": 12345,
  "score_0_10": 7.5,
  "timestamp": "2025-01-17T10:30:00Z",
  "indicators": {
    "sma50": 150.25,
    "sma200": 145.80,
    "rsi": 65.2,
    "macd": 2.1
  },
  "weighted_scores": {
    "sma50vs200": 0.75,
    "rsi_score": 0.65,
    "macd_score": 0.70
  },
  "explanation": {
    "language": "fr",
    "content": "Apple Inc. (AAPL) présente actuellement un score technique de 7.5/10...",
    "quality_score": 0.92,
    "generation_time": 2.3,
    "model_used": "fr_financial_v1.2",
    "fallback_used": false
  },
  "cache_info": {
    "cache_hit": false,
    "cache_key": "analysis_AAPL_fr_detailed_tech"
  }
}
```

### Analyze Portfolio

Analyze a portfolio with multilingual summaries.

**Endpoint:** `POST /analyze-portfolio/{portfolio_id}/`

**Parameters:**
- `portfolio_id` (path): Portfolio identifier
- `language` (optional): Target language for explanation
- `include_individual_stocks` (optional): Include individual stock analyses

**Request Example:**
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/analyze-portfolio/123/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "es",
    "include_individual_stocks": true
  }'
```

**Response Example:**
```json
{
  "portfolio_id": 123,
  "analysis_id": 67890,
  "overall_score": 6.8,
  "timestamp": "2025-01-17T10:30:00Z",
  "total_value": 150000.00,
  "daily_change": 2.5,
  "stocks": [
    {
      "symbol": "AAPL",
      "weight": 0.3,
      "score": 7.5,
      "explanation": {
        "language": "es",
        "content": "Apple Inc. muestra señales técnicas sólidas..."
      }
    }
  ],
  "portfolio_explanation": {
    "language": "es",
    "content": "Su cartera presenta un rendimiento equilibrado...",
    "quality_score": 0.89,
    "generation_time": 3.1
  }
}
```

## Multilingual Explanation Endpoints

### Generate Explanation

Generate or retrieve a multilingual explanation for an existing analysis.

**Endpoint:** `POST /explain/{analysis_id}/`

**Parameters:**
- `analysis_id` (path): Analysis identifier
- `language` (required): Target language
- `detail_level` (optional): Explanation detail level
- `explanation_type` (optional): Type of explanation
- `force_regenerate` (optional): Force new generation even if cached

**Request Example:**
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/explain/12345/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "fr",
    "detail_level": "comprehensive",
    "explanation_type": "technical_analysis",
    "force_regenerate": false
  }'
```

**Response Example:**
```json
{
  "explanation_id": "exp_12345_fr_comp_tech",
  "analysis_id": 12345,
  "language": "fr",
  "detail_level": "comprehensive",
  "explanation_type": "technical_analysis",
  "content": "Une analyse technique approfondie d'Apple Inc. révèle...",
  "quality_score": 0.94,
  "generation_time": 4.2,
  "model_used": "fr_financial_v1.2",
  "fallback_used": false,
  "cache_hit": false,
  "timestamp": "2025-01-17T10:35:00Z",
  "metadata": {
    "word_count": 485,
    "readability_score": 0.78,
    "technical_terms_count": 12
  }
}
```

### Get Explanation

Retrieve an existing explanation.

**Endpoint:** `GET /explanation/{analysis_id}/`

**Query Parameters:**
- `language` (optional): Filter by language
- `detail_level` (optional): Filter by detail level

**Response Example:**
```json
{
  "explanation_id": "exp_12345_fr_comp_tech",
  "analysis_id": 12345,
  "language": "fr",
  "content": "Une analyse technique approfondie...",
  "quality_score": 0.94,
  "cached": true,
  "timestamp": "2025-01-17T10:35:00Z"
}
```

### Bulk Multilingual Generation

Generate explanations in multiple languages simultaneously.

**Endpoint:** `POST /multilingual/bulk/`

**Request Example:**
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/multilingual/bulk/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_ids": [12345, 12346, 12347],
    "languages": ["fr", "es"],
    "detail_level": "summary",
    "explanation_type": "technical_analysis"
  }'
```

**Response Example:**
```json
{
  "batch_id": "batch_20250117_103000",
  "status": "processing",
  "total_requests": 6,
  "estimated_completion": "2025-01-17T10:32:00Z",
  "results": [
    {
      "analysis_id": 12345,
      "language": "fr",
      "status": "completed",
      "explanation_id": "exp_12345_fr_summ_tech"
    },
    {
      "analysis_id": 12345,
      "language": "es",
      "status": "processing"
    }
  ]
}
```

### Parallel Multilingual Generation

Generate explanation in a specific language with high priority.

**Endpoint:** `POST /multilingual/parallel/{analysis_id}/`

**Request Example:**
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/multilingual/parallel/12345/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "fr",
    "detail_level": "detailed",
    "priority": "high"
  }'
```

## Health Check Endpoints

### Multilingual Health Check

Comprehensive health check for multilingual services.

**Endpoint:** `GET /health/multilingual/`

**Authentication:** None required

**Response Example:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-17T10:30:00Z",
  "check_duration_ms": 234.5,
  "multilingual_enabled": true,
  "supported_languages": ["en", "fr", "es"],
  "model_status": {
    "fr": "healthy",
    "es": "healthy"
  },
  "quality_scores": {
    "fr": 0.91,
    "es": 0.89
  },
  "feature_flags": {
    "multilingual_enabled": true,
    "french_enabled": true,
    "spanish_enabled": true,
    "emergency_fallback": false
  },
  "circuit_breakers": {
    "fr": "closed",
    "es": "closed"
  },
  "performance_metrics": {
    "avg_response_time": 2.3,
    "error_rate": 0.02,
    "requests_per_minute": 45,
    "quality_score": 0.90
  }
}
```

### Health Ping

Lightweight health check for load balancers.

**Endpoint:** `GET /health/ping/`

**Authentication:** None required

**Response Example:**
```json
{
  "status": "healthy",
  "multilingual_enabled": true,
  "timestamp": "2025-01-17T10:30:00Z"
}
```

### Feature Flags Status

Get current feature flags status.

**Endpoint:** `GET /health/feature-flags/`

**Authentication:** None required

**Response Example:**
```json
{
  "timestamp": "2025-01-17T10:30:00Z",
  "flags": {
    "multilingual_llm_enabled": {
      "enabled": true,
      "default_value": true
    },
    "french_generation_enabled": {
      "enabled": true,
      "default_value": true
    },
    "spanish_generation_enabled": {
      "enabled": true,
      "default_value": true
    }
  },
  "rollout_percentages": {
    "french_generation_enabled": 100,
    "spanish_generation_enabled": 100
  },
  "emergency_status": {
    "emergency_fallback_enabled": false,
    "circuit_breaker_enabled": true
  }
}
```

## Production Monitoring Endpoints

### Monitoring Status

Get production monitoring service status.

**Endpoint:** `GET /monitoring/status/`

**Authentication:** Required

**Response Example:**
```json
{
  "monitoring_enabled": true,
  "thread_running": true,
  "check_interval": 60,
  "last_check": "2025-01-17T10:29:00Z",
  "thresholds": {
    "cpu_threshold": 80,
    "memory_threshold": 85,
    "disk_threshold": 90,
    "response_time_threshold": 10,
    "error_rate_threshold": 0.1
  },
  "notification_channels": {
    "email_enabled": true,
    "webhook_enabled": false
  }
}
```

### Recent Alerts

Get recent production alerts.

**Endpoint:** `GET /monitoring/alerts/`

**Authentication:** Required

**Query Parameters:**
- `limit` (optional): Number of alerts to return (default: 50)

**Response Example:**
```json
{
  "alerts": [
    {
      "id": "1705656600_1234",
      "level": "warning",
      "title": "High response time detected",
      "message": "Average response time: 8.45s (threshold: 10s)",
      "timestamp": "2025-01-17T10:30:00Z",
      "context": {
        "avg_response_time": 8.45,
        "threshold": 10
      }
    }
  ],
  "count": 1,
  "limit": 50
}
```

### Force Health Check

Trigger an immediate comprehensive health check.

**Endpoint:** `POST /monitoring/health-check/`

**Authentication:** Required

**Response Example:**
```json
{
  "status": "completed",
  "check_duration": 1.23,
  "timestamp": "2025-01-17T10:30:00Z",
  "recent_alerts": [
    {
      "level": "info",
      "title": "Health check completed",
      "message": "All systems operational"
    }
  ]
}
```

## Error Handling

### Error Response Format

All error responses follow this format:

```json
{
  "error": "error_code",
  "message": "Human readable error message",
  "details": {
    "field": "Additional error details"
  },
  "timestamp": "2025-01-17T10:30:00Z",
  "request_id": "req_12345"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `invalid_request` | Invalid request parameters |
| 401 | `authentication_required` | Authentication required |
| 403 | `permission_denied` | Insufficient permissions |
| 404 | `not_found` | Resource not found |
| 429 | `rate_limit_exceeded` | Rate limit exceeded |
| 500 | `internal_server_error` | Internal server error |
| 503 | `service_unavailable` | Service temporarily unavailable |

### Multilingual-Specific Errors

| Error Code | Description |
|------------|-------------|
| `language_not_supported` | Requested language not supported |
| `translation_failed` | Translation service failed |
| `model_unavailable` | Language model unavailable |
| `circuit_breaker_open` | Circuit breaker is open for language |
| `emergency_fallback_active` | System in emergency fallback mode |
| `quality_check_failed` | Generated content failed quality check |

**Example Error Response:**
```json
{
  "error": "circuit_breaker_open",
  "message": "French language service is temporarily unavailable",
  "details": {
    "language": "fr",
    "circuit_state": "open",
    "fallback_available": true,
    "estimated_recovery": "2025-01-17T10:35:00Z"
  },
  "timestamp": "2025-01-17T10:30:00Z",
  "request_id": "req_12345"
}
```

## Rate Limiting

### Rate Limits

| Endpoint Category | Authenticated | Anonymous |
|------------------|---------------|-----------|
| Analysis Endpoints | 100/hour | 10/hour |
| Explanation Endpoints | 200/hour | 20/hour |
| Health Endpoints | 1000/hour | 100/hour |
| Monitoring Endpoints | 500/hour | N/A |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705656600
```

### Rate Limit Exceeded Response

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Try again in 1800 seconds.",
  "details": {
    "limit": 100,
    "window": 3600,
    "retry_after": 1800
  },
  "timestamp": "2025-01-17T10:30:00Z"
}
```

## Examples

### Complete Analysis Workflow

#### 1. Analyze Stock with French Explanation
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/analyze/AAPL/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "fr",
    "detail_level": "detailed",
    "explanation_type": "technical_analysis"
  }'
```

#### 2. Get Additional Language (Spanish)
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/explain/12345/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "es",
    "detail_level": "detailed",
    "explanation_type": "technical_analysis"
  }'
```

#### 3. Check System Health
```bash
curl "https://api.voyageurcompass.com/analytics/health/multilingual/"
```

### Bulk Operations

#### Generate Explanations for Multiple Stocks
```bash
curl -X POST "https://api.voyageurcompass.com/analytics/multilingual/bulk/" \
  -H "Authorization: Token your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_ids": [12345, 12346, 12347],
    "languages": ["fr", "es"],
    "detail_level": "summary"
  }'
```

### Emergency Scenarios

#### Check Emergency Status
```bash
curl "https://api.voyageurcompass.com/analytics/health/feature-flags/" \
  | jq '.emergency_status.emergency_fallback_enabled'
```

#### Get Recent Alerts
```bash
curl -H "Authorization: Token your_token" \
  "https://api.voyageurcompass.com/analytics/monitoring/alerts/?limit=10"
```

### Monitoring Integration

#### Prometheus Metrics
```bash
curl "https://api.voyageurcompass.com/analytics/metrics/"
```

#### Health Check for Load Balancer
```bash
curl "https://api.voyageurcompass.com/analytics/health/ping/"
```

### SDKs and Libraries

#### Python SDK Example
```python
import requests

class VoyageurCompassClient:
    def __init__(self, api_token, base_url="https://api.voyageurcompass.com"):
        self.api_token = api_token
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Token {api_token}',
            'Content-Type': 'application/json'
        })

    def analyze_stock(self, symbol, language='en', detail_level='summary'):
        """Analyze a stock with multilingual explanation."""
        url = f"{self.base_url}/analytics/analyze/{symbol}/"
        data = {
            'language': language,
            'detail_level': detail_level,
            'explanation_type': 'technical_analysis'
        }
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def get_explanation(self, analysis_id, language):
        """Get explanation for existing analysis."""
        url = f"{self.base_url}/analytics/explanation/{analysis_id}/"
        params = {'language': language}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def check_health(self):
        """Check multilingual system health."""
        url = f"{self.base_url}/analytics/health/multilingual/"
        response = requests.get(url)  # No auth required
        response.raise_for_status()
        return response.json()

# Usage example
client = VoyageurCompassClient('your_api_token')

# Analyze AAPL with French explanation
result = client.analyze_stock('AAPL', language='fr', detail_level='detailed')
print(f"Analysis score: {result['score_0_10']}")
print(f"French explanation: {result['explanation']['content']}")

# Check system health
health = client.check_health()
print(f"System status: {health['status']}")
```

#### JavaScript SDK Example
```javascript
class VoyageurCompassClient {
    constructor(apiToken, baseUrl = 'https://api.voyageurcompass.com') {
        this.apiToken = apiToken;
        this.baseUrl = baseUrl;
    }

    async analyzeStock(symbol, options = {}) {
        const { language = 'en', detailLevel = 'summary' } = options;

        const response = await fetch(`${this.baseUrl}/analytics/analyze/${symbol}/`, {
            method: 'POST',
            headers: {
                'Authorization': `Token ${this.apiToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                language,
                detail_level: detailLevel,
                explanation_type: 'technical_analysis'
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        return response.json();
    }

    async checkHealth() {
        const response = await fetch(`${this.baseUrl}/analytics/health/multilingual/`);

        if (!response.ok) {
            throw new Error(`Health Check Failed: ${response.status}`);
        }

        return response.json();
    }
}

// Usage example
const client = new VoyageurCompassClient('your_api_token');

// Analyze AAPL with Spanish explanation
client.analyzeStock('AAPL', { language: 'es', detailLevel: 'detailed' })
    .then(result => {
        console.log(`Analysis score: ${result.score_0_10}`);
        console.log(`Spanish explanation: ${result.explanation.content}`);
    })
    .catch(error => {
        console.error('Analysis failed:', error);
    });
```

---

## API Versioning

Current API version: **v1**

Version is specified in the URL: `/analytics/v1/...`

## Support

- **Documentation:** https://docs.voyageurcompass.com/api/
- **Support Email:** api-support@voyageurcompass.com
- **Status Page:** https://status.voyageurcompass.com
- **Discord Community:** https://discord.gg/voyageurcompass

---

*Last Updated: January 2025*
*API Version: 1.0*