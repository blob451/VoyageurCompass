# VoyageurCompass API Documentation

## Overview

The VoyageurCompass API provides comprehensive financial analysis capabilities powered by advanced AI systems, including LLaMA 3.1 language models and FinBERT sentiment analysis. This documentation covers all available endpoints, request/response formats, and integration examples.

## Base URL
```
https://api.voyageurcompass.com/api/
```

## Authentication

All API requests require JWT authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Authentication Endpoints

#### Login
```http
POST /auth/login/
```

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access": "string",
  "refresh": "string",
  "user": {
    "id": 1,
    "username": "string",
    "email": "string"
  }
}
```

#### Token Refresh
```http
POST /auth/refresh/
```

**Request Body:**
```json
{
  "refresh": "string"
}
```

**Response:**
```json
{
  "access": "string",
  "refresh": "string"
}
```

#### Logout
```http
POST /auth/logout/
```

**Request Body:**
```json
{
  "refresh": "string"
}
```

## Analytics Endpoints

### Enhanced Financial Explanation Generation

#### Generate AI-Enhanced Financial Explanation
```http
POST /analytics/explain/
```

Generate professional investment analysis using LLaMA 3.1 models with optional FinBERT sentiment integration.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "analysis_data": {
    "score_0_10": 7.8,
    "weighted_scores": {
      "w_sma50vs200": 0.15,
      "w_rsi14": 0.08,
      "w_macd12269": 0.12,
      "w_bbpos20": 0.05,
      "w_volsurge": 0.11
    },
    "news_articles": [
      {
        "title": "Apple Reports Strong Q3 Earnings",
        "summary": "Revenue exceeded expectations with strong iPhone sales..."
      }
    ]
  },
  "detail_level": "standard",
  "include_sentiment": true,
  "use_ensemble": false
}
```

**Parameters:**
- `symbol` (string, required): Stock symbol
- `analysis_data` (object, required): Technical analysis data
- `detail_level` (string, optional): "summary", "standard", or "detailed". Default: "standard"
- `include_sentiment` (boolean, optional): Enable FinBERT sentiment analysis. Default: true
- `use_ensemble` (boolean, optional): Use multi-model ensemble. Default: false

**Response:**
```json
{
  "content": "**BUY** - AAPL demonstrates strong technical momentum with a composite score of 7.8/10. The SMA 50/200 crossover indicates bullish momentum (+0.15), while RSI(14) shows moderate bullish divergence (+0.08). Current market sentiment is positive (confidence: 85%, score: +0.42), aligning well with technical indicators. Price target: $185.",
  "recommendation": "BUY",
  "confidence_score": 0.85,
  "generation_time": 1.23,
  "model_used": "llama3.1:8b",
  "cached": false,
  "quality_metrics": {
    "recommendation_clarity": 0.95,
    "technical_coverage": 0.78,
    "content_quality": 0.88,
    "sentiment_integration": 0.82
  },
  "sentiment_data": {
    "sentiment_score": 0.42,
    "sentiment_label": "positive",
    "confidence": 0.85,
    "news_count": 3
  },
  "metadata": {
    "cache_ttl": 180,
    "complexity_score": 0.6,
    "prompt_tokens": 156,
    "completion_tokens": 98
  }
}
```

#### Generate Ensemble Financial Explanation
```http
POST /analytics/explain/ensemble/
```

Generate financial explanation using multi-model ensemble for enhanced accuracy and reliability.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "analysis_data": {
    "score_0_10": 7.8,
    "weighted_scores": {
      "w_sma50vs200": 0.15,
      "w_rsi14": 0.08,
      "w_macd12269": 0.12
    }
  },
  "detail_level": "detailed",
  "strategy": "confidence_weighted",
  "return_all_predictions": false,
  "timeout": 30
}
```

**Parameters:**
- `strategy` (string, optional): "majority_vote", "confidence_weighted", "performance_weighted", or "adaptive_weighted". Default: "confidence_weighted"
- `return_all_predictions` (boolean, optional): Include individual model predictions. Default: false
- `timeout` (integer, optional): Generation timeout in seconds. Default: 30

**Response:**
```json
{
  "content": "**BUY** - AAPL shows exceptional technical and sentiment alignment...",
  "consensus_recommendation": "BUY",
  "consensus_strength": 0.89,
  "generation_time": 2.45,
  "models_used": ["llama3.1:8b", "llama3.1:70b", "sentiment_enhanced"],
  "strategy_applied": "confidence_weighted",
  "quality_metrics": {
    "recommendation_clarity": 0.92,
    "technical_coverage": 0.85,
    "content_quality": 0.91,
    "consensus_strength": 0.89
  },
  "individual_predictions": [
    {
      "model": "llama3.1:8b",
      "content": "Technical analysis suggests...",
      "confidence": 0.84,
      "weight": 1.0,
      "generation_time": 1.2
    }
  ]
}
```

### Technical Analysis Endpoints

#### Get Complete Technical Analysis
```http
GET /analytics/analysis/{symbol}/
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-09-03T10:30:00Z",
  "score_0_10": 7.8,
  "recommendation": "BUY",
  "weighted_scores": {
    "w_sma50vs200": 0.15,
    "w_price_vs_sma50": 0.12,
    "w_rsi14": 0.08,
    "w_macd12269": 0.12,
    "w_bbpos20": 0.05,
    "w_bbwidth20": 0.03,
    "w_volsurge": 0.11,
    "w_obv20": 0.04,
    "w_rel1y": 0.06,
    "w_rel2y": 0.05,
    "w_support_resistance": 0.07,
    "w_candlestick_patterns": 0.08
  },
  "raw_indicators": {
    "sma_50": 175.23,
    "sma_200": 168.45,
    "rsi_14": 62.5,
    "macd_line": 2.45,
    "macd_signal": 1.89,
    "bb_upper": 178.90,
    "bb_lower": 171.20,
    "volume_avg_20": 45670000
  },
  "components": {
    "trend": {
      "direction": "bullish",
      "strength": "moderate",
      "score": 0.72
    },
    "momentum": {
      "direction": "bullish",
      "strength": "strong",
      "score": 0.85
    },
    "volatility": {
      "level": "normal",
      "trend": "decreasing",
      "score": 0.65
    }
  }
}
```

#### Get Performance Metrics
```http
GET /analytics/performance/{symbol}/
```

**Query Parameters:**
- `period` (string, optional): "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y". Default: "1y"

**Response:**
```json
{
  "symbol": "AAPL",
  "period": "1y",
  "performance": {
    "total_return": 0.245,
    "annualised_return": 0.245,
    "volatility": 0.28,
    "sharpe_ratio": 0.87,
    "max_drawdown": -0.15,
    "beta": 1.2
  },
  "relative_performance": {
    "vs_sp500": 0.08,
    "vs_nasdaq": 0.05,
    "vs_sector": 0.12
  },
  "momentum_indicators": {
    "rsi_14": 62.5,
    "momentum_1mo": 0.08,
    "momentum_3mo": 0.15,
    "momentum_6mo": 0.22
  }
}
```

### Sentiment Analysis Endpoints

#### Get Stock Sentiment Analysis
```http
GET /analytics/sentiment/{symbol}/
```

**Query Parameters:**
- `lookback_days` (integer, optional): Number of days to analyse. Default: 7
- `min_confidence` (float, optional): Minimum confidence threshold. Default: 0.6

**Response:**
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-09-03T10:30:00Z",
  "sentiment_score": 0.42,
  "sentiment_label": "positive",
  "confidence": 0.85,
  "news_count": 15,
  "lookback_days": 7,
  "sentiment_distribution": {
    "positive": 0.67,
    "neutral": 0.20,
    "negative": 0.13
  },
  "confidence_distribution": {
    "high": 0.73,
    "medium": 0.20,
    "low": 0.07
  },
  "recent_news": [
    {
      "title": "Apple Announces New Product Line",
      "sentiment_score": 0.65,
      "confidence": 0.89,
      "timestamp": "2025-09-03T08:00:00Z"
    }
  ]
}
```

### Service Status and Monitoring

#### Get AI Service Status
```http
GET /analytics/status/
```

**Response:**
```json
{
  "service_status": "healthy",
  "models_available": {
    "llama3.1:8b": {
      "status": "available",
      "response_time": 1.2,
      "last_check": "2025-09-03T10:29:45Z"
    },
    "llama3.1:70b": {
      "status": "available",
      "response_time": 2.8,
      "last_check": "2025-09-03T10:29:45Z"
    }
  },
  "circuit_breaker_state": "CLOSED",
  "performance_metrics": {
    "total_requests": 1247,
    "success_rate": 0.996,
    "avg_generation_time": 1.42,
    "cache_hit_rate": 0.78,
    "uptime_minutes": 4320
  },
  "sentiment_service": {
    "status": "available",
    "model_loaded": true,
    "avg_processing_time": 0.15
  }
}
```

#### Get Performance Metrics
```http
GET /analytics/metrics/
```

**Query Parameters:**
- `timeframe` (string, optional): "1h", "6h", "24h", "7d". Default: "24h"

**Response:**
```json
{
  "timeframe": "24h",
  "request_metrics": {
    "total_requests": 342,
    "success_rate": 0.995,
    "error_rate": 0.005,
    "avg_response_time": 1.38
  },
  "model_usage": {
    "llama3.1:8b": {
      "requests": 268,
      "avg_time": 1.2,
      "success_rate": 0.996
    },
    "llama3.1:70b": {
      "requests": 74,
      "avg_time": 2.6,
      "success_rate": 0.993
    }
  },
  "quality_metrics": {
    "avg_recommendation_clarity": 0.87,
    "avg_technical_coverage": 0.76,
    "avg_content_quality": 0.89,
    "user_satisfaction": 0.92
  },
  "cache_performance": {
    "hit_rate": 0.79,
    "miss_rate": 0.21,
    "eviction_rate": 0.03,
    "avg_ttl": 165
  }
}
```

## Data Endpoints

### Stock Data

#### Get Stock List
```http
GET /data/stocks/
```

**Query Parameters:**
- `search` (string, optional): Search by symbol or company name
- `sector` (string, optional): Filter by sector
- `limit` (integer, optional): Number of results. Default: 50

#### Get Stock Details
```http
GET /data/stocks/{symbol}/
```

**Response:**
```json
{
  "symbol": "AAPL",
  "company_name": "Apple Inc.",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "market_cap": 2800000000000,
  "price": 175.25,
  "change": 2.45,
  "change_percent": 0.014,
  "volume": 52000000,
  "avg_volume": 48000000,
  "pe_ratio": 28.5,
  "dividend_yield": 0.005,
  "52_week_high": 198.23,
  "52_week_low": 124.17
}
```

### Portfolio Management

#### Get User Portfolio
```http
GET /data/portfolio/
```

#### Add Portfolio Holding
```http
POST /data/portfolio/holdings/
```

**Request Body:**
```json
{
  "symbol": "AAPL",
  "shares": 100,
  "purchase_price": 150.00,
  "purchase_date": "2024-01-15"
}
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid analysis data format",
    "details": {
      "field": "weighted_scores",
      "issue": "Missing required indicator scores"
    },
    "timestamp": "2025-09-03T10:30:00Z",
    "request_id": "req_abc123def456"
  }
}
```

### Error Codes

#### Authentication Errors (401)
- `INVALID_TOKEN`: JWT token is invalid or expired
- `TOKEN_EXPIRED`: Access token has expired
- `REFRESH_REQUIRED`: Token refresh required

#### Client Errors (400)
- `INVALID_REQUEST`: Request format or parameters invalid
- `MISSING_PARAMETER`: Required parameter missing
- `INVALID_SYMBOL`: Stock symbol not found or invalid
- `ANALYSIS_DATA_INVALID`: Technical analysis data format invalid

#### Service Errors (503)
- `LLM_SERVICE_UNAVAILABLE`: Language model service temporarily unavailable
- `SENTIMENT_SERVICE_UNAVAILABLE`: FinBERT sentiment analysis unavailable
- `CIRCUIT_BREAKER_OPEN`: Service temporarily disabled due to errors

#### Rate Limiting (429)
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded
- `GENERATION_LIMIT_EXCEEDED`: Daily explanation generation limit exceeded

## Rate Limits

### Standard Limits
- **Authentication**: 10 requests/minute
- **Technical Analysis**: 60 requests/minute
- **AI Explanations**: 30 requests/minute
- **Ensemble Generation**: 10 requests/minute
- **Service Status**: 120 requests/minute

### Premium Limits
- **Technical Analysis**: 120 requests/minute
- **AI Explanations**: 100 requests/minute
- **Ensemble Generation**: 30 requests/minute

## SDKs and Libraries

### Python SDK
```bash
pip install voyageur-compass-sdk
```

```python
from voyageur_compass import VoyageurClient

client = VoyageurClient(api_key="your-api-key")

# Generate AI explanation
response = client.explain_stock(
    symbol="AAPL",
    analysis_data={
        "score_0_10": 7.8,
        "weighted_scores": {...}
    },
    detail_level="standard",
    include_sentiment=True
)

print(response.content)
print(response.recommendation)
print(response.confidence_score)
```

### JavaScript SDK
```bash
npm install @voyageur-compass/sdk
```

```javascript
import { VoyageurClient } from '@voyageur-compass/sdk';

const client = new VoyageurClient({
  apiKey: 'your-api-key',
  baseURL: 'https://api.voyageurcompass.com'
});

// Generate AI explanation
const response = await client.explainStock({
  symbol: 'AAPL',
  analysisData: {
    score_0_10: 7.8,
    weighted_scores: {...}
  },
  detailLevel: 'standard',
  includeSentiment: true
});

console.log(response.content);
console.log(response.recommendation);
```

## Integration Examples

### Basic Stock Analysis Workflow
```python
import requests

# 1. Get technical analysis
analysis_response = requests.get(
    'https://api.voyageurcompass.com/api/analytics/analysis/AAPL/',
    headers={'Authorization': 'Bearer your-token'}
)
analysis_data = analysis_response.json()

# 2. Generate AI explanation
explain_response = requests.post(
    'https://api.voyageurcompass.com/api/analytics/explain/',
    headers={'Authorization': 'Bearer your-token'},
    json={
        'symbol': 'AAPL',
        'analysis_data': analysis_data,
        'detail_level': 'standard',
        'include_sentiment': True
    }
)
explanation = explain_response.json()

print(f"Recommendation: {explanation['recommendation']}")
print(f"Confidence: {explanation['confidence_score']:.2%}")
print(f"Explanation: {explanation['content']}")
```

### Ensemble Generation with Error Handling
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry strategy
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=0.3,
    status_forcelist=[502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

try:
    response = session.post(
        'https://api.voyageurcompass.com/api/analytics/explain/ensemble/',
        headers={'Authorization': 'Bearer your-token'},
        json={
            'symbol': 'AAPL',
            'analysis_data': analysis_data,
            'detail_level': 'detailed',
            'strategy': 'confidence_weighted',
            'timeout': 30
        },
        timeout=35
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Consensus: {result['consensus_recommendation']}")
        print(f"Strength: {result['consensus_strength']:.2%}")
        print(f"Models: {', '.join(result['models_used'])}")
    elif response.status_code == 503:
        print("Service temporarily unavailable, trying standard generation...")
        # Fallback to standard generation
    else:
        print(f"Error: {response.status_code} - {response.json()}")
        
except requests.exceptions.Timeout:
    print("Request timed out, service may be overloaded")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

### Real-time Sentiment Monitoring
```javascript
const WebSocket = require('ws');

// WebSocket connection for real-time sentiment updates
const ws = new WebSocket('wss://api.voyageurcompass.com/ws/sentiment/', {
  headers: {
    'Authorization': 'Bearer your-token'
  }
});

ws.on('open', function open() {
  // Subscribe to specific stock sentiments
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['AAPL', 'GOOGL', 'MSFT']
  }));
});

ws.on('message', function message(data) {
  const sentimentUpdate = JSON.parse(data);
  
  console.log(`${sentimentUpdate.symbol}: ${sentimentUpdate.sentiment_label} ` +
              `(${(sentimentUpdate.sentiment_score * 100).toFixed(1)}%) ` +
              `confidence: ${(sentimentUpdate.confidence * 100).toFixed(0)}%`);
  
  // Trigger explanation regeneration if sentiment significantly changed
  if (Math.abs(sentimentUpdate.sentiment_score) > 0.5 && sentimentUpdate.confidence > 0.8) {
    generateExplanation(sentimentUpdate.symbol);
  }
});
```

## Best Practices

### Performance Optimisation
1. **Use caching**: Enable caching for frequently requested explanations
2. **Batch requests**: Combine multiple symbol analyses when possible
3. **Choose appropriate detail levels**: Use "summary" for quick insights
4. **Monitor service status**: Check `/analytics/status/` before making requests

### Quality Maximisation
1. **Include sentiment data**: Enable sentiment analysis for enhanced explanations
2. **Use ensemble for critical decisions**: Ensemble generation provides higher accuracy
3. **Provide complete analysis data**: Include all available technical indicators
4. **Monitor quality metrics**: Track recommendation clarity and technical coverage

### Error Handling
1. **Implement retry logic**: Use exponential backoff for temporary failures
2. **Handle circuit breaker**: Fallback to cached or simplified responses
3. **Validate input data**: Ensure analysis data format is correct
4. **Monitor rate limits**: Implement request throttling to avoid limits

### Security
1. **Secure token storage**: Store JWT tokens securely
2. **Implement token refresh**: Handle automatic token renewal
3. **Validate responses**: Verify response structure and content
4. **Log security events**: Monitor for unusual API usage patterns

## Support and Resources

### Documentation Links
- **API Reference**: https://docs.voyageurcompass.com/api/
- **SDK Documentation**: https://docs.voyageurcompass.com/sdks/
- **Integration Guides**: https://docs.voyageurcompass.com/guides/
- **Best Practices**: https://docs.voyageurcompass.com/best-practices/

### Support Channels
- **Technical Support**: api-support@voyageurcompass.com
- **GitHub Issues**: https://github.com/voyageurcompass/api-issues
- **Community Forum**: https://community.voyageurcompass.com
- **Status Page**: https://status.voyageurcompass.com

### Rate Limit Increases
For higher rate limits or custom solutions, contact enterprise@voyageurcompass.com

---

*API Version: 2.1.0*  
*Last Updated: September 3, 2025*  
*Documentation Status: Current*