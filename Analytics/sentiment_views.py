"""
Sentiment Analysis API Views for Analytics app.
Provides endpoints for stock sentiment analysis based on news data.
"""

from django.core.cache import cache
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from Analytics.services.sentiment_analyzer import get_sentiment_analyzer
from Data.services.yahoo_finance import yahoo_finance_service
from Data.models import AnalyticsResults, Stock


class SentimentThrottle(UserRateThrottle):
    """Custom throttle for sentiment analysis endpoints."""
    rate = '50/hour'


@extend_schema(
    summary="Get sentiment analysis for a stock",
    description="Analyze sentiment from recent news articles for a specific stock",
    parameters=[
        OpenApiParameter(
            name='symbol',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description='Stock ticker symbol (e.g., AAPL, MSFT)'
        ),
        OpenApiParameter(
            name='days',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Number of days of news history to analyze (default: 90)'
        ),
        OpenApiParameter(
            name='refresh',
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Force refresh sentiment analysis (default: false)'
        ),
    ],
    responses={
        200: {
            'description': 'Sentiment analysis results',
            'example': {
                'symbol': 'AAPL',
                'sentimentScore': 0.42,
                'sentimentLabel': 'positive',
                'confidence': 0.78,
                'newsCount': 25,
                'lastNewsDate': '2024-01-15T10:30:00Z',
                'distribution': {
                    'positive': 15,
                    'negative': 5,
                    'neutral': 5
                },
                'sources': {
                    'Reuters': {'count': 10, 'avg_score': 0.5},
                    'Bloomberg': {'count': 8, 'avg_score': 0.3}
                },
                'articles': []
            }
        }
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([SentimentThrottle])
def stock_sentiment(request, symbol):
    """
    Get sentiment analysis for a specific stock.
    
    This endpoint analyzes recent news articles to determine market sentiment.
    Results are cached for 5 minutes unless refresh is requested.
    """
    try:
        symbol = symbol.upper()
        days = int(request.GET.get('days', 90))
        refresh = request.GET.get('refresh', 'false').lower() == 'true'
        
        # Check cache first
        sentiment_analyzer = get_sentiment_analyzer()
        cache_key = sentiment_analyzer.generateCacheKey(symbol=symbol, days=days)
        
        if not refresh:
            cached_result = sentiment_analyzer.getCachedSentiment(
                cache_key, symbol, is_recent=(days <= 30)
            )
            if cached_result:
                return Response(cached_result)
        
        # Get sentiment analyzer
        sentiment_analyzer = get_sentiment_analyzer()
        
        # Fetch news
        news_items = yahoo_finance_service.fetchNewsForStock(symbol, days=days, max_items=50)
        
        if not news_items:
            result = {
                'symbol': symbol,
                'sentimentScore': 0.0,
                'sentimentLabel': 'neutral',
                'confidence': 0.0,
                'newsCount': 0,
                'message': 'No news articles found for analysis'
            }
            return Response(result)
        
        # Prepare texts for analysis
        texts = []
        for article in news_items:
            text = yahoo_finance_service.preprocessNewsText(article)
            if text:
                texts.append(text)
        
        # Analyze sentiment
        sentiments = sentiment_analyzer.analyzeSentimentBatch(texts[:30])
        aggregated = sentiment_analyzer.aggregateSentiment(sentiments)
        
        # Get the most recent analysis result from database if available
        try:
            latest_analysis = AnalyticsResults.objects.filter(
                stock__symbol=symbol
            ).latest('as_of')
            
            db_sentiment = {
                'dbSentimentScore': latest_analysis.sentimentScore,
                'dbSentimentLabel': latest_analysis.sentimentLabel,
                'dbNewsCount': latest_analysis.newsCount,
            }
        except AnalyticsResults.DoesNotExist:
            db_sentiment = {}
        
        # Build response
        result = {
            'symbol': symbol,
            'sentimentScore': aggregated.get('sentimentScore', 0.0),
            'sentimentLabel': aggregated.get('sentimentLabel', 'neutral'),
            'confidence': aggregated.get('sentimentConfidence', 0.0),
            'newsCount': len(texts),
            'lastNewsDate': news_items[0]['publishedDate'] if news_items else None,
            'distribution': aggregated.get('distribution', {}),
            'sources': aggregated.get('sourceBreakdown', {}),
            'articles': [
                {
                    'title': article['title'],
                    'publishedDate': article['publishedDate'],
                    'source': article['source'],
                    'sentiment': sentiments[i] if i < len(sentiments) else None
                }
                for i, article in enumerate(news_items[:5])  # Include top 5 articles
            ],
            **db_sentiment
        }
        
        # Cache result with appropriate TTL
        sentiment_analyzer.setCachedSentiment(
            cache_key, result, symbol, is_recent=(days <= 30)
        )
        
        return Response(result)
        
    except Stock.DoesNotExist:
        return Response({
            'error': f'Stock {symbol} not found'
        }, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)