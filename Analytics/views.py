from django.shortcuts import render

# Create your views here.
"""
API Views for Analytics app.
Provides endpoints for stock analysis and trading signals.
"""

from datetime import datetime
from django.core.cache import cache
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from Analytics.engine.ta_engine import TechnicalAnalysisEngine
from Analytics.utils.analysis_logger import AnalysisLogger
from Analytics.services.explanation_service import get_explanation_service
from Data.services.yahoo_finance import yahoo_finance_service
from Data.models import Portfolio, AnalyticsResults, Stock



class AnalysisThrottle(UserRateThrottle):
    """Custom throttle for analysis endpoints."""
    rate = '100/hour'


@extend_schema(
    summary="Analyze a stock",
    description="Run comprehensive analysis on a stock including technical indicators and trading signals",
    parameters=[
        OpenApiParameter(
            name='symbol',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description='Stock ticker symbol (e.g., AAPL, MSFT)'
        ),
        OpenApiParameter(
            name='months',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Number of months to analyze (default: 6)'
        ),
        OpenApiParameter(
            name='sync',
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Sync latest data before analysis (default: false)'
        ),
        OpenApiParameter(
            name='include_explanation',
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Generate explanation using local LLaMA model (default: false)'
        ),
        OpenApiParameter(
            name='explanation_detail',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Explanation detail level: summary, standard, detailed (default: standard)'
        ),
    ],
    responses={
        200: {
            'description': 'Analysis completed successfully',
            'example': {
                'success': True,
                'symbol': 'AAPL',
                'signal': 'BUY',
                'analysis': {}
            }
        },
        400: {'description': 'Invalid parameters'},
        404: {'description': 'Stock not found'},
        429: {'description': 'Rate limit exceeded'}
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def analyze_stock(request, symbol):
    """
    Analyze a stock and generate trading signals.
    
    Path Parameters:
        symbol: Stock ticker symbol
    
    Query Parameters:
        months: Number of months to analyze (default: 6)
        sync: Whether to sync data before analysis (default: false)
        include_explanation: Generate explanation using local LLaMA model (default: false)
        explanation_detail: Explanation detail level (summary/standard/detailed, default: standard)
    
    Returns:
        Comprehensive analysis with trading signals and optional explanations
    """
    # IMMEDIATE DEBUG OUTPUT - This should appear first
    import sys
    print("*" * 100, flush=True)
    print(f"*** ANALYZE_STOCK VIEW CALLED FOR {symbol} ***", flush=True)
    print("*" * 100, flush=True)
    sys.stderr.write(f"STDERR: ANALYZE_STOCK CALLED FOR {symbol}\n")
    sys.stderr.flush()
    
    # Force Django logging - this WILL appear
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"=== DJANGO VIEW CALLED FOR {symbol} ===")

    # Also try print to stderr
    import sys
    print(f"=== STDERR: VIEW CALLED FOR {symbol} ===", file=sys.stderr)

    # Original code continues...
    print("="*50)
    print(f"DJANGO VIEW CALLED FOR {symbol}")
    print("="*50)

    # Get parameters
    symbol = symbol.upper()
    months = int(request.query_params.get('months', 6))
    sync = request.query_params.get('sync', 'false').lower() == 'true'
    include_explanation = request.query_params.get('include_explanation', 'false').lower() == 'true'
    explanation_detail = request.query_params.get('explanation_detail', 'standard')
    
    print(f"[BACKEND] Analysis request received for {symbol}")
    print(f"[BACKEND] Request from user: {request.user.username if hasattr(request.user, 'username') else 'Unknown'}")
    print(f"[BACKEND] Parameters - months: {months}, sync: {sync}, explanation: {include_explanation}, detail: {explanation_detail}")
    
    # Validate parameters
    if months < 1 or months > 24:
        return Response(
            {'error': 'Months parameter must be between 1 and 24'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate explanation detail level
    if explanation_detail not in ['summary', 'standard', 'detailed']:
        return Response(
            {'error': 'explanation_detail must be one of: summary, standard, detailed'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Sync data if explicitly requested (auto-sync is now handled by the engine)
    if sync:
        period = f"{months}mo" if months <= 12 else "2y"
        sync_result = yahoo_finance_service.get_stock_data(
            symbol, 
            period=period, 
            sync_db=True
        )
        
        if 'error' in sync_result:
            return Response(
                {'error': f'Failed to sync data: {sync_result["error"]}'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    # Run analysis
    try:
        print(f"[BACKEND] Starting analysis process for {symbol}")
        
        # Create analysis logger for web-based requests
        analysis_logger = None
        if request.user and hasattr(request.user, 'username'):
            try:
                print(f"[BACKEND] Creating analysis logger for user {request.user.username}")
                analysis_logger = AnalysisLogger(request.user.username, symbol)
                print(f"[BACKEND] AnalysisLogger created successfully: {analysis_logger.log_filename}")
            except Exception as logger_error:
                # Log the error but continue without logger
                print(f"[BACKEND] Failed to create analysis logger: {str(logger_error)}")
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create analysis logger for {symbol}: {str(logger_error)}")
        
        print(f"[BACKEND] Initializing TechnicalAnalysisEngine")
        engine = TechnicalAnalysisEngine()
        
        print(f"[BACKEND] Calling engine.analyze_stock() for {symbol}")
        analysis = engine.analyze_stock(symbol, user=request.user, logger_instance=analysis_logger)
        print(f"[BACKEND] Engine analysis completed successfully")
        
        # New engine always returns successful analysis or raises exception
        
        print(f"[BACKEND] Formatting response data for {symbol}")
        
        # Format response for new TA engine
        analysis_date = analysis.get('analysis_date', datetime.now())
        if hasattr(analysis_date, 'isoformat'):
            analysis_date_str = analysis_date.isoformat()
        else:
            analysis_date_str = str(analysis_date)
        
        print(f"[BACKEND] Processing weighted scores for response")
        weighted_scores_dict = analysis.get('weighted_scores', {})
        print(f"[BACKEND] Original weighted_scores type: {type(weighted_scores_dict)}")
        print(f"[BACKEND] Weighted scores keys: {list(weighted_scores_dict.keys()) if weighted_scores_dict else 'None'}")
        
        # Safe conversion of weighted scores
        safe_weighted_scores = {}
        for k, v in weighted_scores_dict.items():
            try:
                if v is not None:
                    safe_weighted_scores[k] = float(v)
                else:
                    safe_weighted_scores[k] = 0.0
                    print(f"[BACKEND] Warning: weighted_score {k} was None, defaulting to 0.0")
            except (ValueError, TypeError) as e:
                print(f"[BACKEND] Error converting weighted_score {k}={v}: {e}")
                safe_weighted_scores[k] = 0.0
            
        # CRITICAL FIX: Verify database transaction was successful before responding
        analytics_result_id = analysis.get('analytics_result_id')
        if analytics_result_id:
            try:
                # Verify the analysis was actually saved to the database
                from Data.models import AnalyticsResults
                saved_analysis = AnalyticsResults.objects.filter(
                    id=analytics_result_id,
                    user=request.user,
                    stock__symbol=symbol
                ).first()
                
                if not saved_analysis:
                    # Database save failed silently - this is the critical bug we're fixing
                    print(f"[BACKEND] CRITICAL ERROR: Analysis {analytics_result_id} not found in database for {symbol}")
                    logger.error(f"Analysis result ID {analytics_result_id} not found in database for {symbol} user {request.user.id}")
                    
                    # Instead of returning fake success, report the actual error
                    return Response(
                        {'error': f'Analysis completed but failed to save to database. Please try again.'},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                else:
                    print(f"[BACKEND] Verified analysis {analytics_result_id} saved successfully for {symbol}")
                    
                    # Double-check the analysis data matches what we're returning
                    if saved_analysis.stock.symbol != symbol:
                        print(f"[BACKEND] CRITICAL ERROR: Analysis ID {analytics_result_id} belongs to {saved_analysis.stock.symbol}, not {symbol}")
                        logger.error(f"Analysis result ID mismatch: expected {symbol}, found {saved_analysis.stock.symbol}")
                        
                        return Response(
                            {'error': f'Analysis result ID mismatch detected. Please try again.'},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )
                        
            except Exception as db_error:
                print(f"[BACKEND] Database verification error for {symbol}: {str(db_error)}")
                logger.error(f"Database verification failed for {symbol}: {str(db_error)}", exc_info=True)
                
                return Response(
                    {'error': f'Failed to verify analysis save: {str(db_error)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        else:
            print(f"[BACKEND] WARNING: No analytics_result_id returned for {symbol}")
            logger.warning(f"No analytics_result_id returned for {symbol} analysis")

        # Generate explanation if requested
        explanation_data = None
        if include_explanation and analytics_result_id:
            try:
                print(f"[BACKEND] Generating explanation for {symbol} (detail: {explanation_detail})")
                explanation_service = get_explanation_service()
                
                if explanation_service.is_enabled():
                    # Get the saved analysis result for explanation
                    saved_analysis = AnalyticsResults.objects.filter(
                        id=analytics_result_id,
                        user=request.user,
                        stock__symbol=symbol
                    ).first()
                    
                    if saved_analysis:
                        explanation_result = explanation_service.explain_prediction_single(
                            saved_analysis, 
                            detail_level=explanation_detail,
                            user=request.user
                        )
                        
                        if explanation_result:
                            # Update the database with explanation
                            saved_analysis.explanations_json = {
                                'indicators_explained': explanation_result.get('indicators_explained', []),
                                'risk_factors': explanation_result.get('risk_factors', []),
                                'recommendation': explanation_result.get('recommendation', 'HOLD')
                            }
                            saved_analysis.explanation_method = explanation_result.get('method', 'unknown')
                            saved_analysis.explanation_version = '1.0'
                            saved_analysis.narrative_text = explanation_result.get('content', '')
                            saved_analysis.explanation_confidence = explanation_result.get('confidence_score', 0.0)
                            saved_analysis.explained_at = datetime.now()
                            saved_analysis.save()
                            
                            explanation_data = {
                                'content': explanation_result.get('content', ''),
                                'confidence_score': explanation_result.get('confidence_score', 0.0),
                                'detail_level': explanation_detail,
                                'method': explanation_result.get('method', 'unknown'),
                                'generation_time': explanation_result.get('generation_time', 0.0),
                                'word_count': explanation_result.get('word_count', 0),
                                'indicators_explained': explanation_result.get('indicators_explained', []),
                                'risk_factors': explanation_result.get('risk_factors', []),
                                'recommendation': explanation_result.get('recommendation', 'HOLD')
                            }
                            
                            print(f"[BACKEND] Explanation generated successfully for {symbol}")
                        else:
                            print(f"[BACKEND] Failed to generate explanation for {symbol}")
                    else:
                        print(f"[BACKEND] Could not find saved analysis for explanation")
                else:
                    print(f"[BACKEND] Explanation service not available")
                    explanation_data = {
                        'error': 'Explanation service not available',
                        'detail_level': explanation_detail,
                        'method': 'unavailable'
                    }
                    
            except Exception as e:
                print(f"[BACKEND] Error generating explanation: {str(e)}")
                logger.error(f"Explanation generation failed for {symbol}: {str(e)}")
                explanation_data = {
                    'error': f'Explanation generation failed: {str(e)}',
                    'detail_level': explanation_detail,
                    'method': 'error'
                }

        response_data = {
            'success': True,
            'symbol': analysis.get('symbol', symbol),
            'analysis_date': analysis_date_str,
            'horizon': analysis.get('horizon', 'unknown'),
            'composite_score': analysis.get('score_0_10', 0.0),
            'composite_raw': analysis.get('composite_raw'),
            'indicators': analysis.get('components', {}),
            'weighted_scores': safe_weighted_scores,
            'analytics_result_id': analytics_result_id
        }
        
        # Add explanation to response if generated
        if explanation_data:
            response_data['explanation'] = explanation_data
        
        print(f"[BACKEND] Response formatted successfully")
        print(f"[BACKEND] Response data keys: {list(response_data.keys())}")
        print(f"[BACKEND] Database transaction verified, returning response for {symbol}")
        
        # Invalidate analysis history cache for this user since we just created a new analysis
        user_cache_pattern = f"analysis_history:{request.user.id}:*"
        # Django cache doesn't support wildcard deletion, so we'll use a simple approach
        # In production, consider using Redis directly for pattern-based cache invalidation
        cache.delete_many([
            f"analysis_history:{request.user.id}:20:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
            f"analysis_history:{request.user.id}:10:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
            f"analysis_history:{request.user.id}:25:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
            f"analysis_history:{request.user.id}:50:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
        ])
        
        return Response(response_data)
        
    except Exception as e:
        print(f"[BACKEND] EXCEPTION occurred during analysis for {symbol}: {str(e)}")
        print(f"[BACKEND] Exception type: {type(e).__name__}")
        import traceback
        print(f"[BACKEND] Full traceback: {traceback.format_exc()}")
        
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Analysis failed for {symbol}: {str(e)}", exc_info=True)
        return Response(
            {'error': f'Analysis failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Analyze portfolio",
    description="Run analysis on all stocks in a portfolio",
    parameters=[
        OpenApiParameter(
            name='portfolio_id',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description='Portfolio ID'
        ),
    ],
    responses={
        200: {'description': 'Portfolio analysis completed'},
        403: {'description': 'Not authorized to access this portfolio'},
        404: {'description': 'Portfolio not found'}
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analyze_portfolio(request, portfolio_id):
    """
    Analyze all stocks in a portfolio.
    
    Path Parameters:
        portfolio_id: Portfolio ID
    
    Returns:
        Analysis results for all holdings
    """
    # Check portfolio ownership
    try:
        portfolio = Portfolio.objects.get(id=portfolio_id)
        
        if portfolio.user != request.user:
            return Response(
                {'error': 'You are not authorized to analyze this portfolio'},
                status=status.HTTP_403_FORBIDDEN
            )
    except Portfolio.DoesNotExist:
        return Response(
            {'error': 'Portfolio not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Run portfolio analysis (simplified implementation)
    try:
        # For now, disable portfolio analysis - could be implemented later
        # by running individual stock analyses for each holding
        return Response(
            {'error': 'Portfolio analysis not yet implemented with new TA engine'},
            status=status.HTTP_501_NOT_IMPLEMENTED
        )        
        if not analysis.get('success', True):
            return Response(
                {'error': analysis.get('error', 'Analysis failed')},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response(analysis)
        
    except Exception as e:
        return Response(
            {'error': f'Portfolio analysis failed: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Batch stock analysis",
    description="Analyze multiple stocks at once",
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'symbols': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of stock symbols',
                    'example': ['AAPL', 'MSFT', 'GOOGL']
                },
                'months': {
                    'type': 'integer',
                    'description': 'Analysis period in months',
                    'default': 6
                }
            },
            'required': ['symbols']
        }
    },
    responses={
        200: {'description': 'Batch analysis completed'},
        400: {'description': 'Invalid request data'}
    }
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([AnalysisThrottle])
def batch_analysis(request):
    """
    Analyze multiple stocks in a single request.
    
    Request Body:
        symbols: List of stock symbols (max 10)
        months: Analysis period (default: 6)
    
    Returns:
        Analysis results for all requested stocks
    """
    symbols = request.data.get('symbols', [])
    months = int(request.data.get('months', 6))
    
    # Validate input
    if not symbols:
        return Response(
            {'error': 'Symbols list is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if len(symbols) > 10:
        return Response(
            {'error': 'Maximum 10 symbols allowed per batch'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if months < 1 or months > 24:
        return Response(
            {'error': 'Months parameter must be between 1 and 24'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Run analysis for each symbol with new TA engine
    results = {}
    engine = TechnicalAnalysisEngine()
    
    for symbol in symbols:
        symbol = symbol.upper()
        try:
            # Create analysis logger for each symbol in batch
            analysis_logger = None
            if request.user and hasattr(request.user, 'username'):
                try:
                    analysis_logger = AnalysisLogger(request.user.username, symbol)
                except Exception as logger_error:
                    # Log the error but continue without logger
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to create analysis logger for {symbol}: {str(logger_error)}")
            
            analysis = engine.analyze_stock(symbol, user=request.user, logger_instance=analysis_logger)
            
            results[symbol] = {
                'success': True,
                'composite_score': analysis['score_0_10'],
                'composite_raw': analysis['composite_raw'],
                'analysis_date': analysis['analysis_date'].isoformat(),
                'horizon': analysis['horizon']
            }
        except Exception as e:
            results[symbol] = {
                'success': False,
                'error': str(e)
            }
    
    return Response({
        'success': True,
        'results': results,
        'total_analyzed': len(results),
        'successful': sum(1 for r in results.values() if r.get('success'))
    })


@extend_schema(
    summary="Get market overview",
    description="Get analysis of major market indices",
    responses={
        200: {'description': 'Market overview data'}
    }
)
@api_view(['GET'])
@permission_classes([AllowAny])
def market_overview(request):
    """
    Get market overview with major indices analysis.
    
    Returns:
        Analysis of S&P 500, Dow Jones, and NASDAQ
    """
    indices = {
        'SPY': 'S&P 500 ETF',
        'DIA': 'Dow Jones ETF',
        'QQQ': 'NASDAQ ETF',
    }
    
    results = {}
    engine = TechnicalAnalysisEngine()
    
    for symbol, name in indices.items():
        try:
            analysis = engine.analyze_stock(symbol, horizon='short')
            
            results[symbol] = {
                'name': name,
                'composite_score': analysis['score_0_10'],
                'analysis_date': analysis['analysis_date'].isoformat(),
                'horizon': analysis['horizon']
            }
        except:
            results[symbol] = {
                'name': name,
                'error': 'Data unavailable'
            }
    
    # Get market status
    market_status = yahoo_finance_service.get_market_status()
    
    return Response({
        'market_status': market_status,
        'indices': results,
        'timestamp': datetime.now().isoformat()
    })


@extend_schema(
    summary="Get user analysis history",
    description="Get the authenticated user's stock analysis history",
    parameters=[
        OpenApiParameter(
            name='limit',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Maximum number of results to return (default: 20)'
        ),
        OpenApiParameter(
            name='symbol',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description='Filter by stock symbol'
        ),
    ],
    responses={
        200: {
            'description': 'User analysis history retrieved successfully',
            'example': {
                'analyses': [
                    {
                        'id': 123,
                        'symbol': 'AAPL',
                        'score': 7,
                        'analysis_date': '2025-01-14T10:30:00Z',
                        'sector': 'Technology',
                        'industry': 'Consumer Electronics'
                    }
                ],
                'count': 1
            }
        }
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_analysis_history(request):
    """
    Get the authenticated user's analysis history with pagination and field selection.
    
    Query Parameters:
        limit: Maximum number of results per page (default: 20, max: 100)
        offset: Number of results to skip (default: 0)
        symbol: Filter by stock symbol (optional)
        fields: Comma-separated list of fields to include (optional)
                Available: id,symbol,name,score,analysis_date,sector,industry,horizon,composite_raw,components
    
    Returns:
        Paginated list of user's analysis results with metadata
    """
    limit = min(int(request.query_params.get('limit', 20)), 100)  # Cap at 100
    offset = int(request.query_params.get('offset', 0))
    symbol_filter = request.query_params.get('symbol')
    fields_param = request.query_params.get('fields')
    
    # Parse requested fields
    if fields_param:
        requested_fields = set(fields_param.split(','))
        # Validate fields
        valid_fields = {'id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw', 'components'}
        requested_fields = requested_fields.intersection(valid_fields)
    else:
        # Default fields (excluding heavy components field)
        requested_fields = {'id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw'}
    
    # Generate cache key based on user, pagination, filters, and fields
    cache_key = f"analysis_history:{request.user.id}:{limit}:{offset}:{symbol_filter or 'all'}:{','.join(sorted(requested_fields))}"
    
    # Try to get from cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return Response(cached_result)
    
    # Build queryset with optimizations
    queryset = AnalyticsResults.objects.filter(
        user=request.user
    ).select_related('stock').order_by('-as_of')
    
    # Apply symbol filter
    if symbol_filter:
        queryset = queryset.filter(stock__symbol__iexact=symbol_filter)
    
    # Get total count before pagination
    total_count = queryset.count()
    
    # Apply pagination
    paginated_queryset = queryset[offset:offset + limit]
    
    # Serialize the results with field selection
    analyses = []
    for result in paginated_queryset:
        analysis_data = {}
        
        if 'id' in requested_fields:
            analysis_data['id'] = result.id
        if 'symbol' in requested_fields:
            analysis_data['symbol'] = result.stock.symbol
        if 'name' in requested_fields:
            analysis_data['name'] = result.stock.short_name or result.stock.long_name
        if 'score' in requested_fields:
            analysis_data['score'] = result.score_0_10
        if 'analysis_date' in requested_fields:
            analysis_data['analysis_date'] = result.as_of.isoformat()
        if 'sector' in requested_fields:
            analysis_data['sector'] = result.stock.sector or 'Unknown'
        if 'industry' in requested_fields:
            analysis_data['industry'] = result.stock.industry or 'Unknown'
        if 'horizon' in requested_fields:
            analysis_data['horizon'] = result.horizon
        if 'composite_raw' in requested_fields:
            analysis_data['composite_raw'] = float(result.composite_raw) if result.composite_raw else None
        if 'components' in requested_fields:
            analysis_data['components'] = result.components
            
        analyses.append(analysis_data)
    
    # Calculate pagination metadata
    has_next = (offset + limit) < total_count
    has_previous = offset > 0
    
    # Prepare response data
    response_data = {
        'analyses': analyses,
        'pagination': {
            'count': len(analyses),
            'total_count': total_count,
            'limit': limit,
            'offset': offset,
            'has_next': has_next,
            'has_previous': has_previous,
            'next_offset': offset + limit if has_next else None,
            'previous_offset': max(0, offset - limit) if has_previous else None
        }
    }
    
    # Cache the result for 5 minutes (300 seconds)
    cache.set(cache_key, response_data, 300)
    
    return Response(response_data)


@extend_schema(
    summary="Get analysis by ID",
    description="Get a specific analysis result by its ID",
    parameters=[
        OpenApiParameter(
            name='analysis_id',
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description='Analysis result ID'
        ),
    ],
    responses={
        200: {'description': 'Analysis found'},
        404: {'description': 'Analysis not found'},
        403: {'description': 'Not authorized to view this analysis'}
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_analysis_by_id(request, analysis_id):
    """
    Get a specific analysis result by its ID.
    
    Path Parameters:
        analysis_id: Analysis result ID
    
    Returns:
        Analysis result details
    """
    try:
        # Get the analysis result and ensure it belongs to the user
        result = AnalyticsResults.objects.filter(
            id=analysis_id,
            user=request.user
        ).select_related('stock').first()
        
        if not result:
            return Response(
                {'error': 'Analysis not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Format the response - same structure as get_user_latest_analysis
        analysis_data = {
            'id': result.id,
            'symbol': result.stock.symbol,
            'name': result.stock.short_name or result.stock.long_name or f'{result.stock.symbol} Corporation',
            'sector': result.stock.sector or 'Unknown',
            'industry': result.stock.industry or 'Unknown',
            'score': result.score_0_10,
            'analysis_date': result.as_of.isoformat(),
            'horizon': result.horizon,
            'composite_raw': float(result.composite_raw) if result.composite_raw else None,
            'indicators': result.components,
            'weighted_scores': {
                'w_sma50vs200': float(result.w_sma50vs200) if result.w_sma50vs200 else None,
                'w_pricevs50': float(result.w_pricevs50) if result.w_pricevs50 else None,
                'w_rsi14': float(result.w_rsi14) if result.w_rsi14 else None,
                'w_macd12269': float(result.w_macd12269) if result.w_macd12269 else None,
                'w_bbpos20': float(result.w_bbpos20) if result.w_bbpos20 else None,
                'w_bbwidth20': float(result.w_bbwidth20) if result.w_bbwidth20 else None,
                'w_volsurge': float(result.w_volsurge) if result.w_volsurge else None,
                'w_obv20': float(result.w_obv20) if result.w_obv20 else None,
                'w_rel1y': float(result.w_rel1y) if result.w_rel1y else None,
                'w_rel2y': float(result.w_rel2y) if result.w_rel2y else None,
                'w_candlerev': float(result.w_candlerev) if result.w_candlerev else None,
                'w_srcontext': float(result.w_srcontext) if result.w_srcontext else None,
            }
        }
        
        return Response(analysis_data)
        
    except Exception as e:
        return Response(
            {'error': f'Failed to retrieve analysis: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get user's latest analysis for a stock",
    description="Get the user's most recent analysis for a specific stock",
    parameters=[
        OpenApiParameter(
            name='symbol',
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description='Stock ticker symbol'
        ),
    ],
    responses={
        200: {'description': 'Latest analysis found'},
        404: {'description': 'No analysis found for this stock'}
    }
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_latest_analysis(request, symbol):
    """
    Get the user's latest analysis for a specific stock.
    
    Path Parameters:
        symbol: Stock ticker symbol
    
    Returns:
        Latest analysis result for the stock, or 404 if not found
    """
    symbol = symbol.upper()
    
    try:
        # Get the user's latest analysis for this stock
        result = AnalyticsResults.objects.filter(
            user=request.user,
            stock__symbol=symbol
        ).select_related('stock').order_by('-as_of').first()
        
        if not result:
            return Response(
                {'error': f'No analysis found for {symbol}'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Format the response
        analysis_data = {
            'id': result.id,
            'symbol': result.stock.symbol,
            'name': result.stock.short_name or result.stock.long_name or f'{symbol} Corporation',
            'sector': result.stock.sector or 'Unknown',
            'industry': result.stock.industry or 'Unknown',
            'score': result.score_0_10,
            'analysis_date': result.as_of.isoformat(),
            'horizon': result.horizon,
            'composite_raw': float(result.composite_raw) if result.composite_raw else None,
            'indicators': result.components,
            'weighted_scores': {
                'w_sma50vs200': float(result.w_sma50vs200) if result.w_sma50vs200 else None,
                'w_pricevs50': float(result.w_pricevs50) if result.w_pricevs50 else None,
                'w_rsi14': float(result.w_rsi14) if result.w_rsi14 else None,
                'w_macd12269': float(result.w_macd12269) if result.w_macd12269 else None,
                'w_bbpos20': float(result.w_bbpos20) if result.w_bbpos20 else None,
                'w_bbwidth20': float(result.w_bbwidth20) if result.w_bbwidth20 else None,
                'w_volsurge': float(result.w_volsurge) if result.w_volsurge else None,
                'w_obv20': float(result.w_obv20) if result.w_obv20 else None,
                'w_rel1y': float(result.w_rel1y) if result.w_rel1y else None,
                'w_rel2y': float(result.w_rel2y) if result.w_rel2y else None,
                'w_candlerev': float(result.w_candlerev) if result.w_candlerev else None,
                'w_srcontext': float(result.w_srcontext) if result.w_srcontext else None,
            }
        }
        
        return Response(analysis_data)
        
    except Exception as e:
        return Response(
            {'error': f'Failed to retrieve analysis: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )