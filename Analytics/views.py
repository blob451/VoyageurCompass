"""
Analytics API endpoints for financial analysis and trading signals.
Provides technical analysis, explanation generation, and portfolio insights.
"""

from datetime import datetime

from django.core.cache import cache
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle

from Analytics.engine.enhanced_ta_engine import EnhancedTechnicalAnalysisEngine
from Analytics.utils.analysis_logger import AnalysisLogger
from Data.models import AnalyticsResults, Portfolio
from Data.services.yahoo_finance import yahoo_finance_service


def get_stock_sector_industry(stock):
    """
    Get sector and industry for a stock with robust handling and auto-population.
    
    Args:
        stock: Stock model instance
        
    Returns:
        tuple: (sector, industry) with proper fallbacks
    """
    # Get current values, handling None and whitespace-only strings
    sector = (stock.sector or "").strip()
    industry = (stock.industry or "").strip()
    
    # If either is empty, try to auto-populate from Yahoo Finance
    if not sector or not industry:
        try:
            # Import here to avoid circular imports
            from Data.services.yahoo_finance import yahoo_finance_service
            import logging
            
            logger = logging.getLogger(__name__)
            logger.info(f"Auto-populating missing sector/industry data for {stock.symbol}")
            
            # Fetch stock info from Yahoo Finance
            stock_info = yahoo_finance_service.get_stock_info(stock.symbol)
            
            if stock_info and not stock_info.get("error"):
                # Stock record update with valid data
                updated = False
                if not sector and stock_info.get("sector"):
                    stock.sector = stock_info["sector"].strip()
                    sector = stock.sector
                    updated = True
                    
                if not industry and stock_info.get("industry"):
                    stock.industry = stock_info["industry"].strip()
                    industry = stock.industry
                    updated = True
                    
                if updated:
                    stock.save(update_fields=["sector", "industry"])
                    logger.info(f"Updated {stock.symbol} - Sector: {sector}, Industry: {industry}")
                    
        except Exception as e:
            # Auto-population failure handling to maintain API stability
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to auto-populate sector/industry for {stock.symbol}: {e}")
    
    # Return with fallbacks
    return (
        sector if sector else "Unknown",
        industry if industry else "Unknown"
    )


class AnalysisThrottle(UserRateThrottle):
    """Custom throttle for analysis endpoints."""

    rate = "1000/hour"  # Increased for development to prevent issues with rapid testing


@extend_schema(
    summary="Analyze a stock",
    description="Run comprehensive analysis on a stock including technical indicators and trading signals",
    parameters=[
        OpenApiParameter(
            name="symbol",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description="Stock ticker symbol (e.g., AAPL, MSFT)",
        ),
        OpenApiParameter(
            name="months",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Number of months to analyze (default: 6)",
        ),
        OpenApiParameter(
            name="sync",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Sync latest data before analysis (default: false)",
        ),
        OpenApiParameter(
            name="include_explanation",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Generate explanation using local LLaMA model (default: false)",
        ),
        OpenApiParameter(
            name="explanation_detail",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Explanation detail level: summary, standard, detailed (default: standard)",
        ),
        OpenApiParameter(
            name="async",
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Enable async mode - trigger backfill in background if data missing (default: false)",
        ),
    ],
    responses={
        200: {
            "description": "Analysis completed successfully",
            "example": {"success": True, "symbol": "AAPL", "signal": "BUY", "analysis": {}},
        },
        400: {"description": "Invalid parameters"},
        404: {"description": "Stock not found"},
        429: {"description": "Rate limit exceeded"},
    },
)
@api_view(["GET"])
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
        language: Language for explanation (en/fr/es, default: en)

    Returns:
        Comprehensive analysis with trading signals and optional explanations
    """
    import logging

    logger = logging.getLogger(__name__)

    # Parameter extraction
    symbol = symbol.upper()
    months = int(request.query_params.get("months", 6))
    sync = request.query_params.get("sync", "false").lower() == "true"
    include_explanation = request.query_params.get("include_explanation", "false").lower() == "true"
    explanation_detail = request.query_params.get("explanation_detail", "standard")
    language = request.query_params.get("language", "en")

    # Parameters validated

    logger.info(
        f"Analysis request received for {symbol} from user {request.user.username if hasattr(request.user, 'username') else 'Unknown'}"
    )
    logger.debug(
        f"Parameters - months: {months}, sync: {sync}, explanation: {include_explanation}, detail: {explanation_detail}"
    )

    # Parameter validation
    if months < 1 or months > 24:
        return Response({"error": "Months parameter must be between 1 and 24"}, status=status.HTTP_400_BAD_REQUEST)

    # Explanation detail validation
    if explanation_detail not in ["summary", "standard", "detailed"]:
        return Response(
            {"error": "explanation_detail must be one of: summary, standard, detailed"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Credit check and deduction
    try:
        user_profile = request.user.profile
        if not user_profile.has_credits(1):
            return Response(
                {"error": "Insufficient credits. Please purchase more credits to continue."},
                status=status.HTTP_402_PAYMENT_REQUIRED
            )
        logger.info(f"Credit check passed for user {request.user.username}. Current credits: {user_profile.credits}")
    except AttributeError:
        return Response(
            {"error": "User profile not found. Please contact support."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    # Sync data if explicitly requested (auto-sync is now handled by the engine)
    if sync:
        period = f"{months}mo" if months <= 12 else "2y"
        sync_result = yahoo_finance_service.get_stock_data(symbol, period=period, sync_db=True)

        if "error" in sync_result:
            return Response(
                {"error": f'Failed to sync data: {sync_result["error"]}'}, status=status.HTTP_400_BAD_REQUEST
            )

    # Data coverage verification and backfill trigger
    async_mode = request.query_params.get("async", "false").lower() == "true"
    if async_mode:
        from Data.repo.price_reader import PriceReader
        from Data.services.tasks import async_stock_backfill

        # Data adequacy assessment
        required_years = 2  # Standard requirement
        price_reader = PriceReader()
        coverage = price_reader.check_data_coverage(symbol, required_years)
        
        # Determine if backfill is needed
        needs_backfill = (
            not coverage["stock"]["has_data"] or
            coverage["stock"]["gap_count"] > 50
        )
        
        if needs_backfill:
            logger.info(f"Data insufficient for {symbol}, triggering async backfill")
            
            # Trigger async backfill + analysis
            backfill_task = async_stock_backfill.delay(
                symbol=symbol,
                required_years=required_years, 
                user_id=request.user.id
            )
            
            return Response({
                "status": "async_processing",
                "symbol": symbol,
                "message": f"Data backfill initiated for {symbol}. Analysis will start automatically when complete.",
                "task_id": backfill_task.id,
                "check_status_url": f"/api/v1/analytics/status/{symbol}/",
                "estimated_completion": "2-3 minutes"
            }, status=status.HTTP_202_ACCEPTED)

    # Analysis execution
    try:
        logger.info(f"Starting analysis process for {symbol}")

        # Analysis logger initialisation
        analysis_logger = None
        if request.user and hasattr(request.user, "username"):
            try:
                analysis_logger = AnalysisLogger(request.user.username, symbol)
                logger.debug(f"AnalysisLogger created successfully: {analysis_logger.log_filename}")
            except Exception as logger_error:
                logger.warning(f"Failed to create analysis logger for {symbol}: {str(logger_error)}")

        engine = EnhancedTechnicalAnalysisEngine()
        analysis = engine.analyze_stock(symbol, user=request.user, logger_instance=analysis_logger)
        logger.info(f"Engine analysis completed for {symbol}")

        # New engine always returns successful analysis or raises exception

        # Format response for new TA engine
        analysis_date = analysis.get("analysis_date", datetime.now())
        if hasattr(analysis_date, "isoformat"):
            analysis_date_str = analysis_date.isoformat()
        else:
            analysis_date_str = str(analysis_date)

        weighted_scores_dict = analysis.get("weighted_scores", {})

        # Safe conversion of weighted scores
        safe_weighted_scores = {}
        for k, v in weighted_scores_dict.items():
            try:
                if v is not None:
                    safe_weighted_scores[k] = float(v)
                else:
                    safe_weighted_scores[k] = 0.0
                    logger.warning(f"weighted_score {k} was None, defaulting to 0.0 for {symbol}")
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting weighted_score {k}={v} for {symbol}: {e}")
                safe_weighted_scores[k] = 0.0

        # CRITICAL FIX: Verify database transaction was successful before responding
        analytics_result_id = analysis.get("analytics_result_id")
        if analytics_result_id:
            try:
                # Verify the analysis was actually saved to the database
                from Data.models import AnalyticsResults

                saved_analysis = AnalyticsResults.objects.filter(
                    id=analytics_result_id, user=request.user, stock__symbol=symbol
                ).first()

                if not saved_analysis:
                    logger.error(
                        f"Analysis result ID {analytics_result_id} not found in database for {symbol} user {request.user.id}"
                    )
                    return Response(
                        {"error": "Analysis completed but failed to save to database. Please try again."},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )
                else:
                    logger.debug(f"Verified analysis {analytics_result_id} saved successfully for {symbol}")

                    # Double-check the analysis data matches what we're returning
                    if saved_analysis.stock.symbol != symbol:
                        logger.error(
                            f"Analysis result ID mismatch: expected {symbol}, found {saved_analysis.stock.symbol}"
                        )
                        return Response(
                            {"error": "Analysis result ID mismatch detected. Please try again."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        )

                    # FORCE: Always generate summary explanation for better UX (unconditional)
                    try:
                        from Analytics.services.explanation_service import (
                            get_explanation_service,
                        )
                        from Analytics.services.language_detector import detect_request_language

                        explanation_service = get_explanation_service()

                        if explanation_service.enabled:  # Only check if feature is enabled
                            # Explanation existence verification
                            existing = saved_analysis.explanations_json or {}
                            levels = existing.get("levels", {})

                            if "summary" not in levels or not levels["summary"].get("content"):
                                # Smart language detection for auto-generation
                                # Priority: explicit query param > user preferences > request headers
                                detected_language = detect_request_language(
                                    request,
                                    request.user,
                                    explicit_language=language if language != "en" else None
                                )

                                # Use detected language for auto-generation
                                auto_language = detected_language if detected_language in ["fr", "es"] else language
                                logger.info(f"[AUTO-GENERATION] Using language '{auto_language}' for summary auto-generation (query: {language}, detected: {detected_language})")

                                result = explanation_service.explain_prediction_single(
                                    saved_analysis, detail_level="summary", user=request.user, language=auto_language
                                )

                                if result:

                                    # Save to database with forced save
                                    saved_analysis.refresh_from_db()
                                    existing_explanations = saved_analysis.explanations_json or {}
                                    levels = existing_explanations.get("levels", {})
                                    levels["summary"] = {
                                        "content": result.get("content", ""),
                                        "confidence": result.get("confidence_score", 0.0),
                                        "generated_at": datetime.now().isoformat(),
                                        "word_count": result.get("word_count", 0),
                                    }
                                    saved_analysis.explanations_json = {
                                        "levels": levels,
                                        "indicators_explained": result.get("indicators_explained", []),
                                        "risk_factors": result.get("risk_factors", []),
                                        "recommendation": result.get("recommendation", "HOLD"),
                                        "current_level": "summary",
                                    }
                                    # Also update legacy fields
                                    saved_analysis.narrative_text = result.get("content", "")
                                    saved_analysis.explanation_confidence = result.get("confidence_score", 0.0)
                                    saved_analysis.explanation_method = result.get("method", "unknown")
                                    saved_analysis.explained_at = datetime.now()
                                    saved_analysis.save()

                                    # Verify the save worked
                                    saved_analysis.refresh_from_db()
                                    if (
                                        saved_analysis.explanations_json
                                        and "levels" in saved_analysis.explanations_json
                                    ):
                                        if "summary" in saved_analysis.explanations_json["levels"]:
                                            # Summary verification successful
                                            pass
                                        else:
                                            # Summary verification failed
                                            pass
                                    else:
                                        # Explanations JSON verification failed
                                        pass

                                    # Generation attempt completed
                                    # Creating fallback content

                                    # Fallback content generation
                                    score = saved_analysis.score_0_10 if hasattr(saved_analysis, "score_0_10") else 5.0
                                    recommendation = "BUY" if score >= 7 else "HOLD" if score >= 4 else "SELL"
                                    fallback_content = f"{symbol} receives a technical analysis score of {score:.1f}/10, suggesting a {recommendation} position based on current technical indicators."

                                    result = {
                                        "content": fallback_content,
                                        "method": "fallback",
                                        "confidence_score": 0.5,
                                        "word_count": len(fallback_content.split()),
                                        "indicators_explained": ["Technical Analysis"],
                                        "risk_factors": ["Market conditions"],
                                        "recommendation": recommendation,
                                    }

                                    # Save fallback content
                                    saved_analysis.refresh_from_db()
                                    existing_explanations = saved_analysis.explanations_json or {}
                                    levels = existing_explanations.get("levels", {})
                                    levels["summary"] = {
                                        "content": result.get("content", ""),
                                        "confidence": result.get("confidence_score", 0.0),
                                        "generated_at": datetime.now().isoformat(),
                                        "word_count": result.get("word_count", 0),
                                    }
                                    saved_analysis.explanations_json = {
                                        "levels": levels,
                                        "indicators_explained": result.get("indicators_explained", []),
                                        "risk_factors": result.get("risk_factors", []),
                                        "recommendation": result.get("recommendation", "HOLD"),
                                        "current_level": "summary",
                                    }
                                    saved_analysis.narrative_text = result.get("content", "")
                                    saved_analysis.explanation_confidence = result.get("confidence_score", 0.0)
                                    saved_analysis.explanation_method = result.get("method", "fallback")
                                    saved_analysis.explained_at = datetime.now()
                                    saved_analysis.save()
                                    # Fallback explanation saved
                            # Summary already exists
                        else:

                            # Even if service disabled, create minimal explanation
                            score = saved_analysis.score_0_10 if hasattr(saved_analysis, "score_0_10") else 5.0
                            recommendation = "BUY" if score >= 7 else "HOLD" if score >= 4 else "SELL"
                            minimal_content = f"{symbol}: {score:.1f}/10 - {recommendation}"

                            saved_analysis.refresh_from_db()
                            existing_explanations = saved_analysis.explanations_json or {}
                            levels = existing_explanations.get("levels", {})
                            levels["summary"] = {
                                "content": minimal_content,
                                "confidence": 0.3,
                                "generated_at": datetime.now().isoformat(),
                                "word_count": len(minimal_content.split()),
                            }
                            saved_analysis.explanations_json = {
                                "levels": levels,
                                "indicators_explained": [],
                                "risk_factors": [],
                                "recommendation": recommendation,
                                "current_level": "summary",
                            }
                            saved_analysis.narrative_text = minimal_content
                            saved_analysis.explanation_confidence = 0.3
                            saved_analysis.explanation_method = "minimal"
                            saved_analysis.explained_at = datetime.now()
                            saved_analysis.save()

                    except Exception as force_error:
                        logger.error(f"Force generation failed for {symbol}: {str(force_error)}")
                        # Continue execution despite error

                    # Original pre-generation code removed - replaced by forced generation above

            except Exception as db_error:
                logger.error(f"Database verification failed for {symbol}: {str(db_error)}", exc_info=True)
                return Response(
                    {"error": f"Failed to verify analysis save: {str(db_error)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        else:
            logger.warning(f"No analytics_result_id returned for {symbol} analysis")

        # Generate explanation if requested
        explanation_data = None
        if include_explanation and analytics_result_id:
            try:
                explanation_service = get_explanation_service()

                if explanation_service.enabled:  # Use enabled instead of is_enabled() for consistency
                    # Saved analysis retrieval for explanation
                    saved_analysis = AnalyticsResults.objects.filter(
                        id=analytics_result_id, user=request.user, stock__symbol=symbol
                    ).first()

                    if saved_analysis:
                        explanation_result = explanation_service.explain_prediction_single(
                            saved_analysis, detail_level=explanation_detail, user=request.user
                        )

                        if explanation_result:
                            # Update the database with explanation in levels structure
                            existing_explanations = saved_analysis.explanations_json or {}
                            levels = existing_explanations.get("levels", {})
                            levels[explanation_detail] = {
                                "content": explanation_result.get("content", ""),
                                "confidence": explanation_result.get("confidence_score", 0.0),
                                "generated_at": datetime.now().isoformat(),
                                "word_count": explanation_result.get("word_count", 0),
                            }
                            saved_analysis.explanations_json = {
                                "levels": levels,
                                "indicators_explained": explanation_result.get("indicators_explained", []),
                                "risk_factors": explanation_result.get("risk_factors", []),
                                "recommendation": explanation_result.get("recommendation", "HOLD"),
                                "current_level": explanation_detail,
                            }
                            # Also update legacy fields for backward compatibility
                            saved_analysis.explanation_method = explanation_result.get("method", "unknown")
                            saved_analysis.explanation_version = "1.0"
                            saved_analysis.narrative_text = explanation_result.get("content", "")
                            saved_analysis.explanation_confidence = explanation_result.get("confidence_score", 0.0)
                            saved_analysis.explained_at = datetime.now()
                            saved_analysis.save()
                            # Explanation saved successfully

                            explanation_data = {
                                "content": explanation_result.get("content", ""),
                                "confidence_score": explanation_result.get("confidence_score", 0.0),
                                "detail_level": explanation_detail,
                                "method": explanation_result.get("method", "unknown"),
                                "generation_time": explanation_result.get("generation_time", 0.0),
                                "word_count": explanation_result.get("word_count", 0),
                                "indicators_explained": explanation_result.get("indicators_explained", []),
                                "risk_factors": explanation_result.get("risk_factors", []),
                                "recommendation": explanation_result.get("recommendation", "HOLD"),
                            }

                            # Explanation generation completed
                        # Explanation generation attempted
                    # Analysis lookup completed
                    # Service availability checked
                    explanation_data = {
                        "error": "Explanation service not available",
                        "detail_level": explanation_detail,
                        "method": "unavailable",
                    }

            except Exception as e:
                logger.error(f"Error generating explanation for {symbol}: {str(e)}")
                logger.error(f"Explanation generation failed for {symbol}: {str(e)}")
                explanation_data = {
                    "error": f"Explanation generation failed: {str(e)}",
                    "detail_level": explanation_detail,
                    "method": "error",
                }

        # Get sector and industry for response
        if analytics_result_id:
            saved_analysis = AnalyticsResults.objects.filter(
                id=analytics_result_id, user=request.user, stock__symbol=symbol
            ).first()
            if saved_analysis:
                sector, industry = get_stock_sector_industry(saved_analysis.stock)
            else:
                sector, industry = "Unknown", "Unknown"
        else:
            sector, industry = "Unknown", "Unknown"

        response_data = {
            "success": True,
            "symbol": analysis.get("symbol", symbol),
            "name": analysis.get("name", f"{symbol} Corporation"),
            "sector": sector,
            "industry": industry,
            "analysis_date": analysis_date_str,
            "horizon": analysis.get("horizon", "unknown"),
            "composite_score": analysis.get("score_0_10", 0.0),
            "composite_raw": analysis.get("composite_raw"),
            "indicators": analysis.get("components", {}),
            "weighted_scores": safe_weighted_scores,
            "analytics_result_id": analytics_result_id,
        }

        # Conditional explanation inclusion
        if explanation_data:
            response_data["explanation"] = explanation_data

        # Response formatted and verified

        # Analysis history cache invalidation
        user_cache_pattern = f"analysis_history:{request.user.id}:*"
        # Django cache wildcard deletion limitation requires simple invalidation approach
        # In production, consider using Redis directly for pattern-based cache invalidation
        cache.delete_many(
            [
                f"analysis_history:{request.user.id}:20:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
                f"analysis_history:{request.user.id}:10:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
                f"analysis_history:{request.user.id}:25:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
                f"analysis_history:{request.user.id}:50:0:all:{','.join(sorted(['id', 'symbol', 'name', 'score', 'analysis_date', 'sector', 'industry', 'horizon', 'composite_raw']))}",
            ]
        )

        # Deduct credit after successful analysis
        user_profile.subtract_credits(1)
        logger.info(f"Deducted 1 credit from user {request.user.username}. Remaining credits: {user_profile.credits}")

        return Response(response_data)

    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {type(e).__name__}: {str(e)}")
        import traceback

        logger.debug(f"Full traceback for {symbol}: {traceback.format_exc()}")

        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Analysis failed for {symbol}: {str(e)}", exc_info=True)
        return Response({"error": f"Analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Analyze portfolio",
    description="Run analysis on all stocks in a portfolio",
    parameters=[
        OpenApiParameter(
            name="portfolio_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description="Portfolio ID",
        ),
    ],
    responses={
        200: {"description": "Portfolio analysis completed"},
        403: {"description": "Not authorized to access this portfolio"},
        404: {"description": "Portfolio not found"},
    },
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def analyze_portfolio(request, portfolio_id):
    """
    Analyze all stocks in a portfolio.

    Path Parameters:
        portfolio_id: Portfolio ID

    Returns:
        Analysis results for all holdings
    """
    # Portfolio ownership verification
    try:
        portfolio = Portfolio.objects.get(id=portfolio_id)

        if portfolio.user != request.user:
            return Response(
                {"error": "You are not authorized to analyze this portfolio"}, status=status.HTTP_403_FORBIDDEN
            )
    except Portfolio.DoesNotExist:
        return Response({"error": "Portfolio not found"}, status=status.HTTP_404_NOT_FOUND)

    # Portfolio analysis execution
    try:
        # For now, disable portfolio analysis - could be implemented later
        # by running individual stock analyses for each holding
        return Response(
            {"error": "Portfolio analysis not yet implemented with new TA engine"},
            status=status.HTTP_501_NOT_IMPLEMENTED,
        )

    except Exception as e:
        return Response({"error": f"Portfolio analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Batch stock analysis",
    description="Analyze multiple stocks at once",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock symbols",
                    "example": ["AAPL", "MSFT", "GOOGL"],
                },
                "months": {"type": "integer", "description": "Analysis period in months", "default": 6},
            },
            "required": ["symbols"],
        }
    },
    responses={200: {"description": "Batch analysis completed"}, 400: {"description": "Invalid request data"}},
)
@api_view(["POST"])
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
    symbols = request.data.get("symbols", [])
    months = int(request.data.get("months", 6))

    # Input validation
    if not symbols:
        return Response({"error": "Symbols list is required"}, status=status.HTTP_400_BAD_REQUEST)

    if len(symbols) > 10:
        return Response({"error": "Maximum 10 symbols allowed per batch"}, status=status.HTTP_400_BAD_REQUEST)

    if months < 1 or months > 24:
        return Response({"error": "Months parameter must be between 1 and 24"}, status=status.HTTP_400_BAD_REQUEST)

    # Multi-symbol analysis execution
    engine = EnhancedTechnicalAnalysisEngine()
    
    def analyze_single_symbol(symbol):
        """Analyze a single symbol."""
        symbol = symbol.upper()
        try:
            # Batch analysis logger initialisation
            analysis_logger = None
            if request.user and hasattr(request.user, "username"):
                try:
                    analysis_logger = AnalysisLogger(request.user.username, symbol)
                except Exception as logger_error:
                    # Log the error but continue without logger
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to create analysis logger for {symbol}: {str(logger_error)}")

            analysis = engine.analyze_stock(symbol, user=request.user, logger_instance=analysis_logger)

            return symbol, {
                "success": True,
                "composite_score": analysis["score_0_10"],
                "composite_raw": analysis["composite_raw"],
                "analysis_date": analysis["analysis_date"].isoformat(),
                "horizon": analysis["horizon"],
            }
        except Exception as e:
            return symbol, {"success": False, "error": str(e)}

    # Sequential analysis execution
    results = {}
    for symbol in symbols:
        try:
            symbol, analysis_result = analyze_single_symbol(symbol)
            results[symbol] = analysis_result
        except Exception as e:
            results[symbol] = {"success": False, "error": str(e)}

    return Response(
        {
            "success": True,
            "results": results,
            "total_analysed": len(results),
            "successful": sum(1 for r in results.values() if r.get("success")),
        }
    )


@extend_schema(
    summary="Get market overview",
    description="Get analysis of major market indices",
    responses={200: {"description": "Market overview data"}},
)
@api_view(["GET"])
@permission_classes([AllowAny])
def market_overview(request):
    """
    Get market overview with major indices analysis.

    Returns:
        Analysis of S&P 500, Dow Jones, and NASDAQ
    """
    indices = {
        "SPY": "S&P 500 ETF",
        "DIA": "Dow Jones ETF",
        "QQQ": "NASDAQ ETF",
    }

    engine = EnhancedTechnicalAnalysisEngine()

    def analyze_index(symbol, name):
        """Analyze a single index."""
        try:
            analysis = engine.analyze_stock(symbol, horizon="short")

            return symbol, {
                "name": name,
                "composite_score": analysis["score_0_10"],
                "analysis_date": analysis["analysis_date"].isoformat(),
                "horizon": analysis["horizon"],
            }
        except Exception:
            return symbol, {"name": name, "error": "Data unavailable"}

    # Analyze indices and get market status
    results = {}
    for symbol, name in indices.items():
        try:
            symbol, analysis_result = analyze_index(symbol, name)
            results[symbol] = analysis_result
        except Exception as e:
            results[symbol] = {"name": name, "error": str(e)}
    
    # Market status retrieval
    try:
        market_status = yahoo_finance_service.get_market_status()
    except Exception as e:
        market_status = {"error": str(e)}

    return Response({"market_status": market_status, "indices": results, "timestamp": datetime.now().isoformat()})


@extend_schema(
    summary="Get user analysis history",
    description="Get the authenticated user's stock analysis history",
    parameters=[
        OpenApiParameter(
            name="limit",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Maximum number of results to return (default: 20)",
        ),
        OpenApiParameter(
            name="symbol",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=False,
            description="Filter by stock symbol",
        ),
    ],
    responses={
        200: {
            "description": "User analysis history retrieved successfully",
            "example": {
                "analyses": [
                    {
                        "id": 123,
                        "symbol": "AAPL",
                        "score": 7,
                        "analysis_date": "2025-01-14T10:30:00Z",
                        "sector": "Technology",
                        "industry": "Consumer Electronics",
                    }
                ],
                "count": 1,
            },
        }
    },
)
@api_view(["GET"])
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
    limit = min(int(request.query_params.get("limit", 20)), 100)  # Cap at 100
    offset = int(request.query_params.get("offset", 0))
    symbol_filter = request.query_params.get("symbol")
    fields_param = request.query_params.get("fields")

    # Parse requested fields
    if fields_param:
        requested_fields = set(fields_param.split(","))
        # Field validation
        valid_fields = {
            "id",
            "symbol",
            "name",
            "score",
            "analysis_date",
            "sector",
            "industry",
            "horizon",
            "composite_raw",
            "components",
        }
        requested_fields = requested_fields.intersection(valid_fields)
    else:
        # Default fields (excluding heavy components field)
        requested_fields = {
            "id",
            "symbol",
            "name",
            "score",
            "analysis_date",
            "sector",
            "industry",
            "horizon",
            "composite_raw",
        }

    # Generate cache key based on user, pagination, filters, and fields
    cache_key = f"analysis_history:{request.user.id}:{limit}:{offset}:{symbol_filter or 'all'}:{','.join(sorted(requested_fields))}"

    # Try to get from cache first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return Response(cached_result)

    # Optimised queryset construction
    queryset = AnalyticsResults.objects.filter(user=request.user).select_related("stock").order_by("-as_of")

    # Apply symbol filter
    if symbol_filter:
        queryset = queryset.filter(stock__symbol__iexact=symbol_filter)

    # Pre-pagination count determination
    total_count = queryset.count()

    # Apply pagination
    paginated_queryset = queryset[offset : offset + limit]

    # Serialize the results with field selection
    paginated_results = list(paginated_queryset)
    analyses = []
    
    for result in paginated_results:
        analysis_data = {}

        if "id" in requested_fields:
            analysis_data["id"] = result.id
        if "symbol" in requested_fields:
            analysis_data["symbol"] = result.stock.symbol
        if "name" in requested_fields:
            analysis_data["name"] = result.stock.short_name or result.stock.long_name
        if "score" in requested_fields:
            analysis_data["score"] = result.score_0_10
        if "analysis_date" in requested_fields:
            analysis_data["analysis_date"] = result.as_of.isoformat()
        if "sector" in requested_fields:
            sector, industry = get_stock_sector_industry(result.stock)
            analysis_data["sector"] = sector
        if "industry" in requested_fields:
            if "sector" not in requested_fields:  # Avoid calling twice
                sector, industry = get_stock_sector_industry(result.stock)
            analysis_data["industry"] = industry
        if "horizon" in requested_fields:
            analysis_data["horizon"] = result.horizon
        if "composite_raw" in requested_fields:
            analysis_data["composite_raw"] = float(result.composite_raw) if result.composite_raw else None
        if "components" in requested_fields:
            analysis_data["components"] = result.components

        analyses.append(analysis_data)

    # Calculate pagination metadata
    has_next = (offset + limit) < total_count
    has_previous = offset > 0

    # Prepare response data
    response_data = {
        "analyses": analyses,
        "pagination": {
            "count": len(analyses),
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_next": has_next,
            "has_previous": has_previous,
            "next_offset": offset + limit if has_next else None,
            "previous_offset": max(0, offset - limit) if has_previous else None,
        },
    }

    # Cache the result for 5 minutes (300 seconds)
    cache.set(cache_key, response_data, 300)

    return Response(response_data)


@extend_schema(
    summary="Get analysis by ID",
    description="Get a specific analysis result by its ID",
    parameters=[
        OpenApiParameter(
            name="analysis_id",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.PATH,
            required=True,
            description="Analysis result ID",
        ),
    ],
    responses={
        200: {"description": "Analysis found"},
        404: {"description": "Analysis not found"},
        403: {"description": "Not authorized to view this analysis"},
    },
)
@api_view(["GET"])
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
        # User-owned analysis retrieval
        result = AnalyticsResults.objects.filter(id=analysis_id, user=request.user).select_related("stock").first()

        if not result:
            return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)

        # Format the response - same structure as get_user_latest_analysis
        # Get sector and industry with auto-population
        sector, industry = get_stock_sector_industry(result.stock)
        
        analysis_data = {
            "id": result.id,
            "symbol": result.stock.symbol,
            "name": result.stock.short_name or result.stock.long_name or f"{result.stock.symbol} Corporation",
            "sector": sector,
            "industry": industry,
            "score": result.score_0_10,
            "analysis_date": result.as_of.isoformat(),
            "horizon": result.horizon,
            "composite_raw": float(result.composite_raw) if result.composite_raw else None,
            "indicators": result.components,
            "weighted_scores": {
                "w_sma50vs200": float(result.w_sma50vs200) if result.w_sma50vs200 else None,
                "w_pricevs50": float(result.w_pricevs50) if result.w_pricevs50 else None,
                "w_rsi14": float(result.w_rsi14) if result.w_rsi14 else None,
                "w_macd12269": float(result.w_macd12269) if result.w_macd12269 else None,
                "w_bbpos20": float(result.w_bbpos20) if result.w_bbpos20 else None,
                "w_bbwidth20": float(result.w_bbwidth20) if result.w_bbwidth20 else None,
                "w_volsurge": float(result.w_volsurge) if result.w_volsurge else None,
                "w_obv20": float(result.w_obv20) if result.w_obv20 else None,
                "w_rel1y": float(result.w_rel1y) if result.w_rel1y else None,
                "w_rel2y": float(result.w_rel2y) if result.w_rel2y else None,
                "w_candlerev": float(result.w_candlerev) if result.w_candlerev else None,
                "w_srcontext": float(result.w_srcontext) if result.w_srcontext else None,
            },
        }

        return Response(analysis_data)

    except Exception as e:
        return Response(
            {"error": f"Failed to retrieve analysis: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get FinBERT model status",
    description="Check if the FinBERT sentiment analysis model is loaded and ready",
    responses={
        200: {
            "description": "Model status retrieved successfully",
            "example": {
                "model_ready": True,
                "status": "loaded",
                "loading": False,
                "error": None,
                "message": "FinBERT model is ready for sentiment analysis"
            }
        }
    },
)
@api_view(["GET"])
@permission_classes([AllowAny])
def finbert_model_status(request):
    """
    Get the current status of the FinBERT sentiment analysis model.
    
    Returns:
        Model loading status, readiness, and any error information
    """
    try:
        from Analytics.apps import MODEL_STATUS
        
        if MODEL_STATUS['loaded']:
            return Response({
                "model_ready": True,
                "status": "loaded",
                "loading": False,
                "error": None,
                "message": "FinBERT model is ready for sentiment analysis"
            })
        elif MODEL_STATUS['loading']:
            return Response({
                "model_ready": False,
                "status": "loading",
                "loading": True,
                "error": None,
                "message": "FinBERT model is currently loading in background"
            })
        elif MODEL_STATUS['error']:
            return Response({
                "model_ready": False,
                "status": "error",
                "loading": False,
                "error": MODEL_STATUS['error'],
                "message": f"FinBERT model failed to load: {MODEL_STATUS['error']}"
            })
        else:
            return Response({
                "model_ready": False,
                "status": "not_started",
                "loading": False,
                "error": None,
                "message": "FinBERT model loading has not started yet"
            })
            
    except Exception as e:
        return Response({
            "model_ready": False,
            "status": "error",
            "loading": False,
            "error": str(e),
            "message": f"Failed to check model status: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Get user's latest analysis for a stock",
    description="Get the user's most recent analysis for a specific stock",
    parameters=[
        OpenApiParameter(
            name="symbol",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description="Stock ticker symbol",
        ),
    ],
    responses={200: {"description": "Latest analysis found"}, 404: {"description": "No analysis found for this stock"}},
)
@api_view(["GET"])
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
        # Latest user analysis retrieval
        result = AnalyticsResults.objects.filter(user=request.user, stock__symbol=symbol)\
            .select_related("stock")\
            .order_by("-as_of")\
            .first()

        if not result:
            return Response({"error": f"No analysis found for {symbol}"}, status=status.HTTP_404_NOT_FOUND)

        # Get sector and industry with auto-population
        sector, industry = get_stock_sector_industry(result.stock)
        
        # Format the response
        analysis_data = {
            "id": result.id,
            "symbol": result.stock.symbol,
            "name": result.stock.short_name or result.stock.long_name or f"{symbol} Corporation",
            "sector": sector,
            "industry": industry,
            "score": result.score_0_10,
            "analysis_date": result.as_of.isoformat(),
            "horizon": result.horizon,
            "composite_raw": float(result.composite_raw) if result.composite_raw else None,
            "indicators": result.components,
            "weighted_scores": {
                "w_sma50vs200": float(result.w_sma50vs200) if result.w_sma50vs200 else None,
                "w_pricevs50": float(result.w_pricevs50) if result.w_pricevs50 else None,
                "w_rsi14": float(result.w_rsi14) if result.w_rsi14 else None,
                "w_macd12269": float(result.w_macd12269) if result.w_macd12269 else None,
                "w_bbpos20": float(result.w_bbpos20) if result.w_bbpos20 else None,
                "w_bbwidth20": float(result.w_bbwidth20) if result.w_bbwidth20 else None,
                "w_volsurge": float(result.w_volsurge) if result.w_volsurge else None,
                "w_obv20": float(result.w_obv20) if result.w_obv20 else None,
                "w_rel1y": float(result.w_rel1y) if result.w_rel1y else None,
                "w_rel2y": float(result.w_rel2y) if result.w_rel2y else None,
                "w_candlerev": float(result.w_candlerev) if result.w_candlerev else None,
                "w_srcontext": float(result.w_srcontext) if result.w_srcontext else None,
            },
        }

        return Response(analysis_data)

    except Exception as e:
        return Response(
            {"error": f"Failed to retrieve analysis: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get FinBERT model status",
    description="Check if the FinBERT sentiment analysis model is loaded and ready",
    responses={
        200: {
            "description": "Model status retrieved successfully",
            "example": {
                "model_ready": True,
                "status": "loaded",
                "loading": False,
                "error": None,
                "message": "FinBERT model is ready for sentiment analysis"
            }
        }
    },
)
@api_view(["GET"])
@permission_classes([AllowAny])
def finbert_model_status(request):
    """
    Get the current status of the FinBERT sentiment analysis model.
    
    Returns:
        Model loading status, readiness, and any error information
    """
    try:
        from Analytics.apps import MODEL_STATUS
        
        if MODEL_STATUS['loaded']:
            return Response({
                "model_ready": True,
                "status": "loaded",
                "loading": False,
                "error": None,
                "message": "FinBERT model is ready for sentiment analysis"
            })
        elif MODEL_STATUS['loading']:
            return Response({
                "model_ready": False,
                "status": "loading",
                "loading": True,
                "error": None,
                "message": "FinBERT model is currently loading in background"
            })
        elif MODEL_STATUS['error']:
            return Response({
                "model_ready": False,
                "status": "error",
                "loading": False,
                "error": MODEL_STATUS['error'],
                "message": f"FinBERT model failed to load: {MODEL_STATUS['error']}"
            })
        else:
            return Response({
                "model_ready": False,
                "status": "not_started",
                "loading": False,
                "error": None,
                "message": "FinBERT model loading has not started yet"
            })
            
    except Exception as e:
        return Response({
            "model_ready": False,
            "status": "error",
            "loading": False,
            "error": str(e),
            "message": f"Failed to check model status: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
