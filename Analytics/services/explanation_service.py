"""
Financial analysis explanation generation service using LLaMA 3.1 model.
"""

import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from django.conf import settings
from django.core.cache import cache

from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.security_validator import get_security_validator, sanitize_financial_input, validate_financial_output
from Analytics.services.translation_service import get_translation_service
from Data.models import AnalyticsResults

logger = logging.getLogger(__name__)


class ExplanationService:
    """Financial analysis explanation generation service with LLM integration."""

    def __init__(self):
        self._llm_service = None
        self._security_validator = None
        self._translation_service = None
        self.enabled = getattr(settings, "EXPLAINABILITY_ENABLED", True)
        self.cache_ttl = getattr(settings, "EXPLANATION_CACHE_TTL", 300)
        self.security_enabled = getattr(settings, "EXPLANATION_SECURITY_ENABLED", True)

        # Template explanations for LLM fallback scenarios
        self.indicator_templates = {
            "sma50vs200": "Simple Moving Average crossover analysis comparing 50-day vs 200-day periods",
            "pricevs50": "Current price position relative to 50-day Simple Moving Average",
            "rsi14": "Relative Strength Index (14-period) momentum oscillator analysis",
            "macd12269": "MACD indicator (12,26,9) trend following momentum analysis",
            "bbpos20": "Bollinger Bands position showing price relative to statistical bands",
            "bbwidth20": "Bollinger Bands width indicating market volatility levels",
            "volsurge": "Volume surge detection comparing recent to average trading volume",
            "obv20": "On-Balance Volume (20-period) measuring buying/selling pressure",
            "rel1y": "Relative strength performance compared to market over 1-year period",
            "rel2y": "Relative strength performance compared to market over 2-year period",
            "candlerev": "Candlestick reversal pattern recognition for trend changes",
            "srcontext": "Support and resistance level analysis based on price history",
        }

    @property
    def llm_service(self):
        """Lazy-loaded LLM service."""
        if self._llm_service is None:
            logger.info("Initializing LLM service for explanation generation")
            self._llm_service = get_local_llm_service()
        return self._llm_service

    @property
    def security_validator(self):
        """Lazy-loaded security validator."""
        if self._security_validator is None and self.security_enabled:
            self._security_validator = get_security_validator()
        return self._security_validator

    @property
    def translation_service(self):
        """Lazy-loaded translation service."""
        if self._translation_service is None:
            logger.info("Initializing translation service for multilingual explanations")
            self._translation_service = get_translation_service()
        return self._translation_service

    def is_enabled(self) -> bool:
        """Verify explanation service availability status."""
        return self.enabled and self.llm_service.is_available()

    def is_available_or_fallback(self) -> bool:
        """Check if ANY explanation generation is possible (LLM or template)."""
        return self.enabled  # Always true if feature enabled, regardless of LLM availability

    def explain_prediction_single(
        self, analysis_result: Union[Dict[str, Any], AnalyticsResults], detail_level: str = "standard", language: str = "en", user=None, force_regenerate: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Generate detailed explanation for individual stock analysis result."""
        if not self.enabled:
            logger.info("Explanation service disabled")
            return None

        try:
            analysis_data = self._prepare_analysis_data(analysis_result)
            cache_key = self._create_cache_key(analysis_data, detail_level, language, user)
            
            # Skip cache check if force regeneration is requested
            if not force_regenerate:
                cached_explanation = cache.get(cache_key)
                if cached_explanation:
                    logger.info(f"Retrieved cached explanation for {analysis_data.get('symbol', 'unknown')}")
                    return cached_explanation
            else:
                logger.info(f"Force regeneration requested, skipping cache for {analysis_data.get('symbol', 'unknown')}")
                # Clear existing cache entry for this key
                cache.delete(cache_key)

            start_time = time.time()

            # Always try to generate something - LLM preferred, template as fallback
            if self.llm_service.is_available():
                logger.info(f"Using LLM for {analysis_data.get('symbol', 'unknown')} ({detail_level}, {language})")
                explanation = self._generate_multilingual_explanation(analysis_data, detail_level, language)
            else:
                logger.warning(
                    f"LLM unavailable, using template for {analysis_data.get('symbol', 'unknown')} ({detail_level})"
                )
                explanation = self._generate_template_explanation(analysis_data, detail_level)

            # Security validation for generated content
            if explanation and self.security_enabled and self.security_validator:
                content_validation = self.security_validator.validate_output(
                    explanation.get('content', ''), 'explanation'
                )

                if not content_validation['is_valid']:
                    logger.warning(
                        f"Security validation failed for {analysis_data.get('symbol', 'unknown')}: "
                        f"{content_validation['issues']}"
                    )
                    explanation['content'] = content_validation['filtered_content']
                    explanation['security_filtered'] = True
                    explanation['security_issues'] = content_validation['issues']
                else:
                    explanation['security_validated'] = True

            if explanation:
                # Add format validation for detailed explanations
                if detail_level == "detailed" and explanation.get("content"):
                    validation_result = self._validate_detailed_explanation_format(explanation["content"])
                    explanation["format_validation"] = validation_result
                    
                    if not validation_result["is_valid"]:
                        logger.warning(f"Format validation failed for {analysis_data.get('symbol', 'unknown')}: {validation_result['issues']}")
                        
                        # Apply corrective formatting if possible
                        if validation_result.get("suggested_fix"):
                            explanation["content"] = validation_result["suggested_fix"]
                            explanation["format_corrected"] = True
                
                explanation["generation_time"] = time.time() - start_time
                explanation["method"] = "llama" if self.llm_service.is_available() else "template"
                explanation["detail_level"] = detail_level
                explanation["generated_at"] = datetime.now().isoformat()

                cache.set(cache_key, explanation, self.cache_ttl)

                logger.info(
                    f"Generated explanation for {analysis_data.get('symbol', 'unknown')} "
                    f"({explanation['method']}, {explanation['generation_time']:.2f}s)"
                )

                return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")

        return None

    def explain_prediction_batch(
        self,
        analysis_results: List[Union[Dict[str, Any], AnalyticsResults]],
        detail_level: str = "standard",
        user=None,
        force_regenerate: bool = False,
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Generate explanations for multiple analysis results.

        Args:
            analysis_results: List of analysis result dicts or model instances
            detail_level: Detail level for all explanations
            user: User instance for personalization (optional)

        Returns:
            List of explanation results (same order as input)
        """
        explanations = []

        for analysis_result in analysis_results:
            explanation = self.explain_prediction_single(analysis_result, detail_level, user, force_regenerate)
            explanations.append(explanation)

        return explanations

    def build_indicator_explanation(
        self, indicator_name: str, indicator_result: Any, weighted_score: float, context: Dict[str, Any] = None
    ) -> str:
        """
        Build explanation for a specific technical indicator.

        Args:
            indicator_name: Name of the indicator
            indicator_result: Raw indicator calculation result
            weighted_score: Weighted contribution to final score
            context: Additional context information

        Returns:
            Human-readable explanation string
        """
        try:
            base_explanation = self.indicator_templates.get(
                indicator_name, f"{indicator_name} technical analysis indicator"
            )

            # Add weighted score context
            score_impact = "positive" if weighted_score > 0 else "negative" if weighted_score < 0 else "neutral"

            explanation = f"{base_explanation}. Current reading shows {score_impact} impact (weighted score: {weighted_score:.2f})"

            # Add specific indicator details based on type
            if indicator_name == "rsi14" and isinstance(indicator_result, (int, float)):
                if indicator_result > 70:
                    explanation += f". RSI at {indicator_result:.1f} suggests overbought conditions"
                elif indicator_result < 30:
                    explanation += f". RSI at {indicator_result:.1f} suggests oversold conditions"
                else:
                    explanation += f". RSI at {indicator_result:.1f} indicates neutral momentum"

            elif indicator_name == "sma50vs200":
                if weighted_score > 0:
                    explanation += ". 50-day SMA above 200-day SMA (golden cross pattern)"
                elif weighted_score < 0:
                    explanation += ". 50-day SMA below 200-day SMA (death cross pattern)"

            return explanation

        except Exception as e:
            logger.error(f"Error building indicator explanation for {indicator_name}: {str(e)}")
            return f"{indicator_name}: Unable to generate detailed explanation"

    def _prepare_analysis_data(self, analysis_result: Union[Dict[str, Any], AnalyticsResults]) -> Dict[str, Any]:
        """Convert analysis result to standardized dictionary format."""
        if isinstance(analysis_result, dict):
            return analysis_result

        # Handle AnalyticsResults model instance
        try:
            return {
                "symbol": analysis_result.stock.symbol if hasattr(analysis_result, "stock") else "UNKNOWN",
                "score_0_10": float(analysis_result.score_0_10) if hasattr(analysis_result, "score_0_10") else 0.0,
                "composite_raw": (
                    float(analysis_result.composite_raw) if hasattr(analysis_result, "composite_raw") else 0.0
                ),
                "analysis_date": (
                    analysis_result.as_of.isoformat()
                    if hasattr(analysis_result, "as_of")
                    else datetime.now().isoformat()
                ),
                "horizon": analysis_result.horizon if hasattr(analysis_result, "horizon") else "unknown",
                "components": analysis_result.components if hasattr(analysis_result, "components") else {},
                "weighted_scores": {
                    "w_sma50vs200": (
                        float(analysis_result.w_sma50vs200)
                        if hasattr(analysis_result, "w_sma50vs200") and analysis_result.w_sma50vs200
                        else 0.0
                    ),
                    "w_pricevs50": (
                        float(analysis_result.w_pricevs50)
                        if hasattr(analysis_result, "w_pricevs50") and analysis_result.w_pricevs50
                        else 0.0
                    ),
                    "w_rsi14": (
                        float(analysis_result.w_rsi14)
                        if hasattr(analysis_result, "w_rsi14") and analysis_result.w_rsi14
                        else 0.0
                    ),
                    "w_macd12269": (
                        float(analysis_result.w_macd12269)
                        if hasattr(analysis_result, "w_macd12269") and analysis_result.w_macd12269
                        else 0.0
                    ),
                    "w_bbpos20": (
                        float(analysis_result.w_bbpos20)
                        if hasattr(analysis_result, "w_bbpos20") and analysis_result.w_bbpos20
                        else 0.0
                    ),
                    "w_bbwidth20": (
                        float(analysis_result.w_bbwidth20)
                        if hasattr(analysis_result, "w_bbwidth20") and analysis_result.w_bbwidth20
                        else 0.0
                    ),
                    "w_volsurge": (
                        float(analysis_result.w_volsurge)
                        if hasattr(analysis_result, "w_volsurge") and analysis_result.w_volsurge
                        else 0.0
                    ),
                    "w_obv20": (
                        float(analysis_result.w_obv20)
                        if hasattr(analysis_result, "w_obv20") and analysis_result.w_obv20
                        else 0.0
                    ),
                    "w_rel1y": (
                        float(analysis_result.w_rel1y)
                        if hasattr(analysis_result, "w_rel1y") and analysis_result.w_rel1y
                        else 0.0
                    ),
                    "w_rel2y": (
                        float(analysis_result.w_rel2y)
                        if hasattr(analysis_result, "w_rel2y") and analysis_result.w_rel2y
                        else 0.0
                    ),
                    "w_candlerev": (
                        float(analysis_result.w_candlerev)
                        if hasattr(analysis_result, "w_candlerev") and analysis_result.w_candlerev
                        else 0.0
                    ),
                    "w_srcontext": (
                        float(analysis_result.w_srcontext)
                        if hasattr(analysis_result, "w_srcontext") and analysis_result.w_srcontext
                        else 0.0
                    ),
                },
            }
        except Exception as e:
            logger.error(f"Error preparing analysis data: {str(e)}")
            return {"error": "Failed to prepare analysis data"}

    def _generate_llm_explanation(self, analysis_data: Dict[str, Any], detail_level: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Generate explanation using local LLaMA model."""
        try:
            llm_result = self.llm_service.generate_explanation(
                analysis_data=analysis_data, detail_level=detail_level, explanation_type="technical_analysis", language=language
            )

            if llm_result and "content" in llm_result:
                return {
                    "content": llm_result["content"],
                    "confidence_score": llm_result.get("confidence_score", 0.8),
                    "word_count": llm_result.get("word_count", 0),
                    "model_used": llm_result.get("model_used", "llama3.1:70b"),
                    "indicators_explained": self._extract_indicators_explained(analysis_data),
                    "risk_factors": self._extract_risk_factors(analysis_data),
                    "recommendation": self._determine_recommendation(analysis_data.get("score_0_10", 0)),
                }

        except Exception as e:
            logger.error(f"Error generating LLM explanation: {str(e)}")

        return None

    def _generate_multilingual_explanation(
        self, analysis_data: Dict[str, Any], detail_level: str, language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Generate multilingual explanation using translation pipeline for standard/summary modes.

        For detailed mode, uses direct LLM generation.
        For standard/summary modes, generates in English then translates.
        """
        try:
            # For detailed mode, use direct LLM generation (Phase 2 implementation)
            if detail_level == "detailed":
                return self._generate_llm_explanation(analysis_data, detail_level, language)

            # For standard/summary modes, use translation pipeline (Phase 3 implementation)
            if language != "en":
                # Generate English explanation first
                english_explanation = self._generate_llm_explanation(analysis_data, detail_level, "en")

                if not english_explanation or not english_explanation.get("content"):
                    logger.warning(f"Failed to generate English explanation for translation to {language}")
                    return None

                # Translate the content using lazy-loaded translation service
                translated_content = self.translation_service.translate_explanation(
                    english_text=english_explanation["content"],
                    target_language=language,
                    context={
                        "symbol": analysis_data.get("symbol", ""),
                        "score": analysis_data.get("score_0_10", 0),
                        "recommendation": english_explanation.get("recommendation", "HOLD")
                    }
                )

                if translated_content and translated_content.get("translated_text"):
                    # Return translated explanation with enhanced metadata
                    return {
                        "content": translated_content["translated_text"],
                        "confidence_score": english_explanation.get("confidence_score", 0.8),
                        "word_count": len(translated_content["translated_text"].split()),
                        "model_used": english_explanation.get("model_used", "llama3.1:70b"),
                        "indicators_explained": english_explanation.get("indicators_explained", []),
                        "risk_factors": english_explanation.get("risk_factors", []),
                        "recommendation": english_explanation.get("recommendation", "HOLD"),
                        "translation_quality": translated_content.get("quality_score", 0.0),
                        "translation_model": translated_content.get("model_used", "qwen2:latest"),
                        "source_language": "en",
                        "target_language": language,
                        "translation_method": "llm_pipeline"
                    }
                else:
                    logger.error(f"Translation failed for {language}, falling back to English")
                    return english_explanation
            else:
                # English language - use direct LLM generation
                return self._generate_llm_explanation(analysis_data, detail_level, language)

        except Exception as e:
            logger.error(f"Error in multilingual explanation generation: {str(e)}")
            # Fallback to standard LLM explanation
            return self._generate_llm_explanation(analysis_data, detail_level, language)

    def _generate_template_explanation(self, analysis_data: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
        """Generate explanation using template fallback."""
        try:
            symbol = analysis_data.get("symbol", "Unknown")
            score = analysis_data.get("score_0_10", 0)
            weighted_scores = analysis_data.get("weighted_scores", {})

            # Determine recommendation
            recommendation = self._determine_recommendation(score)

            # Find top contributing indicators
            top_indicators = sorted(
                [(k, v) for k, v in weighted_scores.items() if v != 0], key=lambda x: abs(x[1]), reverse=True
            )[:3]

            if detail_level == "summary":
                content = f"{symbol} receives a {score:.1f}/10 analysis score, suggesting a {recommendation} position. "
                if top_indicators:
                    indicator_names = [k.replace("w_", "") for k, v in top_indicators]
                    content += f"Key factors: {', '.join(indicator_names[:2])}."

            elif detail_level == "detailed":
                content = f"Comprehensive analysis of {symbol} yields a score of {score:.1f}/10, indicating a {recommendation} recommendation.\n\n"
                content += "Technical Indicator Analysis:\n"

                for indicator, weight in top_indicators:
                    indicator_name = indicator.replace("w_", "")
                    explanation = self.build_indicator_explanation(
                        indicator_name, weighted_scores.get(indicator), weight
                    )
                    content += f"• {explanation}\n"

                content += f"\nOverall Assessment: The combined technical indicators support a {recommendation} stance "
                content += f"with moderate confidence based on current market conditions."

            else:  # standard
                content = f"{symbol} analysis shows {score:.1f}/10 score ({recommendation}). "
                if top_indicators:
                    top_indicator = top_indicators[0][0].replace("w_", "")
                    impact = "positive" if top_indicators[0][1] > 0 else "negative"
                    content += f"Primary driver: {top_indicator} showing {impact} impact. "
                content += "Consider market conditions and risk tolerance before trading."

            return {
                "content": content,
                "confidence_score": 0.7,  # Lower confidence for template-based
                "word_count": len(content.split()),
                "model_used": "template_fallback",
                "indicators_explained": [k.replace("w_", "") for k, v in top_indicators],
                "risk_factors": self._extract_risk_factors(analysis_data),
                "recommendation": recommendation,
            }

        except Exception as e:
            logger.error(f"Error generating template explanation: {str(e)}")
            return {
                "content": f"Technical analysis completed for {analysis_data.get('symbol', 'Unknown')} with basic template.",
                "confidence_score": 0.5,
                "word_count": 10,
                "model_used": "error_fallback",
                "indicators_explained": [],
                "risk_factors": [],
                "recommendation": "HOLD",
            }

    def _determine_recommendation(self, score: float) -> str:
        """Determine investment recommendation based on score."""
        if score >= 7:
            return "BUY"
        elif score >= 4:
            return "HOLD"
        else:
            return "SELL"

    def _extract_indicators_explained(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract list of indicators that were analyzed."""
        weighted_scores = analysis_data.get("weighted_scores", {})
        return [k.replace("w_", "") for k, v in weighted_scores.items() if v != 0]

    def _extract_risk_factors(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract risk factors based on analysis results."""
        risk_factors = []
        weighted_scores = analysis_data.get("weighted_scores", {})

        # Check for high volatility
        bb_width = weighted_scores.get("w_bbwidth20", 0)
        if abs(bb_width) > 0.5:
            risk_factors.append("High volatility detected")

        # Check for overbought/oversold conditions
        rsi_score = weighted_scores.get("w_rsi14", 0)
        if rsi_score < -0.3:
            risk_factors.append("Potential oversold conditions")
        elif rsi_score > 0.3:
            risk_factors.append("Potential overbought conditions")

        # Check for trend weakness
        sma_score = weighted_scores.get("w_sma50vs200", 0)
        if abs(sma_score) < 0.1:
            risk_factors.append("Uncertain trend direction")

        return risk_factors[:3]  # Limit to top 3 risk factors

    def _validate_detailed_explanation_format(self, content: str) -> Dict[str, Any]:
        """Validate detailed explanation format and suggest fixes."""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "word_count": len(content.split()),
            "section_count": 0,
            "has_proper_headers": False,
            "suggested_fix": None
        }
        
        try:
            import re
            
            # Check word count (should be 600+)
            word_count = len(content.split())
            validation_result["word_count"] = word_count
            
            if word_count < 600:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Insufficient word count: {word_count} (minimum 600)")
            
            # Check for section headers
            section_patterns = [
                r'\*\*\s*Investment\s+Summary\s*:\s*\*\*',
                r'\*\*\s*Technical\s+Analysis\s*:\s*\*\*',
                r'\*\*\s*Risk\s+Assessment\s*:\s*\*\*',
                r'\*\*\s*Entry\s+Strategy\s*:\s*\*\*',
                r'\*\*\s*Market\s+Outlook\s*:\s*\*\*'
            ]
            
            sections_found = 0
            for pattern in section_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    sections_found += 1
            
            validation_result["section_count"] = sections_found
            validation_result["has_proper_headers"] = sections_found >= 3
            
            if sections_found < 3:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Missing section headers: found {sections_found}/5 expected sections")
            
            # Check for malformed headers and suggest fix
            malformed_headers = re.findall(r'\*\*([^*]+?):\*\*+\s*([^*])', content)
            if malformed_headers:
                validation_result["issues"].append("Found malformed headers (extra asterisks)")
                # Apply basic correction
                fixed_content = re.sub(r'\*\*([^*]+?):\*\*+\s*([^*])', r'**\1:** \2', content)
                validation_result["suggested_fix"] = fixed_content
            
        except Exception as e:
            logger.error(f"Error validating explanation format: {str(e)}")
            validation_result["is_valid"] = False
            validation_result["issues"].append("Validation error occurred")
        
        return validation_result

    def _create_cache_key(self, analysis_data: Dict[str, Any], detail_level: str, language: str = "en", user=None) -> str:
        """Create culturally-aware cache key for explanation."""
        symbol = analysis_data.get("symbol", "unknown")
        score = analysis_data.get("score_0_10", 0)
        user_id = user.id if user else "anonymous"

        # Include key weighted scores in cache key for specificity
        weighted_scores = analysis_data.get("weighted_scores", {})
        key_scores = []
        for key in ["w_sma50vs200", "w_rsi14", "w_macd12269"]:
            if key in weighted_scores:
                key_scores.append(f"{key}_{weighted_scores[key]:.2f}")

        # Add cultural context to cache key
        cultural_context = self._get_cultural_context(language, user)
        cultural_suffix = f"_c{cultural_context.get('culture_hash', 'default')}"

        cache_data = f"{symbol}_{score:.1f}_{detail_level}_{language}_{user_id}_{'_'.join(key_scores)}{cultural_suffix}"
        key = hashlib.blake2b(cache_data.encode(), digest_size=16).hexdigest()
        return f"explanation_{key}"

    def _get_cultural_context(self, language: str, user=None) -> Dict[str, Any]:
        """Get cultural context for cache key generation."""
        try:
            import hashlib

            cultural_factors = []

            # Language-specific cultural context
            language_contexts = {
                'fr': ['europe', 'euro_currency', 'cac40_market'],
                'es': ['europe', 'latam', 'euro_currency', 'ibex35_market'],
                'en': ['global', 'usd_currency', 'sp500_market'],
            }

            cultural_factors.extend(language_contexts.get(language, ['global']))

            # User-specific preferences (if available and authenticated)
            if user and hasattr(user, 'profile'):
                # Add timezone-based context
                timezone = getattr(user.profile, 'timezone', None)
                if timezone:
                    if 'Europe' in timezone:
                        cultural_factors.append('europe_timezone')
                    elif 'America' in timezone:
                        cultural_factors.append('americas_timezone')

                # Add preferred currency format
                currency_pref = getattr(user.profile, 'currency_preference', None)
                if currency_pref:
                    cultural_factors.append(f'currency_{currency_pref.lower()}')

            # Create hash of cultural factors for cache key
            cultural_string = '_'.join(sorted(cultural_factors))
            culture_hash = hashlib.blake2b(cultural_string.encode(), digest_size=4).hexdigest()

            return {
                'factors': cultural_factors,
                'culture_hash': culture_hash,
                'language': language,
            }

        except Exception as e:
            logger.error(f"Error generating cultural context: {str(e)}")
            return {'culture_hash': 'default', 'factors': [], 'language': language}

    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "enabled": self.enabled,
            "llm_available": self.llm_service.is_available(),
            "cache_ttl": self.cache_ttl,
            "llm_status": self.llm_service.get_service_status(),
            "template_indicators": len(self.indicator_templates),
        }


# Singleton instance
_explanation_service = None


def get_explanation_service() -> ExplanationService:
    """Get singleton instance of ExplanationService."""
    global _explanation_service
    if _explanation_service is None:
        _explanation_service = ExplanationService()
    return _explanation_service
