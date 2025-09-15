"""
Enhanced translation quality assessment system.
Provides semantic similarity checking, terminology validation, and improved quality scoring.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import json

logger = logging.getLogger(__name__)

# Conditional imports for enhanced quality checking
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.info("sentence-transformers not available - using fallback quality scoring")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.info("numpy not available - using basic similarity calculations")
    NUMPY_AVAILABLE = False


class FinancialTerminologyValidator:
    """Advanced financial terminology validation using multiple approaches."""

    def __init__(self):
        # Comprehensive financial terminology mapping
        self.financial_terms = {
            "en": {
                # Core financial concepts
                "technical_indicators", "moving_average", "bollinger_bands", "relative_strength_index",
                "macd", "rsi", "support_level", "resistance_level", "volume", "volatility",

                # Investment terms
                "buy_recommendation", "sell_recommendation", "hold_recommendation", "bullish", "bearish",
                "portfolio", "diversification", "risk_assessment", "return_on_investment",

                # Market terms
                "market_capitalization", "earnings_per_share", "price_to_earnings_ratio",
                "dividend_yield", "revenue_growth", "profit_margin", "liquidity",

                # Trading terms
                "bid_ask_spread", "order_book", "market_order", "limit_order", "stop_loss",
                "take_profit", "position_sizing", "risk_management"
            },

            "fr": {
                # Core financial concepts
                "indicateurs_techniques", "moyenne_mobile", "bandes_de_bollinger", "indice_de_force_relative",
                "macd", "rsi", "niveau_de_support", "niveau_de_résistance", "volume", "volatilité",

                # Investment terms
                "recommandation_achat", "recommandation_vente", "recommandation_conservation", "haussier", "baissier",
                "portefeuille", "diversification", "évaluation_risque", "retour_sur_investissement",

                # Market terms
                "capitalisation_boursière", "bénéfice_par_action", "ratio_cours_bénéfice",
                "rendement_dividende", "croissance_chiffre_affaires", "marge_bénéficiaire", "liquidité",

                # Trading terms
                "écart_bid_ask", "carnet_ordres", "ordre_marché", "ordre_limite", "stop_loss",
                "prise_bénéfice", "dimensionnement_position", "gestion_risque"
            },

            "es": {
                # Core financial concepts
                "indicadores_técnicos", "media_móvil", "bandas_bollinger", "índice_fuerza_relativa",
                "macd", "rsi", "nivel_soporte", "nivel_resistencia", "volumen", "volatilidad",

                # Investment terms
                "recomendación_compra", "recomendación_venta", "recomendación_mantener", "alcista", "bajista",
                "cartera", "diversificación", "evaluación_riesgo", "retorno_inversión",

                # Market terms
                "capitalización_mercado", "ganancias_por_acción", "relación_precio_ganancias",
                "rendimiento_dividendos", "crecimiento_ingresos", "margen_beneficio", "liquidez",

                # Trading terms
                "diferencial_bid_ask", "libro_órdenes", "orden_mercado", "orden_límite", "stop_loss",
                "toma_beneficios", "tamaño_posición", "gestión_riesgo"
            }
        }

        # Terminology mappings for cross-validation
        self.term_mappings = {
            ("en", "fr"): {
                "technical_indicators": "indicateurs_techniques",
                "moving_average": "moyenne_mobile",
                "buy_recommendation": "recommandation_achat",
                "sell_recommendation": "recommandation_vente",
                "hold_recommendation": "recommandation_conservation",
                "bullish": "haussier",
                "bearish": "baissier",
                "support_level": "niveau_de_support",
                "resistance_level": "niveau_de_résistance",
                "market_capitalization": "capitalisation_boursière",
                "earnings_per_share": "bénéfice_par_action",
                "price_to_earnings_ratio": "ratio_cours_bénéfice",
                "risk_assessment": "évaluation_risque",
                "return_on_investment": "retour_sur_investissement"
            },

            ("en", "es"): {
                "technical_indicators": "indicadores_técnicos",
                "moving_average": "media_móvil",
                "buy_recommendation": "recomendación_compra",
                "sell_recommendation": "recomendación_venta",
                "hold_recommendation": "recomendación_mantener",
                "bullish": "alcista",
                "bearish": "bajista",
                "support_level": "nivel_soporte",
                "resistance_level": "nivel_resistencia",
                "market_capitalization": "capitalización_mercado",
                "earnings_per_share": "ganancias_por_acción",
                "price_to_earnings_ratio": "relación_precio_ganancias",
                "risk_assessment": "evaluación_riesgo",
                "return_on_investment": "retorno_inversión"
            }
        }

    def validate_terminology_accuracy(self, original_text: str, translated_text: str,
                                    source_lang: str, target_lang: str) -> Dict[str, any]:
        """
        Validate financial terminology accuracy in translation.

        Args:
            original_text: Source text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dictionary with terminology validation results
        """
        try:
            validation_result = {
                "terminology_score": 0.0,
                "matched_terms": [],
                "missing_translations": [],
                "incorrect_translations": [],
                "total_financial_terms": 0
            }

            # Get term mappings for language pair
            mapping_key = (source_lang, target_lang)
            if mapping_key not in self.term_mappings:
                return {"terminology_score": 0.5, "note": "No mappings available for this language pair"}

            term_mappings = self.term_mappings[mapping_key]

            # Normalize texts for comparison
            original_normalized = self._normalize_text(original_text)
            translated_normalized = self._normalize_text(translated_text)

            matched_count = 0
            total_terms_found = 0

            # Check each mapped term
            for source_term, target_term in term_mappings.items():
                source_pattern = self._create_term_pattern(source_term)
                target_pattern = self._create_term_pattern(target_term)

                source_found = bool(re.search(source_pattern, original_normalized, re.IGNORECASE))
                target_found = bool(re.search(target_pattern, translated_normalized, re.IGNORECASE))

                if source_found:
                    total_terms_found += 1
                    validation_result["total_financial_terms"] += 1

                    if target_found:
                        matched_count += 1
                        validation_result["matched_terms"].append({
                            "source": source_term,
                            "target": target_term,
                            "found": True
                        })
                    else:
                        validation_result["missing_translations"].append({
                            "source": source_term,
                            "expected_target": target_term,
                            "found": False
                        })

            # Calculate terminology score
            if total_terms_found > 0:
                validation_result["terminology_score"] = matched_count / total_terms_found
            else:
                validation_result["terminology_score"] = 1.0  # No financial terms to validate

            return validation_result

        except Exception as e:
            logger.error(f"Error validating terminology: {str(e)}")
            return {"terminology_score": 0.5, "error": str(e)}

    def _normalize_text(self, text: str) -> str:
        """Normalize text for terminology comparison."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())

        # Replace common punctuation that might interfere with matching
        normalized = re.sub(r'[.,;:!?()"\'-]', ' ', normalized)

        return normalized

    def _create_term_pattern(self, term: str) -> str:
        """Create regex pattern for term matching."""
        # Replace underscores with word boundaries for flexible matching
        pattern = term.replace('_', r'\s+')

        # Add word boundaries to ensure we match complete terms
        return r'\b' + pattern + r'\b'


class SemanticSimilarityChecker:
    """Semantic similarity checking using sentence transformers."""

    def __init__(self):
        self.model = None
        self.model_loaded = False

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight multilingual model for financial texts
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.model_loaded = True
                logger.info("Semantic similarity model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic similarity model: {str(e)}")
                self.model_loaded = False
        else:
            logger.info("Sentence transformers not available - using fallback similarity")

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not self.model_loaded or not self.model:
            # Fallback to simple similarity
            return self._calculate_simple_similarity(text1, text2)

        try:
            # Get embeddings for both texts
            embeddings = self.model.encode([text1, text2])

            if NUMPY_AVAILABLE:
                # Calculate cosine similarity
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                # Convert from [-1, 1] to [0, 1] range
                return (similarity + 1) / 2
            else:
                # Fallback without numpy
                return self._calculate_simple_similarity(text1, text2)

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return self._calculate_simple_similarity(text1, text2)

    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """Simple word-based similarity calculation."""
        try:
            # Tokenize and normalize
            words1 = set(re.findall(r'\w+', text1.lower()))
            words2 = set(re.findall(r'\w+', text2.lower()))

            if not words1 or not words2:
                return 0.0

            # Calculate Jaccard similarity
            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union) if union else 0.0

        except Exception:
            return 0.5


class EnhancedTranslationQualityScorer:
    """Enhanced translation quality scoring with multiple validation approaches."""

    def __init__(self):
        self.terminology_validator = FinancialTerminologyValidator()
        self.semantic_checker = SemanticSimilarityChecker()

        # Updated quality weights
        self.quality_weights = {
            "terminology_accuracy": 0.35,      # Increased weight for terminology
            "semantic_similarity": 0.25,       # New semantic component
            "numerical_preservation": 0.20,    # Maintained importance
            "sentence_structure": 0.15,        # Reduced weight
            "cultural_appropriateness": 0.05   # Reduced weight
        }

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.60,
            "poor": 0.40
        }

    def calculate_enhanced_quality_score(self, original_text: str, translated_text: str,
                                       source_lang: str, target_lang: str) -> Dict[str, any]:
        """
        Calculate comprehensive quality score for translation.

        Args:
            original_text: Original text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dictionary with detailed quality assessment
        """
        try:
            quality_result = {
                "overall_score": 0.0,
                "quality_level": "unknown",
                "component_scores": {},
                "detailed_analysis": {},
                "recommendations": []
            }

            # 1. Terminology accuracy
            terminology_result = self.terminology_validator.validate_terminology_accuracy(
                original_text, translated_text, source_lang, target_lang
            )
            terminology_score = terminology_result.get("terminology_score", 0.5)
            quality_result["component_scores"]["terminology_accuracy"] = terminology_score
            quality_result["detailed_analysis"]["terminology"] = terminology_result

            # 2. Semantic similarity
            semantic_score = self.semantic_checker.calculate_semantic_similarity(
                original_text, translated_text
            )
            quality_result["component_scores"]["semantic_similarity"] = semantic_score

            # 3. Numerical preservation
            numerical_score = self._check_numerical_preservation(original_text, translated_text)
            quality_result["component_scores"]["numerical_preservation"] = numerical_score

            # 4. Sentence structure
            structure_score = self._check_enhanced_sentence_structure(translated_text, target_lang)
            quality_result["component_scores"]["sentence_structure"] = structure_score

            # 5. Cultural appropriateness
            cultural_score = self._check_enhanced_cultural_context(translated_text, target_lang)
            quality_result["component_scores"]["cultural_appropriateness"] = cultural_score

            # Calculate weighted overall score
            overall_score = 0.0
            for component, weight in self.quality_weights.items():
                score = quality_result["component_scores"].get(component, 0.0)
                overall_score += score * weight

            quality_result["overall_score"] = min(1.0, max(0.0, overall_score))

            # Determine quality level
            quality_result["quality_level"] = self._determine_quality_level(overall_score)

            # Generate recommendations
            quality_result["recommendations"] = self._generate_recommendations(quality_result)

            return quality_result

        except Exception as e:
            logger.error(f"Error calculating enhanced quality score: {str(e)}")
            return {
                "overall_score": 0.5,
                "quality_level": "unknown",
                "error": str(e)
            }

    def _check_numerical_preservation(self, original: str, translated: str) -> float:
        """Enhanced numerical preservation check."""
        try:
            # More sophisticated number extraction
            number_patterns = [
                r'\b\d+\.?\d*%\b',      # Percentages
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Currency
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',    # General numbers
                r'\b\d+\.?\d*[KMB]\b'   # Abbreviated numbers (1.5K, 2.3M, etc.)
            ]

            original_numbers = set()
            translated_numbers = set()

            for pattern in number_patterns:
                original_numbers.update(re.findall(pattern, original, re.IGNORECASE))
                translated_numbers.update(re.findall(pattern, translated, re.IGNORECASE))

            if not original_numbers:
                return 1.0  # No numbers to preserve

            # Check preservation
            preserved_count = 0
            for num in original_numbers:
                # Normalize for comparison
                normalized_num = re.sub(r'[,\s]', '', num)
                if any(re.sub(r'[,\s]', '', t_num) == normalized_num for t_num in translated_numbers):
                    preserved_count += 1

            return preserved_count / len(original_numbers)

        except Exception:
            return 0.5

    def _check_enhanced_sentence_structure(self, translated_text: str, target_lang: str) -> float:
        """Enhanced sentence structure validation."""
        try:
            sentences = [s.strip() for s in re.split(r'[.!?]+', translated_text) if s.strip()]
            if not sentences:
                return 0.0

            structure_score = 0.0
            total_checks = 0

            for sentence in sentences:
                words = sentence.split()
                sentence_score = 0.0
                checks = 0

                # Length check
                if 3 <= len(words) <= 50:
                    sentence_score += 1.0
                checks += 1

                # Language-specific structure checks
                if target_lang == "fr":
                    # Check for French article usage
                    french_articles = ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'des']
                    if any(article in sentence.lower().split() for article in french_articles):
                        sentence_score += 1.0
                    checks += 1

                elif target_lang == "es":
                    # Check for Spanish article usage
                    spanish_articles = ['el', 'la', 'los', 'las', 'un', 'una', 'del', 'de', 'al']
                    if any(article in sentence.lower().split() for article in spanish_articles):
                        sentence_score += 1.0
                    checks += 1

                # Check for proper capitalization
                if sentence and sentence[0].isupper():
                    sentence_score += 1.0
                checks += 1

                if checks > 0:
                    structure_score += sentence_score / checks
                    total_checks += 1

            return structure_score / max(1, total_checks)

        except Exception:
            return 0.5

    def _check_enhanced_cultural_context(self, translated_text: str, target_lang: str) -> float:
        """Enhanced cultural context validation."""
        try:
            text_lower = translated_text.lower()

            if target_lang == "fr":
                # French language patterns and cultural context
                french_indicators = {
                    'articles': ['le', 'la', 'les', 'un', 'une', 'des'],
                    'prepositions': ['de', 'du', 'des', 'dans', 'sur', 'avec'],
                    'financial_style': ['analyse', 'recommandation', 'investissement'],
                    'formal_markers': ['nous', 'vous', 'il convient', 'il est']
                }

                score = 0.0
                for category, indicators in french_indicators.items():
                    found = sum(1 for indicator in indicators if indicator in text_lower)
                    if found > 0:
                        score += 0.25

                return min(1.0, score)

            elif target_lang == "es":
                # Spanish language patterns and cultural context
                spanish_indicators = {
                    'articles': ['el', 'la', 'los', 'las', 'un', 'una'],
                    'prepositions': ['de', 'del', 'en', 'con', 'por', 'para'],
                    'financial_style': ['análisis', 'recomendación', 'inversión'],
                    'formal_markers': ['es', 'son', 'está', 'están', 'se recomienda']
                }

                score = 0.0
                for category, indicators in spanish_indicators.items():
                    found = sum(1 for indicator in indicators if indicator in text_lower)
                    if found > 0:
                        score += 0.25

                return min(1.0, score)

            return 0.8  # Default score for unsupported languages

        except Exception:
            return 0.5

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= self.quality_thresholds["excellent"]:
            return "excellent"
        elif score >= self.quality_thresholds["good"]:
            return "good"
        elif score >= self.quality_thresholds["acceptable"]:
            return "acceptable"
        elif score >= self.quality_thresholds["poor"]:
            return "poor"
        else:
            return "very_poor"

    def _generate_recommendations(self, quality_result: Dict[str, any]) -> List[str]:
        """Generate improvement recommendations based on quality analysis."""
        recommendations = []
        component_scores = quality_result.get("component_scores", {})

        if component_scores.get("terminology_accuracy", 1.0) < 0.6:
            recommendations.append("Improve financial terminology translation accuracy")

        if component_scores.get("semantic_similarity", 1.0) < 0.6:
            recommendations.append("Enhance semantic consistency with original meaning")

        if component_scores.get("numerical_preservation", 1.0) < 0.8:
            recommendations.append("Ensure all numerical values are accurately preserved")

        if component_scores.get("sentence_structure", 1.0) < 0.7:
            recommendations.append("Improve sentence structure for target language")

        if component_scores.get("cultural_appropriateness", 1.0) < 0.6:
            recommendations.append("Adapt cultural context for target audience")

        if not recommendations:
            recommendations.append("Translation quality is satisfactory")

        return recommendations


# Singleton instance
_enhanced_quality_scorer = None


def get_enhanced_quality_scorer() -> EnhancedTranslationQualityScorer:
    """Get singleton instance of EnhancedTranslationQualityScorer."""
    global _enhanced_quality_scorer
    if _enhanced_quality_scorer is None:
        _enhanced_quality_scorer = EnhancedTranslationQualityScorer()
    return _enhanced_quality_scorer