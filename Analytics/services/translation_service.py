"""
Translation service for multilingual financial explanations.
Provides high-quality financial translation capabilities using specialised translation models.
"""

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.cache import cache

from Analytics.services.local_llm_service import get_local_llm_service
from Analytics.services.security_validator import get_security_validator, sanitize_financial_input
from Analytics.services.enhanced_translation_quality import get_enhanced_quality_scorer

logger = logging.getLogger(__name__)

# Conditional import for ollama - graceful degradation
try:
    import ollama
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("Ollama not available - translation service will operate in fallback mode")
    ollama = None
    Client = None
    OLLAMA_AVAILABLE = False


class FinancialTerminologyMapper:
    """Financial terminology mapping for accurate translations."""
    
    def __init__(self):
        # English -> French financial terminology
        self.en_to_fr = {
            "technical indicators": "indicateurs techniques",
            "buy recommendation": "recommandation d'achat",
            "sell recommendation": "recommandation de vente", 
            "hold recommendation": "recommandation de conservation",
            "moving average": "moyenne mobile",
            "relative strength index": "indice de force relative",
            "bollinger bands": "bandes de Bollinger",
            "volume surge": "pic de volume",
            "resistance level": "niveau de résistance",
            "support level": "niveau de support",
            "bullish signal": "signal haussier",
            "bearish signal": "signal baissier",
            "price momentum": "momentum des prix",
            "volatility": "volatilité",
            "market capitalisation": "capitalisation boursière",
            "earnings per share": "bénéfice par action",
            "price-to-earnings ratio": "ratio cours/bénéfice",
            "dividend yield": "rendement du dividende",
            "revenue growth": "croissance du chiffre d'affaires",
            "profit margin": "marge bénéficiaire",
            "return on investment": "retour sur investissement",
            "risk assessment": "évaluation du risque",
            "portfolio": "portefeuille",
            "investment strategy": "stratégie d'investissement",
            "financial analysis": "analyse financière",
        }
        
        # English -> Spanish financial terminology
        self.en_to_es = {
            "technical indicators": "indicadores técnicos",
            "buy recommendation": "recomendación de compra",
            "sell recommendation": "recomendación de venta",
            "hold recommendation": "recomendación de mantener",
            "moving average": "media móvil",
            "relative strength index": "índice de fuerza relativa",
            "bollinger bands": "bandas de Bollinger",
            "volume surge": "aumento de volumen",
            "resistance level": "nivel de resistencia",
            "support level": "nivel de soporte",
            "bullish signal": "señal alcista",
            "bearish signal": "señal bajista",
            "price momentum": "momentum del precio",
            "volatility": "volatilidad",
            "market capitalisation": "capitalización de mercado",
            "earnings per share": "ganancias por acción",
            "price-to-earnings ratio": "relación precio-ganancias",
            "dividend yield": "rendimiento de dividendos",
            "revenue growth": "crecimiento de ingresos",
            "profit margin": "margen de beneficio",
            "return on investment": "retorno de inversión",
            "risk assessment": "evaluación de riesgo",
            "portfolio": "cartera",
            "investment strategy": "estrategia de inversión",
            "financial analysis": "análisis financiero",
        }
    
    def get_terminology_context(self, target_language: str) -> str:
        """Get terminology context for translation prompts."""
        if target_language == "fr":
            terms = "\n".join([f"'{en}' -> '{fr}'" for en, fr in list(self.en_to_fr.items())[:15]])
            return f"Key financial terminology translations (English -> French):\n{terms}"
        elif target_language == "es":
            terms = "\n".join([f"'{en}' -> '{es}'" for en, es in list(self.en_to_es.items())[:15]])
            return f"Key financial terminology translations (English -> Spanish):\n{terms}"
        else:
            return "Use accurate financial terminology appropriate for the target language."

    def get_terminology_for_language(self, target_language: str) -> Dict[str, str]:
        """Get terminology dictionary for a specific language."""
        if target_language == "fr":
            return self.en_to_fr
        elif target_language == "es":
            return self.en_to_es
        else:
            return {}


class TranslationQualityScorer:
    """Quality scoring system for translation assessment."""
    
    def __init__(self):
        self.quality_indicators = {
            "financial_terms_preserved": 0.4,
            "sentence_structure_coherent": 0.3,
            "numerical_values_preserved": 0.2,
            "cultural_context_appropriate": 0.1,
        }
    
    def calculate_translation_quality(self, original_text: str, translated_text: str, target_language: str) -> float:
        """
        Calculate translation quality score based on various metrics.
        
        Args:
            original_text: Original English text
            translated_text: Translated text
            target_language: Target language code
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            score = 0.0
            
            # Check if financial terms are preserved
            terminology_score = self._check_financial_terminology(original_text, translated_text, target_language)
            score += terminology_score * self.quality_indicators["financial_terms_preserved"]
            
            # Check sentence structure coherency
            structure_score = self._check_sentence_structure(translated_text)
            score += structure_score * self.quality_indicators["sentence_structure_coherent"]
            
            # Check numerical values preservation
            numerical_score = self._check_numerical_preservation(original_text, translated_text)
            score += numerical_score * self.quality_indicators["numerical_values_preserved"]
            
            # Check cultural context appropriateness
            context_score = self._check_cultural_context(translated_text, target_language)
            score += context_score * self.quality_indicators["cultural_context_appropriate"]
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating translation quality: {str(e)}")
            return 0.5  # Default neutral score
    
    def _check_financial_terminology(self, original: str, translated: str, target_lang: str) -> float:
        """Check if financial terminology is correctly translated."""
        try:
            # Simple heuristic: check if common financial terms appear
            financial_indicators = ['%', '$', 'BUY', 'SELL', 'HOLD']
            preserved_count = sum(1 for indicator in financial_indicators if indicator in translated)
            max_possible = len(financial_indicators)
            
            # Additional check for language-specific terms
            if target_lang == "fr":
                french_terms = ['achat', 'vente', 'conservation', 'analyse', 'recommandation']
                preserved_count += sum(1 for term in french_terms if term.lower() in translated.lower())
                max_possible += len(french_terms)
            elif target_lang == "es":
                spanish_terms = ['compra', 'venta', 'mantener', 'análisis', 'recomendación']
                preserved_count += sum(1 for term in spanish_terms if term.lower() in translated.lower())
                max_possible += len(spanish_terms)
            
            return min(1.0, preserved_count / max(1, max_possible))
            
        except Exception:
            return 0.5
    
    def _check_sentence_structure(self, translated_text: str) -> float:
        """Check sentence structure coherency."""
        try:
            sentences = [s.strip() for s in translated_text.split('.') if s.strip()]
            if not sentences:
                return 0.0
            
            # Simple coherency checks
            coherent_sentences = 0
            for sentence in sentences:
                # Check for reasonable sentence length
                words = sentence.split()
                if 3 <= len(words) <= 50:  # Reasonable sentence length
                    coherent_sentences += 1
                    
            return coherent_sentences / max(1, len(sentences))
            
        except Exception:
            return 0.5
    
    def _check_numerical_preservation(self, original: str, translated: str) -> float:
        """Check if numerical values are preserved."""
        import re
        try:
            # Extract numbers from both texts
            original_numbers = set(re.findall(r'\d+\.?\d*', original))
            translated_numbers = set(re.findall(r'\d+\.?\d*', translated))
            
            if not original_numbers:
                return 1.0  # No numbers to preserve
            
            preserved_numbers = original_numbers.intersection(translated_numbers)
            return len(preserved_numbers) / len(original_numbers)
            
        except Exception:
            return 0.5
    
    def _check_cultural_context(self, translated_text: str, target_language: str) -> float:
        """Check cultural context appropriateness."""
        try:
            # Simple heuristic: check for appropriate language patterns
            if target_language == "fr":
                # Check for French language patterns
                french_patterns = ['de la', 'de le', 'du', 'des', 'une', 'un']
                pattern_count = sum(1 for pattern in french_patterns if pattern in translated_text.lower())
                return min(1.0, pattern_count / 3)  # Expect at least some French patterns
            elif target_language == "es":
                # Check for Spanish language patterns
                spanish_patterns = ['de la', 'del', 'una', 'un', 'el', 'la']
                pattern_count = sum(1 for pattern in spanish_patterns if pattern in translated_text.lower())
                return min(1.0, pattern_count / 3)  # Expect at least some Spanish patterns
            
            return 0.8  # Default good score for other languages
            
        except Exception:
            return 0.5


class TranslationService:
    """
    Translation service for converting English financial explanations to French and Spanish.
    Implements quality scoring, caching, and fallback mechanisms for reliable translations.
    """
    
    def __init__(self):
        self.translation_model = getattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen2:3b")
        self.cache_ttl = getattr(settings, "EXPLANATION_CACHE_TTL", 1800)  # 30 minutes default
        self.translation_timeout = 45  # Shorter timeout for translation
        self.high_quality_cache_ttl = 7200  # 2 hours for high-quality translations
        self.batch_size = 3  # Maximum concurrent translations
        
        # Translation support configuration
        self.supported_languages = ["en", "fr", "es"]
        self.language_names = {
            "en": "English",
            "fr": "French", 
            "es": "Spanish"
        }
        
        # Initialise components
        self.terminology_mapper = FinancialTerminologyMapper()
        self.quality_scorer = TranslationQualityScorer()
        self.enhanced_quality_scorer = get_enhanced_quality_scorer()
        self.security_validator = get_security_validator()
        
        # Caching configuration
        self.cache_prefix = "translation:"
        
        # Performance tracking
        self.translation_metrics = {
            "translations_requested": 0,
            "cache_hits": 0,
            "successful_translations": 0,
            "failed_translations": 0,
            "average_quality_score": 0.0,
        }
        
        self._metrics_lock = threading.Lock()
    
    def translate_explanation(
        self, 
        english_text: str, 
        target_language: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Translate English financial explanation to target language.
        
        Args:
            english_text: Original English explanation
            target_language: Target language code ('fr' or 'es')
            context: Additional context for translation
            
        Returns:
            Dictionary with translation results or None if failed
        """
        if not english_text or not english_text.strip():
            logger.warning("Empty text provided for translation")
            return None
        
        if target_language not in ["fr", "es"]:
            logger.warning(f"Unsupported target language: {target_language}")
            return None
        
        # Update metrics
        with self._metrics_lock:
            self.translation_metrics["translations_requested"] += 1
        
        # Check cache first
        cache_key = self._create_translation_cache_key(english_text, target_language, context)
        cached_result = cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Retrieved translation from cache for {target_language}")
            with self._metrics_lock:
                self.translation_metrics["cache_hits"] += 1
            return cached_result
        
        try:
            # Get LLM service for translation
            llm_service = get_local_llm_service()
            
            if not llm_service.is_available():
                logger.error("LLM service not available for translation")
                # Try enhanced fallback first
                enhanced_fallback = self._create_enhanced_fallback_translation(english_text, target_language, context)
                return enhanced_fallback or self._create_fallback_translation(english_text, target_language)
            
            # Security validation for input
            sanitized_text = sanitize_financial_input(english_text)
            if sanitized_text != english_text:
                logger.warning(f"Input sanitized for translation: {len(english_text) - len(sanitized_text)} chars removed")

            # Build translation prompt
            translation_prompt = self._build_translation_prompt(sanitized_text, target_language, context)
            
            # Generate translation
            start_time = time.time()
            
            response = llm_service.client.generate(
                model=self.translation_model,
                prompt=translation_prompt,
                options=self._get_translation_options(target_language)
            )
            
            generation_time = time.time() - start_time
            
            if not response or "response" not in response:
                logger.error("Invalid response from translation model")
                with self._metrics_lock:
                    self.translation_metrics["failed_translations"] += 1
                enhanced_fallback = self._create_enhanced_fallback_translation(english_text, target_language, context)
                return enhanced_fallback or self._create_fallback_translation(english_text, target_language)
            
            translated_text = response["response"].strip()
            
            # Calculate enhanced quality score
            enhanced_quality_result = self.enhanced_quality_scorer.calculate_enhanced_quality_score(
                sanitized_text, translated_text, "en", target_language
            )
            quality_score = enhanced_quality_result.get("overall_score", 0.5)
            quality_level = enhanced_quality_result.get("quality_level", "unknown")

            # Fallback to basic quality score if enhanced scorer fails
            if quality_score == 0.5 and enhanced_quality_result.get("error"):
                quality_score = self.quality_scorer.calculate_translation_quality(
                    sanitized_text, translated_text, target_language
                )
                quality_level = "basic_scoring"
            
            # Create result with enhanced quality data
            result = {
                "original_text": english_text,
                "sanitized_text": sanitized_text,
                "translated_text": translated_text,
                "target_language": target_language,
                "language_name": self.language_names[target_language],
                "translation_model": self.translation_model,
                "quality_score": quality_score,
                "quality_level": quality_level,
                "enhanced_quality_analysis": enhanced_quality_result,
                "generation_time": generation_time,
                "timestamp": time.time(),
                "word_count_original": len(english_text.split()),
                "word_count_translated": len(translated_text.split()),
                "cached": False,
                "input_sanitized": sanitized_text != english_text,
            }
            
            # Cache successful translation with dynamic TTL based on quality
            if quality_score >= 0.8:  # High-quality translations get longer cache time
                cache.set(cache_key, result, self.high_quality_cache_ttl)
                logger.info(f"Cached high-quality translation with score {quality_score:.2f} for {self.high_quality_cache_ttl}s")
            elif quality_score >= 0.6:  # Standard quality translations
                cache.set(cache_key, result, self.cache_ttl)
                logger.info(f"Cached standard translation with score {quality_score:.2f} for {self.cache_ttl}s")
            
            # Update metrics
            with self._metrics_lock:
                self.translation_metrics["successful_translations"] += 1
                # Update running average quality score
                total_successful = self.translation_metrics["successful_translations"]
                current_avg = self.translation_metrics["average_quality_score"]
                self.translation_metrics["average_quality_score"] = (
                    (current_avg * (total_successful - 1) + quality_score) / total_successful
                )
            
            logger.info(
                f"Translated explanation to {target_language} in {generation_time:.2f}s "
                f"(quality: {quality_score:.2f}, {len(translated_text)} chars)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during translation to {target_language}: {str(e)}")
            with self._metrics_lock:
                self.translation_metrics["failed_translations"] += 1
            enhanced_fallback = self._create_enhanced_fallback_translation(english_text, target_language, context)
            return enhanced_fallback or self._create_fallback_translation(english_text, target_language)
    
    def _build_translation_prompt(self, text: str, target_language: str, context: Optional[Dict[str, Any]]) -> str:
        """Build specialised translation prompt for financial content."""
        target_lang_name = self.language_names[target_language]
        
        # Get financial terminology context
        terminology_context = self.terminology_mapper.get_terminology_context(target_language)
        
        # Context information
        context_info = ""
        if context:
            symbol = context.get("symbol", "")
            score = context.get("score", "")
            if symbol and score:
                context_info = f"\nThis analysis is for stock symbol {symbol} with a score of {score}/10."
        
        # Build comprehensive prompt
        prompt = f"""You are a professional financial translator specialising in investment analysis translations from English to {target_lang_name}.

{terminology_context}

Translation Guidelines:
1. Preserve all financial terminology accurately using the mappings above
2. Maintain all numerical values, percentages, and currency symbols exactly as they appear
3. Keep the professional investment research tone and style
4. Ensure the translation reads naturally in {target_lang_name}
5. Preserve the structure and formatting of the original text
6. Do not add explanations or interpretations - provide only the translation

{context_info}

Text to translate to {target_lang_name}:
{text}

Provide only the {target_lang_name} translation:"""
        
        return prompt
    
    def _get_translation_options(self, target_language: str) -> Dict[str, Any]:
        """Get optimised generation options for translation."""
        # Calculate expected translation length (typically 10-20% longer)
        base_options = {
            "temperature": 0.2,  # Low temperature for accurate translation
            "top_p": 0.7,
            "num_predict": 300,  # Adequate length for translations
            "stop": ["Translation:", "English:", "Original:"],
            "repeat_penalty": 1.1,
            "top_k": 20,
        }
        
        return base_options
    
    def _create_translation_cache_key(self, text: str, target_language: str, context: Optional[Dict[str, Any]]) -> str:
        """Create cache key for translation results."""
        # Create hash of text and parameters
        context_str = ""
        if context:
            context_str = f"_{context.get('symbol', '')}_{context.get('score', '')}"
        
        text_hash = hashlib.blake2b(f"{text}_{target_language}{context_str}".encode(), digest_size=16).hexdigest()
        return f"{self.cache_prefix}{target_language}_{text_hash}"
    
    def _create_enhanced_fallback_translation(self, english_text: str, target_language: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Create enhanced fallback translation with terminology mapping."""
        try:
            logger.info(f"Using enhanced fallback translation for {target_language}")

            # Get terminology context for better fallback
            terminology_context = self.terminology_mapper.get_terminology_context(target_language)

            # Apply basic terminology mapping
            enhanced_text = english_text
            terminology = self.terminology_mapper.get_terminology_for_language(target_language)

            for english_term, translated_term in terminology.items():
                if english_term.lower() in enhanced_text.lower():
                    enhanced_text = enhanced_text.replace(english_term.upper(), translated_term.upper())
                    enhanced_text = enhanced_text.replace(english_term.lower(), translated_term.lower())
                    enhanced_text = enhanced_text.replace(english_term.capitalize(), translated_term.capitalize())

            fallback_text = f"[Enhanced {self.language_names[target_language]} translation] {enhanced_text}"

            return {
                "original_text": english_text,
                "translated_text": fallback_text,
                "target_language": target_language,
                "language_name": self.language_names[target_language],
                "translation_model": "enhanced_fallback",
                "quality_score": 0.5,  # Better than basic fallback
                "generation_time": 0.1,
                "timestamp": time.time(),
                "word_count_original": len(english_text.split()),
                "word_count_translated": len(fallback_text.split()),
                "cached": False,
                "fallback": True,
                "enhanced_fallback": True,
            }

        except Exception as e:
            logger.error(f"Enhanced fallback translation failed: {str(e)}")
            return None

    def _create_fallback_translation(self, english_text: str, target_language: str) -> Dict[str, Any]:
        """Create basic fallback translation result when model translation fails."""
        logger.info(f"Using basic fallback translation for {target_language}")

        fallback_text = f"[{self.language_names[target_language]} translation not available] {english_text}"
        
        return {
            "original_text": english_text,
            "translated_text": fallback_text,
            "target_language": target_language,
            "language_name": self.language_names[target_language],
            "translation_model": "fallback",
            "quality_score": 0.3,  # Low quality score for fallback
            "generation_time": 0.0,
            "timestamp": time.time(),
            "word_count_original": len(english_text.split()),
            "word_count_translated": len(fallback_text.split()),
            "cached": False,
            "fallback": True,
        }
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on common patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code ('en', 'fr', 'es', or 'unknown')
        """
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # French indicators
        french_indicators = ['de la', 'du', 'une', 'avec', 'pour', 'dans', 'sur']
        french_score = sum(1 for indicator in french_indicators if indicator in text_lower)
        
        # Spanish indicators  
        spanish_indicators = ['de la', 'del', 'una', 'con', 'para', 'en', 'sobre']
        spanish_score = sum(1 for indicator in spanish_indicators if indicator in text_lower)
        
        # English indicators
        english_indicators = ['the', 'and', 'with', 'for', 'this', 'that', 'buy', 'sell']
        english_score = sum(1 for indicator in english_indicators if indicator in text_lower)
        
        # Determine language based on scores
        if english_score >= french_score and english_score >= spanish_score:
            return "en"
        elif french_score >= spanish_score:
            return "fr"
        elif spanish_score > 0:
            return "es"
        else:
            return "unknown"
    
    def is_translation_available(self, target_language: str) -> bool:
        """Check if translation is available for target language."""
        if target_language not in self.supported_languages:
            return False
        
        # Check if translation model is available
        llm_service = get_local_llm_service()
        if not llm_service.is_available():
            return False
        
        return llm_service._verify_model_availability(self.translation_model)
    
    def get_translation_metrics(self) -> Dict[str, Any]:
        """Get translation service metrics."""
        with self._metrics_lock:
            metrics = self.translation_metrics.copy()
        
        # Calculate success rate
        total_attempts = metrics["successful_translations"] + metrics["failed_translations"]
        if total_attempts > 0:
            metrics["success_rate"] = metrics["successful_translations"] / total_attempts
            metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["translations_requested"]
        else:
            metrics["success_rate"] = 0.0
            metrics["cache_hit_rate"] = 0.0
        
        return {
            "translation_model": self.translation_model,
            "supported_languages": self.supported_languages,
            "cache_ttl": self.cache_ttl,
            "metrics": metrics,
            "translation_available": {
                lang: self.is_translation_available(lang) for lang in ["fr", "es"]
            }
        }


# Singleton instance
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get singleton instance of TranslationService."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service