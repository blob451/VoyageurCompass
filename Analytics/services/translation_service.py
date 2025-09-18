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
        # Primary and fallback translation models
        self.translation_model = getattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen2:latest")
        self.fallback_translation_models = getattr(settings, "OLLAMA_TRANSLATION_FALLBACK_MODELS", ["phi3:3.8b", "llama3.1:8b"])

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
        
        # Enhanced performance tracking with cache metrics
        self.translation_metrics = {
            "translations_requested": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "successful_translations": 0,
            "failed_translations": 0,
            "average_quality_score": 0.0,
            "model_usage": {},
            "fallback_usage_count": 0,
            "average_translation_time": 0.0,
            "total_cache_entries": 0,
            "cache_memory_usage_mb": 0.0
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
        
        # Enhanced cache lookup with optimization (Phase 3)
        cached_translation = self.optimize_translation_cache(english_text, target_language)
        if cached_translation:
            logger.info(f"Retrieved optimized translation from cache for {target_language}")
            with self._metrics_lock:
                self.translation_metrics["cache_hits"] += 1
                self._update_cache_hit_rate()
            return {
                "translated_text": cached_translation,
                "quality_score": 0.9,  # Assume good quality for cached content
                "model_used": "cached",
                "language": target_language,
                "translation_time": 0.0,
                "cache_hit": True
            }

        # Standard cache lookup fallback
        cache_key = self._create_translation_cache_key(english_text, target_language, context)
        cached_result = cache.get(cache_key)

        if cached_result:
            logger.info(f"Retrieved translation from standard cache for {target_language}")
            with self._metrics_lock:
                self.translation_metrics["cache_hits"] += 1
                self._update_cache_hit_rate()
            return cached_result

        # Record cache miss
        with self._metrics_lock:
            self.translation_metrics["cache_misses"] += 1
            self._update_cache_hit_rate()
        
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

            # Generate translation with model fallback
            start_time = time.time()

            response, model_used = self._generate_translation_with_fallback(
                llm_service, translation_prompt, target_language
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


    def translate_explanations_batch(
        self,
        translations: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Batch translate multiple explanations for improved performance.

        Args:
            translations: List of translation requests, each containing:
                - content: Text to translate
                - target_language: Target language code
                - financial_context: Optional context dict

        Returns:
            List of translation results in same order as input
        """
        if not translations:
            return []

        logger.info(f"Starting batch translation of {len(translations)} explanations")

        # Group translations by target language for optimization
        language_groups = {}
        for i, trans_req in enumerate(translations):
            lang = trans_req.get("target_language", "en")
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append((i, trans_req))

        # Process results in original order
        results = [None] * len(translations)

        # Process each language group
        for target_language, lang_translations in language_groups.items():
            if target_language == "en":
                # No translation needed for English
                for original_index, trans_req in lang_translations:
                    results[original_index] = {
                        "translated_text": trans_req.get("content", ""),
                        "quality_score": 1.0,
                        "model_used": "passthrough",
                        "language": "en",
                        "translation_time": 0.0,
                        "batch_processed": True
                    }
                continue

            # Process non-English translations in batches
            lang_results = self._process_language_batch(lang_translations, target_language)

            # Map results back to original order
            for (original_index, _), result in zip(lang_translations, lang_results):
                results[original_index] = result

        logger.info(f"Completed batch translation with {sum(1 for r in results if r)} successful translations")
        return results

    def _process_language_batch(
        self,
        lang_translations: List[tuple],
        target_language: str
    ) -> List[Optional[Dict[str, Any]]]:
        """Process translations for a specific target language in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(lang_translations)

        # Use ThreadPoolExecutor for parallel translation
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all translation tasks
            future_to_index = {}
            for i, (original_index, trans_req) in enumerate(lang_translations):
                future = executor.submit(
                    self.translate_explanation,
                    trans_req.get("content", ""),
                    target_language,
                    trans_req.get("financial_context", {})
                )
                future_to_index[future] = i

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=self.translation_timeout)
                    if result:
                        result["batch_processed"] = True
                    results[index] = result
                except Exception as e:
                    logger.error(f"Batch translation error for index {index}: {str(e)}")
                    results[index] = None

        return results

    def optimize_translation_cache(self, content: str, target_language: str) -> Optional[str]:
        """
        Optimize cache usage for translation by checking multiple cache keys.
        Implements cache key normalization and fallback mechanisms.
        """
        if target_language == "en":
            return content

        # Try multiple cache key variations for better hit rates
        cache_keys = self._generate_cache_key_variations(content, target_language)

        for cache_key in cache_keys:
            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for translation with key variation: {cache_key[:20]}...")
                return cached_result.get("translated_text")

        return None

    def _generate_cache_key_variations(self, content: str, target_language: str) -> List[str]:
        """Generate multiple cache key variations to improve cache hit rates."""
        # Normalize content for better cache hits
        normalized_content = self._normalize_content_for_cache(content)

        # Generate different cache key variations with enhanced entropy
        keys = []

        # Cache version for invalidation management
        cache_version = "v3.1"

        # Original content hash with enhanced entropy
        content_entropy = f"{len(content)}:{hash(content) % 10000}"
        original_hash = hashlib.blake2b(f"{content}:{content_entropy}".encode(), digest_size=16).hexdigest()
        keys.append(f"translation_{cache_version}_{target_language}_{original_hash}")

        # Normalized content hash with version
        normalized_hash = hashlib.blake2b(f"{normalized_content}:{content_entropy}".encode(), digest_size=16).hexdigest()
        keys.append(f"translation_{cache_version}_{target_language}_norm_{normalized_hash}")

        # Length-based cache key with improved bucketing
        length_bucket = (len(content) // 50) * 50  # Smaller buckets for better granularity
        length_fingerprint = hashlib.blake2b(f"{length_bucket}:{normalized_content[:100]}:{content_entropy}".encode(), digest_size=8).hexdigest()
        keys.append(f"translation_{cache_version}_{target_language}_len_{length_fingerprint}")

        # Semantic fingerprint (for content with similar meaning but different wording)
        semantic_features = self._extract_semantic_features(normalized_content)
        semantic_hash = hashlib.blake2b(f"{semantic_features}:{target_language}".encode(), digest_size=12).hexdigest()
        keys.append(f"translation_{cache_version}_{target_language}_sem_{semantic_hash}")

        return keys

    def _extract_semantic_features(self, content: str) -> str:
        """Extract semantic features for cache key generation."""
        import re

        # Extract key financial terms
        financial_keywords = re.findall(r'\b(?:buy|sell|hold|bullish|bearish|support|resistance|rsi|macd|sma|ema)\b',
                                       content.lower())

        # Extract numerical patterns (percentages, ratios, scores)
        numerical_patterns = re.findall(r'\d+\.?\d*[%]?', content)

        # Create semantic fingerprint
        semantic_signature = f"kw:{len(financial_keywords)}:num:{len(numerical_patterns)}:len:{len(content.split())}"

        return semantic_signature

    def _normalize_content_for_cache(self, content: str) -> str:
        """Normalize content to improve cache hit rates across similar content."""
        import re

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', content.strip())

        # Normalize numbers (round to reduce precision variations)
        def normalize_number(match):
            try:
                num = float(match.group())
                # Round to 2 decimal places
                return f"{num:.2f}"
            except:
                return match.group()

        normalized = re.sub(r'\d+\.\d+', normalize_number, normalized)

        # Normalize currency symbols and percentages
        normalized = re.sub(r'[$£€][\d,]+\.?\d*', '[CURRENCY]', normalized)
        normalized = re.sub(r'\d+\.?\d*%', '[PERCENT]', normalized)

        return normalized

    def warm_translation_cache(
        self,
        content_samples: List[str],
        target_languages: List[str] = None,
        priority_level: str = "standard",
        max_memory_mb: int = 100,
        chunk_size: int = 5
    ) -> Dict[str, Any]:
        """
        Warm translation cache with frequently used content using memory-aware chunking.

        Args:
            content_samples: List of content to pre-translate and cache
            target_languages: Languages to warm cache for (default: fr, es)
            priority_level: Priority level affecting cache TTL
            max_memory_mb: Maximum memory usage for cache warming (default: 100MB)
            chunk_size: Number of items to process per chunk (default: 5)

        Returns:
            Dictionary with warming results and statistics
        """
        if target_languages is None:
            target_languages = ["fr", "es"]

        warming_results = {
            "total_content_samples": len(content_samples),
            "target_languages": target_languages,
            "successful_translations": 0,
            "failed_translations": 0,
            "cache_entries_created": 0,
            "total_warming_time": 0,
            "memory_usage_mb": 0,
            "chunks_processed": 0,
            "warming_details": []
        }

        print(f"Starting memory-aware cache warming for {len(content_samples)} content samples across {len(target_languages)} languages...")
        print(f"Memory limit: {max_memory_mb}MB, Chunk size: {chunk_size}")

        start_time = time.time()

        # Process content samples in memory-aware chunks
        total_chunks = (len(content_samples) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(content_samples))
            chunk_samples = content_samples[start_idx:end_idx]

            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_samples)} samples)...")

            # Monitor memory usage before chunk processing
            initial_memory = self._get_memory_usage_mb()

            chunk_results = self._process_warming_chunk(
                chunk_samples, target_languages, priority_level, start_idx
            )

            # Update results
            warming_results["successful_translations"] += chunk_results["successful"]
            warming_results["failed_translations"] += chunk_results["failed"]
            warming_results["cache_entries_created"] += chunk_results["cache_entries"]
            warming_results["warming_details"].extend(chunk_results["details"])
            warming_results["chunks_processed"] += 1

            # Check memory usage after chunk
            current_memory = self._get_memory_usage_mb()
            memory_increase = current_memory - initial_memory
            warming_results["memory_usage_mb"] = max(warming_results["memory_usage_mb"], current_memory)

            # Memory pressure check
            if current_memory > max_memory_mb:
                print(f"  Memory limit reached ({current_memory:.1f}MB), triggering cache cleanup...")
                self._cleanup_cache_memory()

            # Adaptive delay between chunks based on memory pressure
            if memory_increase > 10:  # If chunk used more than 10MB
                time.sleep(0.5)  # Brief pause to allow GC

        warming_results["total_warming_time"] = time.time() - start_time

        print(f"Memory-aware cache warming completed in {warming_results['total_warming_time']:.2f}s")
        print(f"  Successful: {warming_results['successful_translations']}")
        print(f"  Failed: {warming_results['failed_translations']}")
        print(f"  Cache entries created: {warming_results['cache_entries_created']}")
        print(f"  Peak memory usage: {warming_results['memory_usage_mb']:.1f}MB")
        print(f"  Chunks processed: {warming_results['chunks_processed']}")

        return warming_results

    def _process_warming_chunk(
        self,
        chunk_samples: List[str],
        target_languages: List[str],
        priority_level: str,
        start_index: int
    ) -> Dict[str, Any]:
        """Process a single chunk of content samples for cache warming."""
        chunk_results = {
            "successful": 0,
            "failed": 0,
            "cache_entries": 0,
            "details": []
        }

        for i, content in enumerate(chunk_samples):
            if not content or not content.strip():
                continue

            content_results = {
                "content_index": start_index + i,
                "content_preview": content[:50] + "..." if len(content) > 50 else content,
                "language_results": {}
            }

            for language in target_languages:
                if language == "en":
                    continue  # Skip English as it doesn't need translation

                try:
                    # Perform translation to warm cache
                    translation_result = self.translate_explanation(
                        content,
                        target_language=language,
                        financial_context={
                            "cache_warming": True,
                            "priority": priority_level
                        }
                    )

                    if translation_result and translation_result.get("translated_text"):
                        chunk_results["successful"] += 1
                        content_results["language_results"][language] = {
                            "success": True,
                            "quality_score": translation_result.get("quality_score", 0.0),
                            "model_used": translation_result.get("model_used", "unknown")
                        }

                        # Create additional cache entries with variations
                        self._create_cache_warming_variations(content, language, translation_result)
                        chunk_results["cache_entries"] += 1

                    else:
                        chunk_results["failed"] += 1
                        content_results["language_results"][language] = {
                            "success": False,
                            "error": "Translation failed"
                        }

                except Exception as e:
                    chunk_results["failed"] += 1
                    content_results["language_results"][language] = {
                        "success": False,
                        "error": str(e)
                    }

            chunk_results["details"].append(content_results)

        return chunk_results

    def _create_cache_warming_variations(
        self,
        content: str,
        language: str,
        translation_result: Dict[str, Any]
    ):
        """Create cache variations for improved cache hit rates during warming."""
        # Get cache key variations
        cache_keys = self._generate_cache_key_variations(content, language)

        # Store translation in all cache key variations
        for cache_key in cache_keys:
            try:
                # Use extended TTL for warmed cache entries
                extended_ttl = self.high_quality_cache_ttl * 2  # 4 hours for warmed content

                cache.set(
                    cache_key,
                    translation_result,
                    timeout=extended_ttl
                )
            except Exception as e:
                logger.warning(f"Failed to create cache variation {cache_key}: {str(e)}")

    def warm_common_financial_phrases(self) -> Dict[str, Any]:
        """
        Warm cache with common financial phrases and terminology.
        Pre-populates cache with frequently used financial content.
        """
        common_phrases = [
            "Technical analysis shows bullish signals with strong momentum indicators.",
            "The stock demonstrates oversold conditions with potential for reversal.",
            "Moving averages indicate a strong upward trend continuation.",
            "RSI levels suggest the security is approaching overbought territory.",
            "Volume surge confirms the breakout above resistance levels.",
            "Bollinger Bands indicate increased volatility in the trading range.",
            "MACD crossover signals potential trend change ahead.",
            "Support and resistance levels provide key trading opportunities.",
            "The recommendation is BUY with high confidence based on technical indicators.",
            "HOLD recommendation due to mixed signals in current market conditions.",
            "SELL signal activated as key support levels have been breached.",
            "Momentum indicators suggest continuation of current trend direction.",
            "Price action confirms strong institutional buying interest.",
            "Risk factors include market volatility and sector rotation concerns.",
            "Strong earnings growth supports positive technical outlook."
        ]

        return self.warm_translation_cache(
            content_samples=common_phrases,
            target_languages=["fr", "es"],
            priority_level="high"
        )

    def get_cache_warming_recommendations(
        self,
        usage_patterns: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate cache warming recommendations based on usage patterns.

        Args:
            usage_patterns: Optional dictionary with usage statistics

        Returns:
            List of recommended content for cache warming
        """
        # Default recommendations based on common financial analysis patterns
        recommendations = [
            "Apple Inc. shows strong technical momentum with bullish indicators.",
            "Microsoft Corporation displays mixed signals requiring careful analysis.",
            "Google's technical analysis reveals breakthrough above key resistance.",
            "Tesla demonstrates high volatility with momentum shift patterns.",
            "NVIDIA shows exceptional growth indicators with strong support levels."
        ]

        # Add pattern-based recommendations if usage data available
        if usage_patterns:
            # Extract frequently used symbols
            frequent_symbols = usage_patterns.get("frequent_symbols", [])
            for symbol in frequent_symbols[:5]:  # Top 5 symbols
                recommendations.append(
                    f"{symbol} technical analysis shows {usage_patterns.get('common_sentiment', 'mixed')} signals."
                )

            # Add common score ranges
            common_scores = usage_patterns.get("common_score_ranges", [])
            for score_range in common_scores:
                recommendations.append(
                    f"Analysis indicates {score_range} confidence level with corresponding technical signals."
                )

        return recommendations

    def _generate_translation_with_fallback(
        self,
        llm_service,
        translation_prompt: str,
        target_language: str
    ) -> tuple:
        """
        Generate translation with model fallback chain.

        Returns:
            Tuple of (response, model_used)
        """
        # Try primary model first
        models_to_try = [self.translation_model] + self.fallback_translation_models

        for model_name in models_to_try:
            try:
                # Check model health
                if hasattr(llm_service, 'health_service') and llm_service.health_service:
                    if not llm_service.health_service.is_model_healthy(model_name, llm_service.client):
                        logger.warning(f"Model {model_name} is unhealthy, trying next model")
                        continue

                logger.info(f"Attempting translation with model: {model_name}")

                response = llm_service.client.generate(
                    model=model_name,
                    prompt=translation_prompt,
                    options=self._get_translation_options(target_language)
                )

                if response and "response" in response:
                    logger.info(f"Translation successful with model: {model_name}")
                    return response, model_name

            except Exception as e:
                logger.warning(f"Translation failed with model {model_name}: {str(e)}")
                continue

        # All models failed
        logger.error("All translation models failed")
        return None, None

    def _check_model_availability(self, llm_service, model_name: str) -> bool:
        """Check if a translation model is available."""
        try:
            # Try to list models or ping the specific model
            if hasattr(llm_service.client, 'list'):
                available_models = llm_service.client.list()
                model_names = [model.get('name', '') for model in available_models.get('models', [])]
                return any(model_name in name for name in model_names)
            return True  # Assume available if we can't check
        except Exception as e:
            logger.warning(f"Could not check availability for model {model_name}: {str(e)}")
            return False

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback if psutil not available
            import sys
            return sys.getsizeof(self) / 1024 / 1024

    def _cleanup_cache_memory(self):
        """Clean up cache memory when approaching limits."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Log memory cleanup
            logger.info("Cache memory cleanup triggered - garbage collection completed")

        except Exception as e:
            logger.warning(f"Cache memory cleanup failed: {str(e)}")

    def _update_cache_hit_rate(self):
        """Update cache hit rate calculation."""
        total_requests = self.translation_metrics["cache_hits"] + self.translation_metrics["cache_misses"]
        if total_requests > 0:
            self.translation_metrics["cache_hit_rate"] = (
                self.translation_metrics["cache_hits"] / total_requests
            ) * 100

    def _update_model_usage_metrics(self, model_name: str, is_fallback: bool = False):
        """Update model usage statistics."""
        with self._metrics_lock:
            if model_name not in self.translation_metrics["model_usage"]:
                self.translation_metrics["model_usage"][model_name] = 0
            self.translation_metrics["model_usage"][model_name] += 1

            if is_fallback:
                self.translation_metrics["fallback_usage_count"] += 1

    def _update_translation_time_metrics(self, translation_time: float):
        """Update average translation time."""
        with self._metrics_lock:
            current_avg = self.translation_metrics["average_translation_time"]
            total_translations = self.translation_metrics["successful_translations"]

            if total_translations > 0:
                # Running average calculation
                self.translation_metrics["average_translation_time"] = (
                    (current_avg * (total_translations - 1)) + translation_time
                ) / total_translations

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache and translation metrics."""
        with self._metrics_lock:
            metrics = self.translation_metrics.copy()

            # Add real-time calculations
            metrics["uptime_minutes"] = (time.time() - getattr(self, '_start_time', time.time())) / 60
            metrics["cache_efficiency"] = metrics["cache_hit_rate"] / 100 if metrics["cache_hit_rate"] > 0 else 0

            # Model efficiency metrics
            if metrics["model_usage"]:
                total_model_usage = sum(metrics["model_usage"].values())
                metrics["primary_model_efficiency"] = (
                    metrics["model_usage"].get(self.translation_model, 0) / total_model_usage * 100
                    if total_model_usage > 0 else 0
                )

            return metrics

    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._metrics_lock:
            self.translation_metrics = {
                "translations_requested": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
                "successful_translations": 0,
                "failed_translations": 0,
                "average_quality_score": 0.0,
                "model_usage": {},
                "fallback_usage_count": 0,
                "average_translation_time": 0.0,
                "total_cache_entries": 0,
                "cache_memory_usage_mb": 0.0
            }
            self._start_time = time.time()

# Singleton instance
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get singleton instance of TranslationService."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service