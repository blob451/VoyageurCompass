"""
Local LLM Service for VoyageurCompass Financial Analysis Explanations.
Integrates with Ollama running LLaMA 3.1 70B model for generating natural language explanations.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from django.conf import settings
from django.core.cache import cache
import ollama
from ollama import Client

logger = logging.getLogger(__name__)


class LocalLLMService:
    """Service for interacting with local LLaMA models via Ollama with performance optimization."""
    
    def __init__(self):
        # Model hierarchy for performance optimization
        self.primary_model = "llama3.1:8b"     # Fast model for most requests
        self.detailed_model = "llama3.1:70b"   # Detailed model for complex requests
        self.current_model = self.primary_model  # Start with fast model
        
        self.client = None
        self.max_retries = 3
        self.timeout = 60
        self.generation_timeout = 45  # Reduced from 90 for faster fallback
        self.performance_mode = True  # Enable performance optimizations
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama client connection."""
        try:
            ollama_host = getattr(settings, 'OLLAMA_HOST', 'localhost')
            ollama_port = getattr(settings, 'OLLAMA_PORT', 11434)
            
            self.client = Client(host=f'http://{ollama_host}:{ollama_port}')
            
            # Verify connection and model availability
            primary_available = self._verify_model_availability(self.primary_model)
            detailed_available = self._verify_model_availability(self.detailed_model)
            
            if primary_available:
                logger.info(f"Local LLM service initialized with primary model: {self.primary_model}")
                if detailed_available:
                    logger.info(f"Detailed model also available: {self.detailed_model}")
            elif detailed_available:
                logger.info(f"Local LLM service initialized with detailed model only: {self.detailed_model}")
                self.current_model = self.detailed_model
            else:
                logger.warning(f"No models available, attempting to pull {self.primary_model}")
                try:
                    self._pull_model(self.primary_model)
                except:
                    logger.warning(f"Failed to pull {self.primary_model}, trying {self.detailed_model}")
                    self._pull_model(self.detailed_model)
                
        except Exception as e:
            logger.error(f"Failed to initialize Local LLM service: {str(e)}")
            self.client = None
    
    def _verify_model_availability(self, model_name: str = None) -> bool:
        """Check if the specified model is available."""
        try:
            if not self.client:
                return False
            
            target_model = model_name or self.current_model
            models = self.client.list()
            available_models = [model['name'] for model in models.get('models', [])]
            return target_model in available_models
            
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
    
    def _select_optimal_model(self, detail_level: str) -> str:
        """Select the most appropriate model based on detail level and availability."""
        if not self.performance_mode:
            return self.detailed_model
        
        # For optimal performance and reliability, use the 8B model for all detail levels
        # The difference in quality is minimal compared to the significant performance gain
        if self._verify_model_availability(self.primary_model):
            return self.primary_model
        
        # Fallback to detailed model only if primary is unavailable
        elif self._verify_model_availability(self.detailed_model):
            return self.detailed_model
        
        # Final fallback
        return self.primary_model
    
    def _pull_model(self, model_name: str = None):
        """Pull the specified model if not available."""
        try:
            if not self.client:
                raise Exception("Ollama client not initialized")
            
            target_model = model_name or self.primary_model
            logger.info(f"Pulling model {target_model}...")
            self.client.pull(target_model)
            logger.info(f"Model {target_model} pulled successfully")
            
        except Exception as e:
            logger.error(f"Failed to pull model {target_model}: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        if not self.client:
            return False
        
        # Check if any model is available
        return (self._verify_model_availability(self.primary_model) or 
                self._verify_model_availability(self.detailed_model))
    
    def generate_explanation(self, 
                           analysis_data: Dict[str, Any], 
                           detail_level: str = 'standard',
                           explanation_type: str = 'technical_analysis') -> Optional[Dict[str, Any]]:
        """
        Generate natural language explanation for financial analysis.
        
        Args:
            analysis_data: Dictionary containing analysis results
            detail_level: 'summary', 'standard', or 'detailed'
            explanation_type: Type of explanation to generate
            
        Returns:
            Dictionary with explanation content or None if failed
        """
        if not self.is_available():
            logger.warning("Local LLM service not available")
            return None
        
        # Create cache key
        cache_key = self._create_cache_key(analysis_data, detail_level, explanation_type)
        
        # Check cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Retrieved explanation from cache")
            return cached_result
        
        try:
            # Select optimal model for this request
            selected_model = self._select_optimal_model(detail_level)
            prompt = self._build_prompt(analysis_data, detail_level, explanation_type)
            
            logger.info(f"[LLM SERVICE] Generating {detail_level} explanation using model: {selected_model}")
            logger.info(f"[LLM SERVICE] Target symbol: {analysis_data.get('symbol', 'Unknown')}, Score: {analysis_data.get('score_0_10', 0)}")
            start_time = time.time()
            
            # Add timeout handling
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"LLM generation exceeded {self.generation_timeout} seconds")
            
            # Set timeout (Unix systems only)
            timeout_set = False
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.generation_timeout)
                timeout_set = True
            
            try:
                response = self.client.generate(
                    model=selected_model,
                    prompt=prompt,
                    options=self._get_generation_options(detail_level, selected_model)
                )
                
                if timeout_set:
                    signal.alarm(0)  # Cancel timeout
                    
            except TimeoutError as e:
                if timeout_set:
                    signal.alarm(0)
                logger.warning(f"LLM generation timed out after {self.generation_timeout}s")
                return None
            
            except Exception as e:
                if timeout_set:
                    signal.alarm(0)
                logger.error(f"Error during LLM generation: {str(e)}")
                return None
            
            generation_time = time.time() - start_time
            
            if not response or 'response' not in response:
                logger.error("Invalid response from LLM")
                return None
            
            # Enhanced performance logging with model context
            symbol = analysis_data.get('symbol', 'Unknown')
            if generation_time > 60:
                logger.warning(f"[LLM PERFORMANCE] Slow generation for {symbol}: {generation_time:.2f}s using {selected_model}")
            elif generation_time > 10:
                logger.info(f"[LLM PERFORMANCE] Moderate generation time for {symbol}: {generation_time:.2f}s using {selected_model}")
            else:
                logger.info(f"[LLM PERFORMANCE] Fast generation for {symbol}: {generation_time:.2f}s using {selected_model}")
            
            explanation_content = response['response'].strip()
            
            # Log content metrics
            word_count = len(explanation_content.split())
            logger.info(f"[LLM CONTENT] Generated {word_count} words ({len(explanation_content)} chars) for {symbol}")
            
            result = {
                'content': explanation_content,
                'detail_level': detail_level,
                'explanation_type': explanation_type,
                'generation_time': generation_time,
                'model_used': selected_model,
                'timestamp': time.time(),
                'word_count': len(explanation_content.split()),
                'confidence_score': self._calculate_confidence_score(explanation_content),
                'performance_optimized': self.performance_mode
            }
            
            # Cache the result for 5 minutes
            cache.set(cache_key, result, 300)
            
            logger.info(f"[LLM SUCCESS] Generated {detail_level} explanation for {symbol} in {generation_time:.2f}s ({len(explanation_content)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return None
    
    def _build_prompt(self, analysis_data: Dict[str, Any], detail_level: str, explanation_type: str) -> str:
        """Build the prompt for LLaMA 3.1 70B based on analysis data."""
        
        symbol = analysis_data.get('symbol', 'UNKNOWN')
        score = analysis_data.get('score_0_10', 0)
        indicators = analysis_data.get('components', {})
        weighted_scores = analysis_data.get('weighted_scores', {})
        
        # Optimized shorter base prompt
        base_prompt = f"""Financial Analysis for {symbol}: {score}/10

Key Indicators:
"""
        
        # Add top 3 most significant indicators only
        if weighted_scores:
            sorted_indicators = sorted(weighted_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            for indicator_name, weighted_score in sorted_indicators:
                indicator_value = indicators.get(indicator_name, 'N/A')
                base_prompt += f"- {indicator_name}: {weighted_score:.2f}\n"
        
        # Simplified, focused instructions
        if detail_level == 'summary':
            instruction = f"\n\nExplain in 1-2 sentences: Is {symbol} a BUY/HOLD/SELL and why?"
            
        elif detail_level == 'detailed':
            instruction = f"\n\nProvide investment recommendation for {symbol} with:\n1. BUY/HOLD/SELL decision\n2. Key supporting indicators\n3. Main risks\n4. Confidence level"
            
        else:  # standard
            instruction = f"\n\nAnalyze {symbol}: recommendation, key indicators, and confidence level."
        
        return base_prompt + instruction + "\n\nAnalysis:"
    
    def _get_max_tokens(self, detail_level: str) -> int:
        """Get maximum tokens based on detail level (optimized for performance)."""
        token_limits = {
            'summary': 75,    # Reduced from 150
            'standard': 150,  # Reduced from 300
            'detailed': 250   # Reduced from 500
        }
        return token_limits.get(detail_level, 150)
    
    def _get_generation_options(self, detail_level: str, model_name: str) -> dict:
        """Get optimized generation options based on detail level and model."""
        base_options = {
            'temperature': 0.5,  # Faster sampling than 0.3
            'top_p': 0.85,       # Reduced for faster generation
            'num_predict': self._get_max_tokens(detail_level),
            'stop': ['###', '---', 'END_EXPLANATION', 'END'],
            'repeat_penalty': 1.1
        }
        
        # Optimize context window based on model
        if '8b' in model_name.lower():
            # Smaller model can handle smaller context more efficiently
            base_options['num_ctx'] = 1024
            base_options['temperature'] = 0.6  # Slightly higher for creativity
        else:
            # Larger model with larger context
            base_options['num_ctx'] = 2048
            base_options['temperature'] = 0.4  # Lower for consistency
        
        # Adjust for detail level
        if detail_level == 'summary':
            base_options['temperature'] = 0.7  # Higher for concise generation
            base_options['top_p'] = 0.8
        elif detail_level == 'detailed':
            base_options['temperature'] = 0.3  # Lower for comprehensive analysis
            base_options['top_p'] = 0.9
        
        return base_options
    
    def _calculate_confidence_score(self, content: str) -> float:
        """Calculate confidence score based on content quality (optimized)."""
        try:
            word_count = len(content.split())
            
            # Start with base confidence
            confidence = 0.6  # Increased base confidence
            
            # Length scoring (adjusted for shorter responses)
            if 10 <= word_count <= 300:  # Broader acceptable range
                confidence += 0.25
            
            # Check for key financial terms
            financial_terms = ['buy', 'sell', 'hold', 'score', 'trend', 'indicator', 'risk']
            term_count = sum(1 for term in financial_terms if term.lower() in content.lower())
            confidence += min(0.15, term_count * 0.05)
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.6  # Higher default confidence
    
    def _create_cache_key(self, analysis_data: Dict[str, Any], detail_level: str, explanation_type: str) -> str:
        """Create a cache key for explanation results."""
        # Use symbol, score, and key indicator values for cache key
        symbol = analysis_data.get('symbol', 'UNKNOWN')
        score = analysis_data.get('score_0_10', 0)
        
        # Create a simple hash of the analysis data
        data_str = f"{symbol}_{score}_{detail_level}_{explanation_type}"
        
        # Add some key indicators to make cache more specific
        indicators = analysis_data.get('weighted_scores', {})
        if indicators:
            key_indicators = ['w_sma50vs200', 'w_rsi14', 'w_macd12269']
            for key in key_indicators:
                if key in indicators:
                    data_str += f"_{key}_{indicators[key]}"
        
        return f"llm_explanation_{hash(data_str)}"
    
    def generate_batch_explanations(self, 
                                  analysis_results: List[Dict[str, Any]], 
                                  detail_level: str = 'standard') -> List[Optional[Dict[str, Any]]]:
        """
        Generate explanations for multiple analysis results.
        
        Args:
            analysis_results: List of analysis data dictionaries
            detail_level: Detail level for all explanations
            
        Returns:
            List of explanation results (same order as input)
        """
        explanations = []
        
        for analysis_data in analysis_results:
            explanation = self.generate_explanation(analysis_data, detail_level)
            explanations.append(explanation)
        
        return explanations
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and model information."""
        try:
            if not self.client:
                return {
                    'available': False,
                    'error': 'Client not initialized'
                }
            
            models = self.client.list()
            model_available = self._verify_model_availability()
            
            primary_available = self._verify_model_availability(self.primary_model)
            detailed_available = self._verify_model_availability(self.detailed_model)
            
            return {
                'available': primary_available or detailed_available,
                'primary_model': self.primary_model,
                'detailed_model': self.detailed_model,
                'primary_model_available': primary_available,
                'detailed_model_available': detailed_available,
                'current_model': self.current_model,
                'models_count': len(models.get('models', [])),
                'client_initialized': True,
                'cache_enabled': hasattr(cache, 'get'),
                'performance_mode': self.performance_mode,
                'generation_timeout': self.generation_timeout
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'client_initialized': self.client is not None
            }


# Singleton instance
_llm_service = None

def get_local_llm_service() -> LocalLLMService:
    """Get singleton instance of LocalLLMService."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LocalLLMService()
    return _llm_service