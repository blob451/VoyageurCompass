"""
Circuit breaker implementation for multilingual LLM services.

Provides automatic failure detection, fast-fail mechanisms, and self-healing
capabilities to protect the system from cascading failures.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Fast-fail mode
    HALF_OPEN = "half_open"  # Testing mode


class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    def __init__(self, name: str):
        self.name = name
        self.failure_threshold = getattr(settings, f'{name.upper()}_FAILURE_THRESHOLD', 5)
        self.recovery_timeout = getattr(settings, f'{name.upper()}_RECOVERY_TIMEOUT', 60)  # seconds
        self.success_threshold = getattr(settings, f'{name.upper()}_SUCCESS_THRESHOLD', 2)
        self.timeout = getattr(settings, f'{name.upper()}_TIMEOUT', 30)  # seconds
        self.half_open_max_calls = getattr(settings, f'{name.upper()}_HALF_OPEN_MAX_CALLS', 3)


class MultilingualCircuitBreaker:
    """
    Circuit breaker for multilingual operations with per-language state management.
    """

    def __init__(self, name: str = "multilingual"):
        """Initialize circuit breaker."""
        self.name = name
        self.config = CircuitBreakerConfig(name)

        # Per-language state management
        self.states: Dict[str, CircuitState] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, datetime] = {}
        self.success_counts: Dict[str, int] = {}
        self.half_open_call_counts: Dict[str, int] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Supported languages
        self.supported_languages = ['fr', 'es', 'en']

        # Initialize states
        for lang in self.supported_languages:
            self.states[lang] = CircuitState.CLOSED
            self.failure_counts[lang] = 0
            self.success_counts[lang] = 0
            self.half_open_call_counts[lang] = 0

        # Cache configuration
        self.cache_prefix = f"circuit_breaker:{self.name}:"
        self.cache_ttl = 300  # 5 minutes

        logger.info(f"Circuit breaker '{name}' initialized for languages: {self.supported_languages}")

    def call_with_breaker(
        self,
        func: Callable,
        language: str,
        fallback_func: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            language: Language code for the operation
            fallback_func: Optional fallback function
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from function or fallback

        Raises:
            CircuitBreakerOpenError: When circuit is open and no fallback
        """
        with self._lock:
            # Ensure language is tracked
            if language not in self.states:
                self._initialize_language(language)

            current_state = self._get_current_state(language)

            # Handle open circuit
            if current_state == CircuitState.OPEN:
                if fallback_func:
                    logger.warning(f"Circuit breaker OPEN for {language}, using fallback")
                    return self._execute_fallback(fallback_func, language, *args, **kwargs)
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker OPEN for language: {language}")

            # Handle half-open circuit
            if current_state == CircuitState.HALF_OPEN:
                if self.half_open_call_counts[language] >= self.config.half_open_max_calls:
                    logger.warning(f"Half-open limit reached for {language}, using fallback")
                    if fallback_func:
                        return self._execute_fallback(fallback_func, language, *args, **kwargs)
                    else:
                        raise CircuitBreakerOpenError(f"Half-open limit exceeded for language: {language}")

                self.half_open_call_counts[language] += 1

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success(language)
            return result

        except Exception as e:
            self._on_failure(language, e)

            # Check if circuit should open
            with self._lock:
                if self.states[language] == CircuitState.OPEN and fallback_func:
                    logger.warning(f"Circuit opened for {language} after failure, using fallback")
                    return self._execute_fallback(fallback_func, language, *args, **kwargs)

            # Re-raise if no fallback
            raise

    def _execute_fallback(self, fallback_func: Callable, language: str, *args, **kwargs) -> Any:
        """Execute fallback function with error handling."""
        try:
            result = fallback_func(*args, **kwargs)
            # Mark the fallback execution
            if isinstance(result, dict):
                result['circuit_breaker_triggered'] = True
                result['fallback_language'] = True
                result['original_language'] = language
            return result
        except Exception as e:
            logger.error(f"Fallback function failed for {language}: {str(e)}")
            raise

    def _get_current_state(self, language: str) -> CircuitState:
        """Get current state for language, considering recovery timeout."""
        current_state = self.states[language]

        # Check if open circuit should transition to half-open
        if current_state == CircuitState.OPEN:
            last_failure = self.last_failure_times.get(language)
            if last_failure and datetime.now() - last_failure >= timedelta(seconds=self.config.recovery_timeout):
                self._transition_to_half_open(language)
                return CircuitState.HALF_OPEN

        return current_state

    def _on_success(self, language: str) -> None:
        """Handle successful execution."""
        with self._lock:
            self.success_counts[language] += 1
            current_state = self.states[language]

            if current_state == CircuitState.HALF_OPEN:
                if self.success_counts[language] >= self.config.success_threshold:
                    self._transition_to_closed(language)
                    logger.info(f"Circuit breaker for {language} recovered (HALF_OPEN -> CLOSED)")

            # Update cache
            self._update_cache(language)

    def _on_failure(self, language: str, exception: Exception) -> None:
        """Handle failed execution."""
        with self._lock:
            self.failure_counts[language] += 1
            self.last_failure_times[language] = datetime.now()

            logger.warning(f"Circuit breaker failure for {language}: {str(exception)}")

            current_state = self.states[language]

            # Transition to open if threshold exceeded
            if current_state == CircuitState.CLOSED:
                if self.failure_counts[language] >= self.config.failure_threshold:
                    self._transition_to_open(language)
                    logger.error(f"Circuit breaker OPENED for {language} after {self.failure_counts[language]} failures")

            elif current_state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open(language)
                logger.warning(f"Circuit breaker returned to OPEN for {language} (failure in half-open)")

            # Update cache
            self._update_cache(language)

    def _transition_to_open(self, language: str) -> None:
        """Transition circuit to open state."""
        self.states[language] = CircuitState.OPEN
        self.success_counts[language] = 0
        self.half_open_call_counts[language] = 0

    def _transition_to_half_open(self, language: str) -> None:
        """Transition circuit to half-open state."""
        self.states[language] = CircuitState.HALF_OPEN
        self.success_counts[language] = 0
        self.half_open_call_counts[language] = 0

    def _transition_to_closed(self, language: str) -> None:
        """Transition circuit to closed state."""
        self.states[language] = CircuitState.CLOSED
        self.failure_counts[language] = 0
        self.success_counts[language] = 0
        self.half_open_call_counts[language] = 0

    def _initialize_language(self, language: str) -> None:
        """Initialize tracking for new language."""
        self.states[language] = CircuitState.CLOSED
        self.failure_counts[language] = 0
        self.success_counts[language] = 0
        self.half_open_call_counts[language] = 0
        if language not in self.supported_languages:
            self.supported_languages.append(language)

    def _update_cache(self, language: str) -> None:
        """Update circuit state in cache for monitoring."""
        try:
            cache_key = f"{self.cache_prefix}{language}"
            state_data = {
                'state': self.states[language].value,
                'failure_count': self.failure_counts[language],
                'success_count': self.success_counts[language],
                'last_failure': self.last_failure_times.get(language).isoformat() if language in self.last_failure_times else None,
                'updated_at': datetime.now().isoformat()
            }
            cache.set(cache_key, state_data, self.cache_ttl)
        except Exception as e:
            logger.warning(f"Failed to update cache for circuit breaker {language}: {str(e)}")

    def get_state(self, language: str) -> CircuitState:
        """Get current state for language."""
        with self._lock:
            if language not in self.states:
                self._initialize_language(language)
            return self._get_current_state(language)

    def force_open(self, language: str, reason: str = "Manual override") -> None:
        """Manually open circuit for language."""
        with self._lock:
            if language not in self.states:
                self._initialize_language(language)

            self._transition_to_open(language)
            self.last_failure_times[language] = datetime.now()
            self._update_cache(language)

            logger.warning(f"Circuit breaker manually opened for {language}: {reason}")

    def force_close(self, language: str, reason: str = "Manual override") -> None:
        """Manually close circuit for language."""
        with self._lock:
            if language not in self.states:
                self._initialize_language(language)

            self._transition_to_closed(language)
            self._update_cache(language)

            logger.info(f"Circuit breaker manually closed for {language}: {reason}")

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for language in self.supported_languages:
                self._transition_to_closed(language)
                self._update_cache(language)

            logger.info(f"All circuit breakers reset for {self.name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all languages."""
        with self._lock:
            stats = {
                'name': self.name,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout,
                    'half_open_max_calls': self.config.half_open_max_calls
                },
                'languages': {}
            }

            for language in self.supported_languages:
                current_state = self._get_current_state(language)
                last_failure = self.last_failure_times.get(language)

                stats['languages'][language] = {
                    'state': current_state.value,
                    'failure_count': self.failure_counts[language],
                    'success_count': self.success_counts[language],
                    'half_open_calls': self.half_open_call_counts[language],
                    'last_failure': last_failure.isoformat() if last_failure else None,
                    'time_since_last_failure': (
                        (datetime.now() - last_failure).total_seconds()
                        if last_failure else None
                    ),
                    'can_recover': (
                        current_state == CircuitState.OPEN and
                        last_failure and
                        datetime.now() - last_failure >= timedelta(seconds=self.config.recovery_timeout)
                    )
                }

            return stats

    def is_healthy(self, language: str = None) -> bool:
        """Check if circuit breaker is healthy (not open)."""
        if language:
            return self.get_state(language) != CircuitState.OPEN
        else:
            # Check all languages
            return all(
                self.get_state(lang) != CircuitState.OPEN
                for lang in self.supported_languages
            )


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker instances
_circuit_breakers: Dict[str, MultilingualCircuitBreaker] = {}
_breaker_lock = threading.Lock()


def get_circuit_breaker(name: str = "multilingual") -> MultilingualCircuitBreaker:
    """Get or create circuit breaker instance."""
    global _circuit_breakers

    if name not in _circuit_breakers:
        with _breaker_lock:
            if name not in _circuit_breakers:
                _circuit_breakers[name] = MultilingualCircuitBreaker(name)

    return _circuit_breakers[name]


def call_with_circuit_breaker(
    func: Callable,
    language: str,
    fallback_func: Optional[Callable] = None,
    breaker_name: str = "multilingual",
    *args,
    **kwargs
) -> Any:
    """
    Convenience function to call function with circuit breaker protection.

    Args:
        func: Function to execute
        language: Language code
        fallback_func: Fallback function (optional)
        breaker_name: Circuit breaker name
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result or fallback result
    """
    breaker = get_circuit_breaker(breaker_name)
    return breaker.call_with_breaker(func, language, fallback_func, *args, **kwargs)


def get_all_circuit_breaker_stats() -> Dict[str, Any]:
    """Get stats for all circuit breakers."""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'circuit_breakers': {}
    }

    for name, breaker in _circuit_breakers.items():
        stats['circuit_breakers'][name] = breaker.get_stats()

    return stats