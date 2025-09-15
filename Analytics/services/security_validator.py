"""
Security validation utilities for LLM services.
Provides input sanitization, prompt injection protection, and content filtering.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PromptInjectionDetector:
    """Detects and prevents prompt injection attacks in user input."""

    def __init__(self):
        # Dangerous patterns that could indicate prompt injection
        self.injection_patterns = [
            # Direct system commands
            r'(?i)\bsystem\s*:',
            r'(?i)\bassistant\s*:',
            r'(?i)\buser\s*:',
            r'(?i)\bhuman\s*:',
            r'(?i)\bai\s*:',

            # Instruction manipulation
            r'(?i)ignore\s+(?:previous|all|prior|above)',
            r'(?i)forget\s+(?:previous|all|prior|above)',
            r'(?i)disregard\s+(?:previous|all|prior|above)',
            r'(?i)override\s+(?:previous|all|prior|above)',

            # Role manipulation
            r'(?i)you\s+are\s+now',
            r'(?i)act\s+as\s+(?:if|a|an)',
            r'(?i)pretend\s+(?:to\s+be|you\s+are)',
            r'(?i)simulate\s+(?:being|a|an)',

            # System prompts
            r'(?i)<\|system\|>',
            r'(?i)<\|assistant\|>',
            r'(?i)<\|user\|>',
            r'(?i)<\|im_start\|>',
            r'(?i)<\|im_end\|>',

            # Code injection attempts
            r'```\s*(?:python|javascript|sql|bash|shell)',
            r'(?i)exec\s*\(',
            r'(?i)eval\s*\(',
            r'(?i)import\s+os',
            r'(?i)subprocess\.',

            # Data extraction attempts
            r'(?i)reveal\s+(?:your|the)\s+(?:prompt|instructions|system)',
            r'(?i)show\s+(?:your|the)\s+(?:prompt|instructions|system)',
            r'(?i)what\s+(?:are\s+)?your\s+(?:instructions|rules)',

            # Financial injection specific
            r'(?i)transfer\s+\$',
            r'(?i)wire\s+(?:money|funds)',
            r'(?i)execute\s+(?:trade|order|transaction)',
        ]

        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.injection_patterns]

        # Suspicious character sequences
        self.suspicious_chars = ['<|', '|>', '```', '###', '---']

        # Maximum allowed length for user input
        self.max_input_length = 5000

    def detect_injection(self, text: str) -> Dict[str, any]:
        """
        Detect potential prompt injection in text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with detection results
        """
        if not text:
            return {"is_safe": True, "threats": [], "sanitized_text": ""}

        threats = []
        risk_score = 0.0

        # Check length
        if len(text) > self.max_input_length:
            threats.append(f"Input too long: {len(text)} > {self.max_input_length}")
            risk_score += 0.3

        # Check for injection patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                match = pattern.search(text)
                threats.append(f"Injection pattern detected: {match.group()}")
                risk_score += 0.4

        # Check for suspicious character sequences
        for suspicious in self.suspicious_chars:
            if suspicious in text:
                threats.append(f"Suspicious characters: {suspicious}")
                risk_score += 0.2

        # Check for excessive repetition (potential pattern flooding)
        if self._detect_pattern_flooding(text):
            threats.append("Pattern flooding detected")
            risk_score += 0.3

        # Check for encoding attempts
        if self._detect_encoding_evasion(text):
            threats.append("Encoding evasion detected")
            risk_score += 0.4

        is_safe = risk_score < 0.5 and len(threats) == 0

        return {
            "is_safe": is_safe,
            "threats": threats,
            "risk_score": min(risk_score, 1.0),
            "sanitized_text": self._sanitize_text(text) if not is_safe else text
        }

    def _detect_pattern_flooding(self, text: str) -> bool:
        """Detect repetitive patterns that might be used to overwhelm the system."""
        # Check for excessive repetition of characters
        for char in set(text):
            if text.count(char) > len(text) * 0.3:  # More than 30% of text is same character
                return True

        # Check for repeated words
        words = text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                return True

        return False

    def _detect_encoding_evasion(self, text: str) -> bool:
        """Detect attempts to evade detection through encoding."""
        # Check for unicode normalization evasion
        try:
            import unicodedata
            normalized = unicodedata.normalize('NFKD', text)
            if normalized != text and any(pattern.search(normalized) for pattern in self.compiled_patterns):
                return True
        except:
            pass

        # Check for base64-like patterns
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        if re.search(base64_pattern, text):
            return True

        # Check for hex encoding
        hex_pattern = r'(?:0x)?[0-9a-fA-F]{16,}'
        if re.search(hex_pattern, text):
            return True

        return False

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing dangerous patterns."""
        sanitized = text[:self.max_input_length]

        # Remove dangerous patterns
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('[FILTERED]', sanitized)

        # Remove suspicious character sequences
        for suspicious in self.suspicious_chars:
            sanitized = sanitized.replace(suspicious, '[FILTERED]')

        # Clean up multiple filtered markers
        sanitized = re.sub(r'\[FILTERED\]\s*\[FILTERED\]', '[FILTERED]', sanitized)

        return sanitized.strip()


class ContentFilter:
    """Filters potentially harmful or inappropriate content in LLM outputs."""

    def __init__(self):
        # Financial advice disclaimers that should be present
        self.required_disclaimers = [
            "not financial advice",
            "consult a financial advisor",
            "past performance",
            "do your own research",
        ]

        # Prohibited content patterns
        self.prohibited_patterns = [
            # Investment guarantees
            r'(?i)guaranteed?\s+(?:profit|return|gain)',
            r'(?i)risk-free\s+(?:investment|return)',
            r'(?i)certain\s+(?:profit|gain|success)',

            # Market manipulation
            r'(?i)pump\s+and\s+dump',
            r'(?i)insider\s+(?:information|trading)',
            r'(?i)market\s+manipulation',

            # Personal financial advice
            r'(?i)you\s+should\s+(?:buy|sell|invest)',
            r'(?i)definitely\s+(?:buy|sell|invest)',
            r'(?i)must\s+(?:buy|sell|invest)',

            # Sensitive information
            r'(?i)social\s+security\s+number',
            r'(?i)credit\s+card\s+number',
            r'(?i)bank\s+account\s+number',
        ]

        self.compiled_prohibited = [re.compile(pattern) for pattern in self.prohibited_patterns]

    def validate_content(self, content: str, content_type: str = "explanation") -> Dict[str, any]:
        """
        Validate content for appropriateness and compliance.

        Args:
            content: Content to validate
            content_type: Type of content (explanation, translation, etc.)

        Returns:
            Dictionary with validation results
        """
        if not content:
            return {"is_valid": False, "issues": ["Empty content"], "filtered_content": ""}

        issues = []
        risk_score = 0.0

        # Check for prohibited content
        for pattern in self.compiled_prohibited:
            if pattern.search(content):
                match = pattern.search(content)
                issues.append(f"Prohibited content detected: {match.group()}")
                risk_score += 0.5

        # Check for appropriate disclaimers in financial content
        if content_type == "explanation":
            has_disclaimer = any(disclaimer in content.lower() for disclaimer in self.required_disclaimers)
            if not has_disclaimer and len(content) > 200:  # Only require for substantial content
                issues.append("Missing financial disclaimer")
                risk_score += 0.2

        # Check for excessive confidence in predictions
        confidence_patterns = [
            r'(?i)will\s+(?:definitely|certainly|absolutely)',
            r'(?i)guaranteed?\s+to\s+(?:rise|fall|increase|decrease)',
            r'(?i)100%\s+(?:certain|sure|confident)',
        ]

        for pattern in confidence_patterns:
            if re.search(pattern, content):
                issues.append("Excessive confidence in predictions")
                risk_score += 0.3
                break

        is_valid = risk_score < 0.5 and len(issues) == 0

        return {
            "is_valid": is_valid,
            "issues": issues,
            "risk_score": min(risk_score, 1.0),
            "filtered_content": self._filter_content(content) if not is_valid else content
        }

    def _filter_content(self, content: str) -> str:
        """Filter out problematic content."""
        filtered = content

        # Replace prohibited patterns
        for pattern in self.compiled_prohibited:
            filtered = pattern.sub('[CONTENT FILTERED]', filtered)

        # Add disclaimer if missing
        if len(filtered) > 200 and not any(disclaimer in filtered.lower() for disclaimer in self.required_disclaimers):
            filtered += "\n\nDisclaimer: This analysis is for informational purposes only and is not financial advice. Please consult with a qualified financial advisor before making investment decisions."

        return filtered


class SecurityValidator:
    """Main security validation service for LLM inputs and outputs."""

    def __init__(self):
        self.injection_detector = PromptInjectionDetector()
        self.content_filter = ContentFilter()

        # Security metrics
        self.security_metrics = {
            "inputs_validated": 0,
            "threats_detected": 0,
            "content_filtered": 0,
            "injections_blocked": 0,
        }

    def validate_input(self, text: str, input_type: str = "user_input") -> Dict[str, any]:
        """
        Validate user input for security threats.

        Args:
            text: Input text to validate
            input_type: Type of input (user_input, symbol, etc.)

        Returns:
            Dictionary with validation results and sanitized text
        """
        self.security_metrics["inputs_validated"] += 1

        # Basic input validation
        if not isinstance(text, str):
            return {
                "is_safe": False,
                "threats": ["Invalid input type"],
                "sanitized_text": str(text)[:1000] if text else ""
            }

        # Detect injection attempts
        injection_result = self.injection_detector.detect_injection(text)

        if not injection_result["is_safe"]:
            self.security_metrics["threats_detected"] += 1
            self.security_metrics["injections_blocked"] += 1

            logger.warning(
                f"Security threat detected in {input_type}: "
                f"Threats: {injection_result['threats']}, "
                f"Risk Score: {injection_result['risk_score']:.2f}"
            )

        return injection_result

    def validate_output(self, content: str, content_type: str = "explanation") -> Dict[str, any]:
        """
        Validate LLM output for appropriateness and compliance.

        Args:
            content: Output content to validate
            content_type: Type of content

        Returns:
            Dictionary with validation results
        """
        content_result = self.content_filter.validate_content(content, content_type)

        if not content_result["is_valid"]:
            self.security_metrics["content_filtered"] += 1

            logger.warning(
                f"Content validation failed for {content_type}: "
                f"Issues: {content_result['issues']}, "
                f"Risk Score: {content_result['risk_score']:.2f}"
            )

        return content_result

    def get_security_metrics(self) -> Dict[str, any]:
        """Get security validation metrics."""
        metrics = self.security_metrics.copy()

        # Calculate rates
        if metrics["inputs_validated"] > 0:
            metrics["threat_detection_rate"] = metrics["threats_detected"] / metrics["inputs_validated"]
            metrics["injection_block_rate"] = metrics["injections_blocked"] / metrics["inputs_validated"]
        else:
            metrics["threat_detection_rate"] = 0.0
            metrics["injection_block_rate"] = 0.0

        return metrics


# Singleton instance
_security_validator = None


def get_security_validator() -> SecurityValidator:
    """Get singleton instance of SecurityValidator."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator


def sanitize_financial_input(text: str) -> str:
    """
    Quick sanitization function for financial input.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text safe for LLM processing
    """
    validator = get_security_validator()
    result = validator.validate_input(text, "financial_input")
    return result["sanitized_text"]


def validate_financial_output(content: str) -> Tuple[bool, str]:
    """
    Quick validation function for financial content output.

    Args:
        content: Output content to validate

    Returns:
        Tuple of (is_valid, filtered_content)
    """
    validator = get_security_validator()
    result = validator.validate_output(content, "explanation")
    return result["is_valid"], result["filtered_content"]