"""
Robust logging configuration module for VoyageurCompass.
Provides enhanced error handling and fallback mechanisms for logging infrastructure.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any


class RobustRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced RotatingFileHandler with robust error handling.
    Falls back to console logging if file operations fail.
    """
    
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, 
                 encoding=None, delay=False):
        """Initialise handler with error handling."""
        self.fallback_handler = None
        try:
            # Ensure directory exists
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        except (OSError, IOError, PermissionError) as e:
            # If file handler fails, create fallback console handler
            self.fallback_handler = logging.StreamHandler(sys.stderr)
            self.fallback_handler.setFormatter(
                logging.Formatter('FALLBACK-LOG: %(levelname)s %(asctime)s %(message)s')
            )
            print(f"Warning: Could not initialise log file {filename}: {e}. Using console fallback.", 
                  file=sys.stderr)
    
    def emit(self, record):
        """Emit log record with fallback to console if file operations fail."""
        if self.fallback_handler:
            return self.fallback_handler.emit(record)
        
        try:
            super().emit(record)
        except (OSError, IOError, PermissionError) as e:
            # Create fallback handler if file operations fail
            if not self.fallback_handler:
                self.fallback_handler = logging.StreamHandler(sys.stderr)
                self.fallback_handler.setFormatter(
                    logging.Formatter('FALLBACK-LOG: %(levelname)s %(asctime)s %(message)s')
                )
                print(f"Warning: Log file operation failed: {e}. Switching to console fallback.", 
                      file=sys.stderr)
            
            self.fallback_handler.emit(record)


def create_robust_logging_config(base_dir: Path, log_level: str = "INFO") -> Dict[str, Any]:
    """
    Create robust logging configuration with error handling.
    
    Args:
        base_dir: Base directory for the application
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Dictionary containing logging configuration
    """
    
    # Define log directories
    logs_base_dir = base_dir / "Temp" / "logs"
    log_dirs = {
        "web_analysis": logs_base_dir / "web_analysis" / "current",
        "web_analysis_archived": logs_base_dir / "web_analysis" / "archived",
        "model_training_universal": logs_base_dir / "model_training" / "universal_lstm",
        "model_training_individual": logs_base_dir / "model_training" / "individual_lstm",
        "model_training_sentiment": logs_base_dir / "model_training" / "sentiment",
        "data_collection_stock": logs_base_dir / "data_collection" / "stock_data",
        "data_collection_sector": logs_base_dir / "data_collection" / "sector_data",
        "data_collection_errors": logs_base_dir / "data_collection" / "errors",
        "system_django": logs_base_dir / "system" / "django",
        "system_celery": logs_base_dir / "system" / "celery",
        "system_api": logs_base_dir / "system" / "api",
        "analytics_technical": logs_base_dir / "analytics" / "technical",
        "analytics_sentiment": logs_base_dir / "analytics" / "sentiment",
        "analytics_portfolio": logs_base_dir / "analytics" / "portfolio",
        "security_auth": logs_base_dir / "security" / "auth",
        "security_failed": logs_base_dir / "security" / "failed_attempts",
        "security_api": logs_base_dir / "security" / "api_security",
    }
    
    # Create all log directories with error handling
    for log_dir in log_dirs.values():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr)
    
    # Build logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
                "style": "{",
            },
            "simple": {
                "format": "{levelname} {asctime} {message}",
                "style": "{",
            },
            "security": {
                "format": "{levelname} {asctime} {module} {funcName} {lineno} {message}",
                "style": "{",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "verbose",
            },
        },
        "loggers": {},
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }
    
    # Add file handlers with robust error handling
    file_handlers = [
        ("django_file", log_dirs["system_django"] / "django.log", "verbose", 10485760, 5),
        ("celery_file", log_dirs["system_celery"] / "celery.log", "verbose", 10485760, 5),
        ("api_file", log_dirs["system_api"] / "api.log", "verbose", 10485760, 5),
        ("data_collection_file", log_dirs["data_collection_stock"] / "stock_data.log", "verbose", 10485760, 5),
        ("data_errors_file", log_dirs["data_collection_errors"] / "errors.log", "verbose", 10485760, 5),
        ("analytics_file", log_dirs["analytics_technical"] / "technical_analysis.log", "verbose", 10485760, 5),
        ("sentiment_file", log_dirs["analytics_sentiment"] / "sentiment_analysis.log", "verbose", 10485760, 5),
        ("security_file", log_dirs["security_auth"] / "auth.log", "security", 10485760, 10),
        ("security_failed_file", log_dirs["security_failed"] / "failed_attempts.log", "security", 10485760, 10),
        ("model_training_file", log_dirs["model_training_universal"] / "universal_lstm.log", "verbose", 52428800, 3),
    ]
    
    for handler_name, log_file, formatter, max_bytes, backup_count in file_handlers:
        try:
            config["handlers"][handler_name] = {
                "()": RobustRotatingFileHandler,
                "filename": str(log_file),
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "formatter": formatter,
            }
        except Exception as e:
            print(f"Warning: Could not configure handler {handler_name}: {e}", file=sys.stderr)
            # Skip this handler if it fails
            continue
    
    # Configure loggers with available handlers
    available_handlers = list(config["handlers"].keys())
    
    logger_configs = [
        ("django", ["console", "django_file"], log_level),
        ("django.server", ["console", "api_file"], "INFO"),
        ("django.request", ["console", "api_file"], "WARNING"),
        ("django.security", ["console", "security_file"], "WARNING"),
        ("celery", ["console", "celery_file"], log_level),
        ("celery.task", ["console", "celery_file"], log_level),
        ("Data", ["console", "data_collection_file"], log_level),
        ("Analytics", ["console", "analytics_file"], log_level),
        ("Analytics.engine.sentiment", ["console", "sentiment_file"], log_level),
        ("Analytics.ml", ["console", "model_training_file"], log_level),
        ("Core.auth", ["console", "security_file"], "INFO"),
        ("Core.failed_attempts", ["console", "security_failed_file"], "WARNING"),
        ("data_collection_errors", ["console", "data_errors_file"], "ERROR"),
    ]
    
    for logger_name, handlers, level in logger_configs:
        # Filter handlers to only include those that were successfully created
        valid_handlers = [h for h in handlers if h in available_handlers]
        if not valid_handlers:
            valid_handlers = ["console"]  # Fallback to console if no file handlers available
        
        config["loggers"][logger_name] = {
            "handlers": valid_handlers,
            "level": level,
            "propagate": False,
        }
    
    return config


def validate_logging_setup():
    """
    Validate logging setup and report any issues.
    Returns True if logging is working correctly, False otherwise.
    """
    try:
        # Test basic logging functionality
        logger = logging.getLogger("Core.logging_config")
        logger.info("Logging validation test - infrastructure operational")
        
        # Test each configured logger
        test_loggers = [
            "django", "celery", "Data", "Analytics", 
            "Core.auth", "data_collection_errors"
        ]
        
        for logger_name in test_loggers:
            test_logger = logging.getLogger(logger_name)
            test_logger.debug(f"Validation test for {logger_name} logger")
        
        return True
        
    except Exception as e:
        print(f"Logging validation failed: {e}", file=sys.stderr)
        return False


# Module-level validation on import
if __name__ != "__main__":
    try:
        # Perform basic validation when module is imported
        validate_logging_setup()
    except Exception:
        # Suppress errors during import to prevent application startup issues
        pass