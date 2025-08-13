"""
Performance Test Settings for VoyageurCompass
These settings are used for performance testing in CI/CD
"""

from .test_settings import *  # noqa: F403

# Re-enable migrations for performance testing
# Performance tests need actual database structure
MIGRATION_MODULES = {}

# Use a real cache backend for performance testing
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "performance-test-cache",
    }
}

# Use SQLite for performance testing
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",  # In-memory database for speed
        "OPTIONS": {
            "timeout": 20,
        },
    }
}

# Performance test specific settings
PERFORMANCE_TEST_MODE = True

# Enable minimal logging for performance metrics
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "performance": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}