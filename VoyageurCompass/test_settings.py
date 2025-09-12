"""
Test-specific Django settings configuration for VoyageurCompass project.
Conditionally utilises PostgreSQL in CI environment with SQLite optimisation for local testing performance.
"""

from .settings import *  # noqa: F401,F403
import os

# CI environment detection and database configuration
IS_CI_ENVIRONMENT = os.getenv("CI", "").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL", "")

if IS_CI_ENVIRONMENT and DATABASE_URL:
    # PostgreSQL configuration for comprehensive CI testing environment
    import dj_database_url

    DATABASES = {"default": dj_database_url.parse(DATABASE_URL)}
    # Database migrations enabled for comprehensive CI table creation
    MIGRATION_MODULES = {}

    # Ensure proper database configuration for CI environment
    db_config = DATABASES["default"]
    if not db_config.get("USER"):
        # Set default user if not provided in DATABASE_URL
        db_config["USER"] = os.getenv("POSTGRES_USER", "voyageur_user")
    if not db_config.get("PASSWORD"):
        db_config["PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "test-ci-password")
    if not db_config.get("HOST"):
        db_config["HOST"] = os.getenv("POSTGRES_HOST", "localhost")
    if not db_config.get("PORT"):
        db_config["PORT"] = os.getenv("POSTGRES_PORT", "5432")
else:
    # SQLite configuration for optimised local testing performance
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": ":memory:",
            "TEST": {
                "NAME": ":memory:",
            },
        }
    }
    # Migration system disabled for enhanced local testing performance

    class DisableMigrations:
        def __contains__(self, item):
            return True

        def __getitem__(self, item):
            return None

    MIGRATION_MODULES = DisableMigrations()

# Disable logging during tests
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "null": {
            "class": "logging.NullHandler",
        },
    },
    "root": {
        "handlers": ["null"],
    },
}

# Disable cache during tests
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
}

# Disable Celery during tests
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Password validation disabled for faster test user creation
AUTH_PASSWORD_VALIDATORS = []

# Use faster password hasher for tests
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]

# Disable debug mode
DEBUG = False

# Test-specific secret key
SECRET_KEY = "test-secret-key-for-testing-only"

# Disable SSL redirect for tests - prevents 301 redirects in authentication tests
SECURE_SSL_REDIRECT = False
