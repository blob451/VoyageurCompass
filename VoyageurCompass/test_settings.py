"""
Optimized Django settings for test performance.
"""

from .settings import *

# Test Database Configuration - Ultra-fast SQLite
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
        "OPTIONS": {
            "timeout": 20,
            "check_same_thread": False,
        },
        "TEST": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": ":memory:",
        },
    }
}

# Disable migrations for faster test setup
class DisableMigrations:
    def __contains__(self, item):
        return True
    
    def __getitem__(self, item):
        return None

MIGRATION_MODULES = DisableMigrations()

# Disable cache for tests to prevent Redis dependency
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
}

# Disable middleware that slows down tests
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware", 
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

# Faster password hasher for tests
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.MD5PasswordHasher",
]

# Disable logging for tests
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

# Disable static file collection and compression
STATICFILES_STORAGE = "django.contrib.staticfiles.storage.StaticFilesStorage"
WHITENOISE_COMPRESS_OFFLINE = False

# Disable Celery for tests
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True

# Test-specific optimizations
DEBUG = False
USE_TZ = False  # Faster datetime handling in tests
SECRET_KEY = "test-secret-key-not-for-production"

# Disable template caching
TEMPLATES[0]["OPTIONS"]["loaders"] = [
    "django.template.loaders.filesystem.Loader",
    "django.template.loaders.app_directories.Loader",
]

# Keep essential apps but remove heavy ones
INSTALLED_APPS = [
    "django.contrib.admin",  # Keep for URL compatibility
    "django.contrib.auth",
    "django.contrib.contenttypes", 
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Keep essential third-party apps
    "rest_framework",
    "rest_framework_simplejwt",
    # Our custom apps
    "Analytics",
    "Core",
    "Data",
]