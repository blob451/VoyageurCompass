"""
Django settings configuration for VoyageurCompass financial analytics platform.
Implements environment-based configuration with security enforcement and optimised performance settings.

For comprehensive documentation on Django settings:
https://docs.djangoproject.com/en/5.2/topics/settings/
"""

import environ
import os
import sys
from pathlib import Path
from datetime import timedelta
from django.core.exceptions import ImproperlyConfigured

# Conditional Sentry import with error handling
try:
    import sentry_sdk

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Environment variable configuration
env = environ.Env(
    # Environment variable type casting and defaults
    DEBUG=(bool, False),
    SECURE_SSL_REDIRECT=(bool, False),
    SESSION_COOKIE_SECURE=(bool, False),
    CSRF_COOKIE_SECURE=(bool, False),
)

# Load environment variables from .env file
env_file = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_file):
    environ.Env.read_env(env_file)

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env("DEBUG", default=False)

# SECURITY: Enforce secret key in production
if DEBUG:
    SECRET_KEY = env("SECRET_KEY", default="django-insecure-dev-only-key-replace-in-production")
else:
    SECRET_KEY = env("SECRET_KEY")
    if not SECRET_KEY:
        raise ImproperlyConfigured(
            "SECRET_KEY must be set in environment for production. "
            "Generate with: python -c 'from django.core.management.utils import get_random_secret_key; "
            "print(get_random_secret_key())'"
        )

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])
APP_ENV = env("APP_ENV", default="development")


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party applications
    "rest_framework",
    "rest_framework_simplejwt",
    "corsheaders",
    "drf_spectacular",
    "django_celery_beat",
    # VoyageurCompass applications
    "Analytics",
    "Core",
    "Data",
    "Design",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # Static file serving middleware
    "Core.middleware.compression.IntelligentCompressionMiddleware",  # Response compression
    "Core.middleware.compression.StaticFileCompressionMiddleware",  # Static file compression
    "corsheaders.middleware.CorsMiddleware",  # Standard CORS middleware
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.locale.LocaleMiddleware",  # Language detection and switching
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "Core.backends.BlacklistCheckMiddleware",  # JWT token blacklist verification
    "Core.middleware.deduplication.RequestDeduplicationMiddleware",  # Request deduplication
    "Core.middleware.database_optimizer.DatabaseQueryAnalyzerMiddleware",  # Database query optimisation
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "VoyageurCompass.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.media",
            ],
        },
    },
]

WSGI_APPLICATION = "VoyageurCompass.wsgi.application"


# Database Configuration
# Database password security enforcement
if DEBUG:
    DB_PASSWORD = env("DB_PASSWORD", default="dev-only-password")
else:
    DB_PASSWORD = env("DB_PASSWORD")
    if not DB_PASSWORD:
        raise ImproperlyConfigured(
            "DB_PASSWORD must be set in environment for production"
        )

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("DB_NAME", default="voyageur_compass_db"),
        "USER": env("DB_USER", default="voyageur_user"),
        "PASSWORD": DB_PASSWORD,
        "HOST": env("DB_HOST", default="localhost"),
        "PORT": env("DB_PORT", default="5432"),
        # Connection Management Settings
        "CONN_MAX_AGE": 600,
        "CONN_HEALTH_CHECKS": True,
        # Database Optimization Settings
        "OPTIONS": {
            "connect_timeout": 10,
            "options": "-c statement_timeout=30000",
            "isolation_level": 2,
            "client_encoding": "UTF8",
        },
        # Transaction Settings - Disabled for performance (use selective transactions instead)
        "ATOMIC_REQUESTS": False,
    }
}


# Database engine validation - enforce PostgreSQL usage
def checkDatabaseEngine():
    """Enforce PostgreSQL database engine usage."""
    if "sqlite" in DATABASES["default"]["ENGINE"].lower() and "test" not in sys.argv and "pytest" not in sys.modules:
        raise ImproperlyConfigured("SQLite is not allowed! Configure PostgreSQL in DATABASES setting.")


checkDatabaseEngine()

# Database optimisation settings
SLOW_QUERY_THRESHOLD = env("SLOW_QUERY_THRESHOLD", default=1.0)  # seconds
CACHE_EXPENSIVE_QUERIES = env("CACHE_EXPENSIVE_QUERIES", default=True)
EXPENSIVE_QUERY_THRESHOLD = env("EXPENSIVE_QUERY_THRESHOLD", default=0.5)  # seconds

# Connection pooling settings for external APIs
API_POOL_CONNECTIONS = env("API_POOL_CONNECTIONS", default=100)
API_POOL_MAXSIZE = env("API_POOL_MAXSIZE", default=100)
API_MAX_RETRIES = env("API_MAX_RETRIES", default=3)
API_BACKOFF_FACTOR = env("API_BACKOFF_FACTOR", default=0.3)
API_TIMEOUT = (10, 30)  # (connect_timeout, read_timeout)

# Test database configuration - PostgreSQL for consistency
if "test" in sys.argv or "pytest" in sys.modules:
    # Test database password configuration
    TEST_DB_PASSWORD = env("TEST_DB_PASSWORD", default=DB_PASSWORD)
    
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("TEST_DB_NAME", default="test_voyageur_compass_db"),
            "USER": env("TEST_DB_USER", default="voyageur_user"),
            "PASSWORD": TEST_DB_PASSWORD,
            "HOST": env("TEST_DB_HOST", default="localhost"),
            "PORT": env("TEST_DB_PORT", default="5432"),
            "OPTIONS": {
                "connect_timeout": 10,
            },
            "TEST": {
                "NAME": "test_voyageur_compass_db",
            },
        }
    }

# Redis Cache Configuration
REDIS_HOST = env("REDIS_HOST", default="redis")
REDIS_PORT = env("REDIS_PORT", default="6379")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Production Redis cache configuration with connection pooling and optimization
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": REDIS_URL,
        "TIMEOUT": 300,
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 50,
                "retry_on_timeout": True,
                "socket_keepalive": True,
                "socket_keepalive_options": {},
                "health_check_interval": 30,
            },
            "COMPRESSOR": "django_redis.compressors.zlib.ZlibCompressor",
            "SERIALIZER": "django_redis.serializers.json.JSONSerializer",
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
        "KEY_PREFIX": "voyageur",
        "VERSION": 1,
    },
    # L2 cache for longer-term storage (explanations, translations)
    "l2_cache": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": f"redis://{REDIS_HOST}:{REDIS_PORT}/1",
        "TIMEOUT": 3600,
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 30,
                "retry_on_timeout": True,
                "socket_keepalive": True,
            },
            "COMPRESSOR": "django_redis.compressors.zlib.ZlibCompressor",
            "SERIALIZER": "django_redis.serializers.json.JSONSerializer",
        },
        "KEY_PREFIX": "voyageur_l2",
        "VERSION": 1,
    },
    # Session cache optimised for user sessions
    "sessions": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": f"redis://{REDIS_HOST}:{REDIS_PORT}/2",
        "TIMEOUT": 86400,
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 20,
                "retry_on_timeout": True,
                "socket_keepalive": True,
            },
            "SERIALIZER": "django_redis.serializers.json.JSONSerializer",
        },
        "KEY_PREFIX": "voyageur_session",
        "VERSION": 1,
    },
}

# Celery Configuration
CELERY_BROKER_URL = env("CELERY_BROKER_URL", default="redis://redis:6379/1")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default="redis://redis:6379/2")
CELERY_ACCEPT_CONTENT = ["application/json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "America/Vancouver"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60
CELERY_TASK_SOFT_TIME_LIMIT = 25 * 60
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_WORKER_MAX_TASKS_PER_CHILD = 1000
CELERY_BEAT_SCHEDULE = {}


# Authentication Backends
AUTHENTICATION_BACKENDS = [
    "Core.backends.EmailOrUsernameModelBackend",  # Custom backend for email/username login
    "django.contrib.auth.backends.ModelBackend",  # Fallback to default Django backend
]

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {
            "min_length": 10,
        },
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
LANGUAGE_CODE = "en"
TIME_ZONE = "America/Vancouver"
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Multilingual Support Configuration
LANGUAGES = [
    ('en', 'English'),
    ('fr', 'Français'),
    ('es', 'Español'),
]

# Locale paths for translation files
LOCALE_PATHS = [
    BASE_DIR / 'locale',
]

# Default language for user preferences
DEFAULT_USER_LANGUAGE = env("DEFAULT_USER_LANGUAGE", default="en")

# Language detection settings
LANGUAGE_DETECTION_ENABLED = env.bool("LANGUAGE_DETECTION_ENABLED", default=True)
LANGUAGE_COOKIE_NAME = 'voyageur_language'
LANGUAGE_COOKIE_AGE = 365 * 24 * 60 * 60  # 1 year

# Formatting settings for different locales
USE_THOUSAND_SEPARATOR = True
NUMBER_GROUPING = 3

# Regional financial formatting preferences
FINANCIAL_FORMATTING = {
    'en': {
        'currency_symbol': '$',
        'currency_position': 'before',
        'decimal_separator': '.',
        'thousands_separator': ',',
        'date_format': 'M/d/Y',
        'time_format': 'g:i A',
    },
    'fr': {
        'currency_symbol': '€',
        'currency_position': 'after',
        'decimal_separator': ',',
        'thousands_separator': ' ',
        'date_format': 'd/m/Y',
        'time_format': 'H:i',
    },
    'es': {
        'currency_symbol': '€',
        'currency_position': 'after',
        'decimal_separator': ',',
        'thousands_separator': '.',
        'date_format': 'd/m/Y',
        'time_format': 'H:i',
    },
}


# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "Design" / "staticfiles"
# Only add static dir if it exists
staticDir = BASE_DIR / "Design" / "static"
STATICFILES_DIRS = [staticDir] if staticDir.exists() else []

# WhiteNoise Configuration for static file compression
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
WHITENOISE_COMPRESS_OFFLINE = True
WHITENOISE_COMPRESSION_QUALITY = 85

# Media files (User uploads)
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "Design" / "media"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# Django REST Framework Configuration
REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "user": "1000/hour",
    },
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
}


# JWT Settings
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=60),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN": False,
    "ALGORITHM": "HS256",
    "SIGNING_KEY": SECRET_KEY,
    "VERIFYING_KEY": None,
    "AUDIENCE": None,
    "ISSUER": None,
    "AUTH_HEADER_TYPES": ("Bearer",),
    "AUTH_HEADER_NAME": "HTTP_AUTHORIZATION",
    "USER_ID_FIELD": "id",
    "USER_ID_CLAIM": "user_id",
    "AUTH_TOKEN_CLASSES": ("rest_framework_simplejwt.tokens.AccessToken",),
    "TOKEN_TYPE_CLAIM": "token_type",
}


# CORS configuration with environment-based origin validation
# Uses standard django-cors-headers middleware for enhanced security
if DEBUG:
    # Development: allow common frontend development ports
    corsOrigins = [
        "http://localhost:3000",  # React default
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",  # Additional Vite port
        "http://127.0.0.1:3002",
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
    ]
else:
    # Production/CI: explicit origin configuration or CI defaults
    corsOrigins = env.list("CORS_ALLOWED_ORIGINS", default=[])
    # Allow CI/CD to run without CORS_ALLOWED_ORIGINS by checking for CI environment
    if not corsOrigins and not env.bool("CI", default=False):
        raise ImproperlyConfigured("CORS_ALLOWED_ORIGINS must be set in production!")
    # For CI environment, use safe defaults for testing
    if env.bool("CI", default=False) and not corsOrigins:
        corsOrigins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

CORS_ALLOWED_ORIGINS = corsOrigins

# CSRF Configuration
CSRF_TRUSTED_ORIGINS = env.list("CSRF_TRUSTED_ORIGINS", default=corsOrigins)

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_METHODS = [
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
]

CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]


# API Documentation Settings
SPECTACULAR_SETTINGS = {
    "TITLE": "Voyageur Compass API",
    "DESCRIPTION": "Financial market analysis and portfolio management API",
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "SWAGGER_UI_SETTINGS": {
        "deepLinking": True,
        "persistAuthorization": True,
        "displayOperationId": True,
    },
    "COMPONENT_SPLIT_REQUEST": True,
}


# Security configuration with mandatory header enforcement
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"

# CSRF cookie security
CSRF_COOKIE_SECURE = env("CSRF_COOKIE_SECURE", default=not DEBUG)
CSRF_COOKIE_SAMESITE = "Strict"
CSRF_COOKIE_HTTPONLY = True

if not DEBUG:
    # HTTPS enforcement
    SECURE_SSL_REDIRECT = env("SECURE_SSL_REDIRECT", default=True)
    SESSION_COOKIE_SECURE = env("SESSION_COOKIE_SECURE", default=True)

    # HSTS settings
    SECURE_HSTS_SECONDS = env.int("SECURE_HSTS_SECONDS", default=31536000)  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True

    # Proxy headers for nginx
    USE_X_FORWARDED_HOST = True
    USE_X_FORWARDED_PORT = True
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

    # Session security
    SESSION_COOKIE_SAMESITE = "Strict"
    SESSION_COOKIE_AGE = 86400  # 24 hours
    SESSION_EXPIRE_AT_BROWSER_CLOSE = False

# Session cookie security configuration
SESSION_COOKIE_HTTPONLY = True


# Structured Logging Configuration
LOG_LEVEL = env("LOG_LEVEL", default="INFO")

# Logging directory structure configuration
LOGS_BASE_DIR = BASE_DIR / "Temp" / "logs"
LOG_DIRS = {
    "web_analysis": LOGS_BASE_DIR / "web_analysis" / "current",
    "web_analysis_archived": LOGS_BASE_DIR / "web_analysis" / "archived",
    "model_training_universal": LOGS_BASE_DIR / "model_training" / "universal_lstm",
    "model_training_individual": LOGS_BASE_DIR / "model_training" / "individual_lstm",
    "model_training_sentiment": LOGS_BASE_DIR / "model_training" / "sentiment",
    "data_collection_stock": LOGS_BASE_DIR / "data_collection" / "stock_data",
    "data_collection_sector": LOGS_BASE_DIR / "data_collection" / "sector_data",
    "data_collection_errors": LOGS_BASE_DIR / "data_collection" / "errors",
    "system_django": LOGS_BASE_DIR / "system" / "django",
    "system_celery": LOGS_BASE_DIR / "system" / "celery",
    "system_api": LOGS_BASE_DIR / "system" / "api",
    "analytics_technical": LOGS_BASE_DIR / "analytics" / "technical",
    "analytics_sentiment": LOGS_BASE_DIR / "analytics" / "sentiment",
    "analytics_portfolio": LOGS_BASE_DIR / "analytics" / "portfolio",
    "security_auth": LOGS_BASE_DIR / "security" / "auth",
    "security_failed": LOGS_BASE_DIR / "security" / "failed_attempts",
    "security_api": LOGS_BASE_DIR / "security" / "api_security",
}

# Ensure all log directories exist
for log_dir in LOG_DIRS.values():
    log_dir.mkdir(parents=True, exist_ok=True)

LOGGING = {
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
        "django_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["system_django"] / "django.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "celery_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["system_celery"] / "celery.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "api_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["system_api"] / "api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "data_collection_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["data_collection_stock"] / "stock_data.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "data_errors_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["data_collection_errors"] / "errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "analytics_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["analytics_technical"] / "technical_analysis.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "sentiment_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["analytics_sentiment"] / "sentiment_analysis.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "security_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["security_auth"] / "auth.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "security",
        },
        "security_failed_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["security_failed"] / "failed_attempts.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 10,
            "formatter": "security",
        },
        "model_training_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIRS["model_training_universal"] / "universal_lstm.log",
            "maxBytes": 52428800,  # 50MB
            "backupCount": 3,
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "django_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "django.server": {
            "handlers": ["console", "api_file"],
            "level": "INFO",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console", "api_file"],
            "level": "WARNING",
            "propagate": False,
        },
        "django.security": {
            "handlers": ["console", "security_file"],
            "level": "WARNING",
            "propagate": False,
        },
        "celery": {
            "handlers": ["console", "celery_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "celery.task": {
            "handlers": ["console", "celery_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "Data": {
            "handlers": ["console", "data_collection_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "Analytics": {
            "handlers": ["console", "analytics_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "Analytics.engine.sentiment": {
            "handlers": ["console", "sentiment_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "Analytics.ml": {
            "handlers": ["console", "model_training_file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "Core.auth": {
            "handlers": ["console", "security_file"],
            "level": "INFO",
            "propagate": False,
        },
        "Core.failed_attempts": {
            "handlers": ["console", "security_failed_file"],
            "level": "WARNING",
            "propagate": False,
        },
        "data_collection_errors": {
            "handlers": ["console", "data_errors_file"],
            "level": "ERROR",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
}

# Import robust logging configuration
try:
    from Core.logging_config import create_robust_logging_config, validate_logging_setup
    
    # Replace the default logging configuration with robust version
    LOGGING = create_robust_logging_config(BASE_DIR, LOG_LEVEL)
    
    # Validate logging setup
    logging_status = validate_logging_setup()
    if not logging_status:
        print("Warning: Logging validation failed, using fallback configuration", file=sys.stderr)
        
except ImportError as e:
    print(f"Warning: Could not import robust logging config: {e}. Using default configuration.", file=sys.stderr)
    # Keep existing logging configuration as fallback

# Sentry Error Tracking
SENTRY_DSN = env("SENTRY_DSN", default=None)
if SENTRY_AVAILABLE and SENTRY_DSN:

    def before_send(event, hint):
        """Data sanitisation filter for Sentry events."""
        # List of sensitive keys to redact
        sensitive_keys = ["password", "token", "secret", "api_key", "private_key", "ssn"]

        def redact_dict(d):
            """Recursive sensitive key redaction."""
            if not isinstance(d, dict):
                return d
            for key in list(d.keys()):
                if any(sensitive in str(key).lower() for sensitive in sensitive_keys):
                    d[key] = "[REDACTED]"
                elif isinstance(d[key], dict):
                    redact_dict(d[key])
                elif isinstance(d[key], list):
                    for item in d[key]:
                        if isinstance(item, dict):
                            redact_dict(item)
            return d

        if isinstance(event, dict):
            event = redact_dict(event.copy())
        return event

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        before_send=before_send,
        send_default_pii=False,
        integrations=[
            sentry_sdk.integrations.django.DjangoIntegration(),
            sentry_sdk.integrations.redis.RedisIntegration(),
        ],
        traces_sample_rate=float(env("SENTRY_TRACES_SAMPLE_RATE", default="0.1")),
        environment=APP_ENV,
        release=env("APP_VERSION", default="unknown"),
    )


# Custom Settings for VoyageurCompass
YAHOO_FINANCE_API_TIMEOUT = int(env("YAHOO_FINANCE_API_TIMEOUT", default=30))
DATA_REFRESH_INTERVAL = int(env("DATA_REFRESH_INTERVAL", default=3600))

# Stock synchronization settings
STOCK_DATA_SYNC_THRESHOLD_SECONDS = int(env("STOCK_DATA_SYNC_THRESHOLD_SECONDS", default=3600))

# Local LLM Configuration
OLLAMA_HOST = env("OLLAMA_HOST", default="localhost")
OLLAMA_PORT = int(env("OLLAMA_PORT", default=11434))
OLLAMA_MODEL = env("OLLAMA_MODEL", default="llama3.1:70b")

# Multi-Model LLM Configuration
# Model assignments for different detail levels and use cases
OLLAMA_SUMMARY_MODEL = env("OLLAMA_SUMMARY_MODEL", default="phi3:3.8b")
OLLAMA_STANDARD_MODEL = env("OLLAMA_STANDARD_MODEL", default="phi3:3.8b") 
OLLAMA_DETAILED_MODEL = env("OLLAMA_DETAILED_MODEL", default="llama3.1:8b")
OLLAMA_TRANSLATION_MODEL = env("OLLAMA_TRANSLATION_MODEL", default="qwen2:3b")

# Legacy model configuration for backward compatibility
OLLAMA_PRIMARY_MODEL = env("OLLAMA_PRIMARY_MODEL", default="llama3.1:8b")

# Model health monitoring and circuit breaker configuration
OLLAMA_HEALTH_CHECK_INTERVAL = int(env("OLLAMA_HEALTH_CHECK_INTERVAL", default=30))
OLLAMA_CIRCUIT_BREAKER_THRESHOLD = int(env("OLLAMA_CIRCUIT_BREAKER_THRESHOLD", default=3))

# LLM Performance Configuration
OLLAMA_PERFORMANCE_MODE = env.bool("OLLAMA_PERFORMANCE_MODE", default=True)
OLLAMA_GENERATION_TIMEOUT = int(env("OLLAMA_GENERATION_TIMEOUT", default=60))  # Increased from 45
OLLAMA_GPU_LAYERS = int(env("OLLAMA_GPU_LAYERS", default=-1))  # -1 = use all available

# Connection retry configuration
OLLAMA_RETRY_ATTEMPTS = int(env("OLLAMA_RETRY_ATTEMPTS", default=3))
OLLAMA_RETRY_DELAY = int(env("OLLAMA_RETRY_DELAY", default=1))

# Multilingual LLM Configuration
MULTILINGUAL_LLM_ENABLED = env.bool("MULTILINGUAL_LLM_ENABLED", default=True)

# Language-specific model assignments
LLM_MODELS_BY_LANGUAGE = {
    'en': env("LLM_MODEL_EN", default="llama3.1:8b"),
    'fr': env("LLM_MODEL_FR", default="qwen2:3b"),
    'es': env("LLM_MODEL_ES", default="qwen2:3b"),
}

# Translation service configuration
TRANSLATION_SERVICE_ENABLED = env.bool("TRANSLATION_SERVICE_ENABLED", default=True)
TRANSLATION_CACHE_TTL = int(env("TRANSLATION_CACHE_TTL", default=86400))  # 24 hours
TRANSLATION_TIMEOUT = int(env("TRANSLATION_TIMEOUT", default=30))
TRANSLATION_MAX_RETRIES = int(env("TRANSLATION_MAX_RETRIES", default=2))

# Quality thresholds for multilingual content
TRANSLATION_QUALITY_THRESHOLD = float(env("TRANSLATION_QUALITY_THRESHOLD", default=0.8))
FINANCIAL_TERMINOLOGY_VALIDATION = env.bool("FINANCIAL_TERMINOLOGY_VALIDATION", default=True)

# Cultural adaptation settings
CULTURAL_FORMATTING_ENABLED = env.bool("CULTURAL_FORMATTING_ENABLED", default=True)
REGIONAL_MARKET_DATA_ENABLED = env.bool("REGIONAL_MARKET_DATA_ENABLED", default=True)

# Financial terminology mapping for translations
FINANCIAL_TERMINOLOGY_MAPPING = {
    'fr': {
        'stock': 'action',
        'market_cap': 'capitalisation_boursiere',
        'dividend': 'dividende',
        'earnings': 'benefices',
        'revenue': 'chiffre_affaires',
        'portfolio': 'portefeuille',
        'analysis': 'analyse',
        'recommendation': 'recommandation',
    },
    'es': {
        'stock': 'accion',
        'market_cap': 'capitalizacion_bursatil',
        'dividend': 'dividendo',
        'earnings': 'ganancias',
        'revenue': 'ingresos',
        'portfolio': 'cartera',
        'analysis': 'analisis',
        'recommendation': 'recomendacion',
    },
}

# Explainability Settings
EXPLAINABILITY_ENABLED = env.bool("EXPLAINABILITY_ENABLED", default=True)
EXPLANATION_CACHE_TTL = int(env("EXPLANATION_CACHE_TTL", default=300))  # 5 minutes
EXPLANATION_TIMEOUT = int(env("EXPLANATION_TIMEOUT", default=60))  # 60 seconds
EXPLANATION_MAX_RETRIES = int(env("EXPLANATION_MAX_RETRIES", default=2))

# Stock Market Settings
TRENDING_STOCKS = env.list(
    "TRENDING_STOCKS",
    default=[
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "UNH",
        "DIS",
        "HD",
        "MA",
    ],
)

# API Configuration Values
MAX_COMPARISON_SYMBOLS = int(env("MAX_COMPARISON_SYMBOLS", default=10))
STOCK_ITERATOR_CHUNK_SIZE = int(env("STOCK_ITERATOR_CHUNK_SIZE", default=50))
MARKET_DATA_BATCH_SIZE = int(env("MARKET_DATA_BATCH_SIZE", default=100))
CACHE_TIMEOUT_MINUTES = int(env("CACHE_TIMEOUT_MINUTES", default=10))

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5 MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10 MB

# Temp directory for processing files
TEMP_DIR = BASE_DIR / "Temp"
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
