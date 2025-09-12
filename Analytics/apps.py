import logging
import threading
from django.apps import AppConfig
from django.core.management import call_command

logger = logging.getLogger(__name__)

# Global model status tracking
MODEL_STATUS = {
    'loading': False,
    'loaded': False,
    'error': None
}

# Global cache warming status tracking
CACHE_WARMING_STATUS = {
    'warming': False,
    'warmed': False,
    'error': None
}

def load_finbert_model():
    """Load FinBERT model in background thread."""
    try:
        MODEL_STATUS['loading'] = True
        logger.info("Starting FinBERT model pre-loading in background...")

        from Analytics.services.sentiment_analyzer import get_sentiment_analyzer

        sentiment_analyzer = get_sentiment_analyzer()

        # Trigger model loading by running a small test
        test_texts = ["Market conditions look positive for growth stocks."]
        sentiment_analyzer.analyzeSentimentBatch(test_texts)

        MODEL_STATUS['loaded'] = True
        MODEL_STATUS['loading'] = False
        logger.info("FinBERT model successfully pre-loaded in background")

    except Exception as e:
        MODEL_STATUS['loading'] = False
        MODEL_STATUS['error'] = str(e)
        logger.warning(f"Failed to pre-load FinBERT model: {str(e)}")
        logger.info("FinBERT will be loaded on first sentiment analysis request")

def warm_application_cache():
    """Warm application cache in background thread."""
    try:
        CACHE_WARMING_STATUS['warming'] = True
        logger.info("Starting application cache warming in background...")

        # Run cache warming command with optimized settings for startup
        call_command('warm_cache', 
                    stocks=['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'], 
                    skip_indicators=False,
                    verbosity=0)

        CACHE_WARMING_STATUS['warmed'] = True
        CACHE_WARMING_STATUS['warming'] = False
        logger.info("Application cache successfully warmed in background")

    except Exception as e:
        CACHE_WARMING_STATUS['warming'] = False
        CACHE_WARMING_STATUS['error'] = str(e)
        logger.warning(f"Failed to warm application cache: {str(e)}")
        logger.info("Cache will be populated on demand")


class AnalyticsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Analytics"
    verbose_name = "Analytics Engine"

    def ready(self):
        """Initialise the Analytics app upon Django startup."""
        # Import signal handlers here if needed in the future

        # Start FinBERT model loading in background thread
        # This allows Django to start immediately while model loads asynchronously
        try:
            logger.info("Starting async FinBERT model loading...")
            loading_thread = threading.Thread(target=load_finbert_model, daemon=True)
            loading_thread.start()
            logger.info("FinBERT model loading started in background")

        except Exception as e:
            logger.warning(f"Failed to start async model loading: {str(e)}")
            logger.info("FinBERT will be loaded on first sentiment analysis request")

        # Start cache warming in background thread (after short delay to let Django finish startup)
        try:
            logger.info("Starting async cache warming...")
            
            def delayed_cache_warming():
                import time
                time.sleep(5)  # Wait 5 seconds for Django to fully initialize
                warm_application_cache()
            
            warming_thread = threading.Thread(target=delayed_cache_warming, daemon=True)
            warming_thread.start()
            logger.info("Cache warming started in background")

        except Exception as e:
            logger.warning(f"Failed to start async cache warming: {str(e)}")
            logger.info("Cache will be populated on demand")
