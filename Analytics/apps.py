import logging
import threading
from django.apps import AppConfig

logger = logging.getLogger(__name__)

# Global model status tracking
MODEL_STATUS = {
    'loading': False,
    'loaded': False,
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
            logger.info("Django startup complete - FinBERT loading in background")

        except Exception as e:
            logger.warning(f"Failed to start async model loading: {str(e)}")
            logger.info("FinBERT will be loaded on first sentiment analysis request")
