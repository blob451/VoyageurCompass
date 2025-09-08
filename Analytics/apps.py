from django.apps import AppConfig


class AnalyticsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Analytics'
    verbose_name = 'Analytics Engine'

    def ready(self):
        """
        Initialize the Analytics app.
        This method is called once Django starts.
        """
        # Import signal handlers here if needed in the future

        # Pre-load FinBERT model for faster sentiment analysis
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Starting FinBERT model pre-loading...")

            from Analytics.services.sentiment_analyzer import get_sentiment_analyzer
            sentiment_analyzer = get_sentiment_analyzer()

            # Trigger model loading by running a small test
            test_texts = ["Market conditions look positive for growth stocks."]
            sentiment_analyzer.analyzeSentimentBatch(test_texts)

            logger.info("FinBERT model successfully pre-loaded at startup")

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to pre-load FinBERT model: {str(e)}")
            logger.info("FinBERT will be loaded on first sentiment analysis request")
