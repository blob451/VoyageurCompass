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
        pass