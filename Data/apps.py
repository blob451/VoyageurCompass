from django.apps import AppConfig


class DataConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Data'
    verbose_name = 'Data Management'
    
    def ready(self):
        """
        Initialize the Data app.
        This handles all data models and external API connections.
        """
        # Import signal handlers here if needed in the future
        pass
