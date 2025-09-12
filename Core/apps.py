from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Core"
    verbose_name = "Core System"

    def ready(self):
        """
        Initialize the Core app.
        This handles authentication and utilities.
        """
        # Import signal handlers here if needed in the future
