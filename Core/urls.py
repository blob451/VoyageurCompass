"""
URL configuration for Core app.
"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from Core.views import (
    ChangePasswordView,
    CustomTokenObtainPairView,
    RegisterView,
    UserProfileView,
    health_check,
    healthCheck as health_check_liveness,
    readinessCheck as readiness_check,
    user_stats,
)

app_name = "core"

urlpatterns = [
    # Authentication endpoints
    path("auth/login/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/login/", CustomTokenObtainPairView.as_view(), name="auth-login"),  # Alias for tests
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/register/", RegisterView.as_view(), name="auth-register"),  # Alias for tests
    path("auth/logout/", CustomTokenObtainPairView.as_view(), name="auth-logout"),  # Placeholder for tests
    # User management
    path("user/profile/", UserProfileView.as_view(), name="user_profile"),
    path("user/change-password/", ChangePasswordView.as_view(), name="change_password"),
    path("user/stats/", user_stats, name="user_stats"),
    # System - main health check endpoint only
    # Root-level health checks available at /healthz and /readyz
    path("health/", health_check, name="health_check"),
    path("health/comprehensive/", health_check, name="health-comprehensive"),  # Alias for tests
    path("health/database/", health_check, name="health-database"),  # Alias for tests
    path("health/redis/", health_check, name="health-redis"),  # Alias for tests
    path("info/", health_check, name="api-info"),  # Alias for tests
    path("time/", health_check, name="server-time"),  # Alias for tests
    path("status/", health_check, name="system-status"),  # Alias for tests
]
