"""
URL configuration for Core app.
"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from Core.views import (
    ChangePasswordView,
    CustomTokenObtainPairView,
    LogoutView,
    RegisterView,
    UserProfileView,
    api_info,
    health_check,
    health_comprehensive,
    health_database,
    health_redis,
    healthCheck as health_check_liveness,
    readinessCheck as readiness_check,
    server_time,
    system_status,
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
    path("auth/logout/", LogoutView.as_view(), name="auth-logout"),
    # User management
    path("user/profile/", UserProfileView.as_view(), name="user_profile"),
    path("user/change-password/", ChangePasswordView.as_view(), name="change_password"),
    path("user/stats/", user_stats, name="user_stats"),
    # System - health check endpoints
    # Root-level health checks available at /healthz and /readyz  
    path("health/", health_check, name="health_check"),
    path("health/comprehensive/", health_comprehensive, name="health-comprehensive"),
    path("health/database/", health_database, name="health-database"),
    path("health/redis/", health_redis, name="health-redis"),
    path("info/", api_info, name="api-info"),
    path("time/", server_time, name="server-time"),
    path("status/", system_status, name="system-status"),
]
