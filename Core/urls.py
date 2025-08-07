"""
URL configuration for Core app.
"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from Core.views import (
    CustomTokenObtainPairView,
    RegisterView,
    UserProfileView,
    ChangePasswordView,
    health_check,
    user_stats
)

app_name = 'core'

urlpatterns = [
    # Authentication endpoints
    path('auth/login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/register/', RegisterView.as_view(), name='register'),
    
    # User management
    path('user/profile/', UserProfileView.as_view(), name='user_profile'),
    path('user/change-password/', ChangePasswordView.as_view(), name='change_password'),
    path('user/stats/', user_stats, name='user_stats'),
    
    # System
    path('health/', health_check, name='health_check'),
]