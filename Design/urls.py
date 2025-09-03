"""
URL patterns for Design module.
"""
from django.urls import path
from . import views

app_name = 'design'

urlpatterns = [
    # Static asset serving
    path('static/<path:asset_path>/', views.serve_static_asset, name='static_asset'),
    
    # Media file operations
    path('upload/', views.MediaUploadView.as_view(), name='media_upload'),
    
    # Frontend asset serving
    path('assets/<path:asset_file>', views.frontend_asset_handler, name='frontend_asset'),
    
    # Health check
    path('health/', views.health_check, name='health_check'),
]