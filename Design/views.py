"""
Django views for Design module.
"""
import os
from django.http import HttpResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from .utils import StaticFileHandler, MediaFileHandler, FrontendAssetManager


@require_http_methods(["GET"])
def serve_static_asset(request, asset_path):
    """Serve static assets with appropriate headers."""
    try:
        if not StaticFileHandler.validate_file_extension(asset_path):
            raise Http404("File type not allowed")
        
        file_path = StaticFileHandler.get_static_file_path(asset_path)
        
        if not os.path.exists(file_path):
            raise Http404("File not found")
        
        mime_type = StaticFileHandler.get_file_mime_type(asset_path)
        
        with open(file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type=mime_type)
            
        # Cache headers for static assets
        if not settings.DEBUG:
            response['Cache-Control'] = 'public, max-age=31536000'  # 1 year
            
        return response
        
    except Exception:
        raise Http404("Asset not found")


@method_decorator(login_required, name='dispatch')
class MediaUploadView(View):
    """Handle media file uploads."""
    
    @csrf_exempt
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        """Process file upload."""
        if 'file' not in request.FILES:
            return JsonResponse({
                'error': 'No file provided',
                'success': False
            }, status=400)
        
        uploaded_file = request.FILES['file']
        
        # Validate file
        is_valid, message = MediaFileHandler.validate_upload_file(uploaded_file)
        if not is_valid:
            return JsonResponse({
                'error': message,
                'success': False
            }, status=400)
        
        # Sanitise filename
        safe_filename = MediaFileHandler.sanitize_filename(uploaded_file.name)
        
        # Save file
        try:
            from django.core.files.storage import default_storage
            file_path = default_storage.save(f'uploads/{safe_filename}', uploaded_file)
            
            return JsonResponse({
                'success': True,
                'filename': safe_filename,
                'path': file_path,
                'size': uploaded_file.size
            })
            
        except Exception as e:
            return JsonResponse({
                'error': f'Upload failed: {str(e)}',
                'success': False
            }, status=500)


@require_http_methods(["GET"])
def frontend_asset(request, asset_name, asset_type):
    """Serve compiled frontend assets."""
    try:
        asset_path = FrontendAssetManager.get_asset_path(asset_name, asset_type)
        mime_type = FrontendAssetManager.ASSET_TYPES.get(asset_type, 'application/octet-stream')
        
        with open(asset_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type=mime_type)
        
        # Production caching
        if FrontendAssetManager.is_production_build():
            response['Cache-Control'] = 'public, max-age=86400'  # 24 hours
        else:
            response['Cache-Control'] = 'no-cache'
            
        return response
        
    except Exception:
        raise Http404("Frontend asset not found")


@require_http_methods(["GET"])
def frontend_asset_handler(request, asset_file):
    """Handle frontend asset serving with file parsing."""
    try:
        # Parse asset_name and asset_type from file path
        if '.' not in asset_file:
            raise Http404("Invalid asset filename")
        
        asset_name, asset_type = os.path.splitext(asset_file)
        asset_type = asset_type[1:]  # Remove the dot
        
        return frontend_asset(request, asset_name, asset_type)
        
    except Exception:
        raise Http404("Frontend asset not found")


@require_http_methods(["GET"])
def health_check(request):
    """Design module health check."""
    checks = {
        'static_directory': os.path.exists(settings.STATIC_ROOT) if settings.STATIC_ROOT else False,
        'media_directory': os.path.exists(settings.MEDIA_ROOT),
        'frontend_build': os.path.exists(os.path.join(settings.BASE_DIR, 'Design', 'frontend', 'dist'))
    }
    
    all_healthy = all(checks.values())
    
    return JsonResponse({
        'healthy': all_healthy,
        'checks': checks,
        'module': 'Design'
    }, status=200 if all_healthy else 503)