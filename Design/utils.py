"""
Utility functions for Design module operations.
"""
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import Http404
import mimetypes


class StaticFileHandler:
    """Handles static file operations and validation."""

    ALLOWED_EXTENSIONS = {
        '.css', '.js', '.html', '.htm', '.json', '.xml', '.txt',
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
        '.woff', '.woff2', '.ttf', '.eot', '.otf'
    }

    @classmethod
    def validate_file_extension(cls, filename):
        """Validate file extension against allowed types."""
        _, ext = os.path.splitext(filename.lower())
        return ext in cls.ALLOWED_EXTENSIONS

    @classmethod
    def get_file_mime_type(cls, filename):
        """Determine MIME type for file."""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'

    @classmethod
    def is_safe_path(cls, path):
        """Validate path is safe and within allowed directories."""
        # Prevent directory traversal
        if '..' in path or path.startswith('/'):
            return False
        return True

    @classmethod
    def get_static_file_path(cls, relative_path):
        """Get absolute path for static file."""
        if not cls.is_safe_path(relative_path):
            raise Http404("Invalid file path")

        static_root = getattr(settings, 'STATICFILES_DIRS', [])[0] if getattr(settings, 'STATICFILES_DIRS', []) else settings.STATIC_ROOT
        return os.path.join(static_root, relative_path)


class MediaFileHandler:
    """Handles media file operations and validation."""

    ALLOWED_UPLOAD_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv',
        '.zip', '.tar', '.gz'
    }

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    @classmethod
    def validate_upload_file(cls, uploaded_file):
        """Validate uploaded file."""
        # Check file size
        if uploaded_file.size > cls.MAX_FILE_SIZE:
            return False, "File size exceeds maximum allowed size"

        # Check extension
        filename = uploaded_file.name
        _, ext = os.path.splitext(filename.lower())
        if ext not in cls.ALLOWED_UPLOAD_EXTENSIONS:
            return False, "File type not allowed"

        return True, "Valid file"

    @classmethod
    def sanitize_filename(cls, filename):
        """Sanitise filename for safe storage."""
        # Remove path components
        filename = os.path.basename(filename)

        # Replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')

        return filename.strip()


class FrontendAssetManager:
    """Manages frontend asset compilation and serving."""

    ASSET_TYPES = {
        'js': 'application/javascript',
        'css': 'text/css',
        'map': 'application/json'
    }

    @classmethod
    def get_asset_path(cls, asset_name, asset_type):
        """Get path for compiled frontend asset."""
        if asset_type not in cls.ASSET_TYPES:
            raise ValueError(f"Unsupported asset type: {asset_type}")

        # Frontend build directory
        build_dir = os.path.join(settings.BASE_DIR, 'Design', 'frontend', 'dist')
        asset_path = os.path.join(build_dir, f"{asset_name}.{asset_type}")

        if not os.path.exists(asset_path):
            raise Http404(f"Asset not found: {asset_name}.{asset_type}")

        return asset_path

    @classmethod
    def is_production_build(cls):
        """Check if frontend is production build."""
        return not getattr(settings, 'DEBUG', False)
