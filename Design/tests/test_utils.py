"""
Unit tests for Design module utilities with real file operations.
"""
import os
import tempfile
import shutil
from django.test import TestCase, override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import Http404
from Design.utils import StaticFileHandler, MediaFileHandler, FrontendAssetManager
from .fixtures import DesignTestFileFactory, FileSystemTestUtilities, MediaTestUtilities


class TestStaticFileHandler(TestCase):
    """Test cases for StaticFileHandler utility class."""

    def test_validate_file_extension_allowed(self):
        """Test validation of allowed file extensions."""
        valid_files = [
            'style.css', 'script.js', 'image.png', 'font.woff2',
            'data.json', 'page.html', 'icon.svg'
        ]
        
        for filename in valid_files:
            with self.subTest(filename=filename):
                self.assertTrue(
                    StaticFileHandler.validate_file_extension(filename),
                    f"File {filename} should be allowed"
                )

    def test_validate_file_extension_disallowed(self):
        """Test validation rejects disallowed file extensions."""
        invalid_files = [
            'malware.exe', 'script.php', 'config.conf',
            'database.db', 'archive.zip', 'binary.bin'
        ]
        
        for filename in invalid_files:
            with self.subTest(filename=filename):
                self.assertFalse(
                    StaticFileHandler.validate_file_extension(filename),
                    f"File {filename} should not be allowed"
                )

    def test_get_file_mime_type(self):
        """Test MIME type detection for various file types."""
        mime_tests = [
            ('style.css', 'text/css'),
            ('script.js', 'text/javascript'),
            ('image.png', 'image/png'),
            ('image.jpg', 'image/jpeg'),
            ('data.json', 'application/json')
        ]
        
        for filename, expected_mime in mime_tests:
            with self.subTest(filename=filename):
                actual_mime = StaticFileHandler.get_file_mime_type(filename)
                self.assertEqual(actual_mime, expected_mime)

    def test_is_safe_path_valid(self):
        """Test safe path validation for valid paths."""
        valid_paths = [
            'css/style.css',
            'js/app.js',
            'images/logo.png',
            'fonts/regular.woff2'
        ]
        
        for path in valid_paths:
            with self.subTest(path=path):
                self.assertTrue(
                    StaticFileHandler.is_safe_path(path),
                    f"Path {path} should be safe"
                )

    def test_is_safe_path_invalid(self):
        """Test safe path validation rejects unsafe paths."""
        unsafe_paths = [
            '../../../etc/passwd',
            '..\\windows\\system32',
            '/absolute/path',
            'valid/../unsafe/path'
        ]
        
        for path in unsafe_paths:
            with self.subTest(path=path):
                self.assertFalse(
                    StaticFileHandler.is_safe_path(path),
                    f"Path {path} should not be safe"
                )

    @override_settings(STATICFILES_DIRS=['/test/static'])
    def test_get_static_file_path_valid(self):
        """Test static file path generation for valid paths."""
        test_path = 'css/style.css'
        expected_path = os.path.join('/test/static', test_path)
        
        actual_path = StaticFileHandler.get_static_file_path(test_path)
        self.assertEqual(actual_path, expected_path)

    def test_get_static_file_path_invalid_raises_404(self):
        """Test static file path generation raises Http404 for unsafe paths."""
        unsafe_path = '../../../etc/passwd'
        
        with self.assertRaises(Http404):
            StaticFileHandler.get_static_file_path(unsafe_path)


class TestMediaFileHandler(TestCase):
    """Test cases for MediaFileHandler utility class."""

    def test_validate_upload_file_valid(self):
        """Test validation of valid upload files."""
        # Create small test file
        file_content = b"Test file content"
        test_file = SimpleUploadedFile(
            "test_image.png", 
            file_content, 
            content_type="image/png"
        )
        
        is_valid, message = MediaFileHandler.validate_upload_file(test_file)
        
        self.assertTrue(is_valid)
        self.assertEqual(message, "Valid file")

    def test_validate_upload_file_too_large(self):
        """Test validation rejects files exceeding size limit."""
        # Create oversized file content
        oversized_content = b"x" * (MediaFileHandler.MAX_FILE_SIZE + 1)
        large_file = SimpleUploadedFile(
            "large_image.png",
            oversized_content,
            content_type="image/png"
        )
        
        is_valid, message = MediaFileHandler.validate_upload_file(large_file)
        
        self.assertFalse(is_valid)
        self.assertIn("size exceeds", message)

    def test_validate_upload_file_invalid_extension(self):
        """Test validation rejects disallowed file extensions."""
        file_content = b"Malicious content"
        malicious_file = SimpleUploadedFile(
            "malware.exe",
            file_content,
            content_type="application/x-msdownload"
        )
        
        is_valid, message = MediaFileHandler.validate_upload_file(malicious_file)
        
        self.assertFalse(is_valid)
        self.assertIn("not allowed", message)

    def test_sanitize_filename(self):
        """Test filename sanitisation removes unsafe characters."""
        unsafe_filenames = [
            ('file<name>.txt', 'file_name_.txt'),
            ('path/to/file.pdf', 'file.pdf'),
            ('file:with|unsafe*chars.doc', 'file_with_unsafe_chars.doc'),
            ('  spaced  file  .jpg  ', 'spaced  file  .jpg')
        ]
        
        for unsafe, expected_safe in unsafe_filenames:
            with self.subTest(filename=unsafe):
                safe_filename = MediaFileHandler.sanitize_filename(unsafe)
                self.assertEqual(safe_filename, expected_safe)


class TestFrontendAssetManager(TestCase):
    """Test cases for FrontendAssetManager utility class with real file operations."""

    def setUp(self):
        """Set up real test files and directories."""
        self.file_factory = DesignTestFileFactory()
        
        # Create real frontend directory structure
        self.base_dir = os.path.join(self.file_factory.temp_base, 'test_app')
        self.frontend_dist = os.path.join(self.base_dir, 'Design', 'frontend', 'dist')
        os.makedirs(self.frontend_dist, exist_ok=True)
        
        # Create real asset files
        valid_types = ['js', 'css', 'map']
        for asset_type in valid_types:
            asset_file = os.path.join(self.frontend_dist, f'app.{asset_type}')
            with open(asset_file, 'w', encoding='utf-8') as f:
                if asset_type == 'js':
                    f.write('console.log("test asset");')
                elif asset_type == 'css':
                    f.write('body { margin: 0; }')
                elif asset_type == 'map':
                    f.write('{"version": 3, "sources": ["app.js"]}')
    
    def tearDown(self):
        """Clean up test files."""
        self.file_factory.cleanup()

    def test_get_asset_path_valid_types(self):
        """Test asset path generation for valid asset types with real files."""
        valid_types = ['js', 'css', 'map']
        
        with override_settings(BASE_DIR=self.base_dir):
            for asset_type in valid_types:
                with self.subTest(asset_type=asset_type):
                    asset_path = FrontendAssetManager.get_asset_path('app', asset_type)
                    expected_path = os.path.join(self.base_dir, 'Design', 'frontend', 'dist', f'app.{asset_type}')
                    self.assertEqual(asset_path, expected_path)
                    
                    # Verify file actually exists
                    self.assertTrue(FileSystemTestUtilities.validate_file_exists(asset_path))

    def test_get_asset_path_invalid_type(self):
        """Test asset path generation raises ValueError for invalid types."""
        with override_settings(BASE_DIR=self.base_dir):
            with self.assertRaises(ValueError) as context:
                FrontendAssetManager.get_asset_path('app', 'invalid')
            
            self.assertIn("Unsupported asset type", str(context.exception))

    def test_get_asset_path_missing_file(self):
        """Test asset path generation raises Http404 for missing files."""
        with override_settings(BASE_DIR=self.base_dir):
            with self.assertRaises(Http404):
                FrontendAssetManager.get_asset_path('missing', 'js')

    @override_settings(DEBUG=False)
    def test_is_production_build_debug_false(self):
        """Test production build detection when DEBUG is False using real settings."""
        result = FrontendAssetManager.is_production_build()
        self.assertTrue(result)

    @override_settings(DEBUG=True)
    def test_is_production_build_debug_true(self):
        """Test production build detection when DEBUG is True using real settings."""
        result = FrontendAssetManager.is_production_build()
        self.assertFalse(result)

    def test_asset_types_mapping(self):
        """Test asset type to MIME type mapping completeness."""
        expected_mappings = {
            'js': 'application/javascript',
            'css': 'text/css',
            'map': 'application/json'
        }
        
        self.assertEqual(FrontendAssetManager.ASSET_TYPES, expected_mappings)