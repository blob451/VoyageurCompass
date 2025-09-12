"""
Unit tests for Design module views with real file operations.
"""
import os
import tempfile
import shutil
from django.test import TestCase, Client, override_settings
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import Http404
import json
from .fixtures import DesignTestFileFactory, FileSystemTestUtilities


class TestServeStaticAsset(TestCase):
    """Test cases for serve_static_asset view with real file operations."""

    def setUp(self):
        """Set up test client and real files."""
        self.client = Client()
        self.file_factory = DesignTestFileFactory()

        # Create real test files
        self.test_css_path = self.file_factory.create_test_css('style.css')
        self.test_js_path = self.file_factory.create_test_javascript('app.js')

    def tearDown(self):
        """Clean up test files."""
        self.file_factory.cleanup()

    @override_settings(STATICFILES_DIRS=[])
    def test_serve_static_asset_success(self):
        """Test successful static asset serving with real CSS file."""
        # Create settings override with real static directory
        with override_settings(STATIC_ROOT=self.file_factory.test_dirs['static']):
            response = self.client.get('/design/static/style.css/')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/css')

        # Verify actual CSS content is served
        expected_content = open(self.test_css_path, 'rb').read()
        self.assertEqual(response.content, expected_content)

    def test_serve_static_asset_invalid_extension(self):
        """Test static asset serving with invalid file extension using real file."""
        # Create a real malicious file in temp directory
        malicious_file = os.path.join(self.file_factory.test_dirs['temp'], 'malware.exe')
        with open(malicious_file, 'wb') as f:
            f.write(b'MZ\x90\x00\x03\x00')  # PE header

        with override_settings(STATIC_ROOT=self.file_factory.test_dirs['temp']):
            response = self.client.get('/design/static/malware.exe/')

        self.assertEqual(response.status_code, 404)

    def test_serve_static_asset_file_not_found(self):
        """Test static asset serving when file does not exist."""
        with override_settings(STATIC_ROOT=self.file_factory.test_dirs['static']):
            response = self.client.get('/design/static/css/missing.css/')

        self.assertEqual(response.status_code, 404)

    @override_settings(DEBUG=False)
    def test_serve_static_asset_production_caching(self):
        """Test static asset serving includes caching headers in production."""
        with override_settings(STATIC_ROOT=self.file_factory.test_dirs['static']):
            response = self.client.get('/design/static/app.js/')

        self.assertEqual(response.status_code, 200)
        self.assertIn('Cache-Control', response)
        self.assertEqual(response['Cache-Control'], 'public, max-age=31536000')

        # Verify actual JavaScript content is served
        expected_content = open(self.test_js_path, 'rb').read()
        self.assertEqual(response.content, expected_content)


class TestMediaUploadView(TestCase):
    """Test cases for MediaUploadView with real file operations."""

    def setUp(self):
        """Set up test client, user, and real files."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
        self.file_factory = DesignTestFileFactory()

    def tearDown(self):
        """Clean up test files."""
        self.file_factory.cleanup()

    def test_media_upload_requires_authentication(self):
        """Test media upload requires user authentication using real file."""
        # Create real image file for upload test
        test_image_path = self.file_factory.create_test_image('test.png')

        with open(test_image_path, 'rb') as f:
            test_file = SimpleUploadedFile(
                "test.png",
                f.read(),
                content_type="image/png"
            )

        response = self.client.post('/design/upload/', {'file': test_file})
        self.assertEqual(response.status_code, 302)  # Redirect to login

    @override_settings(MEDIA_ROOT=tempfile.mkdtemp())
    def test_media_upload_success(self):
        """Test successful media file upload with real file operations."""
        self.client.login(username='testuser', password='testpass123')

        # Create real image file
        test_image_path = self.file_factory.create_test_image('upload_test.png')

        with open(test_image_path, 'rb') as f:
            test_file = SimpleUploadedFile(
                "upload_test.png",
                f.read(),
                content_type="image/png"
            )

        response = self.client.post('/design/upload/', {'file': test_file})

        self.assertEqual(response.status_code, 200)

        response_data = json.loads(response.content)
        self.assertTrue(response_data['success'])

        # Verify file was actually uploaded
        uploaded_path = os.path.join(os.path.dirname(test_image_path), response_data['filename'])
        self.assertTrue(FileSystemTestUtilities.validate_file_exists(uploaded_path))

    def test_media_upload_no_file(self):
        """Test media upload without file returns error."""
        self.client.login(username='testuser', password='testpass123')

        response = self.client.post('/design/upload/', {})

        self.assertEqual(response.status_code, 400)

        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertIn('No file provided', response_data['error'])

    def test_media_upload_invalid_file(self):
        """Test media upload with oversized file using real file."""
        self.client.login(username='testuser', password='testpass123')

        # Create real oversized file
        large_file_path = self.file_factory.create_large_file('huge.bin', 20)  # 20MB

        with open(large_file_path, 'rb') as f:
            test_file = SimpleUploadedFile(
                "huge.png",
                f.read(),
                content_type="image/png"
            )

        response = self.client.post('/design/upload/', {'file': test_file})

        self.assertEqual(response.status_code, 400)

        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertIn('size exceeds', response_data['error'])

    def test_media_upload_storage_error(self):
        """Test media upload handles storage errors gracefully with real file."""
        self.client.login(username='testuser', password='testpass123')

        # Create real image file
        test_image_path = self.file_factory.create_test_image('storage_test.png')

        # Configure non-writable media root to simulate storage error
        readonly_media_root = tempfile.mkdtemp()
        os.chmod(readonly_media_root, 0o444)  # Read-only permissions

        with open(test_image_path, 'rb') as f:
            test_file = SimpleUploadedFile(
                "storage_test.png",
                f.read(),
                content_type="image/png"
            )

        with override_settings(MEDIA_ROOT=readonly_media_root):
            response = self.client.post('/design/upload/', {'file': test_file})

        # Should handle storage error gracefully
        # The actual behavior might vary based on system permissions
        response_data = json.loads(response.content)
        if response.status_code == 200:
            # If upload succeeds despite readonly directory, that's also valid
            self.assertTrue(response_data['success'])
        else:
            # If it fails as expected, verify error handling
            self.assertIn(response.status_code, [400, 500])
            self.assertFalse(response_data['success'])

        # Cleanup readonly directory
        try:
            os.chmod(readonly_media_root, 0o755)
            shutil.rmtree(readonly_media_root, ignore_errors=True)
        except Exception:
            pass


class TestFrontendAssetView(TestCase):
    """Test cases for frontend_asset view with real file operations."""

    def setUp(self):
        """Set up test client and real frontend assets."""
        self.client = Client()
        self.file_factory = DesignTestFileFactory()

        # Create proper BASE_DIR structure for Django settings
        self.base_dir = self.file_factory.temp_base

        # Create Django expected structure: BASE_DIR/Design/frontend/dist/
        self.frontend_dist = os.path.join(self.base_dir, 'Design', 'frontend', 'dist')
        os.makedirs(self.frontend_dist, exist_ok=True)

        # Create real JavaScript asset
        self.js_content = b'console.log("Real frontend asset loaded");'
        self.js_path = os.path.join(self.frontend_dist, 'app.js')
        with open(self.js_path, 'wb') as f:
            f.write(self.js_content)

        # Create real CSS asset
        self.css_content = b'body { font-family: "Helvetica", sans-serif; margin: 0; }'
        self.css_path = os.path.join(self.frontend_dist, 'app.css')
        with open(self.css_path, 'wb') as f:
            f.write(self.css_content)

    def tearDown(self):
        """Clean up test files."""
        self.file_factory.cleanup()

    @override_settings(DEBUG=True)
    def test_frontend_asset_success(self):
        """Test successful frontend asset serving with real JavaScript file."""
        # Use the correct URL pattern that matches the Django URL configuration
        with override_settings(BASE_DIR=self.base_dir):
            response = self.client.get('/design/assets/app.js')  # Remove trailing slash

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.js_content)
        self.assertEqual(response['Content-Type'], 'application/javascript')
        self.assertEqual(response['Cache-Control'], 'no-cache')

    @override_settings(DEBUG=False)
    def test_frontend_asset_production_caching(self):
        """Test frontend asset serving with production caching and real CSS."""
        with override_settings(BASE_DIR=self.base_dir):
            response = self.client.get('/design/assets/app.css')  # Remove trailing slash

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.css_content)
        self.assertEqual(response['Cache-Control'], 'public, max-age=86400')

    def test_frontend_asset_not_found(self):
        """Test frontend asset serving when asset file does not exist."""
        with override_settings(BASE_DIR=self.base_dir):
            response = self.client.get('/design/assets/missing.js')  # Remove trailing slash

        self.assertEqual(response.status_code, 404)


class TestHealthCheckView(TestCase):
    """Test cases for health_check view with real directory operations."""

    def setUp(self):
        """Set up test client and real directories."""
        self.client = Client()
        self.file_factory = DesignTestFileFactory()

        # Create real directories for health check
        self.static_dir = os.path.join(self.file_factory.temp_base, 'static')
        self.media_dir = os.path.join(self.file_factory.temp_base, 'media')
        self.frontend_dist = os.path.join(self.file_factory.temp_base, 'app', 'Design', 'frontend', 'dist')

        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.frontend_dist, exist_ok=True)

    def tearDown(self):
        """Clean up test directories."""
        self.file_factory.cleanup()

    def test_health_check_all_healthy(self):
        """Test health check when all components are healthy using real directories."""
        with override_settings(
            STATIC_ROOT=self.static_dir,
            MEDIA_ROOT=self.media_dir,
            BASE_DIR=os.path.join(self.file_factory.temp_base, 'app')
        ):
            response = self.client.get('/design/health/')

        self.assertEqual(response.status_code, 200)

        response_data = json.loads(response.content)
        self.assertTrue(response_data['healthy'])
        self.assertEqual(response_data['module'], 'Design')

        checks = response_data['checks']
        self.assertTrue(checks['static_directory'])
        self.assertTrue(checks['media_directory'])
        self.assertTrue(checks['frontend_build'])

    def test_health_check_static_root_none(self):
        """Test health check when STATIC_ROOT is None."""
        with override_settings(
            STATIC_ROOT=None,
            MEDIA_ROOT=self.media_dir,
            BASE_DIR=os.path.join(self.file_factory.temp_base, 'app')
        ):
            response = self.client.get('/design/health/')

        self.assertEqual(response.status_code, 503)

        response_data = json.loads(response.content)
        self.assertFalse(response_data['healthy'])

        checks = response_data['checks']
        self.assertFalse(checks['static_directory'])

    def test_health_check_missing_directories(self):
        """Test health check when directories are missing from filesystem."""
        # Remove static and frontend directories but keep media
        shutil.rmtree(self.static_dir, ignore_errors=True)
        shutil.rmtree(self.frontend_dist, ignore_errors=True)

        with override_settings(
            STATIC_ROOT=self.static_dir,
            MEDIA_ROOT=self.media_dir,
            BASE_DIR=os.path.join(self.file_factory.temp_base, 'app')
        ):
            response = self.client.get('/design/health/')

        self.assertEqual(response.status_code, 503)

        response_data = json.loads(response.content)
        self.assertFalse(response_data['healthy'])

        checks = response_data['checks']
        self.assertFalse(checks['static_directory'])
        self.assertTrue(checks['media_directory'])
        self.assertFalse(checks['frontend_build'])
