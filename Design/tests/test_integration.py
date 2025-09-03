"""
Integration tests for Design module functionality.
"""
import os
import tempfile
import shutil
from django.test import TestCase, Client, override_settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
import json


class DesignModuleIntegrationTest(TestCase):
    """Integration tests for complete Design module workflows."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.client = Client()
        
        # Create test user
        self.user = User.objects.create_user(
            username='integration_test_user',
            password='test_password_123',
            email='integration@test.com'
        )
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.static_dir = os.path.join(self.temp_dir, 'static')
        self.media_dir = os.path.join(self.temp_dir, 'media')
        self.frontend_dir = os.path.join(self.temp_dir, 'Design', 'frontend', 'dist')
        
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(self.frontend_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @override_settings(
        STATICFILES_DIRS=[],
        STATIC_ROOT=None,
        MEDIA_ROOT=None
    )
    def test_static_file_workflow_complete(self):
        """Test complete static file serving workflow."""
        # Create test static file
        test_css_path = os.path.join(self.static_dir, 'style.css')
        test_css_content = b'body { background-color: #f0f0f0; }'
        
        with open(test_css_path, 'wb') as f:
            f.write(test_css_content)
        
        with override_settings(STATICFILES_DIRS=[self.static_dir]):
            # Test file serving
            response = self.client.get('/design/static/style.css/')
            
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, test_css_content)
            self.assertEqual(response['Content-Type'], 'text/css')

    @override_settings(
        MEDIA_ROOT=None,
        DEFAULT_FILE_STORAGE='django.core.files.storage.FileSystemStorage'
    )
    def test_media_upload_workflow_complete(self):
        """Test complete media upload workflow."""
        with override_settings(MEDIA_ROOT=self.media_dir):
            # Login user
            self.client.login(username='integration_test_user', password='test_password_123')
            
            # Create test image file
            test_image_content = b'fake png content for testing'
            test_image = SimpleUploadedFile(
                "test_upload.png",
                test_image_content,
                content_type="image/png"
            )
            
            # Upload file
            response = self.client.post('/design/upload/', {'file': test_image})
            
            self.assertEqual(response.status_code, 200)
            
            response_data = json.loads(response.content)
            self.assertTrue(response_data['success'])
            self.assertEqual(response_data['filename'], 'test_upload.png')
            
            # Verify file was saved
            uploaded_file_path = os.path.join(self.media_dir, 'uploads', 'test_upload.png')
            self.assertTrue(os.path.exists(uploaded_file_path))
            
            # Verify file content
            with open(uploaded_file_path, 'rb') as f:
                saved_content = f.read()
                self.assertEqual(saved_content, test_image_content)

    @override_settings(
        BASE_DIR=None
    )
    def test_frontend_asset_workflow_complete(self):
        """Test complete frontend asset serving workflow."""
        with override_settings(BASE_DIR=self.temp_dir):
            # Create test frontend assets
            js_asset_path = os.path.join(self.frontend_dir, 'app.js')
            css_asset_path = os.path.join(self.frontend_dir, 'app.css')
            
            js_content = b'console.log("Integration test asset");'
            css_content = b'.integration-test { color: blue; }'
            
            with open(js_asset_path, 'wb') as f:
                f.write(js_content)
            
            with open(css_asset_path, 'wb') as f:
                f.write(css_content)
            
            # Test JavaScript asset serving
            js_response = self.client.get('/design/assets/app.js')
            self.assertEqual(js_response.status_code, 200)
            self.assertEqual(js_response.content, js_content)
            self.assertEqual(js_response['Content-Type'], 'application/javascript')
            
            # Test CSS asset serving
            css_response = self.client.get('/design/assets/app.css')
            self.assertEqual(css_response.status_code, 200)
            self.assertEqual(css_response.content, css_content)
            self.assertEqual(css_response['Content-Type'], 'text/css')

    @override_settings(
        STATIC_ROOT=None,
        MEDIA_ROOT=None,
        BASE_DIR=None
    )
    def test_health_check_integration(self):
        """Test health check integration with actual directory states."""
        with override_settings(
            STATIC_ROOT=self.static_dir,
            MEDIA_ROOT=self.media_dir,
            BASE_DIR=self.temp_dir
        ):
            # All directories should exist
            response = self.client.get('/design/health/')
            
            self.assertEqual(response.status_code, 200)
            
            response_data = json.loads(response.content)
            self.assertTrue(response_data['healthy'])
            self.assertEqual(response_data['module'], 'Design')
            
            checks = response_data['checks']
            self.assertTrue(checks['static_directory'])
            self.assertTrue(checks['media_directory'])
            self.assertTrue(checks['frontend_build'])

    def test_security_file_upload_validation(self):
        """Test security validation in file upload workflow."""
        self.client.login(username='integration_test_user', password='test_password_123')
        
        # Test malicious file upload attempt
        malicious_file = SimpleUploadedFile(
            "malware.exe",
            b"MZ\x90\x00",  # PE header signature
            content_type="application/x-msdownload"
        )
        
        response = self.client.post('/design/upload/', {'file': malicious_file})
        
        self.assertEqual(response.status_code, 400)
        
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertIn('not allowed', response_data['error'])

    def test_security_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        # Attempt path traversal in static file serving
        malicious_paths = [
            '../../../etc/passwd',
            '..\\..\\windows\\system32\\config\\sam',
            'valid/../../../sensitive/file'
        ]
        
        for malicious_path in malicious_paths:
            with self.subTest(path=malicious_path):
                response = self.client.get(f'/design/static/{malicious_path}/')
                self.assertEqual(response.status_code, 404)

    @override_settings(
        MEDIA_ROOT=None
    )
    def test_file_size_limit_enforcement(self):
        """Test file size limit enforcement in upload workflow."""
        with override_settings(MEDIA_ROOT=self.media_dir):
            self.client.login(username='integration_test_user', password='test_password_123')
            
            # Create file exceeding size limit
            oversized_content = b'x' * (11 * 1024 * 1024)  # 11MB (exceeds 10MB limit)
            oversized_file = SimpleUploadedFile(
                "huge_image.png",
                oversized_content,
                content_type="image/png"
            )
            
            response = self.client.post('/design/upload/', {'file': oversized_file})
            
            self.assertEqual(response.status_code, 400)
            
            response_data = json.loads(response.content)
            self.assertFalse(response_data['success'])
            self.assertIn('size exceeds', response_data['error'])

    def test_authentication_required_for_uploads(self):
        """Test authentication requirement for file uploads."""
        # Attempt upload without authentication
        test_file = SimpleUploadedFile(
            "test.png",
            b"fake content",
            content_type="image/png"
        )
        
        response = self.client.post('/design/upload/', {'file': test_file})
        
        # Should redirect to login
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    @override_settings(DEBUG=False)
    def test_production_caching_headers(self):
        """Test caching headers in production environment."""
        with override_settings(
            STATICFILES_DIRS=[self.static_dir],
            BASE_DIR=self.temp_dir
        ):
            # Create test files
            static_file_path = os.path.join(self.static_dir, 'prod.css')
            frontend_asset_path = os.path.join(self.frontend_dir, 'prod.js')
            
            with open(static_file_path, 'wb') as f:
                f.write(b'.prod { color: red; }')
            
            with open(frontend_asset_path, 'wb') as f:
                f.write(b'console.log("production");')
            
            # Test static file caching
            static_response = self.client.get('/design/static/prod.css/')
            self.assertEqual(static_response.status_code, 200)
            self.assertIn('Cache-Control', static_response)
            self.assertEqual(static_response['Cache-Control'], 'public, max-age=31536000')
            
            # Test frontend asset caching
            asset_response = self.client.get('/design/assets/prod.js')
            self.assertEqual(asset_response.status_code, 200)
            self.assertIn('Cache-Control', asset_response)
            self.assertEqual(asset_response['Cache-Control'], 'public, max-age=86400')