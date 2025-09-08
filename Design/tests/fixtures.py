"""
Real test file system fixtures for Design module testing.
"""
import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings


class DesignTestFileFactory:
    """Factory for creating real test files and file system operations."""
    
    def __init__(self):
        """Initialize test file factory with temporary directories."""
        self.temp_base = tempfile.mkdtemp(prefix='design_test_')
        self.test_dirs = {
            'uploads': os.path.join(self.temp_base, 'uploads'),
            'static': os.path.join(self.temp_base, 'static'),
            'media': os.path.join(self.temp_base, 'media'),
            'frontend': os.path.join(self.temp_base, 'frontend'),
            'temp': os.path.join(self.temp_base, 'temp'),
        }
        
        # Create all test directories
        for dir_path in self.test_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def create_test_image(self, filename='test_image.jpg', size=(100, 100)):
        """Create real test image file."""
        try:
            from PIL import Image
            image = Image.new('RGB', size, color='red')
            image_path = os.path.join(self.test_dirs['uploads'], filename)
            image.save(image_path, 'JPEG')
            return image_path
        except ImportError:
            # Fallback: create dummy image file
            image_content = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01'  # JPEG header
            image_path = os.path.join(self.test_dirs['uploads'], filename)
            with open(image_path, 'wb') as f:
                f.write(image_content)
            return image_path
    
    def create_test_css(self, filename='styles.css'):
        """Create real CSS test file."""
        css_content = """
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        """
        css_path = os.path.join(self.test_dirs['static'], filename)
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
        return css_path
    
    def create_test_javascript(self, filename='app.js'):
        """Create real JavaScript test file."""
        js_content = """
        // Test JavaScript file for Design module
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Design module test JavaScript loaded');
            
            // Test function
            function initializeApp() {
                const container = document.querySelector('.container');
                if (container) {
                    container.classList.add('initialized');
                }
            }
            
            // Initialize application
            initializeApp();
            
            // Test API call simulation
            function fetchTestData() {
                return fetch('/api/test/')
                    .then(response => response.json())
                    .then(data => {
                        console.log('Test data loaded:', data);
                        return data;
                    })
                    .catch(error => {
                        console.error('Error loading test data:', error);
                    });
            }
            
            // Export for testing
            window.DesignTestModule = {
                initializeApp,
                fetchTestData
            };
        });
        """
        js_path = os.path.join(self.test_dirs['static'], filename)
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(js_content)
        return js_path
    
    def create_test_html(self, filename='index.html'):
        """Create real HTML test file."""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Voyageur Compass Test</title>
            <link rel="stylesheet" href="styles.css">
        </head>
        <body>
            <div class="container">
                <header class="header">
                    <h1>Voyageur Compass Test Page</h1>
                </header>
                <main>
                    <section class="content">
                        <p>This is a test HTML file for Design module testing.</p>
                        <div class="test-data" data-testid="main-content">
                            <ul>
                                <li>Real HTML structure</li>
                                <li>CSS styling integration</li>
                                <li>JavaScript functionality</li>
                                <li>Asset loading verification</li>
                            </ul>
                        </div>
                    </section>
                </main>
                <footer>
                    <p>© 2024 Voyageur Compass Test Environment</p>
                </footer>
            </div>
            <script src="app.js"></script>
        </body>
        </html>
        """
        html_path = os.path.join(self.test_dirs['frontend'], filename)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return html_path
    
    def create_uploaded_file(self, filename='upload.txt', content='Test file content'):
        """Create real uploaded file for testing."""
        return SimpleUploadedFile(
            filename,
            content.encode('utf-8') if isinstance(content, str) else content,
            content_type='text/plain'
        )
    
    def create_test_pdf(self, filename='document.pdf'):
        """Create real PDF test file."""
        try:
            from reportlab.pdfgen import canvas
            pdf_path = os.path.join(self.test_dirs['uploads'], filename)
            c = canvas.Canvas(pdf_path)
            c.drawString(100, 750, "Test PDF Document")
            c.drawString(100, 700, "Generated for Design module testing")
            c.save()
            return pdf_path
        except ImportError:
            # Fallback: create dummy PDF file
            pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n'
            pdf_path = os.path.join(self.test_dirs['uploads'], filename)
            with open(pdf_path, 'wb') as f:
                f.write(pdf_content)
            return pdf_path
    
    def create_large_file(self, filename='large_file.bin', size_mb=5):
        """Create large test file for size validation testing."""
        file_path = os.path.join(self.test_dirs['temp'], filename)
        size_bytes = size_mb * 1024 * 1024
        
        with open(file_path, 'wb') as f:
            # Write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB chunks
            for _ in range(size_mb):
                f.write(b'0' * chunk_size)
        
        return file_path
    
    def get_file_info(self, file_path):
        """Get real file information for testing."""
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        return {
            'path': file_path,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'is_file': os.path.isfile(file_path),
            'is_dir': os.path.isdir(file_path),
            'permissions': oct(stat.st_mode)[-3:],
            'extension': os.path.splitext(file_path)[1].lower()
        }
    
    def cleanup(self):
        """Clean up all test files and directories."""
        if os.path.exists(self.temp_base):
            shutil.rmtree(self.temp_base, ignore_errors=True)


class FileSystemTestUtilities:
    """Utilities for real file system testing operations."""
    
    @staticmethod
    def validate_file_exists(file_path):
        """Validate file exists with real file system check."""
        return os.path.exists(file_path) and os.path.isfile(file_path)
    
    @staticmethod
    def validate_directory_exists(dir_path):
        """Validate directory exists with real file system check."""
        return os.path.exists(dir_path) and os.path.isdir(dir_path)
    
    @staticmethod
    def get_mime_type(file_path):
        """Get real MIME type of file."""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'
    
    @staticmethod
    def calculate_file_hash(file_path):
        """Calculate real file hash for integrity verification."""
        import hashlib
        if not os.path.exists(file_path):
            return None
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def copy_file(source, destination):
        """Real file copy operation for testing."""
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            print(f"File copy failed: {e}")
            return False
    
    @staticmethod
    def move_file(source, destination):
        """Real file move operation for testing."""
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(source, destination)
            return True
        except Exception as e:
            print(f"File move failed: {e}")
            return False
    
    @staticmethod
    def delete_file(file_path):
        """Real file deletion for testing."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"File deletion failed: {e}")
            return False
    
    @staticmethod
    def create_directory(dir_path):
        """Real directory creation for testing."""
        try:
            os.makedirs(dir_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"Directory creation failed: {e}")
            return False
    
    @staticmethod
    def list_directory_contents(dir_path):
        """Real directory listing for testing."""
        try:
            if not os.path.exists(dir_path):
                return []
            
            contents = []
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                contents.append({
                    'name': item,
                    'path': item_path,
                    'is_file': os.path.isfile(item_path),
                    'is_dir': os.path.isdir(item_path),
                    'size': os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                })
            return contents
        except Exception as e:
            print(f"Directory listing failed: {e}")
            return []
    
    @staticmethod
    def check_disk_space(path=None):
        """Check real disk space for testing."""
        import shutil
        if path is None:
            path = os.getcwd()
        
        try:
            total, used, free = shutil.disk_usage(path)
            return {
                'total': total,
                'used': used,
                'free': free,
                'percent_used': (used / total) * 100 if total > 0 else 0
            }
        except Exception as e:
            print(f"Disk space check failed: {e}")
            return None


class MediaTestUtilities:
    """Utilities for real media file testing."""
    
    @staticmethod
    def validate_image_file(file_path):
        """Validate real image file integrity."""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()  # Verify image integrity
                return True
        except ImportError:
            # Fallback: basic file validation
            return FileSystemTestUtilities.validate_file_exists(file_path)
        except Exception:
            return False
    
    @staticmethod
    def get_image_dimensions(file_path):
        """Get real image dimensions."""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                return img.size
        except ImportError:
            return None
        except Exception:
            return None
    
    @staticmethod
    def resize_image(source_path, destination_path, size=(100, 100)):
        """Real image resize operation for testing."""
        try:
            from PIL import Image
            with Image.open(source_path) as img:
                resized = img.resize(size, Image.Resampling.LANCZOS)
                resized.save(destination_path)
                return True
        except ImportError:
            # Fallback: copy original file
            return FileSystemTestUtilities.copy_file(source_path, destination_path)
        except Exception as e:
            print(f"Image resize failed: {e}")
            return False
    
    @staticmethod
    def validate_file_size(file_path, max_size_mb=5):
        """Validate real file size constraints."""
        if not os.path.exists(file_path):
            return False
        
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes
    
    @staticmethod
    def scan_for_malicious_content(file_path):
        """Basic file security scanning for testing."""
        if not os.path.exists(file_path):
            return False
        
        # Basic checks for potentially malicious files
        suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in suspicious_extensions:
            return False
        
        # Check file size (files that are too large might be suspicious)
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return False
        
        return True


class StaticAssetTestUtilities:
    """Utilities for static asset testing with real file operations."""
    
    @staticmethod
    def validate_css_syntax(css_path):
        """Basic CSS syntax validation for testing."""
        if not os.path.exists(css_path):
            return False
        
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic syntax checks
            brace_count = content.count('{') - content.count('}')
            if brace_count != 0:
                return False
                
            # Check for basic CSS structure
            if ':' not in content and '{' in content:
                return False
                
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_javascript_syntax(js_path):
        """Basic JavaScript syntax validation for testing."""
        if not os.path.exists(js_path):
            return False
        
        try:
            with open(js_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic syntax checks
            paren_count = content.count('(') - content.count(')')
            brace_count = content.count('{') - content.count('}')
            bracket_count = content.count('[') - content.count(']')
            
            if paren_count != 0 or brace_count != 0 or bracket_count != 0:
                return False
                
            return True
        except Exception:
            return False
    
    @staticmethod
    def minify_css(css_path, output_path):
        """Basic CSS minification for testing."""
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic minification: remove comments and extra whitespace
            import re
            # Remove CSS comments
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            # Remove extra whitespace
            content = re.sub(r'\s+', ' ', content)
            content = content.replace(' {', '{').replace('{ ', '{')
            content = content.replace(' }', '}').replace('} ', '}')
            content = content.replace('; ', ';').replace(' ;', ';')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            return True
        except Exception as e:
            print(f"CSS minification failed: {e}")
            return False
    
    @staticmethod
    def compress_image(source_path, destination_path, quality=85):
        """Real image compression for testing."""
        try:
            from PIL import Image
            with Image.open(source_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                
                img.save(destination_path, 'JPEG', quality=quality, optimize=True)
                return True
        except ImportError:
            # Fallback: copy original file
            return FileSystemTestUtilities.copy_file(source_path, destination_path)
        except Exception as e:
            print(f"Image compression failed: {e}")
            return False
