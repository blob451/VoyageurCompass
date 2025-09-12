"""
High-performance response compression middleware with intelligent compression selection.
Implements gzip and brotli compression with dynamic algorithm selection based on content.
"""

import gzip
import logging
import time
from typing import Any

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

from django.http import HttpResponse
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class IntelligentCompressionMiddleware(MiddlewareMixin):
    """Intelligent response compression with algorithm selection and performance monitoring."""
    
    # Compression thresholds
    MIN_COMPRESSION_SIZE = 1024  # Don't compress responses smaller than 1KB
    MAX_COMPRESSION_SIZE = 10 * 1024 * 1024  # Don't compress responses larger than 10MB
    
    # Content types suitable for compression
    COMPRESSIBLE_CONTENT_TYPES = {
        'application/json',
        'application/javascript',
        'application/xml',
        'text/html',
        'text/css',
        'text/javascript',
        'text/plain',
        'text/xml',
        'application/rss+xml',
        'application/atom+xml',
    }
    
    # Compression algorithms with priorities (higher = preferred)
    COMPRESSION_ALGORITHMS = {
        'br': {'priority': 3, 'available': BROTLI_AVAILABLE, 'compress_func': '_compress_brotli'},
        'gzip': {'priority': 2, 'available': True, 'compress_func': '_compress_gzip'},
        'deflate': {'priority': 1, 'available': True, 'compress_func': '_compress_deflate'},
    }
    
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.compression_stats = {
            'total_requests': 0,
            'compressed_requests': 0,
            'compression_ratios': [],
            'algorithm_usage': {'br': 0, 'gzip': 0, 'deflate': 0},
            'total_bytes_saved': 0,
            'average_compression_time': 0.0
        }
        
        logger.info(f"IntelligentCompressionMiddleware initialised (Brotli: {BROTLI_AVAILABLE})")
    
    def process_response(self, request, response):
        """Process response with intelligent compression."""
        
        self.compression_stats['total_requests'] += 1
        
        # Skip compression for non-compressible responses
        if not self._should_compress(request, response):
            return response
        
        # Determine best compression algorithm
        algorithm = self._select_compression_algorithm(request)
        
        if not algorithm:
            return response
        
        # Perform compression
        start_time = time.time()
        compressed_response = self._compress_response(response, algorithm)
        compression_time = time.time() - start_time
        
        if compressed_response:
            # Update statistics
            self._update_compression_stats(response, compressed_response, algorithm, compression_time)
            return compressed_response
        
        return response
    
    def _should_compress(self, request, response) -> bool:
        """Determine if response should be compressed."""
        
        # Check if already compressed
        if response.get('Content-Encoding'):
            return False
        
        # Check response size
        content_length = len(response.content)
        if content_length < self.MIN_COMPRESSION_SIZE or content_length > self.MAX_COMPRESSION_SIZE:
            return False
        
        # Check content type
        content_type = response.get('Content-Type', '').split(';')[0].strip()
        if content_type not in self.COMPRESSIBLE_CONTENT_TYPES:
            return False
        
        # Check if client accepts compression
        accept_encoding = request.META.get('HTTP_ACCEPT_ENCODING', '')
        if not any(alg in accept_encoding for alg in self.COMPRESSION_ALGORITHMS.keys()):
            return False
        
        return True
    
    def _select_compression_algorithm(self, request) -> str:
        """Select optimal compression algorithm based on client capabilities."""
        
        accept_encoding = request.META.get('HTTP_ACCEPT_ENCODING', '').lower()
        
        # Find available algorithms supported by client, ordered by priority
        available_algorithms = []
        for alg, config in self.COMPRESSION_ALGORITHMS.items():
            if config['available'] and alg in accept_encoding:
                available_algorithms.append((alg, config['priority']))
        
        if not available_algorithms:
            return None
        
        # Return highest priority algorithm
        return max(available_algorithms, key=lambda x: x[1])[0]
    
    def _compress_response(self, response: HttpResponse, algorithm: str) -> HttpResponse:
        """Compress response using specified algorithm."""
        
        try:
            compress_func = getattr(self, self.COMPRESSION_ALGORITHMS[algorithm]['compress_func'])
            compressed_content = compress_func(response.content)
            
            if compressed_content and len(compressed_content) < len(response.content):
                # Create compressed response
                compressed_response = HttpResponse(
                    compressed_content,
                    content_type=response.get('Content-Type'),
                    status=response.status_code
                )
                
                # Copy headers
                for header, value in response.items():
                    if header.lower() not in ['content-length', 'content-encoding']:
                        compressed_response[header] = value
                
                # Set compression headers
                compressed_response['Content-Encoding'] = algorithm
                compressed_response['Content-Length'] = len(compressed_content)
                compressed_response['Vary'] = 'Accept-Encoding'
                
                return compressed_response
            
        except Exception as e:
            logger.warning(f"Compression failed with {algorithm}: {str(e)}")
        
        return None
    
    def _compress_brotli(self, content: bytes) -> bytes:
        """Compress content using Brotli algorithm."""
        if not BROTLI_AVAILABLE:
            return None
        
        return brotli.compress(
            content,
            quality=6,  # Balance between compression ratio and speed
            lgwin=22    # Window size
        )
    
    def _compress_gzip(self, content: bytes) -> bytes:
        """Compress content using gzip algorithm."""
        return gzip.compress(
            content,
            compresslevel=6  # Balance between compression ratio and speed
        )
    
    def _compress_deflate(self, content: bytes) -> bytes:
        """Compress content using deflate algorithm."""
        import zlib
        return zlib.compress(content, level=6)
    
    def _update_compression_stats(self, 
                                original_response: HttpResponse,
                                compressed_response: HttpResponse,
                                algorithm: str,
                                compression_time: float):
        """Update compression performance statistics."""
        
        original_size = len(original_response.content)
        compressed_size = len(compressed_response.content)
        compression_ratio = compressed_size / original_size
        bytes_saved = original_size - compressed_size
        
        # Update statistics
        self.compression_stats['compressed_requests'] += 1
        self.compression_stats['compression_ratios'].append(compression_ratio)
        self.compression_stats['algorithm_usage'][algorithm] += 1
        self.compression_stats['total_bytes_saved'] += bytes_saved
        
        # Update average compression time (exponential moving average)
        current_avg = self.compression_stats['average_compression_time']
        self.compression_stats['average_compression_time'] = (
            0.9 * current_avg + 0.1 * compression_time if current_avg else compression_time
        )
        
        # Limit compression ratios list size for memory efficiency
        if len(self.compression_stats['compression_ratios']) > 1000:
            self.compression_stats['compression_ratios'] = (
                self.compression_stats['compression_ratios'][-500:]
            )
        
        logger.debug(f"Compressed response: {original_size}→{compressed_size} bytes "
                    f"({compression_ratio:.2%}) using {algorithm} in {compression_time:.3f}s")
    
    def get_compression_stats(self) -> dict:
        """Retrieve compression performance statistics."""
        
        stats = self.compression_stats.copy()
        
        # Calculate derived statistics
        if stats['compression_ratios']:
            stats['average_compression_ratio'] = sum(stats['compression_ratios']) / len(stats['compression_ratios'])
            stats['best_compression_ratio'] = min(stats['compression_ratios'])
        else:
            stats['average_compression_ratio'] = 0.0
            stats['best_compression_ratio'] = 1.0
        
        stats['compression_rate'] = (
            stats['compressed_requests'] / max(stats['total_requests'], 1)
        )
        
        return stats


class StaticFileCompressionMiddleware(MiddlewareMixin):
    """Specialised compression middleware for static files with aggressive caching."""
    
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.static_cache = {}  # In-memory cache for compressed static files
        self.max_cache_size = 100  # Maximum number of cached files
        
    def process_response(self, request, response):
        """Compress and cache static file responses."""
        
        # Only process static file requests
        if not request.path.startswith('/static/') and not request.path.startswith('/assets/'):
            return response
        
        content_type = response.get('Content-Type', '').split(';')[0].strip()
        
        # Check if suitable for compression
        if (response.status_code == 200 and 
            len(response.content) > 512 and
            content_type in IntelligentCompressionMiddleware.COMPRESSIBLE_CONTENT_TYPES):
            
            # Check cache first
            cache_key = f"{request.path}:{request.META.get('HTTP_ACCEPT_ENCODING', '')}"
            
            if cache_key in self.static_cache:
                cached_response = self.static_cache[cache_key]
                return HttpResponse(
                    cached_response['content'],
                    content_type=content_type,
                    headers=cached_response['headers']
                )
            
            # Compress and cache
            compressed_response = self._compress_static_file(request, response)
            if compressed_response:
                # Cache the compressed response
                if len(self.static_cache) < self.max_cache_size:
                    self.static_cache[cache_key] = {
                        'content': compressed_response.content,
                        'headers': dict(compressed_response.items())
                    }
                
                return compressed_response
        
        return response
    
    def _compress_static_file(self, request, response):
        """Compress static file with optimal settings."""
        
        accept_encoding = request.META.get('HTTP_ACCEPT_ENCODING', '').lower()
        
        try:
            if BROTLI_AVAILABLE and 'br' in accept_encoding:
                # Use maximum compression for static files
                compressed_content = brotli.compress(response.content, quality=11)
                encoding = 'br'
            elif 'gzip' in accept_encoding:
                compressed_content = gzip.compress(response.content, compresslevel=9)
                encoding = 'gzip'
            else:
                return None
            
            if len(compressed_content) < len(response.content):
                compressed_response = HttpResponse(
                    compressed_content,
                    content_type=response.get('Content-Type'),
                    status=response.status_code
                )
                
                # Copy headers and add compression headers
                for header, value in response.items():
                    if header.lower() not in ['content-length', 'content-encoding']:
                        compressed_response[header] = value
                
                compressed_response['Content-Encoding'] = encoding
                compressed_response['Content-Length'] = len(compressed_content)
                compressed_response['Vary'] = 'Accept-Encoding'
                compressed_response['Cache-Control'] = 'public, max-age=31536000'  # 1 year
                
                return compressed_response
        
        except Exception as e:
            logger.warning(f"Static file compression failed: {str(e)}")
        
        return None