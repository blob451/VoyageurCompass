"""
Request deduplication middleware to prevent duplicate concurrent requests.
"""

import hashlib
import json
import logging
import time
from typing import Dict, Optional

from django.core.cache import cache
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class RequestDeduplicationMiddleware(MiddlewareMixin):
    """
    Middleware to prevent duplicate concurrent requests by using Redis-based locking.
    
    This middleware:
    1. Generates unique fingerprints for requests based on user, endpoint, and parameters
    2. Uses Redis locks to prevent concurrent identical requests
    3. Returns cached results for duplicate requests
    4. Implements sliding window deduplication with configurable timeouts
    """

    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.deduplication_window = 5  # seconds
        self.cache_timeout = 30  # seconds for caching results
        self.max_wait_time = 10  # maximum time to wait for a request to complete
        
        # Endpoints that should be deduplicated
        self.deduplicated_endpoints = {
            '/api/analytics/analyze/',
            '/api/analytics/batch-analysis/',
            '/api/analytics/market-overview/',
        }
        
        # Parameters to include in fingerprint generation
        self.fingerprint_params = {
            'symbol', 'symbols', 'months', 'sync', 'include_explanation', 
            'explanation_detail', 'horizon', 'portfolio_id'
        }

    def process_request(self, request):
        """Process incoming request for deduplication."""
        # Only apply deduplication to configured endpoints
        if not any(request.path.startswith(endpoint) for endpoint in self.deduplicated_endpoints):
            return None

        # Skip deduplication for non-analytical operations
        if request.method not in ['GET', 'POST']:
            return None

        # Generate request fingerprint
        fingerprint = self._generate_request_fingerprint(request)
        if not fingerprint:
            return None

        # Check if identical request is already in progress
        lock_key = f"request_lock:{fingerprint}"
        result_key = f"request_result:{fingerprint}"
        
        # Try to acquire lock
        if cache.add(lock_key, True, self.deduplication_window):
            # Lock acquired - this request will proceed
            request._dedup_fingerprint = fingerprint
            request._dedup_is_primary = True
            logger.debug(f"Request acquired lock: {fingerprint}")
            return None
        else:
            # Lock exists - check if there's a cached result
            request._dedup_fingerprint = fingerprint
            request._dedup_is_primary = False
            
            # Wait for result with timeout
            start_time = time.time()
            while time.time() - start_time < self.max_wait_time:
                cached_result = cache.get(result_key)
                if cached_result:
                    logger.info(f"Returning cached result for duplicate request: {fingerprint}")
                    return JsonResponse(cached_result)
                
                # Check if lock is still held
                if not cache.get(lock_key):
                    break
                    
                time.sleep(0.1)  # Small delay before checking again
            
            # If we get here, either timeout or lock released without result
            logger.warning(f"Duplicate request timeout or no result: {fingerprint}")
            return None

    def process_response(self, request, response):
        """Process response for caching and lock release."""
        if not hasattr(request, '_dedup_fingerprint'):
            return response

        fingerprint = request._dedup_fingerprint
        is_primary = getattr(request, '_dedup_is_primary', False)

        if is_primary:
            # Primary request - cache result and release lock
            lock_key = f"request_lock:{fingerprint}"
            result_key = f"request_result:{fingerprint}"
            
            try:
                # Cache successful responses
                if response.status_code == 200 and hasattr(response, 'content'):
                    try:
                        result_data = json.loads(response.content.decode('utf-8'))
                        cache.set(result_key, result_data, self.cache_timeout)
                        logger.debug(f"Cached result for request: {fingerprint}")
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to cache response for {fingerprint}: {e}")
                
            finally:
                # Always release the lock
                cache.delete(lock_key)
                logger.debug(f"Released lock for request: {fingerprint}")

        return response

    def process_exception(self, request, exception):
        """Handle exceptions by releasing locks."""
        if hasattr(request, '_dedup_fingerprint') and getattr(request, '_dedup_is_primary', False):
            fingerprint = request._dedup_fingerprint
            lock_key = f"request_lock:{fingerprint}"
            cache.delete(lock_key)
            logger.warning(f"Released lock due to exception: {fingerprint}")
        return None

    def _generate_request_fingerprint(self, request) -> Optional[str]:
        """Generate a unique fingerprint for the request."""
        try:
            # Base components for fingerprint
            fingerprint_data = {
                'path': request.path,
                'method': request.method,
            }

            # Add user ID if authenticated
            if hasattr(request, 'user') and request.user.is_authenticated:
                fingerprint_data['user_id'] = request.user.id
            else:
                # For anonymous users, use session key or IP
                fingerprint_data['session'] = request.session.session_key or request.META.get('REMOTE_ADDR', 'unknown')

            # Add relevant query parameters
            if request.method == 'GET':
                params = {}
                for param in self.fingerprint_params:
                    if param in request.GET:
                        params[param] = request.GET[param]
                if params:
                    fingerprint_data['params'] = params

            # Add relevant POST data
            elif request.method == 'POST':
                try:
                    if hasattr(request, 'data'):
                        # DRF request
                        post_data = dict(request.data)
                    else:
                        # Django request
                        post_data = dict(request.POST)
                        
                        # Try to parse JSON body
                        if request.content_type == 'application/json':
                            try:
                                body_data = json.loads(request.body.decode('utf-8'))
                                post_data.update(body_data)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass

                    # Filter relevant parameters
                    filtered_data = {}
                    for param in self.fingerprint_params:
                        if param in post_data:
                            value = post_data[param]
                            # Handle list parameters (like symbols)
                            if isinstance(value, list):
                                filtered_data[param] = sorted(value)  # Sort for consistency
                            else:
                                filtered_data[param] = value
                    
                    if filtered_data:
                        fingerprint_data['data'] = filtered_data

                except Exception as e:
                    logger.warning(f"Failed to process POST data for fingerprint: {e}")

            # Generate hash
            fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
            fingerprint_hash = hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()[:16]
            
            logger.debug(f"Generated fingerprint {fingerprint_hash} for {fingerprint_data}")
            return fingerprint_hash

        except Exception as e:
            logger.error(f"Failed to generate request fingerprint: {e}")
            return None


class BatchRequestOptimizer(MiddlewareMixin):
    """
    Middleware to optimize batch requests by combining similar requests.
    
    This middleware:
    1. Collects similar requests within a time window
    2. Combines them into efficient batch operations
    3. Distributes results back to all waiting clients
    """

    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.batch_window = 2  # seconds to collect requests
        self.batch_endpoints = {
            '/api/analytics/analyze/': 'symbols',  # Parameter name for batching
        }
        self.pending_batches: Dict[str, Dict] = {}

    def process_request(self, request):
        """Process request for potential batching."""
        # Check if this endpoint supports batching
        batch_param = None
        for endpoint, param in self.batch_endpoints.items():
            if request.path.startswith(endpoint):
                batch_param = param
                break
        
        if not batch_param:
            return None

        # Only batch GET requests for individual stocks
        if request.method != 'GET':
            return None

        # Extract the symbol from the path (e.g., /api/analytics/analyze/AAPL/)
        path_parts = request.path.strip('/').split('/')
        if len(path_parts) < 4:  # ['api', 'analytics', 'analyze', 'SYMBOL']
            return None

        symbol = path_parts[3].upper()
        
        # Generate batch key based on user and parameters (excluding symbol)
        batch_key = self._generate_batch_key(request, symbol)
        if not batch_key:
            return None

        # Add to pending batch
        if batch_key not in self.pending_batches:
            self.pending_batches[batch_key] = {
                'symbols': [],
                'requests': [],
                'created_at': time.time(),
                'base_request': request,
            }

        batch_info = self.pending_batches[batch_key]
        batch_info['symbols'].append(symbol)
        batch_info['requests'].append(request)

        # Check if batch should be processed
        if (len(batch_info['symbols']) >= 5 or  # Max batch size
            time.time() - batch_info['created_at'] >= self.batch_window):
            
            # Process batch
            return self._process_batch(batch_key)

        # Store request for later processing
        request._batch_key = batch_key
        request._batch_symbol = symbol
        
        # Return a placeholder response - this will be replaced by batch processing
        return None

    def _generate_batch_key(self, request, symbol: str) -> Optional[str]:
        """Generate a key for batching similar requests."""
        try:
            # Include user, common parameters, but exclude symbol
            key_data = {
                'user_id': request.user.id if hasattr(request, 'user') and request.user.is_authenticated else None,
                'params': {k: v for k, v in request.GET.items() if k != 'symbol'}
            }
            
            key_str = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_str.encode('utf-8')).hexdigest()[:12]
            
        except Exception as e:
            logger.error(f"Failed to generate batch key: {e}")
            return None

    def _process_batch(self, batch_key: str):
        """Process a batch of requests."""
        if batch_key not in self.pending_batches:
            return None

        batch_info = self.pending_batches.pop(batch_key)
        symbols = batch_info['symbols']
        requests = batch_info['requests']
        
        logger.info(f"Processing batch request for {len(symbols)} symbols: {symbols}")
        
        # Here you would implement the actual batch processing logic
        # For now, return None to let individual requests proceed
        return None