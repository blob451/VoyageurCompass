"""
Connection pooling manager for external API requests.
"""

import logging
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import requests
from django.conf import settings
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern implementation for external API calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is open")
            else:
                self.state = 'half-open'
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class ConnectionPoolManager:
    """
    Connection pool manager for external API requests with advanced features:
    - HTTP connection pooling and reuse
    - Retry logic with exponential backoff
    - Circuit breaker pattern for failing endpoints
    - Request/response monitoring and metrics
    """
    
    def __init__(self):
        self.sessions: Dict[str, requests.Session] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.request_metrics: Dict[str, Dict] = {}
        
        # Default configuration
        self.pool_connections = getattr(settings, 'API_POOL_CONNECTIONS', 100)
        self.pool_maxsize = getattr(settings, 'API_POOL_MAXSIZE', 100)
        self.max_retries = getattr(settings, 'API_MAX_RETRIES', 3)
        self.backoff_factor = getattr(settings, 'API_BACKOFF_FACTOR', 0.3)
        self.timeout = getattr(settings, 'API_TIMEOUT', (10, 30))  # (connect, read)
        
        # Retry configuration
        self.retry_status_forcelist = [429, 500, 502, 503, 504]
        self.retry_methods = ["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
    
    def get_session(self, base_url: str) -> requests.Session:
        """Get or create a session for the given base URL."""
        parsed_url = urlparse(base_url)
        host_key = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if host_key not in self.sessions:
            self.sessions[host_key] = self._create_session()
            self.circuit_breakers[host_key] = CircuitBreaker()
            self.request_metrics[host_key] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0.0,
                'last_request_time': None,
            }
            logger.info(f"Created new session pool for {host_key}")
        
        return self.sessions[host_key]
    
    def _create_session(self) -> requests.Session:
        """Create a new session with optimized configuration."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=self.retry_status_forcelist,
            method_whitelist=self.retry_methods,
            backoff_factor=self.backoff_factor,
            raise_on_redirect=False,
            raise_on_status=False,
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
            max_retries=retry_strategy,
            pool_block=False,
        )
        
        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'User-Agent': 'VoyageurCompass/1.0 (Financial Analytics Platform)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
        return session
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make a request using connection pooling and circuit breaker protection.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL for the request
            **kwargs: Additional arguments passed to requests
            
        Returns:
            requests.Response object
            
        Raises:
            Exception: If circuit breaker is open or request fails
        """
        parsed_url = urlparse(url)
        host_key = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        session = self.get_session(url)
        circuit_breaker = self.circuit_breakers[host_key]
        metrics = self.request_metrics[host_key]
        
        # Set default timeout if not provided
        kwargs.setdefault('timeout', self.timeout)
        
        # Execute request with circuit breaker protection
        start_time = time.time()
        
        try:
            response = circuit_breaker.call(session.request, method, url, **kwargs)
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(metrics, response_time, success=True)
            
            # Log slow requests
            if response_time > 5.0:
                logger.warning(f"Slow API request: {method} {url} took {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(metrics, response_time, success=False)
            
            logger.error(f"API request failed: {method} {url} - {str(e)}")
            raise e
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Convenience method for GET requests."""
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Convenience method for POST requests."""
        return self.request('POST', url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """Convenience method for PUT requests."""
        return self.request('PUT', url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """Convenience method for DELETE requests."""
        return self.request('DELETE', url, **kwargs)
    
    def _update_metrics(self, metrics: Dict, response_time: float, success: bool):
        """Update request metrics."""
        metrics['total_requests'] += 1
        metrics['last_request_time'] = time.time()
        
        if success:
            metrics['successful_requests'] += 1
        else:
            metrics['failed_requests'] += 1
        
        # Update average response time (exponential moving average)
        if metrics['avg_response_time'] == 0:
            metrics['avg_response_time'] = response_time
        else:
            alpha = 0.1  # Smoothing factor
            metrics['avg_response_time'] = (
                alpha * response_time + (1 - alpha) * metrics['avg_response_time']
            )
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get connection pool metrics."""
        return {
            host: {
                **metrics,
                'circuit_breaker_state': self.circuit_breakers[host].state,
                'circuit_breaker_failures': self.circuit_breakers[host].failure_count,
                'success_rate': (
                    metrics['successful_requests'] / max(metrics['total_requests'], 1) * 100
                ),
            }
            for host, metrics in self.request_metrics.items()
        }
    
    def reset_circuit_breaker(self, host: str):
        """Manually reset a circuit breaker."""
        if host in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[host]
            circuit_breaker.failure_count = 0
            circuit_breaker.state = 'closed'
            circuit_breaker.last_failure_time = None
            logger.info(f"Reset circuit breaker for {host}")
    
    def close_all_sessions(self):
        """Close all active sessions."""
        for host, session in self.sessions.items():
            try:
                session.close()
                logger.debug(f"Closed session for {host}")
            except Exception as e:
                logger.warning(f"Error closing session for {host}: {e}")
        
        self.sessions.clear()
        self.circuit_breakers.clear()
        self.request_metrics.clear()


# Global connection pool manager instance
connection_pool_manager = ConnectionPoolManager()


def get_connection_pool() -> ConnectionPoolManager:
    """Get the global connection pool manager instance."""
    return connection_pool_manager


class PooledHTTPClient:
    """
    HTTP client wrapper that uses connection pooling.
    Drop-in replacement for requests with connection pooling benefits.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url
        self.pool = get_connection_pool()
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request using connection pooling."""
        full_url = self._build_url(url)
        return self.pool.get(full_url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """POST request using connection pooling."""
        full_url = self._build_url(url)
        return self.pool.post(full_url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """PUT request using connection pooling."""
        full_url = self._build_url(url)
        return self.pool.put(full_url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """DELETE request using connection pooling."""
        full_url = self._build_url(url)
        return self.pool.delete(full_url, **kwargs)
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Generic request using connection pooling."""
        full_url = self._build_url(url)
        return self.pool.request(method, full_url, **kwargs)
    
    def _build_url(self, url: str) -> str:
        """Build full URL from base URL and endpoint."""
        if self.base_url and not url.startswith(('http://', 'https://')):
            return f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        return url