"""
Connection pool manager for optimizing database and external service connections.
Provides connection pooling, health monitoring, and automatic failover capabilities.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from django.conf import settings
from django.db import connections

logger = logging.getLogger(__name__)

# Conditional imports for connection pooling
try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis not available - connection pooling limited")
    redis = None
    ConnectionPool = None
    REDIS_AVAILABLE = False

try:
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("Requests library not available")
    requests = None
    REQUESTS_AVAILABLE = False


class DatabaseConnectionPoolManager:
    """Manages database connection pooling and optimization."""

    def __init__(self):
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "connection_errors": 0,
            "average_query_time": 0.0,
        }
        self._stats_lock = threading.Lock()

    def optimize_database_connections(self) -> Dict[str, Any]:
        """Optimize database connection settings based on current usage."""
        try:
            # Get connection info for all databases
            optimization_results = {}

            for db_alias in connections:
                connection = connections[db_alias]

                # Check connection health
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        connection_healthy = True
                except Exception as e:
                    logger.error(f"Database {db_alias} health check failed: {str(e)}")
                    connection_healthy = False

                optimization_results[db_alias] = {
                    "healthy": connection_healthy,
                    "vendor": connection.vendor,
                    "autocommit": connection.get_autocommit(),
                    "in_atomic_block": connection.in_atomic_block,
                }

                # Optimize PostgreSQL connections
                if connection.vendor == 'postgresql' and connection_healthy:
                    self._optimize_postgresql_connection(connection, db_alias)

            return optimization_results

        except Exception as e:
            logger.error(f"Database optimization failed: {str(e)}")
            return {"error": str(e)}

    def _optimize_postgresql_connection(self, connection, db_alias: str):
        """Apply PostgreSQL-specific optimizations."""
        try:
            with connection.cursor() as cursor:
                # Set optimal configuration for performance
                optimizations = [
                    "SET work_mem = '32MB'",
                    "SET maintenance_work_mem = '128MB'",
                    "SET effective_cache_size = '1GB'",
                    "SET random_page_cost = 1.1",
                    "SET default_statistics_target = 100",
                ]

                for optimization in optimizations:
                    try:
                        cursor.execute(optimization)
                        logger.debug(f"Applied optimization to {db_alias}: {optimization}")
                    except Exception as e:
                        logger.warning(f"Failed to apply optimization {optimization}: {str(e)}")

        except Exception as e:
            logger.error(f"PostgreSQL optimization failed for {db_alias}: {str(e)}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get database connection statistics."""
        with self._stats_lock:
            stats = self.connection_stats.copy()

        # Add current connection info
        for db_alias in connections:
            connection = connections[db_alias]
            stats[f"{db_alias}_queries"] = len(connection.queries)

        return stats


class RedisConnectionPoolManager:
    """Manages Redis connection pooling and optimization."""

    def __init__(self):
        self.connection_pools = {}
        self.pool_stats = {
            "pools_created": 0,
            "total_connections": 0,
            "connection_errors": 0,
        }
        self._lock = threading.Lock()

    def create_optimized_redis_pool(
        self,
        redis_url: str,
        pool_name: str = "default",
        max_connections: int = 50
    ) -> Optional[ConnectionPool]:
        """Create optimized Redis connection pool."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available for connection pooling")
            return None

        try:
            # Parse Redis URL
            url_parts = redis.connection.parse_url(redis_url)

            # Create optimized connection pool
            pool = ConnectionPool(
                connection_class=redis.connection.Connection,
                max_connections=max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
                **url_parts
            )

            with self._lock:
                self.connection_pools[pool_name] = pool
                self.pool_stats["pools_created"] += 1
                self.pool_stats["total_connections"] += max_connections

            logger.info(f"Created Redis connection pool '{pool_name}' with {max_connections} connections")
            return pool

        except Exception as e:
            logger.error(f"Failed to create Redis connection pool '{pool_name}': {str(e)}")
            with self._lock:
                self.pool_stats["connection_errors"] += 1
            return None

    def get_redis_client(self, pool_name: str = "default") -> Optional[redis.Redis]:
        """Get Redis client from connection pool."""
        if not REDIS_AVAILABLE:
            return None

        try:
            pool = self.connection_pools.get(pool_name)
            if not pool:
                logger.error(f"Redis connection pool '{pool_name}' not found")
                return None

            client = redis.Redis(connection_pool=pool)
            return client

        except Exception as e:
            logger.error(f"Failed to get Redis client from pool '{pool_name}': {str(e)}")
            return None

    def health_check_pools(self) -> Dict[str, bool]:
        """Check health of all Redis connection pools."""
        pool_health = {}

        for pool_name, pool in self.connection_pools.items():
            try:
                client = redis.Redis(connection_pool=pool)
                client.ping()
                pool_health[pool_name] = True
                logger.debug(f"Redis pool '{pool_name}' health check: OK")
            except Exception as e:
                pool_health[pool_name] = False
                logger.error(f"Redis pool '{pool_name}' health check failed: {str(e)}")

        return pool_health

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get Redis connection pool statistics."""
        with self._lock:
            stats = self.pool_stats.copy()

        # Add per-pool statistics
        for pool_name, pool in self.connection_pools.items():
            try:
                stats[f"{pool_name}_created_connections"] = pool.created_connections
                stats[f"{pool_name}_available_connections"] = len(pool._available_connections)
                stats[f"{pool_name}_in_use_connections"] = len(pool._in_use_connections)
            except AttributeError:
                # Some Redis versions may not have these attributes
                pass

        return stats


class HTTPConnectionPoolManager:
    """Manages HTTP connection pooling for external API calls."""

    def __init__(self):
        self.session_pools = {}
        self.request_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
        }
        self._lock = threading.Lock()

    def create_optimized_session(
        self,
        session_name: str = "default",
        max_retries: int = 3,
        pool_connections: int = 20,
        pool_maxsize: int = 20
    ) -> Optional[requests.Session]:
        """Create optimized HTTP session with connection pooling."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available for HTTP connection pooling")
            return None

        try:
            session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
            )

            # Configure HTTP adapter with connection pooling
            adapter = HTTPAdapter(
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
                max_retries=retry_strategy
            )

            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Set timeouts
            session.timeout = (10, 30)  # Connection timeout, read timeout

            with self._lock:
                self.session_pools[session_name] = session

            logger.info(
                f"Created HTTP session pool '{session_name}' with {pool_connections} connections, "
                f"max size {pool_maxsize}"
            )
            return session

        except Exception as e:
            logger.error(f"Failed to create HTTP session pool '{session_name}': {str(e)}")
            return None

    def get_session(self, session_name: str = "default") -> Optional[requests.Session]:
        """Get HTTP session from pool."""
        return self.session_pools.get(session_name)

    @contextmanager
    def request_with_stats(self, session_name: str = "default"):
        """Context manager for making requests with automatic stats tracking."""
        session = self.get_session(session_name)
        if not session:
            yield None
            return

        start_time = time.time()
        success = False

        try:
            with self._lock:
                self.request_stats["total_requests"] += 1

            yield session
            success = True

        except Exception as e:
            logger.error(f"HTTP request failed: {str(e)}")
            raise

        finally:
            response_time = time.time() - start_time

            with self._lock:
                if success:
                    self.request_stats["successful_requests"] += 1
                else:
                    self.request_stats["failed_requests"] += 1

                # Update average response time
                total_requests = self.request_stats["total_requests"]
                current_avg = self.request_stats["average_response_time"]
                self.request_stats["average_response_time"] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )

    def get_http_stats(self) -> Dict[str, Any]:
        """Get HTTP connection statistics."""
        with self._lock:
            return self.request_stats.copy()


class ConnectionPoolManager:
    """Main connection pool manager coordinating all connection types."""

    def __init__(self):
        self.db_manager = DatabaseConnectionPoolManager()
        self.redis_manager = RedisConnectionPoolManager()
        self.http_manager = HTTPConnectionPoolManager()

        # Initialize connection pools
        self._initialize_pools()

    def _initialize_pools(self):
        """Initialize all connection pools based on settings."""
        try:
            # Initialize Redis pools
            redis_host = getattr(settings, "REDIS_HOST", "redis")
            redis_port = getattr(settings, "REDIS_PORT", "6379")

            # Main Redis pool
            main_redis_url = f"redis://{redis_host}:{redis_port}/0"
            self.redis_manager.create_optimized_redis_pool(
                main_redis_url, "main", max_connections=50
            )

            # Cache Redis pool
            cache_redis_url = f"redis://{redis_host}:{redis_port}/1"
            self.redis_manager.create_optimized_redis_pool(
                cache_redis_url, "cache", max_connections=30
            )

            # Session Redis pool
            session_redis_url = f"redis://{redis_host}:{redis_port}/2"
            self.redis_manager.create_optimized_redis_pool(
                session_redis_url, "sessions", max_connections=20
            )

            # Initialize HTTP session pools
            self.http_manager.create_optimized_session("default", max_retries=3)
            self.http_manager.create_optimized_session("llm_api", max_retries=2, pool_maxsize=10)

            logger.info("Connection pools initialized successfully")

        except Exception as e:
            logger.error(f"Connection pool initialization failed: {str(e)}")

    def optimize_all_connections(self) -> Dict[str, Any]:
        """Optimize all connection types."""
        optimization_results = {
            "database": self.db_manager.optimize_database_connections(),
            "redis_health": self.redis_manager.health_check_pools(),
            "timestamp": time.time(),
        }

        logger.info("Connection optimization completed")
        return optimization_results

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all connection types."""
        return {
            "database": self.db_manager.get_connection_stats(),
            "redis": self.redis_manager.get_pool_stats(),
            "http": self.http_manager.get_http_stats(),
            "timestamp": time.time(),
        }

    def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all connection pools."""
        health_status = {
            "redis_pools": self.redis_manager.health_check_pools(),
            "database_optimized": bool(self.db_manager.optimize_database_connections()),
            "timestamp": time.time(),
        }

        # Overall health assessment
        redis_healthy = all(health_status["redis_pools"].values())
        health_status["overall_healthy"] = redis_healthy and health_status["database_optimized"]

        return health_status


# Singleton instance
_connection_pool_manager = None


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get singleton instance of ConnectionPoolManager."""
    global _connection_pool_manager
    if _connection_pool_manager is None:
        _connection_pool_manager = ConnectionPoolManager()
    return _connection_pool_manager


# Convenience functions
def get_redis_client(pool_name: str = "main") -> Optional[redis.Redis]:
    """Get Redis client from specified pool."""
    manager = get_connection_pool_manager()
    return manager.redis_manager.get_redis_client(pool_name)


def get_http_session(session_name: str = "default") -> Optional[requests.Session]:
    """Get HTTP session from specified pool."""
    manager = get_connection_pool_manager()
    return manager.http_manager.get_session(session_name)