"""
Database query optimization middleware and utilities.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List

from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class DatabaseQueryAnalyzerMiddleware(MiddlewareMixin):
    """
    Middleware to analyze and optimize database queries.
    
    Features:
    - Query performance monitoring
    - Slow query detection and logging
    - Query result caching for expensive operations
    - Database connection pool utilization tracking
    """
    
    def __init__(self, get_response=None):
        super().__init__(get_response)
        self.slow_query_threshold = getattr(settings, 'SLOW_QUERY_THRESHOLD', 1.0)  # seconds
        self.cache_expensive_queries = getattr(settings, 'CACHE_EXPENSIVE_QUERIES', True)
        self.expensive_query_threshold = getattr(settings, 'EXPENSIVE_QUERY_THRESHOLD', 0.5)  # seconds
        
        # Query statistics
        self.query_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'max_time': 0.0,
            'slow_queries': 0
        })

    def process_request(self, request):
        """Initialize query tracking for the request."""
        request._query_start_time = time.time()
        request._initial_query_count = len(connection.queries)
        return None

    def process_response(self, request, response):
        """Analyze queries executed during the request."""
        if not hasattr(request, '_query_start_time'):
            return response
        
        # Calculate request duration
        request_duration = time.time() - request._query_start_time
        
        # Get queries executed during this request
        current_queries = connection.queries[getattr(request, '_initial_query_count', 0):]
        
        if current_queries:
            # Analyze queries
            total_query_time = 0.0
            slow_queries = []
            
            for query in current_queries:
                query_time = float(query.get('time', 0))
                total_query_time += query_time
                
                # Check for slow queries
                if query_time > self.slow_query_threshold:
                    slow_queries.append({
                        'sql': query['sql'][:500],  # Truncate long queries
                        'time': query_time,
                        'path': request.path
                    })
                
                # Update statistics
                sql_hash = self._get_query_hash(query['sql'])
                stats = self.query_stats[sql_hash]
                stats['count'] += 1
                stats['total_time'] += query_time
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['max_time'] = max(stats['max_time'], query_time)
                
                if query_time > self.slow_query_threshold:
                    stats['slow_queries'] += 1
            
            # Log slow queries
            if slow_queries:
                logger.warning(
                    f"Slow queries detected on {request.path}: "
                    f"{len(slow_queries)} queries, total time: {total_query_time:.3f}s"
                )
                for slow_query in slow_queries:
                    logger.warning(f"Slow query ({slow_query['time']:.3f}s): {slow_query['sql']}")
            
            # Add query metrics to response headers (in debug mode)
            if settings.DEBUG:
                response['X-DB-Query-Count'] = str(len(current_queries))
                response['X-DB-Query-Time'] = f"{total_query_time:.3f}"
                response['X-Request-Time'] = f"{request_duration:.3f}"
        
        return response

    def _get_query_hash(self, sql: str) -> str:
        """Generate a hash for the query to group similar queries."""
        import hashlib
        import re
        
        # Normalize SQL by removing values and whitespace
        normalized = re.sub(r'\b\d+\b', '?', sql)  # Replace numbers with ?
        normalized = re.sub(r"'[^']*'", '?', normalized)  # Replace string literals with ?
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = normalized.strip().lower()
        
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]

    def get_query_statistics(self) -> Dict:
        """Get query performance statistics."""
        return dict(self.query_stats)


class DatabaseQueryCache:
    """
    Cache for expensive database query results.
    """
    
    def __init__(self):
        self.cache_prefix = 'db_query_cache'
        self.default_timeout = 300  # 5 minutes
        
    def get_cached_result(self, query_key: str, default=None):
        """Get cached query result."""
        cache_key = f"{self.cache_prefix}:{query_key}"
        return cache.get(cache_key, default)
    
    def cache_result(self, query_key: str, result, timeout: int = None):
        """Cache query result."""
        cache_key = f"{self.cache_prefix}:{query_key}"
        timeout = timeout or self.default_timeout
        cache.set(cache_key, result, timeout)
    
    def generate_key(self, model_name: str, method: str, **kwargs) -> str:
        """Generate cache key for query."""
        import hashlib
        import json
        
        key_data = {
            'model': model_name,
            'method': method,
            'kwargs': kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()[:16]
        
        return f"{model_name}:{method}:{key_hash}"


class DatabaseOptimizer:
    """
    Database optimization utilities and query helpers.
    """
    
    def __init__(self):
        self.query_cache = DatabaseQueryCache()
    
    def optimize_queryset(self, queryset, select_related_fields=None, prefetch_related_fields=None):
        """
        Optimize a queryset with select_related and prefetch_related.
        
        Args:
            queryset: Django QuerySet to optimize
            select_related_fields: Fields for select_related
            prefetch_related_fields: Fields for prefetch_related
            
        Returns:
            Optimized QuerySet
        """
        if select_related_fields:
            queryset = queryset.select_related(*select_related_fields)
        
        if prefetch_related_fields:
            queryset = queryset.prefetch_related(*prefetch_related_fields)
        
        return queryset
    
    def cached_query(self, cache_key: str, query_func, timeout: int = 300, *args, **kwargs):
        """
        Execute query with caching.
        
        Args:
            cache_key: Unique key for caching
            query_func: Function that executes the query
            timeout: Cache timeout in seconds
            *args, **kwargs: Arguments for query_func
            
        Returns:
            Query result (from cache or fresh execution)
        """
        # Try to get from cache
        result = self.query_cache.get_cached_result(cache_key)
        
        if result is not None:
            logger.debug(f"Cache hit for query: {cache_key}")
            return result
        
        # Execute query
        start_time = time.time()
        result = query_func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Cache result if query was expensive
        if execution_time > 0.1:  # Cache queries taking more than 100ms
            self.query_cache.cache_result(cache_key, result, timeout)
            logger.debug(f"Cached expensive query ({execution_time:.3f}s): {cache_key}")
        
        return result
    
    def bulk_create_optimized(self, model_class, objects, batch_size=1000):
        """
        Optimized bulk create with batching.
        
        Args:
            model_class: Django model class
            objects: List of objects to create
            batch_size: Number of objects per batch
            
        Returns:
            List of created objects
        """
        created_objects = []
        
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            batch_objects = model_class.objects.bulk_create(batch, ignore_conflicts=True)
            created_objects.extend(batch_objects)
        
        logger.info(f"Bulk created {len(created_objects)} {model_class.__name__} objects")
        return created_objects
    
    def bulk_update_optimized(self, objects, fields, batch_size=1000):
        """
        Optimized bulk update with batching.
        
        Args:
            objects: List of objects to update
            fields: List of field names to update
            batch_size: Number of objects per batch
        """
        if not objects:
            return
        
        model_class = objects[0].__class__
        updated_count = 0
        
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            updated_count += model_class.objects.bulk_update(batch, fields)
        
        logger.info(f"Bulk updated {updated_count} {model_class.__name__} objects")
    
    def analyze_query_plan(self, queryset):
        """
        Analyze query execution plan (PostgreSQL specific).
        
        Args:
            queryset: Django QuerySet to analyze
            
        Returns:
            Query plan information
        """
        try:
            # Get the SQL query
            sql = str(queryset.query)
            
            # Execute EXPLAIN ANALYZE
            with connection.cursor() as cursor:
                cursor.execute(f"EXPLAIN ANALYZE {sql}")
                plan = cursor.fetchall()
            
            return {
                'sql': sql,
                'plan': [row[0] for row in plan],
                'recommendations': self._analyze_plan_recommendations(plan)
            }
            
        except Exception as e:
            logger.error(f"Query plan analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_plan_recommendations(self, plan) -> List[str]:
        """Analyze query plan and provide optimization recommendations."""
        recommendations = []
        plan_text = ' '.join([row[0] for row in plan]).lower()
        
        # Check for common performance issues
        if 'seq scan' in plan_text and 'filter' in plan_text:
            recommendations.append("Consider adding an index for filtered columns")
        
        if 'sort' in plan_text and 'disk' in plan_text:
            recommendations.append("Query requires disk-based sorting - consider increasing work_mem")
        
        if 'nested loop' in plan_text and 'rows' in plan_text:
            recommendations.append("Nested loop join detected - verify indexes on join columns")
        
        if 'hash join' in plan_text and 'buckets' in plan_text:
            recommendations.append("Hash join may benefit from increased hash_mem_multiplier")
        
        return recommendations
    
    def get_database_statistics(self) -> Dict:
        """Get database performance statistics."""
        try:
            with connection.cursor() as cursor:
                # Get connection count
                cursor.execute("SELECT count(*) FROM pg_stat_activity")
                connection_count = cursor.fetchone()[0]
                
                # Get database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                db_size = cursor.fetchone()[0]
                
                # Get table statistics
                cursor.execute("""
                    SELECT 
                        schemaname, 
                        tablename, 
                        n_tup_ins, 
                        n_tup_upd, 
                        n_tup_del,
                        seq_scan,
                        idx_scan
                    FROM pg_stat_user_tables 
                    ORDER BY seq_scan DESC 
                    LIMIT 10
                """)
                table_stats = cursor.fetchall()
                
                return {
                    'connection_count': connection_count,
                    'database_size': db_size,
                    'table_statistics': [
                        {
                            'schema': row[0],
                            'table': row[1],
                            'inserts': row[2],
                            'updates': row[3],
                            'deletes': row[4],
                            'seq_scans': row[5],
                            'index_scans': row[6]
                        }
                        for row in table_stats
                    ]
                }
                
        except Exception as e:
            logger.error(f"Database statistics collection failed: {str(e)}")
            return {'error': str(e)}


# Global database optimizer instance
_database_optimizer = None


def get_database_optimizer() -> DatabaseOptimizer:
    """Get the global database optimizer instance."""
    global _database_optimizer
    
    if _database_optimizer is None:
        _database_optimizer = DatabaseOptimizer()
    
    return _database_optimizer