"""
Management command to analyze database performance and provide optimization recommendations.
"""

import logging
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from Core.middleware.database_optimizer import get_database_optimizer

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Analyze database performance and provide optimization recommendations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed analysis'
        )
        
        parser.add_argument(
            '--slow-queries',
            action='store_true',
            help='Analyze slow query patterns'
        )
        
        parser.add_argument(
            '--index-recommendations',
            action='store_true',
            help='Suggest missing indexes'
        )
        
        parser.add_argument(
            '--table-stats',
            action='store_true',
            help='Show table usage statistics'
        )

    def handle(self, *args, **options):
        """Execute database performance analysis."""
        self.stdout.write(self.style.SUCCESS('=== Database Performance Analysis ===\n'))
        
        db_optimizer = get_database_optimizer()
        
        try:
            # General database statistics
            self.stdout.write(self.style.HTTP_INFO('Database Statistics'))
            self._show_database_statistics(db_optimizer)
            
            # Connection pool analysis
            self.stdout.write(self.style.HTTP_INFO('\nConnection Analysis'))
            self._show_connection_analysis()
            
            # Table statistics
            if options.get('table_stats', True):
                self.stdout.write(self.style.HTTP_INFO('\nTable Usage Statistics'))
                self._show_table_statistics(db_optimizer)
            
            # Index recommendations
            if options.get('index_recommendations', True):
                self.stdout.write(self.style.HTTP_INFO('\nIndex Recommendations'))
                self._show_index_recommendations()
            
            # Slow query analysis
            if options.get('slow_queries', True):
                self.stdout.write(self.style.HTTP_INFO('\nSlow Query Analysis'))
                self._show_slow_query_analysis()
            
            # Performance recommendations
            self.stdout.write(self.style.HTTP_INFO('\nOptimization Recommendations'))
            self._show_optimization_recommendations()
            
            self.stdout.write(self.style.SUCCESS('\nDatabase analysis completed'))
            
        except Exception as e:
            raise CommandError(f'Database analysis failed: {str(e)}')

    def _show_database_statistics(self, db_optimizer):
        """Show general database statistics."""
        try:
            stats = db_optimizer.get_database_statistics()
            
            if 'error' in stats:
                self.stdout.write(f"WARNING: Could not retrieve statistics: {stats['error']}")
                return
            
            self.stdout.write(f"• Database size: {stats['database_size']}")
            self.stdout.write(f"• Active connections: {stats['connection_count']}")
            
        except Exception as e:
            self.stdout.write(f"WARNING: Statistics error: {str(e)}")

    def _show_connection_analysis(self):
        """Show database connection analysis."""
        try:
            with connection.cursor() as cursor:
                # Get connection info
                cursor.execute("""
                    SELECT 
                        state,
                        count(*) as count
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                    GROUP BY state
                """)
                
                connection_states = cursor.fetchall()
                
                self.stdout.write("Connection states:")
                for state, count in connection_states:
                    self.stdout.write(f"  • {state or 'idle'}: {count}")
                
                # Check for long-running queries
                cursor.execute("""
                    SELECT 
                        query_start,
                        state,
                        left(query, 100) as query_preview
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                      AND query_start < now() - interval '5 minutes'
                      AND state = 'active'
                    ORDER BY query_start
                """)
                
                long_queries = cursor.fetchall()
                
                if long_queries:
                    self.stdout.write("\nWARNING: Long-running queries (>5 minutes):")
                    for start_time, state, query in long_queries:
                        self.stdout.write(f"  • Started: {start_time}")
                        self.stdout.write(f"    Query: {query}...")
                else:
                    self.stdout.write("OK: No long-running queries detected")
                
        except Exception as e:
            self.stdout.write(f"WARNING: Connection analysis error: {str(e)}")

    def _show_table_statistics(self, db_optimizer):
        """Show table usage statistics."""
        try:
            stats = db_optimizer.get_database_statistics()
            
            if 'table_statistics' in stats:
                self.stdout.write("Most accessed tables:")
                
                for table_stat in stats['table_statistics'][:10]:
                    table_name = f"{table_stat['schema']}.{table_stat['table']}"
                    seq_scans = table_stat['seq_scans']
                    idx_scans = table_stat['index_scans']
                    
                    # Calculate scan ratio
                    total_scans = seq_scans + idx_scans
                    if total_scans > 0:
                        seq_ratio = (seq_scans / total_scans) * 100
                        warning = "WARNING: " if seq_ratio > 50 and total_scans > 100 else ""
                    else:
                        seq_ratio = 0
                        warning = ""
                    
                    self.stdout.write(
                        f"  {warning}• {table_name}: "
                        f"{seq_scans} seq / {idx_scans} idx scans "
                        f"({seq_ratio:.1f}% sequential)"
                    )
            
            # Show table sizes
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        schemaname || '.' || tablename as table_name,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables 
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 10
                """)
                
                table_sizes = cursor.fetchall()
                
                self.stdout.write("\nLargest tables:")
                for table_name, size in table_sizes:
                    self.stdout.write(f"  • {table_name}: {size}")
                
        except Exception as e:
            self.stdout.write(f"WARNING: Table statistics error: {str(e)}")

    def _show_index_recommendations(self):
        """Show index recommendations based on query patterns."""
        try:
            with connection.cursor() as cursor:
                # Find tables with high sequential scan ratios
                cursor.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        seq_scan,
                        idx_scan,
                        CASE WHEN seq_scan + idx_scan = 0 THEN 0 
                             ELSE round(100.0 * seq_scan / (seq_scan + idx_scan), 1) 
                        END as seq_scan_ratio
                    FROM pg_stat_user_tables
                    WHERE seq_scan + idx_scan > 100
                      AND seq_scan > idx_scan
                    ORDER BY seq_scan DESC
                    LIMIT 10
                """)
                
                high_seq_scan_tables = cursor.fetchall()
                
                if high_seq_scan_tables:
                    self.stdout.write("Tables that may benefit from indexes:")
                    for schema, table, seq_scans, idx_scans, ratio in high_seq_scan_tables:
                        self.stdout.write(
                            f"  • {schema}.{table}: {ratio}% sequential scans "
                            f"({seq_scans} seq, {idx_scans} idx)"
                        )
                
                # Find unused indexes
                cursor.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan
                    FROM pg_stat_user_indexes
                    WHERE idx_scan < 10
                      AND NOT indexname LIKE '%_pkey'
                    ORDER BY idx_scan
                    LIMIT 10
                """)
                
                unused_indexes = cursor.fetchall()
                
                if unused_indexes:
                    self.stdout.write("\nPossibly unused indexes (consider dropping):")
                    for schema, table, index, scans in unused_indexes:
                        self.stdout.write(f"  • {schema}.{table}.{index}: {scans} scans")
                
        except Exception as e:
            self.stdout.write(f"WARNING: Index analysis error: {str(e)}")

    def _show_slow_query_analysis(self):
        """Show slow query analysis."""
        try:
            with connection.cursor() as cursor:
                # Check if pg_stat_statements is available
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                    )
                """)
                
                has_pg_stat_statements = cursor.fetchone()[0]
                
                if has_pg_stat_statements:
                    cursor.execute("""
                        SELECT 
                            round(mean_exec_time::numeric, 2) as avg_time_ms,
                            calls,
                            round((total_exec_time / 1000)::numeric, 2) as total_time_sec,
                            left(query, 100) as query_preview
                        FROM pg_stat_statements 
                        WHERE mean_exec_time > 100  -- queries slower than 100ms
                        ORDER BY mean_exec_time DESC 
                        LIMIT 10
                    """)
                    
                    slow_queries = cursor.fetchall()
                    
                    if slow_queries:
                        self.stdout.write("Slowest queries (avg execution time):")
                        for avg_time, calls, total_time, query in slow_queries:
                            self.stdout.write(
                                f"  • {avg_time}ms avg ({calls} calls, {total_time}s total)"
                            )
                            self.stdout.write(f"    {query}...")
                    else:
                        self.stdout.write("OK: No particularly slow queries detected")
                else:
                    self.stdout.write("WARNING: pg_stat_statements extension not available")
                    self.stdout.write("    Enable it for detailed query analysis:")
                    self.stdout.write("    ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';")
                
        except Exception as e:
            self.stdout.write(f"WARNING: Slow query analysis error: {str(e)}")

    def _show_optimization_recommendations(self):
        """Show general optimization recommendations."""
        recommendations = [
            "Monitor slow queries with SLOW_QUERY_THRESHOLD setting",
            "Use select_related() and prefetch_related() for related data",
            "Enable query result caching for expensive operations",
            "Add indexes for frequently filtered/sorted columns",
            "Regular VACUUM and ANALYZE maintenance",
            "Use bulk operations for large data modifications",
            "Consider connection pooling (pgBouncer) for high traffic",
            "Use database-level constraints instead of application validation",
            "Implement read replicas for read-heavy workloads",
            "Archive old data to reduce table sizes"
        ]
        
        for recommendation in recommendations:
            self.stdout.write(f"  {recommendation}")
        
        self.stdout.write(f"\nAdditional resources:")
        self.stdout.write(f"  • PostgreSQL Performance Tuning: https://wiki.postgresql.org/wiki/Performance_Optimization")
        self.stdout.write(f"  • Django Database Optimization: https://docs.djangoproject.com/en/stable/topics/db/optimization/")
        self.stdout.write(f"  • EXPLAIN ANALYZE tool: https://explain.depesz.com/")