"""
Optimised pagination utilities for large dataset handling.
Implements cursor-based pagination for better performance on large tables.
"""

from typing import Any, Dict, Optional

from django.core.paginator import Paginator
from django.db import models
from rest_framework.pagination import CursorPagination, PageNumberPagination
from rest_framework.response import Response


class OptimisedCursorPagination(CursorPagination):
    """Cursor-based pagination optimised for time-series data."""
    
    page_size = 100
    max_page_size = 1000
    page_size_query_param = 'page_size'
    ordering = '-date'  # Default ordering for time-series data
    cursor_query_param = 'cursor'
    
    def get_paginated_response(self, data):
        """Enhanced response with performance metadata."""
        return Response({
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'results': data,
            'page_info': {
                'has_next': self.get_next_link() is not None,
                'has_previous': self.get_previous_link() is not None,
                'page_size': self.page_size,
                'total_estimate': getattr(self, 'total_estimate', None)
            }
        })


class OptimisedPageNumberPagination(PageNumberPagination):
    """Page number pagination with count estimation for performance."""
    
    page_size = 50
    max_page_size = 200
    page_size_query_param = 'page_size'
    page_query_param = 'page'
    
    def get_count_estimate(self, queryset):
        """Fast count estimation for large tables."""
        # Use database statistics for count estimation on large tables
        if queryset.count() > 10000:
            # Fast estimate using table statistics
            table_name = queryset.model._meta.db_table
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT reltuples::bigint FROM pg_class WHERE relname = '{table_name}';")
                result = cursor.fetchone()
                return int(result[0]) if result else 0
        return queryset.count()
    
    def get_paginated_response(self, data):
        """Response with estimated counts for performance."""
        count = self.get_count_estimate(self.page.paginator.object_list)
        return Response({
            'count': count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'page_info': {
                'current_page': self.page.number,
                'total_pages': (count + self.page_size - 1) // self.page_size,
                'page_size': self.page_size,
                'count_is_estimate': count > 10000
            },
            'results': data
        })


class StockPricePagination(OptimisedCursorPagination):
    """Specialised pagination for stock price time-series data."""
    
    page_size = 200  # Larger page size for time-series
    max_page_size = 500
    ordering = '-date'
    
    def get_ordering(self, request, queryset, view):
        """Dynamic ordering based on query parameters."""
        ordering_param = request.query_params.get('ordering')
        if ordering_param in ['date', '-date', 'close', '-close']:
            return ordering_param
        return self.ordering


class AnalyticsPagination(OptimisedPageNumberPagination):
    """Pagination optimised for analytics results."""
    
    page_size = 25
    max_page_size = 100
    
    def get_paginated_response(self, data):
        """Enhanced response with analytics-specific metadata."""
        response = super().get_paginated_response(data)
        
        # Add analytics-specific metadata
        if data:
            response.data['analytics_info'] = {
                'date_range': {
                    'start': min(item.get('analysis_date', '') for item in data if item.get('analysis_date')),
                    'end': max(item.get('analysis_date', '') for item in data if item.get('analysis_date'))
                },
                'symbols_count': len(set(item.get('symbol') for item in data if item.get('symbol')))
            }
        
        return response


def get_optimised_queryset(queryset: models.QuerySet, filters: Dict[str, Any] = None) -> models.QuerySet:
    """Apply performance optimisations to querysets."""
    
    # Apply select_related for common foreign keys
    if hasattr(queryset.model, 'stock'):
        queryset = queryset.select_related('stock')
    if hasattr(queryset.model, 'portfolio'):
        queryset = queryset.select_related('portfolio', 'portfolio__user')
    if hasattr(queryset.model, 'user'):
        queryset = queryset.select_related('user')
    
    # Apply filters efficiently
    if filters:
        # Apply date filters first for better index usage
        if 'date_from' in filters:
            queryset = queryset.filter(date__gte=filters['date_from'])
        if 'date_to' in filters:
            queryset = queryset.filter(date__lte=filters['date_to'])
        
        # Apply other filters
        for key, value in filters.items():
            if key not in ['date_from', 'date_to'] and value is not None:
                queryset = queryset.filter(**{key: value})
    
    return queryset


def paginate_queryset(queryset: models.QuerySet, 
                     request,
                     pagination_class: type = OptimisedPageNumberPagination) -> tuple:
    """Helper function to paginate querysets efficiently."""
    
    paginator = pagination_class()
    page = paginator.paginate_queryset(queryset, request)
    
    return page, paginator