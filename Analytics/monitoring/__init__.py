"""
Analytics Monitoring Module

Provides comprehensive observability and performance monitoring capabilities
for the VoyageurCompass Analytics platform.
"""

from .metrics import metrics_collector, get_metrics_collector, timed_metric

__all__ = [
    'metrics_collector',
    'get_metrics_collector', 
    'timed_metric'
]