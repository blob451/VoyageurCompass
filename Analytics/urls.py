"""
URL configuration for Analytics app.
"""

from django.urls import path
from Analytics.views import (
    analyze_stock,
    analyze_portfolio,
    batch_analysis,
    get_user_analysis_history,
    get_user_latest_analysis,
    get_analysis_by_id
)
from Analytics.sentiment_views import stock_sentiment
from Analytics.explanation_views import (
    generate_explanation,
    explanation_service_status,
    get_explanation
)
from Analytics.async_views import (
    batch_analyze,
    batch_explain,
    get_task_status,
    get_batch_status,
    async_performance
)
from Analytics.finetuning_views import (
    generate_dataset,
    start_finetuning,
    get_job_status as get_finetuning_job_status,
    list_jobs,
    list_datasets,
    list_models,
    export_dataset,
    finetuning_status
)
from Analytics.monitoring_views import (
    system_health,
    performance_dashboard,
    metric_history,
    recent_alerts,
    performance_profiles,
    available_metrics,
    record_metric,
    monitoring_status
)
from Analytics.quality_views import (
    quality_dashboard,
    analyze_project,
    analyze_file,
    quality_metrics,
    quality_recommendations,
    quality_service_status
)

app_name = 'analytics'

urlpatterns = [
    path('analyze/<str:symbol>/', analyze_stock, name='analyze_stock'),
    path('stock-analysis/<str:symbol>/', analyze_stock, name='stock-analysis'),  # Alternative name for API consistency
    path('analyze-portfolio/<int:portfolio_id>/', analyze_portfolio, name='analyze_portfolio'),
    path('batch-analysis/', batch_analysis, name='batch_analysis'),
    path('user/history/', get_user_analysis_history, name='user_analysis_history'),
    path('user/<str:symbol>/latest/', get_user_latest_analysis, name='user_latest_analysis'),
    path('analysis/<int:analysis_id>/', get_analysis_by_id, name='get_analysis_by_id'),
    path('sentiment/<str:symbol>/', stock_sentiment, name='stock_sentiment'),
    
    # Explanation endpoints
    path('explain/<int:analysis_id>/', generate_explanation, name='generate_explanation'),
    path('explain/status/', explanation_service_status, name='explain_status'),
    path('explanation/<int:analysis_id>/', get_explanation, name='get_explanation'),
    path('explanation-status/', explanation_service_status, name='explanation_status'),
    
    # Enhanced LLM endpoints
    path('llm/explain/', generate_explanation, name='llm_explain'),
    path('llm/status/', explanation_service_status, name='llm_status'),
    
    # Async Processing endpoints
    path('async/batch-analyze/', batch_analyze, name='batch_analyze'),
    path('async/batch-explain/', batch_explain, name='batch_explain'),
    path('async/task/<str:task_id>/', get_task_status, name='get_task_status'),
    path('async/batch/<str:batch_id>/', get_batch_status, name='get_batch_status'),
    path('async/performance/', async_performance, name='async_performance'),
    
    # Fine-Tuning endpoints
    path('finetuning/generate-dataset/', generate_dataset, name='generate_dataset'),
    path('finetuning/start/', start_finetuning, name='start_finetuning'),
    path('finetuning/job/<str:job_id>/', get_finetuning_job_status, name='get_finetuning_job_status'),
    path('finetuning/jobs/', list_jobs, name='list_finetuning_jobs'),
    path('finetuning/datasets/', list_datasets, name='list_datasets'),
    path('finetuning/models/', list_models, name='list_models'),
    path('finetuning/export/', export_dataset, name='export_dataset'),
    path('finetuning/status/', finetuning_status, name='finetuning_status'),
    
    # Monitoring endpoints
    path('monitoring/health/', system_health, name='system_health'),
    path('monitoring/dashboard/', performance_dashboard, name='performance_dashboard'),
    path('monitoring/metrics/history/', metric_history, name='metric_history'),
    path('monitoring/alerts/', recent_alerts, name='recent_alerts'),
    path('monitoring/profiles/', performance_profiles, name='performance_profiles'),
    path('monitoring/metrics/', available_metrics, name='available_metrics'),
    path('monitoring/record-metric/', record_metric, name='record_metric'),
    path('monitoring/status/', monitoring_status, name='monitoring_status'),
    
    # Code Quality endpoints
    path('quality/dashboard/', quality_dashboard, name='quality_dashboard'),
    path('quality/analyze/', analyze_project, name='analyze_project'),
    path('quality/analyze-file/', analyze_file, name='analyze_file'),
    path('quality/metrics/', quality_metrics, name='quality_metrics'),
    path('quality/recommendations/', quality_recommendations, name='quality_recommendations'),
    path('quality/status/', quality_service_status, name='quality_service_status'),
]