"""
URL configuration for Analytics app.
"""

from django.urls import path

from Analytics.async_views import (
    async_performance,
    batch_analyze,
    batch_explain,
    get_batch_status,
    get_task_status,
)
from Analytics.explanation_views import (
    explanation_service_status,
    generate_explanation,
    get_explanation,
)
from Analytics.finetuning_views import (
    export_dataset,
    finetuning_status,
    generate_dataset,
)
from Analytics.finetuning_views import get_job_status as get_finetuning_job_status
from Analytics.finetuning_views import (
    list_datasets,
    list_jobs,
    list_models,
    start_finetuning,
)
from Analytics.monitoring_views import (
    health_metrics,
    metrics_endpoint,
)
from Analytics.quality_views import (
    analyze_file,
    analyze_project,
    quality_dashboard,
    quality_metrics,
    quality_recommendations,
    quality_service_status,
)
from Analytics.sentiment_views import stock_sentiment
from Analytics.status_views import check_analysis_status
from Analytics.views import (
    analyze_portfolio,
    analyze_stock,
    batch_analysis,
    finbert_model_status,
    get_analysis_by_id,
    get_user_analysis_history,
    get_user_latest_analysis,
)
from Analytics.views_batch import (
    analyze_portfolio_batch,
    batch_analyze_stocks,
    get_batch_performance_stats,
    warm_analysis_cache,
)

app_name = "analytics"

urlpatterns = [
    path("analyze/<str:symbol>/", analyze_stock, name="analyze_stock"),
    path("stock-analysis/<str:symbol>/", analyze_stock, name="stock-analysis"),  # Alternative name for API consistency
    path("analyze-portfolio/<int:portfolio_id>/", analyze_portfolio, name="analyze_portfolio"),
    path("batch-analysis/", batch_analysis, name="batch_analysis"),
    # High-performance batch endpoints
    path("batch/analyze/", batch_analyze_stocks, name="batch_analyze_stocks"),
    path("batch/portfolio/", analyze_portfolio_batch, name="analyze_portfolio_batch"),
    path("batch/portfolio/<int:portfolio_id>/", analyze_portfolio_batch, name="analyze_portfolio_batch_id"),
    path("batch/stats/", get_batch_performance_stats, name="batch_performance_stats"),
    path("batch/warm-cache/", warm_analysis_cache, name="warm_analysis_cache"),
    path("user/history/", get_user_analysis_history, name="user_analysis_history"),
    path("user/<str:symbol>/latest/", get_user_latest_analysis, name="user_latest_analysis"),
    path("analysis/<int:analysis_id>/", get_analysis_by_id, name="get_analysis_by_id"),
    path("sentiment/<str:symbol>/", stock_sentiment, name="stock_sentiment"),
    path("status/<str:symbol>/", check_analysis_status, name="check_analysis_status"),
    path("model/finbert-status/", finbert_model_status, name="finbert_model_status"),
    # Explanation endpoints
    path("explain/<int:analysis_id>/", generate_explanation, name="generate_explanation"),
    path("explain/status/", explanation_service_status, name="explain_status"),
    path("explanation/<int:analysis_id>/", get_explanation, name="get_explanation"),
    path("explanation-status/", explanation_service_status, name="explanation_status"),
    # Enhanced LLM endpoints
    path("llm/explain/", generate_explanation, name="llm_explain"),
    path("llm/status/", explanation_service_status, name="llm_status"),
    # Async Processing endpoints
    path("async/batch-analyze/", batch_analyze, name="batch_analyze"),
    path("async/batch-explain/", batch_explain, name="batch_explain"),
    path("async/task/<str:task_id>/", get_task_status, name="get_task_status"),
    path("async/batch/<str:batch_id>/", get_batch_status, name="get_batch_status"),
    path("async/performance/", async_performance, name="async_performance"),
    # Fine-Tuning endpoints
    path("finetuning/generate-dataset/", generate_dataset, name="generate_dataset"),
    path("finetuning/start/", start_finetuning, name="start_finetuning"),
    path("finetuning/job/<str:job_id>/", get_finetuning_job_status, name="get_finetuning_job_status"),
    path("finetuning/jobs/", list_jobs, name="list_finetuning_jobs"),
    path("finetuning/datasets/", list_datasets, name="list_datasets"),
    path("finetuning/models/", list_models, name="list_models"),
    path("finetuning/export/", export_dataset, name="export_dataset"),
    path("finetuning/status/", finetuning_status, name="finetuning_status"),
    # Monitoring endpoints
    path("monitoring/health/", health_metrics, name="system_health"),
    # Code Quality endpoints
    path("quality/dashboard/", quality_dashboard, name="quality_dashboard"),
    path("quality/analyze/", analyze_project, name="analyze_project"),
    path("quality/analyze-file/", analyze_file, name="analyze_file"),
    path("quality/metrics/", quality_metrics, name="quality_metrics"),
    path("quality/recommendations/", quality_recommendations, name="quality_recommendations"),
    path("quality/status/", quality_service_status, name="quality_service_status"),
    # Prometheus metrics endpoints
    path("metrics/", metrics_endpoint, name="prometheus_metrics"),
    path("health/metrics/", health_metrics, name="health_metrics"),
]
