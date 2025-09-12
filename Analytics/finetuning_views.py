"""
Fine-tuning management views for financial analytics.
Provides endpoints for dataset generation and model training orchestration.
"""

import logging
from datetime import datetime

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from Analytics.services.enhanced_finetuning_service import get_finetuning_manager

logger = logging.getLogger(__name__)


@extend_schema(
    summary="Generate enhanced training dataset",
    description="Generate high-quality training dataset with sentiment-enhanced explanations",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "num_samples": {"type": "integer", "description": "Number of samples to generate", "default": 1000},
                "include_sentiment": {
                    "type": "boolean",
                    "description": "Whether to include sentiment analysis",
                    "default": True,
                },
                "quality_threshold": {
                    "type": "number",
                    "description": "Minimum quality threshold (0.0-1.0)",
                    "default": 0.7,
                },
            },
        }
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_dataset(request):
    """
    Generate enhanced training dataset for model fine-tuning.
    """
    try:
        num_samples = request.data.get("num_samples", 1000)
        include_sentiment = request.data.get("include_sentiment", True)
        quality_threshold = request.data.get("quality_threshold", 0.7)

        # Validate parameters
        if not isinstance(num_samples, int) or num_samples < 1 or num_samples > 10000:
            return Response({"error": "num_samples must be between 1 and 10000"}, status=status.HTTP_400_BAD_REQUEST)

        if not isinstance(quality_threshold, (int, float)) or not 0 <= quality_threshold <= 1:
            return Response(
                {"error": "quality_threshold must be between 0.0 and 1.0"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get fine-tuning manager
        finetuning_manager = get_finetuning_manager()

        # Generate dataset
        logger.info(f"User {request.user} requested dataset generation: {num_samples} samples")

        dataset_metadata = finetuning_manager.generate_enhanced_dataset(
            num_samples=num_samples, include_sentiment=include_sentiment, quality_threshold=quality_threshold
        )

        return Response(
            {
                "message": "Dataset generated successfully",
                "dataset_metadata": dataset_metadata,
                "generated_by": request.user.username,
                "generated_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Dataset generation failed for user {request.user}: {str(e)}")
        return Response({"error": f"Dataset generation failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Start fine-tuning job",
    description="Start a fine-tuning job using specified dataset",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string", "description": "Path to training dataset"},
                "job_name": {"type": "string", "description": "Name for the fine-tuning job"},
                "config_overrides": {"type": "object", "description": "Training configuration overrides"},
            },
            "required": ["dataset_path"],
        }
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def start_finetuning(request):
    """
    Start a fine-tuning job.
    """
    try:
        dataset_path = request.data.get("dataset_path")
        job_name = request.data.get("job_name")
        config_overrides = request.data.get("config_overrides", {})

        if not dataset_path:
            return Response({"error": "dataset_path is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Get fine-tuning manager
        finetuning_manager = get_finetuning_manager()

        # Start fine-tuning job
        logger.info(f"User {request.user} started fine-tuning job: {job_name or 'auto-generated'}")

        job_info = finetuning_manager.start_fine_tuning_job(
            dataset_path=dataset_path, job_name=job_name, config_overrides=config_overrides
        )

        return Response(
            {
                "message": "Fine-tuning job started successfully",
                "job_info": job_info,
                "started_by": request.user.username,
            }
        )

    except Exception as e:
        logger.error(f"Fine-tuning job start failed for user {request.user}: {str(e)}")
        return Response(
            {"error": f"Failed to start fine-tuning job: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get fine-tuning job status",
    description="Get the status and details of a specific fine-tuning job",
    parameters=[
        OpenApiParameter(
            name="job_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            required=True,
            description="Fine-tuning job ID",
        )
    ],
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_job_status(request, job_id):
    """
    Get status of a fine-tuning job.
    """
    try:
        finetuning_manager = get_finetuning_manager()
        job_status = finetuning_manager.get_job_status(job_id)

        if not job_status:
            return Response({"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND)

        return Response(job_status)

    except Exception as e:
        return Response({"error": f"Failed to get job status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(summary="List fine-tuning jobs", description="List all fine-tuning jobs with their status")
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_jobs(request):
    """
    List all fine-tuning jobs.
    """
    try:
        finetuning_manager = get_finetuning_manager()
        jobs_info = finetuning_manager.list_jobs()

        return Response(jobs_info)

    except Exception as e:
        return Response({"error": f"Failed to list jobs: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(summary="List available datasets", description="List all available training datasets")
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_datasets(request):
    """
    List available training datasets.
    """
    try:
        finetuning_manager = get_finetuning_manager()
        datasets = finetuning_manager.list_datasets()

        return Response({"datasets": datasets, "total_count": len(datasets)})

    except Exception as e:
        return Response({"error": f"Failed to list datasets: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(summary="List fine-tuned models", description="List all available fine-tuned models")
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_models(request):
    """
    List available fine-tuned models.
    """
    try:
        finetuning_manager = get_finetuning_manager()
        models = finetuning_manager.list_models()

        return Response({"models": models, "total_count": len(models)})

    except Exception as e:
        return Response({"error": f"Failed to list models: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Export dataset for external training",
    description="Export dataset in format suitable for external fine-tuning platforms",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string", "description": "Path to source dataset"},
                "format": {
                    "type": "string",
                    "enum": ["jsonl", "csv", "huggingface"],
                    "description": "Export format",
                    "default": "jsonl",
                },
            },
            "required": ["dataset_path"],
        }
    },
)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def export_dataset(request):
    """
    Export dataset for external fine-tuning platforms.
    """
    try:
        dataset_path = request.data.get("dataset_path")
        export_format = request.data.get("format", "jsonl")

        if not dataset_path:
            return Response({"error": "dataset_path is required"}, status=status.HTTP_400_BAD_REQUEST)

        if export_format not in ["jsonl", "csv", "huggingface"]:
            return Response(
                {"error": "format must be one of: jsonl, csv, huggingface"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get fine-tuning manager
        finetuning_manager = get_finetuning_manager()

        # Export dataset
        export_path = finetuning_manager.export_dataset_for_external_training(
            dataset_path=dataset_path, format=export_format
        )

        return Response(
            {
                "message": "Dataset exported successfully",
                "export_path": export_path,
                "format": export_format,
                "exported_by": request.user.username,
                "exported_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Dataset export failed for user {request.user}: {str(e)}")
        return Response({"error": f"Dataset export failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Get fine-tuning system status", description="Get overall status and capabilities of the fine-tuning system"
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def finetuning_status(request):
    """
    Get fine-tuning system status and capabilities.
    """
    try:
        from Analytics.services.financial_fine_tuner import FINE_TUNING_AVAILABLE

        finetuning_manager = get_finetuning_manager()
        jobs_info = finetuning_manager.list_jobs()
        datasets = finetuning_manager.list_datasets()
        models = finetuning_manager.list_models()

        return Response(
            {
                "fine_tuning_available": FINE_TUNING_AVAILABLE,
                "dependencies_installed": FINE_TUNING_AVAILABLE,
                "dataset_directory": str(finetuning_manager.dataset_dir),
                "model_directory": str(finetuning_manager.model_dir),
                "total_jobs": jobs_info["total_jobs"],
                "active_jobs": jobs_info["active_jobs"],
                "completed_jobs": jobs_info["completed_jobs"],
                "failed_jobs": jobs_info["failed_jobs"],
                "total_datasets": len(datasets),
                "total_models": len(models),
                "system_ready": True,
                "capabilities": {
                    "dataset_generation": True,
                    "sentiment_enhanced_training": True,
                    "quality_filtering": True,
                    "export_formats": ["jsonl", "csv", "huggingface"],
                    "lora_fine_tuning": FINE_TUNING_AVAILABLE,
                    "model_evaluation": FINE_TUNING_AVAILABLE,
                },
                "required_dependencies": (
                    ["transformers", "peft", "trl", "datasets", "wandb", "accelerate", "bitsandbytes"]
                    if not FINE_TUNING_AVAILABLE
                    else None
                ),
            }
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to get system status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
