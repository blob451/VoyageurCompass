"""
Code Quality Views for Analytics app.
Provides endpoints for code quality analysis and reporting.
"""

import logging
import os
from datetime import datetime

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response

from Analytics.services.code_quality_service import get_code_quality_service

logger = logging.getLogger(__name__)


@extend_schema(
    summary="Get code quality dashboard",
    description="Get comprehensive code quality dashboard with metrics and recommendations",
)
@api_view(["GET"])
@permission_classes([IsAdminUser])
def quality_dashboard(request):
    """
    Get code quality dashboard (admin only).
    """
    try:
        quality_service = get_code_quality_service()
        dashboard_data = quality_service.get_quality_dashboard()

        return Response(dashboard_data)

    except Exception as e:
        logger.error(f"Error getting quality dashboard: {str(e)}")
        return Response(
            {"error": f"Failed to get quality dashboard: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Analyze project code quality",
    description="Run complete code quality analysis on the project (admin only)",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "include_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'File patterns to include (default: ["**/*.py"])',
                },
                "export_report": {
                    "type": "boolean",
                    "description": "Whether to export detailed report",
                    "default": False,
                },
            },
        }
    },
)
@api_view(["POST"])
@permission_classes([IsAdminUser])
def analyze_project(request):
    """
    Run complete project analysis (admin only).
    """
    try:
        include_patterns = request.data.get("include_patterns", ["**/*.py"])
        export_report = request.data.get("export_report", False)

        quality_service = get_code_quality_service()

        logger.info(f"Starting project analysis requested by {request.user}")
        analysis_results = quality_service.analyze_project(include_patterns)

        # Export detailed report if requested
        export_path = None
        if export_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"./code_quality_report_{timestamp}.json"
            quality_service.export_analysis_report(analysis_results, export_path)

        # Return summary (not full detailed results to avoid large response)
        summary_response = {
            "analysis_summary": {
                "files_analyzed": analysis_results["files_analyzed"],
                "total_issues": analysis_results["total_issues"],
                "average_quality_score": analysis_results["average_quality_score"],
                "analyzed_at": analysis_results["analyzed_at"],
            },
            "issue_summary": analysis_results["summary"],
            "export_path": export_path,
            "analyzed_by": request.user.username,
        }

        return Response(summary_response)

    except Exception as e:
        logger.error(f"Error analyzing project: {str(e)}")
        return Response({"error": f"Project analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    summary="Analyze specific file",
    description="Analyze code quality of a specific file",
    parameters=[
        OpenApiParameter(
            name="file_path",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.QUERY,
            required=True,
            description="Path to the file to analyze (relative to project root)",
        )
    ],
)
@api_view(["GET"])
@permission_classes([IsAdminUser])
def analyze_file(request):
    """
    Analyze specific file quality (admin only).
    """
    try:
        file_path = request.GET.get("file_path")

        if not file_path:
            return Response({"error": "file_path parameter is required"}, status=status.HTTP_400_BAD_REQUEST)

        quality_service = get_code_quality_service()

        # Construct full path
        full_path = os.path.join(quality_service.project_root, file_path)

        if not os.path.exists(full_path):
            return Response({"error": f"File not found: {file_path}"}, status=status.HTTP_404_NOT_FOUND)

        if not full_path.endswith(".py"):
            return Response({"error": "Only Python files can be analyzed"}, status=status.HTTP_400_BAD_REQUEST)

        analysis_result = quality_service.get_quality_report(full_path)

        return Response({"file_analysis": analysis_result, "analyzed_by": request.user.username})

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        return Response({"error": f"File analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(summary="Get quality metrics", description="Get overall code quality metrics for the project")
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def quality_metrics(request):
    """
    Get code quality metrics (authenticated users).
    """
    try:
        quality_service = get_code_quality_service()

        # Run quick analysis for metrics
        project_analysis = quality_service.analyze_project(["Analytics/**/*.py"])  # Focus on Analytics

        # Extract key metrics
        metrics = {
            "overall_quality_score": project_analysis["average_quality_score"],
            "files_analyzed": project_analysis["files_analyzed"],
            "total_issues": project_analysis["total_issues"],
            "issue_distribution": project_analysis["summary"]["severity_counts"],
            "top_issue_types": project_analysis["summary"]["top_issues"][:3],
            "quality_level": _get_quality_level(project_analysis["average_quality_score"]),
            "last_updated": project_analysis["analyzed_at"],
        }

        return Response(metrics)

    except Exception as e:
        logger.error(f"Error getting quality metrics: {str(e)}")
        return Response(
            {"error": f"Failed to get quality metrics: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(summary="Get quality recommendations", description="Get code quality improvement recommendations")
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def quality_recommendations(request):
    """
    Get code quality recommendations.
    """
    try:
        quality_service = get_code_quality_service()

        # Get dashboard data which includes recommendations
        dashboard_data = quality_service.get_quality_dashboard()

        recommendations_response = {
            "recommendations": dashboard_data["recommendations"],
            "quality_score": dashboard_data["current_quality_score"],
            "priority_areas": _get_priority_areas(dashboard_data),
            "generated_at": dashboard_data["generated_at"],
        }

        return Response(recommendations_response)

    except Exception as e:
        logger.error(f"Error getting quality recommendations: {str(e)}")
        return Response(
            {"error": f"Failed to get recommendations: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get quality service status", description="Get status and configuration of the code quality service"
)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def quality_service_status(request):
    """
    Get quality service status.
    """
    try:
        quality_service = get_code_quality_service()

        return Response(
            {
                "service_status": "active",
                "project_root": str(quality_service.project_root),
                "analyzer_config": quality_service.analyzer.quality_rules,
                "capabilities": [
                    "file_analysis",
                    "project_analysis",
                    "quality_metrics",
                    "issue_detection",
                    "improvement_recommendations",
                    "quality_scoring",
                    "report_export",
                ],
                "supported_file_types": [".py"],
                "analysis_features": {
                    "complexity_analysis": True,
                    "docstring_coverage": True,
                    "code_style_checks": True,
                    "pattern_detection": True,
                    "quality_scoring": True,
                },
            }
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to get service status: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def _get_quality_level(score: float) -> str:
    """Get quality level description from score."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Needs Improvement"


def _get_priority_areas(dashboard_data: dict) -> list[str]:
    """Extract priority areas from dashboard data."""
    priority_areas = []

    issue_distribution = dashboard_data.get("issue_distribution", {})
    top_issues = dashboard_data.get("top_issue_types", [])

    # High priority if many critical errors
    if issue_distribution.get("error", 0) > 5:
        priority_areas.append("Critical Error Resolution")

    # Check top issue types
    if top_issues:
        top_issue_type = top_issues[0][0] if top_issues[0][1] > 10 else None

        if top_issue_type:
            if "complexity" in top_issue_type:
                priority_areas.append("Code Complexity Reduction")
            elif "docstring" in top_issue_type:
                priority_areas.append("Documentation Improvement")
            elif "line_too_long" in top_issue_type:
                priority_areas.append("Code Formatting")
            elif "function_too_long" in top_issue_type:
                priority_areas.append("Function Refactoring")

    if not priority_areas:
        priority_areas = ["General Code Quality Maintenance"]

    return priority_areas
