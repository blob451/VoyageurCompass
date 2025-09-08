"""
Code Quality Enhancement Service for VoyageurCompass.
Provides code analysis, quality metrics, and improvement suggestions.
"""

import ast
import logging
import re
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzes Python code for quality metrics and issues."""

    def __init__(self):
        self.quality_rules = {
            'max_function_length': 50,
            'max_class_length': 300,
            'max_complexity': 10,
            'min_docstring_coverage': 0.8,
            'max_line_length': 100,
            'forbidden_patterns': [
                r'print\(',  # Should use logging
                r'TODO:',    # Should be tracked properly
                r'FIXME:',   # Should be tracked properly
                r'XXX:'      # Should be tracked properly
            ]
        }

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file for quality metrics.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=file_path)

            # Analyze different aspects
            analysis = {
                'file_path': file_path,
                'analyzed_at': datetime.now().isoformat(),
                'metrics': self._calculate_metrics(tree, content),
                'issues': self._find_issues(tree, content),
                'suggestions': [],
                'quality_score': 0.0
            }

            # Generate suggestions based on issues
            analysis['suggestions'] = self._generate_suggestions(analysis['issues'])

            # Calculate overall quality score
            analysis['quality_score'] = self._calculate_quality_score(analysis['metrics'], analysis['issues'])

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {
                'file_path': file_path,
                'error': str(e),
                'analyzed_at': datetime.now().isoformat(),
                'quality_score': 0.0
            }

    def _calculate_metrics(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Calculate code metrics."""
        metrics = {
            'lines_of_code': len(content.splitlines()),
            'functions': 0,
            'classes': 0,
            'imports': 0,
            'docstring_coverage': 0.0,
            'avg_function_length': 0.0,
            'avg_complexity': 0.0,
            'max_function_length': 0,
            'max_complexity': 0
        }

        function_lengths = []
        complexities = []
        docstring_count = 0
        total_functions = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['functions'] += 1
                total_functions += 1

                # Calculate function length
                func_length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 10
                function_lengths.append(func_length)
                metrics['max_function_length'] = max(metrics['max_function_length'], func_length)

                # Check for docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    docstring_count += 1

                # Calculate complexity (simplified)
                complexity = self._calculate_function_complexity(node)
                complexities.append(complexity)
                metrics['max_complexity'] = max(metrics['max_complexity'], complexity)

            elif isinstance(node, ast.ClassDef):
                metrics['classes'] += 1

                # Check for class docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Constant) and 
                    isinstance(node.body[0].value.value, str)):
                    docstring_count += 1
                    total_functions += 1  # Count classes in docstring coverage

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['imports'] += 1

        # Calculate averages
        if function_lengths:
            metrics['avg_function_length'] = sum(function_lengths) / len(function_lengths)

        if complexities:
            metrics['avg_complexity'] = sum(complexities) / len(complexities)

        if total_functions > 0:
            metrics['docstring_coverage'] = docstring_count / total_functions

        return metrics

    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function (simplified)."""
        complexity = 1  # Base complexity

        for node in ast.walk(func_node):
            # Decision points increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1

        return complexity

    def _find_issues(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Find code quality issues."""
        issues = []
        lines = content.splitlines()

        # Check line lengths
        for i, line in enumerate(lines):
            if len(line) > self.quality_rules['max_line_length']:
                issues.append({
                    'type': 'line_too_long',
                    'severity': 'warning',
                    'line': i + 1,
                    'message': f'Line too long ({len(line)} > {self.quality_rules["max_line_length"]})',
                    'suggestion': 'Consider breaking long lines for better readability'
                })

        # Check for forbidden patterns
        for pattern in self.quality_rules['forbidden_patterns']:
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    issues.append({
                        'type': 'forbidden_pattern',
                        'severity': 'warning',
                        'line': i + 1,
                        'message': f'Found forbidden pattern: {pattern}',
                        'suggestion': self._get_pattern_suggestion(pattern)
                    })

        # AST-based checks
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function length
                func_length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 10
                if func_length > self.quality_rules['max_function_length']:
                    issues.append({
                        'type': 'function_too_long',
                        'severity': 'warning',
                        'line': node.lineno,
                        'function': node.name,
                        'message': f'Function too long ({func_length} > {self.quality_rules["max_function_length"]})',
                        'suggestion': 'Consider breaking into smaller functions'
                    })

                # Check complexity
                complexity = self._calculate_function_complexity(node)
                if complexity > self.quality_rules['max_complexity']:
                    issues.append({
                        'type': 'high_complexity',
                        'severity': 'error',
                        'line': node.lineno,
                        'function': node.name,
                        'message': f'High complexity ({complexity} > {self.quality_rules["max_complexity"]})',
                        'suggestion': 'Refactor to reduce complexity'
                    })

                # Check for missing docstring
                if not (node.body and isinstance(node.body[0], ast.Expr) and 
                       isinstance(node.body[0].value, ast.Constant)):
                    issues.append({
                        'type': 'missing_docstring',
                        'severity': 'info',
                        'line': node.lineno,
                        'function': node.name,
                        'message': 'Function missing docstring',
                        'suggestion': 'Add docstring to document function purpose and parameters'
                    })

        return issues

    def _get_pattern_suggestion(self, pattern: str) -> str:
        """Get suggestion for forbidden pattern."""
        suggestions = {
            r'print\(': 'Use logging instead of print statements',
            r'TODO:': 'Move TODO items to issue tracker',
            r'FIXME:': 'Move FIXME items to issue tracker',
            r'XXX:': 'Move XXX items to issue tracker'
        }
        return suggestions.get(pattern, 'Remove or refactor this pattern')

    def _generate_suggestions(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        issue_counts = {}

        # Count issue types
        for issue in issues:
            issue_type = issue['type']
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # Generate suggestions based on issue patterns
        if issue_counts.get('line_too_long', 0) > 5:
            suggestions.append('Consider using a code formatter like Black to maintain consistent line lengths')

        if issue_counts.get('missing_docstring', 0) > 3:
            suggestions.append('Implement docstring standards across the codebase for better documentation')

        if issue_counts.get('high_complexity', 0) > 0:
            suggestions.append('Refactor complex functions to improve maintainability and testability')

        if issue_counts.get('function_too_long', 0) > 2:
            suggestions.append('Break down large functions into smaller, focused functions')

        return suggestions

    def _calculate_quality_score(self, metrics: Dict[str, Any], issues: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0

        # Deduct points for issues
        for issue in issues:
            if issue['severity'] == 'error':
                score -= 10
            elif issue['severity'] == 'warning':
                score -= 5
            elif issue['severity'] == 'info':
                score -= 2

        # Deduct points for poor metrics
        if metrics['docstring_coverage'] < self.quality_rules['min_docstring_coverage']:
            score -= 15

        if metrics['avg_complexity'] > self.quality_rules['max_complexity'] * 0.7:
            score -= 10

        if metrics['avg_function_length'] > self.quality_rules['max_function_length'] * 0.8:
            score -= 10

        return max(0.0, score)


class CodeQualityService:
    """Main service for code quality analysis and enhancement."""

    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.project_root = Path(__file__).parent.parent.parent  # VoyageurCompass root

    def analyze_project(self, include_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Analyze entire project for code quality.

        Args:
            include_patterns: Patterns of files to include (default: ['**/*.py'])

        Returns:
            Project analysis results
        """
        if include_patterns is None:
            include_patterns = ['**/*.py']

        analysis_results = {
            'project_root': str(self.project_root),
            'analyzed_at': datetime.now().isoformat(),
            'files_analyzed': 0,
            'total_issues': 0,
            'average_quality_score': 0.0,
            'file_results': [],
            'summary': {
                'issue_types': {},
                'severity_counts': {'error': 0, 'warning': 0, 'info': 0},
                'top_issues': [],
                'recommendations': []
            }
        }

        # Find Python files to analyze
        python_files = []
        for pattern in include_patterns:
            python_files.extend(self.project_root.glob(pattern))

        # Filter out __pycache__ and migration files
        python_files = [f for f in python_files if 
                       '__pycache__' not in str(f) and 
                       'migrations' not in str(f) and
                       f.is_file()]

        total_score = 0.0
        issue_type_counts = {}
        all_issues = []

        logger.info(f"Analyzing {len(python_files)} Python files...")

        for file_path in python_files:
            try:
                # Analyze individual file
                file_analysis = self.analyzer.analyze_file(str(file_path))
                analysis_results['file_results'].append(file_analysis)
                analysis_results['files_analyzed'] += 1

                if 'error' not in file_analysis:
                    # Accumulate metrics
                    total_score += file_analysis['quality_score']
                    analysis_results['total_issues'] += len(file_analysis['issues'])

                    # Count issue types
                    for issue in file_analysis['issues']:
                        issue_type = issue['type']
                        severity = issue['severity']

                        issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
                        analysis_results['summary']['severity_counts'][severity] += 1

                        all_issues.append(issue)

            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {str(e)}")

        # Calculate averages and summaries
        if analysis_results['files_analyzed'] > 0:
            analysis_results['average_quality_score'] = total_score / analysis_results['files_analyzed']

        # Top issue types
        analysis_results['summary']['issue_types'] = issue_type_counts
        analysis_results['summary']['top_issues'] = sorted(
            issue_type_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]

        # Generate project-level recommendations
        analysis_results['summary']['recommendations'] = self._generate_project_recommendations(
            analysis_results
        )

        logger.info(f"Project analysis complete: {analysis_results['files_analyzed']} files, "
                   f"{analysis_results['total_issues']} issues, "
                   f"avg quality score: {analysis_results['average_quality_score']:.1f}")

        return analysis_results

    def _generate_project_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate project-level recommendations."""
        recommendations = []
        issue_types = analysis_results['summary']['issue_types']
        severity_counts = analysis_results['summary']['severity_counts']
        avg_score = analysis_results['average_quality_score']

        # Overall quality recommendations
        if avg_score < 60:
            recommendations.append("🔴 CRITICAL: Overall code quality is low. Prioritize immediate refactoring.")
        elif avg_score < 75:
            recommendations.append("🟡 WARNING: Code quality needs improvement. Focus on high-impact issues.")
        elif avg_score >= 90:
            recommendations.append("✅ EXCELLENT: Code quality is high. Maintain current standards.")

        # Specific issue recommendations
        if issue_types.get('high_complexity', 0) > 5:
            recommendations.append("🔧 Refactor complex functions to improve maintainability")

        if issue_types.get('missing_docstring', 0) > 20:
            recommendations.append("📝 Implement docstring standards to improve documentation coverage")

        if issue_types.get('line_too_long', 0) > 50:
            recommendations.append("🎨 Use code formatter (Black/autopep8) to maintain consistent formatting")

        if issue_types.get('function_too_long', 0) > 10:
            recommendations.append("✂️ Break down large functions into smaller, focused functions")

        if severity_counts['error'] > 0:
            recommendations.append(f"❌ Address {severity_counts['error']} critical errors immediately")

        # Best practices recommendations
        if issue_types.get('forbidden_pattern', 0) > 10:
            recommendations.append("🚫 Remove debugging code and implement proper logging")

        return recommendations

    def get_quality_report(self, file_path: str = None) -> Dict[str, Any]:
        """
        Get quality report for specific file or entire project.

        Args:
            file_path: Specific file to analyze (None for entire project)

        Returns:
            Quality report
        """
        if file_path:
            # Single file analysis
            return self.analyzer.analyze_file(file_path)
        else:
            # Full project analysis
            return self.analyze_project()

    def get_quality_dashboard(self) -> Dict[str, Any]:
        """Get quality dashboard data."""
        project_analysis = self.analyze_project()

        # Calculate trend data (would need historical data in production)
        dashboard = {
            'current_quality_score': project_analysis['average_quality_score'],
            'total_files_analyzed': project_analysis['files_analyzed'],
            'total_issues': project_analysis['total_issues'],
            'issue_distribution': project_analysis['summary']['severity_counts'],
            'top_issue_types': project_analysis['summary']['top_issues'][:5],
            'recommendations': project_analysis['summary']['recommendations'][:3],
            'quality_trend': {
                'current': project_analysis['average_quality_score'],
                'previous': 0,  # Would need historical data
                'change': 0
            },
            'files_by_quality': self._categorize_files_by_quality(project_analysis['file_results']),
            'generated_at': datetime.now().isoformat()
        }

        return dashboard

    def _categorize_files_by_quality(self, file_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize files by quality score."""
        categories = {
            'excellent': [],   # 90+
            'good': [],       # 75-89
            'fair': [],       # 60-74
            'poor': []        # <60
        }

        for file_result in file_results:
            if 'error' in file_result:
                categories['poor'].append(file_result['file_path'])
                continue

            score = file_result['quality_score']
            file_path = Path(file_result['file_path']).name  # Just filename

            if score >= 90:
                categories['excellent'].append(file_path)
            elif score >= 75:
                categories['good'].append(file_path)
            elif score >= 60:
                categories['fair'].append(file_path)
            else:
                categories['poor'].append(file_path)

        return categories

    def export_analysis_report(self, analysis_results: Dict[str, Any], output_path: str):
        """Export analysis results to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis report exported to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export analysis report: {str(e)}")
            raise


# Singleton instance
_code_quality_service = None


def get_code_quality_service() -> CodeQualityService:
    """Get singleton instance of CodeQualityService."""
    global _code_quality_service
    if _code_quality_service is None:
        _code_quality_service = CodeQualityService()
    return _code_quality_service
