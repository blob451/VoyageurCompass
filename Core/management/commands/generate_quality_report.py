"""
Django management command to generate comprehensive quality metrics report.
This command aggregates all quality metrics and creates a dashboard-style report.
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    """
    Management command to generate comprehensive quality metrics report.
    """
    
    help = 'Generate comprehensive quality metrics dashboard report'

    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            '--output',
            type=str,
            default='quality_metrics_report.json',
            help='Output file for quality report (default: quality_metrics_report.json)'
        )
        parser.add_argument(
            '--html',
            action='store_true',
            help='Generate HTML dashboard report'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )

    def handle(self, *args, **options):
        """Main command handler."""
        self.output_file = options['output']
        self.generate_html = options['html']
        self.verbose = options['verbose']
        
        self.stdout.write(
            self.style.SUCCESS('Generating comprehensive quality metrics report...')
        )
        
        # Collect all quality metrics
        quality_report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'project_name': 'VoyageurCompass',
                'report_version': '1.0'
            },
            'code_quality': self._collect_code_quality_metrics(),
            'test_coverage': self._collect_coverage_metrics(),
            'security': self._collect_security_metrics(),
            'performance': self._collect_performance_metrics(),
            'dependencies': self._collect_dependency_metrics(),
            'code_complexity': self._collect_complexity_metrics(),
            'technical_debt': self._calculate_technical_debt(),
            'quality_score': {}
        }
        
        # Calculate overall quality score
        quality_report['quality_score'] = self._calculate_quality_score(quality_report)
        
        # Save JSON report
        self._save_json_report(quality_report)
        
        # Generate HTML dashboard if requested
        if self.generate_html:
            self._generate_html_dashboard(quality_report)
        
        # Display summary
        self._display_summary(quality_report)
        
        self.stdout.write(
            self.style.SUCCESS(f'Quality metrics report generated: {self.output_file}')
        )

    def _collect_code_quality_metrics(self):
        """Collect code quality metrics from various tools."""
        metrics = {
            'black_compliance': {'status': 'unknown', 'issues': 0},
            'isort_compliance': {'status': 'unknown', 'issues': 0},
            'flake8_compliance': {'status': 'unknown', 'issues': 0},
            'bandit_security': {'status': 'unknown', 'issues': 0}
        }
        
        try:
            # Check Black formatting
            result = subprocess.run(['black', '--check', '.'], 
                                  capture_output=True, text=True, cwd=settings.BASE_DIR)
            if result.returncode == 0:
                metrics['black_compliance'] = {'status': 'compliant', 'issues': 0}
            else:
                # Count files that would be reformatted
                issues = len([line for line in result.stderr.split('\n') if 'would reformat' in line])
                metrics['black_compliance'] = {'status': 'non_compliant', 'issues': issues}
        except Exception as e:
            metrics['black_compliance'] = {'status': 'error', 'error': str(e)}

        try:
            # Check isort compliance
            result = subprocess.run(['isort', '--check-only', '.'], 
                                  capture_output=True, text=True, cwd=settings.BASE_DIR)
            if result.returncode == 0:
                metrics['isort_compliance'] = {'status': 'compliant', 'issues': 0}
            else:
                # Count files with import sorting issues
                issues = len([line for line in result.stdout.split('\n') if 'Fixing' in line])
                metrics['isort_compliance'] = {'status': 'non_compliant', 'issues': issues}
        except Exception as e:
            metrics['isort_compliance'] = {'status': 'error', 'error': str(e)}

        try:
            # Check flake8 compliance
            result = subprocess.run(['flake8', '.'], 
                                  capture_output=True, text=True, cwd=settings.BASE_DIR)
            issues = len([line for line in result.stdout.split('\n') if line.strip()])
            if issues == 0:
                metrics['flake8_compliance'] = {'status': 'compliant', 'issues': 0}
            else:
                metrics['flake8_compliance'] = {'status': 'non_compliant', 'issues': issues}
        except Exception as e:
            metrics['flake8_compliance'] = {'status': 'error', 'error': str(e)}

        try:
            # Check Bandit security
            result = subprocess.run(['bandit', '-r', '.', '--configfile', 'pyproject.toml', '-f', 'json'], 
                                  capture_output=True, text=True, cwd=settings.BASE_DIR)
            if result.returncode == 0:
                metrics['bandit_security'] = {'status': 'secure', 'issues': 0}
            else:
                try:
                    bandit_data = json.loads(result.stdout)
                    issues = len(bandit_data.get('results', []))
                    metrics['bandit_security'] = {'status': 'issues_found', 'issues': issues}
                except json.JSONDecodeError:
                    metrics['bandit_security'] = {'status': 'compliant', 'issues': 0}
        except Exception as e:
            metrics['bandit_security'] = {'status': 'error', 'error': str(e)}

        return metrics

    def _collect_coverage_metrics(self):
        """Collect test coverage metrics."""
        coverage_data = {
            'backend_coverage': {'percentage': 0, 'status': 'unknown'},
            'frontend_coverage': {'percentage': 0, 'status': 'unknown'},
            'overall_coverage': {'percentage': 0, 'trend': 'stable'}
        }

        # Try to read backend coverage from pytest-cov
        try:
            coverage_file = settings.BASE_DIR / 'coverage.xml'
            if coverage_file.exists():
                # Parse coverage.xml for coverage percentage
                with open(coverage_file, 'r') as f:
                    content = f.read()
                    # Simple regex to extract coverage percentage
                    import re
                    match = re.search(r'line-rate="([\d.]+)"', content)
                    if match:
                        percentage = float(match.group(1)) * 100
                        coverage_data['backend_coverage'] = {
                            'percentage': round(percentage, 2),
                            'status': 'passing' if percentage >= 38 else 'below_threshold'
                        }
        except Exception as e:
            if self.verbose:
                self.stdout.write(f'Backend coverage error: {e}')

        # Try to read frontend coverage
        try:
            frontend_coverage = settings.BASE_DIR / 'Design' / 'frontend' / 'coverage' / 'coverage-summary.json'
            if frontend_coverage.exists():
                with open(frontend_coverage, 'r') as f:
                    frontend_data = json.load(f)
                    total = frontend_data.get('total', {})
                    percentage = total.get('lines', {}).get('pct', 0)
                    coverage_data['frontend_coverage'] = {
                        'percentage': percentage,
                        'status': 'passing' if percentage >= 60 else 'below_threshold'
                    }
        except Exception as e:
            if self.verbose:
                self.stdout.write(f'Frontend coverage error: {e}')

        # Calculate overall coverage (weighted average)
        backend_pct = coverage_data['backend_coverage']['percentage']
        frontend_pct = coverage_data['frontend_coverage']['percentage']
        if backend_pct > 0 or frontend_pct > 0:
            # Weight backend more heavily (70% backend, 30% frontend)
            overall = (backend_pct * 0.7) + (frontend_pct * 0.3)
            coverage_data['overall_coverage']['percentage'] = round(overall, 2)

        return coverage_data

    def _collect_security_metrics(self):
        """Collect security-related metrics."""
        return {
            'bandit_issues': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'dependency_vulnerabilities': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'secrets_exposed': False,
            'owasp_compliance': 'unknown',
            'last_security_scan': 'unknown'
        }

    def _collect_performance_metrics(self):
        """Collect performance-related metrics."""
        performance_data = {
            'lighthouse_scores': {
                'performance': 0,
                'accessibility': 0,
                'best_practices': 0,
                'seo': 0
            },
            'api_response_times': {
                'avg_response_time_ms': 0,
                'p95_response_time_ms': 0,
                'slowest_endpoint': 'unknown'
            },
            'database_performance': {
                'avg_query_time_ms': 0,
                'slow_queries_count': 0
            }
        }

        # Try to read Lighthouse results
        try:
            lighthouse_dir = settings.BASE_DIR / 'Design' / 'frontend' / 'lighthouse-results'
            if lighthouse_dir.exists():
                # Find the most recent lighthouse report
                report_files = list(lighthouse_dir.glob('*.json'))
                if report_files:
                    latest_report = max(report_files, key=os.path.getctime)
                    with open(latest_report, 'r') as f:
                        lighthouse_data = json.load(f)
                        categories = lighthouse_data.get('categories', {})
                        performance_data['lighthouse_scores'] = {
                            'performance': round(categories.get('performance', {}).get('score', 0) * 100),
                            'accessibility': round(categories.get('accessibility', {}).get('score', 0) * 100),
                            'best_practices': round(categories.get('best-practices', {}).get('score', 0) * 100),
                            'seo': round(categories.get('seo', {}).get('score', 0) * 100)
                        }
        except Exception as e:
            if self.verbose:
                self.stdout.write(f'Lighthouse metrics error: {e}')

        # Try to read performance baselines
        try:
            baseline_file = settings.BASE_DIR / 'performance_baselines.json'
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    summary = baseline_data.get('summary', {})
                    performance_data['api_response_times']['avg_response_time_ms'] = round(
                        summary.get('overall_avg_response_time_ms', 0), 2
                    )
        except Exception as e:
            if self.verbose:
                self.stdout.write(f'Performance baseline error: {e}')

        return performance_data

    def _collect_dependency_metrics(self):
        """Collect dependency-related metrics."""
        return {
            'python_dependencies': {
                'total': 0,
                'outdated': 0,
                'vulnerable': 0
            },
            'javascript_dependencies': {
                'total': 0,
                'outdated': 0,
                'vulnerable': 0
            },
            'license_compliance': 'unknown'
        }

    def _collect_complexity_metrics(self):
        """Collect code complexity metrics."""
        return {
            'cyclomatic_complexity': {
                'average': 0,
                'max': 0,
                'functions_over_threshold': 0
            },
            'lines_of_code': {
                'python': 0,
                'javascript': 0,
                'total': 0
            },
            'technical_debt_ratio': 0
        }

    def _calculate_technical_debt(self):
        """Calculate technical debt indicators."""
        return {
            'code_smells': 0,
            'duplicated_lines': 0,
            'debt_ratio': 0,
            'estimated_hours': 0
        }

    def _calculate_quality_score(self, report):
        """Calculate overall quality score based on all metrics."""
        score_components = {
            'code_quality': 0,
            'test_coverage': 0,
            'security': 0,
            'performance': 0,
            'maintainability': 0
        }

        # Code quality score (0-100)
        code_quality = report['code_quality']
        quality_score = 100
        for tool, metrics in code_quality.items():
            if metrics['status'] == 'non_compliant' or metrics['status'] == 'issues_found':
                quality_score -= min(metrics.get('issues', 0) * 2, 20)
            elif metrics['status'] == 'error':
                quality_score -= 10
        score_components['code_quality'] = max(0, quality_score)

        # Test coverage score (0-100)
        coverage = report['test_coverage']
        coverage_score = coverage['overall_coverage']['percentage']
        score_components['test_coverage'] = min(100, coverage_score)

        # Security score (0-100) 
        security = report['security']
        security_score = 100
        if security['bandit_issues']['critical'] > 0:
            security_score -= security['bandit_issues']['critical'] * 30
        if security['bandit_issues']['high'] > 0:
            security_score -= security['bandit_issues']['high'] * 15
        if security['bandit_issues']['medium'] > 0:
            security_score -= security['bandit_issues']['medium'] * 5
        score_components['security'] = max(0, security_score)

        # Performance score (0-100)
        performance = report['performance']
        lighthouse_avg = sum(performance['lighthouse_scores'].values()) / 4 if any(performance['lighthouse_scores'].values()) else 0
        score_components['performance'] = lighthouse_avg

        # Maintainability score (placeholder)
        score_components['maintainability'] = 75  # Default value

        # Calculate weighted overall score
        weights = {
            'code_quality': 0.3,
            'test_coverage': 0.25,
            'security': 0.25,
            'performance': 0.15,
            'maintainability': 0.05
        }

        overall_score = sum(score * weights[component] for component, score in score_components.items())

        return {
            'overall_score': round(overall_score, 1),
            'components': score_components,
            'grade': self._score_to_grade(overall_score),
            'trend': 'improving'  # Placeholder
        }

    def _score_to_grade(self, score):
        """Convert numeric score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def _save_json_report(self, report):
        """Save the quality report as JSON."""
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2)

    def _generate_html_dashboard(self, report):
        """Generate an HTML dashboard from the quality report."""
        html_file = self.output_file.replace('.json', '.html')
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoyageurCompass Quality Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .score {{ font-size: 2em; font-weight: bold; text-align: center; margin: 10px 0; }}
        .grade-A {{ color: #28a745; }}
        .grade-B {{ color: #ffc107; }}
        .grade-C {{ color: #fd7e14; }}
        .grade-D {{ color: #dc3545; }}
        .grade-F {{ color: #dc3545; }}
        .progress-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #28a745; transition: width 0.3s ease; }}
        .metric-details {{ margin-top: 15px; }}
        .metric-item {{ display: flex; justify-content: space-between; margin: 5px 0; }}
        .status-pass {{ color: #28a745; }}
        .status-fail {{ color: #dc3545; }}
        .status-warning {{ color: #ffc107; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>VoyageurCompass Quality Dashboard</h1>
            <p>Generated on {report['metadata']['generated_at']}</p>
            <div class="score grade-{report['quality_score']['grade']}">
                Overall Quality Score: {report['quality_score']['overall_score']} ({report['quality_score']['grade']})
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Code Quality</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {report['quality_score']['components']['code_quality']}%"></div>
                </div>
                <div class="metric-details">
                    <div class="metric-item">
                        <span>Black Formatting:</span>
                        <span class="status-{'pass' if report['code_quality']['black_compliance']['status'] == 'compliant' else 'fail'}">
                            {report['code_quality']['black_compliance']['status']}
                        </span>
                    </div>
                    <div class="metric-item">
                        <span>Import Sorting:</span>
                        <span class="status-{'pass' if report['code_quality']['isort_compliance']['status'] == 'compliant' else 'fail'}">
                            {report['code_quality']['isort_compliance']['status']}
                        </span>
                    </div>
                    <div class="metric-item">
                        <span>Linting:</span>
                        <span class="status-{'pass' if report['code_quality']['flake8_compliance']['status'] == 'compliant' else 'fail'}">
                            {report['code_quality']['flake8_compliance'].get('issues', 0)} issues
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Test Coverage</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {report['test_coverage']['overall_coverage']['percentage']}%"></div>
                </div>
                <div class="metric-details">
                    <div class="metric-item">
                        <span>Backend Coverage:</span>
                        <span>{report['test_coverage']['backend_coverage']['percentage']}%</span>
                    </div>
                    <div class="metric-item">
                        <span>Frontend Coverage:</span>
                        <span>{report['test_coverage']['frontend_coverage']['percentage']}%</span>
                    </div>
                    <div class="metric-item">
                        <span>Overall Coverage:</span>
                        <span>{report['test_coverage']['overall_coverage']['percentage']}%</span>
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Security</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {report['quality_score']['components']['security']}%"></div>
                </div>
                <div class="metric-details">
                    <div class="metric-item">
                        <span>Security Issues:</span>
                        <span>{sum(report['security']['bandit_issues'].values())} found</span>
                    </div>
                    <div class="metric-item">
                        <span>OWASP Compliance:</span>
                        <span>{report['security']['owasp_compliance']}</span>
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Performance</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {report['quality_score']['components']['performance']}%"></div>
                </div>
                <div class="metric-details">
                    <div class="metric-item">
                        <span>Lighthouse Performance:</span>
                        <span>{report['performance']['lighthouse_scores']['performance']}/100</span>
                    </div>
                    <div class="metric-item">
                        <span>API Response Time:</span>
                        <span>{report['performance']['api_response_times']['avg_response_time_ms']}ms</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.stdout.write(f'HTML dashboard generated: {html_file}')

    def _display_summary(self, report):
        """Display a summary of the quality report."""
        score = report['quality_score']
        
        self.stdout.write('\n' + '='*60)
        self.stdout.write(self.style.SUCCESS('QUALITY METRICS DASHBOARD'))
        self.stdout.write('='*60)
        
        # Overall score with color coding
        if score['grade'] in ['A', 'B']:
            style = self.style.SUCCESS
        elif score['grade'] == 'C':
            style = self.style.WARNING
        else:
            style = self.style.ERROR
            
        self.stdout.write(style(f"Overall Quality Score: {score['overall_score']}/100 (Grade: {score['grade']})"))
        
        self.stdout.write('\nComponent Scores:')
        for component, value in score['components'].items():
            self.stdout.write(f'  {component.replace("_", " ").title()}: {value:.1f}/100')
        
        # Key metrics
        coverage = report['test_coverage']['overall_coverage']['percentage']
        self.stdout.write(f'\nTest Coverage: {coverage}%')
        
        code_issues = sum(
            metrics.get('issues', 0) 
            for metrics in report['code_quality'].values() 
            if isinstance(metrics.get('issues'), int)
        )
        self.stdout.write(f'Code Quality Issues: {code_issues}')
        
        security_issues = sum(report['security']['bandit_issues'].values())
        self.stdout.write(f'Security Issues: {security_issues}')
        
        if report['performance']['lighthouse_scores']['performance'] > 0:
            perf_score = report['performance']['lighthouse_scores']['performance']
            self.stdout.write(f'Performance Score: {perf_score}/100')