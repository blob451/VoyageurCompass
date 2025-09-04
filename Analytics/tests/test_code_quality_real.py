"""
Real tests for code quality service.
Tests AST analysis, quality scoring, and recommendation generation.
No mocks - uses real functionality.
"""

import os
import tempfile
from pathlib import Path
from django.test import TestCase

from Analytics.services.code_quality_service import get_code_quality_service


class RealCodeQualityServiceTestCase(TestCase):
    """Real test cases for code quality service."""
    
    def setUp(self):
        """Set up test environment."""
        self.quality_service = get_code_quality_service()
    
    def test_real_service_initialization(self):
        """Test real code quality service initialization."""
        service = get_code_quality_service()
        
        self.assertIsNotNone(service)
        self.assertTrue(hasattr(service, 'analyze_file'))
        self.assertTrue(hasattr(service, 'analyze_project'))
        self.assertTrue(hasattr(service, 'get_quality_dashboard'))
        self.assertTrue(hasattr(service, 'get_quality_report'))
        
        # Verify singleton pattern
        service2 = get_code_quality_service()
        self.assertIs(service, service2)
        
        # Check analyzer initialization
        self.assertIsNotNone(service.analyzer)
        self.assertIsNotNone(service.project_root)
    
    def test_real_file_analysis(self):
        """Test real file analysis functionality."""
        # Create a temporary Python file for analysis
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write('''
def complex_function(a, b, c, d, e, f):
    """This function has high complexity."""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            return a + b + c + d + e + f
                        else:
                            return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0

def simple_function(x):
    """Simple function with low complexity."""
    return x * 2

class ExampleClass:
    def method_without_docstring(self):
        pass
    
    def method_with_docstring(self):
        """This method has a docstring."""
        return True

# Very long line that exceeds the recommended line length limit and should be flagged by the quality analyzer
''')
            temp_file_path = temp_file.name
        
        try:
            # Analyze the file
            result = self.quality_service.analyze_file(temp_file_path)
            
            self.assertIsNotNone(result)
            self.assertIn('quality_score', result)
            self.assertIn('issues', result)
            self.assertIn('metrics', result)
            self.assertIn('file_path', result)
            
            # Check quality score is valid
            score = result['quality_score']
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
            
            # Check that issues were detected
            issues = result['issues']
            self.assertIsInstance(issues, list)
            
            # Should detect complexity and line length issues
            issue_types = [issue['type'] for issue in issues]
            self.assertTrue(any('complexity' in issue_type or 'long' in issue_type.lower() 
                             for issue_type in issue_types))
            
            # Check metrics
            metrics = result['metrics']
            self.assertIn('lines_of_code', metrics)
            self.assertIn('complexity', metrics)
            self.assertGreater(metrics['lines_of_code'], 0)
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_real_project_analysis(self):
        """Test real project analysis functionality."""
        # Analyze a subset of the actual project
        patterns = ['Analytics/services/code_quality_service.py']  # Analyze own file
        
        result = self.quality_service.analyze_project(patterns)
        
        self.assertIsNotNone(result)
        self.assertIn('files_analyzed', result)
        self.assertIn('total_issues', result)
        self.assertIn('average_quality_score', result)
        self.assertIn('files', result)
        self.assertIn('summary', result)
        self.assertIn('analyzed_at', result)
        
        # Should analyze at least one file
        self.assertGreaterEqual(result['files_analyzed'], 1)
        
        # Check summary structure
        summary = result['summary']
        self.assertIn('severity_counts', summary)
        self.assertIn('top_issues', summary)
        self.assertIn('quality_distribution', summary)
        
        # Check severity counts
        severity_counts = summary['severity_counts']
        self.assertIn('error', severity_counts)
        self.assertIn('warning', severity_counts)
        self.assertIn('info', severity_counts)
        
        # Check quality distribution
        quality_dist = summary['quality_distribution']
        self.assertIn('excellent', quality_dist)
        self.assertIn('good', quality_dist)
        self.assertIn('fair', quality_dist)
        self.assertIn('poor', quality_dist)
    
    def test_real_quality_dashboard(self):
        """Test real quality dashboard generation."""
        dashboard = self.quality_service.get_quality_dashboard()
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn('current_quality_score', dashboard)
        self.assertIn('total_files', dashboard)
        self.assertIn('total_issues', dashboard)
        self.assertIn('issue_distribution', dashboard)
        self.assertIn('top_issue_types', dashboard)
        self.assertIn('quality_trends', dashboard)
        self.assertIn('recommendations', dashboard)
        self.assertIn('generated_at', dashboard)
        
        # Check quality score
        score = dashboard['current_quality_score']
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        
        # Check recommendations
        recommendations = dashboard['recommendations']
        self.assertIsInstance(recommendations, list)
        
        # Each recommendation should have proper structure
        for rec in recommendations:
            self.assertIn('priority', rec)
            self.assertIn('description', rec)
            self.assertIn('impact', rec)
    
    def test_real_quality_report_generation(self):
        """Test real quality report generation."""
        # Create a test file with known issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write('''
def bad_function():
    # No docstring, high complexity
    x = 1
    if x:
        if x > 0:
            if x < 10:
                if x != 5:
                    return "complex"
    
    # Unused variable
    unused_var = "not used"
    
    return "done"

class BadClass:
    # No docstring
    def method1(self):
        pass
    
    def method2(self):
        pass
''')
            temp_file_path = temp_file.name
        
        try:
            # Generate quality report
            report = self.quality_service.get_quality_report(temp_file_path)
            
            self.assertIsNotNone(report)
            self.assertIn('summary', report)
            self.assertIn('detailed_issues', report)
            self.assertIn('metrics', report)
            self.assertIn('recommendations', report)
            
            # Check summary
            summary = report['summary']
            self.assertIn('quality_score', summary)
            self.assertIn('total_issues', summary)
            self.assertIn('lines_analyzed', summary)
            
            # Should detect multiple issues
            self.assertGreater(summary['total_issues'], 0)
            
            # Check detailed issues
            issues = report['detailed_issues']
            self.assertIsInstance(issues, list)
            
            # Should have various issue types
            issue_types = {issue['type'] for issue in issues}
            expected_types = ['missing_docstring', 'complexity', 'unused_variable']
            found_types = issue_types.intersection(expected_types)
            self.assertGreater(len(found_types), 0)
            
            # Check recommendations
            recommendations = report['recommendations']
            self.assertIsInstance(recommendations, list)
            self.assertGreater(len(recommendations), 0)
            
        finally:
            os.unlink(temp_file_path)
    
    def test_real_quality_scoring_logic(self):
        """Test real quality scoring logic."""
        # Test with different quality files
        test_cases = [
            {
                'code': '''
def excellent_function(value: int) -> int:
    """
    Calculate double of input value.
    
    Args:
        value: The input integer value
        
    Returns:
        The doubled value
    """
    return value * 2
''',
                'expected_min_score': 85
            },
            {
                'code': '''
def poor_function(a,b,c,d,e,f,g,h):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            if g:
                                if h:
                                    return "very nested"
    return "poor"
''',
                'expected_max_score': 50
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(test_case['code'])
                temp_file_path = temp_file.name
            
            try:
                result = self.quality_service.analyze_file(temp_file_path)
                score = result['quality_score']
                
                if 'expected_min_score' in test_case:
                    self.assertGreaterEqual(score, test_case['expected_min_score'],
                                          f"Test case {i}: Score {score} should be >= {test_case['expected_min_score']}")
                
                if 'expected_max_score' in test_case:
                    self.assertLessEqual(score, test_case['expected_max_score'],
                                       f"Test case {i}: Score {score} should be <= {test_case['expected_max_score']}")
                
            finally:
                os.unlink(temp_file_path)
    
    def test_real_recommendation_generation(self):
        """Test real recommendation generation."""
        # Analyze current test file to get recommendations
        current_file = __file__
        
        result = self.quality_service.analyze_file(current_file)
        report = self.quality_service.get_quality_report(current_file)
        
        recommendations = report['recommendations']
        
        self.assertIsInstance(recommendations, list)
        
        # Each recommendation should have proper structure
        for rec in recommendations:
            self.assertIn('type', rec)
            self.assertIn('description', rec)
            self.assertIn('priority', rec)
            self.assertIn('effort', rec)
            
            # Priority should be valid
            self.assertIn(rec['priority'], ['high', 'medium', 'low'])
            
            # Effort should be valid
            self.assertIn(rec['effort'], ['low', 'medium', 'high'])
    
    def test_real_export_functionality(self):
        """Test real analysis report export functionality."""
        # Analyze a small subset
        analysis_result = self.quality_service.analyze_project(['Analytics/tests/test_code_quality_real.py'])
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            export_path = temp_file.name
        
        try:
            success = self.quality_service.export_analysis_report(analysis_result, export_path)
            self.assertTrue(success)
            
            # Verify file was created and contains data
            self.assertTrue(os.path.exists(export_path))
            
            with open(export_path, 'r') as f:
                import json
                exported_data = json.load(f)
            
            # Should contain the same data structure
            self.assertEqual(exported_data['files_analyzed'], analysis_result['files_analyzed'])
            self.assertEqual(exported_data['total_issues'], analysis_result['total_issues'])
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


if __name__ == '__main__':
    import django
    django.setup()
    from django.test import TestRunner
    runner = TestRunner()
    runner.run_tests(['Analytics.tests.test_code_quality_real'])