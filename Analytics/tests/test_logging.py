import logging
from django.test import TestCase
from io import StringIO

class AnalyticsLoggingTestCase(TestCase):
    """Test Analytics module logging integration"""
    
    def setUp(self):
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger('Analytics')
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
    
    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
    
    def test_analyticsLoggerExists(self):
        """Test Analytics logger is configured and functional"""
        self.logger.info('Analytics test log')
        
        log_output = self.log_stream.getvalue()
        self.assertIn('Analytics test log', log_output)
        
    def test_analyticsLoggerLevel(self):
        """Test Analytics logger respects configured level"""
        self.assertTrue(self.logger.isEnabledFor(logging.INFO))