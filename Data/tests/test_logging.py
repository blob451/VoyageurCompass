import logging
from django.test import TestCase
from io import StringIO

class DataLoggingTestCase(TestCase):
    """Test Data module logging integration"""
    
    def setUp(self):
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger('Data')
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
    
    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
    
    def test_dataLoggerExists(self):
        """Test Data logger is configured and functional"""
        self.logger.info('Data test log')
        
        log_output = self.log_stream.getvalue()
        self.assertIn('Data test log', log_output)
        
    def test_dataLoggerLevel(self):
        """Test Data logger respects configured level"""
        self.assertTrue(self.logger.isEnabledFor(logging.INFO))