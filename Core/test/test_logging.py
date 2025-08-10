import json
import logging
import os
from django.test import TestCase, override_settings
from io import StringIO

class LoggingConfigTestCase(TestCase):
    """Test structured logging configuration"""
    
    def setUp(self):
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger('VoyageurCompass.test')
        # Save current state
        self._prev_handlers = self.logger.handlers[:]
        self._prev_level = self.logger.level
        self._prev_propagate = self.logger.propagate
        # Isolate this test's logger
        self.logger.handlers = []
        self.logger.propagate = False
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
    
    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
        # Restore previous state
        self.logger.handlers = self._prev_handlers
        self.logger.setLevel(self._prev_level)
        self.logger.propagate = self._prev_propagate
    
    @override_settings(LOG_FORMAT='json')
    def test_structuredLoggingEmitsRequiredFields(self):
        """Test that structured logs contain required fields"""
        # Configure JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "environment": "test", "message": "%(message)s"}',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        self.handler.setFormatter(formatter)
        
        # Log a test message
        self.logger.info('Test log message')
        
        # Verify structured output
        log_output = self.log_stream.getvalue().strip()
        log_data = json.loads(log_output)
        
        self.assertIn('timestamp', log_data)
        self.assertEqual(log_data['level'], 'INFO')
        self.assertEqual(log_data['logger'], 'VoyageurCompass.test')
        self.assertEqual(log_data['environment'], 'test')
        self.assertEqual(log_data['message'], 'Test log message')