from unittest.mock import patch
from django.test import TestCase, Client
from django.db import connection

class HealthCheckTestCase(TestCase):
    """Test health check endpoints"""
    
    def setUp(self):
        self.client = Client()
    
    def test_healthzReturns200Quickly(self):
        """Test /healthz returns 200 status quickly"""
        response = self.client.get('/healthz')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_readyzReturns200WhenDbReachable(self):
        """Test /readyz returns 200 when database is reachable"""
        response = self.client.get('/readyz')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'ready')
        self.assertEqual(data['database'], 'connected')
        self.assertIn('timestamp', data)
    
    @patch('django.db.connection.cursor')
    def test_readyzReturnsNon200WhenDbUnreachable(self, mock_cursor):
        """Test /readyz returns non-200 when database is unreachable"""
        # Mock database connection failure
        mock_cursor.side_effect = Exception('Database connection failed')
        
        response = self.client.get('/readyz')
        
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data['status'], 'not ready')
        self.assertEqual(data['database'], 'disconnected')
        self.assertIn('error', data)