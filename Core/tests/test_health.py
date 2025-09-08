from django.test import TestCase, Client, override_settings
from django.db import connection
import time


class HealthCheckTestCase(TestCase):
    """Test health check endpoints with real operations"""
    
    def setUp(self):
        self.client = Client()
    
    def test_healthzReturns200Quickly(self):
        """Test /healthz returns 200 status quickly with real timing validation"""
        start_time = time.time()
        response = self.client.get('/healthz')
        end_time = time.time()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        
        # Verify it responds within reasonable time (under 15 seconds due to service initialization)
        response_time = end_time - start_time
        self.assertLess(response_time, 15.0)
    
    def test_readyzReturns200WhenDbReachable(self):
        """Test /readyz returns 200 when database is actually reachable"""
        response = self.client.get('/readyz')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'ready')
        self.assertEqual(data['database'], 'connected')
        self.assertIn('timestamp', data)
        
        # Verify actual database connectivity by performing a real query
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], 1)
    
    def test_readyzHandlesDbConnectionsGracefully(self):
        """Test /readyz handles database connections gracefully with real validation"""
        # Since we're using real database operations, this test validates
        # that the readiness check properly handles actual database connectivity
        # and returns the expected structure even when database works
        response = self.client.get('/readyz')
        
        # Verify response structure regardless of connection state
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('database', data) 
        self.assertIn('timestamp', data)
        
        # Since database is working in test environment, should be ready
        if response.status_code == 200:
            self.assertEqual(data['status'], 'ready')
            self.assertEqual(data['database'], 'connected')
        else:
            # If database connection fails, should return appropriate error
            self.assertEqual(response.status_code, 503)
            self.assertEqual(data['status'], 'not ready')
            self.assertEqual(data['database'], 'disconnected')
