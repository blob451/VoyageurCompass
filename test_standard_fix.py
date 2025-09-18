#!/usr/bin/env python
"""Test script to verify the Standard explanation fix"""

import os
import sys
import django
import json

# Add the project to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'VoyageurCompass.settings')
django.setup()

from django.test import RequestFactory
from django.contrib.auth.models import User
from Data.models import AnalyticsResults
from Analytics.explanation_views import get_explanation

def test_standard_explanation_fix():
    """Test the Standard explanation French fix"""

    print("Testing Standard explanation French fix...")
    print("=" * 50)

    # Create test user and request factory
    factory = RequestFactory()
    try:
        user = User.objects.get(username='admin')  # Assuming admin user exists
    except User.DoesNotExist:
        print("Admin user not found, creating test user...")
        user = User.objects.create_user('testuser', 'test@example.com', 'testpass')

    # Get the latest analysis result for testing
    latest_analysis = AnalyticsResults.objects.filter(user=user).order_by('-as_of').first()

    if not latest_analysis:
        print("No analysis results found for user. Please run an analysis first.")
        return

    print(f"Testing with analysis ID: {latest_analysis.id} ({latest_analysis.stock.symbol})")

    # Test 1: Get original explanation (should be English)
    print("\n1. Testing original explanation retrieval...")
    request = factory.get(f'/analytics/explanation/{latest_analysis.id}/', {'detail_level': 'summary'})
    request.user = user

    response = get_explanation(request, latest_analysis.id)
    if response.status_code == 200:
        data = response.data
        if data.get('has_explanation'):
            explanation = data.get('explanation', {})
            print(f"   Original language: {explanation.get('language', 'unknown')}")
            print(f"   Content preview: {explanation.get('content', '')[:100]}...")
        else:
            print("   No explanation found")
    else:
        print(f"   Error: {response.status_code}")

    # Test 2: Request French translation (should trigger on-demand translation)
    print("\n2. Testing French on-demand translation...")
    request = factory.get(f'/analytics/explanation/{latest_analysis.id}/', {
        'detail_level': 'summary',
        'language': 'fr'
    })
    request.user = user

    response = get_explanation(request, latest_analysis.id)
    if response.status_code == 200:
        data = response.data
        if data.get('has_explanation'):
            explanation = data.get('explanation', {})
            multilingual = explanation.get('multilingual', {})

            print(f"   Returned language: {explanation.get('language', 'unknown')}")
            print(f"   Translation success: {multilingual.get('translated', False)}")
            print(f"   Source language: {multilingual.get('source_language', 'unknown')}")
            print(f"   Target language: {multilingual.get('target_language', 'unknown')}")
            print(f"   Content preview: {explanation.get('content', '')[:100]}...")

            if multilingual.get('translated'):
                print("   SUCCESS: On-demand translation worked!")
            else:
                print(f"   FAILED: {multilingual.get('error', 'Unknown error')}")
        else:
            print("   No explanation found")
    else:
        print(f"   Error: {response.status_code}")

    # Test 3: Request Spanish translation
    print("\n3. Testing Spanish on-demand translation...")
    request = factory.get(f'/analytics/explanation/{latest_analysis.id}/', {
        'detail_level': 'summary',
        'language': 'es'
    })
    request.user = user

    response = get_explanation(request, latest_analysis.id)
    if response.status_code == 200:
        data = response.data
        if data.get('has_explanation'):
            explanation = data.get('explanation', {})
            multilingual = explanation.get('multilingual', {})

            print(f"   Returned language: {explanation.get('language', 'unknown')}")
            print(f"   Translation success: {multilingual.get('translated', False)}")

            if multilingual.get('translated'):
                print("   SUCCESS: Spanish on-demand translation worked!")
        else:
            print("   No explanation found")
    else:
        print(f"   Error: {response.status_code}")

    print("\n" + "=" * 50)
    print("Standard explanation fix test completed!")

if __name__ == "__main__":
    test_standard_explanation_fix()