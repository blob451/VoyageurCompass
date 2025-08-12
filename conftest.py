"""
Global pytest configuration for VoyageurCompass.
"""

import os
import sys
from pathlib import Path

import django
from django.conf import settings

# Add the project root to the Python path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Configure Django settings for tests
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "VoyageurCompass.settings")


def pytest_configure():
    """Configure Django settings for pytest."""
    if not settings.configured:
        django.setup()
