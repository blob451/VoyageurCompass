"""
Global pytest configuration for VoyageurCompass - Performance Optimized.
"""

import os
import sys
from pathlib import Path

import django
from django.conf import settings

import pytest

# Add the project root to the Python path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Configure Django settings for tests with performance settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "VoyageurCompass.test_settings")


def pytest_configure(config):
    """Configure Django settings for pytest with performance optimizations."""
    if not settings.configured:
        django.setup()

    # Enable fast test mode if parallel execution detected
    if hasattr(config.option, "numprocesses") and config.option.numprocesses:
        os.environ["TEST_FAST_MODE"] = "true"


@pytest.fixture(scope="session", autouse=True)
def enable_db_access_for_all_tests(django_db_setup, django_db_blocker):
    """Enable database access for all tests without requiring django_db marker."""
    with django_db_blocker.unblock():
        yield


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    """Mock network calls to prevent actual API requests in tests."""
    import requests

    def mock_get(*args, **kwargs):
        raise Exception("Network calls are disabled in tests. Use mock data instead.")

    def mock_post(*args, **kwargs):
        raise Exception("Network calls are disabled in tests. Use mock data instead.")

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)


@pytest.fixture
def fast_user(db):
    """Create a user for testing quickly without password hashing overhead."""
    from django.contrib.auth.models import User

    return User.objects.create(
        username="testuser",
        email="test@example.com",
        password="md5$salt$hashedpassword",  # Pre-hashed for speed
    )


@pytest.fixture
def sample_stock(db):
    """Create a sample stock for testing."""
    from Data.models import Stock

    return Stock.objects.create(symbol="AAPL", name="Apple Inc.", data_source="test")


@pytest.fixture
def sample_portfolio(db, fast_user):
    """Create a sample portfolio for testing."""
    from Data.models import Portfolio

    return Portfolio.objects.create(name="Test Portfolio", user=fast_user)
