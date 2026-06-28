import pytest
from fastapi.testclient import TestClient
import sys

sys.path.append('/Users/hkc619/Documents/PY/project/market-regime-platform/backend')
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client