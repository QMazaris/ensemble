import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.api.main import app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Ensemble Pipeline API" in response.json()["message"]

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_pipeline_status():
    """Test the pipeline status endpoint."""
    response = client.get("/pipeline/status")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "message" in response.json()

def test_get_config():
    """Test the config endpoint."""
    response = client.get("/config")
    assert response.status_code == 200
    assert "config" in response.json()

def test_list_models():
    """Test the models list endpoint."""
    response = client.get("/models/list")
    assert response.status_code == 200
    assert "models" in response.json()

def test_list_files():
    """Test the files list endpoint."""
    response = client.get("/files/list")
    assert response.status_code == 200
    assert "files" in response.json() 