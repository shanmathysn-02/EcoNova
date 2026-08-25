import pytest
from src.api.app import create_app

@pytest.fixture
def client():
    # Use a dummy config for testing if necessary, or just use the main one
    app = create_app("config.yaml")
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the GET /api/v1/health endpoint."""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is False
    assert data['model_version'] is None
    
    # Check that request ID is injected
    assert 'X-Request-ID' in response.headers

def test_model_info(client):
    """Test the GET /api/v1/model/info endpoint."""
    response = client.get('/api/v1/model/info')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['success'] is False
    assert "Model is not loaded" in data['message']
    assert data['model'] is None
    
    assert 'X-Request-ID' in response.headers

def test_invalid_route(client):
    """Test that an invalid route returns a correctly formatted JSON error."""
    response = client.get('/api/v1/invalid_endpoint_does_not_exist')
    assert response.status_code == 404
    
    data = response.get_json()
    assert data['success'] is False
    assert data['error_code'] == 'HTTP_404'
    assert 'message' in data
    assert 'request_id' in data
