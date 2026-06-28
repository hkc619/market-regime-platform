def test_health_returns_service_status(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert data["service"] == "market-regime-inference-api"
    assert "model_loaded" in data
    assert "model_version" in data

def test_database_health_check(client):
    response = client.get("/api/v1/health/db")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert data["result"] == 1