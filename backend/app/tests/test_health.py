def test_health_returns_service_status(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert data["service"] == "market-regime-inference-api"
    assert "model_loaded" in data
    assert "model_version" in data