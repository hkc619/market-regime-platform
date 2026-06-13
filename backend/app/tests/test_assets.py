def test_supported_assets(client):
    response = client.get("/supported-assets")

    assert response.status_code == 200

    data = response.json()

    assert data["validated_inference_assets"] == ["SPY"]
    assert "available_data_assets" in data
    assert "SPY" in data["available_data_assets"]
    assert data["default_asset"] == "SPY"
    assert "note" in data