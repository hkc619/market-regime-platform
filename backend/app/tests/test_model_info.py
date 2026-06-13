def test_model_info_returns_metadata(client):
    response = client.get("/model-info")

    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()

        assert data["version"] == "v1"
        assert data["model_name"] == "CNN-GRU Fusion"
        assert data["task"] == "Market Regime Classification"
        assert data["num_classes"] == 4
        assert data["classes"] == [
            "Trending-Down",
            "Trans-Down",
            "Trans-Up",
            "Trending-Up",
        ]
        assert "input_config" in data
        assert "model_config" in data
        assert "metrics" in data