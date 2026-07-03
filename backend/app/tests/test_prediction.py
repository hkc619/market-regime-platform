import math

def test_post_latest_prediction_success(client):
    payload = {
        "ticker": "SPY"
    }

    response = client.post("/api/v1/predictions/latest", json=payload)
    assert response.status_code == 200

    if response.status_code == 200:
        data = response.json()

        
        assert data["ticker"] == "SPY"
        assert "prediction_id" in data
        assert "as_of_date" in data
        assert "predicted_class" in data
        assert "predicted_regime" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "input_window" in data

        assert data["predicted_regime"]
        

        assert 0 <= data["confidence"] <= 1
        assert data["input_window"]["model_input_rows"] == 60
        assert data["input_window"]["feature_dim"] > 0

def test_prediction_probabilities_sum_to_one(client):
    payload = {
        "ticker": "SPY"
    }

    response = client.post("/api/v1/predictions/latest", json=payload)
    assert response.status_code == 200

    if response.status_code == 200:
        data = response.json()

        prob_dict = data["probabilities"]
        prob_sum = sum(prob_dict.values())

        assert math.isclose(prob_sum, 1.0, rel_tol=1e-4, abs_tol=1e-5)


def test_get_latest_prediction_success(client):
    # 先 POST 一次，確保 DB 裡至少有 prediction history
    post_response = client.post(
        "/api/v1/predictions/latest",
        json={"ticker": "SPY"},
        headers={"X-Request-ID": "test-get-latest-setup-001"},
    )

    assert post_response.status_code == 200

    get_response = client.get("/api/v1/predictions/latest?ticker=SPY")

    assert get_response.status_code == 200

    data = get_response.json()

    assert data["ticker"] == "SPY"
    assert "prediction_id" in data
    assert "predicted_regime" in data
    assert "probabilities" in data


def test_invalid_ticker_returns_error(client):
    response = client.post(
        "/api/v1/predictions/latest",
        json={"ticker": "INVALID_TICKER"},
        headers={"X-Request-ID": "test-invalid-ticker-001"},
    )

    assert response.status_code in [404, 422]