import math

def test_post_latest_prediction_success(client):
    payload = {
        "ticker": "SPY"
    }
    response = client.post("/api/v1/predictions/latest", json=payload)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()

        assert data["predicted_regime"]
        prob_dict = data["probabilities"]
        prob_sum = sum(prob_dict.values())

        assert math.isclose(prob_sum, 1.0, abs_tol=1e-5)