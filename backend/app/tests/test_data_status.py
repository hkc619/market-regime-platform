def test_get_spy_market_data_status(client):
    response = client.get("/api/v1/data/status?ticker=SPY")

    assert response.status_code == 200

    data = response.json()

    assert data["ticker"] == "SPY"
    assert data["available_days"] >= 252
    assert data["required_days"] == 252
    assert data["is_ready_for_prediction"] is True
    assert data["start_date"] is not None
    assert data["end_date"] is not None


def test_get_unknown_ticker_market_data_status(client):
    response = client.get("/api/v1/data/status?ticker=UNKNOWN")

    assert response.status_code == 404


def test_get_macro_daily_status(client):
    response = client.get("/api/v1/data/macro/status")

    assert response.status_code == 200

    data = response.json()

    assert data["total_rows"] > 0
    assert data["start_date"] is not None
    assert data["end_date"] is not None

def test_get_latest_window_success(client):
    response = client.get("/api/v1/data/window?ticker=SPY&lookback=312")

    assert response.status_code == 200

    data = response.json()

    assert data["ticker"] == "SPY"
    assert data["available_days"] == 312
    assert data["required_days"] == 312
    assert data["is_ready"] is True
    assert data["start_date"] is not None
    assert data["end_date"] is not None

def test_window_order_is_ascending(client):
    pass

def test_invalid_ticker_window(client):
    pass

def test_insufficient_data(client):
    pass