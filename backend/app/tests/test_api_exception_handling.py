from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api import backtest, data_refresh, macro_refresh, prediction_db
from app.api.error_handler import app_error_handler
from app.core.exceptions import (
    AppError, ExternalDataFetchError, InsufficientFeatureDataError,
    InsufficientRawDataError, InvalidExternalDataError, ModelInferenceError,
    PredictionSaveError, TickerNotFoundError,
)
from app.db.session import get_db


@pytest.fixture(params=['prediction', 'backtest', 'market', 'daily', 'monthly'])
def endpoint(request, monkeypatch):
    kind = request.param
    app = FastAPI()
    app.add_exception_handler(AppError, app_error_handler)
    model_state = SimpleNamespace(model_loaded=True, error=None)
    app.state.model_state = model_state
    db = object()
    app.dependency_overrides[get_db] = lambda: db

    @app.middleware('http')
    async def request_id(req, call_next):
        req.state.request_id = 'test-request'
        return await call_next(req)

    service = Mock()
    payload = {'ticker': 'SPY'}
    expected_args = {'db': db, 'request_id': 'test-request'}
    if kind == 'prediction':
        module, path = prediction_db, '/predictions/latest'
        monkeypatch.setattr(module, 'create_latest_prediction', service)
        expected_args.update(ticker='SPY', model_state=model_state)
    elif kind == 'backtest':
        module, path = backtest, '/backtest/test'
        monkeypatch.setattr(module, 'backtest_for_range', service)
        payload.update(sup0='QQQ', sup1='TLT', start_date='2025-01-02', end_date='2025-01-03')
        expected_args.update(ticker='SPY', sup0='QQQ', sup1='TLT',
                             start_date=date(2025, 1, 2), end_date=date(2025, 1, 3),
                             model_state=model_state)
    elif kind == 'market':
        module, path = data_refresh, '/data/refresh/market'
        monkeypatch.setattr(module, 'refresh_market_data', service)
        expected_args.update(ticker='SPY')
    else:
        module, path = macro_refresh, f'/data/refresh/macro/{kind}'
        macro_service = SimpleNamespace(**{f'refresh_{kind}_macro': service})
        app.dependency_overrides[module.get_macro_data_service] = lambda: macro_service
        payload = None
    app.include_router(module.router, prefix='/api/v1')
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, path='/api/v1' + path, payload=payload,
                              service=service, kind=kind, expected_args=expected_args,
                              model_state=model_state)


@pytest.mark.parametrize('error_type', [
    TickerNotFoundError, InsufficientRawDataError, InsufficientFeatureDataError,
    ExternalDataFetchError, InvalidExternalDataError, ModelInferenceError,
    PredictionSaveError,
])
def test_known_errors_keep_status_and_code(endpoint, error_type):
    endpoint.service.side_effect = error_type('Known failure')
    response = endpoint.client.post(endpoint.path, json=endpoint.payload)
    assert response.status_code == error_type.status_code
    assert response.json() == {
        'status': 'error', 'error_code': error_type.error_code, 'message': 'Known failure',
    }
    endpoint.service.assert_called_once_with(**endpoint.expected_args)


@pytest.mark.parametrize('error_type', [RuntimeError, ValueError])
def test_unexpected_errors_return_500(endpoint, error_type):
    endpoint.service.side_effect = error_type('internal diagnostic')
    response = endpoint.client.post(endpoint.path, json=endpoint.payload)
    assert response.status_code == 500
    assert 'internal diagnostic' not in response.text
    expected = 'prediction_failed' if endpoint.kind in ('prediction', 'backtest') else 'data_refresh_failed'
    assert response.json()['detail']['error'] == expected


def test_http_exception_is_preserved(endpoint):
    endpoint.service.side_effect = HTTPException(409, 'Conflict', headers={'X-Test': 'preserved'})
    response = endpoint.client.post(endpoint.path, json=endpoint.payload)
    assert response.status_code == 409
    assert response.json() == {'detail': 'Conflict'}
    assert response.headers['X-Test'] == 'preserved'


def test_success_keeps_response_and_service_arguments(endpoint):
    if endpoint.kind == 'prediction':
        result = {
            'prediction_id': 1, 'ticker': 'SPY', 'as_of_date': '2025-01-03',
            'predicted_class': 3, 'predicted_regime': 'Trending-Up', 'confidence': 0.8,
            'probabilities': {'Trending-Up': 0.8, 'Transition-Up': 0.2},
            'model_version': 'v2', 'created_at': '2025-01-03T12:00:00',
            'input_window': {'raw_window_rows': 312, 'feature_rows': 60,
                             'model_input_rows': 60, 'feature_dim': 73,
                             'input_start_date': '2024-10-01', 'input_end_date': '2025-01-03'},
        }
    elif endpoint.kind == 'backtest':
        result = {'ticker': 'SPY', 'num_predictions': 2, 'predictions': []}
    else:
        result = {'latest_before': '2025-01-02', 'latest_after': '2025-01-03',
                  'rows_fetched': 1, 'rows_inserted_or_updated': 1,
                  'status': 'success', 'message': 'Refreshed successfully.'}
        result.update({'ticker': 'SPY'} if endpoint.kind == 'market' else {'type_of_macro': endpoint.kind})
    endpoint.service.return_value = result
    response = endpoint.client.post(endpoint.path, json=endpoint.payload)
    assert response.status_code == 200
    assert response.json() == result
    endpoint.service.assert_called_once_with(**endpoint.expected_args)


@pytest.mark.parametrize('start,end', [('invalid', '2025-01-03'), ('2025-01-02', 'invalid'), ('2025-01-04', '2025-01-03')])
def test_backtest_invalid_dates_do_not_call_service(start, end, monkeypatch):
    app = FastAPI()
    app.state.model_state = SimpleNamespace(model_loaded=True, error=None)
    app.dependency_overrides[get_db] = lambda: object()

    @app.middleware('http')
    async def request_id(req, call_next):
        req.state.request_id = 'test-request'
        return await call_next(req)

    service = Mock()
    monkeypatch.setattr(backtest, 'backtest_for_range', service)
    app.include_router(backtest.router)
    with TestClient(app) as client:
        response = client.post('/backtest/test', json={
            'ticker': 'SPY', 'sup0': 'QQQ', 'sup1': 'TLT',
            'start_date': start, 'end_date': end,
        })
    assert response.status_code == 422
    service.assert_not_called()
