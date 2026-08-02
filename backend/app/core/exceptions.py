class AppError(Exception):
    """Base class for application-level errors."""
    status_code = 500
    error_code = "APP_ERROR"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TickerNotFoundError(AppError):
    status_code = 404
    error_code = "TICKER_NOT_FOUND"


class InsufficientRawDataError(AppError):
    status_code = 422
    error_code = "INSUFFICIENT_RAW_DATA"


class InsufficientFeatureDataError(AppError):
    status_code = 422
    error_code = "INSUFFICIENT_FEATURE_DATA"


class ModelInferenceError(AppError):
    status_code = 500
    error_code = "MODEL_INFERENCE_ERROR"


class PredictionSaveError(AppError):
    status_code = 500
    error_code = "PREDICTION_SAVE_ERROR"

class MacroNotFoundError(AppError):
    status_code = 404
    error_code = "MACRO_NOT_FOUND"
    

class ExternalDataFetchError(AppError):
    status_code = 502
    error_code = "EXTERNAL_DATA_FETCH_ERROR"

class InvalidExternalDataError(AppError):
    status_code = 502
    error_code = "INVALID_EXTERNAL_DATA"

class NoNewMarketDataError(AppError):
    """
    Internal control-flow exception.

    Usually should not be mapped to API error.
    Refresh endpoint should return status='no_new_data' with HTTP 200.
    """
    pass

class NoSuccessfulBacktestPredictionsError(AppError):
    pass

class InvalidPredictionError(AppError):
    pass