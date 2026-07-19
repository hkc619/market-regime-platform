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


class NoNewMarketDataError(AppError):
    status_code = 200
    error_code = "NO_NEW_MARKET_DATA"


class InvalidExternalDataError(AppError):
    status_code = 502
    error_code = "INVALID_EXTERNAL_DATA"