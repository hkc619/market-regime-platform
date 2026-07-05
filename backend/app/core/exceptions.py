class AppError(Exception):
    """Base class for application-level errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class TickerNotFoundError(AppError):
    pass


class InsufficientRawDataError(AppError):
    pass


class InsufficientFeatureDataError(AppError):
    pass


class ModelInferenceError(AppError):
    pass


class PredictionSaveError(AppError):
    pass

class MacroNotFoundError(AppError):
    pass