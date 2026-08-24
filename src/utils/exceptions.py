class DRSystemError(Exception):
    """Base exception for all domain-specific errors."""
    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

class InvalidInputError(DRSystemError):
    def __init__(self, message: str = "Invalid input provided"):
        super().__init__(message, "INVALID_INPUT")

class ModelLoadError(DRSystemError):
    def __init__(self, message: str = "Failed to load AI model"):
        super().__init__(message, "MODEL_LOAD_ERROR")

class QualityRejectedError(DRSystemError):
    def __init__(self, message: str = "Image quality check failed"):
        super().__init__(message, "QUALITY_REJECTED")

class PredictionError(DRSystemError):
    def __init__(self, message: str = "Error occurred during model prediction"):
        super().__init__(message, "PREDICTION_ERROR")

class PreprocessingError(DRSystemError):
    def __init__(self, message: str = "Error occurred during image preprocessing"):
        super().__init__(message, "PREPROCESSING_ERROR")
