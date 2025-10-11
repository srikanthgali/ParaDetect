"""Custom exceptions for ParaDetect pipeline"""

import sys
import traceback


class ParaDetectException(Exception):
    """Base exception class for ParaDetect"""

    def __init__(self, error_message: str):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = self._get_error_detail()

    def _get_error_detail(self) -> str:
        """Get detailed error information including file and line number"""
        _, _, exc_tb = sys.exc_info()
        if exc_tb is None:
            return self.error_message

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{self.error_message}]"

    def __str__(self):
        return self.error_detail


class ConfigurationError(ParaDetectException):
    """Configuration related errors"""

    pass


class DataError(ParaDetectException):
    """Data related errors"""

    pass


class DeploymentError(ParaDetectException):
    """Deployment related errors"""

    pass


class ModelError(ParaDetectException):
    """Model related errors"""

    pass


class DataIngestionError(ParaDetectException):
    """Exception raised during data ingestion"""

    pass


class DataPreprocessingError(ParaDetectException):
    """Exception raised during data preprocessing"""

    pass


class DataValidationError(ParaDetectException):
    """Exception raised during data validation"""

    pass


class ModelTrainingError(ParaDetectException):
    """Exception raised during model training"""

    pass


class ModelPredictionError(ParaDetectException):
    """Exception raised during model prediction"""

    pass


class MLPipelineException(ParaDetectException):
    """Exception for ML pipeline operations"""

    pass


class ModelTrainingError(MLPipelineException):
    """Exception for model training errors"""

    pass


class ModelEvaluationError(MLPipelineException):
    """Exception for model evaluation errors"""

    pass


class ModelValidationError(MLPipelineException):
    """Exception for model validation errors"""

    pass


class ModelRegistrationError(MLPipelineException):
    """Exception for model registration errors"""

    pass


class ModelPredictionError(MLPipelineException):
    """Exception for model prediction errors"""

    pass


class DeviceError(ParaDetectException):
    """Exception for device-related errors"""

    pass


class CheckpointError(MLPipelineException):
    """Exception for checkpoint operations"""

    pass
