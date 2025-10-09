"""Custom exceptions for ParaDetect"""

class ParaDetectError(Exception):
    """Base exception for ParaDetect"""
    pass

class ConfigurationError(ParaDetectError):
    """Configuration related errors"""
    pass

class DataError(ParaDetectError):
    """Data related errors"""
    pass

class ModelError(ParaDetectError):
    """Model related errors"""
    pass

class DeploymentError(ParaDetectError):
    """Deployment related errors"""
    pass

class ValidationError(ParaDetectError):
    """Validation related errors"""
    pass
