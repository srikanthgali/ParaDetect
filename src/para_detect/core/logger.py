import logging
import logging.config
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

from para_detect.entities.logger_config import LoggerConfig


class LoggerFactory:
    """Factory class for creating loggers following industry standards"""

    _configured = False
    _config = None

    @classmethod
    def setup_logging(cls, config: LoggerConfig, environment: str = "development"):
        """Setup logging configuration once for the entire application"""
        if cls._configured:
            return

        cls._config = config

        # Create log directory
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging based on environment and config
        if environment == "production" or config.structured:
            cls._setup_structured_logging(config)
        else:
            cls._setup_standard_logging(config)

        cls._configured = True

    @classmethod
    def _setup_structured_logging(cls, config: LoggerConfig):
        """Production/structured logging setup"""
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                    "class": (
                        "pythonjsonlogger.jsonlogger.JsonFormatter"
                        if config.json_format
                        else "logging.Formatter"
                    ),
                },
                "detailed": {"format": config.format},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "json" if config.json_format else "detailed",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": config.level,
                    "formatter": "json" if config.json_format else "detailed",
                    "filename": config.log_file,
                    "maxBytes": (
                        config.rotation.get("max_bytes", 10485760)
                        if config.rotation
                        else 10485760
                    ),
                    "backupCount": (
                        config.rotation.get("backup_count", 5) if config.rotation else 5
                    ),
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": str(Path(config.log_dir) / "error.log"),
                    "maxBytes": (
                        config.rotation.get("max_bytes", 10485760)
                        if config.rotation
                        else 10485760
                    ),
                    "backupCount": (
                        config.rotation.get("backup_count", 5) if config.rotation else 5
                    ),
                },
            },
            "loggers": {
                "para_detect": {
                    "level": config.level,
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False,
                }
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
        }

        # Add component-specific loggers
        if config.loggers:
            for logger_name, logger_config in config.loggers.items():
                logging_config["loggers"][logger_name] = {
                    "level": logger_config.get("level", config.level),
                    "handlers": logger_config.get("handlers", ["console", "file"]),
                    "propagate": logger_config.get("propagate", False),
                }

        logging.config.dictConfig(logging_config)

    @classmethod
    def _setup_standard_logging(cls, config: LoggerConfig):
        """Development/standard logging setup"""
        # Setup root logger
        level = getattr(logging, config.level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers
        root_logger.handlers = []

        # Create formatter
        formatter = logging.Formatter(config.format)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation
        if config.log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                config.log_file,
                maxBytes=(
                    config.rotation.get("max_bytes", 10485760)
                    if config.rotation
                    else 10485760
                ),
                backupCount=(
                    config.rotation.get("backup_count", 5) if config.rotation else 5
                ),
            )
            file_handler.setLevel(getattr(logging, config.level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance"""
        if not cls._configured:
            # Use default configuration if not set up
            from para_detect.core.config_manager import ConfigurationManager

            config_manager = ConfigurationManager()
            logging_config = config_manager.get_logging_config()
            cls.setup_logging(logging_config)

        return logging.getLogger(name)


class MLOpsLogger:
    """Specialized logger for ML Operations with experiment tracking"""

    def __init__(self, component: str, experiment_id: Optional[str] = None):
        self.component = component
        self.experiment_id = experiment_id or self._generate_experiment_id()

        # Get logger
        self.logger = LoggerFactory.get_logger(f"para_detect.{component}")

        # Add context to all log messages
        self.context = {"component": component, "experiment_id": self.experiment_id}

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"

    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with context"""
        context = {**self.context, **kwargs}
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
        full_message = f"{message} | {context_str}"
        getattr(self.logger, level)(full_message)

    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log_with_context("info", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log_with_context("debug", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log_with_context("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log_with_context("error", message, **kwargs)

    def log_model_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """Log model training metrics"""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        epoch_info = f"epoch={epoch}" if epoch is not None else ""
        self.info(
            f"Model metrics: {metrics_str}", type="model_metrics", epoch=epoch_info
        )

    def log_data_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        info_str = ", ".join([f"{k}={v}" for k, v in dataset_info.items()])
        self.info(f"Dataset info: {info_str}", type="data_info")

    def log_model_artifact(self, artifact_path: str, artifact_type: str):
        """Log model artifacts"""
        self.info(
            f"Model artifact saved: {artifact_path}",
            type="artifact",
            artifact_type=artifact_type,
        )

    def log_prediction(
        self,
        input_text: str,
        prediction: Dict[str, Any],
        confidence: float,
        latency_ms: float,
    ):
        """Log prediction requests"""
        self.info(
            f"Prediction made",
            type="prediction",
            input_length=len(input_text),
            prediction=str(prediction),
            confidence=confidence,
            latency_ms=latency_ms,
        )


# Convenience functions
def get_logger(name: str = "para_detect") -> logging.Logger:
    """Get a standard logger"""
    return LoggerFactory.get_logger(name)


def get_mlops_logger(
    component: str, experiment_id: Optional[str] = None
) -> MLOpsLogger:
    """Get an MLOps-specific logger"""
    return MLOpsLogger(component, experiment_id)
