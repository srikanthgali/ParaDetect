import logging
import logging.config
import logging.handlers
import sys
import json
import gzip
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta

from para_detect.entities.logger_config import LoggerConfig


class TimestampedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """Custom handler that creates timestamped log files"""

    def __init__(
        self,
        filename,
        when="midnight",
        interval=1,
        backupCount=0,
        encoding=None,
        delay=False,
        utc=False,
        atTime=None,
        compress_old_logs=True,
    ):
        self.compress_old_logs = compress_old_logs
        super().__init__(
            filename, when, interval, backupCount, encoding, delay, utc, atTime
        )

    def getFilesToDelete(self):
        """Override to handle compressed files and timestamped names"""
        files = super().getFilesToDelete()
        # Also consider compressed files
        if self.compress_old_logs:
            compressed_files = []
            for f in files:
                gz_file = f + ".gz"
                if Path(gz_file).exists():
                    compressed_files.append(gz_file)
            files.extend(compressed_files)
        return files

    def doRollover(self):
        """Override to add compression and better naming"""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Create timestamped filename
        current_time = int(self.rolloverAt - self.interval)
        timestamp = datetime.fromtimestamp(current_time).strftime("%Y%m%d_%H%M%S")

        # Get base filename without extension
        base_path = Path(self.baseFilename)
        base_name = base_path.stem
        extension = base_path.suffix

        # Create new filename with timestamp
        dfn = base_path.parent / f"{base_name}_{timestamp}{extension}"

        if Path(self.baseFilename).exists():
            Path(self.baseFilename).rename(dfn)

            # Compress the rotated file if enabled
            if self.compress_old_logs:
                self._compress_file(dfn)

        # Delete old files
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                Path(s).unlink(missing_ok=True)

        # Calculate next rollover time
        self.rolloverAt = self.rolloverAt + self.interval

        if not self.delay:
            self.stream = self._open()

    def _compress_file(self, filepath):
        """Compress the log file"""
        try:
            compressed_path = str(filepath) + ".gz"
            with open(filepath, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            Path(filepath).unlink()  # Remove original file
        except Exception as e:
            # If compression fails, keep the original file
            print(f"Warning: Failed to compress log file {filepath}: {e}")


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
        """Production/structured logging setup with date-based rotation"""

        log_dir = Path(config.log_dir)

        # Use base filenames without timestamps - let the handler add them
        main_log_file = log_dir / "para_detect.log"
        error_log_file = log_dir / "para_detect_error.log"
        debug_log_file = log_dir / "para_detect_debug.log"

        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": (
                    {
                        "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                        "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s",
                    }
                    if config.json_format
                    else {"format": config.format}
                ),
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
                },
                "simple": {"format": "%(asctime)s [%(levelname)s] - %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                },
                "main_file": {
                    "()": TimestampedRotatingFileHandler,
                    "filename": str(main_log_file),
                    "when": (
                        config.rotation.get("when", "midnight")
                        if config.rotation
                        else "midnight"
                    ),
                    "interval": (
                        config.rotation.get("interval", 1) if config.rotation else 1
                    ),
                    "backupCount": (
                        config.rotation.get("backup_count", 7) if config.rotation else 7
                    ),
                    "level": config.level,
                    "formatter": "json" if config.json_format else "detailed",
                    "compress_old_logs": (
                        config.rotation.get("compress", True)
                        if config.rotation
                        else True
                    ),
                },
                "error_file": {
                    "()": TimestampedRotatingFileHandler,
                    "filename": str(error_log_file),
                    "when": "midnight",
                    "interval": 1,
                    "backupCount": 30,  # Keep error logs longer
                    "level": "ERROR",
                    "formatter": "detailed",
                    "compress_old_logs": True,
                },
                "debug_file": (
                    {
                        "()": TimestampedRotatingFileHandler,
                        "filename": str(debug_log_file),
                        "when": "midnight",
                        "interval": 1,
                        "backupCount": 3,  # Keep debug logs shorter
                        "level": "DEBUG",
                        "formatter": "detailed",
                        "compress_old_logs": True,
                    }
                    if config.level == "DEBUG"
                    else None
                ),
            },
            "loggers": {
                "para_detect": {
                    "level": config.level,
                    "handlers": ["console", "main_file", "error_file"]
                    + (["debug_file"] if config.level == "DEBUG" else []),
                    "propagate": False,
                }
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
        }

        # Remove None handlers
        logging_config["handlers"] = {
            k: v for k, v in logging_config["handlers"].items() if v is not None
        }

        # Add component-specific loggers
        if config.loggers:
            for logger_name, logger_config in config.loggers.items():
                full_logger_name = f"para_detect.{logger_name}"
                logging_config["loggers"][full_logger_name] = {
                    "level": logger_config.get("level", config.level),
                    "handlers": logger_config.get("handlers", ["console", "main_file"]),
                    "propagate": logger_config.get("propagate", False),
                }

        logging.config.dictConfig(logging_config)

    @classmethod
    def _setup_standard_logging(cls, config: LoggerConfig):
        """Development/standard logging setup with date-based rotation"""
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

        # File handler with date-based rotation
        if config.log_file:
            # Use the base log file path without adding extra timestamps
            file_handler = TimestampedRotatingFileHandler(
                filename=config.log_file,  # Use as-is from config
                when=(
                    config.rotation.get("when", "midnight")
                    if config.rotation
                    else "midnight"
                ),
                interval=config.rotation.get("interval", 1) if config.rotation else 1,
                backupCount=(
                    config.rotation.get("backup_count", 7) if config.rotation else 7
                ),
                compress_old_logs=(
                    config.rotation.get("compress", True) if config.rotation else True
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

    @classmethod
    def cleanup_old_logs(cls, log_dir: str, retention_days: int = 30):
        """Clean up old log files beyond retention period"""
        try:
            log_path = Path(log_dir)
            if not log_path.exists():
                return

            cutoff_date = datetime.now() - timedelta(days=retention_days)

            for log_file in log_path.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink(missing_ok=True)
                    print(f"Cleaned up old log file: {log_file}")

        except Exception as e:
            print(f"Error cleaning up logs: {e}")


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

        if self.logger.handlers and hasattr(self.logger.handlers[0], "formatter"):
            # For JSON logging, add context as structured data
            if "json" in str(type(self.logger.handlers[0].formatter)).lower():
                extra = {"context": context}
                getattr(self.logger, level)(message, extra=extra)
                return

        # For standard logging, add context as string
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
        metrics_data = {"type": "model_metrics", "metrics": metrics, "epoch": epoch}
        self.info("Model metrics logged", **metrics_data)

    def log_data_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information"""
        self.info("Dataset information", type="data_info", **dataset_info)

    def log_model_artifact(self, artifact_path: str, artifact_type: str):
        """Log model artifacts"""
        self.info(
            f"Model artifact saved: {artifact_path}",
            type="artifact",
            artifact_type=artifact_type,
            artifact_path=artifact_path,
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
            "Prediction completed",
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
