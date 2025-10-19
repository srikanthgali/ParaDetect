"""
ParaDetect package initialization with proper logging setup
This is the APPLICATION BOOTSTRAP - run when package is imported
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import logging

for logger_name in ["botocore", "boto3", "urllib3", "s3transfer", "sagemaker"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def _configure_dev_environment_optimizations():
    """Configure environment variables and PyTorch settings based on deployment context"""
    environment = (
        os.getenv("ENVIRONMENT") or os.getenv("PARA_DETECT_ENV") or "development"
    )

    if environment == "development":
        # 🍎 Local macOS/MPS development optimizations
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

        import torch

        torch.set_num_threads(1)
    else:
        # Default minimal setup
        import torch

    return environment


# Apply environment-specific optimizations
current_environment = _configure_dev_environment_optimizations()

# Add src to path if not already there
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Global configuration manager instance (singleton)
config_manager = None
_initialized = False
_initialization_lock = threading.Lock()


def initialize_app(environment: Optional[str] = None, force_reinit: bool = False):
    """Initialize application with proper logging and configuration"""
    global config_manager, _initialized

    # Thread-safe initialization
    with _initialization_lock:
        if _initialized and not force_reinit:
            return config_manager

        try:
            # Import here to avoid circular imports
            from para_detect.core.config_manager import ConfigurationManager
            from para_detect.core.logger import LoggerFactory

            # Initialize configuration manager (singleton)
            if config_manager is None or force_reinit:
                config_manager = ConfigurationManager()

            # Get environment - prioritize parameter, then env var, then config
            resolved_environment = (
                environment
                or os.getenv("ENVIRONMENT")
                or config_manager.environment
                or "development"
            )

            # Setup logging based on configuration
            logging_config = config_manager.get_logging_config()
            LoggerFactory.setup_logging(logging_config, resolved_environment)

            # Setup log cleanup for production environments
            if resolved_environment == "production":
                _schedule_log_cleanup(logging_config)

            # Only log initialization once
            from para_detect.core.logger import get_logger

            logger = get_logger("para_detect.init")
            logger.info(f"ParaDetect initialized - Environment: {resolved_environment}")
            # logger.debug(f"Config loaded from: {config_manager.base_config}")

            _initialized = True
            return config_manager

        except Exception as e:
            print(f"Error initializing ParaDetect: {e}")
            # Create minimal fallback configuration
            try:
                from para_detect.entities.logger_config import LoggerConfig
                from para_detect.core.logger import LoggerFactory

                fallback_config = LoggerConfig(
                    level="INFO",
                    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                    log_dir="artifacts/logs/",
                    rotation={
                        "when": "midnight",
                        "interval": 1,
                        "backup_count": 3,
                        "compress": True,
                        "retention_days": 7,
                    },
                    structured=False,
                    json_format=False,
                )
                LoggerFactory.setup_logging(fallback_config, "development")
                _initialized = True

                # Try to get a logger for error reporting
                try:
                    logger = LoggerFactory.get_logger("para_detect.init")
                    logger.warning(
                        "Initialized with fallback configuration due to error"
                    )
                except Exception:
                    print("Warning: Initialized with fallback configuration")

            except Exception as fallback_error:
                print(
                    f"Critical: Even fallback initialization failed: {fallback_error}"
                )

            return config_manager


def _schedule_log_cleanup(logging_config):
    """Schedule periodic log cleanup for production environments"""
    try:

        def cleanup_worker():
            """Background worker for log cleanup"""
            from para_detect.core.logger import LoggerFactory

            while True:
                try:
                    retention_days = logging_config.retention_days
                    LoggerFactory.cleanup_old_logs(
                        logging_config.log_dir, retention_days
                    )

                    # Log cleanup success
                    logger = LoggerFactory.get_logger("para_detect.cleanup")
                    logger.debug(
                        f"Log cleanup completed - retention: {retention_days} days"
                    )

                    # Run cleanup daily
                    time.sleep(86400)  # 24 hours

                except Exception as e:
                    print(f"Error in log cleanup worker: {e}")
                    time.sleep(3600)  # Retry in 1 hour on error

        # Start cleanup worker in background daemon thread
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.name = "ParaDetect-LogCleanup"
        cleanup_thread.start()

        print("Log cleanup worker started for production environment")

    except Exception as e:
        print(f"Warning: Could not schedule log cleanup: {e}")


def get_config_manager():
    """Get configuration manager, initializing if needed (singleton access)"""
    global config_manager

    if config_manager is None:
        config_manager = initialize_app()

    return config_manager


def reinitialize_app(environment: Optional[str] = None):
    """Force reinitialize the application (useful for testing or environment changes)"""
    global _initialized
    _initialized = False
    return initialize_app(environment, force_reinit=True)


def get_application_info():
    """Get current application initialization status"""
    global config_manager, _initialized

    return {
        "initialized": _initialized,
        "config_manager_loaded": config_manager is not None,
        "environment": config_manager.environment if config_manager else "unknown",
        "project_root": str(project_root),
    }


# Import functions but delay initialization until actually needed
def _lazy_import():
    """Lazy import of logger functions to avoid circular dependencies"""
    try:
        from para_detect.core.logger import get_logger, get_mlops_logger

        return get_logger, get_mlops_logger
    except ImportError as e:
        print(f"Warning: Could not import logger functions: {e}")

        # Provide fallback functions
        import logging

        def fallback_get_logger(name="para_detect"):
            return logging.getLogger(name)

        def fallback_get_mlops_logger(component, experiment_id=None):
            return fallback_get_logger(f"para_detect.{component}")

        return fallback_get_logger, fallback_get_mlops_logger


# Lazy loading of logger functions
def get_logger(name: str = "para_detect"):
    """Get a standard logger (lazy loaded)"""
    get_logger_fn, _ = _lazy_import()
    return get_logger_fn(name)


def get_mlops_logger(component: str, experiment_id: Optional[str] = None):
    """Get an MLOps-specific logger (lazy loaded)"""
    _, get_mlops_logger_fn = _lazy_import()
    return get_mlops_logger_fn(component, experiment_id)


# Auto-initialize when module is imported (but only if not in test environment)
def _auto_initialize():
    """Auto-initialize application on import if appropriate"""
    try:
        # Skip auto-initialization in test environments or if explicitly disabled
        if (
            os.getenv("PARA_DETECT_NO_AUTO_INIT") == "1"
            or os.getenv("PYTEST_CURRENT_TEST")
            or "pytest" in sys.modules
        ):
            print("Skipping auto-initialization (test environment or disabled)")
            return

        # Auto-initialize with environment detection
        initialize_app()

    except Exception as e:
        print(f"Warning: Auto-initialization failed: {e}")
        print("Manual initialization may be required")


# Export commonly used functions and classes
__all__ = [
    "get_logger",
    "get_mlops_logger",
    "get_config_manager",
    "initialize_app",
    "reinitialize_app",
    "get_application_info",
]

# Auto-initialize on import
_auto_initialize()
