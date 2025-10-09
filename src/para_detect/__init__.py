"""
ParaDetect package initialization with proper logging setup
This is the APPLICATION BOOTSTRAP - run when package is imported
"""

import os
import sys
from pathlib import Path

# Add src to path if not already there
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Global configuration manager instance
config_manager = None
_initialized = False


def initialize_app():
    """Initialize application with proper logging and configuration"""
    global config_manager, _initialized

    if _initialized:
        return config_manager

    try:
        # Import here to avoid circular imports
        from para_detect.core.config_manager import ConfigurationManager
        from para_detect.core.logger import LoggerFactory

        # Initialize configuration manager
        config_manager = ConfigurationManager()

        # Get environment
        environment = os.getenv("ENVIRONMENT", config_manager.environment)

        # Setup logging based on configuration
        logging_config = config_manager.get_logging_config()
        LoggerFactory.setup_logging(logging_config, environment)

        # Only log initialization once
        from para_detect.core.logger import get_logger

        logger = get_logger("para_detect.init")
        logger.info(f"ParaDetect initialized - Environment: {environment}")

        _initialized = True
        return config_manager

    except Exception as e:
        print(f"Error initializing ParaDetect: {e}")
        # Create minimal fallback configuration
        from para_detect.entities.logger_config import LoggerConfig
        from para_detect.core.logger import LoggerFactory

        fallback_config = LoggerConfig(
            level="INFO",
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            log_dir="artifacts/logs/",
        )
        LoggerFactory.setup_logging(fallback_config, "development")
        _initialized = True
        return None


# Lazy initialization - only when actually needed
def get_config_manager():
    """Get configuration manager, initializing if needed"""
    global config_manager
    if config_manager is None:
        config_manager = initialize_app()
    return config_manager


# Import functions but delay initialization
from para_detect.core.logger import get_logger, get_mlops_logger

# Export commonly used functions and classes
__all__ = ["get_logger", "get_mlops_logger", "get_config_manager", "initialize_app"]
