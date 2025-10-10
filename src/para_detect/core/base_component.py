"""Base classes for ParaDetect pipeline"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import logging
from box import ConfigBox
from para_detect import get_logger

from para_detect.core.exceptions import ParaDetectException, ConfigurationError


class BaseComponent(ABC):
    """Base class for all pipeline components"""

    def __init__(self, config: Union[ConfigBox, Dict[str, Any]]):
        """
        Initialize base component.

        Args:
            config: Configuration object (ConfigBox preferred) or dictionary
        """
        # Convert to ConfigBox if it's a dictionary
        if isinstance(config, dict):
            self.config = ConfigBox(config)
        else:
            self.config = config

        # Setup logger for the component
        self.logger = get_logger(self.__class__.__name__)

        # Validate configuration on initialization
        if not self.validate_config():
            raise ConfigurationError(
                f"Invalid configuration for {self.__class__.__name__}"
            )

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the component logic"""
        pass

    def validate_config(self) -> bool:
        """
        Validate component configuration.
        Override in subclasses for specific validation logic.

        Returns:
            bool: True if configuration is valid
        """
        return self.config is not None

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Safely get configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation like 'data.raw_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            keys = key.split(".")
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, AttributeError):
            return default
