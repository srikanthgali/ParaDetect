"""Base classes for ParaDetect pipeline"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import logging
from box import ConfigBox
from para_detect import get_logger

from para_detect.core.exceptions import ParaDetectException, ConfigurationError


class BaseModel(ABC):
    """Base class for ML models"""

    def __init__(self, config: Union[ConfigBox, Dict[str, Any]]):
        """
        Initialize base model.

        Args:
            config: Configuration object (ConfigBox preferred) or dictionary
        """
        # Convert to ConfigBox if it's a dictionary
        if isinstance(config, dict):
            self.config = ConfigBox(config)
        else:
            self.config = config

        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_model(self) -> None:
        """Load the model"""
        pass

    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction"""
        pass

    @abstractmethod
    def train(self, train_data, val_data) -> None:
        """Train the model"""
        pass

    def validate_model_config(self) -> bool:
        """
        Validate model-specific configuration.
        Override in subclasses for specific validation logic.

        Returns:
            bool: True if configuration is valid
        """
        return self.config is not None
