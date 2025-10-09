import os
from box import ConfigBox
from para_detect.entities.logger_config import LoggerConfig, ComponentLoggerConfig
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from para_detect.constants import *
from para_detect.utils.helpers import read_yaml


@dataclass
class ConfigurationManager:
    """Main configuration management for ParaDetect"""

    def __init__(self, base_config_file_path=BASE_CONFIG_FILE_PATH):
        self.base_config = read_yaml(base_config_file_path)
        self.environment = self._get_environment()
        self._apply_environment_overrides()
        # Load additional configuration files
        self._load_additional_configs()

    def _get_environment(self) -> str:
        """Get current environment from config or environment variable"""
        return (
            os.getenv("ENVIRONMENT")
            or self.base_config.project.environment
            or "development"
        )

    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if (
            hasattr(self.base_config, "environments")
            and self.environment in self.base_config.environments
        ):
            env_config = self.base_config.environments[self.environment]

            # Deep merge environment overrides
            for section, values in env_config.items():
                if section in self.base_config:
                    self.base_config[section].update(values)

    def _load_additional_configs(self):
        """Load additional configuration files"""
        try:
            # Load model configs
            self.deberta_config = read_yaml(DEBERTA_CONFIG_FILE_PATH)
            self.training_config = read_yaml(TRAINING_CONFIG_FILE_PATH)

            # Load deployment configs based on environment
            if self.environment == "production":
                self.deployment_config = read_yaml(AWS_CONFIG_FILE_PATH)
            elif self.environment == "development":
                self.deployment_config = read_yaml(LOCAL_CONFIG_FILE_PATH)
            else:
                # Default to AWS config
                self.deployment_config = read_yaml(AWS_CONFIG_FILE_PATH)

        except Exception as e:
            print(f"Warning: Error loading additional configs: {e}")
            # Create empty configs as fallback
            self.deberta_config = ConfigBox({})
            self.training_config = ConfigBox({})
            self.deployment_config = ConfigBox({})

    def get_logging_config(self) -> LoggerConfig:
        """
        Load logging configuration from config.yaml into LoggerConfig entity
        """
        try:
            logging_config = self.base_config.logging

            # Extract base logging configuration
            level = logging_config.get("level", "INFO")
            format_str = logging_config.get(
                "format",
                "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
            )
            log_dir = logging_config.get("log_dir", "artifacts/logs/")

            # Extract component-specific loggers
            component_loggers = None
            if hasattr(logging_config, "loggers"):
                component_loggers = dict(logging_config.loggers)

            # Extract rotation settings
            rotation_config = None
            if hasattr(logging_config, "rotation"):
                rotation_config = dict(logging_config.rotation)

            # Extract structured logging settings
            structured = logging_config.structured or False
            json_format = logging_config.json_format or False

            return LoggerConfig(
                level=level,
                format=format_str,
                log_dir=log_dir,
                loggers=component_loggers,
                rotation=rotation_config,
                structured=structured,
                json_format=json_format,
            )

        except Exception as e:
            # Return default config if there's an error
            print(f"Warning: Error loading logging config, using defaults: {e}")
            return LoggerConfig(
                level="INFO",
                format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
                log_dir="artifacts/logs/",
            )

    def get_component_logger_config(
        self, component_name: str
    ) -> Optional[ComponentLoggerConfig]:
        """Get configuration for a specific component logger"""
        logging_config = self.get_logging_config()

        if logging_config.loggers and component_name in logging_config.loggers:
            component_config = logging_config.loggers[component_name]
            return ComponentLoggerConfig(
                name=component_name,
                level=component_config.level or logging_config.level,
                handlers=component_config.handlers or ["console", "file"],
                propagate=component_config.propagate or False,
            )

        return None

    def get_model_config(self) -> ConfigBox:
        """Get model configuration from deberta_config.yaml"""
        return self.deberta_config

    def get_training_config(self) -> ConfigBox:
        """Get training configuration from training_config.yaml"""
        return self.training_config

    def get_data_config(self) -> ConfigBox:
        """Get data configuration from base config"""
        return self.base_config.get("data", ConfigBox({}))

    def get_deployment_config(self) -> ConfigBox:
        """Get deployment configuration from environment-specific file"""
        return self.deployment_config

    def get_project_config(self) -> ConfigBox:
        """Get project configuration from base config"""
        return self.base_config.get("project", ConfigBox({}))

    def get_monitoring_config(self) -> ConfigBox:
        """Get monitoring configuration from base config"""
        return self.base_config.get("monitoring", ConfigBox({}))

    # Additional convenience methods for specific configs
    def get_deberta_config(self) -> ConfigBox:
        """Get DeBERTa specific configuration"""
        return self.deberta_config

    def get_aws_config(self) -> ConfigBox:
        """Get AWS deployment configuration"""
        try:
            aws_config = read_yaml(AWS_CONFIG_FILE_PATH)
            return aws_config
        except Exception:
            return ConfigBox({})

    def get_local_config(self) -> ConfigBox:
        """Get local deployment configuration"""
        try:
            local_config = read_yaml(LOCAL_CONFIG_FILE_PATH)
            return local_config
        except Exception:
            return ConfigBox({})
