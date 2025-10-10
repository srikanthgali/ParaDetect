import os
from box import ConfigBox
from para_detect.entities import (
    LoggerConfig,
    ComponentLoggerConfig,
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataValidationConfig,
    PipelineConfig,
)
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from para_detect.constants import *
from para_detect.core.exceptions import ConfigurationError
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

    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration with pipeline-specific state files"""
        try:
            pipeline_config = self.base_config.get("pipeline", ConfigBox({}))

            # Extract state files configuration
            state_files_config = pipeline_config.state_files or {}
            state_files = {key: Path(path) for key, path in state_files_config.items()}

            # Provide defaults if not specified
            default_state_files = {
                "data_pipeline": Path("artifacts/states/data_pipeline_state.json"),
                "training_pipeline": Path(
                    "artifacts/states/training_pipeline_state.json"
                ),
                "inference_pipeline": Path(
                    "artifacts/states/inference_pipeline_state.json"
                ),
                "monitoring_pipeline": Path(
                    "artifacts/states/monitoring_pipeline_state.json"
                ),
                "deployment_pipeline": Path(
                    "artifacts/states/deployment_pipeline_state.json"
                ),
            }

            # Merge with defaults
            for key, default_path in default_state_files.items():
                if key not in state_files:
                    state_files[key] = default_path

            return PipelineConfig(
                artifacts_dir=Path(pipeline_config.artifacts_dir or "artifacts/"),
                checkpoints_dir=Path(
                    pipeline_config.checkpoints_dir or "artifacts/checkpoints/"
                ),
                state_files=state_files,
                enable_state_persistence=pipeline_config.enable_state_persistence
                or True,
                state_auto_save=pipeline_config.state_auto_save or True,
                state_retention_days=pipeline_config.state_retention_days or 30,
                enable_pipeline_locks=pipeline_config.enable_pipeline_locks or True,
            )

        except Exception as e:
            self.logger.warning(
                f"Error loading pipeline config, using defaults: {str(e)}"
            )
            # Return default config with pipeline-specific state files
            default_state_files = {
                "data_pipeline": Path("artifacts/states/data_pipeline_state.json"),
                "training_pipeline": Path(
                    "artifacts/states/training_pipeline_state.json"
                ),
                "inference_pipeline": Path(
                    "artifacts/states/inference_pipeline_state.json"
                ),
                "monitoring_pipeline": Path(
                    "artifacts/states/monitoring_pipeline_state.json"
                ),
                "deployment_pipeline": Path(
                    "artifacts/states/deployment_pipeline_state.json"
                ),
            }

            return PipelineConfig(
                artifacts_dir=Path("artifacts/"),
                checkpoints_dir=Path("artifacts/checkpoints/"),
                state_files=default_state_files,
                enable_state_persistence=True,
                state_auto_save=True,
                state_retention_days=30,
                enable_pipeline_locks=True,
            )

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

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get data ingestion configuration from config.yaml

        Returns:
            DataIngestionConfig: Configured data ingestion entity
        """
        try:
            # Get data ingestion section from config.yaml
            ingestion_config = self.base_config.data_ingestion

            # Create the entity with proper type conversion
            return DataIngestionConfig(
                dataset_name=ingestion_config.dataset_name,
                source_type=ingestion_config.source_type,
                raw_data_dir=Path(ingestion_config.raw_data_dir),
                dataset_filename=ingestion_config.dataset_filename,
                sample_size=ingestion_config.sample_size or None,
                random_state=ingestion_config.random_state or DEFAULT_RANDOM_STATE,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load data ingestion config: {str(e)}"
            ) from e

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        """Get data preprocessing configuration"""
        try:
            preprocessing_config = self.base_config.data_preprocessing

            return DataPreprocessingConfig(
                text_column=preprocessing_config.text_column,
                label_column=preprocessing_config.label_column,
                source_column=preprocessing_config.source_column,
                remove_duplicates=preprocessing_config.remove_duplicates,
                min_text_length=preprocessing_config.min_text_length,
                max_text_length=preprocessing_config.max_text_length,
                lowercase=preprocessing_config.lowercase,
                strip_whitespace=preprocessing_config.strip_whitespace,
                remove_special_chars=preprocessing_config.remove_special_chars,
                balance_classes=preprocessing_config.balance_classes,
                processed_data_dir=Path(preprocessing_config.processed_data_dir),
                processed_filename=preprocessing_config.processed_filename,
                random_state=preprocessing_config.random_state or DEFAULT_RANDOM_STATE,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load data preprocessing config: {str(e)}"
            ) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        """Get data validation configuration"""
        try:
            validation_config = self.base_config.data_validation

            return DataValidationConfig(
                expected_columns=validation_config.expected_columns,
                required_columns=validation_config.required_columns,
                text_column=validation_config.text_column,
                label_column=validation_config.label_column,
                min_text_length=validation_config.min_text_length,
                max_text_length=validation_config.max_text_length,
                expected_labels=validation_config.expected_labels,
                min_samples_per_class=validation_config.min_samples_per_class,
                max_null_percentage=validation_config.max_null_percentage,
                validation_report_dir=Path(validation_config.validation_report_dir),
                report_filename=validation_config.report_filename,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load data validation config: {str(e)}"
            ) from e
