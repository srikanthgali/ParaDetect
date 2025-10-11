import os
from box import ConfigBox
from para_detect.entities import (
    LoggerConfig,
    ComponentLoggerConfig,
    DataIngestionConfig,
    DataPreprocessingConfig,
    DataValidationConfig,
    PipelineConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig,
    ModelRegistrationConfig,
    ModelValidationConfig,
    InferenceConfig,
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
        """Apply environment-specific configuration overrides with deep merge support"""
        if (
            hasattr(self.base_config, "environments")
            and self.environment in self.base_config.environments
        ):
            env_config = self.base_config.environments[self.environment]

            # Deep merge environment overrides
            for section, values in env_config.items():
                if section in self.base_config:
                    if isinstance(values, dict):
                        # Handle nested configurations like logging.rotation
                        current_section = self.base_config[section]
                        if isinstance(current_section, dict):
                            # Deep merge for nested dictionaries
                            self._deep_merge_dict(current_section, values)
                        else:
                            # Replace if not a dict
                            self.base_config[section] = values
                    else:
                        # Replace primitive values
                        self.base_config[section] = values
                else:
                    # Add new sections
                    self.base_config[section] = values

    def _deep_merge_dict(self, target: dict, source: dict):
        """Recursively merge source dict into target dict"""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                self._deep_merge_dict(target[key], value)
            else:
                # Replace or add the value
                target[key] = value

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
                # Default to local config for development
                self.deployment_config = read_yaml(LOCAL_CONFIG_FILE_PATH)

        except Exception as e:
            print(f"Warning: Error loading additional configs: {e}")
            # Create empty configs as fallback
            self.deberta_config = ConfigBox({})
            self.training_config = ConfigBox({})
            self.deployment_config = ConfigBox({})

    def get_logging_config(self) -> LoggerConfig:
        """Load logging configuration from config.yaml into LoggerConfig entity"""
        try:
            logging_config = self.base_config.logging
            level = logging_config.level
            format_str = logging_config.format
            log_dir = logging_config.log_dir

            # Handle component loggers
            component_loggers = None
            if hasattr(logging_config, "loggers"):
                component_loggers = dict(logging_config.loggers)

            # Handle rotation configuration with new fields
            rotation_config = None
            if hasattr(logging_config, "rotation"):
                rotation_config = dict(logging_config.rotation)
                # Ensure all rotation fields have defaults
                rotation_defaults = {
                    "when": "midnight",
                    "interval": 1,
                    "backup_count": 7,
                    "compress": True,
                    "retention_days": 30,
                }
                for key, default_value in rotation_defaults.items():
                    if key not in rotation_config:
                        rotation_config[key] = default_value

            # Handle structured logging options
            structured = getattr(logging_config, "structured", False)
            json_format = getattr(logging_config, "json_format", False)

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
                rotation={
                    "when": "midnight",
                    "interval": 1,
                    "backup_count": 7,
                    "compress": True,
                    "retention_days": 30,
                },
                structured=False,
                json_format=False,
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
                handlers=component_config.handlers,
                propagate=component_config.propagate,
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
            pipeline_config = self.base_config.pipeline
            state_files_config = pipeline_config.state_files
            state_files = {key: Path(path) for key, path in state_files_config.items()}
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
            for key, default_path in default_state_files.items():
                if key not in state_files:
                    state_files[key] = default_path
            return PipelineConfig(
                artifacts_dir=Path(pipeline_config.artifacts_dir),
                checkpoints_dir=Path(pipeline_config.checkpoints_dir),
                state_files=state_files,
                enable_state_persistence=pipeline_config.enable_state_persistence,
                state_auto_save=pipeline_config.state_auto_save,
                state_retention_days=pipeline_config.state_retention_days,
                enable_pipeline_locks=pipeline_config.enable_pipeline_locks,
            )

        except Exception as e:
            print(f"Warning: Error loading pipeline config, using defaults: {str(e)}")
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
        """Get data ingestion configuration from config.yaml"""
        try:
            # Get data ingestion section from config.yaml
            ingestion_config = self.base_config.data_ingestion

            # Create the entity with proper type conversion
            return DataIngestionConfig(
                dataset_name=ingestion_config.dataset_name,
                source_type=ingestion_config.source_type,
                raw_data_dir=Path(ingestion_config.raw_data_dir),
                dataset_filename=ingestion_config.dataset_filename,
                sample_size=ingestion_config.sample_size,
                random_state=ingestion_config.random_state,
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
                random_state=preprocessing_config.random_state,
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

    # ML Component Configurations
    def get_model_training_config(self) -> ModelTrainingConfig:
        """Get model training configuration"""
        try:
            training_config = self.training_config.training
            model_config = self.deberta_config.model
            dataset_config = self.training_config.dataset
            peft_config = None
            if training_config.use_peft:
                lora_config = getattr(self.deberta_config, "lora", ConfigBox({}))
                if lora_config:
                    peft_config = {
                        "task_type": lora_config.task_type,
                        "r": lora_config.r,
                        "lora_alpha": lora_config.lora_alpha,
                        "lora_dropout": lora_config.lora_dropout,
                        "bias": lora_config.bias,
                        "target_modules": lora_config.target_modules
                        or DEFAULT_LORA_TARGET_MODULES,
                        "inference_mode": lora_config.inference_mode,
                    }
            return ModelTrainingConfig(
                model_name_or_path=model_config.model_name_or_path,
                tokenizer_name_or_path=model_config.tokenizer_name_or_path
                or model_config.model_name_or_path,
                num_labels=model_config.num_labels,
                output_dir=Path(training_config.output_dir),
                num_train_epochs=training_config.num_train_epochs,
                per_device_train_batch_size=training_config.per_device_train_batch_size,
                per_device_eval_batch_size=training_config.per_device_eval_batch_size,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                learning_rate=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                warmup_steps=training_config.warmup_steps,
                warmup_ratio=training_config.warmup_ratio,
                max_grad_norm=training_config.max_grad_norm,
                fp16=training_config.fp16,
                bf16=training_config.bf16,
                max_length=dataset_config.max_length,
                text_column=dataset_config.text_column,
                label_column=dataset_config.label_column,
                train_path=dataset_config.train_path,
                validation_split=dataset_config.validation_split,
                test_split=dataset_config.test_split,
                eval_strategy=training_config.evaluation.eval_strategy,
                eval_steps=training_config.evaluation.eval_steps,
                save_strategy=training_config.evaluation.save_strategy,
                save_steps=training_config.evaluation.save_steps,
                save_total_limit=training_config.evaluation.save_total_limit,
                load_best_model_at_end=training_config.evaluation.load_best_model_at_end,
                metric_for_best_model=training_config.evaluation.metric_for_best_model,
                greater_is_better=training_config.evaluation.greater_is_better,
                early_stopping_patience=training_config.evaluation.early_stopping_patience,
                early_stopping_threshold=training_config.evaluation.early_stopping_threshold,
                resume_from_checkpoint=training_config.resume_from_checkpoint,
                checkpoint_interval_steps=training_config.checkpoint_interval_steps,
                use_peft=training_config.use_peft,
                peft_config=peft_config,
                logging_steps=training_config.logging_steps,
                report_to=training_config.report_to,
                run_name=training_config.run_name,
                seed=training_config.seed,
                device_preference=training_config.device_preference,
                device_map=model_config.loading.device_map,
                torch_dtype_loading=model_config.loading.torch_dtype_loading,
                low_cpu_mem_usage=model_config.loading.low_cpu_mem_usage,
                trust_remote_code=model_config.loading.trust_remote_code,
                save_model=model_config.saving.save_model,
                torch_dtype_saving=model_config.saving.torch_dtype_saving,
                safe_serialization=model_config.saving.safe_serialization,
                save_metadata=model_config.saving.save_metadata,
                save_config=model_config.saving.save_config,
                save_tokenizer=model_config.saving.save_tokenizer,
                create_model_card=model_config.saving.create_model_card,
                save_training_args=model_config.saving.save_training_args,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load model training config: {str(e)}"
            ) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Get model evaluation configuration"""
        try:
            eval_config = self.training_config.evaluation

            return ModelEvaluationConfig(
                metrics=eval_config.metrics,
                save_best_by=eval_config.save_best_by,
                eval_batch_size=eval_config.eval_batch_size,
                max_length=dataset_config.max_length,
                evaluation_output_dir=Path(eval_config.evaluation_output_dir),
                save_confusion_matrix=eval_config.save_confusion_matrix,
                save_classification_report=eval_config.save_classification_report,
                save_roc_curve=eval_config.save_roc_curve,
                save_precision_recall_curve=eval_config.save_precision_recall_curve,
                perform_calibration_analysis=eval_config.perform_calibration_analysis,
                calibration_bins=eval_config.calibration_bins,
                compute_per_class_metrics=eval_config.compute_per_class_metrics,
                device_preference=eval_config.device_preference,
                text_column=dataset_config.text_column,
                label_column=dataset_config.label_column,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load model evaluation config: {str(e)}"
            ) from e

    def get_model_validation_config(self) -> ModelValidationConfig:
        """Get model validation configuration"""
        try:
            validation_config = self.training_config.validation
            return ModelValidationConfig(
                min_accuracy=validation_config.min_accuracy,
                min_f1=validation_config.min_f1,
                min_precision=validation_config.min_precision,
                min_recall=validation_config.min_recall,
                min_auc=validation_config.min_auc,
                max_brier_score=validation_config.max_brier_score,
                max_ece=validation_config.max_ece,
                calibration_bins=validation_config.calibration_bins,
                min_per_class_f1=validation_config.min_per_class_f1,
                max_class_imbalance_ratio=validation_config.max_class_imbalance_ratio,
                perform_fairness_checks=validation_config.perform_fairness_checks,
                max_demographic_parity_diff=validation_config.max_demographic_parity_diff,
                max_equalized_odds_diff=validation_config.max_equalized_odds_diff,
                check_prediction_distribution=validation_config.check_prediction_distribution,
                max_prediction_skew=validation_config.max_prediction_skew,
                allowed_label_values=validation_config.allowed_label_values,
                validation_output_dir=Path(validation_config.validation_output_dir),
                save_validation_report=validation_config.save_validation_report,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load model validation config: {str(e)}"
            ) from e

    def get_model_registration_config(self) -> ModelRegistrationConfig:
        """Get model registration configuration"""
        try:
            registration_config = self.training_config.registration
            return ModelRegistrationConfig(
                use_hf=registration_config.use_hf,
                hf_repo_id=registration_config.hf_repo_id,
                hf_token=registration_config.hf_token,
                push_to_hub=registration_config.push_to_hub,
                private_repo=registration_config.private_repo,
                use_mlflow=registration_config.use_mlflow,
                mlflow_tracking_uri=registration_config.mlflow_tracking_uri,
                mlflow_experiment_name=registration_config.mlflow_experiment_name,
                mlflow_model_name=registration_config.mlflow_model_name,
                mlflow_stage=registration_config.mlflow_stage,
                use_sagemaker=registration_config.use_sagemaker,
                sagemaker_role=registration_config.sagemaker_role,
                sagemaker_bucket=registration_config.sagemaker_bucket,
                sagemaker_model_name=registration_config.sagemaker_model_name,
                local_registry_dir=Path(registration_config.local_registry_dir),
                model_description=registration_config.model_description,
                model_tags=registration_config.model_tags,
                license=registration_config.license,
                require_validation_pass=registration_config.require_validation_pass,
                dry_run=registration_config.dry_run,
                force_overwrite=registration_config.force_overwrite,
                generate_model_card=registration_config.generate_model_card,
                model_card_template=registration_config.model_card_template,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load model registration config: {str(e)}"
            ) from e

    def get_inference_config(self) -> InferenceConfig:
        """Get inference configuration"""
        try:
            inference_config = self.base_config.inference
            return InferenceConfig(
                model_path=inference_config.model_path,
                tokenizer_path=inference_config.tokenizer_path,
                device_preference=inference_config.device_preference,
                batch_size=inference_config.batch_size,
                max_length=inference_config.max_length,
                text_column=inference_config.text_column,
                preprocessing_enabled=inference_config.preprocessing_enabled,
                include_probabilities=inference_config.include_probabilities,
                include_confidence=inference_config.include_confidence,
                confidence_threshold=inference_config.confidence_threshold,
                enable_monitoring=inference_config.enable_monitoring,
                log_predictions=inference_config.log_predictions,
                chunk_size=inference_config.chunk_size,
                progress_bar=inference_config.progress_bar,
                skip_errors=inference_config.skip_errors,
                max_retries=inference_config.max_retries,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load inference config: {str(e)}"
            ) from e

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'training.learning_rate')"""
        try:
            keys = key.split(".")
            value = self.base_config

            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value
        except Exception:
            return default
