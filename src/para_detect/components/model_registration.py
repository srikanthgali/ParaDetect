"""
Model Registration Component for ParaDetect
Handles model publishing to various registries (HuggingFace Hub, MLflow, SageMaker)
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os
import textwrap

# Hugging Face Hub
try:
    from huggingface_hub import HfApi, Repository, login

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# MLflow
try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# AWS SageMaker
try:
    import boto3
    import sagemaker

    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False

from para_detect.utils.helpers import convert_to_serializable
from para_detect.utils.s3_manager import S3Manager
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from para_detect.entities.model_registration_config import ModelRegistrationConfig
from para_detect.core.exceptions import ModelRegistrationError
from para_detect import get_logger


class ModelRegistrar:
    """
    Model registration component supporting multiple registries.

    Supports:
    - Hugging Face Hub (public/private repositories)
    - MLflow Model Registry
    - AWS SageMaker Model Registry
    - Local model registry
    - Model card generation
    - Version management
    """

    def __init__(
        self,
        config: ModelRegistrationConfig,
        run_id: str,
        s3_manager: Optional[S3Manager] = None,
    ):
        """
        Initialize model registrar.

        Args:
            config: Registration configuration
        """
        self.config = config
        self.run_id = run_id
        self.logger = get_logger(self.__class__.__name__)

        # Keep S3 manager for S3 operations
        self.s3_manager = s3_manager
        self.s3_enabled = s3_manager is not None

        # Registration results
        self.registration_results = {}

    def register_model(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        validation_passed: bool = True,
    ) -> Dict[str, Any]:
        """
        Register model across configured registries.
        """
        try:
            self.logger.info("📦 Starting model registration...")

            self.config.registration_output_dir.mkdir(parents=True, exist_ok=True)

            # Check validation requirement
            if self.config.require_validation_pass and not validation_passed:
                self.logger.warning(
                    "⚠️ Model failed validation but registration required validation pass"
                )
                raise ModelRegistrationError(
                    "Model validation failed and require_validation_pass is True"
                )

            # Validate inputs
            if not Path(model_path).exists():
                raise ModelRegistrationError(f"Model path does not exist: {model_path}")

            tokenizer_path = tokenizer_path or model_path
            metadata = metadata or {}

            # Ensure metadata is JSON serializable from the start
            metadata = convert_to_serializable(metadata)

            # Initialize results
            results = {
                "success": True,
                "registrations": {},
                "errors": {},
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "dry_run": self.config.dry_run,
            }

            # Register to each configured registry
            registration_attempted = False

            if self.config.use_hf and HF_AVAILABLE:
                registration_attempted = True
                try:
                    hf_result = self._register_to_huggingface(
                        model_path, tokenizer_path, metadata
                    )
                    results["registrations"]["huggingface"] = hf_result
                    self.logger.info("✅ HuggingFace registration completed")
                except Exception as e:
                    error_msg = f"HuggingFace registration failed: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    results["errors"]["huggingface"] = error_msg
                    results["success"] = False

            if self.config.use_mlflow and MLFLOW_AVAILABLE:
                registration_attempted = True
                try:
                    mlflow_result = self._register_to_mlflow(model_path, metadata)
                    results["registrations"]["mlflow"] = mlflow_result
                    self.logger.info("✅ MLflow registration completed")
                except Exception as e:
                    error_msg = f"MLflow registration failed: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    results["errors"]["mlflow"] = error_msg
                    results["success"] = False

            if self.config.use_sagemaker and SAGEMAKER_AVAILABLE:
                registration_attempted = True
                try:
                    sm_result = self._register_to_sagemaker(model_path, metadata)
                    results["registrations"]["sagemaker"] = sm_result
                    self.logger.info("✅ SageMaker registration completed")
                except Exception as e:
                    error_msg = f"SageMaker registration failed: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    results["errors"]["sagemaker"] = error_msg
                    results["success"] = False

            # Always register to local registry
            try:
                local_result = self._register_to_local(
                    model_path, tokenizer_path, metadata
                )
                results["registrations"]["local"] = local_result
                self.logger.info("✅ Local registration completed")
            except Exception as e:
                error_msg = f"Local registration failed: {str(e)}"
                self.logger.error(f"❌ {error_msg}")
                results["errors"]["local"] = error_msg
                results["success"] = False

            # Generate model card
            if self.config.generate_model_card:
                try:
                    model_card_path = self._generate_model_card(model_path, metadata)
                    results["model_card_path"] = str(model_card_path)
                    self.logger.info(f"📄 Model card generated: {model_card_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Model card generation failed: {str(e)}")

            self.registration_results = results

            results_file_path = (
                self.config.registration_output_dir / f"registration_results.json"
            )
            # Convert to serializable format
            serializable_results = convert_to_serializable(self.registration_results)

            with open(results_file_path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)

            if self.s3_enabled and hasattr(self, "registration_results"):
                self._upload_registration_results_to_s3(
                    self.registration_results, model_path
                )

            if results["success"]:
                self.logger.info("✅ Model registration completed successfully")
            else:
                self.logger.warning("⚠️ Model registration completed with errors")

            return results

        except Exception as e:
            raise ModelRegistrationError(f"Model registration failed: {str(e)}") from e

    def _register_to_huggingface(
        self, model_path: str, tokenizer_path: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register model to Hugging Face Hub."""
        if not HF_AVAILABLE:
            raise ModelRegistrationError("Hugging Face Hub libraries not available")

        if not self.config.hf_repo_id:
            raise ModelRegistrationError(
                "hf_repo_id required for Hugging Face registration"
            )

        if self.config.dry_run:
            return {"status": "dry_run", "repo_id": self.config.hf_repo_id}

        try:
            # Login if token provided
            if self.config.hf_token:
                login(token=self.config.hf_token)

            # Load model and tokenizer
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # Push to hub
            if self.config.push_to_hub:
                model.push_to_hub(
                    self.config.hf_repo_id,
                    private=self.config.private_repo,
                    commit_message=f"Upload ParaDetect model - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                )

                tokenizer.push_to_hub(
                    self.config.hf_repo_id,
                    private=self.config.private_repo,
                    commit_message=f"Upload ParaDetect tokenizer - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                )

                # Create model card
                self._create_hf_model_card(self.config.hf_repo_id, metadata)

                return {
                    "status": "success",
                    "repo_id": self.config.hf_repo_id,
                    "url": f"https://huggingface.co/{self.config.hf_repo_id}",
                    "private": self.config.private_repo,
                }
            else:
                return {"status": "skipped", "reason": "push_to_hub is False"}

        except Exception as e:
            raise ModelRegistrationError(f"Hugging Face registration failed: {str(e)}")

    def _register_to_mlflow(
        self, model_path: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register model to MLflow."""
        if not MLFLOW_AVAILABLE:
            raise ModelRegistrationError("MLflow not available")

        if self.config.dry_run:
            return {
                "status": "dry_run",
                "tracking_uri": self.config.mlflow_tracking_uri,
            }

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

            # Set experiment
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            with mlflow.start_run():
                # Load model
                model = AutoModelForSequenceClassification.from_pretrained(model_path)

                # Log model
                mlflow.pytorch.log_model(
                    model, "model", registered_model_name=self.config.mlflow_model_name
                )

                # Log metrics
                if "metrics" in metadata:
                    for key, value in metadata["metrics"].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)

                # Log parameters
                if "training_config" in metadata:
                    for key, value in metadata["training_config"].items():
                        if isinstance(value, (str, int, float, bool)):
                            mlflow.log_param(key, value)

                # Get run info
                run = mlflow.active_run()
                run_id = run.info.run_id

                # Register model version
                if self.config.mlflow_stage:
                    client = mlflow.tracking.MlflowClient()
                    model_version = client.create_model_version(
                        name=self.config.mlflow_model_name,
                        source=f"runs:/{run_id}/model",
                        description=self.config.model_description or "ParaDetect model",
                    )

                    # Transition to specified stage
                    client.transition_model_version_stage(
                        name=self.config.mlflow_model_name,
                        version=model_version.version,
                        stage=self.config.mlflow_stage,
                    )

                    return {
                        "status": "success",
                        "run_id": run_id,
                        "model_name": self.config.mlflow_model_name,
                        "model_version": model_version.version,
                        "stage": self.config.mlflow_stage,
                        "tracking_uri": self.config.mlflow_tracking_uri,
                    }
                else:
                    return {
                        "status": "success",
                        "run_id": run_id,
                        "tracking_uri": self.config.mlflow_tracking_uri,
                    }

        except Exception as e:
            raise ModelRegistrationError(f"MLflow registration failed: {str(e)}")

    def _register_to_sagemaker(
        self, model_path: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register model to AWS SageMaker."""
        if not SAGEMAKER_AVAILABLE:
            raise ModelRegistrationError("SageMaker libraries not available")

        if self.config.dry_run:
            return {"status": "dry_run", "model_name": self.config.sagemaker_model_name}

        try:
            # This is a simplified example - real implementation would need proper SageMaker setup
            session = sagemaker.Session()

            # Package model (simplified)
            model_name = (
                self.config.sagemaker_model_name
                or f"paradetect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

            # In real implementation, you would:
            # 1. Package the model in SageMaker format
            # 2. Upload to S3
            # 3. Create SageMaker model
            # 4. Register in model registry

            # For now, just return success with placeholder
            return {
                "status": "success",
                "model_name": model_name,
                "note": "SageMaker registration requires additional setup",
            }

        except Exception as e:
            raise ModelRegistrationError(f"SageMaker registration failed: {str(e)}")

    def _register_to_local(
        self, model_path: str, tokenizer_path: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register model to local registry."""
        try:
            # Create version directory in the configured registry location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_dir = self.config.local_registry_dir / timestamp
            self.version_dir = version_dir
            version_dir.mkdir(parents=True, exist_ok=True)

            if self.config.dry_run:
                return {"status": "dry_run", "version_dir": str(version_dir)}

            # Copy model files to registry (clean production copy)
            self.logger.info(f"📦 Copying model to registry: {version_dir}")

            model_dest = version_dir / "model"
            shutil.copytree(model_path, model_dest, dirs_exist_ok=True)

            # Handle tokenizer path
            if tokenizer_path != model_path:
                tokenizer_dest = version_dir / "tokenizer"
                shutil.copytree(tokenizer_path, tokenizer_dest, dirs_exist_ok=True)
            else:
                # If tokenizer and model are in the same path, just reference the model directory
                tokenizer_dest = model_dest

            # Enhanced metadata with JSON serialization fix
            metadata_enhanced = {
                **metadata,
                "version": timestamp,
                "registration_timestamp": datetime.now().isoformat(),
                "model_path": str(model_dest),
                "tokenizer_path": str(tokenizer_dest),
                "registry_type": "local",
                "registry_location": str(version_dir),
                "production_ready": True,  # Mark as production-ready
                "tags": self.config.model_tags or {},
                "description": self.config.model_description,
                "license": self.config.license,
            }

            # Convert any non-serializable objects to serializable formats
            metadata_enhanced = convert_to_serializable(metadata_enhanced)

            # Save metadata
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata_enhanced, f, indent=2)

            # Generate model card in registry location
            if self.config.generate_model_card:
                try:
                    self._generate_model_card(str(model_dest), metadata)
                    self.logger.info(f"📄 Model card generated in registry")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to generate model card in registry: {str(e)}"
                    )

            # Update registry index
            self._update_local_registry_index(timestamp, metadata_enhanced)

            # Create a "latest" marker file instead of symlink
            self._update_latest_marker(timestamp, version_dir)

            # Upload the registered model to S3
            s3_upload_success = False
            if self.s3_enabled:
                s3_upload_success = self._upload_registered_model_to_s3(
                    version_dir, timestamp, metadata_enhanced
                )

            return {
                "status": "success",
                "version": timestamp,
                "version_dir": str(version_dir),
                "metadata_path": str(metadata_path),
                "model_path": str(model_dest),
                "tokenizer_path": str(tokenizer_dest),
                "registry_location": str(self.config.local_registry_dir),
                "s3_upload_success": s3_upload_success,
                "s3_enabled": self.s3_enabled,
            }

        except Exception as e:
            raise ModelRegistrationError(f"Local registration failed: {str(e)}") from e

    def _update_latest_marker(self, version: str, version_dir: Path):
        """Update 'latest' marker to point to the newest model version (without symlinks)."""
        try:
            latest_file = self.config.local_registry_dir / "LATEST"

            # Create a simple text file containing the latest version info
            latest_info = {
                "version": version,
                "version_dir": str(version_dir),
                "model_path": str(version_dir / "model"),
                "tokenizer_path": str(
                    version_dir / "model"
                ),  # Same as model for most cases
                "updated_at": datetime.now().isoformat(),
            }

            with open(latest_file, "w") as f:
                json.dump(latest_info, f, indent=2)

            self.logger.info(f"📝 Updated LATEST marker to version {version}")

        except Exception as e:
            self.logger.warning(f"Failed to update latest marker: {str(e)}")

    def get_latest_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the latest registered model."""
        try:
            latest_file = self.config.local_registry_dir / "LATEST"

            if not latest_file.exists():
                return None

            with open(latest_file, "r") as f:
                latest_info = json.load(f)

            # Verify the version directory still exists
            version_dir = Path(latest_info["version_dir"])
            if not version_dir.exists():
                self.logger.warning(
                    f"Latest version directory not found: {version_dir}"
                )
                return None

            return latest_info

        except Exception as e:
            self.logger.warning(f"Failed to get latest model info: {str(e)}")
            return None

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models in the local registry."""
        try:
            index_path = self.config.local_registry_dir / "registry_index.json"

            if not index_path.exists():
                return []

            with open(index_path, "r") as f:
                index = json.load(f)

            return index.get("models", [])

        except Exception as e:
            self.logger.warning(f"Failed to list available models: {str(e)}")
            return []

    def get_model_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Get model information by version."""
        try:
            models = self.list_available_models()

            for model in models:
                if model.get("version") == version:
                    # Verify the model directory exists
                    model_path = Path(model.get("model_path", ""))
                    if model_path.exists():
                        return model
                    else:
                        self.logger.warning(
                            f"Model version {version} directory not found: {model_path}"
                        )

            return None

        except Exception as e:
            self.logger.warning(f"Failed to get model by version {version}: {str(e)}")
            return None

    def _upload_registered_model_to_s3(
        self, version_dir: Path, version: str, metadata: Dict[str, Any]
    ) -> bool:
        """Upload the registered model directory to S3."""
        try:
            if not self.s3_enabled:
                self.logger.warning("S3 not enabled - skipping model upload")
                return False

            self.logger.info(f"📤 Uploading registered model to S3: version {version}")

            # S3 key for the registered model
            s3_prefix = f"artifacts/model_registry/{version}"

            # Prepare S3 metadata
            s3_metadata = {
                "component": "model_registration",
                "version": version,
                "run_id": self.run_id,
                "registry_type": "production",
                "production_ready": "true",
                "model_name": str(metadata.get("model_name", "")),
                "accuracy": str(metadata.get("metrics", {}).get("accuracy", "")),
                "f1_score": str(metadata.get("metrics", {}).get("f1", "")),
                "license": str(self.config.license or ""),
                "uploaded_at": datetime.now().isoformat(),
            }

            # Use S3Manager's enhanced directory upload
            upload_results = self.s3_manager.upload_directory_with_structure(
                local_dir=version_dir,
                s3_prefix=s3_prefix,
                folder_type="artifacts",
                exclude_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store", "*.tmp"],
                preserve_empty_dirs=True,
                metadata=s3_metadata,
            )

            if upload_results["success"]:
                self.logger.info(
                    f"✅ Successfully uploaded registered model to S3: {s3_prefix}"
                )
                self.logger.info(
                    f"   📊 Upload stats: {upload_results['files_uploaded']} files, "
                    f"{upload_results['directories_created']} directories"
                )

                return True
            else:
                self.logger.warning("❌ Failed to upload registered model to S3")
                if upload_results.get("failed_uploads", 0) > 0:
                    self.logger.warning(
                        f"   Failed uploads: {upload_results['failed_uploads']}"
                    )
                return False

        except Exception as e:
            self.logger.warning(f"Error uploading registered model to S3: {e}")
            return False

    def _upload_registration_results_to_s3(
        self, results: Dict[str, Any], model_path: str
    ) -> bool:
        """Upload complete registration results to S3."""
        try:
            # Upload the entire local registry directory structure
            registry_upload_success = self._upload_complete_registry_to_s3()

            # Upload individual registration results
            component_upload_success = (
                self._upload_component_registration_results_to_s3(results, model_path)
            )

            return registry_upload_success and component_upload_success

        except Exception as e:
            self.logger.warning(f"Error uploading registration results to S3: {e}")
            return False

    def _upload_complete_registry_to_s3(self) -> bool:
        """Upload the complete local registry to S3."""
        try:
            if not self.version_dir.exists():
                return True  # Nothing to upload

            s3_prefix = f"artifacts/model_registry/{self.version_dir}"

            self.logger.info(f"📂 Uploading complete model registry to S3: {s3_prefix}")

            # Prepare metadata for registry backup
            registry_metadata = {
                "component": "complete_model_registry",
                "run_id": self.run_id,
                "backup_type": "complete_registry",
                "backup_timestamp": datetime.now().isoformat(),
                "registry_location": str(self.version_dir),
            }

            # Upload entire registry directory
            upload_results = self.s3_manager.upload_directory_with_structure(
                local_dir=self.version_dir,
                s3_prefix=s3_prefix,
                folder_type="artifacts",
                exclude_patterns=[".git", "__pycache__", "*.pyc", ".DS_Store", "*.tmp"],
                preserve_empty_dirs=True,
                metadata=registry_metadata,
            )

            if upload_results["success"]:
                self.logger.info(
                    f"✅ Complete registry uploaded to S3: {upload_results['files_uploaded']} files"
                )
                return True
            else:
                self.logger.warning("❌ Failed to upload complete registry to S3")
                return False

        except Exception as e:
            self.logger.warning(f"Error uploading complete registry to S3: {e}")
            return False

    def _upload_component_registration_results_to_s3(
        self, results: Dict[str, Any], model_path: str
    ) -> bool:
        """Upload component-specific registration results to S3."""
        try:
            regs = results.get("registrations", {}) if isinstance(results, dict) else {}
            errors = results.get("errors", {}) if isinstance(results, dict) else {}

            metadata = {
                "registration_success": str(results.get("success", "")),
                "registrations_count": str(len(regs)),
                "errors_count": str(len(errors)),
                "model_path": str(model_path),
                "registered_at": datetime.now().isoformat(),
            }

            upload_results = self.s3_manager.upload_component_results(
                results_dir=self.config.registration_output_dir,
                component_name="registrations",
                run_id=self.run_id,
                metadata=metadata,
            )

            if upload_results.get("success"):
                self.logger.info(
                    f"Uploaded registration results to S3: {upload_results.get('files_uploaded', 0)} files"
                )
                return True
            else:
                self.logger.warning("Failed to upload registration results to S3")
                return False

        except Exception as e:
            self.logger.warning(f"Error uploading registration results to S3: {e}")
            return False

    def _update_local_registry_index(self, version: str, metadata: Dict[str, Any]):
        """Update local registry index."""
        index_path = self.config.local_registry_dir / "registry_index.json"

        # Load existing index
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {"models": [], "latest": None}

        # Add new entry with JSON serializable metadata
        serializable_metadata = convert_to_serializable(metadata)

        index["models"].append(
            {
                "version": version,
                "timestamp": serializable_metadata.get("registration_timestamp"),
                "metrics": serializable_metadata.get("metrics", {}),
                "model_path": serializable_metadata.get("model_path"),
                "description": self.config.model_description,
            }
        )

        # Update latest
        index["latest"] = version

        # Save index
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _create_hf_model_card(self, repo_id: str, metadata: Dict[str, Any]):
        """Create model card for Hugging Face Hub."""
        if not HF_AVAILABLE:
            return

        try:
            api = HfApi()

            model_card_content = self._generate_model_card_content(metadata)

            # Upload model card
            api.upload_file(
                path_or_fileobj=model_card_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add model card",
            )

        except Exception as e:
            self.logger.warning(f"Failed to create HF model card: {str(e)}")

    def _generate_model_card(self, model_path: str, metadata: Dict[str, Any]) -> Path:
        """Generate comprehensive model card."""
        model_card_content = self._generate_model_card_content(metadata)

        # Save to model directory as README.md (standard name)
        model_card_path = Path(model_path) / "README.md"

        # Also save as MODEL_CARD.md for backup
        model_card_backup_path = Path(model_path) / "MODEL_CARD.md"

        try:
            with open(model_card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)

            # Create backup copy
            with open(model_card_backup_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)

            self.logger.info(f"📄 Model card saved: {model_card_path}")
            self.logger.info(f"📄 Model card backup: {model_card_backup_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save model card: {str(e)}")
            raise

        return model_card_path

    def _generate_model_card_content(self, metadata: Dict[str, Any]) -> str:
        """Generate model card content using the comprehensive template."""
        metrics = metadata.get("metrics", {})
        training_config = metadata.get("training_config", {})
        validation_results = metadata.get("validation_results", {})
        pipeline_metadata = metadata.get("pipeline_metadata", {})

        # Extract key metrics with fallbacks - using correct metric keys from evaluation
        accuracy = metrics.get("accuracy", "N/A")
        f1_score = metrics.get("f1", metrics.get("f1_weighted", "N/A"))
        precision = metrics.get("precision", metrics.get("precision_weighted", "N/A"))
        recall = metrics.get("recall", metrics.get("recall_weighted", "N/A"))

        # Format percentages
        def format_metric(value):
            if isinstance(value, (int, float)):
                return f"{value:.2%}"
            return str(value)

        # Extract training configuration with defaults
        base_model = training_config.get(
            "model_name_or_path", "microsoft/deberta-v3-large"
        )
        num_epochs = training_config.get(
            "num_train_epochs", training_config.get("num_epochs", "3")
        )
        batch_size = training_config.get(
            "per_device_train_batch_size", training_config.get("batch_size", "32")
        )
        learning_rate = training_config.get("learning_rate", "2e-4")
        max_length = training_config.get("max_length", "512")
        use_peft = training_config.get("use_peft", False)

        # Extract LoRA configuration
        peft_config = metadata.get("peft_config", {})
        lora_r = peft_config.get("r", "64") if isinstance(peft_config, dict) else "64"
        lora_alpha = (
            peft_config.get("lora_alpha", "128")
            if isinstance(peft_config, dict)
            else "128"
        )
        lora_dropout = (
            peft_config.get("lora_dropout", "0.1")
            if isinstance(peft_config, dict)
            else "0.1"
        )

        # Get class-wise metrics if available
        class_metrics = metrics.get("per_class_metrics", {})
        # Handle both possible key formats for class-wise metrics
        human_metrics = class_metrics.get("human", class_metrics.get("0", {}))
        ai_metrics = class_metrics.get("ai", class_metrics.get("1", {}))

        human_precision = human_metrics.get("precision", precision)
        human_recall = human_metrics.get("recall", recall)
        human_f1 = human_metrics.get("f1-score", human_metrics.get("f1", f1_score))
        ai_precision = ai_metrics.get("precision", precision)
        ai_recall = ai_metrics.get("recall", recall)
        ai_f1 = ai_metrics.get("f1-score", ai_metrics.get("f1", f1_score))

        # Get author info from pipeline metadata
        author = pipeline_metadata.get("author", "Model Developer")
        organization = pipeline_metadata.get("organization", "Independent Research")
        repository_url = pipeline_metadata.get(
            "repository_url", "https://github.com/your-repo/para-detect"
        )
        training_date = pipeline_metadata.get(
            "training_date", datetime.now().strftime("%Y-%m-%d")
        )

        # Generate LoRA configuration section
        lora_config_section = ""
        if use_peft:
            lora_config_section = f"""### LoRA Configuration
- **Rank (r)**: {lora_r}
- **Alpha**: {lora_alpha}
- **Dropout**: {lora_dropout}
- **Target Modules**: query_proj, key_proj, value_proj, dense, output.dense
- **Bias**: all
"""

        # Generate usage section based on PEFT usage
        if use_peft:
            usage_section = f"""```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("your-repo/paradetect-deberta-v3-lora")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "{base_model}",
    num_labels=2
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "your-repo/paradetect-deberta-v3-lora")

# Prediction function
def predict_text_origin(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length={max_length},
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)

    human_prob = probabilities[0][0].item()
    ai_prob = probabilities[0][1].item()

    return {{
        "prediction": "AI" if prediction.item() == 1 else "Human",
        "confidence": max(human_prob, ai_prob),
        "human_probability": human_prob,
        "ai_probability": ai_prob
    }}

# Example usage
text = "Your text here..."
result = predict_text_origin(text)
print(f"Prediction: {{result['prediction']}} (Confidence: {{result['confidence']:.1%}})")
```"""
        else:
            usage_section = f"""```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-repo/paradetect-deberta-v3-lora")
model = AutoModelForSequenceClassification.from_pretrained("your-repo/paradetect-deberta-v3-lora")

# Prediction function
def predict_text_origin(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length={max_length},
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1)

    human_prob = probabilities[0][0].item()
    ai_prob = probabilities[0][1].item()

    return {{
        "prediction": "AI" if prediction.item() == 1 else "Human",
        "confidence": max(human_prob, ai_prob),
        "human_probability": human_prob,
        "ai_probability": ai_prob
    }}

# Example usage
text = "Your text here..."
result = predict_text_origin(text)
print(f"Prediction: {{result['prediction']}} (Confidence: {{result['confidence']:.1%}})")
```"""

        # Generate the model card content
        # NOTE: The initial '---' and final '---' have no indentation.
        # The text block is dedented to remove the leading spaces from the final output.
        model_card_content = f"""\
---
language: en
license: {self.config.license}
library_name: transformers
pipeline_tag: text-classification
tags:
- ai-detection
- text-classification
- deberta-v3
- lora
- peft
datasets:
- artem9k/ai-text-detection-pile
metrics:
- accuracy
- f1
- precision
- recall
base_model: {base_model}
model_name: paradetect-deberta-v3-lora
---

# ParaDetect: DeBERTa Fine-tuned for AI vs Human Text Detection

## Model Description

{self.config.model_description or "ParaDetect is a fine-tuned DeBERTa model using LoRA (Low-Rank Adaptation) for detecting AI-generated vs human-written text. This model achieves high accuracy in distinguishing between human and AI-generated content, making it effective for academic integrity, content verification, and research applications."}

## Model Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: {"LoRA (Low-Rank Adaptation)" if use_peft else "Full Fine-tuning"}
- **Trainable Parameters**: {"~28M parameters (6% of total)" if use_peft else "All model parameters"}
- **Task**: Binary text classification (Human: 0, AI: 1)
- **Dataset**: AI Text Detection Pile (cleaned)
- **Training Framework**: Hugging Face Transformers{" + PEFT" if use_peft else ""}

## Performance Metrics

### Test Set Results
- **Accuracy**: {format_metric(accuracy)}
- **Precision (Weighted)**: {format_metric(precision)}
- **Recall (Weighted)**: {format_metric(recall)}
- **F1-Score (Weighted)**: {format_metric(f1_score)}

### Class-wise Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|---------|----------|
| **Human (0)** | {format_metric(human_precision)} | {format_metric(human_recall)} | {format_metric(human_f1)} |
| **AI (1)** | {format_metric(ai_precision)} | {format_metric(ai_recall)} | {format_metric(ai_f1)} |

## Training Details

{lora_config_section}### Training Parameters
- **Epochs**: {num_epochs}
- **Batch Size**: {batch_size} (train/eval)
- **Learning Rate**: {learning_rate}
- **Optimizer**: AdamW
- **Weight Decay**: {training_config.get('weight_decay', '0.01')}
- **Warmup Ratio**: {training_config.get('warmup_ratio', '0.1')}
- **Max Gradient Norm**: {training_config.get('max_grad_norm', '1.0')}
- **Max Length**: {max_length} tokens

### Early Stopping
- **Patience**: {training_config.get('early_stopping_patience', '5')} evaluation steps
- **Metric**: {training_config.get('metric_for_best_model', 'F1-score')}
- **Threshold**: {training_config.get('early_stopping_threshold', '0.001')}

## Usage

### Quick Start

{usage_section}

### Gradio Interface

```python
import gradio as gr

# Create interface
def analyze_text(text):
    result = predict_text_origin(text)
    return (
        f"{{result['prediction']}} ({{result['confidence']:.1%}} confidence)",
        {{
            "Human": result['human_probability'],
            "AI": result['ai_probability']
        }}
    )

demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=10, placeholder="Enter text to analyze..."),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="ParaDetect - AI vs Human Text Detection",
    description="Detect whether text is written by humans or generated by AI"
)

demo.launch()
```

## Technical Specifications

- **Input**: Text (up to {max_length} tokens)
- **Output**: Binary classification with confidence scores
- **Inference Speed**: ~100ms per text
- **Memory Usage**: {"Optimized with LoRA (reduced by ~94%)" if use_peft else "Standard model memory usage"}
- **GPU Support**: CUDA-enabled for faster inference

## Training Dataset

- **Source**: artem9k/ai-text-detection-pile
- **Size**: {training_config.get('dataset_size', 'Variable')} samples
- **Split**: {training_config.get('train_split', '70')}% train, {training_config.get('val_split', '15')}% validation, {training_config.get('test_split', '15')}% test
- **Balance**: Equal distribution of human vs AI text
- **Text Length**: 10-{max_length} tokens

## Limitations and Considerations

- **Language**: Optimized for English text
- **Text Length**: Best performance on moderate-length texts
- **Domain**: Performance may vary on very recent AI models
- **Context**: Performance may vary on highly technical or domain-specific content
- **Updates**: Regular retraining recommended as AI models evolve

## Intended Use Cases

### Primary Applications
- Academic integrity verification
- Content authenticity checking
- Research and analysis
- Educational demonstrations
- Journalism and fact-checking

### Not Recommended For
- Legal evidence without human verification
- Automated content moderation decisions
- High-stakes authentication without additional validation

## Ethical Considerations

- **Bias**: Model trained on specific dataset; may not represent all text types
- **Fairness**: Regular evaluation across different demographics recommended
- **Transparency**: Predictions are probabilistic, not definitive
- **Human Oversight**: Critical decisions should involve human judgment

## Model Card Authors

- **Developer**: {author}
- **Organization**: {organization}
- **Contact**: {repository_url}

## Citation

```bibtex
@misc{{paradetect{datetime.now().year},
  title={{ParaDetect: AI vs Human Text Detection with {base_model.split('/')[-1]}}},
  author={{{author}}},
  year={{{datetime.now().year}}},
  url={{{repository_url}}},
  note={{Fine-tuned {"using LoRA for efficient parameter adaptation" if use_peft else "for AI text detection"}}}
}}
```

## Additional Resources

- **📁 GitHub Repository**: [ParaDetect]({repository_url})
- **📊 Dataset**: [AI Text Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile)
- **🎯 Demo**: Gradio Interface (see usage above)
- **📈 Training Details**: See training configuration section
- **🔍 Performance Analysis**: See metrics section

## Version History

- **v1.0**: Initial release with {base_model.split('/')[-1]}{" + LoRA" if use_peft else ""}
- **Training Date**: {training_date}
- **Model Size**: {"~28M trainable parameters" if use_peft else f"~{training_config.get('total_parameters', '435M')} parameters"}
- **Performance**: {format_metric(accuracy)} test accuracy

---

*Model registered on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return textwrap.dedent(model_card_content)
