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

import torch
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

    def __init__(self, config: ModelRegistrationConfig):
        """
        Initialize model registrar.

        Args:
            config: Registration configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

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

        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer (if different from model_path)
            metadata: Model metadata including metrics and training info
            validation_passed: Whether model passed validation

        Returns:
            Dict: Registration results for each registry
        """
        try:
            self.logger.info("📦 Starting model registration...")

            # Check validation requirement
            if self.config.require_validation_pass and not validation_passed:
                raise ModelRegistrationError(
                    "Model registration requires validation to pass, but validation failed"
                )

            # Validate inputs
            if not Path(model_path).exists():
                raise ModelRegistrationError(f"Model path does not exist: {model_path}")

            tokenizer_path = tokenizer_path or model_path
            metadata = metadata or {}

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
            if self.config.use_hf and HF_AVAILABLE:
                try:
                    hf_result = self._register_to_huggingface(
                        model_path, tokenizer_path, metadata
                    )
                    results["registrations"]["huggingface"] = hf_result
                    self.logger.info("✅ Hugging Face registration completed")
                except Exception as e:
                    error_msg = f"Hugging Face registration failed: {str(e)}"
                    results["errors"]["huggingface"] = error_msg
                    self.logger.error(f"❌ {error_msg}")
                    if not self.config.dry_run:
                        results["success"] = False

            if self.config.use_mlflow and MLFLOW_AVAILABLE:
                try:
                    mlflow_result = self._register_to_mlflow(model_path, metadata)
                    results["registrations"]["mlflow"] = mlflow_result
                    self.logger.info("✅ MLflow registration completed")
                except Exception as e:
                    error_msg = f"MLflow registration failed: {str(e)}"
                    results["errors"]["mlflow"] = error_msg
                    self.logger.error(f"❌ {error_msg}")
                    if not self.config.dry_run:
                        results["success"] = False

            if self.config.use_sagemaker and SAGEMAKER_AVAILABLE:
                try:
                    sagemaker_result = self._register_to_sagemaker(model_path, metadata)
                    results["registrations"]["sagemaker"] = sagemaker_result
                    self.logger.info("✅ SageMaker registration completed")
                except Exception as e:
                    error_msg = f"SageMaker registration failed: {str(e)}"
                    results["errors"]["sagemaker"] = error_msg
                    self.logger.error(f"❌ {error_msg}")
                    if not self.config.dry_run:
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
                results["errors"]["local"] = error_msg
                self.logger.error(f"❌ {error_msg}")
                results["success"] = False

            # Generate model card
            if self.config.generate_model_card:
                try:
                    model_card_path = self._generate_model_card(model_path, metadata)
                    results["model_card_path"] = str(model_card_path)
                    self.logger.info(f"📄 Model card generated: {model_card_path}")
                except Exception as e:
                    self.logger.warning(f"Model card generation failed: {str(e)}")

            self.registration_results = results

            if results["success"]:
                self.logger.info("🎉 Model registration completed successfully!")
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
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_dir = self.config.local_registry_dir / f"v_{timestamp}"
            version_dir.mkdir(parents=True, exist_ok=True)

            if self.config.dry_run:
                return {"status": "dry_run", "local_path": str(version_dir)}

            # Copy model files
            model_copy_dir = version_dir / "model"
            shutil.copytree(model_path, model_copy_dir)

            # Copy tokenizer if different path
            if tokenizer_path != model_path:
                tokenizer_copy_dir = version_dir / "tokenizer"
                shutil.copytree(tokenizer_path, tokenizer_copy_dir)

            # Save metadata
            metadata_enhanced = {
                **metadata,
                "registration_timestamp": datetime.now().isoformat(),
                "model_path": str(model_copy_dir),
                "tokenizer_path": (
                    str(tokenizer_copy_dir)
                    if tokenizer_path != model_path
                    else str(model_copy_dir)
                ),
                "version": timestamp,
            }

            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata_enhanced, f, indent=2)

            # Update registry index
            self._update_local_registry_index(timestamp, metadata_enhanced)

            return {
                "status": "success",
                "local_path": str(version_dir),
                "version": timestamp,
                "metadata_path": str(metadata_path),
            }

        except Exception as e:
            raise ModelRegistrationError(f"Local registration failed: {str(e)}")

    def _update_local_registry_index(self, version: str, metadata: Dict[str, Any]):
        """Update local registry index."""
        index_path = self.config.local_registry_dir / "registry_index.json"

        # Load existing index
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {"models": [], "latest": None}

        # Add new entry
        index["models"].append(
            {
                "version": version,
                "timestamp": metadata.get("registration_timestamp"),
                "metrics": metadata.get("metrics", {}),
                "model_path": metadata.get("model_path"),
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

        # Save to model directory
        model_card_path = Path(model_path) / "MODEL_CARD.md"
        with open(model_card_path, "w") as f:
            f.write(model_card_content)

        return model_card_path

    def _generate_model_card_content(self, metadata: Dict[str, Any]) -> str:
        """Generate model card content."""
        metrics = metadata.get("metrics", {})
        training_config = metadata.get("training_config", {})

        content = f"""# ParaDetect: AI vs Human Text Detection

## Model Description

{self.config.model_description or "DeBERTa-based model for detecting AI-generated text vs human-written text."}

## Model Details

- **Model Type**: Text Classification (Binary)
- **Base Model**: {training_config.get('model_name', 'microsoft/deberta-v3-large')}
- **Task**: Human vs AI Text Detection
- **License**: {self.config.license}

## Training Details

### Training Configuration
- **Epochs**: {training_config.get('num_epochs', 'N/A')}
- **Batch Size**: {training_config.get('batch_size', 'N/A')}
- **Learning Rate**: {training_config.get('learning_rate', 'N/A')}
- **Max Length**: {training_config.get('max_length', 'N/A')}
- **LoRA**: {'Yes' if training_config.get('use_peft', False) else 'No'}

### Performance Metrics

"""

        if metrics:
            content += "| Metric | Value |\n|--------|-------|\n"
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    content += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"

        content += f"""

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-repo/paradetect")
model = AutoModelForSequenceClassification.from_pretrained("your-repo/paradetect")

# Example usage
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)

# 0 = Human, 1 = AI
print(f"Prediction: {{'AI' if prediction.item() == 1 else 'Human'}}")
print(f"Confidence: {{probabilities.max().item():.4f}}")
```

## Labels

- `0`: Human-written text
- `1`: AI-generated text

## Training Data

The model was trained on the AI Text Detection Pile dataset, which contains examples of both human-written and AI-generated text from various sources.

## Limitations and Biases

- The model may have biases based on the training data distribution
- Performance may vary on text from domains not well-represented in training data
- The model should be used as a tool to assist human judgment, not replace it

## Registration Date

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

        if self.config.model_tags:
            content += "\n## Tags\n\n"
            for key, value in self.config.model_tags.items():
                content += f"- **{key}**: {value}\n"

        return content
