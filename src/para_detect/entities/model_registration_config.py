from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class ModelRegistrationConfig:
    """Configuration for model registration and publishing"""

    # Hugging Face Hub configuration
    use_hf: bool = True
    hf_repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    push_to_hub: bool = True
    private_repo: bool = False

    # MLflow configuration
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "paradetect"
    mlflow_model_name: str = "paradetect-deberta"
    mlflow_stage: str = "Staging"  # None, Staging, Production, Archived

    # AWS SageMaker configuration
    use_sagemaker: bool = False
    sagemaker_role: Optional[str] = None
    sagemaker_bucket: Optional[str] = None
    sagemaker_model_name: Optional[str] = None

    # Local registry configuration
    local_registry_dir: Path = Path("artifacts/model_registry")

    # Model metadata
    model_description: Optional[str] = None
    model_tags: Optional[Dict[str, str]] = None
    license: str = "apache-2.0"

    # Registration validation
    require_validation_pass: bool = True
    dry_run: bool = False
    force_overwrite: bool = False

    # Model card generation
    generate_model_card: bool = True
    model_card_template: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not any([self.use_hf, self.use_mlflow, self.use_sagemaker]):
            # Default to local registry if no remote registry is configured
            pass

        if self.use_hf and self.push_to_hub and not self.hf_repo_id:
            raise ValueError("hf_repo_id is required when push_to_hub is True")

        if self.use_mlflow and not self.mlflow_tracking_uri:
            # Use default local MLflow if no URI provided
            object.__setattr__(self, "mlflow_tracking_uri", "file:./mlruns")

        if self.mlflow_stage not in [None, "Staging", "Production", "Archived"]:
            raise ValueError(
                "mlflow_stage must be None, 'Staging', 'Production', or 'Archived'"
            )

        # Ensure local registry directory exists
        self.local_registry_dir.mkdir(parents=True, exist_ok=True)
