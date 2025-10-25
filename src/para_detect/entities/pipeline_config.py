from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from enum import Enum


class PipelineType(Enum):
    """Supported pipeline types"""

    DATA = "data_pipeline"
    TRAINING = "training_pipeline"
    INFERENCE = "inference_pipeline"
    MONITORING = "monitoring_pipeline"
    DEPLOYMENT = "deployment_pipeline"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for pipeline execution and state management"""

    artifacts_dir: Path
    checkpoints_dir: Path
    state_files: Dict[str, Path]
    enable_state_persistence: bool = True
    state_auto_save: bool = True
    state_retention_days: int = 30
    enable_pipeline_locks: bool = True
    use_s3_cache: bool = False
    s3_backup_enabled: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure directories exist
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Ensure state file directories exist
        for state_file in self.state_files.values():
            state_file.parent.mkdir(parents=True, exist_ok=True)

    def get_state_file(self, pipeline_type: PipelineType) -> Path:
        """Get state file path for specific pipeline type"""
        return self.state_files.get(
            pipeline_type.value,
            self.artifacts_dir / f"{pipeline_type.value}_state.json",
        )

    def get_checkpoint_dir(self, pipeline_type: PipelineType) -> Path:
        """Get checkpoint directory for specific pipeline type"""
        checkpoint_dir = self.checkpoints_dir / pipeline_type.value
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def get_lock_file(self, pipeline_type: PipelineType) -> Path:
        """Get lock file path for specific pipeline type"""
        return self.artifacts_dir / f"{pipeline_type.value}.lock"
