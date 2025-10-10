"""Base pipeline class with common functionality"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import json
import fcntl
import os
import time
from datetime import datetime, timedelta

from para_detect.core.config_manager import ConfigurationManager
from para_detect.entities.pipeline_config import PipelineConfig, PipelineType
from para_detect.core.exceptions import ParaDetectException
from para_detect import get_logger


class BasePipeline(ABC):
    """
    Base class for all ParaDetect pipelines with standardized state management.

    Provides:
    - Pipeline-specific state persistence
    - Process locking to prevent concurrent execution
    - Checkpoint management
    - State cleanup and retention
    """

    def __init__(
        self,
        pipeline_type: PipelineType,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize base pipeline.

        Args:
            pipeline_type: Type of pipeline (DATA, TRAINING, etc.)
            config_manager: Optional configuration manager
        """
        self.pipeline_type = pipeline_type
        self.config_manager = config_manager or ConfigurationManager()
        self.logger = get_logger(f"{pipeline_type.value}_pipeline")

        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get_pipeline_config()

        # Pipeline-specific paths
        self.state_file = self.pipeline_config.get_state_file(pipeline_type)
        self.checkpoint_dir = self.pipeline_config.get_checkpoint_dir(pipeline_type)
        self.lock_file = self.pipeline_config.get_lock_file(pipeline_type)

        # Pipeline state
        self.pipeline_state = self._get_initial_state()
        self.lock_handle = None

        # Load previous state if persistence is enabled
        if self.pipeline_config.enable_state_persistence:
            self._load_pipeline_state()

    @abstractmethod
    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial pipeline state. Override in subclasses."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute pipeline. Override in subclasses."""
        pass

    def _acquire_lock(self) -> bool:
        """Acquire pipeline lock to prevent concurrent execution."""
        if not self.pipeline_config.enable_pipeline_locks:
            return True

        try:
            self.lock_handle = open(self.lock_file, "w")
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write process info to lock file
            lock_info = {
                "pid": os.getpid(),
                "pipeline_type": self.pipeline_type.value,
                "acquired_at": datetime.now().isoformat(),
                "hostname": os.uname().nodename,
            }
            json.dump(lock_info, self.lock_handle, indent=2)
            self.lock_handle.flush()

            self.logger.info(f"🔒 Acquired lock for {self.pipeline_type.value}")
            return True

        except (IOError, OSError) as e:
            if self.lock_handle:
                self.lock_handle.close()
                self.lock_handle = None

            self.logger.error(
                f"❌ Failed to acquire lock for {self.pipeline_type.value}: {str(e)}"
            )
            return False

    def _release_lock(self) -> None:
        """Release pipeline lock."""
        if self.lock_handle:
            try:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
                self.lock_handle.close()

                # Remove lock file
                if self.lock_file.exists():
                    self.lock_file.unlink()

                self.logger.info(f"🔓 Released lock for {self.pipeline_type.value}")

            except Exception as e:
                self.logger.warning(f"Error releasing lock: {str(e)}")
            finally:
                self.lock_handle = None

    def _load_pipeline_state(self) -> None:
        """Load pipeline state from persistent storage."""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    saved_state = json.load(f)

                # Merge saved state with current state
                self.pipeline_state.update(saved_state.get("pipeline_state", {}))

                self.logger.info(
                    f"📂 Loaded {self.pipeline_type.value} state from: {self.state_file}"
                )

                # Clean old states if retention is configured
                self._cleanup_old_states()

        except Exception as e:
            self.logger.warning(
                f"Could not load {self.pipeline_type.value} state: {str(e)}"
            )

    def _save_pipeline_state(self) -> None:
        """Save pipeline state to persistent storage."""
        if (
            not self.pipeline_config.enable_state_persistence
            or not self.pipeline_config.state_auto_save
        ):
            return

        try:
            state_data = {
                "pipeline_type": self.pipeline_type.value,
                "pipeline_state": self.pipeline_state,
                "saved_at": datetime.now().isoformat(),
                "version": "1.0.0",
            }

            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            self.logger.debug(
                f"💾 {self.pipeline_type.value} state saved to: {self.state_file}"
            )

        except Exception as e:
            self.logger.warning(
                f"Could not save {self.pipeline_type.value} state: {str(e)}"
            )

    def _cleanup_old_states(self) -> None:
        """Clean up old state files based on retention policy."""
        try:
            if self.pipeline_config.state_retention_days <= 0:
                return

            cutoff_date = datetime.now() - timedelta(
                days=self.pipeline_config.state_retention_days
            )

            # Check state file age
            if self.state_file.exists():
                file_time = datetime.fromtimestamp(self.state_file.stat().st_mtime)
                if file_time < cutoff_date:
                    # Archive instead of delete
                    archive_name = f"{self.state_file.stem}_archived_{file_time.strftime('%Y%m%d')}.json"
                    archive_path = self.state_file.parent / "archived" / archive_name
                    archive_path.parent.mkdir(exist_ok=True)

                    self.state_file.rename(archive_path)
                    self.logger.info(f"📦 Archived old state file to: {archive_path}")

        except Exception as e:
            self.logger.warning(f"Error during state cleanup: {str(e)}")

    def save_checkpoint(
        self, checkpoint_name: str, additional_data: Optional[Dict] = None
    ) -> bool:
        """Save a named checkpoint."""
        try:
            checkpoint_data = {
                "pipeline_type": self.pipeline_type.value,
                "checkpoint_name": checkpoint_name,
                "timestamp": datetime.now().isoformat(),
                "pipeline_state": self.pipeline_state.copy(),
                "additional_data": additional_data or {},
            }

            checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"

            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            self.logger.info(
                f"💾 Checkpoint '{checkpoint_name}' saved for {self.pipeline_type.value}"
            )
            return True

        except Exception as e:
            self.logger.warning(
                f"Could not save checkpoint '{checkpoint_name}': {str(e)}"
            )
            return False

    def __enter__(self):
        """Context manager entry - acquire lock"""
        if not self._acquire_lock():
            raise ParaDetectException(
                f"Could not acquire lock for {self.pipeline_type.value} pipeline"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release lock"""
        self._release_lock()
