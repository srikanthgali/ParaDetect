"""
Standalone Data Pipeline for ParaDetect
Can be used independently or as part of other pipelines
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import os
import json
from datetime import datetime

from para_detect.core.base_pipeline import BasePipeline
from para_detect.core.config_manager import ConfigurationManager
from para_detect.components.data_ingestion import DataIngestion
from para_detect.components.data_preprocessing import DataPreprocessing
from para_detect.components.data_validation import DataValidation
from para_detect.core.exceptions import ParaDetectException
from para_detect import get_logger
from para_detect.entities.pipeline_config import PipelineType


class DataPipeline(BasePipeline):
    """
    Standalone data processing pipeline for ParaDetect with smart resumption capabilities.

    Features:
    - Automatic step skipping if outputs exist and are valid
    - Pipeline state persistence
    - Smart resumption from last successful step
    - Configurable force rerun options
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize data pipeline with pipeline-specific state management.

        Args:
            config_manager: Optional configuration manager. If None, creates new instance.
        """
        self.config_manager = config_manager or ConfigurationManager()
        super().__init__(PipelineType.DATA, config_manager)
        self.logger = get_logger(self.__class__.__name__)

        # Pipeline state tracking
        self.pipeline_state = {
            "ingestion_completed": False,
            "preprocessing_completed": False,
            "validation_completed": False,
            "validation_passed": False,
            "last_run_timestamp": None,
            "pipeline_version": "1.0.0",
        }

        # Paths tracking
        self.data_paths = {
            "raw_data": None,
            "processed_data": None,
            "validation_report": None,
        }

        # State persistence
        # Load data paths from state if available
        if "data_paths" in self.pipeline_state:
            self.data_paths.update(self.pipeline_state["data_paths"])

        # Load previous state if exists
        self._load_pipeline_state()

    def _load_pipeline_state(self) -> None:
        """Load pipeline state from persistent storage."""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    saved_state = json.load(f)

                # Merge saved state with current state
                self.pipeline_state.update(saved_state.get("pipeline_state", {}))
                self.data_paths.update(saved_state.get("data_paths", {}))

                self.logger.info("📂 Loaded previous pipeline state")
                self.logger.info(
                    f"Last run: {self.pipeline_state.get('last_run_timestamp', 'Never')}"
                )

        except Exception as e:
            self.logger.warning(f"Could not load pipeline state: {str(e)}")

    def _save_pipeline_state(self) -> None:
        """Save pipeline state to persistent storage."""
        try:
            state_data = {
                "pipeline_state": self.pipeline_state,
                "data_paths": self.data_paths,
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            self.logger.debug("💾 Pipeline state saved")

        except Exception as e:
            self.logger.warning(f"Could not save pipeline state: {str(e)}")

    def _check_step_output_exists(self, step_name: str) -> bool:
        """Check if step output exists and is valid."""
        try:
            if step_name == "ingestion":
                path = self.data_paths.get("raw_data")
                if path and Path(path).exists():
                    # Additional validation: check if file is not empty
                    return Path(path).stat().st_size > 0

            elif step_name == "preprocessing":
                path = self.data_paths.get("processed_data")
                if path and Path(path).exists():
                    return Path(path).stat().st_size > 0

            elif step_name == "validation":
                path = self.data_paths.get("validation_report")
                if path and Path(path).exists():
                    return Path(path).stat().st_size > 0

            return False

        except Exception as e:
            self.logger.warning(f"Error checking {step_name} output: {str(e)}")
            return False

    def _should_skip_step(self, step_name: str, force_rerun: bool = False) -> bool:
        """Determine if a step should be skipped."""
        if force_rerun:
            return False

        # Check if step was completed and output exists
        step_completed = self.pipeline_state.get(f"{step_name}_completed", False)
        output_exists = self._check_step_output_exists(step_name)

        should_skip = step_completed and output_exists

        if should_skip:
            self.logger.info(
                f"⏭️  Skipping {step_name} - already completed with valid output"
            )

        return should_skip

    def run_full_pipeline(
        self, force_rerun: bool = False, force_steps: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Run the complete data pipeline with smart resumption.

        Args:
            force_rerun: If True, reruns all steps even if already completed
            force_steps: List of specific steps to force rerun (e.g., ['preprocessing', 'validation'])

        Returns:
            Dict containing pipeline results and paths

        Raises:
            ParaDetectException: If any pipeline step fails
        """
        try:
            self.logger.info("🚀 Starting data pipeline with smart resumption...")

            # Normalize force_steps
            force_steps = force_steps or []

            # Step 1: Data Ingestion
            force_ingestion = force_rerun or "ingestion" in force_steps
            if not self._should_skip_step("ingestion", force_ingestion):
                self.data_paths["raw_data"] = self.run_data_ingestion()
                self.pipeline_state["ingestion_completed"] = True
                self._save_pipeline_state()
            else:
                self.logger.info(
                    f"📁 Using existing raw data: {self.data_paths['raw_data']}"
                )

            # Step 2: Data Preprocessing
            force_preprocessing = force_rerun or "preprocessing" in force_steps
            if not self._should_skip_step("preprocessing", force_preprocessing):
                self.data_paths["processed_data"] = self.run_data_preprocessing()
                self.pipeline_state["preprocessing_completed"] = True
                self._save_pipeline_state()
            else:
                self.logger.info(
                    f"📁 Using existing processed data: {self.data_paths['processed_data']}"
                )

            # Step 3: Data Validation
            force_validation = force_rerun or "validation" in force_steps
            if not self._should_skip_step("validation", force_validation):
                validation_passed, report_path = self.run_data_validation()
                self.data_paths["validation_report"] = report_path
                self.pipeline_state["validation_completed"] = True
                self.pipeline_state["validation_passed"] = validation_passed
                self._save_pipeline_state()
            else:
                self.logger.info(
                    f"📁 Using existing validation report: {self.data_paths['validation_report']}"
                )
                # Still need to check if validation passed
                if not self.pipeline_state.get("validation_passed", False):
                    self.logger.warning(
                        "⚠️ Previous validation failed - may need to rerun"
                    )

            # Final validation check
            if not self.pipeline_state["validation_passed"]:
                raise ParaDetectException(
                    f"Data validation failed. Check report: {self.data_paths['validation_report']}"
                )

            # Update final state
            self.pipeline_state["last_run_timestamp"] = datetime.now().isoformat()
            self._save_pipeline_state()

            result = {
                "success": True,
                "processed_data_path": self.data_paths["processed_data"],
                "validation_report_path": self.data_paths["validation_report"],
                "pipeline_state": self.pipeline_state.copy(),
                "data_paths": self.data_paths.copy(),
                "steps_executed": self._get_executed_steps_summary(),
            }

            self.logger.info("✅ Data pipeline completed successfully!")
            self.logger.info(f"📊 Final dataset: {self.data_paths['processed_data']}")
            self.logger.info(
                f"📋 Validation report: {self.data_paths['validation_report']}"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Data pipeline failed: {str(e)}")
            raise ParaDetectException(f"Data pipeline failed: {str(e)}") from e

    @staticmethod
    def run_data_pipeline_safely():
        """Run data pipeline with automatic locking and checkpointing."""
        try:
            # Context manager automatically handles locking
            with DataPipeline() as pipeline:
                result = pipeline.run_full_pipeline()

                # Save checkpoint after successful completion
                pipeline.save_checkpoint("post_completion", {"success": True})

                return result

        except ParaDetectException as e:
            print(f"Pipeline execution failed: {e}")
            return None

    def _get_executed_steps_summary(self) -> Dict[str, str]:
        """Get summary of which steps were executed vs skipped."""
        return {
            "ingestion": (
                "completed" if self.pipeline_state["ingestion_completed"] else "failed"
            ),
            "preprocessing": (
                "completed"
                if self.pipeline_state["preprocessing_completed"]
                else "failed"
            ),
            "validation": (
                "completed" if self.pipeline_state["validation_completed"] else "failed"
            ),
        }

    def run_data_ingestion(self) -> str:
        """Run data ingestion step."""
        self.logger.info("📥 Step 1: Data Ingestion")

        ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(ingestion_config)
        raw_data_path = data_ingestion.run()

        self.logger.info(f"✅ Data ingestion completed: {raw_data_path}")
        return raw_data_path

    def run_data_preprocessing(self) -> str:
        """Run data preprocessing step."""
        if not self.data_paths["raw_data"]:
            raise ParaDetectException(
                "Data ingestion must be completed before preprocessing"
            )

        self.logger.info("🔄 Step 2: Data Preprocessing")

        preprocessing_config = self.config_manager.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(preprocessing_config)
        processed_data_path = data_preprocessing.run(self.data_paths["raw_data"])

        self.logger.info(f"✅ Data preprocessing completed: {processed_data_path}")
        return processed_data_path

    def run_data_validation(self) -> Tuple[bool, str]:
        """Run data validation step."""
        if not self.data_paths["processed_data"]:
            raise ParaDetectException(
                "Data preprocessing must be completed before validation"
            )

        self.logger.info("🔍 Step 3: Data Validation")

        validation_config = self.config_manager.get_data_validation_config()
        data_validation = DataValidation(validation_config)
        validation_passed, validation_report = data_validation.run(
            self.data_paths["processed_data"]
        )

        status = "✅ PASSED" if validation_passed else "❌ FAILED"
        self.logger.info(f"{status} Data validation completed: {validation_report}")

        return validation_passed, validation_report

    def reset_pipeline_state(self, steps: Optional[list] = None) -> None:
        """
        Reset pipeline state for specific steps or all steps.

        Args:
            steps: List of steps to reset ('ingestion', 'preprocessing', 'validation')
                  If None, resets all steps
        """
        steps = steps or ["ingestion", "preprocessing", "validation"]

        for step in steps:
            if step == "ingestion":
                self.pipeline_state["ingestion_completed"] = False
                self.data_paths["raw_data"] = None
            elif step == "preprocessing":
                self.pipeline_state["preprocessing_completed"] = False
                self.data_paths["processed_data"] = None
            elif step == "validation":
                self.pipeline_state["validation_completed"] = False
                self.pipeline_state["validation_passed"] = False
                self.data_paths["validation_report"] = None

        self._save_pipeline_state()
        self.logger.info(f"🔄 Reset pipeline state for steps: {steps}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status with detailed information."""
        return {
            "pipeline_state": self.pipeline_state.copy(),
            "data_paths": self.data_paths.copy(),
            "files_exist": {
                "raw_data": self._check_step_output_exists("ingestion"),
                "processed_data": self._check_step_output_exists("preprocessing"),
                "validation_report": self._check_step_output_exists("validation"),
            },
            "is_complete": all(
                [
                    self.pipeline_state["ingestion_completed"],
                    self.pipeline_state["preprocessing_completed"],
                    self.pipeline_state["validation_completed"],
                    self.pipeline_state["validation_passed"],
                ]
            ),
            "can_resume": any(
                [
                    self.pipeline_state["ingestion_completed"],
                    self.pipeline_state["preprocessing_completed"],
                    self.pipeline_state["validation_completed"],
                ]
            ),
        }

    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for data pipeline."""
        return {
            "ingestion_completed": False,
            "preprocessing_completed": False,
            "validation_completed": False,
            "validation_passed": False,
            "last_run_timestamp": None,
            "pipeline_version": "1.0.0",
            "data_paths": {
                "raw_data": None,
                "processed_data": None,
                "validation_report": None,
            },
        }

    def _save_pipeline_state(self) -> None:
        """Override to include data paths in state."""
        # Update state with current data paths
        self.pipeline_state["data_paths"] = self.data_paths
        super()._save_pipeline_state()

    def run(
        self, force_rerun: bool = False, force_steps: Optional[list] = None
    ) -> Dict[str, Any]:
        """Run with context manager for automatic locking."""
        return self.run_full_pipeline(force_rerun, force_steps)


# Enhanced standalone execution with command-line options
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ParaDetect Data Pipeline")
    parser.add_argument(
        "--force-rerun", action="store_true", help="Force rerun all steps"
    )
    parser.add_argument(
        "--force-steps",
        nargs="*",
        choices=["ingestion", "preprocessing", "validation"],
        help="Force rerun specific steps",
    )
    parser.add_argument(
        "--reset",
        nargs="*",
        choices=["ingestion", "preprocessing", "validation"],
        help="Reset specific steps",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show pipeline status and exit"
    )

    args = parser.parse_args()

    pipeline = DataPipeline()

    if args.status:
        status = pipeline.get_pipeline_status()
        print(f"📊 Pipeline Status:")
        print(f"  Complete: {status['is_complete']}")
        print(f"  Can Resume: {status['can_resume']}")
        print(
            f"  Last Run: {status['pipeline_state'].get('last_run_timestamp', 'Never')}"
        )
        exit(0)

    if args.reset:
        pipeline.reset_pipeline_state(args.reset)
        print(f"🔄 Reset completed for: {args.reset}")

    try:
        result = pipeline.run_full_pipeline(
            force_rerun=args.force_rerun, force_steps=args.force_steps
        )
        print(
            f"✅ Pipeline completed! Dataset ready at: {result['processed_data_path']}"
        )
        print(f"📊 Steps executed: {result['steps_executed']}")

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        exit(1)
