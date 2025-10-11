"""
Complete Training Pipeline for ParaDetect
Orchestrates: Data Pipeline + Model Training + Evaluation + Validation + Registration
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from para_detect.core.base_pipeline import BasePipeline
from para_detect.core.config_manager import ConfigurationManager
from para_detect.pipelines.data_pipeline import DataPipeline
from para_detect.components.model_training import ModelTrainer
from para_detect.components.model_evaluation import ModelEvaluator
from para_detect.components.model_validation import ModelValidator
from para_detect.components.model_registration import ModelRegistrar
from para_detect.core.exceptions import MLPipelineException
from para_detect.entities.pipeline_config import PipelineType
from para_detect import get_logger


class TrainingPipeline(BasePipeline):
    """
    Complete training pipeline orchestrator for ParaDetect.

    Orchestrates the complete ML training workflow:
    1. Data Pipeline (ingestion → preprocessing → validation)
    2. Model Training (DeBERTa fine-tuning with LoRA)
    3. Model Evaluation (comprehensive metrics and analysis)
    4. Model Validation (quality and fairness checks)
    5. Model Registration (HuggingFace Hub, MLflow, local registry)

    Features:
    - Checkpoint resumption for interrupted runs
    - Comprehensive state management
    - Artifact tracking and metadata preservation
    - Configurable validation gates
    - Multi-registry model publication
    """

    def __init__(
        self,
        config_manager: Optional[ConfigurationManager] = None,
        run_id: Optional[str] = None,
    ):
        """
        Initialize training pipeline.

        Args:
            config_manager: Configuration manager instance
            run_id: Optional run identifier for tracking
        """
        super().__init__(PipelineType.TRAINING, config_manager)

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = get_logger(self.__class__.__name__)

        # Initialize components
        self.data_pipeline = DataPipeline(self.config_manager)
        self.model_trainer = None
        self.model_evaluator = None
        self.model_validator = None
        self.model_registrar = None

        # Artifacts tracking
        self.training_artifacts = {
            "run_id": self.run_id,
            "start_time": None,
            "end_time": None,
            "processed_data_path": None,
            "model_path": None,
            "evaluation_results": None,
            "validation_results": None,
            "registration_results": None,
            "training_successful": False,
            "artifacts_dir": str(self.checkpoint_dir / self.run_id),
        }

        # Create run-specific artifacts directory
        self.run_artifacts_dir = Path(self.training_artifacts["artifacts_dir"])
        self.run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for training pipeline."""
        return {
            "run_id": self.run_id,
            "data_pipeline_completed": False,
            "model_training_completed": False,
            "model_evaluation_completed": False,
            "model_validation_completed": False,
            "model_registration_completed": False,
            "training_successful": False,
            "current_step": "initialization",
            "artifacts": self.training_artifacts.copy(),
            "last_checkpoint": None,
        }

    def run(
        self,
        use_existing_data: bool = False,
        data_path: Optional[str] = None,
        resume_from_step: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Args:
            use_existing_data: Skip data pipeline and use existing data
            data_path: Path to existing processed data
            resume_from_step: Step to resume from ('data', 'training', 'evaluation', 'validation', 'registration')

        Returns:
            Dict: Complete training results and artifacts
        """
        try:
            self.logger.info(f"🚀 Starting training pipeline - Run ID: {self.run_id}")
            self.training_artifacts["start_time"] = datetime.now().isoformat()

            # Determine starting step
            start_step = resume_from_step or self._determine_resume_step()
            self.logger.info(f"📍 Starting from step: {start_step}")

            # Step 1: Data Pipeline
            if start_step in [None, "data"] and not use_existing_data:
                self._run_data_pipeline()
            elif use_existing_data and data_path:
                self._use_existing_data(data_path)
            elif self.pipeline_state.get("data_pipeline_completed"):
                self.logger.info("✅ Data pipeline already completed - skipping")
                self.training_artifacts["processed_data_path"] = self.pipeline_state[
                    "artifacts"
                ].get("processed_data_path")
            else:
                raise MLPipelineException(
                    "No data available. Run data pipeline or provide existing data path."
                )

            # Step 2: Model Training
            if start_step in [None, "data", "training"]:
                self._run_model_training()
            elif self.pipeline_state.get("model_training_completed"):
                self.logger.info("✅ Model training already completed - skipping")
                self.training_artifacts["model_path"] = self.pipeline_state[
                    "artifacts"
                ].get("model_path")

            # Step 3: Model Evaluation
            if start_step in [None, "data", "training", "evaluation"]:
                self._run_model_evaluation()
            elif self.pipeline_state.get("model_evaluation_completed"):
                self.logger.info("✅ Model evaluation already completed - skipping")
                self.training_artifacts["evaluation_results"] = self.pipeline_state[
                    "artifacts"
                ].get("evaluation_results")

            # Step 4: Model Validation
            if start_step in [None, "data", "training", "evaluation", "validation"]:
                self._run_model_validation()
            elif self.pipeline_state.get("model_validation_completed"):
                self.logger.info("✅ Model validation already completed - skipping")
                self.training_artifacts["validation_results"] = self.pipeline_state[
                    "artifacts"
                ].get("validation_results")

            # Step 5: Model Registration (conditional)
            validation_passed = self.training_artifacts["validation_results"][
                "validation_passed"
            ]
            registration_config = self.config_manager.get_model_registration_config()

            if validation_passed and not registration_config.dry_run:
                if start_step in [
                    None,
                    "data",
                    "training",
                    "evaluation",
                    "validation",
                    "registration",
                ]:
                    self._run_model_registration()
                elif self.pipeline_state.get("model_registration_completed"):
                    self.logger.info(
                        "✅ Model registration already completed - skipping"
                    )
                    self.training_artifacts["registration_results"] = (
                        self.pipeline_state["artifacts"].get("registration_results")
                    )
            else:
                self.logger.warning(
                    "⚠️ Skipping model registration (validation failed or dry run mode)"
                )

            # Finalize pipeline
            self._finalize_pipeline()

            return self.training_artifacts

        except Exception as e:
            self.logger.error(f"❌ Training pipeline failed: {str(e)}")
            self.training_artifacts["error"] = str(e)
            self.training_artifacts["training_successful"] = False
            self.pipeline_state["training_successful"] = False
            self._save_pipeline_state()
            raise MLPipelineException(f"Training pipeline failed: {str(e)}") from e

    def _determine_resume_step(self) -> Optional[str]:
        """Determine which step to resume from based on pipeline state."""
        if not self.pipeline_state.get("data_pipeline_completed"):
            return "data"
        elif not self.pipeline_state.get("model_training_completed"):
            return "training"
        elif not self.pipeline_state.get("model_evaluation_completed"):
            return "evaluation"
        elif not self.pipeline_state.get("model_validation_completed"):
            return "validation"
        elif not self.pipeline_state.get("model_registration_completed"):
            return "registration"
        else:
            self.logger.info("🔄 All steps completed, re-running registration")
            return "registration"

    def _run_data_pipeline(self) -> None:
        """Execute data pipeline step."""
        try:
            self.logger.info("📊 Step 1: Data Pipeline")
            self.pipeline_state["current_step"] = "data_pipeline"

            # Run data pipeline
            data_result = self.data_pipeline.run_full_pipeline()

            # Update state
            self.training_artifacts["processed_data_path"] = data_result[
                "processed_data_path"
            ]
            self.pipeline_state["data_pipeline_completed"] = True
            self.pipeline_state["artifacts"]["processed_data_path"] = data_result[
                "processed_data_path"
            ]

            self.logger.info(
                f"✅ Data pipeline completed: {data_result['processed_data_path']}"
            )
            self._save_checkpoint("data_pipeline_completed")

        except Exception as e:
            raise MLPipelineException(f"Data pipeline failed: {str(e)}") from e

    def _use_existing_data(self, data_path: str) -> None:
        """Use existing processed data."""
        if not Path(data_path).exists():
            raise MLPipelineException(f"Data path does not exist: {data_path}")

        self.training_artifacts["processed_data_path"] = data_path
        self.pipeline_state["data_pipeline_completed"] = True
        self.pipeline_state["artifacts"]["processed_data_path"] = data_path

        self.logger.info(f"📊 Using existing data: {data_path}")

    def _run_model_training(self) -> None:
        """Execute model training step."""
        try:
            self.logger.info("🎯 Step 2: Model Training")
            self.pipeline_state["current_step"] = "model_training"

            # Initialize trainer
            training_config = self.config_manager.get_model_training_config()
            # Set run-specific output directory
            training_config = training_config.__class__(
                **{
                    **training_config.__dict__,
                    "output_dir": self.run_artifacts_dir / "model",
                }
            )

            self.model_trainer = ModelTrainer(training_config)

            # Prepare datasets
            datasets = self.model_trainer.prepare_datasets(
                self.training_artifacts["processed_data_path"]
            )

            # Build model
            model = self.model_trainer.build_model()

            # Train model
            training_result = self.model_trainer.train(datasets)

            # Save model
            model_path = self.model_trainer.save()

            # Update state
            self.training_artifacts["model_path"] = model_path
            self.training_artifacts["training_metrics"] = training_result
            self.pipeline_state["model_training_completed"] = True
            self.pipeline_state["artifacts"]["model_path"] = model_path
            self.pipeline_state["artifacts"]["training_metrics"] = training_result

            self.logger.info(f"✅ Model training completed: {model_path}")
            self._save_checkpoint("model_training_completed")

        except Exception as e:
            raise MLPipelineException(f"Model training failed: {str(e)}") from e

    def _run_model_evaluation(self) -> None:
        """Execute model evaluation step."""
        try:
            self.logger.info("📈 Step 3: Model Evaluation")
            self.pipeline_state["current_step"] = "model_evaluation"

            # Initialize evaluator
            evaluation_config = self.config_manager.get_model_evaluation_config()
            # Set run-specific output directory
            evaluation_config = evaluation_config.__class__(
                **{
                    **evaluation_config.__dict__,
                    "evaluation_output_dir": self.run_artifacts_dir / "evaluation",
                }
            )

            self.model_evaluator = ModelEvaluator(evaluation_config)

            # Load model
            self.model_evaluator.load_model(self.training_artifacts["model_path"])

            # Get test dataset path from training results
            test_dataset_path = None
            if "test_dataset_path" in self.training_artifacts.get(
                "training_metrics", {}
            ):
                test_dataset_path = self.training_artifacts["training_metrics"][
                    "test_dataset_path"
                ]
            else:
                # Fallback: look for test dataset in model output directory
                model_dir = Path(self.training_artifacts["model_path"]).parent
                tokenized_data_dir = model_dir / "tokenized_data"
                test_path = tokenized_data_dir / "test"
                if test_path.exists():
                    test_dataset_path = str(test_path)
                else:
                    # Last resort: use the processed data file (not ideal, but backward compatible)
                    self.logger.warning(
                        "⚠️ Test dataset not found, using full processed data for evaluation"
                    )
                    test_dataset_path = self.training_artifacts["processed_data_path"]

            self.logger.info(f"📊 Using test dataset: {test_dataset_path}")

            # Run evaluation on test data
            evaluation_result = self.model_evaluator.evaluate(test_dataset_path)

            # Update state
            self.training_artifacts["evaluation_results"] = evaluation_result
            self.pipeline_state["model_evaluation_completed"] = True
            self.pipeline_state["artifacts"]["evaluation_results"] = evaluation_result

            self.logger.info(f"✅ Model evaluation completed")
            self.logger.info(
                f"   Test Accuracy: {evaluation_result['metrics']['accuracy']:.4f}"
            )
            self.logger.info(
                f"   Test F1-score: {evaluation_result['metrics']['f1']:.4f}"
            )

            self._save_checkpoint("model_evaluation_completed")

        except Exception as e:
            raise MLPipelineException(f"Model evaluation failed: {str(e)}") from e

    def _run_model_validation(self) -> None:
        """Execute model validation step."""
        try:
            self.logger.info("🔍 Step 4: Model Validation")
            self.pipeline_state["current_step"] = "model_validation"

            # Initialize validator
            validation_config = self.config_manager.get_model_validation_config()
            # Set run-specific output directory
            validation_config = validation_config.__class__(
                **{
                    **validation_config.__dict__,
                    "validation_output_dir": self.run_artifacts_dir / "validation",
                }
            )

            self.model_validator = ModelValidator(validation_config)

            # Get predictions from evaluation results
            if (
                hasattr(self.model_evaluator, "predictions")
                and self.model_evaluator.predictions is not None
            ):
                predictions = self.model_evaluator.predictions
                true_labels = self.model_evaluator.true_labels
                probabilities = self.model_evaluator.probabilities
            else:
                raise MLPipelineException(
                    "No predictions available from evaluation step"
                )

            # Run validation
            validation_result = self.model_validator.validate_model(
                predictions=predictions,
                true_labels=true_labels,
                probabilities=probabilities,
                metadata=self.training_artifacts["evaluation_results"],
            )

            # Update state
            self.training_artifacts["validation_results"] = validation_result
            self.pipeline_state["model_validation_completed"] = True
            self.pipeline_state["artifacts"]["validation_results"] = validation_result

            validation_status = (
                "✅ PASSED" if validation_result["validation_passed"] else "❌ FAILED"
            )
            self.logger.info(f"{validation_status} Model validation completed")

            if not validation_result["validation_passed"]:
                self.logger.warning("⚠️ Model failed validation checks:")
                for issue in validation_result["validation_issues"]:
                    self.logger.warning(f"   - {issue}")

            self._save_checkpoint("model_validation_completed")

        except Exception as e:
            raise MLPipelineException(f"Model validation failed: {str(e)}") from e

    def _run_model_registration(self) -> None:
        """Execute model registration step."""
        try:
            self.logger.info("📦 Step 5: Model Registration")
            self.pipeline_state["current_step"] = "model_registration"

            # Initialize registrar
            registration_config = self.config_manager.get_model_registration_config()
            self.model_registrar = ModelRegistrar(registration_config)

            # Prepare metadata
            metadata = {
                "run_id": self.run_id,
                "training_metrics": self.training_artifacts.get("training_metrics", {}),
                "evaluation_results": self.training_artifacts.get(
                    "evaluation_results", {}
                ),
                "validation_results": self.training_artifacts.get(
                    "validation_results", {}
                ),
                "model_path": self.training_artifacts["model_path"],
                "training_config": self.config_manager.get_model_training_config().__dict__,
                "training_timestamp": self.training_artifacts["start_time"],
            }

            # Register model
            registration_result = self.model_registrar.register_model(
                model_path=self.training_artifacts["model_path"],
                metadata=metadata,
                validation_passed=self.training_artifacts["validation_results"][
                    "validation_passed"
                ],
            )

            # Update state
            self.training_artifacts["registration_results"] = registration_result
            self.pipeline_state["model_registration_completed"] = True
            self.pipeline_state["artifacts"][
                "registration_results"
            ] = registration_result

            if registration_result["success"]:
                self.logger.info(f"✅ Model registration completed successfully")
                for registry, result in registration_result["registrations"].items():
                    self.logger.info(
                        f"   {registry}: {result.get('status', 'unknown')}"
                    )
            else:
                self.logger.warning(f"⚠️ Model registration completed with errors")
                for registry, error in registration_result["errors"].items():
                    self.logger.error(f"   {registry}: {error}")

            self._save_checkpoint("model_registration_completed")

        except Exception as e:
            raise MLPipelineException(f"Model registration failed: {str(e)}") from e

    def _finalize_pipeline(self) -> None:
        """Finalize the training pipeline."""
        self.training_artifacts["end_time"] = datetime.now().isoformat()
        self.training_artifacts["training_successful"] = True

        self.pipeline_state["training_successful"] = True
        self.pipeline_state["current_step"] = "completed"
        self.pipeline_state["completed_at"] = self.training_artifacts["end_time"]

        # Save final artifacts metadata
        artifacts_metadata = {
            "pipeline_artifacts": self.training_artifacts,
            "pipeline_state": self.pipeline_state,
            "config_snapshot": {
                "training_config": self.config_manager.get_model_training_config().__dict__,
                "evaluation_config": self.config_manager.get_model_evaluation_config().__dict__,
                "validation_config": self.config_manager.get_model_validation_config().__dict__,
                "registration_config": self.config_manager.get_model_registration_config().__dict__,
            },
        }

        artifacts_file = self.run_artifacts_dir / "pipeline_artifacts.json"
        with open(artifacts_file, "w") as f:
            json.dump(artifacts_metadata, f, indent=2, default=str)

        self._save_pipeline_state()

        self.logger.info("🎉 Training pipeline completed successfully!")
        self.logger.info(f"📁 Artifacts saved to: {self.run_artifacts_dir}")

    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save a pipeline checkpoint."""
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "pipeline_state": self.pipeline_state,
            "training_artifacts": self.training_artifacts,
            "timestamp": datetime.now().isoformat(),
        }

        self.save_checkpoint(checkpoint_name, checkpoint_data)
        self._save_pipeline_state()

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status."""
        return {
            "run_id": self.run_id,
            "pipeline_state": self.pipeline_state.copy(),
            "training_artifacts": self.training_artifacts.copy(),
            "current_step": self.pipeline_state.get("current_step", "unknown"),
            "is_complete": self.pipeline_state.get("training_successful", False),
            "artifacts_dir": str(self.run_artifacts_dir),
        }


def main():
    """CLI interface for training pipeline."""
    parser = argparse.ArgumentParser(description="ParaDetect Training Pipeline")

    parser.add_argument("--run-id", type=str, help="Unique run identifier")

    parser.add_argument("--config", type=str, help="Path to custom config file")

    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )

    parser.add_argument(
        "--resume-from-step",
        type=str,
        choices=["data", "training", "evaluation", "validation", "registration"],
        help="Specific step to resume from",
    )

    parser.add_argument(
        "--use-existing-data",
        type=str,
        help="Path to existing processed data (skip data pipeline)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no model registration)",
    )

    parser.add_argument(
        "--status", action="store_true", help="Show training status and exit"
    )

    args = parser.parse_args()

    try:
        # Initialize configuration manager
        if args.config:
            config_manager = ConfigurationManager(args.config)
        else:
            config_manager = ConfigurationManager()

        # Override dry_run if specified
        if args.dry_run:
            # This would require modifying the config - for now just log
            print("🔧 Dry-run mode enabled")

        # Initialize pipeline
        pipeline = TrainingPipeline(config_manager, args.run_id)

        # Show status if requested
        if args.status:
            status = pipeline.get_training_status()
            print(f"📊 Training Pipeline Status:")
            print(f"  Run ID: {status['run_id']}")
            print(f"  Current Step: {status['current_step']}")
            print(f"  Complete: {status['is_complete']}")
            print(f"  Artifacts Dir: {status['artifacts_dir']}")
            return

        # Run pipeline
        with pipeline:
            result = pipeline.run(
                use_existing_data=bool(args.use_existing_data),
                data_path=args.use_existing_data,
                resume_from_step=args.resume_from_step,
            )

        # Print results
        if result["training_successful"]:
            print("🎉 Training pipeline completed successfully!")
            print(f"📁 Model saved to: {result['model_path']}")
            print(f"📊 Run ID: {result['run_id']}")

            # Print key metrics
            if "evaluation_results" in result:
                metrics = result["evaluation_results"]["metrics"]
                print(f"📈 Final Metrics:")
                print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"   F1-score: {metrics.get('f1', 0):.4f}")

            # Print validation status
            if "validation_results" in result:
                validation_passed = result["validation_results"]["validation_passed"]
                print(
                    f"🔍 Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}"
                )

            # Print registration status
            if "registration_results" in result:
                reg_success = result["registration_results"]["success"]
                print(
                    f"📦 Registration: {'✅ SUCCESS' if reg_success else '❌ FAILED'}"
                )
        else:
            print("❌ Training pipeline failed!")
            if "error" in result:
                print(f"Error: {result['error']}")

    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
