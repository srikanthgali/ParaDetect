"""
Complete Training Pipeline for ParaDetect
Orchestrates: Data Pipeline + Model Training + Evaluation + Validation + Registration
"""

import argparse
import json
import hashlib
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

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
    - Simple timestamp-based run IDs
    - Automatic resumption of incomplete runs
    - State-driven pipeline execution
    - Clean artifact management
    - Configurable validation gates
    - Multi-registry model publication
    """

    def __init__(
        self,
        config_manager: Optional[ConfigurationManager] = None,
        force_new_run: bool = False,
        dry_run: bool = False,
    ):
        """
        Initialize training pipeline.

        Args:
            config_manager: Configuration manager instance
            force_new_run: Force new run (don't resume incomplete)
            dry_run: Skip model registration step
        """
        # Initialize config manager first
        self.config_manager = config_manager or ConfigurationManager()

        # Store dry_run parameter
        self.dry_run = dry_run

        # Generate run ID
        self.run_id = self._generate_run_id()

        # Initialize base pipeline
        super().__init__(PipelineType.TRAINING, self.config_manager)
        self.logger = get_logger(self.__class__.__name__)

        # Determine resumption strategy
        self.resume_info = self._determine_resumption_strategy(force_new_run)

        # Set up directories and state
        self._setup_directories()
        self.training_artifacts = self._initialize_artifacts()

        # Initialize components (lazy loading)
        self.data_pipeline = None
        self.model_trainer = None
        self.model_evaluator = None
        self.model_validator = None
        self.model_registrar = None

        self.logger.info(f"🔧 Training Pipeline - {self.resume_info['status']}")
        self.logger.info(f"📁 Working directory: {self.run_dir}")

    def _generate_run_id(self) -> str:
        """Generate simple timestamp-based run ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _determine_resumption_strategy(self, force_new_run: bool) -> Dict[str, Any]:
        """
        Determine resumption strategy based on existing state.

        Returns:
            Dict with resumption information
        """
        if force_new_run:
            return {
                "status": "Starting new run (forced)",
                "resume_from": None,
                "previous_run": None,
                "action": "new",
            }

        # Check for incomplete runs
        incomplete_run = self._find_most_recent_incomplete_run()
        if incomplete_run:
            # Resume the incomplete run
            self.run_id = incomplete_run["run_id"]
            return {
                "status": f"Resuming incomplete run: {incomplete_run['run_id']}",
                "resume_from": incomplete_run["last_completed_step"],
                "previous_run": incomplete_run,
                "action": "resume",
            }

        # Check if current config has a recent successful run
        recent_successful = self._find_recent_successful_run()
        if recent_successful:
            return {
                "status": f"Starting new run (previous successful: {recent_successful['run_id']})",
                "resume_from": None,
                "previous_run": recent_successful,
                "action": "new",
            }

        # First time running
        return {
            "status": "Starting first training run",
            "resume_from": None,
            "previous_run": None,
            "action": "new",
        }

    def _find_most_recent_incomplete_run(self) -> Optional[Dict[str, Any]]:
        """Find the most recent incomplete run."""
        # Use checkpoint_dir from base class
        state_file = self.checkpoint_dir / "current_run.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file, "r") as f:
                current_run_info = json.load(f)

            run_id = current_run_info.get("run_id")
            if not run_id:
                return None

            run_dir = self.checkpoint_dir / run_id
            if not run_dir.exists():
                # Cleanup stale state
                state_file.unlink()
                return None

            # Check if run is actually incomplete
            if self._is_run_complete(run_id):
                # Run is complete, cleanup state
                state_file.unlink()
                return None

            return {
                "run_id": run_id,
                "last_completed_step": current_run_info.get("last_completed_step"),
                "started_at": current_run_info.get("started_at"),
                "config_hash": current_run_info.get("config_hash"),
            }

        except Exception as e:
            self.logger.warning(f"Could not read current run state: {e}")
            # Cleanup corrupted state
            if state_file.exists():
                state_file.unlink()
            return None

    def _find_recent_successful_run(self) -> Optional[Dict[str, Any]]:
        """Find recent successful run with same config."""
        config_hash = self._get_config_hash()

        # Look for recent successful runs (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)

        if not self.checkpoint_dir.exists():
            return None

        successful_runs = []
        for run_dir in self.checkpoint_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.replace("_", "").isdigit():
                continue

            if not self._is_run_complete(run_dir.name):
                continue

            # Check if run is recent
            try:
                run_date = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")
                if run_date < cutoff_date:
                    continue
            except ValueError:
                continue

            # Check config compatibility
            artifacts_file = run_dir / "pipeline_artifacts.json"
            if artifacts_file.exists():
                try:
                    with open(artifacts_file, "r") as f:
                        artifacts = json.load(f)

                    run_config_hash = artifacts.get("config_snapshot", {}).get(
                        "config_hash"
                    )
                    if run_config_hash == config_hash:
                        successful_runs.append(
                            {
                                "run_id": run_dir.name,
                                "completed_at": artifacts.get(
                                    "pipeline_artifacts", {}
                                ).get("end_time"),
                                "config_hash": run_config_hash,
                            }
                        )
                except Exception:
                    continue

        # Return most recent successful run
        if successful_runs:
            successful_runs.sort(key=lambda x: x["completed_at"] or "", reverse=True)
            return successful_runs[0]

        return None

    def _setup_directories(self):
        """Set up directory structure without symlinks."""
        self.run_dir = self.checkpoint_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Update current run state
        self._update_current_run_state()

    def _update_current_run_state(self):
        """Update the current run state file."""
        state_file = self.checkpoint_dir / "current_run.json"

        state = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "config_hash": self._get_config_hash(),
            "last_completed_step": None,
            "status": "active",
        }

        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not update current run state: {e}")

    def _get_config_hash(self) -> str:
        """Get configuration hash for compatibility checking."""
        training_config = self.config_manager.get_model_training_config()

        # Include key parameters that affect training compatibility
        config_elements = {
            "model_name": training_config.model_name_or_path,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.per_device_train_batch_size,
            "epochs": training_config.num_train_epochs,
            "max_length": training_config.max_length,
            "use_peft": training_config.use_peft,
        }

        if training_config.use_peft and training_config.peft_config:
            config_elements.update(
                {
                    "lora_r": training_config.peft_config.get("r", 8),
                    "lora_alpha": training_config.peft_config.get("lora_alpha", 16),
                }
            )

        config_str = json.dumps(config_elements, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _initialize_artifacts(self) -> Dict[str, Any]:
        """Initialize training artifacts."""
        artifacts_file = self.run_dir / "pipeline_artifacts.json"

        # Try to load existing artifacts if resuming
        if self.resume_info["action"] == "resume" and artifacts_file.exists():
            try:
                with open(artifacts_file, "r") as f:
                    data = json.load(f)
                artifacts = data.get("pipeline_artifacts", {})
                self.logger.info("📂 Loaded existing training artifacts")
                return artifacts
            except Exception as e:
                self.logger.warning(f"Could not load existing artifacts: {e}")

        # Create new artifacts
        return {
            "run_id": self.run_id,
            "start_time": None,
            "end_time": None,
            "processed_data_path": None,
            "model_path": None,
            "evaluation_results": None,
            "validation_results": None,
            "registration_results": None,
            "training_successful": False,
            "artifacts_dir": str(self.run_dir),
            "config_hash": self._get_config_hash(),
        }

    def _get_initial_state(self) -> Dict[str, Any]:
        """Get initial state for training pipeline."""
        return {
            "run_id": getattr(self, "run_id", "unknown"),
            "data_pipeline_completed": False,
            "model_training_completed": False,
            "model_evaluation_completed": False,
            "model_validation_completed": False,
            "model_registration_completed": False,
            "training_successful": False,
            "current_step": "initialization",
            "last_checkpoint": None,
        }

    def run(
        self,
        use_existing_data: bool = False,
        data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the training pipeline with automatic resumption.

        The pipeline automatically determines where to start based on
        previous state - no manual step specification needed.
        """
        try:
            self.logger.info(f"🚀 Starting training pipeline - Run ID: {self.run_id}")
            self.training_artifacts["start_time"] = datetime.now().isoformat()

            # Auto-determine starting point
            start_step = self._determine_start_step(use_existing_data, data_path)
            self.logger.info(f"📍 Starting from: {start_step}")

            # Execute pipeline steps
            self._execute_pipeline_steps(start_step, use_existing_data, data_path)

            # Finalize
            self._finalize_pipeline()

            return self.training_artifacts

        except Exception as e:
            self.logger.error(f"❌ Training pipeline failed: {str(e)}")
            self.training_artifacts["error"] = str(e)
            self.training_artifacts["training_successful"] = False
            self._save_artifacts()
            raise MLPipelineException(f"Training pipeline failed: {str(e)}") from e

    def _determine_start_step(
        self, use_existing_data: bool, data_path: Optional[str]
    ) -> str:
        """Automatically determine where to start the pipeline."""

        # If resuming, use the resume point
        if self.resume_info["action"] == "resume":
            last_step = self.resume_info["resume_from"]
            if last_step == "data_pipeline":
                return "model_training"
            elif last_step == "model_training":
                return "model_evaluation"
            elif last_step == "model_evaluation":
                return "model_validation"
            elif last_step == "model_validation":
                return "model_registration"
            else:
                return "data_pipeline"

        # For new runs, start from the beginning unless data is provided
        if use_existing_data and data_path:
            return "model_training"

        return "data_pipeline"

    def _execute_pipeline_steps(
        self, start_step: str, use_existing_data: bool, data_path: Optional[str]
    ):
        """Execute pipeline steps starting from the specified step."""

        steps = [
            "data_pipeline",
            "model_training",
            "model_evaluation",
            "model_validation",
            "model_registration",
        ]
        start_index = steps.index(start_step)

        for step in steps[start_index:]:
            if step == "data_pipeline":
                if use_existing_data and data_path:
                    self._use_existing_data(data_path)
                else:
                    self._run_data_pipeline()
            elif step == "model_training":
                self._run_model_training(use_existing_data, data_path)
            elif step == "model_evaluation":
                self._run_model_evaluation()
            elif step == "model_validation":
                self._run_model_validation()
            elif step == "model_registration":
                validation_passed = self.training_artifacts["validation_results"][
                    "validation_passed"
                ]
                registration_config = (
                    self.config_manager.get_model_registration_config()
                )
                skip_registration = self.dry_run or registration_config.dry_run

                if not skip_registration:
                    self._run_model_registration()
                else:
                    if self.dry_run:
                        self.logger.info(
                            "⏭️ Skipping model registration (pipeline dry run enabled)"
                        )
                    else:
                        self.logger.info(
                            "⏭️ Skipping model registration (config dry run enabled)"
                        )

            # Update progress after each step
            self._update_step_progress(step)

    def _update_step_progress(self, completed_step: str):
        """Update progress in current run state."""
        state_file = self.checkpoint_dir / "current_run.json"

        try:
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
            else:
                state = {"run_id": self.run_id}

            state["last_completed_step"] = completed_step
            state["last_updated"] = datetime.now().isoformat()

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Could not update step progress: {e}")

    def _finalize_pipeline(self):
        """Finalize pipeline and cleanup state."""
        self.training_artifacts["end_time"] = datetime.now().isoformat()
        self.training_artifacts["training_successful"] = True

        # Save final artifacts
        self._save_artifacts()

        # Create completion marker
        completion_marker = self.run_dir / "TRAINING_COMPLETE"
        with open(completion_marker, "w") as f:
            f.write(f"Training completed at: {self.training_artifacts['end_time']}\n")
            f.write(f"Run ID: {self.run_id}\n")

        # Cleanup current run state
        current_run_state = self.checkpoint_dir / "current_run.json"
        if current_run_state.exists():
            current_run_state.unlink()

        self.logger.info("🎉 Training pipeline completed successfully!")
        self.logger.info(f"📁 Artifacts saved to: {self.run_dir}")

    def _save_artifacts(self):
        """Save training artifacts."""
        artifacts_metadata = {
            "pipeline_artifacts": self.training_artifacts,
            "config_snapshot": {
                "config_hash": self._get_config_hash(),
                "training_config": self.config_manager.get_model_training_config().__dict__,
                "evaluation_config": self.config_manager.get_model_evaluation_config().__dict__,
                "validation_config": self.config_manager.get_model_validation_config().__dict__,
                "registration_config": self.config_manager.get_model_registration_config().__dict__,
            },
            "saved_at": datetime.now().isoformat(),
        }

        artifacts_file = self.run_dir / "pipeline_artifacts.json"
        with open(artifacts_file, "w") as f:
            json.dump(artifacts_metadata, f, indent=2, default=str)

    def _is_run_complete(self, run_id: str) -> bool:
        """Check if a run is complete."""
        run_dir = self.checkpoint_dir / run_id
        return (run_dir / "TRAINING_COMPLETE").exists()

    def _resolve_processed_data_path(self) -> Optional[str]:
        """
        Resolve processed data path from artifacts or fallback to latest data.

        Returns:
            str: Path to processed data file
        """
        # Try to get from current artifacts
        processed_path = self.training_artifacts.get("processed_data_path")
        if processed_path and Path(processed_path).exists():
            return processed_path

        # Try to load from saved artifacts
        artifacts_file = self.run_dir / "pipeline_artifacts.json"
        if artifacts_file.exists():
            try:
                with open(artifacts_file, "r") as f:
                    saved_artifacts = json.load(f)

                saved_path = saved_artifacts.get("pipeline_artifacts", {}).get(
                    "processed_data_path"
                )
                if saved_path and Path(saved_path).exists():
                    return saved_path
            except Exception as e:
                self.logger.warning(f"Could not load saved artifacts: {e}")

        # Fallback: find the most recent processed data file
        data_dir = Path("data/processed")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                # Get the most recently modified CSV file
                latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                self.logger.info(f"📂 Using fallback data path: {latest_file}")
                return str(latest_file)

        # Last resort: check common locations
        common_paths = [
            "data/processed/ai_text_detection_pile_processed.csv",
            "data/processed/AI_Text_Detection_Pile_sampled_50k.csv",
        ]

        for path in common_paths:
            if Path(path).exists():
                self.logger.info(f"📂 Using common data path: {path}")
                return path

        return None

    # Pipeline step implementations
    def _run_data_pipeline(self):
        """Execute data pipeline step."""
        self.logger.info("📊 Step 1: Data Pipeline")

        if self.data_pipeline is None:
            self.data_pipeline = DataPipeline(self.config_manager)

        data_result = self.data_pipeline.run_full_pipeline()
        self.training_artifacts["processed_data_path"] = data_result[
            "processed_data_path"
        ]

        self._save_artifacts()

        self.logger.info(
            f"✅ Data pipeline completed: {data_result['processed_data_path']}"
        )

    def _use_existing_data(self, data_path: str) -> None:
        """Use existing processed data."""
        if not Path(data_path).exists():
            raise MLPipelineException(f"Data path does not exist: {data_path}")

        self.training_artifacts["processed_data_path"] = data_path
        self.logger.info(f"📊 Using existing data: {data_path}")

    def _run_model_training(
        self, use_existing_data: bool, existing_data_path: Optional[str] = None
    ):
        """Execute model training step."""
        self.logger.info("🎯 Step 2: Model Training")

        # Resolve the data path
        data_path = (
            existing_data_path
            if use_existing_data and existing_data_path
            else self._resolve_processed_data_path()
        )
        if not data_path:
            raise MLPipelineException(
                "No processed data found. Please run data pipeline first or provide --use-existing-data"
            )
        # Update artifacts with resolved path
        self.training_artifacts["processed_data_path"] = data_path
        self._save_artifacts()  # Persist immediately

        # Initialize trainer with simple run-specific output directory
        training_config = self.config_manager.get_model_training_config()
        from dataclasses import replace

        training_config = replace(
            training_config,
            output_dir=self.run_dir / "model",
            train_path=data_path,  # Ensure config has the data path
        )

        self.model_trainer = ModelTrainer(training_config)

        # Build model and prepare datasets
        model = self.model_trainer.build_model()
        datasets = self.model_trainer.prepare_datasets(data_path)

        # Train and save
        training_result = self.model_trainer.train(datasets)
        model_path = self.model_trainer.save()

        self.training_artifacts["model_path"] = model_path
        self.training_artifacts["training_metrics"] = training_result

        self.logger.info(f"✅ Model training completed: {model_path}")

    def _run_model_evaluation(self):
        """Execute model evaluation step."""
        self.logger.info("📈 Step 3: Model Evaluation")

        # Initialize evaluator
        evaluation_config = self.config_manager.get_model_evaluation_config()
        from dataclasses import replace

        evaluation_config = replace(
            evaluation_config, evaluation_output_dir=self.run_dir / "evaluation"
        )

        self.model_evaluator = ModelEvaluator(evaluation_config)

        # Load model and evaluate
        self.model_evaluator.load_model(self.training_artifacts["model_path"])

        # Get test dataset path from training results
        test_dataset_path = self.training_artifacts.get("training_metrics", {}).get(
            "test_dataset_path"
        )
        if not test_dataset_path:
            # Fallback to processed data
            test_dataset_path = self.training_artifacts["processed_data_path"]

        evaluation_result = self.model_evaluator.evaluate(test_dataset_path)
        self.training_artifacts["evaluation_results"] = evaluation_result

        self.logger.info(f"✅ Model evaluation completed")
        self.logger.info(
            f"   Test Accuracy: {evaluation_result['metrics']['accuracy']:.4f}"
        )
        self.logger.info(f"   Test F1-score: {evaluation_result['metrics']['f1']:.4f}")

    def _run_model_validation(self):
        """Execute model validation step."""
        self.logger.info("🔍 Step 4: Model Validation")

        # Initialize validator
        validation_config = self.config_manager.get_model_validation_config()
        from dataclasses import replace

        validation_config = replace(
            validation_config, validation_output_dir=self.run_dir / "validation"
        )

        self.model_validator = ModelValidator(validation_config)

        # Get predictions from evaluation results
        evaluation_results = self.training_artifacts["evaluation_results"]
        predictions = evaluation_results.get("predictions", [])
        true_labels = evaluation_results.get("true_labels", [])
        probabilities = evaluation_results.get("probabilities", [])

        # Run validation
        validation_result = self.model_validator.validate_model(
            predictions=predictions,
            true_labels=true_labels,
            probabilities=probabilities,
            metadata=evaluation_results,
        )

        self.training_artifacts["validation_results"] = validation_result

        validation_status = (
            "✅ PASSED" if validation_result["validation_passed"] else "❌ FAILED"
        )
        self.logger.info(f"{validation_status} Model validation completed")

    def _run_model_registration(self):
        """Execute model registration step."""
        self.logger.info("📦 Step 5: Model Registration")

        # Initialize registrar
        registration_config = self.config_manager.get_model_registration_config()

        # Override dry_run if set at pipeline level
        if self.dry_run:
            registration_config.dry_run = True

        self.model_registrar = ModelRegistrar(registration_config)

        # Log the paths for better visibility
        training_model_path = self.training_artifacts["model_path"]
        self.logger.info(f"📁 Training model location: {training_model_path}")
        self.logger.info(
            f"📦 Registry location: {registration_config.local_registry_dir}"
        )

        # Prepare comprehensive metadata for model card generation
        training_config = self.config_manager.get_model_training_config()
        evaluation_config = self.config_manager.get_model_evaluation_config()

        # Extract actual training results and metrics
        training_metrics = self.training_artifacts.get("training_metrics", {})
        evaluation_results = self.training_artifacts.get("evaluation_results", {})
        validation_results = self.training_artifacts.get("validation_results", {})

        # Prepare comprehensive metadata
        metadata = {
            # Training configuration (serializable format)
            "training_config": {
                "model_name_or_path": training_config.model_name_or_path,
                "num_train_epochs": training_config.num_train_epochs,
                "per_device_train_batch_size": training_config.per_device_train_batch_size,
                "per_device_eval_batch_size": training_config.per_device_eval_batch_size,
                "learning_rate": float(training_config.learning_rate),
                "weight_decay": float(training_config.weight_decay),
                "warmup_ratio": float(training_config.warmup_ratio),
                "max_grad_norm": float(training_config.max_grad_norm),
                "max_length": training_config.max_length,
                "use_peft": training_config.use_peft,
                "fp16": training_config.fp16,
                "bf16": training_config.bf16,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "early_stopping_patience": training_config.early_stopping_patience,
                "metric_for_best_model": training_config.metric_for_best_model,
                "eval_strategy": training_config.eval_strategy,
                "eval_steps": training_config.eval_steps,
                "seed": training_config.seed,
            },
            # PEFT/LoRA configuration if available
            "peft_config": {},
            # Training metrics and results
            "training_results": training_metrics,
            # Evaluation metrics (the key ones for model card)
            "metrics": evaluation_results.get("metrics", {}),
            # Validation results
            "validation_results": validation_results,
            # Dataset information
            "dataset_info": {
                "source": "artem9k/ai-text-detection-pile",
                "processed_data_path": self.training_artifacts.get(
                    "processed_data_path"
                ),
                "train_split": float(
                    training_config.validation_split + training_config.test_split
                ),  # Remaining for train
                "val_split": float(training_config.validation_split),
                "test_split": float(training_config.test_split),
            },
            # Pipeline metadata
            "pipeline_metadata": {
                "run_id": self.run_id,
                "training_date": datetime.now().strftime("%Y-%m-%d"),
                "environment": self.config_manager.environment,
                "author": "Srikanth Gali",
                "organization": "Independent Research",
                "repository_url": "https://github.com/srikanthgali/para_detect",
                "contact": "https://github.com/srikanthgali/para_detect/issues",
                "training_artifacts_path": str(
                    self.run_dir
                ),  # Link back to training run
                "config_hash": self._get_config_hash(),
            },
            # Provenance tracking
            "provenance": {
                "training_run_id": self.run_id,
                "training_artifacts_dir": str(self.run_dir),
                "source_model_path": str(training_model_path),
                "created_from": "training_pipeline",
            },
        }

        # Add PEFT configuration details if available
        if training_config.use_peft and training_config.peft_config:
            peft_config = training_config.peft_config
            if hasattr(peft_config, "__dict__"):
                peft_dict = peft_config.__dict__
            else:
                peft_dict = peft_config

            metadata["peft_config"] = {
                "r": peft_dict.get("r", 64),
                "lora_alpha": peft_dict.get("lora_alpha", 128),
                "lora_dropout": float(peft_dict.get("lora_dropout", 0.1)),
                "bias": peft_dict.get("bias", "all"),
                "target_modules": peft_dict.get("target_modules", []),
                "task_type": str(peft_dict.get("task_type", "SEQ_CLS")),
            }

            # Add PEFT info to training config
            metadata["training_config"]["lora_r"] = metadata["peft_config"]["r"]
            metadata["training_config"]["lora_alpha"] = metadata["peft_config"][
                "lora_alpha"
            ]
            metadata["training_config"]["lora_dropout"] = metadata["peft_config"][
                "lora_dropout"
            ]

        # Determine validation status
        validation_passed = (
            validation_results.get("validation_passed", False)
            if validation_results
            else False
        )

        # Register model with comprehensive metadata
        try:
            registration_result = self.model_registrar.register_model(
                model_path=self.training_artifacts["model_path"],
                tokenizer_path=self.training_artifacts[
                    "model_path"
                ],  # Same path typically
                metadata=metadata,
                validation_passed=validation_passed,
            )

            self.training_artifacts["registration_results"] = registration_result

            # Enhanced logging for registration results
            if registration_result.get("success", False):
                self.logger.info("✅ Model registration completed successfully")

                # Log detailed registry information
                if "local" in registration_result.get("registrations", {}):
                    local_result = registration_result["registrations"]["local"]
                    registry_path = local_result.get("registry_location")
                    model_version = local_result.get("version")
                    model_path = local_result.get("model_path")

                    self.logger.info(f"   📍 Registry: {registry_path}")
                    self.logger.info(f"   🏷️  Version: {model_version}")
                    self.logger.info(f"   🤖 Model: {model_path}")
                    self.logger.info(f"   📄 Latest info: {registry_path}/LATEST")

                    # Log convenient access paths
                    if registry_path:
                        registry_dir = Path(registry_path)
                        latest_file = registry_dir / "LATEST"
                        if latest_file.exists():
                            self.logger.info(
                                f"   💡 To load latest model, read: {latest_file}"
                            )
            else:
                self.logger.warning("⚠️ Model registration completed with errors")
                # Log any errors
                errors = registration_result.get("errors", {})
                for registry, error in errors.items():
                    self.logger.warning(f"   {registry}: {error}")

        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {str(e)}")
            # Don't fail the entire pipeline for registration issues
            self.training_artifacts["registration_results"] = {
                "success": False,
                "error": str(e),
            }

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status."""
        return {
            "run_id": self.run_id,
            "current_step": self.pipeline_state.get("current_step", "unknown"),
            "is_complete": self.pipeline_state.get("training_successful", False),
            "artifacts_dir": str(self.run_dir),
            "resume_info": self.resume_info,
        }


def main():
    """CLI interface for training pipeline."""
    parser = argparse.ArgumentParser(description="ParaDetect Training Pipeline")

    parser.add_argument("--config", type=str, help="Path to custom config file")
    parser.add_argument(
        "--force-new-run", action="store_true", help="Force new run (don't resume)"
    )
    parser.add_argument(
        "--use-existing-data", type=str, help="Path to existing processed data"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument(
        "--status", action="store_true", help="Show training status and exit"
    )
    parser.add_argument(
        "--list-runs", action="store_true", help="List all runs and exit"
    )

    args = parser.parse_args()

    try:
        # Initialize configuration manager
        config_manager = (
            ConfigurationManager(args.config) if args.config else ConfigurationManager()
        )

        # Override dry_run if specified
        if args.dry_run:
            # This would need to be implemented in config_manager
            pass

        if args.list_runs:
            # Show available runs
            pipeline = TrainingPipeline(config_manager, force_new_run=True)
            runs_dir = pipeline.checkpoint_dir
            if runs_dir.exists():
                print("📋 Available training runs:")
                for run_dir in runs_dir.iterdir():
                    if run_dir.is_dir() and run_dir.name.replace("_", "").isdigit():
                        status = (
                            "✅ Complete"
                            if (run_dir / "TRAINING_COMPLETE").exists()
                            else "⏸️ Incomplete"
                        )
                        print(f"   {run_dir.name}: {status}")
            else:
                print("No training runs found.")
            return

        # Initialize pipeline
        pipeline = TrainingPipeline(config_manager, args.force_new_run, args.dry_run)

        # Show status if requested
        if args.status:
            status = pipeline.get_training_status()
            print("📊 Training Pipeline Status:")
            print(f"   Run ID: {status['run_id']}")
            print(f"   Current Step: {status['current_step']}")
            print(f"   Is Complete: {status['is_complete']}")
            print(f"   Artifacts Dir: {status['artifacts_dir']}")
            print(f"   Resume Info: {status['resume_info']['status']}")
            return

        # Run pipeline
        with pipeline:
            result = pipeline.run(
                use_existing_data=bool(args.use_existing_data),
                data_path=args.use_existing_data,
            )

        # Print results
        if result["training_successful"]:
            print("🎉 Training pipeline completed successfully!")
            print(f"📁 Artifacts: {result['artifacts_dir']}")
        else:
            print("❌ Training pipeline failed")

    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
