"""
Model Training Component for ParaDetect
Handles DeBERTa fine-tuning with LoRA support, checkpointing, and resumption
"""

import os
import json
import shutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

from para_detect.entities.model_training_config import ModelTrainingConfig
from para_detect.core.exceptions import ModelTrainingError, DeviceError, CheckpointError
from para_detect.constants import (
    DEFAULT_LORA_TARGET_MODULES,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    DEVICE_PRIORITY,
    DEFAULT_RANDOM_STATE,
)
from para_detect.utils.helpers import create_directories
from para_detect import get_logger
from para_detect.constants import LABEL_MAPPING, REVERSE_LABEL_MAPPING


class ModelTrainer:
    """
    Model training component with support for:
    - DeBERTa fine-tuning with full or LoRA adaptation
    - Smart checkpoint resumption
    - Dataset tokenization caching
    - Multi-GPU support
    - Comprehensive logging and monitoring
    """

    def __init__(
        self,
        config: ModelTrainingConfig,
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[AutoModelForSequenceClassification] = None,
    ):
        """
        Initialize model trainer.

        Args:
            config: Training configuration
            tokenizer: Pre-loaded tokenizer (optional)
            model: Pre-loaded model (optional)
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Set random seeds for reproducibility
        set_seed(config.seed)

        # Initialize device
        self.device = self._detect_device()
        self.logger.info(f"🔧 Using device: {self.device}")

        # Model and tokenizer
        self.tokenizer = tokenizer
        self.model = model

        # Training state
        self.training_state = {
            "tokenization_completed": False,
            "model_loaded": False,
            "training_completed": False,
            "last_checkpoint": None,
            "best_metrics": None,
        }

        # Paths
        self.checkpoint_dir = self.config.output_dir / "checkpoints"
        self.tokenized_data_dir = self.config.output_dir / "tokenized_data"
        self.logs_dir = self.config.output_dir / "logs"

        # Create directories
        for directory in [self.checkpoint_dir, self.tokenized_data_dir, self.logs_dir]:
            create_directories(directory)

    def _detect_device(self) -> torch.device:
        """Detect optimal device for training."""
        try:
            if self.config.device_preference:
                if self.config.device_preference == "auto":
                    pass  # Continue with auto-detection
                else:
                    device = torch.device(self.config.device_preference)
                    if device.type == "cuda" and not torch.cuda.is_available():
                        self.logger.warning(
                            "CUDA requested but not available, falling back to auto-detection"
                        )
                    elif device.type == "mps" and not torch.backends.mps.is_available():
                        self.logger.warning(
                            "MPS requested but not available, falling back to auto-detection"
                        )
                    else:
                        return device

            # Auto-detection
            for device_type in DEVICE_PRIORITY:
                if device_type == "cuda" and torch.cuda.is_available():
                    device = torch.device("cuda")
                    self.logger.info(
                        f"🚀 CUDA available: {torch.cuda.get_device_name(0)}"
                    )
                    self.logger.info(
                        f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                    )
                    return device
                elif device_type == "mps" and torch.backends.mps.is_available():
                    self.logger.info("🍎 Using Apple Metal Performance Shaders (MPS)")
                    return torch.device("mps")
                elif device_type == "cpu":
                    self.logger.info("💻 Using CPU for training")
                    return torch.device("cpu")

            return torch.device("cpu")

        except Exception as e:
            raise DeviceError(f"Failed to detect device: {str(e)}") from e

    def prepare_datasets(self, data_path: Optional[str] = None) -> DatasetDict:
        """
        Load and tokenize datasets with caching support.

        Args:
            data_path: Path to training data (CSV or dataset directory)

        Returns:
            DatasetDict: Tokenized train/validation datasets
        """
        try:
            self.logger.info("📊 Preparing datasets...")

            # Check if tokenized datasets already exist
            cached_train_path = self.tokenized_data_dir / "train"
            cached_val_path = self.tokenized_data_dir / "validation"
            cached_test_path = self.tokenized_data_dir / "test"

            if (
                cached_train_path.exists()
                and cached_val_path.exists()
                and cached_test_path.exists()
                and not self._should_retokenize()
            ):
                self.logger.info("📂 Loading cached tokenized datasets...")
                train_dataset = load_from_disk(str(cached_train_path))
                val_dataset = load_from_disk(str(cached_val_path))
                test_dataset = load_from_disk(str(cached_test_path))

                return DatasetDict(
                    {
                        "train": train_dataset,
                        "validation": val_dataset,
                        "test": test_dataset,
                    }
                )

            # Load raw data
            data_path = data_path or self.config.train_path
            if not data_path:
                raise ModelTrainingError("No training data path provided")

            self.logger.info(f"📥 Loading data from: {data_path}")

            if Path(data_path).suffix == ".csv":
                df = pd.read_csv(data_path)
            else:
                raise ModelTrainingError(f"Unsupported data format: {data_path}")

            # Validate required columns
            required_columns = [self.config.text_column, self.config.label_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ModelTrainingError(f"Missing required columns: {missing_columns}")

            # Create train/validation/test splits
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df[self.config.text_column].tolist(),
                df[self.config.label_column].tolist(),
                test_size=self.config.validation_split
                + self.config.test_split,  # Ex: 30% for validation + test
                random_state=self.config.seed,
                stratify=df[self.config.label_column].tolist(),
            )

            # Split the remaining 30% into validation (15%) and test (15%)
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts,
                temp_labels,
                test_size=self.config.test_split
                / (
                    self.config.validation_split + self.config.test_split
                ),  # Half of 30% = 15% each
                random_state=self.config.seed,
                stratify=temp_labels,
            )

            self.logger.info(f"📈 Dataset splits:")
            self.logger.info(
                f"   Train: {len(train_texts):,} samples ({len(train_texts)/len(df)*100:.1f}%)"
            )
            self.logger.info(
                f"   Validation: {len(val_texts):,} samples ({len(val_texts)/len(df)*100:.1f}%)"
            )
            self.logger.info(
                f"   Test: {len(test_texts):,} samples ({len(test_texts)/len(df)*100:.1f}%)"
            )

            # Create datasets
            train_dataset = Dataset.from_dict(
                {"text": train_texts, "labels": train_labels}
            )

            val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

            test_dataset = Dataset.from_dict(
                {"text": test_texts, "labels": test_labels}
            )

            # Tokenize datasets
            self.logger.info("🔤 Tokenizing datasets...")
            train_dataset = train_dataset.map(
                self._tokenize_function, batched=True, desc="Tokenizing train data"
            )

            val_dataset = val_dataset.map(
                self._tokenize_function, batched=True, desc="Tokenizing validation data"
            )

            test_dataset = test_dataset.map(
                self._tokenize_function, batched=True, desc="Tokenizing test data"
            )

            # Remove text columns (no longer needed)
            train_dataset = train_dataset.remove_columns(["text"])
            val_dataset = val_dataset.remove_columns(["text"])
            test_dataset = test_dataset.remove_columns(["text"])

            # Cache tokenized datasets
            self.logger.info("💾 Caching tokenized datasets...")
            train_dataset.save_to_disk(str(cached_train_path))
            val_dataset.save_to_disk(str(cached_val_path))
            test_dataset.save_to_disk(str(cached_test_path))

            # Save test dataset path for evaluation
            test_data_path = self.tokenized_data_dir / "test_dataset_path.txt"
            with open(test_data_path, "w") as f:
                f.write(str(cached_test_path))

            self.training_state["tokenization_completed"] = True

            return DatasetDict(
                {
                    "train": train_dataset,
                    "validation": val_dataset,
                    "test": test_dataset,
                }
            )

        except Exception as e:
            raise ModelTrainingError(f"Failed to prepare datasets: {str(e)}") from e

    def _tokenize_function(self, examples):
        """Tokenize text examples."""
        if not self.tokenizer:
            raise ModelTrainingError("Tokenizer not loaded")

        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors=None,
        )

    def _should_retokenize(self) -> bool:
        """Check if datasets should be retokenized."""
        # Check if tokenizer has changed
        tokenizer_config_path = self.tokenized_data_dir / "tokenizer_config.json"

        if not tokenizer_config_path.exists():
            return True

        try:
            with open(tokenizer_config_path, "r") as f:
                saved_config = json.load(f)

            current_config = {
                "model_name": self.config.model_name_or_path,
                "max_length": self.config.max_length,
            }

            return saved_config != current_config

        except Exception:
            return True

    def build_model(self) -> AutoModelForSequenceClassification:
        """
        Load and configure model with optional LoRA adaptation.

        Returns:
            AutoModelForSequenceClassification: Configured model
        """
        try:
            self.logger.info(f"🤖 Loading model: {self.config.model_name_or_path}")

            # Load tokenizer if not provided
            if not self.tokenizer:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name_or_path or self.config.model_name_or_path
                )
                self.logger.info("✅ Tokenizer loaded")

            # Load base model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name_or_path,
                num_labels=len(LABEL_MAPPING),
                torch_dtype=torch.float32,
                device_map=self.config.device_map,
                id2label=REVERSE_LABEL_MAPPING,
                label2id=LABEL_MAPPING,
            )

            self.logger.info(
                f"✅ Base model loaded: {model.num_parameters():,} parameters"
            )

            # Apply LoRA if configured
            if self.config.use_peft:
                model = self._apply_lora(model)

            # Move to device
            model = model.to(self.device)

            self.model = model
            self.training_state["model_loaded"] = True

            return model

        except Exception as e:
            raise ModelTrainingError(f"Failed to build model: {str(e)}") from e

    def _apply_lora(self, model: AutoModelForSequenceClassification) -> PeftModel:
        """Apply LoRA adaptation to model."""
        try:
            self.logger.info("🔧 Applying LoRA adaptation...")

            # Get LoRA config
            peft_config = self.config.peft_config or {}

            lora_config = LoraConfig(
                task_type=peft_config.task_type,
                r=peft_config.r,
                lora_alpha=peft_config.lora_alpha,
                lora_dropout=peft_config.lora_dropout,
                bias=peft_config.bias,
                target_modules=peft_config.target_modules,
                inference_mode=peft_config.inference_mode,
            )

            # Apply LoRA
            model = get_peft_model(model, lora_config)

            # Print trainable parameters
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())

            self.logger.info(f"🎯 LoRA Configuration:")
            self.logger.info(f"   Rank: {lora_config.r}")
            self.logger.info(f"   Alpha: {lora_config.lora_alpha}")
            self.logger.info(f"   Dropout: {lora_config.lora_dropout}")
            self.logger.info(f"   Target modules: {lora_config.target_modules}")
            self.logger.info(
                f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
            )

            return model

        except Exception as e:
            raise ModelTrainingError(f"Failed to apply LoRA: {str(e)}") from e

    def train(self, datasets: DatasetDict) -> Dict[str, Any]:
        """
        Train model with checkpoint support and resumption.
        Uses only train and validation sets, test set is reserved for evaluation.

        Args:
            datasets: Tokenized training datasets (must include train, validation, test)

        Returns:
            Dict: Training results and metrics
        """
        try:
            self.logger.info("🚀 Starting model training...")

            if not self.model:
                raise ModelTrainingError("Model not loaded. Call build_model() first.")

            # Verify we have the required datasets
            required_sets = ["train", "validation", "test"]
            missing_sets = [s for s in required_sets if s not in datasets]
            if missing_sets:
                raise ModelTrainingError(f"Missing required datasets: {missing_sets}")

            # Log dataset sizes
            self.logger.info(f"📊 Dataset sizes:")
            self.logger.info(f"   Train: {len(datasets['train']):,} samples")
            self.logger.info(f"   Validation: {len(datasets['validation']):,} samples")
            self.logger.info(
                f"   Test: {len(datasets['test']):,} samples (reserved for evaluation)"
            )

            # Setup training arguments
            training_args = self._create_training_arguments()

            # Setup trainer with only train/validation sets
            trainer = self._create_trainer(training_args, datasets)

            # Check for checkpoint resumption
            resume_from_checkpoint = self._get_resume_checkpoint()

            if resume_from_checkpoint:
                self.logger.info(f"🔄 Resuming training from: {resume_from_checkpoint}")

            # Start training
            start_time = datetime.now()

            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            end_time = datetime.now()
            training_duration = end_time - start_time

            self.logger.info(f"✅ Training completed!")
            self.logger.info(f"   Duration: {training_duration}")
            self.logger.info(f"   Final loss: {train_result.training_loss:.4f}")
            self.logger.info(f"   Training steps: {train_result.global_step}")

            # Get final evaluation metrics on validation set
            eval_result = trainer.evaluate()

            # Prepare results
            results = {
                "success": True,
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "training_duration": str(training_duration),
                "validation_metrics": eval_result,  # Renamed to be clear this is validation
                "model_path": str(self.config.output_dir),
                "checkpoint_dir": str(self.checkpoint_dir),
                "device_used": str(self.device),
                "num_parameters": (
                    self.model.num_parameters()
                    if hasattr(self.model, "num_parameters")
                    else None
                ),
                "test_dataset_path": str(
                    self.tokenized_data_dir / "test"
                ),  # Path to test set for evaluation
            }

            self.training_state["training_completed"] = True
            self.training_state["best_metrics"] = eval_result

            return results

        except Exception as e:
            self.logger.error(f"❌ Training failed: {str(e)}")
            raise ModelTrainingError(f"Training failed: {str(e)}") from e

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from config."""
        return TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            # Precision
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            # Evaluation
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            # Saving
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            # Best model
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            # Logging
            logging_steps=self.config.logging_steps,
            logging_dir=str(self.logs_dir),
            report_to=self.config.report_to or [],
            run_name=self.config.run_name,
            # Data loading
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=self.config.remove_unused_columns,
            # Reproducibility
            seed=self.config.seed,
        )

    def _create_trainer(
        self, training_args: TrainingArguments, datasets: DatasetDict
    ) -> Trainer:
        """Create Hugging Face trainer."""
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Early stopping callback
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            )

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )

    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def _get_resume_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint for resumption."""
        if not self.config.resume_from_checkpoint:
            return None

        try:
            # Find all checkpoint directories
            checkpoints = [
                d
                for d in self.config.output_dir.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]

            if not checkpoints:
                self.logger.info("No checkpoints found for resumption")
                return None

            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
            latest_checkpoint = checkpoints[-1]

            self.logger.info(f"Found latest checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint)

        except Exception as e:
            self.logger.warning(f"Error finding checkpoint: {str(e)}")
            return None

    def save(self) -> str:
        """
        Save final model and tokenizer.

        Returns:
            str: Path to saved model
        """
        try:
            self.logger.info("💾 Saving final model...")

            if not self.model:
                raise ModelTrainingError("No model to save")

            # Save model
            if not self.config.saving.save_model:
                self.logger.info("Model saving is disabled in config")
                return ""

            final_model_dir = self.config.output_dir / "final_model"
            final_model_dir.mkdir(exist_ok=True)

            self.model.save_pretrained(str(final_model_dir))
            self.logger.info(f"✅ Model saved to: {final_model_dir}")

            # Save tokenizer
            if self.config.saving.save_tokenizer and self.tokenizer:
                self.tokenizer.save_pretrained(str(final_model_dir))
                self.logger.info(f"✅ Tokenizer saved to: {final_model_dir}")

            # Save metadata
            if self.config.saving.save_metadata:
                metadata = {
                    "model_name": self.config.model_name_or_path,
                    "training_config": {
                        "num_epochs": self.config.num_train_epochs,
                        "batch_size": self.config.per_device_train_batch_size,
                        "learning_rate": self.config.learning_rate,
                        "max_length": self.config.max_length,
                        "use_peft": self.config.use_peft,
                    },
                    "training_state": self.training_state,
                    "device_used": str(self.device),
                    "saved_at": datetime.now().isoformat(),
                }

                metadata_path = final_model_dir / "training_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                self.logger.info(f"✅ Metadata saved to: {metadata_path}")

            return str(final_model_dir)

        except Exception as e:
            raise ModelTrainingError(f"Failed to save model: {str(e)}") from e
