"""
Model Evaluation Component for ParaDetect
Comprehensive evaluation with metrics, visualizations, and calibration analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

# Import calibration_curve from the correct module
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Fallback for very old versions
    from sklearn.metrics import calibration_curve

from sklearn.calibration import CalibratedClassifierCV
import torch
import torch.nn.functional as f
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import Dataset, load_from_disk
import warnings

warnings.filterwarnings("ignore")

from para_detect.entities.model_evaluation_config import ModelEvaluationConfig
from para_detect.core.exceptions import ModelEvaluationError, DeviceError
from para_detect.constants import DEVICE_PRIORITY, REVERSE_LABEL_MAPPING
from para_detect import get_logger
from para_detect.utils.s3_manager import S3Manager


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics computation and visualization.

    Features:
    - Standard classification metrics (accuracy, precision, recall, F1, AUC)
    - ROC and Precision-Recall curves
    - Confusion matrix with class-wise analysis
    - Calibration analysis and reliability diagrams
    - Per-class performance breakdown
    - Prediction confidence analysis
    """

    def __init__(
        self,
        config: ModelEvaluationConfig,
        run_id: str,
        s3_manager: Optional[S3Manager] = None,
    ):
        """
        Initialize model evaluator.

        Args:
            config: Evaluation configuration
            run_id: Unique run identifier
            s3_manager: Optional S3 manager for remote storage
        """
        self.config = config
        self.run_id = run_id
        self.logger = get_logger(self.__class__.__name__)

        # Initialize device
        self.device = self._detect_device()
        self.logger.info(f"🔧 Using device: {self.device}")

        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.classifier = None

        # Keep S3 manager for S3 operations
        self.s3_manager = s3_manager
        self.s3_enabled = s3_manager is not None

        # Results storage
        self.evaluation_results = {}
        self.predictions = None
        self.true_labels = None
        self.probabilities = None

    def _detect_device(self) -> torch.device:
        """Detect optimal device for evaluation."""
        try:
            if self.config.device_preference:
                device = torch.device(self.config.device_preference)
                if device.type == "cuda" and not torch.cuda.is_available():
                    self.logger.warning(
                        "CUDA requested but not available, falling back to CPU"
                    )
                    return torch.device("cpu")
                elif device.type == "mps" and not torch.backends.mps.is_available():
                    self.logger.warning(
                        "MPS requested but not available, falling back to CPU"
                    )
                    return torch.device("cpu")
                return device

            # Auto-detection
            for device_type in DEVICE_PRIORITY:
                if device_type == "cuda" and torch.cuda.is_available():
                    return torch.device("cuda")
                elif device_type == "mps" and torch.backends.mps.is_available():
                    return torch.device("mps")
                elif device_type == "cpu":
                    return torch.device("cpu")

            return torch.device("cpu")

        except Exception as e:
            raise DeviceError(f"Failed to detect device: {str(e)}") from e

    def load_model(self, model_path: str) -> None:
        """
        Load model and tokenizer for evaluation.

        Args:
            model_path: Path to saved model directory
        """
        try:
            self.logger.info(f"📥 Loading model from: {model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, torch_dtype=torch.float32, device_map=None
            )
            self.model.to(self.device)
            self.model.eval()

            # Create pipeline for easier inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True,
                truncation=True,
                max_length=self.config.max_length,
            )

            self.logger.info("✅ Model loaded successfully")

        except Exception as e:
            raise ModelEvaluationError(f"Failed to load model: {str(e)}") from e

    def evaluate(
        self, dataset: Union[Dataset, str, pd.DataFrame, Path]
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            dataset: Test dataset (Dataset, Parquet path, DataFrame, or path to tokenized test dataset)

        Returns:
            Dict: Comprehensive evaluation results
        """
        try:
            self.logger.info("📊 Starting model evaluation...")

            if not self.model or not self.tokenizer:
                raise ModelEvaluationError("Model not loaded. Call load_model() first.")

            # Handle tokenized dataset path (from training pipeline)
            if isinstance(dataset, (str, Path)) and Path(dataset).is_dir():
                # Check if this is a tokenized dataset directory
                if (Path(dataset) / "dataset_info.json").exists():
                    self.logger.info(
                        f"📂 Loading tokenized test dataset from: {dataset}"
                    )
                    test_dataset = load_from_disk(str(dataset))

                    # Extract texts and labels from tokenized dataset
                    texts = self.tokenizer.batch_decode(
                        test_dataset["input_ids"], skip_special_tokens=True
                    )
                    labels = test_dataset["labels"]

                    self.logger.info(f"📈 Evaluating on {len(texts):,} test samples...")

                    # Generate predictions using the model directly (more efficient for tokenized data)
                    predictions, probabilities = (
                        self._generate_predictions_from_tokenized(test_dataset)
                    )
                else:
                    # Regular file/directory, use existing logic
                    texts, labels = self._prepare_evaluation_data(dataset)
                    self.logger.info(f"📈 Evaluating on {len(texts):,} samples...")
                    predictions, probabilities = self._generate_predictions(texts)
            else:
                # Use existing logic for other input types
                texts, labels = self._prepare_evaluation_data(dataset)
                self.logger.info(f"📈 Evaluating on {len(texts):,} samples...")
                predictions, probabilities = self._generate_predictions(texts)

            # Normalize probs to 1D positive class
            probabilities = self._to_binary_probs(probabilities)

            # Store for analysis
            self.predictions = np.asarray(predictions)
            self.true_labels = np.asarray(labels)
            self.probabilities = probabilities

            # Compute metrics
            metrics = self.compute_metrics(
                self.predictions, self.true_labels, self.probabilities
            )

            # Visualizations
            if any(
                [
                    self.config.save_confusion_matrix,
                    self.config.save_roc_curve,
                    self.config.save_precision_recall_curve,
                ]
            ):
                self._generate_visualizations(
                    self.predictions, self.true_labels, self.probabilities
                )

            # Calibration analysis
            if (
                self.config.perform_calibration_analysis
                and self.probabilities is not None
            ):
                _ = self._analyze_calibration(self.probabilities, self.true_labels)

            # Per-class analysis
            if self.config.compute_per_class_metrics:
                _ = self._compute_per_class_metrics(self.predictions, self.true_labels)

            # Prepare final results (include preds/labels/probs for validator)
            self.evaluation_results = {
                "metrics": metrics,
                "predictions": self.predictions.tolist(),
                "true_labels": self.true_labels.tolist(),
                "probabilities": (
                    self.probabilities.tolist()
                    if self.probabilities is not None
                    else None
                ),
                "evaluation_config": {
                    "num_samples": int(len(self.true_labels)),
                    "device_used": str(self.device),
                    "max_length": self.config.max_length,
                    "batch_size": self.config.eval_batch_size,
                },
                "timestamp": datetime.now().isoformat(),
                "model_performance_summary": self._create_performance_summary(metrics),
            }

            self._save_evaluation_results()

            if self.s3_enabled:
                self._upload_evaluation_results_to_s3(self.evaluation_results)

            self.logger.info("✅ Evaluation completed successfully!")
            return self.evaluation_results

        except Exception as e:
            raise ModelEvaluationError(f"Evaluation failed: {str(e)}") from e

    def _generate_predictions_from_tokenized(
        self, dataset: Dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions from already tokenized dataset efficiently."""
        try:
            self.model.eval()
            preds = []
            probs = []
            batch_size = self.config.eval_batch_size

            with torch.no_grad():
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i : i + batch_size]

                    # Prepare batch tensors
                    input_ids = torch.tensor(batch["input_ids"], device=self.device)
                    attention_mask = torch.tensor(
                        batch["attention_mask"], device=self.device
                    )

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    logits = outputs.logits

                    # Get predictions and probabilities
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    p = f.softmax(logits, dim=1).cpu().numpy()

                    preds.extend(batch_preds)
                    probs.extend(p)

            return np.array(preds), np.array(probs)

        except Exception as e:
            raise ModelEvaluationError(
                f"Failed to generate predictions from tokenized data: {str(e)}"
            ) from e

    def _prepare_evaluation_data(
        self, dataset: Union[Dataset, str, pd.DataFrame]
    ) -> Tuple[List[str], List[int]]:
        """Prepare evaluation data from various input formats."""
        try:
            if isinstance(dataset, Dataset):
                texts = dataset[self.config.text_column]
                labels = dataset[self.config.label_column]
            elif isinstance(dataset, str):
                # Load from Parquet file
                df = pd.read_parquet(dataset)
                texts = df[self.config.text_column].tolist()
                labels = df[self.config.label_column].tolist()
            elif isinstance(dataset, pd.DataFrame):
                texts = dataset[self.config.text_column].tolist()
                labels = dataset[self.config.label_column].tolist()
            else:
                raise ModelEvaluationError(f"Unsupported dataset type: {type(dataset)}")

            # Validate data
            if len(texts) != len(labels):
                raise ModelEvaluationError("Mismatch between texts and labels length")

            if len(texts) == 0:
                raise ModelEvaluationError("Empty dataset provided")

            return texts, labels

        except Exception as e:
            raise ModelEvaluationError(
                f"Failed to prepare evaluation data: {str(e)}"
            ) from e

    def _generate_predictions(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for raw texts using the model directly."""
        try:
            self.model.eval()
            preds = []
            probs = []
            bs = self.config.eval_batch_size

            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    batch_texts = texts[i : i + bs]

                    # Tokenize batch
                    encoded = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        max_length=self.config.max_length,
                        return_tensors="pt",
                    )

                    # Move to device
                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded["attention_mask"].to(self.device)

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    logits = outputs.logits

                    # Get predictions and probabilities
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                    p = f.softmax(logits, dim=1).cpu().numpy()

                    preds.extend(batch_preds)
                    probs.extend(p)

            return np.array(preds), np.array(probs)

        except Exception as e:
            raise ModelEvaluationError(
                f"Failed to generate predictions: {str(e)}"
            ) from e

    def _to_binary_probs(
        self, probabilities: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Ensure 1D positive-class probabilities for binary tasks."""
        if probabilities is None:
            return None
        probs = np.asarray(probabilities)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        if probs.ndim > 1:
            return probs.reshape(-1)
        return probs

    def compute_metrics(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            predictions: Model predictions
            references: True labels
            probabilities: Prediction probabilities

        Returns:
            Dict: Computed metrics
        """
        try:
            metrics = {}

            # Basic classification metrics
            metrics["accuracy"] = accuracy_score(references, predictions)

            # Precision, recall, F1 (macro and weighted)
            precision_macro, recall_macro, f1_macro, _ = (
                precision_recall_fscore_support(
                    references, predictions, average="macro", zero_division=0
                )
            )
            precision_weighted, recall_weighted, f1_weighted, _ = (
                precision_recall_fscore_support(
                    references, predictions, average="weighted", zero_division=0
                )
            )

            metrics.update(
                {
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "f1_macro": f1_macro,
                    "precision_weighted": precision_weighted,
                    "recall_weighted": recall_weighted,
                    "f1_weighted": f1_weighted,
                    "f1": f1_weighted,  # Main F1 score for compatibility
                }
            )

            # ROC AUC and PR AUC (if probabilities available)
            if probabilities is not None:
                try:
                    metrics["roc_auc"] = roc_auc_score(references, probabilities)
                    metrics["precision_recall_auc"] = average_precision_score(
                        references, probabilities
                    )
                except ValueError as e:
                    self.logger.warning(f"Could not compute AUC metrics: {str(e)}")
                    metrics["roc_auc"] = 0.0
                    metrics["precision_recall_auc"] = 0.0

            # Class distribution
            unique, counts = np.unique(references, return_counts=True)
            class_distribution = dict(zip(unique.tolist(), counts.tolist()))
            metrics["class_distribution"] = class_distribution

            # Prediction distribution
            unique_pred, counts_pred = np.unique(predictions, return_counts=True)
            pred_distribution = dict(zip(unique_pred.tolist(), counts_pred.tolist()))
            metrics["prediction_distribution"] = pred_distribution

            return metrics

        except Exception as e:
            raise ModelEvaluationError(f"Failed to compute metrics: {str(e)}") from e

    def _compute_per_class_metrics(
        self, predictions: np.ndarray, references: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class performance metrics."""
        try:
            per_class_metrics = {}

            # Get per-class precision, recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                references, predictions, average=None, zero_division=0
            )

            for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
                class_name = REVERSE_LABEL_MAPPING.get(i, f"class_{i}")
                per_class_metrics[class_name] = {
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "support": int(s),
                }

            return per_class_metrics

        except Exception as e:
            self.logger.warning(f"Failed to compute per-class metrics: {str(e)}")
            return {}

    def _analyze_calibration(
        self, probabilities: np.ndarray, references: np.ndarray
    ) -> Dict[str, float]:
        """Analyze model calibration."""
        try:
            calibration_metrics = {}

            # Brier score
            calibration_metrics["brier_score"] = brier_score_loss(
                references, probabilities
            )

            # Calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                references, probabilities, n_bins=self.config.calibration_bins
            )

            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = references[in_bin].mean()
                    avg_confidence_in_bin = probabilities[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            calibration_metrics["expected_calibration_error"] = ece

            # Save calibration plot
            if self.config.save_roc_curve:  # Reuse this flag for calibration plots
                self._plot_calibration_curve(
                    fraction_of_positives, mean_predicted_value
                )

            return calibration_metrics

        except Exception as e:
            self.logger.warning(f"Failed to analyze calibration: {str(e)}")
            return {}

    def _generate_visualizations(
        self,
        predictions: np.ndarray,
        references: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> None:
        """Generate evaluation visualizations."""
        try:
            # Set style
            plt.style.use("default")
            sns.set_palette("husl")

            # Confusion Matrix
            if self.config.save_confusion_matrix:
                self._plot_confusion_matrix(predictions, references)

            # ROC Curve
            if self.config.save_roc_curve and probabilities is not None:
                self._plot_roc_curve(references, probabilities)

            # Precision-Recall Curve
            if self.config.save_precision_recall_curve and probabilities is not None:
                self._plot_precision_recall_curve(references, probabilities)

        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {str(e)}")

    def _plot_confusion_matrix(
        self, predictions: np.ndarray, references: np.ndarray
    ) -> None:
        """Plot and save confusion matrix."""
        try:
            plt.figure(figsize=(8, 6))

            cm = confusion_matrix(references, predictions)
            labels = [
                REVERSE_LABEL_MAPPING.get(i, f"Class {i}") for i in range(len(cm))
            ]

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Add accuracy text
            accuracy = accuracy_score(references, predictions)
            plt.text(
                0.5,
                -0.1,
                f"Accuracy: {accuracy:.3f}",
                transform=plt.gca().transAxes,
                ha="center",
            )

            plt.tight_layout()

            save_path = self.config.evaluation_output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"📊 Confusion matrix saved: {save_path}")

        except Exception as e:
            self.logger.warning(f"Failed to plot confusion matrix: {str(e)}")

    def _plot_roc_curve(
        self, references: np.ndarray, probabilities: np.ndarray
    ) -> None:
        """Plot and save ROC curve."""
        try:
            plt.figure(figsize=(8, 6))

            fpr, tpr, _ = roc_curve(references, probabilities)
            auc_score = roc_auc_score(references, probabilities)

            plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
            plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            save_path = self.config.evaluation_output_dir / "roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"📈 ROC curve saved: {save_path}")

        except Exception as e:
            self.logger.warning(f"Failed to plot ROC curve: {str(e)}")

    def _plot_precision_recall_curve(
        self, references: np.ndarray, probabilities: np.ndarray
    ) -> None:
        """Plot and save Precision-Recall curve."""
        try:
            plt.figure(figsize=(8, 6))

            precision, recall, _ = precision_recall_curve(references, probabilities)
            avg_precision = average_precision_score(references, probabilities)

            plt.plot(
                recall,
                precision,
                linewidth=2,
                label=f"PR Curve (AP = {avg_precision:.3f})",
            )

            # Baseline (proportion of positive class)
            baseline = references.mean()
            plt.axhline(
                y=baseline,
                color="k",
                linestyle="--",
                linewidth=1,
                label=f"Baseline (AP = {baseline:.3f})",
            )

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            save_path = self.config.evaluation_output_dir / "precision_recall_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"📈 Precision-Recall curve saved: {save_path}")

        except Exception as e:
            self.logger.warning(f"Failed to plot Precision-Recall curve: {str(e)}")

    def _plot_calibration_curve(
        self, fraction_of_positives: np.ndarray, mean_predicted_value: np.ndarray
    ) -> None:
        """Plot and save calibration curve."""
        try:
            plt.figure(figsize=(8, 6))

            plt.plot(
                mean_predicted_value,
                fraction_of_positives,
                "s-",
                linewidth=2,
                label="Model",
            )
            plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly Calibrated")

            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title("Calibration Curve (Reliability Diagram)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            save_path = self.config.evaluation_output_dir / "calibration_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"📊 Calibration curve saved: {save_path}")

        except Exception as e:
            self.logger.warning(f"Failed to plot calibration curve: {str(e)}")

    def _save_classification_report(
        self, predictions: np.ndarray, references: np.ndarray
    ) -> None:
        """Save detailed classification report."""
        try:
            target_names = [
                REVERSE_LABEL_MAPPING.get(i, f"Class {i}") for i in range(2)
            ]
            report = classification_report(
                references, predictions, target_names=target_names
            )

            report_path = (
                self.config.evaluation_output_dir / "classification_report.txt"
            )
            with open(report_path, "w") as f:
                f.write("Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)

            self.logger.info(f"📋 Classification report saved: {report_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save classification report: {str(e)}")

    def _create_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create human-readable performance summary."""
        summary = {}

        # Overall performance
        accuracy = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)

        if accuracy >= 0.9 and f1 >= 0.9:
            summary["overall_performance"] = "Excellent"
        elif accuracy >= 0.85 and f1 >= 0.85:
            summary["overall_performance"] = "Good"
        elif accuracy >= 0.75 and f1 >= 0.75:
            summary["overall_performance"] = "Fair"
        else:
            summary["overall_performance"] = "Poor"

        # Key metrics summary
        summary["key_metrics"] = f"Accuracy: {accuracy:.3f}, F1: {f1:.3f}"

        if "roc_auc" in metrics:
            summary["key_metrics"] += f", AUC: {metrics['roc_auc']:.3f}"

        # Calibration summary
        if "brier_score" in metrics:
            brier = metrics["brier_score"]
            if brier <= 0.1:
                summary["calibration"] = "Well calibrated"
            elif brier <= 0.2:
                summary["calibration"] = "Reasonably calibrated"
            else:
                summary["calibration"] = "Poorly calibrated"

        return summary

    def _save_evaluation_results(self) -> None:
        """Save complete evaluation results."""
        try:
            results_path = self.config.evaluation_output_dir / "evaluation_results.json"

            with open(results_path, "w") as f:
                json.dump(self.evaluation_results, f, indent=2)

            self.logger.info(f"💾 Evaluation results saved: {results_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save evaluation results: {str(e)}")

    def _upload_evaluation_results_to_s3(self, results: Dict[str, Any]) -> bool:
        """Upload evaluation results to S3."""
        try:
            if not hasattr(self.config, "evaluation_output_dir"):
                return False

            metadata = {
                "accuracy": str(results.get("metrics", {}).get("accuracy", "")),
                "f1_score": str(results.get("metrics", {}).get("f1", "")),
                "model_path": str(getattr(self.config, "model_path", "")),
            }

            # Use S3Manager's component results upload
            upload_results = self.s3_manager.upload_component_results(
                results_dir=self.config.evaluation_output_dir,
                component_name="evaluations",
                run_id=self.run_id,
                metadata=metadata,
            )

            if upload_results["success"]:
                self.logger.info(
                    f"Uploaded evaluation results to S3: {upload_results['files_uploaded']} files"
                )
                return True
            else:
                self.logger.warning("Failed to upload evaluation results to S3")
                return False

        except Exception as e:
            self.logger.warning(f"Error uploading evaluation results to S3: {e}")
            return False
