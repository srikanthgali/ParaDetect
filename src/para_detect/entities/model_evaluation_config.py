from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for model evaluation component"""

    # Evaluation metrics
    metrics: List[str] = None
    save_best_by: str = "f1"

    # Evaluation parameters
    eval_batch_size: int = 32
    max_length: int = 512

    # Output configuration
    evaluation_output_dir: Path = Path("artifacts/evaluation")
    save_confusion_matrix: bool = True
    save_classification_report: bool = True
    save_roc_curve: bool = True
    save_precision_recall_curve: bool = True

    # Calibration analysis
    perform_calibration_analysis: bool = True
    calibration_bins: int = 10

    # Per-class analysis
    compute_per_class_metrics: bool = True

    # Device configuration
    device_preference: Optional[str] = None

    # Data configuration
    text_column: str = "text"
    label_column: str = "generated"

    def __post_init__(self):
        """Validate configuration and set defaults"""
        if self.metrics is None:
            object.__setattr__(
                self,
                "metrics",
                [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "precision_recall_auc",
                ],
            )

        if self.save_best_by not in self.metrics:
            raise ValueError(
                f"save_best_by '{self.save_best_by}' must be in metrics list"
            )

        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")

        # Ensure output directory exists
        self.evaluation_output_dir.mkdir(parents=True, exist_ok=True)
