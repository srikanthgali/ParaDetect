from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class ModelValidationConfig:
    """Configuration for model validation and post-training checks"""

    # Threshold validation
    min_accuracy: float = 0.85
    min_f1: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80
    min_auc: float = 0.85

    # Calibration validation
    max_brier_score: float = 0.25
    max_ece: float = 0.1  # Expected Calibration Error
    calibration_bins: int = 10

    # Class-specific validation
    min_per_class_f1: float = 0.75
    max_class_imbalance_ratio: float = 3.0

    # Fairness and bias checks
    perform_fairness_checks: bool = True
    max_demographic_parity_diff: float = 0.1
    max_equalized_odds_diff: float = 0.1

    # Distribution checks
    check_prediction_distribution: bool = True
    max_prediction_skew: float = 0.8  # Maximum ratio of one class

    # Allowed label values
    allowed_label_values: List[int] = None

    # Output configuration
    validation_output_dir: Path = Path("artifacts/validation")
    save_validation_report: bool = True

    def __post_init__(self):
        """Validate configuration and set defaults"""
        if self.allowed_label_values is None:
            object.__setattr__(self, "allowed_label_values", [0, 1])

        # Validate threshold ranges
        thresholds = [
            self.min_accuracy,
            self.min_f1,
            self.min_precision,
            self.min_recall,
            self.min_auc,
        ]

        for threshold in thresholds:
            if not 0 <= threshold <= 1:
                raise ValueError("All metric thresholds must be between 0 and 1")

        if self.max_brier_score < 0:
            raise ValueError("max_brier_score must be non-negative")

        if not 0 <= self.max_ece <= 1:
            raise ValueError("max_ece must be between 0 and 1")

        if self.max_class_imbalance_ratio <= 1:
            raise ValueError("max_class_imbalance_ratio must be greater than 1")

        # Ensure output directory exists
        self.validation_output_dir.mkdir(parents=True, exist_ok=True)
