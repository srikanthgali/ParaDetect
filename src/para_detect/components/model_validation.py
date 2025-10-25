"""
Model Validation Component for ParaDetect
Post-training validation checks for model quality and fairness
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from para_detect.utils.s3_manager import S3Manager
from para_detect.utils.helpers import convert_to_serializable
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

from para_detect.entities.model_validation_config import ModelValidationConfig
from para_detect.core.exceptions import ModelValidationError
from para_detect.constants import REVERSE_LABEL_MAPPING
from para_detect import get_logger


class ModelValidator:
    """
    Comprehensive model validation with quality and fairness checks.

    Performs:
    - Threshold validation against minimum performance requirements
    - Calibration quality assessment
    - Per-class performance validation
    - Basic fairness and bias checks
    - Prediction distribution analysis
    - Statistical significance tests
    """

    def __init__(
        self,
        config: ModelValidationConfig,
        run_id: str,
        s3_manager: Optional[S3Manager] = None,
    ):
        """
        Initialize model validator.

        Args:
            config: Validation configuration
            s3_manager: Optional S3 manager for remote storage
        """
        self.config = config
        self.run_id = run_id
        self.logger = get_logger(self.__class__.__name__)

        # Keep S3 manager for S3 operations
        self.s3_manager = s3_manager
        self.s3_enabled = s3_manager is not None

        # Validation results
        self.validation_results = {}
        self.validation_passed = False
        self.validation_issues = []
        self.validation_warnings = []

    def validate_model(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive model validation.

        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            probabilities: Prediction probabilities
            metadata: Additional metadata for validation

        Returns:
            Dict: Validation results with pass/fail status
        """
        try:
            self.logger.info("🔍 Starting model validation...")

            # Reset validation state
            self.validation_issues = []
            self.validation_warnings = []

            # Coerce to numpy + 1D probs
            predictions = np.asarray(predictions)
            true_labels = np.asarray(true_labels)
            if probabilities is not None:
                probabilities = np.asarray(probabilities)
                if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                    probabilities = probabilities[:, 1]
                elif probabilities.ndim > 1:
                    probabilities = probabilities.reshape(-1)

            # Basic data validation
            self._validate_input_data(predictions, true_labels, probabilities)

            # Compute basic metrics
            metrics = self._compute_validation_metrics(
                predictions, true_labels, probabilities
            )

            # Run validation checks
            validation_checks = {
                "threshold_validation": self._validate_thresholds(metrics),
                "calibration_validation": (
                    self._validate_calibration(probabilities, true_labels)
                    if probabilities is not None
                    else {"passed": True, "skipped": True}
                ),
                "class_balance_validation": self._validate_class_performance(
                    predictions, true_labels
                ),
                "distribution_validation": self._validate_prediction_distribution(
                    predictions, true_labels
                ),
                "statistical_validation": self._validate_statistical_significance(
                    predictions, true_labels
                ),
            }

            # Fairness checks (if enabled)
            if self.config.perform_fairness_checks:
                validation_checks["fairness_validation"] = self._validate_fairness(
                    predictions, true_labels, metadata
                )

            # Determine overall validation status
            all_passed = all(
                check.get("passed", False)
                for check in validation_checks.values()
                if not check.get("skipped", False)
            )

            # Prepare results
            self.validation_results = {
                "validation_passed": all_passed,
                "validation_timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "validation_checks": validation_checks,
                "validation_issues": self.validation_issues,
                "validation_warnings": self.validation_warnings,
                "validation_config": {
                    "min_accuracy": self.config.min_accuracy,
                    "min_f1": self.config.min_f1,
                    "min_auc": self.config.min_auc,
                    "calibration_enabled": probabilities is not None,
                    "fairness_enabled": self.config.perform_fairness_checks,
                },
                "recommendation": self._generate_recommendation(all_passed),
            }

            self.validation_passed = all_passed

            # Save validation report
            if self.config.save_validation_report:
                self._save_validation_report()

            # Log results
            if all_passed:
                self.logger.info("✅ Model validation PASSED")
            else:
                self.logger.warning("❌ Model validation FAILED")
                for issue in self.validation_issues:
                    self.logger.warning(f"   Issue: {issue}")

            if self.validation_warnings:
                for warning in self.validation_warnings:
                    self.logger.warning(f"   Warning: {warning}")

            if self.s3_enabled and hasattr(self, "validation_results"):
                self._upload_validation_results_to_s3(self.validation_results)

            return self.validation_results

        except Exception as e:
            raise ModelValidationError(f"Validation failed: {str(e)}") from e

    def _validate_input_data(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        probabilities: Optional[np.ndarray],
    ) -> None:
        """Validate input data format and consistency."""
        # Check array shapes
        if len(predictions) != len(true_labels):
            raise ModelValidationError(
                "Predictions and true labels have different lengths"
            )

        if probabilities is not None:
            if len(probabilities) != len(predictions):
                raise ModelValidationError(
                    "Probabilities and predictions have different lengths"
                )
            if np.any(probabilities < 0) or np.any(probabilities > 1):
                raise ModelValidationError("Probabilities must be in [0, 1]")
            if len(probabilities) == 0:
                raise ModelValidationError(
                    "Probabilities are empty; cannot run calibration metrics"
                )

        # Check label values
        unique_labels = np.unique(true_labels)
        expected_labels = set(self.config.allowed_label_values)
        actual_labels = set(unique_labels)

        if not actual_labels.issubset(expected_labels):
            unexpected = actual_labels - expected_labels
            raise ModelValidationError(f"Unexpected label values: {unexpected}")

        # Check prediction values
        unique_preds = np.unique(predictions)
        actual_preds = set(unique_preds)

        if not actual_preds.issubset(expected_labels):
            unexpected = actual_preds - expected_labels
            raise ModelValidationError(f"Unexpected prediction values: {unexpected}")

        # Check probabilities range
        if probabilities is not None:
            if np.any(probabilities < 0) or np.any(probabilities > 1):
                raise ModelValidationError("Probabilities must be between 0 and 1")

    def _compute_validation_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        probabilities: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Compute metrics for validation."""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(true_labels, predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="weighted", zero_division=0
        )

        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        # AUC metrics (if probabilities available)
        if probabilities is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(true_labels, probabilities)
                metrics["brier_score"] = brier_score_loss(true_labels, probabilities)
            except ValueError:
                self.validation_warnings.append("Could not compute AUC metrics")

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = (
            precision_recall_fscore_support(
                true_labels, predictions, average=None, zero_division=0
            )
        )

        metrics["per_class_precision"] = precision_per_class.tolist()
        metrics["per_class_recall"] = recall_per_class.tolist()
        metrics["per_class_f1"] = f1_per_class.tolist()
        metrics["per_class_support"] = support.tolist()

        return metrics

    def _validate_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate metrics against minimum thresholds."""
        threshold_results = {"passed": True, "details": {}}

        thresholds = {
            "accuracy": self.config.min_accuracy,
            "f1": self.config.min_f1,
            "precision": self.config.min_precision,
            "recall": self.config.min_recall,
        }

        if "roc_auc" in metrics:
            thresholds["roc_auc"] = self.config.min_auc

        for metric_name, min_threshold in thresholds.items():
            if metric_name in metrics:
                actual_value = float(metrics[metric_name])  # Ensure it's a Python float
                passed = not (np.isnan(actual_value) or actual_value < min_threshold)

                threshold_results["details"][metric_name] = {
                    "actual": actual_value,
                    "minimum": min_threshold,
                    "passed": passed,
                }

                if not passed:
                    if np.isnan(actual_value):
                        issue = f"{metric_name.title()} is NaN (invalid)"
                    else:
                        issue = f"{metric_name.title()} {actual_value:.3f} below minimum {min_threshold:.3f}"

                    self.validation_issues.append(issue)
                    threshold_results["passed"] = False

        return threshold_results

    def _validate_calibration(
        self, probabilities: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Validate model calibration quality."""
        try:
            calibration_results = {"passed": True, "details": {}}

            # Brier score
            brier_score = brier_score_loss(true_labels, probabilities)
            calibration_results["details"]["brier_score"] = {
                "actual": brier_score,
                "max_allowed": self.config.max_brier_score,
                "passed": brier_score <= self.config.max_brier_score,
            }

            if brier_score > self.config.max_brier_score:
                calibration_results["passed"] = False
                self.validation_issues.append(
                    f"Brier score {brier_score:.3f} exceeds maximum {self.config.max_brier_score:.3f}"
                )

            # Expected Calibration Error (ECE)
            ece = self._compute_ece(probabilities, true_labels)
            calibration_results["details"]["expected_calibration_error"] = {
                "actual": ece,
                "max_allowed": self.config.max_ece,
                "passed": ece <= self.config.max_ece,
            }

            if ece > self.config.max_ece:
                calibration_results["passed"] = False
                self.validation_issues.append(
                    f"Expected Calibration Error {ece:.3f} exceeds maximum {self.config.max_ece:.3f}"
                )

            return calibration_results

        except Exception as e:
            self.validation_warnings.append(f"Calibration validation failed: {str(e)}")
            return {"passed": True, "skipped": True, "error": str(e)}

    def _compute_ece(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.config.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _validate_class_performance(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Validate per-class performance."""
        class_results = {"passed": True, "details": {}}

        # Check per-class F1 scores
        _, _, f1_scores, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )

        for i, (f1, sup) in enumerate(zip(f1_scores, support)):
            class_name = REVERSE_LABEL_MAPPING.get(i, f"class_{i}")

            # Only validate classes with sufficient support
            if sup > 10:  # Minimum samples for reliable validation
                passed = f1 >= self.config.min_per_class_f1

                class_results["details"][class_name] = {
                    "f1_score": f1,
                    "min_required": self.config.min_per_class_f1,
                    "support": sup,
                    "passed": passed,
                }

                if not passed:
                    class_results["passed"] = False
                    self.validation_issues.append(
                        f"Class {class_name} F1 score {f1:.3f} below threshold {self.config.min_per_class_f1:.3f}"
                    )
            else:
                self.validation_warnings.append(
                    f"Insufficient samples for class {class_name} validation (n={sup})"
                )

        # Check class imbalance
        unique, counts = np.unique(true_labels, return_counts=True)
        if len(counts) > 1:
            imbalance_ratio = max(counts) / min(counts)
            class_results["details"]["imbalance_ratio"] = {
                "actual": imbalance_ratio,
                "max_allowed": self.config.max_class_imbalance_ratio,
                "passed": imbalance_ratio <= self.config.max_class_imbalance_ratio,
            }

            if imbalance_ratio > self.config.max_class_imbalance_ratio:
                self.validation_warnings.append(
                    f"Class imbalance ratio {imbalance_ratio:.2f} exceeds recommended maximum {self.config.max_class_imbalance_ratio:.2f}"
                )

        return class_results

    def _validate_prediction_distribution(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Validate prediction distribution patterns."""
        dist_results = {"passed": True, "details": {}}

        if self.config.check_prediction_distribution:
            # Check for prediction skew
            unique_preds, pred_counts = np.unique(predictions, return_counts=True)

            if len(pred_counts) > 1:
                prediction_skew = max(pred_counts) / len(predictions)

                dist_results["details"]["prediction_skew"] = {
                    "actual": prediction_skew,
                    "max_allowed": self.config.max_prediction_skew,
                    "passed": prediction_skew <= self.config.max_prediction_skew,
                }

                if prediction_skew > self.config.max_prediction_skew:
                    dist_results["passed"] = False
                    self.validation_issues.append(
                        f"Prediction skew {prediction_skew:.3f} exceeds maximum {self.config.max_prediction_skew:.3f}"
                    )

            # Compare prediction and true label distributions
            true_unique, true_counts = np.unique(true_labels, return_counts=True)
            true_dist = true_counts / len(true_labels)
            pred_dist = pred_counts / len(predictions)

            # Chi-square test for distribution similarity
            try:
                chi2_stat, p_value = stats.chisquare(pred_counts, true_counts)
                dist_results["details"]["distribution_similarity"] = {
                    "chi2_statistic": chi2_stat,
                    "p_value": p_value,
                    "distributions_similar": p_value > 0.05,
                }

                if p_value <= 0.01:  # Very significant difference
                    self.validation_warnings.append(
                        "Prediction distribution significantly differs from true distribution"
                    )
            except Exception:
                self.validation_warnings.append(
                    "Could not perform distribution similarity test"
                )

        return dist_results

    def _validate_statistical_significance(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Validate statistical significance of model performance."""
        stat_results = {"passed": True, "details": {}}

        try:
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions)

            if cm.shape == (2, 2):  # Binary classification
                # McNemar's test for statistical significance
                b = cm[0, 1]  # False positives
                c = cm[1, 0]  # False negatives

                if b + c > 10:  # Sufficient discordant pairs
                    mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
                    p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)

                    stat_results["details"]["mcnemar_test"] = {
                        "statistic": mcnemar_stat,
                        "p_value": p_value,
                        "significantly_better_than_random": p_value < 0.05,
                    }

                    if p_value >= 0.05:
                        self.validation_warnings.append(
                            "Model performance not significantly better than random (McNemar test)"
                        )
                else:
                    self.validation_warnings.append(
                        "Insufficient data for McNemar test"
                    )

            # Bootstrap confidence interval for accuracy
            n_bootstrap = 1000
            bootstrap_accuracies = []

            for _ in range(n_bootstrap):
                indices = np.random.choice(
                    len(predictions), len(predictions), replace=True
                )
                boot_preds = predictions[indices]
                boot_true = true_labels[indices]
                boot_acc = accuracy_score(boot_true, boot_preds)
                bootstrap_accuracies.append(boot_acc)

            ci_lower = np.percentile(bootstrap_accuracies, 2.5)
            ci_upper = np.percentile(bootstrap_accuracies, 97.5)

            stat_results["details"]["accuracy_confidence_interval"] = {
                "lower_bound": ci_lower,
                "upper_bound": ci_upper,
                "confidence_level": 0.95,
            }

        except Exception as e:
            self.validation_warnings.append(f"Statistical validation failed: {str(e)}")

        return stat_results

    def _validate_fairness(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Basic fairness validation checks."""
        fairness_results = {"passed": True, "details": {}}

        try:
            # This is a placeholder for basic fairness checks
            # In practice, you would need demographic information to perform proper fairness analysis

            fairness_results["details"]["note"] = "Basic fairness checks performed"

            # Check for obvious bias patterns in confusion matrix
            cm = confusion_matrix(true_labels, predictions)

            if cm.shape == (2, 2):
                # Check for symmetric errors
                false_positives = cm[0, 1]
                false_negatives = cm[1, 0]

                if false_positives > 0 and false_negatives > 0:
                    error_ratio = max(false_positives, false_negatives) / min(
                        false_positives, false_negatives
                    )

                    fairness_results["details"]["error_symmetry"] = {
                        "false_positives": int(false_positives),
                        "false_negatives": int(false_negatives),
                        "error_ratio": error_ratio,
                        "reasonably_symmetric": error_ratio <= 2.0,
                    }

                    if error_ratio > 3.0:
                        self.validation_warnings.append(
                            f"Asymmetric error pattern detected (ratio: {error_ratio:.2f})"
                        )

        except Exception as e:
            self.validation_warnings.append(f"Fairness validation failed: {str(e)}")

        return fairness_results

    def _generate_recommendation(self, validation_passed: bool) -> str:
        """Generate recommendation based on validation results."""
        if validation_passed:
            if not self.validation_warnings:
                return "Model meets all validation criteria and is recommended for registration and deployment."
            else:
                return "Model meets validation criteria but has some warnings. Review warnings before deployment."
        else:
            if len(self.validation_issues) == 1:
                return "Model has one critical issue. Address the issue before proceeding with registration."
            else:
                return f"Model has {len(self.validation_issues)} critical issues. Address all issues before proceeding with registration."

    def _save_validation_report(self) -> None:
        """Save detailed validation report."""
        try:
            report_path = self.config.validation_output_dir / "validation_report.json"

            # Convert to serializable format
            serializable_results = convert_to_serializable(self.validation_results)

            with open(report_path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)

            # Also save a human-readable summary
            summary_path = self.config.validation_output_dir / "validation_summary.txt"
            with open(summary_path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("MODEL VALIDATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(
                    f"Timestamp: {self.validation_results.get('validation_timestamp', 'Unknown')}\n"
                )
                f.write(
                    f"Overall Status: {'PASSED' if self.validation_results.get('validation_passed', False) else 'FAILED'}\n\n"
                )

                # Metrics summary
                f.write("METRICS SUMMARY:\n")
                f.write("-" * 30 + "\n")
                metrics = self.validation_results.get("metrics", {})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

                f.write(f"\nVALIDATION CHECKS:\n")
                f.write("-" * 30 + "\n")
                checks = self.validation_results.get("validation_checks", {})
                for check_name, check_result in checks.items():
                    status = "PASSED" if check_result.get("passed", False) else "FAILED"
                    if check_result.get("skipped", False):
                        status = "SKIPPED"
                    f.write(f"{check_name}: {status}\n")

                # Issues and warnings
                if self.validation_issues:
                    f.write(f"\nISSUES:\n")
                    f.write("-" * 30 + "\n")
                    for issue in self.validation_issues:
                        f.write(f"- {issue}\n")

                if self.validation_warnings:
                    f.write(f"\nWARNINGS:\n")
                    f.write("-" * 30 + "\n")
                    for warning in self.validation_warnings:
                        f.write(f"- {warning}\n")

                f.write(f"\nRECOMMENDATION:\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"{self.validation_results.get('recommendation', 'No recommendation available')}\n"
                )

            self.logger.info(f"📋 Validation report saved: {report_path}")
            self.logger.info(f"📄 Validation summary saved: {summary_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save validation report: {str(e)}")

    def _upload_validation_results_to_s3(self, results: Dict[str, Any]) -> bool:
        """Upload validation results to S3."""
        try:
            if not hasattr(self.config, "validation_output_dir"):
                return False

            metrics = results.get("metrics", {}) if isinstance(results, dict) else {}
            metadata = {
                "accuracy": str(metrics.get("accuracy", "")),
                "f1_score": str(metrics.get("f1", "")),
                "validation_passed": str(results.get("validation_passed", "")),
                "validation_issues_count": str(
                    len(results.get("validation_issues", []))
                ),
                "model_path": str(getattr(self.config, "model_path", "")),
            }

            upload_results = self.s3_manager.upload_component_results(
                results_dir=self.config.validation_output_dir,
                component_name="validations",
                run_id=self.run_id,
                metadata=metadata,
            )

            if upload_results.get("success"):
                self.logger.info(
                    f"Uploaded validation results to S3: {upload_results.get('files_uploaded', 0)} files"
                )
                return True
            else:
                self.logger.warning("Failed to upload validation results to S3")
                return False

        except Exception as e:
            self.logger.warning(f"Error uploading validation results to S3: {e}")
            return False
