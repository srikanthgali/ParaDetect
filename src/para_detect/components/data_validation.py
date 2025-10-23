"""Data validation module for ParaDetect"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from para_detect.core.base_component import BaseComponent
from para_detect.core.config_manager import ConfigurationManager
from para_detect.core.exceptions import DataValidationError
from para_detect.entities.data_validation_config import DataValidationConfig
from para_detect.utils.s3_manager import S3Manager


class DataValidation(BaseComponent):
    """
    Data validation component for ParaDetect pipeline.

    Validates data quality, schema compliance, and data integrity
    before training or inference.
    """

    def __init__(
        self, config: DataValidationConfig, config_manager: ConfigurationManager = None
    ):
        """
        Initialize DataValidation component.

        Args:
            config: DataValidationConfig object containing validation parameters
            config_manager: ConfigurationManager for S3 configuration access
        """
        self.validation_config = config
        self.config_manager = config_manager or ConfigurationManager()

        # Pass the config as a dictionary to the base class for backward compatibility
        super().__init__(config.__dict__)

        # Create directories
        self.validation_config.validation_report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 manager
        self._initialize_s3_manager()

        self.validation_results = {}
        self.validation_passed = True

        self.logger.info("DataValidation component initialized")

    def _initialize_s3_manager(self):
        """Initialize S3 manager with automatic credential detection."""
        try:
            self.s3_manager = S3Manager(config_manager=self.config_manager)
            self.s3_enabled = True

            # Create bucket if it doesn't exist
            self.s3_manager.create_bucket_if_not_exists()

            self.logger.info("S3 integration enabled successfully")

            # Log credential info for debugging
            cred_info = self.s3_manager.get_credential_info()
            self.logger.info(f"AWS Account: {cred_info.get('account', 'Unknown')}")
            self.logger.info(f"Environment: {cred_info.get('environment', 'Unknown')}")

        except Exception as e:
            self.s3_manager = None
            self.s3_enabled = False
            self.logger.warning(f"S3 integration disabled: {e}")

    def run(self, data_path: str) -> Tuple[bool, str]:
        """
        Execute data validation process with S3 backup.

        Args:
            data_path: Path to the data file to validate

        Returns:
            Tuple[bool, str]: (validation_passed, report_path)

        Raises:
            DataValidationError: If validation process fails
        """
        try:
            self.logger.info(f"Starting data validation for: {data_path}")

            # Check if validation report exists in S3 cache first
            local_report_path = None
            if self.s3_enabled and self._should_use_s3_cache():
                local_report_path = self._try_download_from_s3_cache(data_path)

            if local_report_path and Path(local_report_path).exists():
                self.logger.info(
                    f"Using cached validation report from S3: {local_report_path}"
                )
                # Load validation results from cached report
                validation_passed = self._load_validation_results_from_report(
                    local_report_path
                )
                return validation_passed, local_report_path

            # Proceed with normal validation if no S3 cache
            # Load data
            df = self._load_data(data_path)

            # Run validation checks
            self._validate_schema(df)
            self._validate_data_quality(df)
            self._validate_text_content(df)
            self._validate_labels(df)
            self._validate_class_distribution(df)

            # Generate and save validation report
            report_path = self._generate_validation_report(data_path, df)

            # Upload validation report to S3
            if self.s3_enabled:
                self._upload_validation_report_to_s3(report_path, data_path)

            self.logger.info(
                f"Data validation completed. Validation passed: {self.validation_passed}"
            )
            self.logger.info(f"Validation report saved to: {report_path}")

            return self.validation_passed, report_path

        except Exception as e:
            error_msg = f"Data validation failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataValidationError(error_msg) from e

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from file."""
        try:
            df = pd.read_parquet(data_path)
            self.logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            raise DataValidationError(
                f"Failed to load data from {data_path}: {str(e)}"
            ) from e

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate data schema and column structure."""
        self.logger.info("Validating data schema...")

        schema_results = {}

        # Check for expected columns
        missing_columns = set(self.config.expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(self.config.expected_columns)

        schema_results["missing_columns"] = list(missing_columns)
        schema_results["extra_columns"] = list(extra_columns)
        schema_results["actual_columns"] = list(df.columns)

        # Check for required columns
        missing_required = set(self.config.required_columns) - set(df.columns)
        schema_results["missing_required_columns"] = list(missing_required)

        if missing_required:
            self.validation_passed = False
            self.logger.error(f"Missing required columns: {missing_required}")

        if missing_columns:
            self.logger.warning(f"Missing expected columns: {missing_columns}")

        if extra_columns:
            self.logger.info(f"Extra columns found: {extra_columns}")

        # Check data types
        dtypes_info = {}
        for col in df.columns:
            dtypes_info[col] = str(df[col].dtype)

        schema_results["column_dtypes"] = dtypes_info
        schema_results["validation_passed"] = len(missing_required) == 0

        self.validation_results["schema"] = schema_results
        self.logger.info(
            f"Schema validation completed. Passed: {schema_results['validation_passed']}"
        )

    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Validate overall data quality."""
        self.logger.info("Validating data quality...")

        quality_results = {}

        # Check for null values
        null_counts = df.isnull().sum().to_dict()
        null_percentages = (df.isnull().sum() / len(df)).to_dict()

        quality_results["null_counts"] = null_counts
        quality_results["null_percentages"] = null_percentages

        # Check if null percentage exceeds threshold
        high_null_columns = {
            col: pct
            for col, pct in null_percentages.items()
            if pct > self.config.max_null_percentage
        }

        if high_null_columns:
            self.validation_passed = False
            self.logger.error(f"Columns with high null percentage: {high_null_columns}")

        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)

        quality_results["duplicate_rows"] = int(duplicate_count)
        quality_results["duplicate_percentage"] = float(duplicate_percentage)

        # Check for empty rows
        empty_rows = (df == "").all(axis=1).sum()
        quality_results["empty_rows"] = int(empty_rows)

        # Overall data quality score
        quality_score = 1.0 - max(null_percentages.values()) - duplicate_percentage
        quality_results["quality_score"] = float(quality_score)

        quality_results["validation_passed"] = len(high_null_columns) == 0

        self.validation_results["data_quality"] = quality_results
        self.logger.info(
            f"Data quality validation completed. Quality score: {quality_score:.3f}"
        )

    def _validate_text_content(self, df: pd.DataFrame) -> None:
        """Validate text content quality."""
        self.logger.info("Validating text content...")

        text_results = {}

        if self.config.text_column not in df.columns:
            text_results["validation_passed"] = False
            text_results["error"] = f"Text column '{self.config.text_column}' not found"
            self.validation_results["text_content"] = text_results
            self.validation_passed = False
            return

        text_series = df[self.config.text_column].astype(str)

        # Calculate text length statistics
        text_lengths = text_series.str.len()

        text_results["length_stats"] = {
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
            "mean": float(text_lengths.mean()),
            "median": float(text_lengths.median()),
            "std": float(text_lengths.std()),
        }

        # Check text length constraints
        too_short = (text_lengths < self.config.min_text_length).sum()
        too_long = (text_lengths > self.config.max_text_length).sum()

        text_results["texts_too_short"] = int(too_short)
        text_results["texts_too_long"] = int(too_long)
        text_results["texts_within_range"] = int(len(df) - too_short - too_long)

        # Check for empty or whitespace-only texts
        empty_texts = (text_series.str.strip() == "").sum()
        text_results["empty_texts"] = int(empty_texts)

        # Check text diversity
        unique_texts = text_series.nunique()
        text_results["unique_texts"] = int(unique_texts)
        text_results["text_diversity_ratio"] = float(unique_texts / len(df))

        # Validation criteria
        length_validation_passed = (too_short == 0) and (too_long == 0)
        empty_validation_passed = empty_texts == 0

        text_results["validation_passed"] = (
            length_validation_passed and empty_validation_passed
        )

        if not length_validation_passed:
            self.validation_passed = False
            self.logger.error(
                f"Text length validation failed. Too short: {too_short}, Too long: {too_long}"
            )

        if not empty_validation_passed:
            self.validation_passed = False
            self.logger.error(f"Found {empty_texts} empty texts")

        self.validation_results["text_content"] = text_results
        self.logger.info(
            f"Text content validation completed. Passed: {text_results['validation_passed']}"
        )

    def _validate_labels(self, df: pd.DataFrame) -> None:
        """Validate label quality and distribution."""
        self.logger.info("Validating labels...")

        label_results = {}

        if self.config.label_column not in df.columns:
            label_results["validation_passed"] = False
            label_results["error"] = (
                f"Label column '{self.config.label_column}' not found"
            )
            self.validation_results["labels"] = label_results
            self.validation_passed = False
            return

        labels = df[self.config.label_column]

        # Get unique labels
        unique_labels = sorted(labels.unique())
        label_results["unique_labels"] = [
            int(x) if pd.notna(x) else None for x in unique_labels
        ]

        # Check for missing labels
        null_labels = labels.isnull().sum()
        label_results["null_labels"] = int(null_labels)

        # Check if labels match expected values
        expected_set = set(self.config.expected_labels)
        actual_set = set(x for x in unique_labels if pd.notna(x))

        unexpected_labels = actual_set - expected_set
        missing_labels = expected_set - actual_set

        label_results["unexpected_labels"] = list(unexpected_labels)
        label_results["missing_expected_labels"] = list(missing_labels)

        # Label distribution
        label_distribution = labels.value_counts().to_dict()
        label_results["label_distribution"] = {
            str(k): int(v) for k, v in label_distribution.items()
        }

        # Validation criteria
        labels_valid = len(unexpected_labels) == 0 and null_labels == 0

        label_results["validation_passed"] = labels_valid

        if not labels_valid:
            self.validation_passed = False
            if unexpected_labels:
                self.logger.error(f"Unexpected labels found: {unexpected_labels}")
            if null_labels > 0:
                self.logger.error(f"Found {null_labels} null labels")

        self.validation_results["labels"] = label_results
        self.logger.info(f"Label validation completed. Passed: {labels_valid}")

    def _validate_class_distribution(self, df: pd.DataFrame) -> None:
        """Validate class distribution and balance."""
        self.logger.info("Validating class distribution...")

        class_results = {}

        if self.config.label_column not in df.columns:
            class_results["validation_passed"] = False
            class_results["error"] = (
                f"Label column '{self.config.label_column}' not found"
            )
            self.validation_results["class_distribution"] = class_results
            return

        labels = df[self.config.label_column]
        label_counts = labels.value_counts()

        class_results["class_counts"] = label_counts.to_dict()
        class_results["total_samples"] = int(len(df))

        # Calculate class percentages
        class_percentages = (label_counts / len(df) * 100).to_dict()
        class_results["class_percentages"] = {
            str(k): float(v) for k, v in class_percentages.items()
        }

        # Check minimum samples per class
        min_samples = label_counts.min()
        class_results["min_samples_per_class"] = int(min_samples)

        insufficient_classes = label_counts[
            label_counts < self.config.min_samples_per_class
        ]
        class_results["insufficient_classes"] = insufficient_classes.to_dict()

        # Calculate class balance ratio
        if len(label_counts) >= 2:
            balance_ratio = min(label_counts) / max(label_counts)
            class_results["balance_ratio"] = float(balance_ratio)
            class_results["is_balanced"] = (
                balance_ratio >= 0.5
            )  # Reasonable balance threshold
        else:
            class_results["balance_ratio"] = 1.0
            class_results["is_balanced"] = True

        # Validation criteria
        sufficient_samples = min_samples >= self.config.min_samples_per_class

        class_results["validation_passed"] = sufficient_samples

        if not sufficient_samples:
            self.validation_passed = False
            self.logger.error(
                f"Insufficient samples in classes: {insufficient_classes.to_dict()}"
            )

        self.validation_results["class_distribution"] = class_results
        self.logger.info(
            f"Class distribution validation completed. Balance ratio: {class_results.get('balance_ratio', 'N/A')}"
        )

    def _generate_validation_report(self, data_path: str, df: pd.DataFrame) -> str:
        """Generate comprehensive validation report."""
        self.logger.info("Generating validation report...")

        # Create comprehensive report
        report = {
            "metadata": {
                "data_path": data_path,
                "validation_timestamp": datetime.now().isoformat(),
                "dataset_shape": df.shape,
                "validation_passed": self.validation_passed,
            },
            "configuration": {
                "expected_columns": self.config.expected_columns,
                "required_columns": self.config.required_columns,
                "min_text_length": self.config.min_text_length,
                "max_text_length": self.config.max_text_length,
                "expected_labels": self.config.expected_labels,
                "min_samples_per_class": self.config.min_samples_per_class,
                "max_null_percentage": self.config.max_null_percentage,
            },
            "validation_results": self.validation_results,
            "summary": self._generate_summary(),
        }

        # Save report
        report_path = self.config.validation_report_dir / self.config.report_filename
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Validation report saved to: {report_path}")

        return str(report_path)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            "overall_validation_passed": self.validation_passed,
            "total_checks": len(self.validation_results),
            "passed_checks": sum(
                1
                for result in self.validation_results.values()
                if result.get("validation_passed", False)
            ),
            "failed_checks": [],
        }

        # Collect failed checks
        for check_name, result in self.validation_results.items():
            if not result.get("validation_passed", True):
                summary["failed_checks"].append(check_name)

        summary["success_rate"] = (
            summary["passed_checks"] / summary["total_checks"]
            if summary["total_checks"] > 0
            else 0
        )

        return summary

    def _get_s3_cache_key(self, data_path: str) -> str:
        """Generate S3 cache key for the current validation configuration."""
        # Create hash based on data file and validation config
        data_hash = hashlib.md5(data_path.encode()).hexdigest()[:8]

        cache_components = [
            str(self.validation_config.min_text_length),
            str(self.validation_config.max_text_length),
            str(self.validation_config.min_samples_per_class),
            str(self.validation_config.max_null_percentage),
            ",".join(str(x) for x in self.validation_config.expected_labels),
        ]

        config_string = "_".join(cache_components)
        config_hash = hashlib.md5(config_string.encode()).hexdigest()[:12]

        return f"artifacts/reports/cache/validation_{data_hash}_{config_hash}.json"

    def _should_use_s3_cache(self) -> bool:
        """Determine if S3 cache should be checked."""
        return getattr(self.validation_config, "use_s3_cache", True)

    def _try_download_from_s3_cache(self, data_path: str) -> Optional[str]:
        """Try to download validation report from S3 cache."""
        try:
            cache_key = self._get_s3_cache_key(data_path)
            local_cache_path = (
                self.validation_config.validation_report_dir
                / self.validation_config.report_filename
            )

            self.logger.info(f"Checking S3 cache for validation report: {cache_key}")

            # Check if file exists in S3
            if not self.s3_manager.file_exists(cache_key, "data"):
                self.logger.info("No cached validation report found in S3")
                return None

            # Download cached version
            success = self.s3_manager.download_file(
                s3_path=cache_key, local_path=local_cache_path, folder_type="data"
            )

            if success and local_cache_path.exists():
                self.logger.info(
                    "Successfully downloaded cached validation report from S3"
                )
                return str(local_cache_path)
            else:
                self.logger.warning(
                    "Failed to download cached validation report from S3"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error checking S3 cache for validation report: {e}")
            return None

    def _load_validation_results_from_report(self, report_path: str) -> bool:
        """Load validation results from a cached report."""
        try:
            with open(report_path, "r") as f:
                report = json.load(f)

            self.validation_results = report.get("validation_results", {})
            validation_passed = report.get("metadata", {}).get(
                "validation_passed", False
            )

            self.logger.info(
                f"Loaded validation results from cached report: {validation_passed}"
            )
            return validation_passed

        except Exception as e:
            self.logger.warning(
                f"Error loading validation results from cached report: {e}"
            )
            return False

    def _upload_validation_report_to_s3(self, report_path: str, data_path: str) -> bool:
        """Upload validation report to S3 for backup and caching."""
        try:
            # Upload original validation report
            report_s3_key = "artifacts/reports/data_validation_report.json"

            # Prepare metadata
            metadata = {
                "component": "data_validation",
                "data_path": data_path,
                "validation_passed": str(self.validation_passed),
                "total_checks": str(len(self.validation_results)),
                "passed_checks": str(
                    sum(
                        1
                        for result in self.validation_results.values()
                        if result.get("validation_passed", False)
                    )
                ),
                "validated_at": datetime.now().isoformat(),
            }

            # Upload to main artifacts folder
            success1 = self.s3_manager.upload_file(
                local_path=report_path,
                s3_path=report_s3_key,
                folder_type="artifacts",
                metadata=metadata,
            )

            # Upload to cache folder for future use
            cache_key = self._get_s3_cache_key(data_path)
            success2 = self.s3_manager.upload_file(
                local_path=report_path,
                s3_path=cache_key,
                folder_type="data",
                metadata=metadata,
            )

            if success1:
                self.logger.info(
                    f"Successfully uploaded validation report to S3: {report_s3_key}"
                )
            if success2:
                self.logger.info(
                    f"Successfully cached validation report to S3: {cache_key}"
                )

            return success1 and success2

        except Exception as e:
            self.logger.warning(f"Error uploading validation report to S3: {e}")
            return False

    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status."""
        return {
            "validation_passed": self.validation_passed,
            "validation_results": self.validation_results,
            "summary": self._generate_summary(),
        }
