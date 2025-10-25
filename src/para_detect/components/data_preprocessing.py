"""Data preprocessing module for ParaDetect"""

import hashlib
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime
import json

from para_detect.core.base_component import BaseComponent
from para_detect.core.config_manager import ConfigurationManager
from para_detect.core.exceptions import DataPreprocessingError
from para_detect.entities.data_preprocessing_config import DataPreprocessingConfig
from para_detect.utils.s3_manager import S3Manager
from para_detect.constants import HUMAN_LABEL, AI_LABEL


class DataPreprocessing(BaseComponent):
    """
    Data preprocessing component for ParaDetect pipeline.

    Handles text cleaning, label encoding, duplicate removal,
    class balancing, and other preprocessing tasks.
    """

    def __init__(
        self, config: DataPreprocessingConfig, s3_manager: Optional[S3Manager] = None
    ):
        """
        Initialize DataPreprocessing component.

        Args:
            config: DataPreprocessingConfig object containing preprocessing parameters
            s3_manager: Optional S3Manager instance for S3 operations
        """
        # Keep the typed config for component-specific operations
        self.preprocessing_config = config

        # Keep S3 manager for S3 operations
        self.s3_manager = s3_manager
        self.s3_enabled = s3_manager is not None

        # Pass the config as a dictionary to the base class for backward compatibility
        super().__init__(config.__dict__)

        # Create directories
        self.preprocessing_config.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("DataPreprocessing component initialized")

    def run(self, input_data_path: str) -> str:
        """
        Execute data preprocessing process with S3 backup.

        Args:
            input_data_path: Path to the raw data file

        Returns:
            str: Path to the processed data file

        Raises:
            DataPreprocessingError: If preprocessing fails
        """
        try:
            self.logger.info(f"Starting data preprocessing from: {input_data_path}")

            # Store input path for cache key generation
            self._current_input_path = input_data_path

            # Check if processed data exists in S3 cache first
            local_processed_path = None
            if self.s3_enabled and self._should_use_s3_cache():
                local_processed_path = self._try_download_from_s3_cache(input_data_path)

            if local_processed_path and Path(local_processed_path).exists():
                self.logger.info(
                    f"Using cached processed data from S3: {local_processed_path}"
                )
                return local_processed_path

            # Load data
            df = self._load_data(input_data_path)

            # Log initial statistics
            self._log_preprocessing_stats(df, "Initial")

            # Apply preprocessing steps
            df = self._create_binary_labels_from_source(df)
            df = self._clean_data(df)
            df = self._process_text(df)
            df = self._process_labels(df)
            df = self._filter_data(df)

            if self.config.balance_classes:
                df = self._balance_classes(df)

            # Final validation and shuffle
            df = self._finalize_data(df)

            # Save processed data
            output_path = self._save_processed_data(df)

            # Upload to S3 for backup and caching
            if self.s3_enabled:
                self._upload_processed_data_to_s3(output_path, df)

            # Log final statistics
            self._log_preprocessing_stats(df, "Final")

            self.logger.info(
                f"Data preprocessing completed successfully. Processed data saved to: {output_path}"
            )
            return output_path

        except Exception as e:
            error_msg = f"Data preprocessing failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataPreprocessingError(error_msg) from e

    def _load_data(self, input_path: str) -> pd.DataFrame:
        """Load data from input file."""
        try:
            df = pd.read_parquet(input_path)
            self.logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            raise DataPreprocessingError(
                f"Failed to load data from {input_path}: {str(e)}"
            ) from e

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset by removing duplicates and handling missing values."""
        initial_shape = df.shape

        # Remove duplicates
        if self.config.remove_duplicates:
            df = df.drop_duplicates(subset=[self.config.text_column])
            self.logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")

        # Handle missing values in text column
        missing_text = df[self.config.text_column].isnull().sum()
        if missing_text > 0:
            self.logger.warning(
                f"Found {missing_text} missing values in text column, dropping them"
            )
            df = df.dropna(subset=[self.config.text_column])

        # Handle missing values in label column
        missing_labels = df[self.config.label_column].isnull().sum()
        if missing_labels > 0:
            self.logger.warning(
                f"Found {missing_labels} missing values in label column, dropping them"
            )
            df = df.dropna(subset=[self.config.label_column])

        df = df.reset_index(drop=True)
        self.logger.info(f"After cleaning: {df.shape}")

        return df

    def _create_binary_labels_from_source(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels from source column."""
        self.logger.info("Creating binary labels from source column...")

        # Analyze source distribution
        self.logger.info("=== Source Distribution ===")
        source_counts = df[self.config.source_column].value_counts()
        self.logger.info(f"Source counts: {source_counts.to_dict()}")

        # Define source mappings
        human_sources = ["human", "wiki", "reddit", "book", "wikipedia"]
        ai_sources = ["gpt", "chatgpt", "ai", "generated", "claude", "palm"]

        def create_binary_label(source):
            source_lower = str(source).lower()
            if any(human_src in source_lower for human_src in human_sources):
                return HUMAN_LABEL  # Human
            elif any(ai_src in source_lower for ai_src in ai_sources):
                return AI_LABEL  # AI
            else:
                # For unknown sources, mark for manual review
                return -1  # Unknown

        # Create binary labels
        df[self.config.label_column] = df[self.config.source_column].apply(
            create_binary_label
        )

        # Check label distribution
        self.logger.info("=== Binary Label Distribution ===")
        label_counts = df[self.config.label_column].value_counts()
        self.logger.info(f"Human (0): {label_counts.get(0, 0)}")
        self.logger.info(f"AI (1): {label_counts.get(1, 0)}")
        self.logger.info(f"Unknown (-1): {label_counts.get(-1, 0)}")

        # Handle unknown sources
        if -1 in label_counts.index:
            unknown_sources = df[df[self.config.label_column] == -1][
                self.config.source_column
            ].unique()
            self.logger.warning(f"Unknown sources found: {unknown_sources}")

            # For now, assign unknown sources to human (0) - you can adjust this logic
            df.loc[df[self.config.label_column] == -1, self.config.label_column] = (
                HUMAN_LABEL
            )
            self.logger.info("Assigned unknown sources to human label (0)")

            # Log final distribution after handling unknowns
            final_counts = df[self.config.label_column].value_counts()
            self.logger.info(f"Final label distribution: {final_counts.to_dict()}")

        return df

    def _process_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean text data."""
        self.logger.info("Processing text data...")

        # Convert to string and handle NaN
        df[self.config.text_column] = df[self.config.text_column].astype(str)

        # Apply text cleaning
        df["processed_text"] = df[self.config.text_column].apply(self._clean_text)

        # Calculate text statistics
        df["text_length"] = df["processed_text"].str.len()
        df["word_count"] = df["processed_text"].str.split().str.len()

        self.logger.info(
            f"Text length stats - Min: {df['text_length'].min()}, "
            f"Max: {df['text_length'].max()}, "
            f"Mean: {df['text_length'].mean():.1f}"
        )

        return df

    def _clean_text(self, text: str) -> str:
        """
        Clean individual text samples.

        Args:
            text: Raw text to clean

        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""

        text = str(text)

        # Strip whitespace
        if self.config.strip_whitespace:
            text = text.strip()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters if specified
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s.,!?;:"\'-]', "", text)

        # Remove multiple consecutive punctuation
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        # Convert to lowercase if specified
        if self.config.lowercase:
            text = text.lower()

        return text.strip()

    def _process_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate labels."""
        self.logger.info("Processing labels...")

        # Ensure labels are in correct format
        unique_labels = df[self.config.label_column].unique()
        self.logger.info(f"Unique labels found: {unique_labels}")

        # Convert labels to integers if they're not already
        try:
            df[self.config.label_column] = df[self.config.label_column].astype(int)
        except ValueError as e:
            # Handle string labels
            self.logger.info("Converting string labels to integers...")
            label_mapping = self._create_label_mapping(df[self.config.label_column])
            df[self.config.label_column] = df[self.config.label_column].map(
                label_mapping
            )
            self.logger.info(f"Label mapping used: {label_mapping}")

        # Validate label values
        final_labels = df[self.config.label_column].unique()
        expected_labels = [0, 1]  # Binary classification

        if not all(label in expected_labels for label in final_labels):
            self.logger.warning(f"Unexpected labels found: {final_labels}")

        return df

    def _create_label_mapping(self, labels: pd.Series) -> Dict[str, int]:
        """Create mapping from string labels to integers."""
        unique_labels = labels.unique()

        # Predefined mappings for common cases
        human_indicators = ["human", "real", "original", "0"]
        ai_indicators = ["ai", "generated", "artificial", "gpt", "chatgpt", "1"]

        mapping = {}
        for label in unique_labels:
            label_str = str(label).lower()
            if any(indicator in label_str for indicator in human_indicators):
                mapping[label] = HUMAN_LABEL
            elif any(indicator in label_str for indicator in ai_indicators):
                mapping[label] = AI_LABEL
            else:
                # Default mapping for unknown labels
                if len(mapping) == 0:
                    mapping[label] = HUMAN_LABEL
                else:
                    mapping[label] = AI_LABEL

        return mapping

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on text length and other criteria."""
        initial_shape = df.shape

        # Filter by text length
        mask = (df["text_length"] >= self.config.min_text_length) & (
            df["text_length"] <= self.config.max_text_length
        )
        df = df[mask].reset_index(drop=True)

        removed_count = initial_shape[0] - df.shape[0]
        self.logger.info(
            f"Filtered out {removed_count} samples based on text length criteria"
        )

        return df

    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance classes by undersampling the majority class."""
        label_counts = df[self.config.label_column].value_counts()
        self.logger.info(
            f"Class distribution before balancing: {label_counts.to_dict()}"
        )

        if len(label_counts) < 2:
            self.logger.warning("Only one class found, skipping class balancing")
            return df

        # Find minimum class size
        min_class_size = label_counts.min()

        # Sample equal amounts from each class
        balanced_dfs = []
        for label in label_counts.index:
            class_df = df[df[self.config.label_column] == label]
            sampled_df = class_df.sample(
                n=min_class_size, random_state=self.config.random_state
            )
            balanced_dfs.append(sampled_df)

        # Combine and shuffle
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        df_balanced = df_balanced.sample(
            frac=1, random_state=self.config.random_state
        ).reset_index(drop=True)

        final_counts = df_balanced[self.config.label_column].value_counts()
        self.logger.info(
            f"Class distribution after balancing: {final_counts.to_dict()}"
        )

        return df_balanced

    def _finalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize the processed dataset."""
        # Select final columns
        final_columns = ["processed_text", self.config.label_column]

        if self.config.source_column in df.columns:
            final_columns.append(self.config.source_column)

        # Create final dataset
        df_final = df[final_columns].copy()
        df_final = df_final.rename(columns={"processed_text": self.config.text_column})

        # Final shuffle
        df_final = df_final.sample(
            frac=1, random_state=self.config.random_state
        ).reset_index(drop=True)

        return df_final

    def _save_processed_data(self, df: pd.DataFrame) -> str:
        """Save processed data to file."""
        output_path = self.config.processed_data_dir / self.config.processed_filename
        df.to_parquet(output_path, index=False, engine="pyarrow")

        # Save metadata
        metadata = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "class_distribution": df[self.config.label_column].value_counts().to_dict(),
            "text_length_stats": {
                "min": int(df[self.config.text_column].str.len().min()),
                "max": int(df[self.config.text_column].str.len().max()),
                "mean": float(df[self.config.text_column].str.len().mean()),
            },
            "preprocessing_config": {
                "min_text_length": self.config.min_text_length,
                "max_text_length": self.config.max_text_length,
                "balance_classes": self.config.balance_classes,
                "remove_duplicates": self.config.remove_duplicates,
                "lowercase": self.config.lowercase,
                "random_state": self.config.random_state,
            },
            "created_at": datetime.now().isoformat(),
        }

        metadata_path = self.config.processed_data_dir / "preprocessing_metadata.json"

        with open(metadata_path, "w") as f:
            f.write(json.dumps(metadata, indent=2))

        self.logger.info(f"Metadata saved to: {metadata_path}")

        return str(output_path)

    def _get_s3_cache_key(self, input_data_path: str) -> str:
        """Generate S3 cache key for the current preprocessing configuration."""
        # Create a more stable hash based on filename and file stats
        input_path = Path(input_data_path)

        # Use filename and file size for consistent hashing
        cache_input_components = [input_path.name]

        # Add file size if file exists (for cache invalidation when file changes)
        if input_path.exists():
            cache_input_components.append(str(input_path.stat().st_size))

        input_string = "_".join(cache_input_components)
        input_hash = hashlib.md5(input_string.encode()).hexdigest()[:8]

        # Preprocessing config components
        cache_components = [
            str(self.preprocessing_config.min_text_length),
            str(self.preprocessing_config.max_text_length),
            str(self.preprocessing_config.balance_classes),
            str(self.preprocessing_config.remove_duplicates),
            str(self.preprocessing_config.lowercase),
            str(self.preprocessing_config.random_state),
        ]

        config_string = "_".join(cache_components)
        config_hash = hashlib.md5(config_string.encode()).hexdigest()[:12]

        return f"data/processed/cache/preprocessed_{input_hash}_{config_hash}.parquet"

    def _should_use_s3_cache(self) -> bool:
        """Determine if S3 cache should be checked."""
        return getattr(self.preprocessing_config, "use_s3_cache", True)

    def _upload_processed_data_to_s3(self, data_path: str, df: pd.DataFrame) -> bool:
        """Upload processed data and metadata to S3 for backup and caching."""
        try:
            # Define S3 paths
            processed_s3_key = (
                f"data/processed/{self.preprocessing_config.processed_filename}"
            )
            metadata_s3_key = f"data/processed/preprocessing_metadata.json"

            # Prepare metadata for S3 object metadata
            s3_metadata = {
                "component": "data_preprocessing",
                "shape": str(df.shape),
                "columns": ",".join(df.columns.tolist()),
                "class_distribution": str(
                    df[self.preprocessing_config.label_column].value_counts().to_dict()
                ),
                "min_text_length": str(self.preprocessing_config.min_text_length),
                "max_text_length": str(self.preprocessing_config.max_text_length),
                "balance_classes": str(self.preprocessing_config.balance_classes),
                "processed_at": datetime.now().isoformat(),
                "random_state": str(self.preprocessing_config.random_state),
            }

            # Upload processed data file
            success1 = self.s3_manager.upload_file(
                local_path=data_path,
                s3_path=processed_s3_key,
                folder_type="data",
                metadata=s3_metadata,
            )

            # Upload metadata file
            metadata_local_path = (
                self.config.processed_data_dir / "preprocessing_metadata.json"
            )
            success2 = self.s3_manager.upload_file(
                local_path=str(metadata_local_path),
                s3_path=metadata_s3_key,
                folder_type="data",
                metadata=s3_metadata,
            )

            # Upload to cache folder for future use
            cache_key = self._get_s3_cache_key(
                getattr(self, "_current_input_path", data_path)
            )
            cache_metadata_key = cache_key.replace(".parquet", "_metadata.json")

            success3 = self.s3_manager.upload_file(
                local_path=data_path,
                s3_path=cache_key,
                folder_type="data",
                metadata=s3_metadata,
            )

            success4 = self.s3_manager.upload_file(
                local_path=str(metadata_local_path),
                s3_path=cache_metadata_key,
                folder_type="data",
                metadata=s3_metadata,
            )

            # Log results
            if success1:
                self.logger.info(
                    f"Successfully uploaded processed data to S3: {processed_s3_key}"
                )
            if success2:
                self.logger.info(
                    f"Successfully uploaded metadata to S3: {metadata_s3_key}"
                )
            if success3:
                self.logger.info(
                    f"Successfully cached processed data to S3: {cache_key}"
                )
            if success4:
                self.logger.info(
                    f"Successfully cached metadata to S3: {cache_metadata_key}"
                )

            return success1 and success2 and success3 and success4

        except Exception as e:
            self.logger.warning(
                f"Error uploading processed data and metadata to S3: {e}"
            )
            return False

    def _try_download_from_s3_cache(self, input_data_path: str) -> Optional[str]:
        """Try to download preprocessed data and metadata from S3 cache."""
        try:
            cache_key = self._get_s3_cache_key(input_data_path)
            cache_metadata_key = cache_key.replace(".parquet", "_metadata.json")

            local_cache_path = (
                self.preprocessing_config.processed_data_dir
                / self.preprocessing_config.processed_filename
            )
            local_metadata_path = (
                self.preprocessing_config.processed_data_dir
                / "preprocessing_metadata.json"
            )

            self.logger.info(f"Checking S3 cache for preprocessed data: {cache_key}")
            self.logger.info(f"Checking S3 cache for metadata: {cache_metadata_key}")

            # Check if both files exist in S3
            data_exists = self.s3_manager.file_exists(cache_key, "data")
            metadata_exists = self.s3_manager.file_exists(cache_metadata_key, "data")

            self.logger.info(f"Cache data exists: {data_exists}")
            self.logger.info(f"Cache metadata exists: {metadata_exists}")

            if not (data_exists and metadata_exists):
                self.logger.info("No complete cached preprocessed data found in S3")
                return None

            # Download both files
            success1 = self.s3_manager.download_file(
                s3_path=cache_key, local_path=local_cache_path, folder_type="data"
            )

            success2 = self.s3_manager.download_file(
                s3_path=cache_metadata_key,
                local_path=local_metadata_path,
                folder_type="data",
            )

            if (
                success1
                and success2
                and local_cache_path.exists()
                and local_metadata_path.exists()
            ):
                self.logger.info(
                    "Successfully downloaded cached preprocessed data and metadata from S3"
                )
                return str(local_cache_path)
            else:
                self.logger.warning(
                    "Failed to download complete cached preprocessed data from S3"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error checking S3 cache for preprocessed data: {e}")
            return None

    def _log_preprocessing_stats(self, df: pd.DataFrame, stage: str) -> None:
        """Log statistics at different preprocessing stages."""
        self.logger.info(f"=== {stage} Dataset Statistics ===")
        self.logger.info(f"Shape: {df.shape}")

        if self.config.label_column in df.columns:
            label_dist = df[self.config.label_column].value_counts().to_dict()
            self.logger.info(f"Label distribution: {label_dist}")

        if "text_length" in df.columns:
            text_stats = df["text_length"].describe()
            self.logger.info(
                f"Text length - Min: {text_stats['min']}, Max: {text_stats['max']}, Mean: {text_stats['mean']:.1f}"
            )
