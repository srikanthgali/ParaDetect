"""Data preprocessing module for ParaDetect"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging


from para_detect.core.base_component import BaseComponent
from para_detect.core.exceptions import DataPreprocessingError
from para_detect.entities.data_preprocessing_config import DataPreprocessingConfig
from para_detect.constants import HUMAN_LABEL, AI_LABEL


class DataPreprocessing(BaseComponent):
    """
    Data preprocessing component for ParaDetect pipeline.

    Handles text cleaning, label encoding, duplicate removal,
    class balancing, and other preprocessing tasks.
    """

    def __init__(self, config: DataPreprocessingConfig):
        """
        Initialize DataPreprocessing component.

        Args:
            config: DataPreprocessingConfig object containing preprocessing parameters
        """
        self.config = config

        super().__init__(config)

        # Create directories
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("DataPreprocessing component initialized")

    def run(self, input_data_path: str) -> str:
        """
        Execute data preprocessing process.

        Args:
            input_data_path: Path to the raw data file

        Returns:
            str: Path to the processed data file

        Raises:
            DataPreprocessingError: If preprocessing fails
        """
        try:
            self.logger.info(f"Starting data preprocessing from: {input_data_path}")

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
            df = pd.read_csv(input_path)
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
        df.to_csv(output_path, index=False)

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
        }

        metadata_path = self.config.processed_data_dir / "preprocessing_metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to: {metadata_path}")

        return str(output_path)

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
