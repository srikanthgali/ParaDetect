"""Data ingestion module for ParaDetect"""

import os
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import Tuple, Optional

from para_detect.core.base_component import BaseComponent
from para_detect.core.exceptions import DataIngestionError
from para_detect.entities.data_ingestion_config import DataIngestionConfig


class DataIngestion(BaseComponent):
    """
    Data ingestion component for ParaDetect pipeline.

    Handles downloading and loading data from various sources including:
    - HuggingFace datasets
    - Local files
    - Remote URLs
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion component.

        Args:
            config: DataIngestionConfig entity object containing ingestion parameters
        """
        # Keep the typed config for component-specific operations
        self.ingestion_config = config

        # Pass the config as a dictionary to the base class for backward compatibility
        # The base class will convert it to ConfigBox
        super().__init__(config.__dict__)

        # Create directories
        self.ingestion_config.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"DataIngestion component initialized with source: {config.source_type}"
        )

    def validate_config(self) -> bool:
        """
        Validate data ingestion specific configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Check if required config values are present
            required_fields = [
                "dataset_name",
                "source_type",
                "raw_data_dir",
                "dataset_filename",
            ]

            for field in required_fields:
                if not hasattr(self.ingestion_config, field):
                    self.logger.error(f"Missing required config field: {field}")
                    return False

                value = getattr(self.ingestion_config, field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    self.logger.error(
                        f"Invalid value for config field {field}: {value}"
                    )
                    return False

            # Validate source type
            valid_sources = ["huggingface", "local", "url"]
            if self.ingestion_config.source_type not in valid_sources:
                self.logger.error(
                    f"Invalid source_type: {self.ingestion_config.source_type}"
                )
                return False

            # Validate paths
            if not isinstance(self.ingestion_config.raw_data_dir, Path):
                self.logger.error("raw_data_dir must be a Path object")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Config validation failed: {str(e)}")
            return False

    def run(self) -> str:
        """
        Execute data ingestion process.

        Returns:
            str: Path to the ingested data file

        Raises:
            DataIngestionError: If ingestion fails
        """
        try:
            self.logger.info("Starting data ingestion process...")

            if self.ingestion_config.source_type == "huggingface":
                data_path = self._ingest_from_huggingface()
            elif self.ingestion_config.source_type == "local":
                data_path = self._ingest_from_local()
            else:
                raise DataIngestionError(
                    f"Unsupported source type: {self.ingestion_config.source_type}"
                )

            self.logger.info(
                f"Data ingestion completed successfully. Data saved to: {data_path}"
            )
            return data_path

        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataIngestionError(error_msg) from e

    def _ingest_from_huggingface(self) -> str:
        """
        Ingest data from HuggingFace Hub.

        Returns:
            str: Path to the saved CSV file
        """
        try:
            self.logger.info(
                f"Loading dataset from HuggingFace: {self.config.dataset_name}"
            )

            # Load dataset
            dataset = load_dataset(self.config.dataset_name)

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset["train"])

            self.logger.info(f"Original dataset shape: {df.shape}")
            self.logger.info(f"Dataset columns: {df.columns.tolist()}")

            # Apply sampling if specified
            if self.config.sample_size is not None:
                if self.config.sample_size < len(df):
                    df = df.sample(
                        n=self.config.sample_size, random_state=self.config.random_state
                    ).reset_index(drop=True)
                    self.logger.info(
                        f"Sampled dataset to {self.config.sample_size} rows"
                    )

            # Save to CSV
            output_path = self.config.raw_data_dir / self.config.dataset_filename
            df.to_csv(output_path, index=False)

            self.logger.info(f"Dataset saved to: {output_path}")
            self.logger.info(f"Final dataset shape: {df.shape}")

            # Log sample data for verification
            self._log_data_sample(df)

            return str(output_path)

        except Exception as e:
            raise DataIngestionError(
                f"Failed to ingest data from HuggingFace: {str(e)}"
            ) from e

    def _ingest_from_local(self) -> str:
        """
        Ingest data from local file.

        Returns:
            str: Path to the data file
        """
        try:
            # Look for existing data files
            data_file = self.config.raw_data_dir / self.config.dataset_filename

            if not data_file.exists():
                # Look for any CSV files in the directory
                csv_files = list(self.config.raw_data_dir.glob("*.csv"))
                if not csv_files:
                    raise DataIngestionError(
                        f"No data files found in {self.config.raw_data_dir}"
                    )

                data_file = csv_files[0]
                self.logger.warning(f"Using first available CSV file: {data_file}")

            # Validate file
            if not data_file.exists():
                raise DataIngestionError(f"Data file not found: {data_file}")

            # Load and validate data
            df = pd.read_csv(data_file)
            self.logger.info(f"Loaded local dataset from: {data_file}")
            self.logger.info(f"Dataset shape: {df.shape}")

            # Log sample data for verification
            self._log_data_sample(df)

            return str(data_file)

        except Exception as e:
            raise DataIngestionError(
                f"Failed to ingest data from local file: {str(e)}"
            ) from e

    def _log_data_sample(self, df: pd.DataFrame, n_samples: int = 3) -> None:
        """
        Log sample data for verification.

        Args:
            df: DataFrame to sample from
            n_samples: Number of samples to log
        """
        try:
            self.logger.info("=== Dataset Sample ===")
            for i in range(min(n_samples, len(df))):
                sample = df.iloc[i]
                text_preview = (
                    str(sample.get("text", "N/A"))[:100] + "..."
                    if len(str(sample.get("text", ""))) > 100
                    else str(sample.get("text", "N/A"))
                )
                self.logger.info(f"Sample {i+1}:")
                self.logger.info(f"  Text: {text_preview}")
                self.logger.info(f"  Label: {sample.get('generated', 'N/A')}")
                self.logger.info(f"  Source: {sample.get('source', 'N/A')}")

        except Exception as e:
            self.logger.warning(f"Could not log data sample: {str(e)}")

    def get_data_info(self, file_path: str) -> dict:
        """
        Get comprehensive information about the dataset.

        Args:
            file_path: Path to the data file

        Returns:
            dict: Dataset information including shape, columns, types, etc.
        """
        try:
            df = pd.read_csv(file_path)

            info = {
                "file_path": file_path,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "duplicate_rows": df.duplicated().sum(),
            }

            # Add column-specific info
            if "text" in df.columns:
                info["text_stats"] = {
                    "avg_length": df["text"].str.len().mean(),
                    "min_length": df["text"].str.len().min(),
                    "max_length": df["text"].str.len().max(),
                }

            if "generated" in df.columns:
                info["label_distribution"] = df["generated"].value_counts().to_dict()

            return info

        except Exception as e:
            self.logger.error(f"Failed to get data info: {str(e)}")
            return {"error": str(e)}
