"""Data ingestion module for ParaDetect"""

import hashlib
import os
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from para_detect.core.base_component import BaseComponent
from para_detect.core.config_manager import ConfigurationManager
from para_detect.core.exceptions import DataIngestionError
from para_detect.entities.data_ingestion_config import DataIngestionConfig
from para_detect.utils.s3_manager import S3Manager


class DataIngestion(BaseComponent):
    """
    Data ingestion component for ParaDetect pipeline.

    Handles downloading and loading data from various sources including:
    - HuggingFace datasets
    - Local files
    - Remote URLs
    """

    def __init__(
        self, config: DataIngestionConfig, s3_manager: Optional[S3Manager] = None
    ):
        """
        Initialize DataIngestion component.

        Args:
            config: DataIngestionConfig entity object containing ingestion parameters
            s3_manager: Optional S3Manager instance for S3 operations
        """
        # Keep the typed config for component-specific operations
        self.ingestion_config = config

        # Keep S3 manager for S3 operations
        self.s3_manager = s3_manager
        self.s3_enabled = s3_manager is not None

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
            valid_sources = ["huggingface", "local", "url", "s3"]
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
        Execute data ingestion process with S3 backup.

        Returns:
            str: Path to the ingested data file

        Raises:
            DataIngestionError: If ingestion fails
        """
        try:
            self.logger.info("Starting data ingestion process...")

            # Check if data exists in S3 cache first
            local_data_path = None
            if self.s3_enabled and self._should_use_s3_cache():
                local_data_path = self._try_download_from_s3_cache()

            if local_data_path and Path(local_data_path).exists():
                self.logger.info(f"Using cached data from S3: {local_data_path}")
                return local_data_path

            # Proceed with normal ingestion if no S3 cache
            if self.ingestion_config.source_type == "huggingface":
                data_path = self._ingest_from_huggingface()
            elif self.ingestion_config.source_type == "local":
                data_path = self._ingest_from_local()
            elif self.ingestion_config.source_type == "s3":
                data_path = self._ingest_from_s3()
            else:
                raise DataIngestionError(
                    f"Unsupported source type: {self.ingestion_config.source_type}"
                )

            # Upload to S3 for backup and caching
            if self.s3_enabled:
                self._upload_data_to_s3(data_path)

            self.logger.info(
                f"Data ingestion completed successfully. Data saved to: {data_path}"
            )
            return data_path

        except Exception as e:
            error_msg = f"Data ingestion failed: {str(e)}"
            self.logger.error(error_msg)
            raise DataIngestionError(error_msg) from e

    def _ingest_from_s3(self) -> str:
        """
        Ingest data directly from S3 bucket.

        Returns:
            str: Path to the downloaded data file
        """
        try:
            if not self.s3_enabled:
                raise DataIngestionError("S3 integration not available")

            # Parse S3 path from dataset_name
            dataset_name = self.ingestion_config.dataset_name

            if dataset_name.startswith("s3://"):
                # Full S3 URI provided - extract bucket and key
                s3_path = dataset_name.replace("s3://", "").split("/", 1)[1]
            else:
                # Assume it's a key in the configured bucket
                s3_path = dataset_name

            self.logger.info(f"Downloading dataset from S3: {s3_path}")

            # Download from S3
            local_path = (
                self.ingestion_config.raw_data_dir
                / self.ingestion_config.dataset_filename
            )

            success = self.s3_manager.download_file(
                s3_path=s3_path, local_path=local_path, folder_type="data"
            )

            if not success:
                raise DataIngestionError(f"Failed to download from S3: {s3_path}")

            # Validate downloaded data
            if not local_path.exists() or local_path.stat().st_size == 0:
                raise DataIngestionError(
                    f"Downloaded file is empty or missing: {local_path}"
                )

            # Load and validate data structure
            df = pd.read_parquet(local_path)
            self.logger.info(f"Downloaded dataset shape: {df.shape}")

            # Apply sampling if specified
            if self.ingestion_config.sample_size is not None:
                if self.ingestion_config.sample_size < len(df):
                    df = df.sample(
                        n=self.ingestion_config.sample_size,
                        random_state=self.ingestion_config.random_state,
                    ).reset_index(drop=True)

                    # Save the sampled version
                    df.to_parquet(local_path, index=False, engine="pyarrow")
                    self.logger.info(
                        f"Sampled dataset to {self.ingestion_config.sample_size} rows"
                    )

            # Log sample data for verification
            self._log_data_sample(df)

            return str(local_path)

        except Exception as e:
            raise DataIngestionError(f"Failed to ingest data from S3: {str(e)}") from e

    def _get_s3_cache_key(self) -> str:
        """Generate S3 cache key for the current dataset configuration."""
        cache_components = [
            self.ingestion_config.dataset_name,
            str(self.ingestion_config.sample_size),
            str(self.ingestion_config.random_state),
        ]

        cache_string = "_".join(str(c) for c in cache_components)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()[:12]

        return f"data/raw/cache/{self.ingestion_config.dataset_name.replace('/', '_')}_{cache_hash}.parquet"

    def _should_use_s3_cache(self) -> bool:
        """Determine if S3 cache should be checked."""
        return (
            self.ingestion_config.source_type
            != "s3"  # Don't cache if source is already S3
            and getattr(self.ingestion_config, "use_s3_cache", True)
        )

    def _try_download_from_s3_cache(self) -> Optional[str]:
        """Try to download data from S3 cache."""
        try:
            cache_key = self._get_s3_cache_key()
            local_cache_path = (
                self.ingestion_config.raw_data_dir
                / self.ingestion_config.dataset_filename
            )

            self.logger.info(f"Checking S3 cache: {cache_key}")

            # Check if file exists in S3
            if not self.s3_manager.file_exists(cache_key, "data"):
                self.logger.info("No cached version found in S3")
                return None

            # Download cached version
            success = self.s3_manager.download_file(
                s3_path=cache_key, local_path=local_cache_path, folder_type="data"
            )

            if success and local_cache_path.exists():
                self.logger.info("Successfully downloaded cached data from S3")
                return str(local_cache_path)
            else:
                self.logger.warning("Failed to download cached data from S3")
                return None

        except Exception as e:
            self.logger.warning(f"Error checking S3 cache: {e}")
            return None

    def _upload_data_to_s3(self, data_path: str) -> bool:
        """Upload processed data to S3 for backup and caching."""
        try:
            # Upload original data file
            original_s3_key = f"data/raw/{self.ingestion_config.dataset_filename}"

            # Prepare metadata
            metadata = {
                "dataset_name": self.ingestion_config.dataset_name,
                "source_type": self.ingestion_config.source_type,
                "sample_size": (
                    str(self.ingestion_config.sample_size)
                    if self.ingestion_config.sample_size
                    else "full"
                ),
                "random_state": str(self.ingestion_config.random_state),
                "ingested_at": datetime.now().isoformat(),
                "component": "data_ingestion",
            }

            # Upload to main data folder
            success1 = self.s3_manager.upload_file(
                local_path=data_path,
                s3_path=original_s3_key,
                folder_type="data",
                metadata=metadata,
            )

            # Upload to cache folder for future use
            cache_key = self._get_s3_cache_key()
            success2 = self.s3_manager.upload_file(
                local_path=data_path,
                s3_path=cache_key,
                folder_type="data",
                metadata=metadata,
            )

            if success1:
                self.logger.info(f"Successfully uploaded data to S3: {original_s3_key}")
            if success2:
                self.logger.info(f"Successfully cached data to S3: {cache_key}")

            return success1 and success2

        except Exception as e:
            self.logger.warning(f"Error uploading data to S3: {e}")
            return False

    def _ingest_from_huggingface(self) -> str:
        """
        Ingest data from HuggingFace Hub.

        Returns:
            str: Path to the saved Parquet file
        """
        try:
            self.logger.info(
                f"Loading dataset from HuggingFace: {self.ingestion_config.dataset_name}"
            )

            # Load dataset
            dataset = load_dataset(self.ingestion_config.dataset_name)

            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset["train"])

            self.logger.info(f"Original dataset shape: {df.shape}")
            self.logger.info(f"Dataset columns: {df.columns.tolist()}")

            # Apply sampling if specified
            if self.ingestion_config.sample_size is not None:
                if self.ingestion_config.sample_size < len(df):
                    df = df.sample(
                        n=self.ingestion_config.sample_size,
                        random_state=self.ingestion_config.random_state,
                    ).reset_index(drop=True)
                    self.logger.info(
                        f"Sampled dataset to {self.ingestion_config.sample_size} rows"
                    )

            # Save to Parquet
            output_path = (
                self.ingestion_config.raw_data_dir
                / self.ingestion_config.dataset_filename
            )
            df.to_parquet(output_path, index=False, engine="pyarrow")

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
            data_file = (
                self.ingestion_config.raw_data_dir
                / self.ingestion_config.dataset_filename
            )

            if not data_file.exists():
                # Look for any Parquet files in the directory
                parquet_files = list(
                    self.ingestion_config.raw_data_dir.glob("*.parquet")
                )
                if not parquet_files:
                    raise DataIngestionError(
                        f"No data files found in {self.ingestion_config.raw_data_dir}"
                    )

                data_file = parquet_files[0]
                self.logger.warning(f"Using first available Parquet file: {data_file}")

            # Validate file
            if not data_file.exists():
                raise DataIngestionError(f"Data file not found: {data_file}")

            # Load and validate data
            df = pd.read_parquet(data_file)
            self.logger.info(f"Loaded local dataset from: {data_file}")
            self.logger.info(f"Dataset shape: {df.shape}")

            # Apply sampling if specified
            if self.ingestion_config.sample_size is not None:
                if self.ingestion_config.sample_size < len(df):
                    df = df.sample(
                        n=self.ingestion_config.sample_size,
                        random_state=self.ingestion_config.random_state,
                    ).reset_index(drop=True)

                    # Save the sampled version
                    df.to_parquet(data_file, index=False, engine="pyarrow")
                    self.logger.info(
                        f"Sampled dataset to {self.ingestion_config.sample_size} rows"
                    )

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

    def list_s3_datasets(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List datasets available in S3."""
        try:
            if not self.s3_enabled:
                return []

            files = self.s3_manager.list_files(prefix, "data")

            datasets = []
            for file_info in files:
                datasets.append(
                    {
                        "s3_path": file_info["key"],
                        "size_bytes": file_info["size"],
                        "last_modified": file_info["last_modified"].isoformat(),
                        "etag": file_info["etag"],
                    }
                )

            return datasets

        except Exception as e:
            self.logger.error(f"Failed to list S3 datasets: {e}")
            return []

    def get_data_info(self, file_path: str) -> dict:
        """
        Get comprehensive information about the dataset.

        Args:
            file_path: Path to the data file

        Returns:
            dict: Dataset information including shape, columns, types, etc.
        """
        try:
            df = pd.read_parquet(file_path)

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
