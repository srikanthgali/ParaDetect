from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion component"""

    dataset_name: str
    source_type: str
    raw_data_dir: Path
    dataset_filename: str
    sample_size: Optional[int] = None
    random_state: int = 42
    use_s3_cache: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.source_type not in ["huggingface", "local", "url"]:
            raise ValueError(f"Invalid source_type: {self.source_type}")

        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got: {self.sample_size}")
