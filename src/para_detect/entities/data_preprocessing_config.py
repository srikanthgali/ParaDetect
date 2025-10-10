from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataPreprocessingConfig:
    """Configuration for data preprocessing component"""

    text_column: str
    label_column: str
    source_column: str
    remove_duplicates: bool
    min_text_length: int
    max_text_length: int
    lowercase: bool
    strip_whitespace: bool
    remove_special_chars: bool
    balance_classes: bool
    processed_data_dir: Path
    processed_filename: str
    random_state: int = 42

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.min_text_length < 1:
            raise ValueError(
                f"min_text_length must be positive, got: {self.min_text_length}"
            )

        if self.max_text_length <= self.min_text_length:
            raise ValueError(f"max_text_length must be greater than min_text_length")
