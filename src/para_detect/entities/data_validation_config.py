from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation component"""

    expected_columns: List[str]
    required_columns: List[str]
    text_column: str
    label_column: str
    min_text_length: int
    max_text_length: int
    expected_labels: List[Union[int, str]]
    min_samples_per_class: int
    max_null_percentage: float
    validation_report_dir: Path
    report_filename: str
    use_s3_cache: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not all(col in self.expected_columns for col in self.required_columns):
            raise ValueError("All required_columns must be in expected_columns")

        if self.max_null_percentage < 0 or self.max_null_percentage > 1:
            raise ValueError(
                f"max_null_percentage must be between 0 and 1, got: {self.max_null_percentage}"
            )

        if self.min_samples_per_class < 1:
            raise ValueError(
                f"min_samples_per_class must be positive, got: {self.min_samples_per_class}"
            )
