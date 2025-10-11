from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference pipeline"""

    # Model configuration
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    device_preference: Optional[str] = None  # auto, cuda, mps, cpu

    # Inference parameters
    batch_size: int = 32
    max_length: int = 512

    # Processing configuration
    text_column: str = "text"
    preprocessing_enabled: bool = True

    # Output configuration
    include_probabilities: bool = True
    include_confidence: bool = True
    confidence_threshold: float = 0.5

    # Monitoring configuration
    enable_monitoring: bool = True
    log_predictions: bool = False  # For privacy, default to False

    # Batch processing configuration
    chunk_size: int = 1000  # For large batch processing
    progress_bar: bool = True

    # Error handling
    skip_errors: bool = True
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
