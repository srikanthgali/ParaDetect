from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Configuration for model training component"""

    # Model configuration
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    num_labels: int = 2

    # Tokenizer configuration
    max_length: int = 512
    text_column: str = "text"
    label_column: str = "generated"
    padding: bool = True
    truncation: bool = True
    return_tensors: str = "None"  # 'pt', 'tf', 'np', or 'None'

    # Training parameters
    output_dir: Path = Path("artifacts/models")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Precision and optimization
    fp16: bool = False
    bf16: bool = True

    # Dataset configuration
    max_length: int = 512
    text_column: str = "text"
    label_column: str = "generated"
    train_path: Optional[str] = None
    validation_split: float = 0.15
    test_split: float = 0.15

    # Evaluation and checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Resumption and checkpointing
    resume_from_checkpoint: bool = True
    checkpoint_interval_steps: int = 500

    # LoRA/PEFT configuration
    use_peft: bool = True
    peft_config: Optional[Dict[str, Any]] = None

    # Logging and monitoring
    logging_steps: int = 100
    report_to: Optional[List[str]] = None
    run_name: Optional[str] = None

    # Data handling
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = False

    # Model Loading Configuration
    device_map: str = "auto"  # auto, cpu, or specific device mapping
    torch_dtype_loading: str = "auto"  # auto, float32, float16, bfloat16
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False

    # Model Saving Configuration
    save_model: bool = True
    torch_dtype_saving: str = "float32"
    safe_serialization: bool = True  # Use safetensors format
    save_metadata: bool = True
    save_config: bool = True
    save_tokenizer: bool = True
    create_model_card: bool = True
    save_training_args: bool = True

    # Random seed
    seed: int = 42

    # Device configuration
    device_preference: Optional[str] = None  # auto, cuda, mps, cpu

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")

        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.validation_split < 0 or self.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.test_split < 0 or self.test_split >= 1:
            raise ValueError("test_split must be between 0 and 1")

        if self.eval_strategy not in ["no", "steps", "epoch"]:
            raise ValueError("eval_strategy must be 'no', 'steps', or 'epoch'")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)"""

    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    bias: str = "none"  # none, all, lora_only
    target_modules: Optional[List[str]] = None
    task_type: str = "SEQ_CLS"
    inference_mode: bool = False

    def __post_init__(self):
        """Validate LoRA configuration"""
        if self.r <= 0:
            raise ValueError("LoRA rank (r) must be positive")

        if self.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")

        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("LoRA dropout must be between 0 and 1")

        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError("bias must be 'none', 'all', or 'lora_only'")
