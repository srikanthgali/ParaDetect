from pathlib import Path

# Configuration paths
BASE_CONFIG_DIR = Path("configs")
MODEL_CONFIG_DIR = Path("configs/model_configs")
DEPLOYMENT_CONFIG_DIR = Path("configs/deployment")

BASE_CONFIG_FILE_PATH = BASE_CONFIG_DIR / "config.yaml"
DEBERTA_CONFIG_FILE_PATH = MODEL_CONFIG_DIR / "deberta_config.yaml"
TRAINING_CONFIG_FILE_PATH = MODEL_CONFIG_DIR / "training_config.yaml"
AWS_CONFIG_FILE_PATH = DEPLOYMENT_CONFIG_DIR / "aws_config.yaml"
LOCAL_CONFIG_FILE_PATH = DEPLOYMENT_CONFIG_DIR / "local_config.yaml"
COLAB_CONFIG_FILE_PATH = DEPLOYMENT_CONFIG_DIR / "colab_config.yaml"

# Artifact directories
ARTIFACTS_DIR = Path("artifacts")
RAW_DATA_DIR = ARTIFACTS_DIR / "raw_data"
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "processed_data"
INTERIM_DATA_DIR = ARTIFACTS_DIR / "interim_data"
VALIDATION_REPORT_DIR = ARTIFACTS_DIR / "validation_reports"
LOGS_DIR = ARTIFACTS_DIR / "logs"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
MODEL_REGISTRY_DIR = ARTIFACTS_DIR / "model_registry"
STATES_DIR = ARTIFACTS_DIR / "states"

# Data processing constants
DEFAULT_RANDOM_STATE = 42
MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 10
DEFAULT_TRAIN_SPLIT = 0.7
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15

# Label mappings
HUMAN_LABEL = 0
AI_LABEL = 1
LABEL_MAPPING = {"human": HUMAN_LABEL, "ai": AI_LABEL}
REVERSE_LABEL_MAPPING = {HUMAN_LABEL: "human", AI_LABEL: "ai"}

# Model constants
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-large"
DEFAULT_TOKENIZER_NAME = "microsoft/deberta-v3-large"
DEFAULT_MAX_LENGTH = 512
DEFAULT_NUM_LABELS = 2

# Training constants
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01

# LoRA constants
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGET_MODULES = [
    "query_proj",
    "key_proj",
    "value_proj",
    "dense",
    "output.dense",
]

# Evaluation constants
DEFAULT_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "precision_recall_auc",
]

# Validation thresholds
DEFAULT_MIN_ACCURACY = 0.85
DEFAULT_MIN_F1 = 0.85
DEFAULT_MIN_AUC = 0.85

# Device detection priority
DEVICE_PRIORITY = ["cuda", "mps", "cpu"]

# File extensions
MODEL_FILE_EXTENSIONS = [".bin", ".safetensors", ".pt", ".pth"]
CONFIG_FILE_EXTENSIONS = [".json", ".yaml", ".yml"]
TOKENIZER_FILE_EXTENSIONS = [".json", ".txt", ".model"]
