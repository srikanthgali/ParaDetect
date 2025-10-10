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
