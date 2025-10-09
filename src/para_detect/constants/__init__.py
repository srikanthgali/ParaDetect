from pathlib import Path

BASE_CONFIG_DIR = Path("configs")
MODEL_CONFIG_DIR = Path("configs/model_configs")
DEPLOYMENT_CONFIG_DIR = Path("configs/deployment")

BASE_CONFIG_FILE_PATH = BASE_CONFIG_DIR / "config.yaml"
DEBERTA_CONFIG_FILE_PATH = MODEL_CONFIG_DIR / "deberta_config.yaml"
TRAINING_CONFIG_FILE_PATH = MODEL_CONFIG_DIR / "training_config.yaml"
AWS_CONFIG_FILE_PATH = DEPLOYMENT_CONFIG_DIR / "aws_config.yaml"
LOCAL_CONFIG_FILE_PATH = DEPLOYMENT_CONFIG_DIR / "local_config.yaml"
COLAB_CONFIG_FILE_PATH = DEPLOYMENT_CONFIG_DIR / "colab_config.yaml"
