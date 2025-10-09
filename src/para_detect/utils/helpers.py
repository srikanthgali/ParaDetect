import yaml
from box import ConfigBox
from box.exceptions import BoxValueError, BoxKeyError
from typing import Any
from ensure import ensure_annotations
from pathlib import Path
import logging
import json


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path (Path): Path to the YAML file.

    Returns:
        ConfigBox: Parsed configuration.
    """
    try:
        with open(path, "r") as yaml_file:
            content: Any = yaml.safe_load(yaml_file)
            logging.info(f"YAML file: {path} loaded successfully")
        return ConfigBox(content)
    except BoxValueError as e:
        logging.error(f"Error reading YAML file: {e}")
        raise ValueError(f"Error in config values: {e}")
    except BoxKeyError as e:
        logging.error(f"Error reading YAML file: {e}")
        raise KeyError(f"Error in config keys: {e}")
    except Exception as e:
        logging.error(f"Error reading YAML file: {e}")
        raise e


@ensure_annotations
def create_directories(path: Path) -> None:
    """Create directories if they don't exist"""
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory created: {path}")


@ensure_annotations
def save_json(data: dict, path: Path) -> None:
    """Save data as JSON file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"JSON file saved: {path}")


@ensure_annotations
def load_json(path: Path) -> dict:
    """Load data from JSON file"""
    with open(path, "r") as f:
        data = json.load(f)
    logging.info(f"JSON file loaded: {path}")
    return data
