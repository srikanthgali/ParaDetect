import yaml
from box import ConfigBox
from box.exceptions import BoxValueError, BoxKeyError
from typing import Any, Optional
from ensure import ensure_annotations
from pathlib import Path
import logging
import json
import numpy as np
import torch
from para_detect.constants import DEVICE_PRIORITY
from para_detect.core.exceptions import DeviceError


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


@ensure_annotations
def detect_device(
    device_preference: Optional[str] = None, logger: Optional[logging.Logger] = None
) -> torch.device:
    """
    Detect optimal device (CUDA -> MPS -> CPU) with an optional user preference.

    Args:
        device_preference: "auto" (or None), "cuda", "mps", or "cpu"
        logger: optional logger; falls back to module logger if not provided

    Returns:
        torch.device

    Raises:
        DeviceError: if detection fails unexpectedly
    """
    log = logger or logging.getLogger(__name__)
    try:
        pref = (device_preference or "").strip().lower()
        if pref and pref != "auto":
            # Respect explicit preference if available; otherwise warn and fall back
            try:
                device = torch.device(pref)
                if device.type == "cuda" and not torch.cuda.is_available():
                    log.warning(
                        "CUDA requested but not available, falling back to auto-detection"
                    )
                elif device.type == "mps" and not (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    log.warning(
                        "MPS requested but not available, falling back to auto-detection"
                    )
                else:
                    return device
            except Exception as e:
                log.warning(
                    f"Requested device '{device_preference}' is invalid: {e}. Falling back to auto-detection."
                )

        # Auto-detection by priority
        for device_type in DEVICE_PRIORITY:
            if device_type == "cuda" and torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                log.info(f"🚀 CUDA available: {name}")
                log.info(f"   GPU Memory: {props.total_memory / 1e9:.1f} GB")
                return torch.device("cuda")
            if (
                device_type == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                log.info("🍎 Using Apple Metal Performance Shaders (MPS)")
                return torch.device("mps")
            if device_type == "cpu":
                log.info("💻 Using CPU")
                return torch.device("cpu")

        # Final fallback
        return torch.device("cpu")

    except Exception as e:
        raise DeviceError(f"Failed to detect device: {str(e)}") from e


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable types to JSON-serializable formats."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif hasattr(obj, "__dict__"):
        # Handle custom objects
        return convert_to_serializable(obj.__dict__)
    else:
        return obj
