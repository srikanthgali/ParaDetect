from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class LoggerConfig:
    """Enhanced logger configuration entity matching config.yaml structure"""
    level: str
    format: str
    log_dir: str
    log_file: Optional[str] = None

    # Component-specific logging
    loggers: Optional[Dict[str, Dict[str, Any]]] = None

    # Log rotation settings
    rotation: Optional[Dict[str, Any]] = None

    # Structured logging options
    structured: bool = False
    json_format: bool = False

    def __post_init__(self):
        """Auto-generate log_file from log_dir if not provided"""
        if not self.log_file and self.log_dir:
            from pathlib import Path
            log_path = Path(self.log_dir) / "para_detect.log"
            object.__setattr__(self, 'log_file', str(log_path))


@dataclass(frozen=True)
class ComponentLoggerConfig:
    """Configuration for individual component loggers"""
    name: str
    level: str
    handlers: List[str]
    propagate: bool = False


@dataclass(frozen=True)
class HandlerConfig:
    """Configuration for log handlers"""
    name: str
    type: str  # 'console', 'file', 'rotating_file', 'syslog', etc.
    level: str
    format: str
    filename: Optional[str] = None
    max_bytes: Optional[int] = None
    backup_count: Optional[int] = None
