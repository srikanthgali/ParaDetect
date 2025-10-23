from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List


@dataclass(frozen=True)
class S3Config:
    """Configuration entity for AWS S3 operations and layout."""

    # Core bucket settings
    bucket_name: str
    region: str
    storage_class: str

    # Bucket features
    enable_versioning: bool
    enable_lifecycle: bool

    # Upload/transfer settings (values in MB where applicable)
    multipart_threshold: int  # Files larger than this use multipart upload
    multipart_chunksize: int  # Size of each part in multipart upload
    max_concurrency: int

    # Retry settings
    max_retries: int
    retry_mode: str

    # Folder structure inside the bucket
    folders: Dict[str, str] = field(default_factory=dict)

    # Lifecycle management
    lifecycle_rules: Optional[List[Dict[str, Any]]] = None

    # Optional encryption settings (not required, but commonly used)
    server_side_encryption: Optional[str] = None  # e.g., "AES256" or "aws:kms"
    kms_key_id: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Bucket and region sanity
        if not self.bucket_name or not isinstance(self.bucket_name, str):
            raise ValueError("bucket_name must be a non-empty string")
        if not self.region or not isinstance(self.region, str):
            raise ValueError("region must be a non-empty string")

        # Storage class validation
        valid_storage = {"STANDARD", "STANDARD_IA", "GLACIER", "DEEP_ARCHIVE"}
        if self.storage_class not in valid_storage:
            raise ValueError(f"storage_class must be one of {sorted(valid_storage)}")

        # Retry mode validation
        valid_retry_modes = {"standard", "adaptive", "legacy"}
        if self.retry_mode not in valid_retry_modes:
            raise ValueError(f"retry_mode must be one of {sorted(valid_retry_modes)}")

        # Concurrency and sizes
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.multipart_chunksize < 5:
            # AWS requires multipart chunk size >= 5MB
            raise ValueError("multipart_chunksize must be >= 5 MB")
        if self.multipart_threshold < self.multipart_chunksize:
            raise ValueError("multipart_threshold must be >= multipart_chunksize (MB)")

        # Normalize folders to ensure trailing slash
        normalized_folders = {}
        for k, v in (self.folders or {}).items():
            if not isinstance(v, str):
                raise ValueError(f"folders['{k}'] must be a string path")
            normalized = v if v.endswith("/") else f"{v}/"
            # Remove any leading slash to keep keys relative within bucket
            normalized = normalized.lstrip("/")

        object.__setattr__(self, "folders", normalized_folders)

        # Lifecycle rules normalization
        rules = self.lifecycle_rules if self.lifecycle_rules is not None else []
        if not isinstance(rules, list):
            raise ValueError("lifecycle_rules must be a list of rule dictionaries")
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(f"lifecycle_rules[{idx}] must be a dictionary")
            # Minimal structural validation (optional fields are allowed)
            if "id" not in rule or "status" not in rule:
                raise ValueError(
                    f"lifecycle_rules[{idx}] must include 'id' and 'status' keys"
                )
            if rule["status"] not in {"Enabled", "Disabled"}:
                raise ValueError(
                    f"lifecycle_rules[{idx}].status must be 'Enabled' or 'Disabled'"
                )
        object.__setattr__(self, "lifecycle_rules", rules)

        # Encryption sanity
        if self.kms_key_id is not None and not self.server_side_encryption:
            raise ValueError("kms_key_id provided but server_side_encryption is None")

    @property
    def multipart_threshold_bytes(self) -> int:
        """Multipart threshold converted to bytes."""
        return int(self.multipart_threshold) * 1024 * 1024

    @property
    def multipart_chunksize_bytes(self) -> int:
        """Multipart chunk size converted to bytes."""
        return int(self.multipart_chunksize) * 1024 * 1024
