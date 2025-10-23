"""
AWS S3 utility module for ParaDetect project.
Provides comprehensive S3 operations with proper error handling and logging.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import hashlib
import mimetypes

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

from para_detect.core.logger import get_logger
from para_detect.core.config_manager import ConfigurationManager
from para_detect.core.exceptions import ConfigurationError


class S3Manager:
    """
    AWS S3 utility class for managing model artifacts, checkpoints, and inference results.

    This class provides a comprehensive interface for S3 operations including:
    - File uploads/downloads with progress tracking
    - Batch operations for multiple files
    - Automatic retry and error handling
    - Metadata management
    - Folder structure management
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize S3Manager with configuration.

        Args:
            config_manager: Configuration manager instance. If None, creates a new one.
        """
        self.logger = get_logger(self.__class__.__name__)

        # Initialize configuration
        if config_manager is None:
            self.config_manager = ConfigurationManager()
        else:
            self.config_manager = config_manager

        # Detect environment
        self._detect_environment()

        # Load AWS credentials
        self._load_aws_credentials()

        # Load S3 configuration
        self._load_s3_config()

        # Initialize S3 client
        self._initialize_s3_client()

    def _detect_environment(self):
        """Detect the deployment environment."""
        if self._is_ec2_instance():
            # Running on EC2
            self.environment = "ec2"
            self.logger.info("Detected EC2 environment")
        else:
            # Local development
            self.environment = "local"
            self.logger.info("Detected local development environment")

    def _is_ec2_instance(self):
        """Check if running on EC2 instance."""
        try:
            import requests

            # Try to access EC2 instance metadata
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-id", timeout=1
            )
            return response.status_code == 200
        except:
            return False

    def _load_aws_credentials(self):
        """
        Load AWS credentials using the standard AWS credential chain.
        No explicit credential management - let boto3 handle it.
        """
        try:
            # Create session using default credential chain
            self.session = boto3.Session()

            # Determine credential source for logging
            self.credential_source = self._detect_credential_source()

            self.logger.info(f"Using AWS credentials from: {self.credential_source}")

        except Exception as e:
            self.logger.error(f"Failed to initialize AWS session: {e}")
            raise ConfigurationError(f"AWS session initialization error: {e}")

    def _detect_credential_source(self) -> str:
        """Detect which credential source is being used."""
        if self.environment in ["ec2", "ecs", "lambda", "eks"]:
            return "iam_role"
        elif all(
            k in os.environ for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ):
            return "environment_variables"
        elif Path.home().joinpath(".aws", "credentials").exists():
            return "aws_credentials_file"
        else:
            return "unknown"

    def _load_s3_config(self) -> None:
        """Load S3 configuration from config files."""
        try:
            self.s3_config = self.config_manager.get_s3_config()
            self.bucket_name = self.s3_config.bucket_name
            self.region = self.s3_config.region
            self.folders = dict(self.s3_config.folders or {})
            self.logger.info(
                f"S3 configuration loaded: bucket={self.s3_config.bucket_name}, region={self.s3_config.region}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load S3 configuration: {e}")
            raise ConfigurationError(f"S3 configuration error: {e}")

    def _initialize_s3_client(self) -> None:
        """Initialize S3 client with proper configuration."""
        try:
            # Configure transfer settings for better performance
            transfer_config = TransferConfig(
                multipart_threshold=self.s3_config.multipart_threshold_bytes,
                max_concurrency=self.s3_config.max_concurrency,
                multipart_chunksize=self.s3_config.multipart_chunksize_bytes,
                use_threads=True,
            )

            # Configure boto3 client
            config = Config(
                region_name=self.s3_config.region,
                retries={
                    "max_attempts": self.s3_config.max_retries,
                    "mode": self.s3_config.retry_mode,
                },
                max_pool_connections=50,
            )

            # Initialize S3 client
            self.s3_client = self.session.client("s3", config=config)
            self.transfer_config = transfer_config

            # Test credentials
            self._test_credentials()

            self.logger.info("S3 client initialized successfully")

        except (NoCredentialsError, PartialCredentialsError) as e:
            self.logger.error(f"AWS credentials not found or incomplete: {e}")
            raise ConfigurationError(
                "AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "environment variables, configure AWS credentials file, or ensure IAM role is attached."
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            raise ConfigurationError(f"S3 client initialization error: {e}")

    def _test_credentials(self) -> None:
        """Test AWS credentials by making a test call."""
        try:
            # Test with STS get-caller-identity (lightweight call)
            sts_client = self.session.client("sts", region_name=self.region)
            identity = sts_client.get_caller_identity()

            self.logger.info(
                f"AWS credentials validated for account: {identity.get('Account')}"
            )
            self.logger.debug(f"User ARN: {identity.get('Arn')}")

            # Test S3 access
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.logger.info(f"S3 bucket '{self.bucket_name}' is accessible")
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    self.logger.warning(
                        f"S3 bucket '{self.bucket_name}' does not exist"
                    )
                elif error_code == "403":
                    self.logger.warning(f"No access to S3 bucket '{self.bucket_name}'")
                else:
                    self.logger.warning(f"S3 bucket test failed: {error_code}")

        except Exception as e:
            self.logger.error(f"Credential validation failed: {e}")
            raise

    def get_credential_info(self) -> Dict[str, Any]:
        """Get information about current credentials (for debugging)."""
        try:
            sts_client = self.session.client("sts", region_name=self.region)
            identity = sts_client.get_caller_identity()

            info = {
                "account": identity.get("Account"),
                "user_id": identity.get("UserId"),
                "arn": identity.get("Arn"),
                "environment": self.environment,
                "region": self.region,
                "bucket_name": self.bucket_name,
                "credential_source": self.credential_source,
            }

            return info

        except Exception as e:
            self.logger.error(f"Failed to get credential info: {e}")
            return {"error": str(e)}

    def create_bucket_if_not_exists(self) -> bool:
        """Create S3 bucket if it doesn't exist."""
        try:
            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.logger.info(f"Bucket '{self.bucket_name}' already exists")
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise e

            # Create bucket
            self.s3_client.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={"LocationConstraint": self.region},
            )

            self.logger.info(f"Created S3 bucket: {self.bucket_name}")

            # Enable versioning if configured
            if self.s3_config.enable_versioning:
                self.s3_client.put_bucket_versioning(
                    Bucket=self.bucket_name,
                    VersioningConfiguration={"Status": "Enabled"},
                )
                self.logger.info("Enabled versioning on S3 bucket")

            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "BucketAlreadyOwnedByYou":
                self.logger.info(f"Bucket '{self.bucket_name}' already owned by you")
                return True
            else:
                self.logger.error(f"Failed to create bucket: {e}")
                return False
        except Exception as e:
            self.logger.error(f"Unexpected error creating bucket: {e}")
            return False

    def _get_s3_key(self, s3_path: str, folder_type: Optional[str] = None) -> str:
        """
        Generate S3 key with proper folder structure.

        Args:
            s3_path: Relative path within the bucket
            folder_type: Type of folder (checkpoints, artifacts, logs, results, models, datasets)

        Returns:
            Complete S3 key
        """
        if folder_type and folder_type in self.folders:
            prefix = self.folders[folder_type]
            return f"{prefix}{s3_path.lstrip('/')}"
        return s3_path.lstrip("/")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def upload_file(
        self,
        local_path: Union[str, Path],
        s3_path: str,
        folder_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path
            s3_path: S3 object key (relative path)
            folder_type: Folder type for automatic prefixing
            metadata: Additional metadata to store with the file
            progress_callback: Optional callback for upload progress

        Returns:
            True if upload successful, False otherwise
        """
        local_path = Path(local_path)

        if not local_path.exists():
            self.logger.error(f"Local file not found: {local_path}")
            return False

        try:
            s3_key = self._get_s3_key(s3_path, folder_type)

            # Prepare metadata
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            # Add content type if available
            content_type, _ = mimetypes.guess_type(str(local_path))
            if content_type:
                extra_args["ContentType"] = content_type

            # Add file hash as metadata
            file_hash = self._calculate_file_hash(local_path)
            if "Metadata" not in extra_args:
                extra_args["Metadata"] = {}
            extra_args["Metadata"]["file_hash"] = file_hash
            extra_args["Metadata"]["upload_timestamp"] = datetime.utcnow().isoformat()

            # Upload file
            self.logger.info(
                f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}"
            )

            if progress_callback:
                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs=extra_args,
                    Config=self.transfer_config,
                    Callback=progress_callback,
                )
            else:
                self.s3_client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    ExtraArgs=extra_args,
                    Config=self.transfer_config,
                )

            self.logger.info(f"Successfully uploaded {local_path} to S3")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            self.logger.error(f"Failed to upload file to S3 (Error: {error_code}): {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during file upload: {e}")
            return False

    def download_file(
        self,
        s3_path: str,
        local_path: Union[str, Path],
        folder_type: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> bool:
        """
        Download a file from S3.

        Args:
            s3_path: S3 object key (relative path)
            local_path: Local destination path
            folder_type: Folder type for automatic prefixing
            progress_callback: Optional callback for download progress

        Returns:
            True if download successful, False otherwise
        """
        local_path = Path(local_path)

        try:
            s3_key = self._get_s3_key(s3_path, folder_type)

            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            self.logger.info(
                f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}"
            )

            if progress_callback:
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    str(local_path),
                    Config=self.transfer_config,
                    Callback=progress_callback,
                )
            else:
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    str(local_path),
                    Config=self.transfer_config,
                )

            self.logger.info(f"Successfully downloaded file to {local_path}")
            return True

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                self.logger.error(
                    f"File not found in S3: s3://{self.bucket_name}/{s3_key}"
                )
            else:
                self.logger.error(
                    f"Failed to download file from S3 (Error: {error_code}): {e}"
                )
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during file download: {e}")
            return False

    def list_files(
        self, prefix: str = "", folder_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List files in S3 bucket with given prefix.

        Args:
            prefix: S3 key prefix to filter files
            folder_type: Folder type for automatic prefixing

        Returns:
            List of file information dictionaries
        """
        try:
            s3_prefix = self._get_s3_key(prefix, folder_type)

            self.logger.info(
                f"Listing files with prefix: s3://{self.bucket_name}/{s3_prefix}"
            )

            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=s3_prefix
            )

            files = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    files.append(
                        {
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"],
                            "etag": obj["ETag"].strip('"'),
                        }
                    )

            self.logger.info(f"Found {len(files)} files")
            return files

        except ClientError as e:
            self.logger.error(f"Failed to list files in S3: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during file listing: {e}")
            return []

    def delete_file(self, s3_path: str, folder_type: Optional[str] = None) -> bool:
        """
        Delete a file from S3.

        Args:
            s3_path: S3 object key (relative path)
            folder_type: Folder type for automatic prefixing

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            s3_key = self._get_s3_key(s3_path, folder_type)

            self.logger.info(f"Deleting file: s3://{self.bucket_name}/{s3_key}")

            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)

            self.logger.info(f"Successfully deleted file from S3")
            return True

        except ClientError as e:
            self.logger.error(f"Failed to delete file from S3: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during file deletion: {e}")
            return False

    def upload_directory(
        self,
        local_dir: Union[str, Path],
        s3_prefix: str,
        folder_type: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """
        Upload an entire directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix for uploaded files
            folder_type: Folder type for automatic prefixing
            exclude_patterns: List of patterns to exclude

        Returns:
            Dictionary mapping local file paths to upload success status
        """
        local_dir = Path(local_dir)
        exclude_patterns = exclude_patterns or []
        results = {}

        if not local_dir.exists() or not local_dir.is_dir():
            self.logger.error(
                f"Local directory not found or not a directory: {local_dir}"
            )
            return results

        # Find all files to upload
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Check exclusion patterns
                relative_path = file_path.relative_to(local_dir)
                if any(pattern in str(relative_path) for pattern in exclude_patterns):
                    continue

                # Generate S3 key
                s3_file_path = f"{s3_prefix.rstrip('/')}/{relative_path}".replace(
                    "\\", "/"
                )

                # Upload file
                success = self.upload_file(file_path, s3_file_path, folder_type)
                results[str(file_path)] = success

        successful_uploads = sum(1 for success in results.values() if success)
        self.logger.info(
            f"Directory upload completed: {successful_uploads}/{len(results)} files uploaded"
        )

        return results

    def sync_directory(
        self,
        local_dir: Union[str, Path],
        s3_prefix: str,
        folder_type: Optional[str] = None,
        delete_extra: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync local directory with S3 prefix.

        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix to sync with
            folder_type: Folder type for automatic prefixing
            delete_extra: Whether to delete files in S3 that don't exist locally

        Returns:
            Sync results summary
        """
        local_dir = Path(local_dir)
        results = {"uploaded": 0, "skipped": 0, "deleted": 0, "errors": 0}

        if not local_dir.exists():
            self.logger.error(f"Local directory not found: {local_dir}")
            return results

        # Get existing S3 files
        s3_files = {f["key"]: f for f in self.list_files(s3_prefix, folder_type)}
        s3_prefix_full = self._get_s3_key(s3_prefix, folder_type)

        # Upload/update local files
        local_files_processed = set()
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix_full}{relative_path}".replace("\\", "/")
                local_files_processed.add(s3_key)

                # Check if file needs uploading
                needs_upload = True
                if s3_key in s3_files:
                    local_hash = self._calculate_file_hash(file_path)
                    s3_etag = s3_files[s3_key]["etag"]
                    if local_hash == s3_etag:
                        needs_upload = False
                        results["skipped"] += 1

                if needs_upload:
                    success = self.upload_file(
                        file_path, str(relative_path), folder_type
                    )
                    if success:
                        results["uploaded"] += 1
                    else:
                        results["errors"] += 1

        # Delete extra S3 files if requested
        if delete_extra:
            for s3_key in s3_files:
                if s3_key not in local_files_processed:
                    relative_key = s3_key.replace(s3_prefix_full, "", 1)
                    if self.delete_file(relative_key, folder_type):
                        results["deleted"] += 1
                    else:
                        results["errors"] += 1

        self.logger.info(f"Directory sync completed: {results}")
        return results

    def get_file_metadata(
        self, s3_path: str, folder_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an S3 file.

        Args:
            s3_path: S3 object key (relative path)
            folder_type: Folder type for automatic prefixing

        Returns:
            File metadata dictionary or None if not found
        """
        try:
            s3_key = self._get_s3_key(s3_path, folder_type)

            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)

            return {
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "etag": response["ETag"].strip('"'),
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                self.logger.warning(f"File not found: s3://{self.bucket_name}/{s3_key}")
            else:
                self.logger.error(f"Failed to get file metadata: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting file metadata: {e}")
            return None

    def file_exists(self, s3_path: str, folder_type: Optional[str] = None) -> bool:
        """
        Check if a file exists in S3.

        Args:
            s3_path: S3 object key (relative path)
            folder_type: Folder type for automatic prefixing

        Returns:
            True if file exists, False otherwise
        """
        return self.get_file_metadata(s3_path, folder_type) is not None


# Convenience functions for common operations
def get_s3_manager() -> S3Manager:
    """Get a configured S3Manager instance."""
    return S3Manager()


def upload_model_checkpoint(
    checkpoint_path: Union[str, Path], model_name: str, version: str
) -> bool:
    """
    Upload model checkpoint to S3.

    Args:
        checkpoint_path: Local path to checkpoint
        model_name: Name of the model
        version: Version/timestamp of the checkpoint

    Returns:
        True if upload successful
    """
    s3_manager = get_s3_manager()
    s3_path = f"{model_name}/{version}/"

    if Path(checkpoint_path).is_dir():
        results = s3_manager.upload_directory(checkpoint_path, s3_path, "checkpoints")
        return all(results.values())
    else:
        filename = Path(checkpoint_path).name
        s3_file_path = f"{s3_path}{filename}"
        return s3_manager.upload_file(checkpoint_path, s3_file_path, "checkpoints")


def download_model_checkpoint(
    model_name: str, version: str, local_path: Union[str, Path]
) -> bool:
    """
    Download model checkpoint from S3.

    Args:
        model_name: Name of the model
        version: Version/timestamp of the checkpoint
        local_path: Local destination path

    Returns:
        True if download successful
    """
    s3_manager = get_s3_manager()
    s3_path = f"{model_name}/{version}/"

    # List files in checkpoint directory
    files = s3_manager.list_files(s3_path, "checkpoints")

    if not files:
        return False

    # Download all files
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for file_info in files:
        # Extract relative path
        full_key = file_info["key"]
        prefix = s3_manager._get_s3_key(s3_path, "checkpoints")
        relative_path = full_key.replace(prefix, "", 1).lstrip("/")

        local_file_path = local_path / relative_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        if s3_manager.download_file(relative_path, local_file_path, "checkpoints"):
            success_count += 1

    return success_count == len(files)
