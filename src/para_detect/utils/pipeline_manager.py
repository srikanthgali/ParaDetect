"""
Pipeline Management Utility

Provides command-line tools for managing ParaDetect pipelines:
- Force unlock pipelines
- Clean old states
- Show pipeline statuses
- Pipeline health checks
"""

import json
import fcntl
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import asdict

# Use TYPE_CHECKING for type hints to avoid circular imports
if TYPE_CHECKING:
    from para_detect.core.config_manager import ConfigurationManager

from para_detect.entities.pipeline_config import PipelineType, PipelineConfig
from para_detect.core.exceptions import ParaDetectException
from para_detect import get_logger


class PipelineManager:
    """
    Utility class for managing ParaDetect pipelines.

    Provides operations for:
    - Pipeline state management
    - Lock management
    - Health monitoring
    - Cleanup operations
    """

    def __init__(self, config_manager: Optional["ConfigurationManager"] = None):
        """Initialize pipeline manager."""
        # Lazy import to avoid circular dependency
        if config_manager is None:
            from para_detect.core.config_manager import ConfigurationManager

            self.config_manager = ConfigurationManager()
        else:
            self.config_manager = config_manager

        self.pipeline_config = self.config_manager.get_pipeline_config()
        self.logger = get_logger(self.__class__.__name__)

    def get_all_pipeline_types(self) -> List[PipelineType]:
        """Get all available pipeline types."""
        return list(PipelineType)

    def force_unlock_pipeline(self, pipeline_type: PipelineType) -> bool:
        """
        Force unlock a pipeline by removing its lock file.

        Args:
            pipeline_type: Type of pipeline to unlock

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            lock_file = self.pipeline_config.get_lock_file(pipeline_type)

            if not lock_file.exists():
                self.logger.info(f"🔓 No lock file found for {pipeline_type.value}")
                return True

            # Try to read lock info before removing
            try:
                with open(lock_file, "r") as f:
                    lock_info = json.load(f)
                    self.logger.info(
                        f"🔍 Lock info: PID {lock_info.get('pid')}, "
                        f"acquired at {lock_info.get('acquired_at')}"
                    )
            except Exception:
                self.logger.warning("Could not read lock file contents")

            # Remove lock file
            lock_file.unlink()
            self.logger.info(f"✅ Force unlocked {pipeline_type.value} pipeline")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to unlock {pipeline_type.value}: {str(e)}")
            return False

    def cleanup_old_states(
        self, retention_days: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Clean up old state files across all pipelines.

        Args:
            retention_days: Number of days to retain. Uses config default if None.

        Returns:
            Dict with cleanup statistics per pipeline
        """
        retention_days = retention_days or self.pipeline_config.state_retention_days
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        cleanup_stats = {}

        for pipeline_type in self.get_all_pipeline_types():
            try:
                state_file = self.pipeline_config.get_state_file(pipeline_type)
                archived_count = 0

                if state_file.exists():
                    file_time = datetime.fromtimestamp(state_file.stat().st_mtime)

                    if file_time < cutoff_date:
                        # Archive the file
                        archive_dir = state_file.parent / "archived"
                        archive_dir.mkdir(exist_ok=True)

                        archive_name = f"{state_file.stem}_archived_{file_time.strftime('%Y%m%d_%H%M%S')}.json"
                        archive_path = archive_dir / archive_name

                        state_file.rename(archive_path)
                        archived_count = 1

                        self.logger.info(
                            f"📦 Archived {pipeline_type.value} state to: {archive_path}"
                        )

                # Also clean old checkpoints
                checkpoint_dir = self.pipeline_config.get_checkpoint_dir(pipeline_type)
                checkpoint_cleaned = self._cleanup_old_checkpoints(
                    checkpoint_dir, cutoff_date
                )

                cleanup_stats[pipeline_type.value] = {
                    "states_archived": archived_count,
                    "checkpoints_cleaned": checkpoint_cleaned,
                }

            except Exception as e:
                self.logger.warning(f"Error cleaning {pipeline_type.value}: {str(e)}")
                cleanup_stats[pipeline_type.value] = {"error": str(e)}

        return cleanup_stats

    def _cleanup_old_checkpoints(
        self, checkpoint_dir: Path, cutoff_date: datetime
    ) -> int:
        """Clean old checkpoint files."""
        cleaned_count = 0

        if not checkpoint_dir.exists():
            return 0

        try:
            for checkpoint_file in checkpoint_dir.glob("*.json"):
                file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)

                if file_time < cutoff_date:
                    checkpoint_file.unlink()
                    cleaned_count += 1

        except Exception as e:
            self.logger.warning(f"Error cleaning checkpoints: {str(e)}")

        return cleaned_count

    def get_pipeline_status(self, pipeline_type: PipelineType) -> Dict[str, Any]:
        """
        Get detailed status for a specific pipeline.

        Args:
            pipeline_type: Type of pipeline to check

        Returns:
            Dict with pipeline status information
        """
        try:
            state_file = self.pipeline_config.get_state_file(pipeline_type)
            lock_file = self.pipeline_config.get_lock_file(pipeline_type)
            checkpoint_dir = self.pipeline_config.get_checkpoint_dir(pipeline_type)

            status = {
                "pipeline_type": pipeline_type.value,
                "state_file_exists": state_file.exists(),
                "state_file_path": str(state_file),
                "is_locked": lock_file.exists(),
                "lock_file_path": str(lock_file),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_count": 0,
                "last_modified": None,
                "state_data": None,
                "lock_info": None,
            }

            # Get state file info
            if state_file.exists():
                stat = state_file.stat()
                status["last_modified"] = datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat()
                status["file_size"] = stat.st_size

                # Try to load state data
                try:
                    with open(state_file, "r") as f:
                        status["state_data"] = json.load(f)
                except Exception as e:
                    status["state_load_error"] = str(e)

            # Get lock info
            if lock_file.exists():
                try:
                    with open(lock_file, "r") as f:
                        status["lock_info"] = json.load(f)
                except Exception as e:
                    status["lock_read_error"] = str(e)

            # Count checkpoints
            if checkpoint_dir.exists():
                status["checkpoint_count"] = len(list(checkpoint_dir.glob("*.json")))

            return status

        except Exception as e:
            return {"pipeline_type": pipeline_type.value, "error": str(e)}

    def get_all_pipeline_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all pipeline types."""
        statuses = {}

        for pipeline_type in self.get_all_pipeline_types():
            statuses[pipeline_type.value] = self.get_pipeline_status(pipeline_type)

        return statuses

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on pipeline system.

        Returns:
            Dict with health check results
        """
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "issues": [],
            "warnings": [],
            "config_valid": True,
            "directories_accessible": True,
            "pipelines": {},
        }

        try:
            # Check configuration
            config_dict = asdict(self.pipeline_config)
            health["config"] = config_dict

            # Check directory accessibility
            dirs_to_check = [
                self.pipeline_config.artifacts_dir,
                self.pipeline_config.checkpoints_dir,
            ]

            for dir_path in dirs_to_check:
                if not dir_path.exists():
                    health["warnings"].append(f"Directory does not exist: {dir_path}")
                elif not os.access(dir_path, os.W_OK):
                    health["issues"].append(f"Directory not writable: {dir_path}")
                    health["directories_accessible"] = False

            # Check each pipeline
            for pipeline_type in self.get_all_pipeline_types():
                pipeline_health = self._check_pipeline_health(pipeline_type)
                health["pipelines"][pipeline_type.value] = pipeline_health

                if pipeline_health.get("issues"):
                    health["issues"].extend(pipeline_health["issues"])
                if pipeline_health.get("warnings"):
                    health["warnings"].extend(pipeline_health["warnings"])

            # Determine overall health
            if health["issues"]:
                health["overall_health"] = "unhealthy"
            elif health["warnings"]:
                health["overall_health"] = "warning"

        except Exception as e:
            health["overall_health"] = "error"
            health["issues"].append(f"Health check failed: {str(e)}")

        return health

    def _check_pipeline_health(self, pipeline_type: PipelineType) -> Dict[str, Any]:
        """Check health of a specific pipeline."""
        pipeline_health = {
            "type": pipeline_type.value,
            "healthy": True,
            "issues": [],
            "warnings": [],
        }

        try:
            state_file = self.pipeline_config.get_state_file(pipeline_type)
            lock_file = self.pipeline_config.get_lock_file(pipeline_type)

            # Check for stale locks
            if lock_file.exists():
                try:
                    stat = lock_file.stat()
                    lock_age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)

                    if lock_age > timedelta(hours=2):  # Configurable threshold
                        pipeline_health["warnings"].append(
                            f"Potentially stale lock file (age: {lock_age})"
                        )

                    # Try to read lock info and check if process is still running
                    with open(lock_file, "r") as f:
                        lock_info = json.load(f)
                        pid = lock_info.get("pid")

                        if pid and not self._is_process_running(pid):
                            pipeline_health["issues"].append(
                                f"Lock file exists but process {pid} is not running"
                            )
                            pipeline_health["healthy"] = False

                except Exception as e:
                    pipeline_health["warnings"].append(
                        f"Could not validate lock: {str(e)}"
                    )

            # Check state file corruption
            if state_file.exists():
                try:
                    with open(state_file, "r") as f:
                        json.load(f)
                except json.JSONDecodeError:
                    pipeline_health["issues"].append(
                        "State file is corrupted (invalid JSON)"
                    )
                    pipeline_health["healthy"] = False

        except Exception as e:
            pipeline_health["issues"].append(f"Health check error: {str(e)}")
            pipeline_health["healthy"] = False

        return pipeline_health

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)  # Send null signal to check if process exists
            return True
        except OSError:
            return False

    def repair_pipeline(
        self, pipeline_type: PipelineType, auto_confirm: bool = False
    ) -> bool:
        """
        Attempt to repair a pipeline by cleaning stale locks and corrupted states.

        Args:
            pipeline_type: Pipeline to repair
            auto_confirm: Skip confirmation prompts

        Returns:
            bool: True if repair was successful
        """
        try:
            self.logger.info(
                f"🔧 Starting repair for {pipeline_type.value} pipeline..."
            )

            health = self._check_pipeline_health(pipeline_type)

            if health["healthy"] and not health["warnings"]:
                self.logger.info(
                    f"✅ {pipeline_type.value} pipeline is already healthy"
                )
                return True

            repairs_made = []

            # Handle stale locks
            lock_file = self.pipeline_config.get_lock_file(pipeline_type)
            if lock_file.exists():
                try:
                    with open(lock_file, "r") as f:
                        lock_info = json.load(f)
                        pid = lock_info.get("pid")

                    if pid and not self._is_process_running(pid):
                        if auto_confirm or self._confirm_action(
                            f"Remove stale lock for PID {pid}?"
                        ):
                            lock_file.unlink()
                            repairs_made.append("Removed stale lock file")

                except Exception as e:
                    self.logger.warning(f"Could not check lock file: {str(e)}")

            # Handle corrupted state files
            state_file = self.pipeline_config.get_state_file(pipeline_type)
            if state_file.exists():
                try:
                    with open(state_file, "r") as f:
                        json.load(f)
                except json.JSONDecodeError:
                    if auto_confirm or self._confirm_action(
                        f"Backup and reset corrupted state file?"
                    ):
                        # Backup corrupted file
                        backup_path = state_file.with_suffix(
                            f".corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                        )
                        state_file.rename(backup_path)
                        repairs_made.append(
                            f"Backed up corrupted state to {backup_path}"
                        )

            if repairs_made:
                self.logger.info(f"✅ Repair completed for {pipeline_type.value}:")
                for repair in repairs_made:
                    self.logger.info(f"  - {repair}")
                return True
            else:
                self.logger.info(f"ℹ️  No repairs needed for {pipeline_type.value}")
                return True

        except Exception as e:
            self.logger.error(f"❌ Repair failed for {pipeline_type.value}: {str(e)}")
            return False

    def _confirm_action(self, message: str) -> bool:
        """Get user confirmation for an action."""
        response = input(f"{message} (y/N): ").strip().lower()
        return response in ["y", "yes"]


def main():
    """Command-line interface for pipeline management."""
    parser = argparse.ArgumentParser(
        description="ParaDetect Pipeline Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Force unlock a specific pipeline
  python -m para_detect.utils.pipeline_manager --unlock data_pipeline

  # Clean old states (older than 30 days)
  python -m para_detect.utils.pipeline_manager --cleanup-states

  # Clean with custom retention (7 days)
  python -m para_detect.utils.pipeline_manager --cleanup-states --retention-days 7

  # Show all pipeline statuses
  python -m para_detect.utils.pipeline_manager --status-all

  # Show specific pipeline status
  python -m para_detect.utils.pipeline_manager --status data_pipeline

  # Perform health check
  python -m para_detect.utils.pipeline_manager --health-check

  # Repair a pipeline
  python -m para_detect.utils.pipeline_manager --repair data_pipeline
        """,
    )

    # Action arguments
    parser.add_argument(
        "--unlock",
        choices=[pt.value for pt in PipelineType],
        help="Force unlock a specific pipeline",
    )

    parser.add_argument(
        "--cleanup-states",
        action="store_true",
        help="Clean old state files across all pipelines",
    )

    parser.add_argument(
        "--retention-days",
        type=int,
        help="Number of days to retain states (used with --cleanup-states)",
    )

    parser.add_argument(
        "--status-all", action="store_true", help="Show status for all pipelines"
    )

    parser.add_argument(
        "--status",
        choices=[pt.value for pt in PipelineType],
        help="Show status for specific pipeline",
    )

    parser.add_argument(
        "--health-check", action="store_true", help="Perform system health check"
    )

    parser.add_argument(
        "--repair",
        choices=[pt.value for pt in PipelineType],
        help="Attempt to repair a pipeline",
    )

    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip confirmation prompts (use with caution)",
    )

    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()

    # Initialize manager
    try:
        manager = PipelineManager()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline manager: {str(e)}")
        sys.exit(1)

    # Execute actions
    try:
        if args.unlock:
            pipeline_type = PipelineType(args.unlock)
            success = manager.force_unlock_pipeline(pipeline_type)
            if not success:
                sys.exit(1)

        elif args.cleanup_states:
            stats = manager.cleanup_old_states(args.retention_days)
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print("🧹 Cleanup completed:")
                for pipeline, data in stats.items():
                    if "error" in data:
                        print(f"  {pipeline}: Error - {data['error']}")
                    else:
                        print(
                            f"  {pipeline}: {data['states_archived']} states archived, "
                            f"{data['checkpoints_cleaned']} checkpoints cleaned"
                        )

        elif args.status_all:
            statuses = manager.get_all_pipeline_statuses()
            if args.json:
                print(json.dumps(statuses, indent=2))
            else:
                print("📊 Pipeline Status Summary:")
                for pipeline, status in statuses.items():
                    if "error" in status:
                        print(f"  {pipeline}: ❌ Error - {status['error']}")
                    else:
                        lock_status = (
                            "🔒 Locked" if status["is_locked"] else "🔓 Unlocked"
                        )
                        state_status = (
                            "📄 Has State"
                            if status["state_file_exists"]
                            else "📄 No State"
                        )
                        print(
                            f"  {pipeline}: {lock_status}, {state_status}, "
                            f"{status['checkpoint_count']} checkpoints"
                        )

        elif args.status:
            pipeline_type = PipelineType(args.status)
            status = manager.get_pipeline_status(pipeline_type)
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"📊 {args.status} Pipeline Status:")
                if "error" in status:
                    print(f"  ❌ Error: {status['error']}")
                else:
                    print(
                        f"  State File: {'✅ Exists' if status['state_file_exists'] else '❌ Missing'}"
                    )
                    print(
                        f"  Lock Status: {'🔒 Locked' if status['is_locked'] else '🔓 Unlocked'}"
                    )
                    print(f"  Checkpoints: {status['checkpoint_count']}")
                    if status["last_modified"]:
                        print(f"  Last Modified: {status['last_modified']}")

        elif args.health_check:
            health = manager.health_check()
            if args.json:
                print(json.dumps(health, indent=2))
            else:
                status_emoji = {
                    "healthy": "✅",
                    "warning": "⚠️",
                    "unhealthy": "❌",
                    "error": "💥",
                }

                print(
                    f"🏥 System Health Check: {status_emoji.get(health['overall_health'], '❓')} {health['overall_health'].upper()}"
                )

                if health["issues"]:
                    print("\n❌ Issues:")
                    for issue in health["issues"]:
                        print(f"  - {issue}")

                if health["warnings"]:
                    print("\n⚠️  Warnings:")
                    for warning in health["warnings"]:
                        print(f"  - {warning}")

                print(f"\n📊 Pipeline Health:")
                for pipeline, pipeline_health in health["pipelines"].items():
                    status = (
                        "✅ Healthy" if pipeline_health["healthy"] else "❌ Unhealthy"
                    )
                    print(f"  {pipeline}: {status}")

        elif args.repair:
            pipeline_type = PipelineType(args.repair)
            success = manager.repair_pipeline(pipeline_type, args.auto_confirm)
            if not success:
                sys.exit(1)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
