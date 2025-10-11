"""
Basic logging usage examples
"""

# Method 1: Simple logging (most common)
from para_detect import get_logger

logger = get_logger("para_detect.api")
logger.info("API server starting")
logger.debug("Debug information")
logger.warning("This is a warning")
logger.error("This is an error")
