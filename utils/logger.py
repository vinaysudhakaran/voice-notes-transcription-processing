import sys
from loguru import logger
from datetime import datetime
from pathlib import Path


LOG_LEVEL: str = "DEBUG"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"

logger.remove()

LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{name}:{function}</cyan> - "
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    colorize=True,
    backtrace=True,
    diagnose=True,
)


logger.add(
    LOG_FILE,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{function}:{line} - {message}",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    backtrace=True,
    diagnose=True,
)


logger.add(
    ERROR_LOG_FILE,
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{function}:{line} - {message}\n{exception}",
    rotation="10 MB",
    retention="90 days",
    compression="zip",
    backtrace=True,
    diagnose=True,
)


__all__ = ["logger"]
