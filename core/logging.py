"""
Logging configuration for the Agentic Trading System.
"""
import sys
from pathlib import Path
from loguru import logger
from core.config import settings


def setup_logging():
    """Configure loguru logger with appropriate settings."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # File handler for all logs
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )
    
    # Separate file for errors
    error_log_path = log_path.parent / "errors.log"
    logger.add(
        str(error_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="zip",
    )
    
    # Separate file for trading decisions
    trading_log_path = log_path.parent / "trading_decisions.log"
    logger.add(
        str(trading_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: record["extra"].get("TRADE_DECISION"),
        rotation="50 MB",
        retention="365 days",
        compression="zip",
    )

    # File for orchestrator portfolio plans
    portfolio_log_path = log_path.parent / "portfolio_plans.log"
    logger.add(
        str(portfolio_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: record["extra"].get("PORTFOLIO_PLAN"),
        rotation="50 MB",
        retention="365 days",
        compression="zip",
    )
    
    logger.info(f"Logging initialized - Level: {settings.log_level}, File: {settings.log_file}")
    
    # Bridge standard logging to loguru for third-party modules
    try:
        import logging
        class LoguruHandler(logging.Handler):
            def emit(self, record):
                try:
                    level = logger.level(record.levelname).name
                except Exception:
                    level = record.levelno
                logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

        root_logger = logging.getLogger()
        root_logger.handlers = []
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(LoguruHandler())
    except Exception:
        # If bridging fails, continue with loguru only
        pass
    return logger


# Initialize logger
log = setup_logging()

