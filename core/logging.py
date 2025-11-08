"""
Logging configuration for the Agentic Trading System.
"""
import importlib
import logging
import sys
from pathlib import Path

from loguru import logger

from core.config import settings


def _resolve_log_path(default_path: Path) -> Path:
    try:
        default_path.parent.mkdir(parents=True, exist_ok=True)
        return default_path
    except Exception:
        fallback_dir = Path.cwd() / "logs"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / default_path.name


def _configure_logfire() -> None:
    if not settings.logfire_token:
        return

    try:
        logfire = importlib.import_module("logfire")
        logfire.configure(
            token=settings.logfire_token,
            service_name=settings.app_name,
            environment=settings.environment,
        )

        # Attach Logfire handler to loguru
        handler_factory = getattr(logfire, "loguru_handler", None)
        if callable(handler_factory):
            logger.add(
                handler_factory(),
                level=settings.log_level,
                enqueue=True,
            )
            logger.info("Logfire loguru handler installed")
        else:
            logger.warning("Logfire loguru handler unavailable; falling back to std logging bridge")

        # Ensure std logging is also instrumented for third-party libs
        try:
            configure_mod = importlib.import_module("logfire.integrations.logging")
            configure_logfire_logging = getattr(configure_mod, "configure_logging", None)
            if callable(configure_logfire_logging):
                configure_logfire_logging()
        except Exception as exc:
            logger.debug("Logfire logging integration fallback failed: %s", exc)
            instrument = getattr(logfire, "instrument_python_logging", None)
            if callable(instrument):
                instrument()
    except Exception as exc:
        logger.warning("Logfire integration disabled: %s", exc)


def setup_logging():
    """Configure loguru logger with appropriate settings."""

    # Remove default handler
    logger.remove()
    logger.configure(
        extra={
            "cluster": settings.cluster_name,
            "instance": settings.agent_instance_id,
        }
    )

    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # File handler for all logs
    base_log_path = Path(settings.log_file)
    log_path = _resolve_log_path(base_log_path)

    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
    )

    # Separate file for errors
    error_log_path = _resolve_log_path(log_path.parent / "errors.log")
    logger.add(
        str(error_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="zip",
    )

    # Separate file for trading decisions
    trading_log_path = _resolve_log_path(log_path.parent / "trading_decisions.log")
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
    portfolio_log_path = _resolve_log_path(log_path.parent / "portfolio_plans.log")
    logger.add(
        str(portfolio_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        filter=lambda record: record["extra"].get("PORTFOLIO_PLAN"),
        rotation="50 MB",
        retention="365 days",
        compression="zip",
    )

    logger.info(
        "Logging initialized - Level: %s, File: %s",
        settings.log_level,
        log_path,
    )

    # Bridge standard logging to loguru for third-party modules
    class LoguruHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except Exception:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(LoguruHandler())

    _configure_logfire()
    logger.bind(cluster=settings.cluster_name, instance=settings.agent_instance_id).info(
        "Log context bound for cluster=%s instance=%s",
        settings.cluster_name,
        settings.agent_instance_id,
    )
    return logger


# Initialize logger
log = setup_logging()

