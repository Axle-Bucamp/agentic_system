"""
Logging configuration for the Agentic Trading System.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from redis import Redis

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

        handler_factory = getattr(logfire, "loguru_handler", None)
        if callable(handler_factory):
            logger.add(handler_factory(), level=settings.log_level, enqueue=True)
            logger.info("Logfire loguru handler installed")
        else:
            logger.warning("Logfire loguru handler unavailable; falling back to std logging bridge")

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


class RedisLogSink:
    """Loguru sink that writes log records to Redis capped list."""

    def __init__(self, key: str, max_entries: int) -> None:
        self.key = key
        self.max_entries = max_entries
        try:
            self.client = Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
            )
            # Probe connection
            self.client.ping()
            self._available = True
        except Exception as exc:
            logger.warning("Redis log sink unavailable: %s", exc)
            self._available = False

    def write(self, message: Any) -> None:
        if not self._available:
            return
        try:
            record: Dict[str, Any] = message.record
            payload = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "name": record["name"],
                "function": record["function"],
                "line": record["line"],
                "extra": record.get("extra", {}),
            }
            self.client.rpush(self.key, json.dumps(payload))
            self.client.ltrim(self.key, -self.max_entries, -1)
        except Exception as exc:
            # Downgrade to debug to avoid recursive logging
            logger.debug("Failed to push log entry to Redis: %s", exc)


def setup_logging():
    """Configure loguru logger with appropriate settings."""

    logger.remove()
    logger.configure(extra={"cluster": settings.cluster_name, "instance": settings.agent_instance_id})

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

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

    error_log_path = _resolve_log_path(log_path.parent / "errors.log")
    logger.add(
        str(error_log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="50 MB",
        retention="90 days",
        compression="zip",
    )

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

    if settings.log_redis_enabled:
        redis_sink = RedisLogSink(settings.log_redis_list_key, settings.log_redis_max_entries)
        logger.add(redis_sink, level="INFO", enqueue=False)

    logger.info("Logging initialized - Level: %s, File: %s", settings.log_level, log_path)

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


log = setup_logging()

