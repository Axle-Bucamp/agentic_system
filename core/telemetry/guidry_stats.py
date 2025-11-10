"""
Telemetry helpers capturing Guidry Cloud API behaviour for CAMEL agents.

The workforce relies heavily on the forecasting service provided by
guidry-cloud.com.  This module records rolling request statistics so that
agents can reason about latency, error rates, and asset availability
issues.  The aggregated metrics are exposed to CAMEL toolkits via
`guidry_cloud_stats.summary()`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from threading import Lock
from time import time
from typing import Deque, Dict, List, Optional


@dataclass
class RequestSample:
    """A single forecasting API request sample."""

    endpoint: str
    status: int
    duration_secs: float
    error: Optional[str]
    timestamp: float


class GuidryCloudStats:
    """Thread-safe collector of forecasting API request statistics."""

    def __init__(self, history_size: int = 200) -> None:
        self._history: Deque[RequestSample] = deque(maxlen=history_size)
        self._lock = Lock()

        self._total_calls = 0
        self._success_calls = 0
        self._rate_limit_hits = 0
        self._timeouts = 0
        self._disabled_assets: Dict[str, int] = {}
        self._last_error: Optional[str] = None

    def record_success(self, endpoint: str, status: int, duration_secs: float) -> None:
        sample = RequestSample(
            endpoint=endpoint,
            status=status,
            duration_secs=duration_secs,
            error=None,
            timestamp=time(),
        )
        with self._lock:
            self._history.append(sample)
            self._total_calls += 1
            self._success_calls += 1

    def record_failure(
        self,
        endpoint: str,
        status: int,
        duration_secs: float,
        error: Optional[str] = None,
        *,
        timeout: bool = False,
        rate_limited: bool = False,
        disabled_asset: Optional[str] = None,
    ) -> None:
        sample = RequestSample(
            endpoint=endpoint,
            status=status,
            duration_secs=duration_secs,
            error=error,
            timestamp=time(),
        )
        with self._lock:
            self._history.append(sample)
            self._total_calls += 1
            self._last_error = error
            if rate_limited:
                self._rate_limit_hits += 1
            if timeout:
                self._timeouts += 1
            if disabled_asset:
                ticker = disabled_asset.upper()
                self._disabled_assets[ticker] = self._disabled_assets.get(ticker, 0) + 1

    def summary(self) -> Dict[str, object]:
        with self._lock:
            history = list(self._history)
            total = self._total_calls
            success = self._success_calls
            latency_values = [sample.duration_secs for sample in history]
            last_five = history[-5:]

        avg_latency = mean(latency_values) if latency_values else None
        p95_latency = _percentile(latency_values, 95) if latency_values else None

        recent_events: List[Dict[str, object]] = []
        for sample in last_five:
            recent_events.append(
                {
                    "endpoint": sample.endpoint,
                    "status": sample.status,
                    "duration": round(sample.duration_secs, 4),
                    "error": sample.error,
                    "age_seconds": round(time() - sample.timestamp, 2),
                }
            )

        return {
            "total_calls": total,
            "success_calls": success,
            "success_rate": round(success / total, 4) if total else None,
            "average_latency": round(avg_latency, 4) if avg_latency is not None else None,
            "p95_latency": round(p95_latency, 4) if p95_latency is not None else None,
            "rate_limit_hits": self._rate_limit_hits,
            "timeouts": self._timeouts,
            "disabled_assets": dict(self._disabled_assets),
            "last_error": self._last_error,
            "recent_samples": recent_events,
        }

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._total_calls = 0
            self._success_calls = 0
            self._rate_limit_hits = 0
            self._timeouts = 0
            self._disabled_assets.clear()
            self._last_error = None


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


guidry_cloud_stats = GuidryCloudStats()

__all__ = ["GuidryCloudStats", "RequestSample", "guidry_cloud_stats"]

