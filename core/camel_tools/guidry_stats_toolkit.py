"""
CAMEL toolkit exposing Guidry Cloud forecasting API telemetry statistics.

Agents can call the provided FunctionTool to understand latency, success
rates, and disabled asset history for the forecasting service, allowing
them to reason about degraded conditions before attempting expensive tool
calls.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.telemetry.guidry_stats import guidry_cloud_stats

try:  # pragma: no cover - optional dependency
    from camel.toolkits import FunctionTool
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FunctionTool = None  # type: ignore
    CAMEL_TOOLS_AVAILABLE = False


class GuidryStatsToolkit:
    """Expose telemetry snapshots from the guidry-cloud forecasting API."""

    async def initialize(self) -> None:
        """Provided for interface parity; nothing to initialise."""

    def get_stats_tool(self):
        if not CAMEL_TOOLS_AVAILABLE or FunctionTool is None:
            raise ImportError("CAMEL function tools not installed")

        async def get_guidry_cloud_api_stats() -> Dict[str, Any]:
            """
            Return aggregated telemetry for guidry-cloud forecasting requests.

            Includes success rate, latency percentiles, rate limit counts,
            and tracked disabled assets.  Call this before intensive
            forecasting operations to understand the current reliability
            of the external service.
            """

            return guidry_cloud_stats.summary()

        get_guidry_cloud_api_stats.__name__ = "get_guidry_cloud_api_stats"
        tool = FunctionTool(get_guidry_cloud_api_stats)
        try:  # pragma: no cover - schema normalisation
            schema = dict(tool.get_openai_tool_schema())
        except Exception:
            schema = {
                "type": "function",
                "function": {
                    "name": "get_guidry_cloud_api_stats",
                    "description": get_guidry_cloud_api_stats.__doc__ or "",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

        function_schema = schema.setdefault("function", {})
        function_schema["name"] = "get_guidry_cloud_api_stats"
        function_schema.setdefault("description", get_guidry_cloud_api_stats.__doc__ or "")
        tool.openai_tool_schema = schema
        return tool

    def get_all_tools(self):
        """Return the complete tool collection provided by this toolkit."""
        return [self.get_stats_tool()]


__all__ = ["GuidryStatsToolkit"]

