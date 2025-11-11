"""
CAMEL toolkit exposing the weight review pipeline as a callable tool.

Agents can invoke the tool to trigger an immediate weight review cycle and
retrieve the resulting snapshot (weights, explanation, metrics).  This allows
the CAMEL workforce to reason about agent weighting adjustments without
waiting for the scheduled memory agent cycle.
"""

from __future__ import annotations

from typing import Any, Dict

from core.pipelines.review_pipeline import WeightReviewPipeline, REDIS_REVIEW_KEY
from core.redis_client import redis_client

try:  # pragma: no cover - optional dependency
    from camel.toolkits import FunctionTool
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FunctionTool = None  # type: ignore
    CAMEL_TOOLS_AVAILABLE = False


class ReviewPipelineToolkit:
    """Expose the weight review pipeline via CAMEL function tools."""

    def __init__(self, redis_client_override=None):
        self.redis = redis_client_override or redis_client

    async def initialize(self) -> None:
        """Placeholder to mirror other toolkit interfaces."""

    def get_run_review_tool(self):
        if not CAMEL_TOOLS_AVAILABLE or FunctionTool is None:
            raise ImportError("CAMEL function tools not installed")

        async def run_weight_review(trigger: str = "camel_tool") -> Dict[str, Any]:
            """
            Execute the weight review pipeline and return the latest snapshot.

            Args:
                trigger: Label describing the caller (defaults to 'camel_tool').

            Returns:
                Dictionary containing updated weights and the stored snapshot.
            """

            pipeline = WeightReviewPipeline(self.redis)
            weights = await pipeline.run(trigger=trigger)
            snapshot = await self.redis.get_json(REDIS_REVIEW_KEY)  # type: ignore[attr-defined]
            return {
                "success": True,
                "weights": weights,
                "snapshot": snapshot,
            }

        run_weight_review.__name__ = "run_weight_review"
        tool = FunctionTool(run_weight_review)

        try:  # pragma: no cover - schema normalisation
            schema = dict(tool.get_openai_tool_schema())
        except Exception:
            schema = {
                "type": "function",
                "function": {
                    "name": "run_weight_review",
                    "description": run_weight_review.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "trigger": {
                                "type": "string",
                                "description": "Label describing the caller or reason for review.",
                            }
                        },
                        "required": [],
                    },
                },
            }

        function_schema = schema.setdefault("function", {})
        function_schema["name"] = "run_weight_review"
        function_schema.setdefault("description", run_weight_review.__doc__ or "")
        tool.openai_tool_schema = schema
        return tool

    def get_all_tools(self):
        """Return the tool collection provided by this toolkit."""
        return [self.get_run_review_tool()]


__all__ = ["ReviewPipelineToolkit"]


