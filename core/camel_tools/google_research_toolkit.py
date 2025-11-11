"""
Toolkit wrapping CAMEL's SearchToolkit helpers (Google/Wikipedia) and exposing
them as FunctionTool instances for the trading workforce.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.logging import log

try:  # pragma: no cover - optional dependency
    from camel.toolkits import FunctionTool
    from camel.toolkits.search_toolkit import SearchToolkit
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FunctionTool = None  # type: ignore
    SearchToolkit = None  # type: ignore
    CAMEL_TOOLS_AVAILABLE = False


class GoogleResearchToolkit:
    """Expose search helpers (Google/Wikipedia) as FunctionTool objects."""

    def __init__(self) -> None:
        self._search_toolkit: SearchToolkit | None = None

    async def initialize(self) -> None:
        if not CAMEL_TOOLS_AVAILABLE or SearchToolkit is None:
            raise ImportError("camel.toolkits.search_toolkit.SearchToolkit is not available.")
        if self._search_toolkit is None:
            self._search_toolkit = SearchToolkit()
            log.info("SearchToolkit initialised for GoogleResearchToolkit.")

    def _wrap(self, fn_name: str, tool_name: str, description: str) -> FunctionTool:
        if not CAMEL_TOOLS_AVAILABLE or FunctionTool is None or SearchToolkit is None:
            raise ImportError("CAMEL toolkits are not available.")
        if self._search_toolkit is None:
            raise RuntimeError("SearchToolkit not initialised. Call initialize() first.")

        fn = getattr(self._search_toolkit, fn_name)
        tool = FunctionTool(fn)
        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
        tool.openai_tool_schema = schema
        return tool

    def get_tools(self) -> List[FunctionTool]:
        """Return the bundled Google/Wikipedia search tools."""
        google_tool = self._wrap(
            fn_name="search_google",
            tool_name="search_google",
            description="Search Google and return the top results.",
        )
        wiki_tool = self._wrap(
            fn_name="search_wiki",
            tool_name="search_wikipedia",
            description="Search Wikipedia and return summary snippets.",
        )
        return [google_tool, wiki_tool]

    def get_all_tools(self) -> List[Any]:
        return self.get_tools()


__all__ = ["GoogleResearchToolkit"]


