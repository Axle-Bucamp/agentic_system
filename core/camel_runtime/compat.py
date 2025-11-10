"""
Compatibility patches for third-party CAMEL components.
"""

from __future__ import annotations

from core.logging import log


def patch_search_toolkit() -> None:
    """Wrap legacy SearchToolkit tools in FunctionTool for newer CAMEL releases."""
    try:
        from camel.toolkits import FunctionTool
        from camel.toolkits.search_toolkit import SearchToolkit
    except ImportError:  # pragma: no cover - optional dependency
        return

    original_get_tools = getattr(SearchToolkit, "get_tools", None)
    if not callable(original_get_tools):
        return

    if getattr(SearchToolkit.get_tools, "_ats_patched", False):  # type: ignore[attr-defined]
        return

    def _patched_get_tools(self, *args, **kwargs):
        tools = original_get_tools(self, *args, **kwargs)
        wrapped = []
        for tool in tools or []:
            if isinstance(tool, FunctionTool):
                wrapped.append(tool)
            else:
                try:
                    wrapped.append(FunctionTool(tool))
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("Failed to wrap search toolkit tool %s: %s", tool, exc)
        return wrapped

    _patched_get_tools._ats_patched = True  # type: ignore[attr-defined]
    SearchToolkit.get_tools = _patched_get_tools  # type: ignore[assignment]
    log.debug("Patched CAMEL SearchToolkit.get_tools to return FunctionTool instances")


__all__ = ["patch_search_toolkit"]

