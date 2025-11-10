"""
Playwright-powered toolkit for CAMEL agents.

Provides headless browsing and lightweight scraping using Playwright. Agents can
call the exposed tool to fetch rendered page content, titles, and preview
snippets. Requires `pip install playwright` and `playwright install`.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from core.logging import log

try:  # pragma: no cover - optional dependency
    from camel.toolkits import FunctionTool
    CAMEL_TOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FunctionTool = None  # type: ignore
    CAMEL_TOOLS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from playwright.async_api import Browser, Error as PlaywrightError, async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    Browser = None  # type: ignore
    PlaywrightError = Exception  # type: ignore
    PLAYWRIGHT_AVAILABLE = False


class PlaywrightToolkit:
    """Expose Playwright browsing helpers as CAMEL FunctionTools."""

    def __init__(self) -> None:
        self._browser_lock = asyncio.Lock()
        self._browser: Optional[Browser] = None

    async def initialize(self) -> None:
        """Validate that Playwright is available."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is not installed. Install with `pip install playwright` "
                "and run `playwright install` to download browser binaries."
            )

    async def _ensure_browser(self) -> Browser:
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is not installed. Install with `pip install playwright` "
                "and run `playwright install` to download browser binaries."
            )

        async with self._browser_lock:
            if self._browser:
                return self._browser

            playwright = await async_playwright().start()
            try:
                browser = await playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                await playwright.stop()
                raise RuntimeError(
                    "Failed to launch Playwright Chromium browser. "
                    "Ensure `playwright install` has been executed."
                ) from exc

            self._browser = browser

            async def _close_browser() -> None:
                try:
                    await browser.close()
                finally:
                    await playwright.stop()

            # register finalizer to close when loop stops
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._finalize_on_shutdown(_close_browser))
            except RuntimeError:
                pass

            return browser

    @staticmethod
    async def _finalize_on_shutdown(callback) -> None:
        try:
            await asyncio.sleep(0)
        finally:
            await callback()

    @asynccontextmanager
    async def _page_context(self):
        browser = await self._ensure_browser()
        page = await browser.new_page()
        try:
            yield page
        finally:
            await page.close()

    async def browse_url(
        self,
        url: str,
        wait_for_selector: Optional[str] = None,
        timeout_seconds: float = 15.0,
        max_characters: int = 2000,
    ) -> Dict[str, Any]:
        """Navigate to a URL headlessly and return a rendered snippet."""

        await self.initialize()

        try:
            async with self._page_context() as page:
                await page.goto(url, timeout=timeout_seconds * 1000)
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=timeout_seconds * 1000)

                title = await page.title()
                content = await page.inner_text("body", timeout=timeout_seconds * 1000)
                snippet = content.strip()
                if len(snippet) > max_characters:
                    snippet = snippet[: max(0, max_characters - 1)].rstrip() + "…"

                return {
                    "success": True,
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "length": len(content),
                }
        except PlaywrightError as exc:
            log.warning("Playwright navigation failed for %s: %s", url, exc)
            return {
                "success": False,
                "url": url,
                "error": str(exc),
            }
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Unexpected Playwright error for %s: %s", url, exc)
            return {
                "success": False,
                "url": url,
                "error": str(exc),
            }

    def get_browse_tool(self):
        """Return the FunctionTool wrapping the Playwright browse helper."""
        if not CAMEL_TOOLS_AVAILABLE:
            raise ImportError("CAMEL function tools not installed")

        toolkit_instance = self

        async def browse_url(
            url: str,
            wait_for_selector: Optional[str] = None,
            timeout_seconds: float = 15.0,
            max_characters: int = 2000,
        ) -> Dict[str, Any]:
            """
            Load the specified URL headlessly and return a structured snapshot.

            Args:
                url: The URL to navigate to (must include scheme).
                wait_for_selector: Optional CSS selector to wait for before scraping.
                timeout_seconds: Maximum time to wait for navigation/selector.
                max_characters: Maximum number of body characters to return.
            """

            return await toolkit_instance.browse_url(
                url=url,
                wait_for_selector=wait_for_selector,
                timeout_seconds=timeout_seconds,
                max_characters=max_characters,
            )

        browse_url.__name__ = "browse_url"
        browse_url.__doc__ = (
            "Navigate to a URL headlessly using Playwright and return a snippet of the rendered page."
        )

        tool = FunctionTool(browse_url)
        try:  # pragma: no cover - schema normalisation
            schema = dict(tool.get_openai_tool_schema())
        except Exception:
            schema = {
                "type": "function",
                "function": {
                    "name": browse_url.__name__,
                    "description": browse_url.__doc__ or browse_url.__name__,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "wait_for_selector": {"type": "string"},
                            "timeout_seconds": {"type": "number"},
                            "max_characters": {"type": "integer"},
                        },
                        "required": ["url"],
                    },
                },
            }

        function_schema = schema.setdefault("function", {})
        function_schema["name"] = browse_url.__name__
        function_schema.setdefault("description", browse_url.__doc__ or browse_url.__name__)
        tool.openai_tool_schema = schema
        return tool

    def get_all_tools(self) -> List[Any]:
        """Return all FunctionTools provided by this toolkit."""
        return [self.get_browse_tool()]


__all__ = ["PlaywrightToolkit"]

