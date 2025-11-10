"""
Camel Runtime Bootstrap

Provides shared factories and registries for configuring CAMEL-AI
societies, toolkits, and runtime services used across the trading system.
"""

from core.camel_runtime.compat import patch_search_toolkit

patch_search_toolkit()

from .runtime import CamelTradingRuntime
from .registries import ToolkitRegistry
from .societies import TradingWorkforceSociety

__all__ = [
    "CamelTradingRuntime",
    "ToolkitRegistry",
    "TradingWorkforceSociety",
]

