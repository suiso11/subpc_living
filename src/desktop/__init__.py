"""Windows native desktop client for SUBPC BUDDY."""

from .api import DesktopApi, DesktopApiError
from .config import DesktopSettings

__all__ = ["DesktopApi", "DesktopApiError", "DesktopSettings"]
