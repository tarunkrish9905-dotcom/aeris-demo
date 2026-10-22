"""
AERIS Frontend & Plotly Dash Engineering User Interface.

Provides the multi-page analyst web portal, API client, and mounting adapter
for the AERIS flight telemetry analysis platform.
"""

from aeris.frontend.app import create_dash_app, mount_dash_app
from aeris.frontend.client import AerisApiClient, AerisApiError

__all__ = [
    "AerisApiClient",
    "AerisApiError",
    "create_dash_app",
    "mount_dash_app",
]
