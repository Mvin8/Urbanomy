"""Main QGIS plugin object."""

from __future__ import annotations

from qgis.core import QgsApplication

from .bootstrap import ensure_runtime_on_path


class UrbanomyQgisPlugin:
    """Register the Urbanomy processing provider inside QGIS."""

    def __init__(self, iface) -> None:
        self.iface = iface
        self._provider = None

    def initGui(self) -> None:
        ensure_runtime_on_path()
        if self._provider is None:
            from .provider import UrbanomyProvider

            self._provider = UrbanomyProvider()
        QgsApplication.processingRegistry().addProvider(self._provider)

    def unload(self) -> None:
        if self._provider is None:
            return
        QgsApplication.processingRegistry().removeProvider(self._provider)
        self._provider = None
