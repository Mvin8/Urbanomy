"""QGIS Processing provider for Urbanomy backend workflows."""

from __future__ import annotations

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .algorithms.estimate_land_value_via_api import EstimateLandValueViaApiAlgorithm
from .bootstrap import plugin_root


class UrbanomyProvider(QgsProcessingProvider):
    """Expose the minimal Urbanomy backend workflow in the Processing Toolbox."""

    def loadAlgorithms(self) -> None:
        self.addAlgorithm(EstimateLandValueViaApiAlgorithm())

    def id(self) -> str:
        return "urbanomy"

    def name(self) -> str:
        return self.tr("Urbanomy")

    def longName(self) -> str:
        return self.tr("Urbanomy")

    def icon(self) -> QIcon:
        return QIcon(str(plugin_root() / "icon.svg"))

    def tr(self, text: str) -> str:
        return QCoreApplication.translate("UrbanomyProvider", text)
