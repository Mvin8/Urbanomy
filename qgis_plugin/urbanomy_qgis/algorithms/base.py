"""Base class for Urbanomy processing algorithms."""

from __future__ import annotations

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import QgsProcessingAlgorithm, QgsProcessingException


class UrbanomyAlgorithmBase(QgsProcessingAlgorithm):
    """Shared helpers for all Urbanomy QGIS algorithms."""

    HELP = ""

    def tr(self, text: str) -> str:
        return QCoreApplication.translate(self.__class__.__name__, text)

    def group(self) -> str:
        return self.tr("Urbanomy")

    def groupId(self) -> str:
        return "urbanomy"

    def shortHelpString(self) -> str:
        return self.tr(self.HELP.strip())

    def _required_layer(self, parameters, name: str, context):
        layer = self.parameterAsVectorLayer(parameters, name, context)
        if layer is None:
            raise QgsProcessingException(f"Missing required layer parameter '{name}'.")
        return layer
