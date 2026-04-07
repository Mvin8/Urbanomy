"""QGIS Processing algorithm that calls the Urbanomy backend API."""

from __future__ import annotations

import json

from qgis.core import (
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer,
)

from ..utils import geojson_to_sink, layer_to_geojson, post_json
from .base import UrbanomyAlgorithmBase


class EstimateLandValueViaApiAlgorithm(UrbanomyAlgorithmBase):
    """Call the Urbanomy backend and return a layer with predicted land values."""

    INPUT = "INPUT"
    API_URL = "API_URL"
    PROMPT = "PROMPT"
    TIMEOUT = "TIMEOUT"
    OUTPUT = "OUTPUT"

    HELP = """
    Sends the selected polygon layer to the Urbanomy backend service and returns
    a new layer containing `land_value` and `land_value_per_100m2`.

    The backend should expose POST /api/land-value-per-100m2 and be configured
    with the CatBoost model and feature schema.
    """

    def name(self) -> str:
        return "estimate_land_value_via_api"

    def displayName(self) -> str:
        return self.tr("Estimate land value via Urbanomy API")

    def createInstance(self):
        return EstimateLandValueViaApiAlgorithm()

    def initAlgorithm(self, config=None) -> None:
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                self.tr("Blocks layer"),
                [QgsProcessing.TypeVectorPolygon],
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.API_URL,
                self.tr("Urbanomy backend URL"),
                defaultValue="http://127.0.0.1:8765/api/land-value-per-100m2",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROMPT,
                self.tr("Prompt"),
                defaultValue="Вычисли стоимость земли за сотку для кварталов",
                multiLine=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TIMEOUT,
                self.tr("Request timeout, seconds"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=300.0,
                minValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr("Predicted land values"),
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        layer = self._required_layer(parameters, self.INPUT, context)
        geojson_payload = layer_to_geojson(layer, feedback)
        if not list(geojson_payload.get("features") or []):
            raise QgsProcessingException("Input layer is empty.")

        api_url = self.parameterAsString(parameters, self.API_URL, context).strip()
        prompt = self.parameterAsString(parameters, self.PROMPT, context).strip()
        timeout_seconds = self.parameterAsDouble(parameters, self.TIMEOUT, context)
        if not api_url:
            raise QgsProcessingException("Backend URL must not be empty.")

        payload = {
            "prompt": prompt or "Вычисли стоимость земли за сотку для кварталов",
            "layer_name": layer.name(),
            "layer_geojson": geojson_payload,
        }
        feedback.pushInfo(f"Sending {len(geojson_payload.get('features') or [])} features to {api_url}")
        response = post_json(api_url, payload, timeout_seconds=timeout_seconds)
        if not response.get("ok", False):
            raise QgsProcessingException(str(response.get("error", "Backend returned an error.")))

        message = str(response.get("message", "")).strip()
        if message:
            feedback.pushInfo(message)
        stats = response.get("stats")
        if isinstance(stats, dict):
            feedback.pushInfo(json.dumps(stats, ensure_ascii=False))

        layer_geojson = response.get("layer_geojson")
        if not isinstance(layer_geojson, dict):
            raise QgsProcessingException("Backend response is missing 'layer_geojson'.")
        return {
            self.OUTPUT: geojson_to_sink(
                self,
                parameters,
                context,
                self.OUTPUT,
                layer_geojson,
                fallback_crs=layer.crs(),
            )
        }
