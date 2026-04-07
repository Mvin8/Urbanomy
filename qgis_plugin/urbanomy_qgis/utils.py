"""Shared helpers for QGIS Processing algorithms."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    NULL,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsJsonUtils,
    QgsProcessingException,
    QgsWkbTypes,
)


def layer_to_geojson(layer, feedback) -> dict[str, Any]:
    """Serialize a QGIS layer to a GeoJSON FeatureCollection."""
    features: list[dict[str, Any]] = []
    field_names = [field.name() for field in layer.fields()]
    for feature in layer.getFeatures():
        geometry = feature.geometry()
        if geometry is None or geometry.isEmpty():
            geometry_payload = None
        else:
            geometry_payload = json.loads(geometry.asJson())
        properties = {name: _serialize_attribute(feature[name]) for name in field_names}
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": geometry_payload,
            }
        )
    feedback.pushInfo(f"Loaded {len(features)} features from layer '{layer.name()}'.")
    payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": features,
    }
    crs_authid = layer.crs().authid()
    if crs_authid:
        payload["crs"] = {"type": "name", "properties": {"name": crs_authid}}
    return payload


def geojson_to_sink(
    algorithm,
    parameters,
    context,
    output_name: str,
    payload: dict[str, Any],
    *,
    fallback_crs: QgsCoordinateReferenceSystem | None = None,
):
    """Write a GeoJSON FeatureCollection to a QGIS feature sink."""
    if str(payload.get("type", "")).strip() != "FeatureCollection":
        raise QgsProcessingException("Backend did not return a GeoJSON FeatureCollection.")
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise QgsProcessingException("Backend GeoJSON payload is missing 'features'.")

    attribute_names = _collect_attribute_names(raw_features)
    fields = QgsFields()
    for name in attribute_names:
        values = [feature.get("properties", {}).get(name) for feature in raw_features if isinstance(feature, dict)]
        fields.append(QgsField(str(name), _qvariant_type(values)))

    sink_wkb = _geojson_wkb_type(raw_features)
    sink_crs = _geojson_crs(payload, fallback_crs=fallback_crs)
    sink, dest_id = algorithm.parameterAsSink(parameters, output_name, context, fields, sink_wkb, sink_crs)
    if sink is None:
        raise QgsProcessingException(f"Could not create output sink '{output_name}'.")

    for raw_feature in raw_features:
        if not isinstance(raw_feature, dict):
            continue
        feature = QgsFeature(fields)
        properties = raw_feature.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        feature.setAttributes([_normalize_output_value(properties.get(name)) for name in attribute_names])
        geometry_payload = raw_feature.get("geometry")
        if geometry_payload is not None:
            geometry = QgsJsonUtils.geometryFromGeoJson(json.dumps(geometry_payload, ensure_ascii=False))
            if geometry is not None and not geometry.isEmpty():
                feature.setGeometry(geometry)
        sink.addFeature(feature, QgsFeatureSink.FastInsert)

    return dest_id


def post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    """Send a JSON POST request to the backend."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(timeout_seconds)) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise QgsProcessingException(f"Backend HTTP error {exc.code}: {details}") from exc
    except URLError as exc:
        raise QgsProcessingException(f"Could not connect to backend: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QgsProcessingException(f"Backend returned invalid JSON: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise QgsProcessingException("Backend response must be a JSON object.")
    return parsed


def _serialize_attribute(value: Any) -> Any:
    if value is None:
        return None
    if value == NULL:
        return None
    if hasattr(value, "isNull") and callable(getattr(value, "isNull")):
        try:
            if value.isNull():
                return None
        except Exception:
            pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_output_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() in {"NULL", "NONE", "NAN"}:
        return None
    return value


def _qvariant_type(values: list[Any]) -> QVariant.Type:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return QVariant.Bool
        if isinstance(value, int) and not isinstance(value, bool):
            return QVariant.LongLong
        if isinstance(value, float):
            return QVariant.Double
        return QVariant.String
    return QVariant.String


def _collect_attribute_names(raw_features: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_feature in raw_features:
        if not isinstance(raw_feature, dict):
            continue
        properties = raw_feature.get("properties", {})
        if not isinstance(properties, dict):
            continue
        for name in properties.keys():
            text = str(name)
            if text in seen:
                continue
            seen.add(text)
            ordered.append(text)
    return ordered


def _geojson_wkb_type(raw_features: list[dict[str, Any]]) -> QgsWkbTypes.Type:
    for raw_feature in raw_features:
        if not isinstance(raw_feature, dict):
            continue
        geometry_payload = raw_feature.get("geometry")
        if geometry_payload is None:
            continue
        geometry = QgsJsonUtils.geometryFromGeoJson(json.dumps(geometry_payload, ensure_ascii=False))
        if geometry is None or geometry.isEmpty():
            continue
        return geometry.wkbType()
    return QgsWkbTypes.Unknown


def _geojson_crs(
    payload: dict[str, Any],
    *,
    fallback_crs: QgsCoordinateReferenceSystem | None = None,
) -> QgsCoordinateReferenceSystem:
    qgs_crs = QgsCoordinateReferenceSystem()
    crs_payload = payload.get("crs")
    if not isinstance(crs_payload, dict):
        return fallback_crs if fallback_crs is not None else qgs_crs
    properties = crs_payload.get("properties")
    if not isinstance(properties, dict):
        return fallback_crs if fallback_crs is not None else qgs_crs
    name = str(properties.get("name") or "").strip()
    if not name:
        return fallback_crs if fallback_crs is not None else qgs_crs
    if qgs_crs.createFromString(name):
        return qgs_crs
    return fallback_crs if fallback_crs is not None else qgs_crs
