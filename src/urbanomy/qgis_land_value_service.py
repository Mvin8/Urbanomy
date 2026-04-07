"""HTTP service for QGIS land-value estimation workflows."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .methods.land_value_modeling.land_price_estimation import LandPriceEstimator


@dataclass(frozen=True)
class QgisLandValueServiceConfig:
    """Runtime configuration for the dedicated QGIS backend."""

    model_path: str
    orig_features: list[str]
    categorical_features: list[str]
    radius_list: list[float] | None = None
    use_service_features: bool = False
    service_features: list[str] | None = None
    predictions_in_log_scale: bool = True
    id_column: str = "id"
    area_column: str = "site_area"
    total_price_column: str = "land_value"
    unit_price_column: str = "land_value_per_100m2"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "QgisLandValueServiceConfig":
        model_path = str(raw.get("model_path", "")).strip()
        if not model_path:
            raise ValueError("Config must provide 'model_path'.")
        orig_features = [str(item).strip() for item in list(raw.get("orig_features") or []) if str(item).strip()]
        categorical_features = [
            str(item).strip() for item in list(raw.get("categorical_features") or []) if str(item).strip()
        ]
        if not orig_features:
            raise ValueError("Config must provide non-empty 'orig_features'.")
        if not categorical_features:
            raise ValueError("Config must provide non-empty 'categorical_features'.")
        radius_list = raw.get("radius_list")
        if radius_list is not None:
            radius_list = [float(item) for item in list(radius_list)]
        service_features = raw.get("service_features")
        if service_features is not None:
            service_features = [str(item).strip() for item in list(service_features) if str(item).strip()]
        return cls(
            model_path=model_path,
            orig_features=orig_features,
            categorical_features=categorical_features,
            radius_list=radius_list,
            use_service_features=bool(raw.get("use_service_features", False)),
            service_features=service_features,
            predictions_in_log_scale=bool(raw.get("predictions_in_log_scale", True)),
            id_column=str(raw.get("id_column", "id")).strip() or "id",
            area_column=str(raw.get("area_column", "site_area")).strip() or "site_area",
            total_price_column=str(raw.get("total_price_column", "land_value")).strip() or "land_value",
            unit_price_column=(
                str(raw.get("unit_price_column", "land_value_per_100m2")).strip() or "land_value_per_100m2"
            ),
        )


@dataclass
class QgisLandValueServiceState:
    """Live service state shared across HTTP requests."""

    config: QgisLandValueServiceConfig
    model: Any
    lock: threading.Lock


def load_qgis_land_value_service_config(path: str | Path) -> QgisLandValueServiceConfig:
    """Load service configuration from a JSON file."""
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a JSON object.")
    config = QgisLandValueServiceConfig.from_mapping(raw)
    model_path = Path(config.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()
    return QgisLandValueServiceConfig(
        model_path=str(model_path),
        orig_features=config.orig_features,
        categorical_features=config.categorical_features,
        radius_list=config.radius_list,
        use_service_features=config.use_service_features,
        service_features=config.service_features,
        predictions_in_log_scale=config.predictions_in_log_scale,
        id_column=config.id_column,
        area_column=config.area_column,
        total_price_column=config.total_price_column,
        unit_price_column=config.unit_price_column,
    )


def create_qgis_land_value_service_state(
    *,
    config_path: str | Path,
) -> QgisLandValueServiceState:
    """Create the runtime state and load the CatBoost model once."""
    config = load_qgis_land_value_service_config(config_path)
    model = CatBoostRegressor()
    model.load_model(config.model_path)
    return QgisLandValueServiceState(config=config, model=model, lock=threading.Lock())


def predict_land_value_for_geojson(
    *,
    layer_geojson: dict[str, Any],
    state: QgisLandValueServiceState,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Estimate land value for one GeoJSON layer."""
    blocks = _geojson_to_gdf(layer_geojson)
    if blocks.empty:
        raise ValueError("Input layer is empty.")
    if blocks.geometry is None:
        raise ValueError("Input layer must contain geometry.")

    config = state.config
    if config.id_column not in blocks.columns:
        blocks[config.id_column] = blocks.index

    area_values = _resolve_area_values(blocks, area_column=config.area_column)
    blocks[config.area_column] = area_values

    for feature in config.orig_features:
        if feature in config.categorical_features or feature not in blocks.columns:
            continue
        blocks[feature] = pd.to_numeric(blocks[feature], errors="coerce")

    with state.lock:
        estimator = LandPriceEstimator(
            model=state.model,
            blocks=blocks,
            radius_list=config.radius_list,
            orig_features=config.orig_features,
            categorical_features=config.categorical_features,
            use_service_features=config.use_service_features,
            service_features=config.service_features,
        )
        predicted = estimator.predict_prices(
            total_price_column=config.total_price_column,
            include_unit_price=False,
            predictions_in_log_scale=config.predictions_in_log_scale,
        )

    cleaned = pd.DataFrame(
        {
            config.area_column: pd.to_numeric(predicted[config.area_column], errors="coerce"),
            config.total_price_column: pd.to_numeric(predicted[config.total_price_column], errors="coerce"),
        },
        index=predicted.index,
    )
    cleaned[config.unit_price_column] = cleaned[config.total_price_column] / cleaned[config.area_column] * 100.0
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    upper_quantile = float(pd.to_numeric(cleaned[config.unit_price_column], errors="coerce").quantile(0.95))
    if not np.isfinite(upper_quantile):
        upper_quantile = 0.0
    outlier_mask = cleaned[config.unit_price_column] > upper_quantile
    if bool(outlier_mask.any()):
        cleaned.loc[outlier_mask, config.unit_price_column] = upper_quantile
        cleaned.loc[outlier_mask, config.total_price_column] = (
            cleaned.loc[outlier_mask, config.unit_price_column]
            * cleaned.loc[outlier_mask, config.area_column]
            / 100.0
        )

    blocks[config.area_column] = pd.to_numeric(cleaned[config.area_column], errors="coerce").to_numpy()
    blocks[config.total_price_column] = pd.to_numeric(cleaned[config.total_price_column], errors="coerce").to_numpy()
    blocks[config.unit_price_column] = pd.to_numeric(cleaned[config.unit_price_column], errors="coerce").to_numpy()

    stats = {
        "rows": int(len(blocks)),
        "area_column": config.area_column,
        "total_price_column": config.total_price_column,
        "unit_price_column": config.unit_price_column,
        "unit_price_upper_quantile": upper_quantile,
        "outliers_clipped_rows": int(outlier_mask.sum()),
    }
    return blocks, stats


def run_qgis_land_value_service(
    *,
    state: QgisLandValueServiceState,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run the HTTP server for QGIS requests."""
    server = ThreadingHTTPServer((host, int(port)), _make_handler(state))
    try:
        print(f"Urbanomy QGIS land-value service listening on http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _make_handler(state: QgisLandValueServiceState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - HTTP API
            path = urlparse(self.path).path
            if path == "/healthz":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "service": "urbanomy-qgis-land-value",
                        "model_path": state.config.model_path,
                    },
                )
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})

        def do_POST(self) -> None:  # noqa: N802 - HTTP API
            path = urlparse(self.path).path
            if path != "/api/land-value-per-100m2":
                _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
                return
            try:
                payload = _read_json_body(self)
                prompt = str(payload.get("prompt", "")).strip()
                layer_name = str(payload.get("layer_name", "blocks")).strip() or "blocks"
                layer_geojson = payload.get("layer_geojson")
                if not isinstance(layer_geojson, dict):
                    raise ValueError("Request must provide 'layer_geojson' as a GeoJSON object.")
                predicted, stats = predict_land_value_for_geojson(
                    layer_geojson=layer_geojson,
                    state=state,
                )
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return

            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "prompt": prompt,
                    "message": (
                        "Стоимость земли (land_value) и стоимость земли за сотку "
                        "(land_value_per_100m2) рассчитаны для всех кварталов."
                    ),
                    "layer_name": f"{layer_name}_with_land_value",
                    "layer_geojson": _gdf_to_geojson(predicted),
                    "stats": stats,
                },
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_length = handler.headers.get("Content-Length", "0").strip() or "0"
    length = int(raw_length)
    body = handler.rfile.read(length) if length > 0 else b"{}"
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    return payload


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _geojson_to_gdf(payload: dict[str, Any]) -> gpd.GeoDataFrame:
    if str(payload.get("type", "")).strip() != "FeatureCollection":
        raise ValueError("layer_geojson must be a GeoJSON FeatureCollection.")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON FeatureCollection must contain a 'features' list.")
    crs = _read_geojson_crs(payload)
    return gpd.GeoDataFrame.from_features(features, crs=crs)


def _gdf_to_geojson(blocks: gpd.GeoDataFrame) -> dict[str, Any]:
    payload = json.loads(blocks.to_json(drop_id=False))
    if not isinstance(payload, dict):
        raise ValueError("Could not serialize predicted layer to GeoJSON.")
    crs_authid = None
    if blocks.crs is not None:
        try:
            crs_authid = blocks.crs.to_string()
        except Exception:
            crs_authid = None
    if crs_authid:
        payload["crs"] = {"type": "name", "properties": {"name": crs_authid}}
    return payload


def _read_geojson_crs(payload: dict[str, Any]) -> str | None:
    crs_payload = payload.get("crs")
    if not isinstance(crs_payload, dict):
        return None
    properties = crs_payload.get("properties")
    if not isinstance(properties, dict):
        return None
    name = properties.get("name")
    return str(name).strip() or None


def _resolve_area_values(blocks: gpd.GeoDataFrame, *, area_column: str) -> np.ndarray:
    if area_column in blocks.columns:
        area_values = pd.to_numeric(blocks[area_column], errors="coerce").to_numpy(copy=True)
    else:
        area_values = np.full(len(blocks), np.nan, dtype=float)
    invalid_mask = ~np.isfinite(area_values) | (area_values <= 0)
    if invalid_mask.any():
        area_values[invalid_mask] = _metric_geometry_area(blocks)[invalid_mask]
    return area_values


def _metric_geometry_area(blocks: gpd.GeoDataFrame) -> np.ndarray:
    try:
        utm_crs = blocks.estimate_utm_crs()
    except Exception:
        utm_crs = None
    if utm_crs is not None:
        try:
            return blocks.to_crs(utm_crs).geometry.area.to_numpy()
        except Exception:
            pass
    return blocks.geometry.area.to_numpy()


def main() -> None:
    """CLI entry point for the QGIS land-value backend."""
    parser = argparse.ArgumentParser(description="Run the Urbanomy QGIS land-value service.")
    parser.add_argument("--config", required=True, help="Path to the service JSON config.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=8765, type=int, help="Bind port.")
    args = parser.parse_args()

    state = create_qgis_land_value_service_state(config_path=args.config)
    run_qgis_land_value_service(state=state, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
