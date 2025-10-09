"""Utilities for preparing investment-metrics input datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import geopandas as gpd
import pandas as pd

# ``urbanomy.methods.investment_potential.constants.DEFAULT_IP_VALUE`` uses the
# same literal value ("ip_value"); duplicating it here avoids an import cycle
# when this module is imported ahead of ``investment_potential.constants``.
DEFAULT_IP_VALUE: str = "ip_value"


INVESTMENT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "price_pred",
    "price_per_sotka",
    "site_area",
    "living_area",
    "non_living_area",
    "build_floor_area",
    "share",
    DEFAULT_IP_VALUE,
)

DEFAULT_SCENARIO_KEEP_COLUMNS: tuple[str, ...] = (
    "y_log_pred",
    "price_pred",
    "price_per_sotka",
    "is_scn",
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
    "land_use",
    "share",
    "footprint_area",
    "build_floor_area",
    "living_area",
    "non_living_area",
    "population",
    "site_area",
    "fsi",
    "gsi",
    "mxi",
    "l",
    "morphotype",
    "area_accessibility",
    "geometry",
)

DEFAULT_ALLOWED_IP_USES: tuple[str, ...] = (
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
)


@dataclass(frozen=True)
class InvestmentInputSpec:
    """Schema describing the columns required for investment attractiveness.

    Parameters
    ----------
    required : Sequence[str]
        Columns that must be present in the input GeoDataFrame.
    optional : Sequence[str]
        Columns that are desirable but can be imputed with ``defaults`` when
        missing.
    defaults : Mapping[str, float]
        Default values to use for optional columns when absent or containing
        nulls.
    geometry_column : str, optional
        Name of the geometry column in the target GeoDataFrame (default:
        ``"geometry"``).
    """

    required: Sequence[str]
    optional: Sequence[str]
    defaults: Mapping[str, float]
    geometry_column: str = "geometry"

    def enforce(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Validate and reorder a GeoDataFrame according to the specification.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input GeoDataFrame containing at least the required columns.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of ``gdf`` with columns ordered as ``geometry`` + required +
            optional. Missing optional columns are filled using ``defaults``.

        Raises
        ------
        TypeError
            If ``gdf`` is not a GeoDataFrame.
        ValueError
            If the geometry column or any required column is missing.
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Expected GeoDataFrame input.")

        if self.geometry_column not in gdf.columns:
            raise ValueError(f"Geometry column '{self.geometry_column}' is missing.")

        missing = [col for col in self.required if col not in gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        ordered_columns: list[str] = [self.geometry_column]
        ordered_columns.extend(self.required)
        ordered_columns.extend([col for col in self.optional if col in gdf.columns])

        trimmed = gdf.loc[:, ordered_columns].copy()
        for col in self.optional:
            default_value = self.defaults.get(col, 0.0)
            if col not in trimmed.columns:
                trimmed[col] = default_value
            else:
                trimmed[col] = trimmed[col].fillna(default_value)

        return trimmed


INPUT_SPEC = InvestmentInputSpec(
    required=("land_use", "price_pred"),
    optional=(
        "site_area",
        "living_area",
        "non_living_area",
        "build_floor_area",
        "share",
        "price_per_sotka",
        DEFAULT_IP_VALUE,
    ),
    defaults={
        "site_area": 0.0,
        "living_area": 0.0,
        "non_living_area": 0.0,
        "build_floor_area": 0.0,
        "share": 1.0,
        "price_per_sotka": 0.0,
        DEFAULT_IP_VALUE: 0.0,
    },
)


def _ensure_geodataframe(data: gpd.GeoDataFrame | pd.DataFrame) -> gpd.GeoDataFrame:
    """Coerce a pandas DataFrame into a GeoDataFrame when necessary.

    Parameters
    ----------
    data : geopandas.GeoDataFrame or pandas.DataFrame
        Input data structure expected to contain a ``geometry`` column.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame version of ``data`` preserving the original CRS.

    Raises
    ------
    ValueError
        If a pandas DataFrame lacks the ``geometry`` column.
    TypeError
        If ``data`` is neither a GeoDataFrame nor a DataFrame.
    """
    if isinstance(data, gpd.GeoDataFrame):
        return data
    if isinstance(data, pd.DataFrame):
        if "geometry" not in data.columns:
            raise ValueError("Expected DataFrame with a 'geometry' column.")
        return gpd.GeoDataFrame(data, geometry="geometry", crs=getattr(data, "crs", None))
    raise TypeError("Expected GeoDataFrame or DataFrame input.")


def _build_ip_value_lookup(
    base_gdf: gpd.GeoDataFrame,
    allowed_uses: Iterable[str],
    *,
    ip_type_column: str,
    ip_value_column: str,
) -> pd.DataFrame:
    """Aggregate baseline IP values by land-use type.

    Parameters
    ----------
    base_gdf : geopandas.GeoDataFrame
        Reference dataset containing baseline IP values per land-use type.
    allowed_uses : Iterable[str]
        Iterable of land-use codes to retain when computing the lookup table.
    ip_type_column : str
        Column containing land-use type identifiers.
    ip_value_column : str
        Column holding baseline IP values.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame mapping ``ip_type_column`` to averaged baseline
        values under the alias ``ip_value_from_base``.

    Raises
    ------
    ValueError
        If required columns are absent in ``base_gdf``.
    """
    if ip_type_column not in base_gdf.columns:
        raise ValueError(f"Column '{ip_type_column}' is missing in base_gdf.")
    if ip_value_column not in base_gdf.columns:
        raise ValueError(f"Column '{ip_value_column}' is missing in base_gdf.")

    working = base_gdf.copy()
    working[ip_type_column] = working[ip_type_column].astype(str).str.lower()
    allowed = tuple(allowed_uses)
    if allowed:
        working = working[working[ip_type_column].isin(allowed)]

    return (
        working.groupby(ip_type_column, as_index=False)[ip_value_column]
        .mean()
        .rename(columns={ip_value_column: "ip_value_from_base"})
    )


def _prepare_with_base(
    polygon_gdf: gpd.GeoDataFrame,
    base_gdf: gpd.GeoDataFrame,
    *,
    keep_columns: Sequence[str] | None,
    allowed_uses: Iterable[str],
    land_use_column: str,
    ip_type_column: str,
    scenario_flag_column: str,
    land_use_prefix_pattern: str,
    ip_value_column: str,
) -> gpd.GeoDataFrame:
    """Filter, normalise, and enrich scenario polygons using baseline data.

    Parameters
    ----------
    polygon_gdf : geopandas.GeoDataFrame
        Scenario polygons containing at least geometry and land-use data.
    base_gdf : geopandas.GeoDataFrame
        Baseline potential dataset used to impute IP values.
    keep_columns : Sequence[str] or None, optional
        Columns to retain from ``polygon_gdf`` (defaults to
        ``DEFAULT_SCENARIO_KEEP_COLUMNS``).
    allowed_uses : Iterable[str]
        Land-use codes that are permitted in the resulting dataset.
    land_use_column : str
        Column containing land-use codes in the scenario data.
    ip_type_column : str
        Column name for the derived IP type.
    scenario_flag_column : str
        Column indicating scenario membership (used for filtering when present).
    land_use_prefix_pattern : str
        Regex pattern removed from land-use codes.
    ip_value_column : str
        Column receiving the imputed IP values.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered scenario dataset joined with baseline IP values.

    Raises
    ------
    ValueError
        If the land-use column is missing.
    """
    geometry_column = polygon_gdf.geometry.name
    keep_columns = tuple(keep_columns or DEFAULT_SCENARIO_KEEP_COLUMNS)
    existing_keep = [col for col in keep_columns if col in polygon_gdf.columns]

    if land_use_column not in polygon_gdf.columns:
        raise ValueError(f"Column '{land_use_column}' is missing in polygon_gdf.")

    ordered_columns: list[str] = [geometry_column]
    ordered_columns.extend(
        col for col in existing_keep if col not in {geometry_column, land_use_column}
    )
    if land_use_column != geometry_column:
        ordered_columns.insert(1, land_use_column)

    working = polygon_gdf.loc[:, ordered_columns].copy()

    working[land_use_column] = (
        working[land_use_column]
        .astype(str)
        .str.replace(land_use_prefix_pattern, "", regex=True)
    )

    if scenario_flag_column in working.columns:
        mask = working[scenario_flag_column].fillna(False).astype(bool)
        working = working.loc[mask].reset_index(drop=True)

    working[ip_type_column] = working[land_use_column].astype(str).str.lower()
    working = working[
        working[ip_type_column].notna() & (working[ip_type_column] != "none")
    ]

    base_lookup = _build_ip_value_lookup(
        base_gdf,
        allowed_uses=allowed_uses,
        ip_type_column=ip_type_column,
        ip_value_column=ip_value_column,
    )

    working = working.merge(base_lookup, on=ip_type_column, how="left")
    working[ip_value_column] = working.pop("ip_value_from_base").fillna(0.0)

    return working


def prepare_investment_input(
    gdf: gpd.GeoDataFrame,
    project_potential: gpd.GeoDataFrame | pd.DataFrame | None = None,
    *,
    keep_columns: Sequence[str] | None = None,
    allowed_uses: Iterable[str] | None = None,
    land_use_column: str = "land_use",
    ip_type_column: str = "ip_type",
    scenario_flag_column: str = "is_scn",
    land_use_prefix_pattern: str = r"^LandUse\.",
    ip_value_column: str = DEFAULT_IP_VALUE,
) -> gpd.GeoDataFrame:
    """Prepare a GeoDataFrame for investment-metrics calculation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Scenario dataset to be normalised and validated.
    project_potential : geopandas.GeoDataFrame or pandas.DataFrame or None, optional
        Baseline potential dataset used to impute IP values. When provided,
        scenario polygons are filtered to ``scenario_flag_column == True`` and
        joined with baseline IP values.
    keep_columns : Sequence[str] or None, optional
        Columns to preserve when filtering scenario polygons.
    allowed_uses : Iterable[str] or None, optional
        Land-use codes allowed when computing baseline lookups.
    land_use_column : str, optional
        Column containing land-use codes (default ``"land_use"``).
    ip_type_column : str, optional
        Column name used for the derived IP type (default ``"ip_type"``).
    scenario_flag_column : str, optional
        Column indicating scenario membership (default ``"is_scn"``).
    land_use_prefix_pattern : str, optional
        Regex pattern removed from land-use codes (default ``r"^LandUse\."``).
    ip_value_column : str, optional
        Column receiving the imputed IP values (default ``DEFAULT_IP_VALUE``).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame trimmed and ordered according to :data:`INPUT_SPEC`.
    """

    polygon_gdf = _ensure_geodataframe(gdf)
    if project_potential is not None:
        base_ready = _ensure_geodataframe(project_potential)
        polygon_gdf = _prepare_with_base(
            polygon_gdf,
            base_ready,
            keep_columns=keep_columns,
            allowed_uses=tuple(allowed_uses or DEFAULT_ALLOWED_IP_USES),
            land_use_column=land_use_column,
            ip_type_column=ip_type_column,
            scenario_flag_column=scenario_flag_column,
            land_use_prefix_pattern=land_use_prefix_pattern,
            ip_value_column=ip_value_column,
        )

    return INPUT_SPEC.enforce(polygon_gdf)


__all__ = [
    "INVESTMENT_NUMERIC_COLUMNS",
    "DEFAULT_SCENARIO_KEEP_COLUMNS",
    "DEFAULT_ALLOWED_IP_USES",
    "InvestmentInputSpec",
    "INPUT_SPEC",
    "prepare_investment_input",
]
