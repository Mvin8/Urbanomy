from __future__ import annotations

from typing import Any, BinaryIO, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from blocksnet.analysis.accessibility import area_accessibility
from blocksnet.analysis.indicators import calculate_development_indicators
from blocksnet.analysis.morphotypes import get_strelka_morphotypes
from blocksnet.config import log_config
from blocksnet.enums import LandUse
from blocksnet.machine_learning.regression import DensityRegressor
from blocksnet.relations import generate_adjacency_graph

from . import constants as _constants

DataFrameLike = Union[pd.DataFrame, gpd.GeoDataFrame]
BlocksInput = Union[BinaryIO, DataFrameLike]
AccessibilityInput = Union[BinaryIO, pd.DataFrame]


class LandDataPreparator:
    """Provide a reusable interface for preparing block-level geospatial data."""

    DEFAULT_OUTPUT_COLUMNS: Sequence[str] = _constants.DEFAULT_OUTPUT_COLUMNS

    def __init__(
        self,
        scenario_blocks_source: BlocksInput,
        context_blocks_source: BlocksInput,
        accessibility_matrix_source: AccessibilityInput,
        *,
        adjacency_radius: float = _constants.DEFAULT_ADJACENCY_RADIUS,
        sqm_per_person: float = _constants.DEFAULT_SQM_PER_PERSON,
        output_columns: Optional[Sequence[str]] = None,
        log_level: str = 'WARNING',
    ) -> None:
        """Create a preparator configured with scenario, context, and matrices.

        Parameters
        ----------
        scenario_blocks_source : BlocksInput
            Base source representing scenario blocks.
        context_blocks_source : BlocksInput
            Base source representing context blocks.
        accessibility_matrix_source : AccessibilityInput
            Source providing the area accessibility matrix.
        adjacency_radius : float, optional
            Radius (metres) for adjacency graph construction.
        sqm_per_person : float, optional
            Square metres per person used when estimating population.
        output_columns : Sequence[str], optional
            Desired output column ordering. Defaults to
            :data:`~urbanomy.methods.land_value_modeling.constants.DEFAULT_OUTPUT_COLUMNS`.
        log_level : str, optional
            Logging level forwarded to ``blocksnet`` utilities.
        """
        self._scenario_source = scenario_blocks_source
        self._context_source = context_blocks_source
        self._accessibility_source = accessibility_matrix_source
        self.adjacency_radius = adjacency_radius
        self.sqm_per_person = sqm_per_person
        self.output_columns = list(output_columns) if output_columns else list(self.DEFAULT_OUTPUT_COLUMNS)
        self._density_regressor = DensityRegressor()
        self._accessibility_cache: Optional[pd.DataFrame] = None
        self._last_prepared: Optional[gpd.GeoDataFrame] = None
        log_config.set_logger_level(log_level)

    def prepare(
        self,
        scenario_blocks: Optional[BlocksInput] = None,
        context_blocks: Optional[BlocksInput] = None,
        accessibility_matrix: Optional[AccessibilityInput] = None,
    ) -> gpd.GeoDataFrame:
        """Prepare scenario and context blocks with engineered features.

        Parameters
        ----------
        scenario_blocks : BlocksInput, optional
            Scenario blocks override. Falls back to ``scenario_blocks_source``
            when omitted.
        context_blocks : BlocksInput, optional
            Context blocks override. Falls back to ``context_blocks_source``
            when omitted.
        accessibility_matrix : AccessibilityInput, optional
            Accessibility matrix override. Falls back to the cached source
            matrix when omitted.

        Returns
        -------
        geopandas.GeoDataFrame
            Prepared dataset containing engineered indicators and metadata.
        """
        scenario_source = scenario_blocks if scenario_blocks is not None else self._scenario_source
        context_source = context_blocks if context_blocks is not None else self._context_source
        blocks = self._build_blocks(scenario_source, context_source)
        self._ensure_land_use_enum(blocks)
        self._clamp_land_use(blocks)
        adjacency_graph = generate_adjacency_graph(blocks, self.adjacency_radius)
        density_df = self._calculate_density(blocks, adjacency_graph)
        self._attach_development_indicators(blocks, density_df)
        self._append_morphotypes(blocks)
        self._append_accessibility(blocks, accessibility_matrix)
        prepared = self._cleanup(blocks)
        prepared['id'] = prepared.index
        self._last_prepared = prepared.copy()
        return prepared.copy()

    def _build_blocks(
        self,
        scenario_source: BlocksInput,
        context_source: BlocksInput,
    ) -> gpd.GeoDataFrame:
        """Combine scenario and context blocks into a unified GeoDataFrame.

        Parameters
        ----------
        scenario_source : BlocksInput
            Scenario block data or stream.
        context_source : BlocksInput
            Context block data or stream.

        Returns
        -------
        geopandas.GeoDataFrame
            Concatenated blocks with ``site_area`` and ``is_scn`` columns.
        """
        scenario = self._resolve_blocks_input(scenario_source)
        context = self._resolve_blocks_input(context_source)
        if scenario.crs and context.crs and scenario.crs != context.crs:
            context = context.to_crs(scenario.crs)
        blocks = pd.concat([scenario, context], ignore_index=True)
        blocks = gpd.GeoDataFrame(blocks, geometry=scenario.geometry.name, crs=scenario.crs)
        blocks['site_area'] = blocks.geometry.area
        blocks['is_scn'] = LandDataPreparator.mark_scenario_blocks(blocks, scenario)
        return blocks

    def _resolve_blocks_input(
        self,
        source: BlocksInput,
    ) -> gpd.GeoDataFrame:
        """Load blocks data from either an in-memory object or binary stream.

        Parameters
        ----------
        source : BlocksInput
            Either a pandas/GeoPandas object or a binary file-like object that
            yields pickled data.

        Returns
        -------
        geopandas.GeoDataFrame
            Validated GeoDataFrame copy derived from the input ``source``.
        """
        loaded = self._load_dataframe_from_source(source)
        return self._ensure_geodataframe(loaded)

    @staticmethod
    def _load_dataframe_from_source(source: BlocksInput) -> DataFrameLike:
        """Load a DataFrame or GeoDataFrame from the given input source.

        Parameters
        ----------
        source : BlocksInput
            DataFrame-like object or binary stream containing pickled data.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            Copy of the loaded structure.

        Raises
        ------
        TypeError
            If ``source`` is neither a DataFrame-like object nor a readable
            binary stream.
        """
        if isinstance(source, (pd.DataFrame, gpd.GeoDataFrame)):
            return source.copy()
        if hasattr(source, 'read'):
            binary_source = LandDataPreparator._reset_stream(source)
            return pd.read_pickle(binary_source)
        raise TypeError('Ожидался GeoDataFrame/DataFrame или бинарный поток с данными.')

    @staticmethod
    def _reset_stream(stream: BinaryIO) -> BinaryIO:
        """Rewind a binary stream to the beginning when supported.

        Parameters
        ----------
        stream : BinaryIO
            File-like object that may expose ``seek``.

        Returns
        -------
        BinaryIO
            The same stream, rewound to the beginning when possible.
        """
        seek = getattr(stream, 'seek', None)
        if callable(seek):
            seek(0)
        return stream

    @staticmethod
    def _ensure_geodataframe(data: DataFrameLike) -> gpd.GeoDataFrame:
        """Validate that input data can be represented as a GeoDataFrame.

        Parameters
        ----------
        data : pandas.DataFrame or geopandas.GeoDataFrame
            Input structure expected to include a ``geometry`` column.

        Returns
        -------
        geopandas.GeoDataFrame
            Copy of the data coerced to GeoDataFrame.

        Raises
        ------
        ValueError
            If the ``geometry`` column is missing.
        """
        if isinstance(data, gpd.GeoDataFrame):
            return data.copy()
        if 'geometry' not in data.columns:
            raise ValueError("Переданный DataFrame должен содержать колонку 'geometry'.")
        crs = getattr(data, 'crs', None)
        return gpd.GeoDataFrame(data.copy(), geometry='geometry', crs=crs)

    @staticmethod
    def mark_scenario_blocks(
        blocks: gpd.GeoDataFrame,
        scenario: gpd.GeoDataFrame,
    ) -> np.ndarray:
        """Compute a boolean mask identifying scenario blocks.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Combined blocks dataset.
        scenario : geopandas.GeoDataFrame
            Scenario subset used to mark the blocks.

        Returns
        -------
        numpy.ndarray
            Boolean mask aligned to ``blocks.index`` with ``True`` for scenario
            polygons.
        """
        if scenario.empty:
            return np.zeros(len(blocks), dtype=bool)

        scenario_geometry = scenario[['geometry']]
        if scenario_geometry.crs != blocks.crs:
            scenario_geometry = scenario_geometry.to_crs(blocks.crs)

        joined = gpd.sjoin(
            blocks[['geometry']].reset_index().rename(columns={'index': '_idx'}),
            scenario_geometry,
            how='inner',
            predicate='intersects',
        )
        scenario_indices = joined['_idx'].unique()
        return blocks.index.isin(scenario_indices)

    def _clamp_land_use(self, blocks: gpd.GeoDataFrame) -> None:
        """Limit land-use share columns to the [0, 1] interval in-place.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks containing percentage share columns that correspond to
            :class:`~blocksnet.enums.LandUse` values.
        """
        for land_use in LandUse:
            column = land_use.value
            if column in blocks.columns:
                blocks[column] = blocks[column].clip(upper=1)

    def _ensure_land_use_enum(self, blocks: gpd.GeoDataFrame) -> None:
        """Coerce ``land_use`` column to ``LandUse`` enum values when possible."""
        if 'land_use' not in blocks.columns:
            return

        def coerce(value: Any) -> Any:
            if isinstance(value, LandUse) or value is None:
                return value

            text = str(value).strip()
            if not text:
                return value

            try:
                return LandUse(text)
            except ValueError:
                pass

            name = text.upper()
            if name.startswith("LANDUSE."):
                name = name.split(".", 1)[1]
            try:
                return LandUse[name]
            except KeyError:
                return value

        blocks['land_use'] = blocks['land_use'].apply(coerce)

    def _calculate_density(self, blocks: gpd.GeoDataFrame, adjacency_graph) -> pd.DataFrame:
        """Evaluate density indicators with basic post-processing.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks enriched with geometric fields.
        adjacency_graph : Any
            Graph describing neighbourhood relations between blocks.

        Returns
        -------
        pandas.DataFrame
            Density indicators aligned to ``blocks.index``.
        """
        density_df = self._density_regressor.evaluate(blocks, adjacency_graph).copy()
        density_df['fsi'] = density_df['fsi'].clip(lower=0)
        density_df['gsi'] = density_df['gsi'].clip(lower=0, upper=1)
        density_df['mxi'] = density_df['mxi'].clip(lower=0, upper=1)
        density_df.loc[blocks['residential'] == 0, 'mxi'] = 0
        return density_df

    def _attach_development_indicators(self, blocks: gpd.GeoDataFrame, density_df: pd.DataFrame) -> None:
        """Append development indicators derived from density metrics.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks dataset receiving the indicator columns.
        density_df : pandas.DataFrame
            Output of :meth:`_calculate_density` containing density metrics.
        """
        density_df = density_df.copy()
        density_df['site_area'] = blocks['site_area']
        indicators = calculate_development_indicators(density_df)
        population = (indicators['living_area'] // self.sqm_per_person).fillna(0)
        indicators['population'] = population.astype(int)
        blocks.loc[:, indicators.columns] = indicators

    def _append_morphotypes(self, blocks: gpd.GeoDataFrame) -> None:
        """Join morphological classifications to the blocks dataset.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset whose rows will be annotated with morphotype labels.
        """
        morphotypes = get_strelka_morphotypes(blocks)
        blocks.loc[:, morphotypes.columns] = morphotypes

    def _append_accessibility(
        self,
        blocks: gpd.GeoDataFrame,
        accessibility_matrix: Optional[AccessibilityInput],
    ) -> None:
        """Attach area accessibility metrics to blocks in-place.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Blocks dataset being enriched with accessibility metrics.
        accessibility_matrix : AccessibilityInput, optional
            Optional override for the accessibility matrix source.
        """
        matrix = (
            self._resolve_accessibility_input(accessibility_matrix)
            if accessibility_matrix is not None
            else self._load_accessibility_matrix()
        )
        area_acc = area_accessibility(matrix, blocks)
        blocks.loc[:, area_acc.columns] = area_acc

    def _load_accessibility_matrix(self) -> pd.DataFrame:
        """Load or reuse the cached accessibility matrix.

        Returns
        -------
        pandas.DataFrame
            Accessibility matrix suitable for ``area_accessibility``.
        """
        if self._accessibility_cache is None:
            self._accessibility_cache = self._resolve_accessibility_input(self._accessibility_source)
        return self._accessibility_cache.copy()

    @staticmethod
    def _resolve_accessibility_input(source: AccessibilityInput) -> pd.DataFrame:
        """Convert an accessibility input into a DataFrame.

        Parameters
        ----------
        source : AccessibilityInput
            DataFrame or binary stream containing pickled accessibility data.

        Returns
        -------
        pandas.DataFrame
            Copy of the accessibility matrix.

        Raises
        ------
        TypeError
            If the source cannot be interpreted as a DataFrame.
        """
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if hasattr(source, 'read'):
            binary_source = LandDataPreparator._reset_stream(source)
            loaded = pd.read_pickle(binary_source)
            if not isinstance(loaded, pd.DataFrame):
                raise TypeError('Ожидался DataFrame доступности.')
            return loaded
        raise TypeError('Ожидался DataFrame или бинарный поток с матрицей доступности.')

    def _cleanup(self, blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove intermediate columns and enforce output ordering.

        Parameters
        ----------
        blocks : geopandas.GeoDataFrame
            Dataset containing intermediate columns to be trimmed.

        Returns
        -------
        geopandas.GeoDataFrame
            Cleaned view limited to the configured ``output_columns``.
        """
        cleaned = blocks.copy()
        for prefix in ('capacity', 'count'):
            drop_cols = [col for col in cleaned.columns if col.startswith(prefix)]
            if drop_cols:
                cleaned = cleaned.drop(columns=drop_cols)
        geom_col = cleaned.geometry.name
        keep_cols = [col for col in self.output_columns if col in cleaned.columns]
        if 'is_scn' in cleaned.columns and 'is_scn' not in keep_cols:
            keep_cols.append('is_scn')
        ordered_cols = keep_cols + ([geom_col] if geom_col not in keep_cols else [])
        return cleaned[ordered_cols].copy()
