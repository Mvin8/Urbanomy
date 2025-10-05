from __future__ import annotations

from typing import BinaryIO, Optional, Sequence, Union

import geopandas as gpd
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
    """Подготовка данных геоблоков в переиспользуемый интерфейс."""

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
        """Возвращает подготовленный GeoDataFrame."""
        scenario_source = scenario_blocks if scenario_blocks is not None else self._scenario_source
        context_source = context_blocks if context_blocks is not None else self._context_source
        blocks = self._build_blocks(scenario_source, context_source)
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
        scenario = self._resolve_blocks_input(scenario_source)
        context = self._resolve_blocks_input(context_source)
        blocks = pd.concat([scenario, context], ignore_index=True)
        blocks = gpd.GeoDataFrame(blocks, geometry=scenario.geometry.name, crs=scenario.crs)
        blocks['site_area'] = blocks.geometry.area
        return blocks

    def _resolve_blocks_input(
        self,
        source: BlocksInput,
    ) -> gpd.GeoDataFrame:
        loaded = self._load_dataframe_from_source(source)
        return self._ensure_geodataframe(loaded)

    @staticmethod
    def _load_dataframe_from_source(source: BlocksInput) -> DataFrameLike:
        if isinstance(source, (pd.DataFrame, gpd.GeoDataFrame)):
            return source.copy()
        if hasattr(source, 'read'):
            binary_source = LandDataPreparator._reset_stream(source)
            return pd.read_pickle(binary_source)
        raise TypeError('Ожидался GeoDataFrame/DataFrame или бинарный поток с данными.')

    @staticmethod
    def _reset_stream(stream: BinaryIO) -> BinaryIO:
        seek = getattr(stream, 'seek', None)
        if callable(seek):
            seek(0)
        return stream

    @staticmethod
    def _ensure_geodataframe(data: DataFrameLike) -> gpd.GeoDataFrame:
        if isinstance(data, gpd.GeoDataFrame):
            return data.copy()
        if 'geometry' not in data.columns:
            raise ValueError("Переданный DataFrame должен содержать колонку 'geometry'.")
        crs = getattr(data, 'crs', None)
        return gpd.GeoDataFrame(data.copy(), geometry='geometry', crs=crs)

    def _clamp_land_use(self, blocks: gpd.GeoDataFrame) -> None:
        for land_use in LandUse:
            column = land_use.value
            if column in blocks.columns:
                blocks[column] = blocks[column].clip(upper=1)

    def _calculate_density(self, blocks: gpd.GeoDataFrame, adjacency_graph) -> pd.DataFrame:
        density_df = self._density_regressor.evaluate(blocks, adjacency_graph).copy()
        density_df['fsi'] = density_df['fsi'].clip(lower=0)
        density_df['gsi'] = density_df['gsi'].clip(lower=0, upper=1)
        density_df['mxi'] = density_df['mxi'].clip(lower=0, upper=1)
        density_df.loc[blocks['residential'] == 0, 'mxi'] = 0
        return density_df

    def _attach_development_indicators(self, blocks: gpd.GeoDataFrame, density_df: pd.DataFrame) -> None:
        density_df = density_df.copy()
        density_df['site_area'] = blocks['site_area']
        indicators = calculate_development_indicators(density_df)
        population = (indicators['living_area'] // self.sqm_per_person).fillna(0)
        indicators['population'] = population.astype(int)
        blocks.loc[:, indicators.columns] = indicators

    def _append_morphotypes(self, blocks: gpd.GeoDataFrame) -> None:
        morphotypes = get_strelka_morphotypes(blocks)
        blocks.loc[:, morphotypes.columns] = morphotypes

    def _append_accessibility(
        self,
        blocks: gpd.GeoDataFrame,
        accessibility_matrix: Optional[AccessibilityInput],
    ) -> None:
        matrix = (
            self._resolve_accessibility_input(accessibility_matrix)
            if accessibility_matrix is not None
            else self._load_accessibility_matrix()
        )
        area_acc = area_accessibility(matrix, blocks)
        blocks.loc[:, area_acc.columns] = area_acc

    def _load_accessibility_matrix(self) -> pd.DataFrame:
        if self._accessibility_cache is None:
            self._accessibility_cache = self._resolve_accessibility_input(self._accessibility_source)
        return self._accessibility_cache.copy()

    @staticmethod
    def _resolve_accessibility_input(source: AccessibilityInput) -> pd.DataFrame:
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
        cleaned = blocks.copy()
        for prefix in ('capacity', 'count'):
            drop_cols = [col for col in cleaned.columns if col.startswith(prefix)]
            if drop_cols:
                cleaned = cleaned.drop(columns=drop_cols)
        if 'land_use' in cleaned.columns:
            cleaned['land_use'] = (
                cleaned['land_use']
                .astype(str)
                .str.replace(r'^LandUse\.', '', regex=True)
            )
        geom_col = cleaned.geometry.name
        keep_cols = [col for col in self.output_columns if col in cleaned.columns]
        ordered_cols = keep_cols + ([geom_col] if geom_col not in keep_cols else [])
        return cleaned[ordered_cols].copy()
