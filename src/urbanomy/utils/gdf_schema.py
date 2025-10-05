# gdf_schema.py

from loguru import logger
import geopandas as gpd
import pandera as pa
import pandas as pd
from shapely.geometry.base import BaseGeometry
from pandera.typing.geopandas import GeoSeries

from .df_schema import DfSchema

DEFAULT_CRS = 4326

class GdfSchema(DfSchema):
    """База для geopandas.GeoDataFrame-схем."""
    # переопределяем поле geometry
    geometry: GeoSeries

    @classmethod
    def _geometry_types(cls) -> set[type[BaseGeometry]]:
        """
        Должен вернуть набор классов Geometry
        (Point, LineString, Polygon и т.п.).
        """
        raise NotImplementedError(
            "Нужно в подклассе описать конкретные типы геометрии"
        )

    @classmethod
    def _check_instance(cls, df):
        if not isinstance(df, gpd.GeoDataFrame):
            raise ValueError("Ожидается geopandas.GeoDataFrame")

    @pa.dataframe_parser
    @classmethod
    def _warn_crs(cls, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Проверяем, что CRS проективная. Если нет — предупреждаем.
        """
        crs = df.crs
        if crs is None or not crs.is_projected:
            recommended = df.estimate_utm_crs()
            logger.warning(
                f"Текущий CRS {crs.to_epsg() if crs else 'None'} "
                f"не проективный. Рекомендуемый: EPSG:{recommended.to_epsg()}"
            )
        return df

    @pa.check("geometry")
    @classmethod
    def _check_geometry(cls, series: pd.Series) -> pd.Series:
        """
        Проверяем каждый элемент: 
        принадлежит ли он к одному из допустимых типов.
        """
        allowed = cls._geometry_types()
        return series.map(
            lambda geom: any(isinstance(geom, t) for t in allowed)
        )

    @classmethod
    def create_empty(cls, crs: int = DEFAULT_CRS) -> gpd.GeoDataFrame:
        # возвращает пустой GeoDataFrame
        return gpd.GeoDataFrame(
            [], columns=cls._columns(), crs=f"EPSG:{crs}"
        )
