# utils/validation.py

from shapely.geometry import Polygon
from pandera.typing.geopandas import GeoSeries, GeoDataFrame
from .gdf_schema import GdfSchema

class LandUseSchema(GdfSchema):
    """
    Схема для GeoDataFrame с полигонами землепользования.
    - geometry: поле GeoSeries с элементами Polygon
    - остальные колонки (показатели, потенциальные столбцы) берутся динамически из GdfSchema
    """
    geometry: GeoSeries[Polygon]

    @classmethod
    def _geometry_types(cls) -> set[type]:
        # единственный разрешённый тип геометрии — Polygon
        return {Polygon}
    
    class Config(GdfSchema.Config):
        # Отменяем фильтрацию «лишних» колонок
        strict = False

LandUseDF = GeoDataFrame[LandUseSchema]
