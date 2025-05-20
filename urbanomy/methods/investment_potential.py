import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd


class LandUseInvestmentAnalyzer:
    """
    Анализатор инвестиционной привлекательности по типам землепользования.

    Атрибуты:
        LAND_USE_TO_POTENTIAL_COLUMN (dict): Соответствие типа землепользования -> колонка с потенциалом.
        DEFAULT_WEIGHTS (dict): Весовые коэффициенты по умолчанию.
    """

    LAND_USE_TO_POTENTIAL_COLUMN = {
        'residential_individual': 'Потенциал развития жилой застройки типа ИЖС',
        'residential_lowrise': 'Потенциал развития малоэтажной жилой застройки',
        'residential_midrise': 'Потенциал развития среднеэтажной жилой застройки',
        'residential_multistorey': 'Потенциал развития многоэтажной жилой застройки',
        'business': 'Потенциал развития застройки общественно-деловой зоны',
        'recreation': 'Потенциал развития застройки рекреационной зоны',
        'special': 'Потенциал развития застройки зоны специального назначения',
        'industrial': 'Потенциал развития застройки промышленной зоны',
        'agriculture': 'Потенциал развития застройки сельскохозяйственной зоны',
        'transport': 'Потенциал развития застройки транспортной зоны'
    }

    DEFAULT_WEIGHTS = {
        'residential_individual': {
            'Население': 1.3,
            'Социальное обеспечение': 1.4,
            'Экологическая ситуация': 1.5,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.2,
            'Транспортное обеспечение': 1.1,
            'default': 1.0
        },
        'residential_lowrise': {
            'Население': 1.4,
            'Социальное обеспечение': 1.3,
            'Экологическая ситуация': 1.4,
            'Транспортное обеспечение': 1.2,
            'default': 1.0
        },
        'residential_midrise': {
            'Средняя этажность': 1.5,
            'Население': 1.4,
            'Социальное обеспечение': 1.3,
            'Транспортное обеспечение': 1.2,
            'default': 1.0
        },
        'residential_multistorey': {
            'Средняя этажность': 1.5,
            'Население': 1.4,
            'Транспортное обеспечение': 1.3,
            'Социальное обеспечение': 1.2,
            'default': 1.0
        },
        'business': {
            'Транспортное обеспечение': 1.5,
            'Население': 1.4,
            'Социальное обеспечение (комфорт)': 1.3,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.2,
            'default': 1.0
        },
        'recreation': {
            'Экологическая ситуация': 1.5,
            'Социальное обеспечение (комфорт)': 1.4,
            'Транспортное обеспечение': 1.2,
            'Население': 0.8,
            'default': 1.0
        },
        'special': {
            'Потенциал размещения порта': 1.5,
            'Транспортное обеспечение': 1.4,
            'Потенциал размещения логистического, складского комплекса': 1.3,
            'default': 1.0
        },
        'industrial': {
            'Потенциал размещения логистического, складского комплекса': 1.5,
            'Транспортное обеспечение': 1.4,
            'Экологическая ситуация': 0.8,
            'Население': 0.9,
            'default': 1.0
        },
        'agriculture': {
            'Экологическая ситуация': 1.5,
            'Население': 0.8,
            'Транспортное обеспечение': 1.2,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.1,
            'default': 1.0
        },
        'transport': {
            'Потенциал размещения логистического, складского комплекса': 1.5,
            'Количество аэропортов местного значения': 1.4,
            'Средняя доступность до близлежащего крупного населенного пункта': 1.3,
            'default': 1.0
        }
    }

    def __init__(self, weights: dict = None, weights_path: str = None):
        """
        Инициализация анализатора.

        Args:
            weights (dict, optional): Пользовательские веса в виде словаря.
            weights_path (str, optional): Путь к JSON-файлу с пользовательскими весами.

        Приоритет:
            1) weights (если передан)
            2) weights_path (если передан)
            3) DEFAULT_WEIGHTS
        """
        if weights is not None:
            self.weights = weights
        elif weights_path:
            with open(weights_path, 'r', encoding='utf-8') as f:
                self.weights = json.load(f)
        else:
            self.weights = self.DEFAULT_WEIGHTS

        self.land_use_to_potential = self.LAND_USE_TO_POTENTIAL_COLUMN

    def compute_scores(self, polygon_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Вычисляет инвестиционную привлекательность для каждого типа землепользования.

        Args:
            polygon_gdf (GeoDataFrame): Исходный GeoDataFrame с атрибутами и потенциалами.

        Returns:
            GeoDataFrame с добавленными столбцами ИП_<land_use>.
        """
        potential_cols = list(self.land_use_to_potential.values())
        attributes = [
            col for col in polygon_gdf.columns
            if polygon_gdf[col].dtype.kind in 'if'
            and col not in potential_cols
            and (((polygon_gdf[col] >= -5) & (polygon_gdf[col] <= 5) & (polygon_gdf[col] != 0)).any())
        ]

        for lu, pot_col in self.land_use_to_potential.items():
            score_col = f'ИП_{lu}'

            def calc(row):
                pot = row.get(pot_col, None)
                if pd.isna(pot):
                    return None
                vals = []
                for attr in attributes:
                    if pd.notna(row.get(attr)):
                        w = self.weights.get(lu, {}).get(attr, self.weights[lu]['default'])
                        vals.append(row[attr] * w)
                if not vals:
                    return None
                return round(sum(vals) / len(vals) * (pot / 5), 1)

            polygon_gdf[score_col] = polygon_gdf.apply(calc, axis=1)

        return polygon_gdf

    def plot_attribute_weights(self, land_use_key: str, ax=None):
        """
        Визуализация весов атрибутов для заданного типа землепользования.

        Args:
            land_use_key (str): Ключ типа землепользования из словаря weights.
            ax (matplotlib.axes.Axes, optional): Ось для рисования. Если None, создается новая.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        weights_for_lu = self.weights.get(land_use_key, {}).copy()
        weights_for_lu.pop('default', None)

        strong = {k: v for k, v in weights_for_lu.items() if v > 1}
        weak = {k: v for k, v in weights_for_lu.items() if v < 1}
        all_w = {**strong, **weak}
        items = sorted(all_w.items(), key=lambda x: x[1], reverse=True)

        labels, vals = zip(*items)
        colors = ['green' if v > 1 else 'red' for v in vals]

        ax.barh(labels, vals, color=colors)
        ax.set_xlim(0, max(vals) + 0.5)
        ax.invert_yaxis()
        ax.set_title(f'Влияние факторов: {land_use_key}', fontsize=10)
        return ax


    def visualize_investment_maps(
            self,
            polygon_gdf: gpd.GeoDataFrame,
            cols: int = 4
        ) -> plt.Figure:
            """
            Отрисовывает сетку карт инвестиционной привлекательности по типам землепользования.

            Args:
                polygon_gdf (GeoDataFrame): GeoDataFrame с колонками ИП_<land_use>.
                cols (int): Количество столбцов в сетке.
            Returns:
                matplotlib.figure.Figure
            """
            # теперь берём нужный маппинг прямо из self
            land_use_to_potential = self.land_use_to_potential

            # собираем все столбцы score
            score_cols = [f'ИП_{lu}' for lu in land_use_to_potential.keys()]
            vmin = polygon_gdf[score_cols].min().min()
            vmax = polygon_gdf[score_cols].max().max()

            n = len(score_cols)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            axes = axes.flatten()

            for i, lu in enumerate(land_use_to_potential.keys()):
                col = f'ИП_{lu}'
                ax = axes[i]
                mean_val = polygon_gdf[col].mean()

                polygon_gdf.plot(
                    column=col,
                    cmap='RdYlGn',
                    legend=False,
                    ax=ax,
                    edgecolor='black',
                    vmin=vmin,
                    vmax=vmax
                )
                ax.set_title(f"{lu} (средн.: {mean_val:.1f})", fontsize=12)
                ax.axis('off')

            # отключаем лишние оси
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            # единая цветовая шкала
            fig.subplots_adjust(right=0.88)
            cax = fig.add_axes([1.1, 0.15, 0.04, 0.8])
            sm = plt.cm.ScalarMappable(
                cmap='RdYlGn',
                norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm._A = []
            fig.colorbar(sm, cax=cax, label='Инвестиционная привлекательность')
            plt.suptitle(
                "Инвестиционная привлекательность по типам землепользования",
                fontsize=16, y=0.99
            )
            plt.tight_layout()
            return fig
