# investment_potential/land_use.py
from __future__ import annotations
import math
from typing import Dict, Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pandera import check_types

from ...utils.validation import LandUseDF
from .constants import LAND_USE_TO_POTENTIAL_COLUMN, LAND_USE_WEIGHTS



class LandUseScoreAnalyzer:
    """Анализатор инвестиционной привлекательности по видам землепользования."""

    def __init__(self,
                 weights: Dict[str, Dict[str, float]] | None = None,
                 weights_path: str | None = None):
        if weights is not None:
            self.weights = weights
        elif weights_path:
            import json
            with open(weights_path, "r", encoding="utf-8") as f:
                self.weights = json.load(f)
        else:
            self.weights = LAND_USE_WEIGHTS
        self.land_use_to_potential = LAND_USE_TO_POTENTIAL_COLUMN

    # ------------------------------------------------------------------ #
    # основные методы
    # ------------------------------------------------------------------ #
    def compute_scores(self, polygon_gdf: LandUseDF) -> LandUseDF:
        pot_cols = list(self.land_use_to_potential.values())
        attrs = [
            c for c in polygon_gdf.select_dtypes("number").columns
            if c not in pot_cols and ((polygon_gdf[c].between(-5, 5) & (polygon_gdf[c] != 0)).any())
        ]

        for lu, pot_col in self.land_use_to_potential.items():
            score_col = f"ИП_{lu}"

            def calc(row: pd.Series):
                pot = row.get(pot_col)
                if pd.isna(pot):
                    return None
                vals = [
                    row[a] * self.weights.get(lu, {}).get(a, self.weights[lu]["default"])
                    for a in attrs if pd.notna(row[a])
                ]
                return round(sum(vals) / len(vals) * (pot / 5), 1) if vals else None

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
                "Оценка видов использования территории",
                fontsize=16, y=0.99
            )
            plt.tight_layout()
            return fig
