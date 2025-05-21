# df_schema.py

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Index
from loguru import logger

class DfSchema(pa.DataFrameModel):
    """Базовая схема для pandas.DataFrame."""
    # Индекс — целочисленный, уникальный
    idx: Index[int] = pa.Field(unique=True)

    class Config:
        strict = "filter"            # запретить лишние колонки
        add_missing_columns = True   # добавить отсутствующие
        coerce = True                # привести типы

    @classmethod
    def _check_instance(cls, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Ожидается pandas.DataFrame")

    @classmethod
    def _check_len(cls, df: pd.DataFrame):
        if df.shape[0] == 0:
            raise ValueError("DataFrame не должен быть пустым")

    @classmethod
    def _check_multi(cls, df: pd.DataFrame):
        if df.index.nlevels > 1 or df.columns.nlevels > 1:
            raise ValueError("Многомерные индексы не поддерживаются")

    @classmethod
    def _reset_index_name(cls, df: pd.DataFrame):
        # сбросим имя индекса, если есть
        if df.index.name is not None:
            df.index.name = None

    @classmethod
    def _before_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        # сюда можно добавить пред-обработку перед валидацией
        return df

    @classmethod
    def _after_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        # сюда — пост-обработку (логирование, трансформации и т.п.)
        return df

    @classmethod
    def validate(cls, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = df.copy()
        cls._check_instance(df)
        cls._check_len(df)
        cls._check_multi(df)
        cls._reset_index_name(df)

        df = cls._before_validate(df)
        df = super().validate(df, **kwargs)
        df = cls._after_validate(df)
        return df.copy()

    @classmethod
    def _columns(cls) -> list[str]:
        return list(cls.to_schema().columns.keys())

    @classmethod
    def create_empty(cls) -> pd.DataFrame:
        # возвращает пустой DataFrame с нужными колонками
        return pd.DataFrame([], columns=cls._columns())

    @pa.dataframe_parser
    @classmethod
    def _enforce_column_order(cls, df: pd.DataFrame) -> pd.DataFrame:
        # приведёт порядок колонок к тому, что задано в схеме
        return df[cls._columns()]
