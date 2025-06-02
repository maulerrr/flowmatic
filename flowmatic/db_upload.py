import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, Float, DateTime, Text, Boolean
from typing import Dict, Any


def build_postgres_url(
    username: str,
    password: str,
    host: str,
    port: int,
    database: str,
) -> str:
    """
    Construct a SQLAlchemy PostgreSQL URL string.
    """
    return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"


def infer_sqlalchemy_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infer a SQLAlchemy dtype mapping for each column in `df`.
    - Integer columns → Integer()
    - Float columns   → Float()
    - DatetimeIndex or datetime-like → DateTime()
    - Boolean         → Boolean()
    - Everything else → Text()
    """
    dtype_map: Dict[str, Any] = {}
    for col, series in df.items():
        if pd.api.types.is_integer_dtype(series.dtype):
            dtype_map[col] = Integer()
        elif pd.api.types.is_float_dtype(series.dtype):
            dtype_map[col] = Float()
        elif pd.api.types.is_bool_dtype(series.dtype):
            dtype_map[col] = Boolean()
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            dtype_map[col] = DateTime()
        else:
            dtype_map[col] = Text()
    return dtype_map


def upload_df_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    db_url: str,
    if_exists: str = "replace",
    index: bool = False,
    custom_dtypes: Dict[str, Any] = None,
) -> None:
    """
    Upload a DataFrame `df` to a PostgreSQL table via SQLAlchemy:
    - `table_name`: target table in the database.
    - `db_url`: SQLAlchemy URL (e.g. "postgresql+psycopg2://user:pw@host:port/db").
    - `if_exists`: action if table exists (replace/append/fail).
    - `index`: whether to write DataFrame’s index as a column.
    - `custom_dtypes`: optional override of dtype mapping.
    """
    engine = create_engine(db_url)
    dtype_map = custom_dtypes or infer_sqlalchemy_types(df)
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=index,
        dtype=dtype_map,
        method="multi",
    )
