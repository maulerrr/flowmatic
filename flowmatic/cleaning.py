import pandas as pd
import numpy as np

def impute_missing(df: pd.DataFrame, method="time") -> pd.DataFrame:
    # split numeric vs. other columns
    numeric = df.select_dtypes(include=[np.number]).copy()
    others  = df.drop(columns=numeric.columns)

    if method == "time":
        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="raise")

        # convert object->native dtypes to avoid deprecation
        numeric = numeric.infer_objects()

        try:
            numeric = numeric.interpolate(method='time')
        except ValueError:
            # fallback if something goes wrong
            numeric = numeric.ffill().bfill()

    elif method == "ffill":
        numeric = numeric.ffill().bfill()
    else:
        raise ValueError("method must be 'time' or 'ffill'")

    # recombine
    return pd.concat([numeric, others], axis=1)[df.columns]

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df.index.duplicated(keep='first')]

def cap_outliers(df: pd.DataFrame, lower_quantile=0.01, upper_quantile=0.99) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    lower = numeric.quantile(lower_quantile)
    upper = numeric.quantile(upper_quantile)
    # clip only numeric columns
    clipped = numeric.clip(lower=lower, upper=upper, axis=1)
    return pd.concat([clipped, df.drop(columns=numeric.columns)], axis=1)[df.columns]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    df = impute_missing(df, method="time")
    df = cap_outliers(df)
    return df
