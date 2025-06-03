import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

def load_local(path: str) -> pd.DataFrame:
    """
    Load a local CSV or JSON file into a DataFrame,
    ensuring a DatetimeIndex.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Try to coerce the index to datetime
    try:
        df.index = pd.to_datetime(df.index, errors="raise")
        return df
    except Exception:
        # If index isn’t already datetime, try to find a datetime-like column
        datetime_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if datetime_cols:
            dt_col = datetime_cols[0]
            df[dt_col] = pd.to_datetime(df[dt_col], errors="raise")
            return df.set_index(dt_col)
        else:
            raise KeyError("No datetime-like column found in local file.")


def load_hf(dataset_name: str, split="train", token: str = None) -> pd.DataFrame:
    """
    Load a time-series dataset from Hugging Face Hub.
    Detects any column containing 'date' or 'time' and sets it as index.
    """
    # Load the specified split as a Dataset, then convert to pandas
    ds = load_dataset(dataset_name, split=split, use_auth_token=token)
    df = pd.DataFrame(ds)

    # Identify any datetime-like column
    datetime_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

    if not datetime_cols:
        raise KeyError(
            f"No datetime-like column found in HF dataset '{dataset_name}'. "
            f"Expected a column name containing 'date' or 'time'."
        )

    dt_col = datetime_cols[0]
    df[dt_col] = pd.to_datetime(df[dt_col], errors="raise")
    return df.set_index(dt_col)


def ingest(source: str, **kwargs) -> pd.DataFrame:
    """
    Unified interface for loading data:
      - If 'source' ends with .csv or .json → load local file
      - Otherwise → treat 'source' as a Hugging Face dataset ID
    """
    if source.endswith((".csv", ".json")):
        return load_local(source)
    else:
        return load_hf(source, **kwargs)
