import os
import pandas as pd
from datasets import load_dataset

def load_local(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif ext == ".json":
        df = pd.read_json(path).set_index('datetime')
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # coerce index to datetime
    df.index = pd.to_datetime(df.index, errors="raise")
    return df

def load_hf(dataset_name: str, split="train", token: str = None) -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split, use_auth_token=token)
    df = pd.DataFrame(ds)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.set_index('datetime')

def ingest(source: str, **kwargs) -> pd.DataFrame:
    if source.endswith(('.csv', '.json')):
        return load_local(source)
    else:
        return load_hf(source, **kwargs)
