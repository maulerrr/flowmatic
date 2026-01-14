import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump, load


def _numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).fillna(method="ffill").fillna(method="bfill")


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.01, random_state: int = 42) -> Tuple[IsolationForest, dict]:
    X = _numeric_matrix(df)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    scores = -model.score_samples(X)
    metrics = {"score_mean": float(np.mean(scores)), "score_std": float(np.std(scores))}
    return model, metrics


def save_model(model: IsolationForest, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)


def load_model(path: str) -> IsolationForest:
    return load(path)
