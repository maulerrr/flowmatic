from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
MODEL_DATASETS = ROOT / "models" / "datasets"


@dataclass
class FrameSpec:
    name: str
    source: str
    frame: pd.DataFrame
    numeric_columns: list[str]
    timestamp_column: str | None
    target_column: str | None
    label_column: str | None = None


def load_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)


def infer_timestamp_column(frame: pd.DataFrame) -> str | None:
    candidates = [c for c in frame.columns if any(k in c.lower() for k in ["date", "time", "timestamp"])]
    for column in candidates + list(frame.columns):
        parsed = pd.to_datetime(frame[column].head(300), errors="coerce")
        if parsed.notna().mean() >= 0.6:
            return str(column)
    return None


def infer_numeric_columns(frame: pd.DataFrame) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().mean() >= 0.75:
            result.append(str(column))
    return result


def infer_target_column(frame: pd.DataFrame, numeric_columns: list[str]) -> str | None:
    priority = ["Traffic_Density", "Speed_kmh", "OT", "traffic", "value"]
    for target in priority:
        if target in numeric_columns:
            return target
    return numeric_columns[-1] if numeric_columns else None


def infer_label_column(frame: pd.DataFrame) -> str | None:
    for column in ["Severity", "Event_Type", "label", "target"]:
        if column in frame.columns and frame[column].nunique(dropna=True) <= 32:
            return column
    return None


def frame_spec(name: str, path: Path, nrows: int | None = None) -> FrameSpec:
    frame = load_csv(path, nrows=nrows)
    numeric = infer_numeric_columns(frame)
    return FrameSpec(
        name=name,
        source=str(path),
        frame=frame,
        numeric_columns=numeric,
        timestamp_column=infer_timestamp_column(frame),
        target_column=infer_target_column(frame, numeric),
        label_column=infer_label_column(frame),
    )


def load_named_frame(name: str, nrows: int | None = None) -> FrameSpec:
    paths = {
        "astana": DATA_ROOT / "astana_synthetic_data.csv",
        "hf_ett": MODEL_DATASETS / "hf_ett_h1_energy" / "ETTh1.csv",
        "hf_weather": MODEL_DATASETS / "hf_weather" / "weather.csv",
        "hf_traffic": MODEL_DATASETS / "hf_traffic" / "traffic.csv",
        "pems_metr_la": MODEL_DATASETS / "pems_metr_la" / "sensors.csv",
        "pems_bay": MODEL_DATASETS / "pems_bay" / "sensors.csv",
    }
    if name not in paths:
        raise ValueError(f"Unknown dataset: {name}")
    return frame_spec(name, paths[name], nrows=nrows)


class Standardizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        self.std[self.std == 0] = 1
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        clean = np.nan_to_num(values, nan=self.mean)
        return (clean - self.mean) / self.std

    def inverse_target(self, values: np.ndarray | torch.Tensor, target_index: int) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
        return array * self.std[target_index] + self.mean[target_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
        }


class WindowForecastDataset(Dataset):
    def __init__(
        self,
        spec: FrameSpec,
        seq_len: int,
        horizon: int,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
    ) -> None:
        self.spec = spec
        self.seq_len = seq_len
        self.horizon = horizon
        self.feature_columns = feature_columns or spec.numeric_columns[:16]
        self.target_column = target_column or spec.target_column or self.feature_columns[-1]
        self.target_index = self.feature_columns.index(self.target_column)
        values = spec.frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.scaler = Standardizer().fit(values)
        self.values = self.scaler.transform(values).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.values) - self.seq_len - self.horizon + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = index + self.seq_len
        target_at = end + self.horizon - 1
        x = self.values[start:end]
        y = self.values[target_at, self.target_index]
        return torch.tensor(x), torch.tensor([y], dtype=torch.float32)


class SequenceReconstructionDataset(Dataset):
    def __init__(self, spec: FrameSpec, seq_len: int, feature_columns: list[str] | None = None) -> None:
        self.spec = spec
        self.seq_len = seq_len
        self.feature_columns = feature_columns or spec.numeric_columns[:16]
        values = spec.frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.scaler = Standardizer().fit(values)
        self.values = self.scaler.transform(values).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.values) - self.seq_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.seq_len]
        return torch.tensor(x), torch.tensor(x)


class MaskedImputationDataset(Dataset):
    def __init__(
        self,
        spec: FrameSpec,
        seq_len: int,
        feature_columns: list[str] | None = None,
        mask_ratio: float = 0.2,
    ) -> None:
        self.spec = spec
        self.seq_len = seq_len
        self.feature_columns = feature_columns or spec.numeric_columns[:16]
        self.mask_ratio = mask_ratio
        values = spec.frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.scaler = Standardizer().fit(values)
        self.values = self.scaler.transform(values).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.values) - self.seq_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.seq_len]
        mask = np.random.default_rng(index).random(x.shape) > self.mask_ratio
        masked = x.copy()
        masked[~mask] = 0.0
        return (
            torch.tensor(masked, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(mask.astype(np.float32), dtype=torch.float32),
        )


class SequenceClassificationDataset(Dataset):
    def __init__(self, spec: FrameSpec, seq_len: int, label_column: str | None = None) -> None:
        self.spec = spec
        self.seq_len = seq_len
        self.feature_columns = spec.numeric_columns[:16]
        self.label_column = label_column or spec.label_column
        if not self.label_column:
            raise ValueError("Classification dataset needs a label column")
        labels = spec.frame[self.label_column].astype(str).fillna("Unknown")
        self.class_to_id = {label: i for i, label in enumerate(sorted(labels.unique()))}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        values = spec.frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.scaler = Standardizer().fit(values)
        self.values = self.scaler.transform(values).astype(np.float32)
        self.labels = labels.map(self.class_to_id).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return max(0, len(self.values) - self.seq_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.seq_len]
        y = self.labels[index + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)


class GraphForecastDataset(Dataset):
    def __init__(self, spec: FrameSpec, seq_len: int, horizon: int, max_nodes: int = 32) -> None:
        self.spec = spec
        self.seq_len = seq_len
        self.horizon = horizon
        self.node_columns = spec.numeric_columns[:max_nodes]
        values = spec.frame[self.node_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.scaler = Standardizer().fit(values)
        self.values = self.scaler.transform(values).astype(np.float32)
        corr = np.corrcoef(np.nan_to_num(self.values).T)
        corr = np.nan_to_num(np.abs(corr), nan=0.0)
        np.fill_diagonal(corr, 1.0)
        row_sum = corr.sum(axis=1, keepdims=True)
        self.adjacency = (corr / np.maximum(row_sum, 1e-6)).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.values) - self.seq_len - self.horizon + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.seq_len].T
        y = self.values[index + self.seq_len + self.horizon - 1]
        return torch.tensor(x), torch.tensor(y)


def split_dataset(dataset: Dataset, train_ratio: float = 0.7, val_ratio: float = 0.15):
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return (
        torch.utils.data.Subset(dataset, range(0, train_end)),
        torch.utils.data.Subset(dataset, range(train_end, val_end)),
        torch.utils.data.Subset(dataset, range(val_end, total)),
    )


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
