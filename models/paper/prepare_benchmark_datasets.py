from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
DATASETS = MODELS / "datasets"
REPORTS = MODELS / "paper" / "reports"
ASTANA_PATH = ROOT / "data" / "astana_synthetic_data.csv"

HF_DATASETS = [
    {
        "name": "hf_ett_h1_energy",
        "repo_id": "pkr7098/time-series-forecasting-datasets",
        "filename": "ETTh1.csv",
        "train_key": "hf_ett",
    },
    {
        "name": "hf_weather",
        "repo_id": "pkr7098/time-series-forecasting-datasets",
        "filename": "weather.csv",
        "train_key": "hf_weather",
    },
    {
        "name": "hf_traffic",
        "repo_id": "pkr7098/time-series-forecasting-datasets",
        "filename": "traffic.csv",
        "train_key": "hf_traffic",
    },
]

TRAFFIC_BENCHMARKS = [
    {
        "name": "pems_metr_la",
        "train_key": "pems_metr_la",
        "filename": "sensors.csv",
        "hf_repo_id": "witgaw/METR-LA",
        "hf_files": ["train.parquet", "val.parquet", "test.parquet"],
        "value_column": "x_t+0_d0",
    },
    {
        "name": "pems_bay",
        "train_key": "pems_bay",
        "filename": "sensors.csv",
        "hf_repo_id": "witgaw/PEMS-BAY",
        "hf_files": ["train.parquet", "val.parquet", "test.parquet"],
        "value_column": "x_t+0_d0",
    },
]


def read_env_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            return value or None
    return None


def download_hf_dataset(config: dict[str, str], token: str | None) -> dict[str, Any]:
    target_dir = DATASETS / config["name"]
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / config["filename"]
    if target_path.exists() and target_path.stat().st_size > 0:
        frame = pd.read_csv(target_path, nrows=5)
        return {
            "name": config["name"],
            "trainKey": config["train_key"],
            "status": "cached",
            "path": str(target_path),
            "rows": int(sum(1 for _ in open(target_path, encoding="utf-8")) - 1),
            "columns": len(frame.columns),
        }
    if hf_hub_download is None:
        return {"name": config["name"], "trainKey": config["train_key"], "status": "skipped", "reason": "huggingface_hub unavailable"}
    try:
        path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
            repo_type="dataset",
            token=token,
            local_dir=target_dir,
        )
        frame = pd.read_csv(path, nrows=5)
        return {
            "name": config["name"],
            "trainKey": config["train_key"],
            "status": "downloaded",
            "path": str(path),
            "rows": int(sum(1 for _ in open(path, encoding="utf-8")) - 1),
            "columns": len(frame.columns),
            "source": f"https://huggingface.co/datasets/{config['repo_id']}",
        }
    except Exception as exc:
        return {"name": config["name"], "trainKey": config["train_key"], "status": "failed", "reason": str(exc)}


def download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def npz_to_wide_csv(npz_path: Path, csv_path: Path, max_rows: int | None = None) -> dict[str, Any]:
    import numpy as np

    archive = np.load(npz_path)
    arrays = [archive[key] for key in archive.files if archive[key].ndim >= 2]
    if not arrays:
        raise ValueError(f"No 2D arrays found in {npz_path}")
    stacked = np.concatenate(arrays, axis=0)
    if stacked.ndim == 3:
        # (timesteps, nodes, features) -> use first feature channel per node
        values = stacked[:, :, 0]
    else:
        values = stacked
    if max_rows is not None:
        values = values[:max_rows]
    columns = [f"sensor_{index}" for index in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "timestamp", pd.date_range("2024-01-01", periods=len(frame), freq="5min"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return {"rows": len(frame), "columns": len(frame.columns), "nodes": len(columns)}


def parquet_benchmark_to_csv(config: dict[str, Any], token: str | None, max_rows: int | None = 12000) -> dict[str, Any]:
    if hf_hub_download is None:
        return {"name": config["name"], "trainKey": config["train_key"], "status": "skipped", "reason": "huggingface_hub unavailable"}

    target_dir = DATASETS / config["name"]
    csv_path = target_dir / config["filename"]
    if csv_path.exists() and csv_path.stat().st_size > 0:
        frame = pd.read_csv(csv_path, nrows=5)
        return {
            "name": config["name"],
            "trainKey": config["train_key"],
            "status": "cached",
            "path": str(csv_path),
            "rows": int(sum(1 for _ in open(csv_path, encoding="utf-8")) - 1),
            "columns": len(frame.columns),
        }

    frames: list[pd.DataFrame] = []
    value_column = config["value_column"]
    for filename in config["hf_files"]:
        path = hf_hub_download(
            repo_id=config["hf_repo_id"],
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=target_dir / "_cache",
        )
        part = pd.read_parquet(path, columns=["node_id", "t0_timestamp", value_column])
        frames.append(part)

    merged = pd.concat(frames, ignore_index=True)
    wide = merged.pivot_table(index="t0_timestamp", columns="node_id", values=value_column, aggfunc="first")
    wide = wide.sort_index()
    wide.columns = [f"sensor_{column}" for column in wide.columns]
    wide = wide.reset_index().rename(columns={"t0_timestamp": "timestamp"})
    if max_rows is not None:
        wide = wide.head(max_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(csv_path, index=False)
    return {
        "name": config["name"],
        "trainKey": config["train_key"],
        "status": "converted",
        "path": str(csv_path),
        "source": f"https://huggingface.co/datasets/{config['hf_repo_id']}",
        "rows": len(wide),
        "columns": len(wide.columns),
        "nodes": len(wide.columns) - 1,
    }


def prepare_traffic_benchmark(config: dict[str, Any], max_rows: int | None = 12000, token: str | None = None) -> dict[str, Any]:
    if config.get("hf_repo_id"):
        return parquet_benchmark_to_csv(config, token, max_rows=max_rows)
    target_dir = DATASETS / config["name"]
    csv_path = target_dir / config["filename"]
    if csv_path.exists() and csv_path.stat().st_size > 0:
        frame = pd.read_csv(csv_path, nrows=5)
        return {
            "name": config["name"],
            "trainKey": config["train_key"],
            "status": "cached",
            "path": str(csv_path),
            "rows": int(sum(1 for _ in open(csv_path, encoding="utf-8")) - 1),
            "columns": len(frame.columns),
        }

    cache_dir = target_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in config["sources"]:
        npz_path = cache_dir / Path(url).name
        try:
            if not npz_path.exists():
                print(f"[DOWNLOAD] {config['name']} <- {url}")
                download_url(url, npz_path)
            summary = npz_to_wide_csv(npz_path, csv_path, max_rows=max_rows)
            return {
                "name": config["name"],
                "trainKey": config["train_key"],
                "status": "converted",
                "path": str(csv_path),
                "source": url,
                **summary,
            }
        except Exception as exc:
            last_error = str(exc)
            if npz_path.exists():
                npz_path.unlink(missing_ok=True)
    return {
        "name": config["name"],
        "trainKey": config["train_key"],
        "status": "failed",
        "reason": last_error or "no sources succeeded",
    }


def ensure_astana() -> dict[str, Any]:
    if not ASTANA_PATH.exists():
        return {"name": "astana_synthetic", "trainKey": "astana", "status": "missing", "path": str(ASTANA_PATH)}
    rows = int(sum(1 for _ in open(ASTANA_PATH, encoding="utf-8")) - 1)
    return {"name": "astana_synthetic", "trainKey": "astana", "status": "ready", "path": str(ASTANA_PATH), "rows": rows}


def write_dataset_index(entries: list[dict[str, Any]]) -> None:
    index = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "root": str(DATASETS),
        "families": sorted({entry.get("trainKey") for entry in entries if entry.get("trainKey")}),
        "entries": entries,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "dataset_manifest.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    ready = [entry for entry in entries if entry.get("status") in {"ready", "cached", "downloaded", "converted"}]
    print(json.dumps({"ready": len(ready), "total": len(entries), "manifest": str(REPORTS / "dataset_manifest.json")}, indent=2))


def main() -> None:
    token = read_env_token()
    entries: list[dict[str, Any]] = [ensure_astana()]
    for config in HF_DATASETS:
        entries.append(download_hf_dataset(config, token))
    for config in TRAFFIC_BENCHMARKS:
        entries.append(prepare_traffic_benchmark(config, token=token))
    write_dataset_index(entries)


if __name__ == "__main__":
    main()
