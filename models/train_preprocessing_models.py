from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
DATASETS_DIR = MODELS_DIR / "datasets"
ASTANA_PATH = ROOT / "data" / "astana_synthetic_data.csv"


HF_DATASETS = [
    {
        "name": "hf_ett_h1_energy",
        "repo_id": "pkr7098/time-series-forecasting-datasets",
        "filename": "ETTh1.csv",
        "purpose": "energy load and transformer telemetry",
    },
    {
        "name": "hf_weather",
        "repo_id": "pkr7098/time-series-forecasting-datasets",
        "filename": "weather.csv",
        "purpose": "weather telemetry for smart-city context",
    },
    {
        "name": "hf_traffic",
        "repo_id": "pkr7098/time-series-forecasting-datasets",
        "filename": "traffic.csv",
        "purpose": "traffic sensor forecasting benchmark",
    },
]


@dataclass
class DatasetBundle:
    name: str
    source: str
    purpose: str
    frame: pd.DataFrame


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return clean_scalar(value)


def write_artifact(name: str, data: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / name).write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def load_hf_dataset(config: dict[str, str], token: str | None, max_rows: int) -> DatasetBundle | None:
    if hf_hub_download is None:
        return None
    try:
        path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
            repo_type="dataset",
            token=token,
            local_dir=DATASETS_DIR / config["name"],
        )
        frame = pd.read_csv(path, nrows=max_rows)
        return DatasetBundle(
            name=config["name"],
            source=f"https://huggingface.co/datasets/{config['repo_id']}/resolve/main/{config['filename']}",
            purpose=config["purpose"],
            frame=frame,
        )
    except Exception as exc:
        print(f"[WARN] Could not load {config['name']}: {exc}")
        return None


def load_datasets(token: str | None, max_rows: int) -> list[DatasetBundle]:
    bundles = [
        DatasetBundle(
            name="astana_synthetic",
            source=str(ASTANA_PATH),
            purpose="local Astana traffic smart-city telemetry",
            frame=pd.read_csv(ASTANA_PATH, nrows=max_rows),
        )
    ]
    for config in HF_DATASETS:
        bundle = load_hf_dataset(config, token, max_rows)
        if bundle is not None:
            bundles.append(bundle)
    return bundles


def infer_timestamp_column(frame: pd.DataFrame) -> str | None:
    candidates = [col for col in frame.columns if any(k in col.lower() for k in ["time", "date", "timestamp"])]
    for col in candidates + list(frame.columns):
        sample = pd.to_datetime(frame[col].head(200), errors="coerce")
        if sample.notna().mean() > 0.6:
            return str(col)
    return None


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().mean() >= 0.7:
            columns.append(str(col))
    return columns


def categorical_columns(frame: pd.DataFrame, numeric: list[str]) -> list[str]:
    return [str(col) for col in frame.columns if str(col) not in set(numeric)]


def profile_numeric(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "mad": 0}
    median = float(values.median())
    mad = float((values - median).abs().median())
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0) or 0),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": median,
        "mad": mad,
        "quantiles": {str(k): float(v) for k, v in quantiles.items()},
    }


def build_schema_model(bundles: list[DatasetBundle]) -> dict[str, Any]:
    datasets = {}
    for bundle in bundles:
        nums = numeric_columns(bundle.frame)
        cats = categorical_columns(bundle.frame, nums)
        datasets[bundle.name] = {
            "source": bundle.source,
            "purpose": bundle.purpose,
            "rows": int(len(bundle.frame)),
            "columns": [str(col) for col in bundle.frame.columns],
            "timestampColumn": infer_timestamp_column(bundle.frame),
            "numericColumns": nums,
            "categoricalColumns": cats,
            "locationColumns": [col for col in bundle.frame.columns if any(k in col.lower() for k in ["lat", "lon", "lng", "location"])],
            "targetCandidates": [
                col
                for col in bundle.frame.columns
                if any(k in col.lower() for k in ["target", "severity", "event", "traffic", "speed", "load", "value", "ot"])
            ],
        }
    return {"type": "schema_inference", "datasets": datasets}


def build_imputation_model(bundles: list[DatasetBundle]) -> dict[str, Any]:
    rules = {}
    for bundle in bundles:
        nums = numeric_columns(bundle.frame)
        dataset_rules = {}
        for col in bundle.frame.columns:
            if col in nums:
                dataset_rules[str(col)] = {"strategy": "median", "value": profile_numeric(bundle.frame[col])["median"]}
            else:
                mode = bundle.frame[col].dropna().mode()
                dataset_rules[str(col)] = {"strategy": "mode", "value": None if mode.empty else str(mode.iloc[0])}
        rules[bundle.name] = dataset_rules
    return {"type": "imputation", "rules": rules}


def build_scaling_model(bundles: list[DatasetBundle]) -> dict[str, Any]:
    profiles = {}
    for bundle in bundles:
        profiles[bundle.name] = {
            col: profile_numeric(bundle.frame[col]) for col in numeric_columns(bundle.frame)
        }
    return {"type": "standard_and_minmax_scaling", "profiles": profiles}


def build_anomaly_model(bundles: list[DatasetBundle]) -> dict[str, Any]:
    thresholds = {}
    for bundle in bundles:
        dataset_thresholds = {}
        for col in numeric_columns(bundle.frame):
            profile = profile_numeric(bundle.frame[col])
            robust_sigma = 1.4826 * (profile["mad"] or profile["std"] or 1)
            dataset_thresholds[col] = {
                "median": profile["median"],
                "mad": profile["mad"],
                "robustSigma": robust_sigma,
                "lower": profile["median"] - 3 * robust_sigma,
                "upper": profile["median"] + 3 * robust_sigma,
                "zThreshold": 3,
            }
        thresholds[bundle.name] = dataset_thresholds
    return {"type": "robust_numeric_anomaly_detector", "thresholds": thresholds}


def build_drift_model(bundles: list[DatasetBundle]) -> dict[str, Any]:
    bins = {}
    for bundle in bundles:
        dataset_bins = {}
        for col in numeric_columns(bundle.frame):
            values = pd.to_numeric(bundle.frame[col], errors="coerce").dropna()
            if values.empty:
                continue
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))).tolist()
            dataset_bins[col] = {"method": "psi_quantile_bins", "edges": edges}
        bins[bundle.name] = dataset_bins
    return {"type": "population_stability_drift_detector", "bins": bins, "alertThreshold": 0.2}


def fit_linear_forecaster(frame: pd.DataFrame, target: str, timestamp: str | None) -> dict[str, Any]:
    series = pd.to_numeric(frame[target], errors="coerce").dropna().reset_index(drop=True)
    if len(series) < 8:
        return {"target": target, "status": "not_enough_rows"}
    rows = []
    y = []
    for idx in range(3, len(series)):
        rows.append([1.0, float(series[idx - 1]), float(series[idx - 2]), float(series[idx - 3])])
        y.append(float(series[idx]))
    x = np.array(rows)
    target_values = np.array(y)
    coef = np.linalg.pinv(x.T @ x) @ x.T @ target_values
    prediction = x @ coef
    mae = float(np.abs(prediction - target_values).mean())
    return {
        "target": target,
        "timestamp": timestamp,
        "features": ["bias", "lag_1", "lag_2", "lag_3"],
        "coefficients": coef.tolist(),
        "mae": mae,
    }


def build_forecasting_model(bundles: list[DatasetBundle], schema: dict[str, Any]) -> dict[str, Any]:
    models = {}
    for bundle in bundles:
        meta = schema["datasets"][bundle.name]
        candidates = [col for col in meta["targetCandidates"] if col in meta["numericColumns"]]
        target = candidates[0] if candidates else (meta["numericColumns"][0] if meta["numericColumns"] else None)
        models[bundle.name] = fit_linear_forecaster(bundle.frame, target, meta["timestampColumn"]) if target else {"status": "no_numeric_target"}
    return {"type": "autoregressive_linear_forecaster", "models": models}


def build_classification_model(bundles: list[DatasetBundle]) -> dict[str, Any]:
    models = {}
    for bundle in bundles:
        label = next((col for col in ["Severity", "Event_Type"] if col in bundle.frame.columns), None)
        nums = numeric_columns(bundle.frame)
        if label is None or not nums:
            models[bundle.name] = {"status": "no_label"}
            continue
        classes = {}
        for class_name, group in bundle.frame.groupby(label):
            classes[str(class_name)] = {
                "prior": float(len(group) / len(bundle.frame)),
                "numeric": {col: profile_numeric(group[col]) for col in nums},
            }
        models[bundle.name] = {"label": label, "classes": classes}
    return {"type": "gaussian_naive_bayes_baseline", "models": models}


def build_clustering_model(bundles: list[DatasetBundle], k: int = 4) -> dict[str, Any]:
    models = {}
    rng = np.random.default_rng(42)
    for bundle in bundles:
        nums = numeric_columns(bundle.frame)[:12]
        if len(nums) < 2:
            models[bundle.name] = {"status": "not_enough_numeric_features"}
            continue
        matrix = bundle.frame[nums].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std == 0] = 1
        z = (matrix - mean) / std
        if len(z) < k:
            models[bundle.name] = {"status": "not_enough_rows"}
            continue
        centroids = z[rng.choice(len(z), size=k, replace=False)]
        for _ in range(20):
            distances = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            for cluster in range(k):
                if np.any(labels == cluster):
                    centroids[cluster] = z[labels == cluster].mean(axis=0)
        models[bundle.name] = {
            "features": nums,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "centroids": centroids.tolist(),
        }
    return {"type": "kmeans_operational_regime_clustering", "models": models}


def build_cleaning_policy(schema: dict[str, Any], anomaly: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "smart_city_cleaning_policy",
        "rules": {
            "dedupe": ["timestampColumn", "locationColumns", "entityId"],
            "dropRowsWithEmptyTimestampWhenTimestampRequired": True,
            "clipNumericOutliersUsing": "anomaly_detector_model.thresholds",
            "normalizeTimezone": "UTC",
            "trimStrings": True,
            "emptyStringsToNull": True,
        },
        "schemaReference": schema["datasets"],
        "anomalyReference": anomaly["thresholds"],
    }


def build_feature_engineering(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "smart_city_feature_engineering",
        "recipes": {
            "time": ["hour", "day_of_week", "is_weekend", "month"],
            "lags": [1, 2, 3, 6, 12],
            "rollingWindows": [3, 6, 12],
            "location": ["lat_lon_grid_500m", "distance_to_city_center_if_available"],
            "categorical": ["frequency_encoding", "top_k_one_hot"],
        },
        "datasets": schema["datasets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows-per-dataset", type=int, default=50000)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    args = parser.parse_args()

    token = os.getenv(args.hf_token_env)
    bundles = load_datasets(token, args.max_rows_per_dataset)
    if len(bundles) == 1:
        print("[WARN] Only local Astana data loaded; Hugging Face downloads were unavailable.")

    schema = build_schema_model(bundles)
    imputation = build_imputation_model(bundles)
    scaling = build_scaling_model(bundles)
    anomaly = build_anomaly_model(bundles)
    drift = build_drift_model(bundles)
    forecasting = build_forecasting_model(bundles, schema)
    classification = build_classification_model(bundles)
    clustering = build_clustering_model(bundles)
    cleaning = build_cleaning_policy(schema, anomaly)
    features = build_feature_engineering(schema)

    artifacts = {
        "schema_inference_model.json": schema,
        "imputation_model.json": imputation,
        "scaling_model.json": scaling,
        "anomaly_detector_model.json": anomaly,
        "drift_detector_model.json": drift,
        "forecasting_model.json": forecasting,
        "classification_model.json": classification,
        "clustering_model.json": clustering,
        "cleaning_policy_model.json": cleaning,
        "feature_engineering_model.json": features,
    }
    for name, artifact in artifacts.items():
        write_artifact(name, artifact)

    manifest = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "datasets": [
            {"name": bundle.name, "source": bundle.source, "purpose": bundle.purpose, "rows": int(len(bundle.frame))}
            for bundle in bundles
        ],
        "artifacts": list(artifacts.keys()),
        "usage": "python models/sample_usage.py",
    }
    write_artifact("manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
