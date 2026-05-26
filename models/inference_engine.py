"""Shared Flowmatic checkpoint inference for CLI and model-inference service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from infer import build_model

ALIASES = {
    "Speed_kmh": ["Speed_kmh", "averageSpeedKph", "speed", "speedKph"],
    "Traffic_Density": ["Traffic_Density", "trafficDensity", "vehicleCount", "occupancyPercent"],
    "Latitude": ["Latitude", "latitude", "lat"],
    "Longitude": ["Longitude", "longitude", "lon", "lng"],
    "Event_ID": ["Event_ID", "eventId", "id"],
    "OT": ["OT", "value", "load", "temperatureC", "airQualityIndex"],
}


def resolve_device(device: str | None = None) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_value(payload: dict[str, Any], column: str, fallback: float = 0.0) -> float:
    for key in ALIASES.get(column, [column]):
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return fallback
    return fallback


def normalized_feature_vector(metadata: dict[str, Any], payload: dict[str, Any]) -> np.ndarray:
    feature_columns = metadata["task"].get("featureColumns") or metadata["task"].get("nodeColumns") or []
    scaler = metadata["task"].get("scaler", {})
    mean = np.array(scaler.get("mean") or [0.0] * len(feature_columns), dtype=np.float32)
    std = np.array(scaler.get("std") or [1.0] * len(feature_columns), dtype=np.float32)
    std[std == 0] = 1.0
    values = np.array(
        [get_value(payload, column, float(mean[idx])) for idx, column in enumerate(feature_columns)],
        dtype=np.float32,
    )
    return (values - mean) / std


def denormalize_target(metadata: dict[str, Any], output: float) -> float:
    task = metadata.get("task", {})
    target_index = task.get("targetIndex")
    scaler = task.get("scaler", {})
    if target_index is None:
        return output
    mean = scaler.get("mean", [])
    std = scaler.get("std", [])
    if target_index >= len(mean) or target_index >= len(std):
        return output
    return output * float(std[target_index]) + float(mean[target_index])


def load_checkpoint_bundle(run_dir: Path, device: torch.device | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    device = device or resolve_device()
    model_pt = run_dir / "model.pt"
    metadata_path = run_dir / "metadata.json"

    if model_pt.exists():
        checkpoint = torch.load(model_pt, map_location=device, weights_only=False)
        metadata = checkpoint["metadata"]
        model = build_model(metadata)
        model.load_state_dict(checkpoint["state_dict"])
    elif (run_dir / "model.safetensors").exists() and metadata_path.exists():
        from safetensors.torch import load_file

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = build_model(metadata)
        state_dict = load_file(str(run_dir / "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No model.pt or model.safetensors+metadata.json in {run_dir}")

    model.to(device)
    model.eval()
    return model, metadata


def predict_payload(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    payload: dict[str, Any],
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or next(model.parameters()).device
    seq_len = int(metadata["seqLen"])
    kind = metadata["kind"]
    vector = normalized_feature_vector(metadata, payload)

    with torch.no_grad():
        if kind == "stgcn_forecast":
            nodes = len(metadata["task"]["nodeColumns"])
            x = torch.tensor(np.tile(vector[:nodes], (seq_len, 1)).T[None, :, :], dtype=torch.float32, device=device)
            adjacency = torch.tensor(metadata["task"]["adjacency"], dtype=torch.float32, device=device)
            raw = model(x, adjacency).detach().cpu().numpy()[0].tolist()
            return {"mode": "graph_forecast", "nodeForecast": raw}
        if kind == "saits_imputer":
            x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32, device=device)
            mask = torch.ones_like(x)
            raw = model(x, mask).detach().cpu().numpy()[0, -1].tolist()
            return {"mode": "imputation", "imputedLastStep": raw}
        if kind in {"autoencoder", "tranad_anomaly"}:
            x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32, device=device)
            recon = model(x)
            mse = torch.mean((recon - x) ** 2).item()
            return {"mode": "anomaly_reconstruction", "anomalyScore": mse}
        if kind == "transformer_classifier":
            x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32, device=device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
            id_to_class = metadata["task"]["idToClass"]
            best = int(np.argmax(probs))
            return {
                "mode": "classification",
                "class": id_to_class.get(str(best), id_to_class.get(best, str(best))),
                "probability": float(probs[best]),
                "probabilities": {
                    str(id_to_class.get(str(i), id_to_class.get(i, i))): float(v) for i, v in enumerate(probs)
                },
            }

        x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32, device=device)
        raw = float(model(x).detach().cpu().numpy()[0][0])
        return {
            "mode": "forecast",
            "normalizedPrediction": raw,
            "prediction": denormalize_target(metadata, raw),
            "target": metadata["task"].get("target"),
        }


def run_checkpoint_inference(run_dir: Path, payload: dict[str, Any], device: str | None = None) -> dict[str, Any]:
    torch_device = resolve_device(device)
    model, metadata = load_checkpoint_bundle(run_dir, torch_device)
    result = predict_payload(model, metadata, payload, torch_device)
    return {
        "run": run_dir.name,
        "kind": metadata["kind"],
        "result": result,
        "device": str(torch_device),
        "modelCard": str(run_dir / "model_card.md") if (run_dir / "model_card.md").exists() else None,
    }
