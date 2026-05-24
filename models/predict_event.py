from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from infer import build_model


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "models" / "checkpoints"


ALIASES = {
    "Speed_kmh": ["Speed_kmh", "averageSpeedKph", "speed", "speedKph"],
    "Traffic_Density": ["Traffic_Density", "trafficDensity", "vehicleCount", "occupancyPercent"],
    "Latitude": ["Latitude", "latitude", "lat"],
    "Longitude": ["Longitude", "longitude", "lon", "lng"],
    "Event_ID": ["Event_ID", "eventId", "id"],
    "OT": ["OT", "value", "load", "temperatureC", "airQualityIndex"],
}


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
    values = np.array([get_value(payload, column, float(mean[idx])) for idx, column in enumerate(feature_columns)], dtype=np.float32)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--payload")
    parser.add_argument("--payload-b64")
    args = parser.parse_args()
    raw_payload = (
        base64.b64decode(args.payload_b64).decode("utf-8")
        if args.payload_b64
        else args.payload if args.payload is not None else (sys.stdin.read() or "{}")
    )
    payload = json.loads(raw_payload)
    run_dir = CHECKPOINTS / args.run
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    model = build_model(metadata)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    seq_len = int(metadata["seqLen"])
    kind = metadata["kind"]
    vector = normalized_feature_vector(metadata, payload)
    with torch.no_grad():
        if kind == "stgcn_forecast":
            nodes = len(metadata["task"]["nodeColumns"])
            x = torch.tensor(np.tile(vector[:nodes], (seq_len, 1)).T[None, :, :], dtype=torch.float32)
            adjacency = torch.tensor(metadata["task"]["adjacency"], dtype=torch.float32)
            raw = model(x, adjacency).numpy()[0].tolist()
            result = {"mode": "graph_forecast", "nodeForecast": raw}
        elif kind == "saits_imputer":
            x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32)
            mask = torch.ones_like(x)
            raw = model(x, mask).numpy()[0, -1].tolist()
            result = {"mode": "imputation", "imputedLastStep": raw}
        elif kind in {"autoencoder", "tranad_anomaly"}:
            x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32)
            recon = model(x)
            mse = torch.mean((recon - x) ** 2).item()
            result = {"mode": "anomaly_reconstruction", "anomalyScore": mse}
        elif kind == "transformer_classifier":
            x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).numpy()[0]
            id_to_class = metadata["task"]["idToClass"]
            best = int(np.argmax(probs))
            result = {
                "mode": "classification",
                "class": id_to_class.get(str(best), id_to_class.get(best, str(best))),
                "probability": float(probs[best]),
                "probabilities": {str(id_to_class.get(str(i), id_to_class.get(i, i))): float(v) for i, v in enumerate(probs)},
            }
        else:
            x = torch.tensor(np.tile(vector, (seq_len, 1))[None, :, :], dtype=torch.float32)
            raw = float(model(x).numpy()[0][0])
            result = {
                "mode": "forecast",
                "normalizedPrediction": raw,
                "prediction": denormalize_target(metadata, raw),
                "target": metadata["task"].get("target"),
            }

    print(
        json.dumps(
            {
                "run": args.run,
                "kind": kind,
                "result": result,
                "modelCard": str(run_dir / "model_card.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
