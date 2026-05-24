from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def load_artifact(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def process_event(event: dict[str, Any], dataset: str = "astana_synthetic") -> dict[str, Any]:
    imputer = load_artifact("imputation_model.json")["rules"][dataset]
    scaler = load_artifact("scaling_model.json")["profiles"][dataset]
    anomaly = load_artifact("anomaly_detector_model.json")["thresholds"][dataset]

    cleaned = dict(event)
    for column, rule in imputer.items():
        if cleaned.get(column) in (None, ""):
            cleaned[column] = rule["value"]

    scaled: dict[str, float] = {}
    anomalies: list[dict[str, Any]] = []
    for column, profile in scaler.items():
        if column not in cleaned:
            continue
        try:
            value = float(cleaned[column])
        except (TypeError, ValueError):
            continue
        std = profile.get("std") or 1
        scaled[column] = (value - profile["mean"]) / std

        limits = anomaly.get(column)
        if limits and (value < limits["lower"] or value > limits["upper"]):
            anomalies.append(
                {
                    "column": column,
                    "value": value,
                    "lower": limits["lower"],
                    "upper": limits["upper"],
                }
            )

    return {
        "cleaned": cleaned,
        "scaled": scaled,
        "anomalies": anomalies,
    }


if __name__ == "__main__":
    manifest = load_artifact("manifest.json")
    sample = {
        "Event_ID": 999999,
        "Timestamp": "2026-04-09 12:00:00",
        "Vehicle_Type": "Car",
        "Speed_kmh": 118,
        "Latitude": 51.122808,
        "Longitude": 71.511946,
        "Event_Type": "Normal",
        "Severity": "Low",
        "Traffic_Density": 82.5,
    }
    print(json.dumps({"manifest": manifest["version"], "result": process_event(sample)}, indent=2))
