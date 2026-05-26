from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
TABLES = MODELS / "paper" / "tables"
REPORTS = MODELS / "reports"

PRODUCTION_SLOTS: list[dict[str, Any]] = [
    {
        "slot": "traffic_forecast",
        "kind": "patchtst_forecast",
        "dataset": "astana",
        "metric": "test_rmse",
        "direction": "min",
        "priority": 1,
        "canonicalRun": "q1_astana_patchtst_density_forecaster",
    },
    {
        "slot": "traffic_anomaly",
        "kind": "tranad_anomaly",
        "dataset": "astana",
        "metric": "test_rmse",
        "direction": "min",
        "priority": 1,
        "canonicalRun": "q1_astana_tranad_anomaly_detector",
    },
    {
        "slot": "weather_forecast",
        "kind": "timesblock_forecast",
        "dataset": "hf_weather",
        "metric": "test_rmse",
        "direction": "min",
        "priority": 1,
        "canonicalRun": "q1_hf_weather_timesblock_forecaster",
    },
    {
        "slot": "graph_traffic_forecast",
        "kind": "stgcn_forecast",
        "dataset": "hf_traffic",
        "metric": "test_rmse",
        "direction": "min",
        "priority": 2,
        "canonicalRun": "q1_hf_traffic_stgcn_forecaster",
    },
    {
        "slot": "imputation",
        "kind": "saits_imputer",
        "dataset": "astana",
        "metric": "test_masked_mse",
        "direction": "min",
        "priority": 8,
        "canonicalRun": "q1_astana_saits_imputer",
    },
    {
        "slot": "severity_classification",
        "kind": "transformer_classifier",
        "dataset": "astana",
        "metric": "test_macro_f1",
        "direction": "max",
        "priority": 7,
        "canonicalRun": "q1_astana_transformer_severity_classifier",
    },
    {
        "slot": "energy_forecast",
        "kind": "dlinear_forecast",
        "dataset": "hf_ett",
        "metric": "test_rmse",
        "direction": "min",
        "priority": 20,
        "canonicalRun": "q1_hf_ett_dlinear_energy_forecaster",
    },
    {
        "slot": "traffic_speed_forecast",
        "kind": "itransformer_forecast",
        "dataset": "astana",
        "metric": "test_rmse",
        "direction": "min",
        "priority": 5,
        "canonicalRun": "q1_astana_itransformer_speed_forecaster",
    },
]


def load_leaderboard_rows() -> list[dict[str, Any]]:
    csv_path = TABLES / "checkpoint_leaderboard.csv"
    if not csv_path.exists():
        return leaderboard_from_checkpoints()
    with csv_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def leaderboard_from_checkpoints() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(CHECKPOINTS.iterdir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        test = metadata.get("metrics", {}).get("test", {})
        rows.append(
            {
                "name": run_dir.name,
                "kind": metadata["kind"],
                "dataset": metadata["dataset"],
                "test_rmse": test.get("test_rmse"),
                "test_mae": test.get("test_mae"),
                "test_accuracy": test.get("test_accuracy"),
                "test_macro_f1": test.get("test_macro_f1"),
                "test_masked_mse": test.get("test_masked_mse"),
            }
        )
    return rows


def parse_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_seed_variant(name: str) -> bool:
    return "_seed" in name


def is_production_candidate(name: str) -> bool:
    return name.startswith("q1_") and not name.startswith("ablation_")


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    return parse_float(row.get(metric))


def pick_best(rows: list[dict[str, Any]], slot: dict[str, Any]) -> dict[str, Any] | None:
    canonical = slot.get("canonicalRun")
    if canonical:
        canonical_row = next((row for row in rows if row.get("name") == canonical), None)
        if canonical_row and (CHECKPOINTS / str(canonical) / "metadata.json").exists():
            return canonical_row

    metric = str(slot["metric"])
    direction = str(slot["direction"])
    candidates = [
        row
        for row in rows
        if row.get("kind") == slot["kind"]
        and row.get("dataset") == slot["dataset"]
        and not is_seed_variant(str(row.get("name", "")))
        and is_production_candidate(str(row.get("name", "")))
        and metric_value(row, metric) is not None
    ]
    if not candidates:
        return None
    reverse = direction == "max"
    return sorted(candidates, key=lambda row: metric_value(row, metric) or 0, reverse=reverse)[0]


def load_checkpoint_metadata(run_name: str) -> dict[str, Any] | None:
    metadata_path = CHECKPOINTS / run_name / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_portfolio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for slot in PRODUCTION_SLOTS:
        winner = pick_best(rows, slot)
        if not winner:
            continue
        run_name = str(winner["name"])
        metadata = load_checkpoint_metadata(run_name) or {}
        task = metadata.get("task", {})
        production = metadata.get("production", {})
        selected.append(
            {
                "slot": slot["slot"],
                "run": run_name,
                "kind": winner["kind"],
                "dataset": winner["dataset"],
                "priority": slot["priority"],
                "metric": slot["metric"],
                "metricValue": metric_value(winner, str(slot["metric"])),
                "metrics": metadata.get("metrics", {}),
                "capabilities": {
                    "modality": infer_modality(str(winner["dataset"])),
                    "tasks": infer_tasks(str(winner["kind"])),
                    "sensorKinds": infer_sensor_kinds(str(winner["dataset"]), str(winner["kind"])),
                    "requiresGeo": winner["kind"] == "stgcn_forecast",
                    "seqLen": metadata.get("seqLen"),
                    "horizon": metadata.get("horizon"),
                    "inputContract": {
                        "featureColumns": task.get("featureColumns") or task.get("nodeColumns"),
                        "target": task.get("target"),
                        "labelColumn": task.get("labelColumn"),
                        "timestampField": None,
                    },
                },
                "production": production,
            }
        )
    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "version": "phase2",
        "maxModels": 8,
        "sourceLeaderboard": str(TABLES / "checkpoint_leaderboard.csv"),
        "models": selected,
    }


def infer_modality(dataset: str) -> str:
    if "weather" in dataset:
        return "weather"
    if "ett" in dataset or "energy" in dataset:
        return "energy"
    if "traffic" in dataset or "astana" in dataset or "pems" in dataset or "metr" in dataset:
        return "traffic"
    return "generic"


def infer_tasks(kind: str) -> list[str]:
    if kind.endswith("_forecast"):
        return ["forecast"]
    if kind.endswith("_anomaly") or kind == "autoencoder":
        return ["anomaly", "repair"]
    if kind.endswith("_imputer"):
        return ["imputation"]
    if kind.endswith("_classifier"):
        return ["classification"]
    return ["forecast"]


def infer_sensor_kinds(dataset: str, kind: str) -> list[str]:
    modality = infer_modality(dataset)
    if modality == "weather":
        return ["weather", "air"]
    if modality == "energy":
        return ["generic"]
    if kind == "saits_imputer":
        return ["traffic", "weather", "air", "generic"]
    return ["traffic", "generic"]


def main() -> None:
    rows = load_leaderboard_rows()
    portfolio = build_portfolio(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "production_portfolio.json"
    output.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "portfolio": str(output),
                "selected": len(portfolio["models"]),
                "runs": [item["run"] for item in portfolio["models"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
