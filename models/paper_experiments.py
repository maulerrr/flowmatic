from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

MODELS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODELS_ROOT.parent
sys.path.insert(0, str(MODELS_ROOT))

from flowml.data import WindowForecastDataset, load_named_frame, save_json, split_dataset
from flowml.models import DLinearForecaster, PatchTSTForecaster
from flowml.training import evaluate_supervised, train_supervised
from infer import build_model
from train import train_experiment, set_seed


PAPER_ROOT = MODELS_ROOT / "paper"
TABLES = PAPER_ROOT / "tables"
REPORTS = PAPER_ROOT / "reports"
CHECKPOINTS = MODELS_ROOT / "checkpoints"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_multi_seed(base_config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        set_seed(seed)
        cfg = copy.deepcopy(base_config)
        cfg["seed"] = seed
        for exp in cfg["experiments"]:
            seeded = copy.deepcopy(exp)
            seeded["name"] = f"{exp['name']}_seed{seed}"
            print(f"[MULTI-SEED] {seed} {seeded['name']}")
            metadata = train_experiment(seeded, cfg)
            rows.append(
                {
                    "protocol": "multi_seed",
                    "seed": seed,
                    "name": seeded["name"],
                    "baseName": exp["name"],
                    "kind": metadata["kind"],
                    "dataset": metadata["dataset"],
                    **flatten_metrics(metadata),
                }
            )
    return rows


def run_ablation(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    ablations = [
        {
            "name": "ablation_seq24_patchtst_astana",
            "kind": "patchtst_forecast",
            "dataset": "astana",
            "target": "Traffic_Density",
            "seq_len": 24,
        },
        {
            "name": "ablation_seq96_patchtst_astana",
            "kind": "patchtst_forecast",
            "dataset": "astana",
            "target": "Traffic_Density",
            "seq_len": 96,
        },
        {
            "name": "ablation_seq24_dlinear_ett",
            "kind": "dlinear_forecast",
            "dataset": "hf_ett",
            "target": "OT",
            "seq_len": 24,
        },
        {
            "name": "ablation_seq96_dlinear_ett",
            "kind": "dlinear_forecast",
            "dataset": "hf_ett",
            "target": "OT",
            "seq_len": 96,
        },
    ]
    for ablation in ablations:
        cfg = copy.deepcopy(base_config)
        cfg["seq_len"] = ablation.pop("seq_len")
        cfg["epochs"] = max(4, int(base_config["epochs"]) // 2)
        print(f"[ABLATION] {ablation['name']}")
        metadata = train_experiment(ablation, cfg)
        rows.append(
            {
                "protocol": "ablation",
                "seed": cfg["seed"],
                "name": metadata["name"],
                "baseName": metadata["kind"],
                "kind": metadata["kind"],
                "dataset": metadata["dataset"],
                "seqLen": cfg["seq_len"],
                **flatten_metrics(metadata),
            }
        )
    return rows


def run_cross_dataset_transfer(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    pairs = [
        ("hf_ett", "hf_weather", "dlinear"),
        ("hf_weather", "hf_ett", "dlinear"),
        ("hf_ett", "hf_weather", "patchtst"),
        ("hf_weather", "hf_ett", "patchtst"),
    ]
    for source, target, model_kind in pairs:
        set_seed(int(base_config["seed"]))
        source_spec = load_named_frame(source, nrows=base_config["max_rows"].get(source))
        target_spec = load_named_frame(target, nrows=base_config["max_rows"].get(target))
        feature_count = min(8, len(source_spec.numeric_columns), len(target_spec.numeric_columns))
        seq_len = int(base_config["seq_len"])
        horizon = int(base_config["horizon"])
        source_ds = WindowForecastDataset(
            source_spec,
            seq_len,
            horizon,
            feature_columns=source_spec.numeric_columns[:feature_count],
            target_column=source_spec.numeric_columns[feature_count - 1],
        )
        target_ds = WindowForecastDataset(
            target_spec,
            seq_len,
            horizon,
            feature_columns=target_spec.numeric_columns[:feature_count],
            target_column=target_spec.numeric_columns[feature_count - 1],
        )
        train, val, _ = split_dataset(source_ds)
        _, _, target_test = split_dataset(target_ds)
        model = (
            DLinearForecaster(seq_len=seq_len, input_dim=feature_count)
            if model_kind == "dlinear"
            else PatchTSTForecaster(input_dim=feature_count, seq_len=seq_len)
        )
        print(f"[TRANSFER] {model_kind} {source} -> {target}")
        train_metrics = train_supervised(
            model,
            train,
            val,
            epochs=max(4, int(base_config["epochs"]) // 2),
            batch_size=int(base_config["batch_size"]),
            lr=float(base_config["lr"]),
        )
        transfer_metrics = evaluate_supervised(model, target_test, batch_size=int(base_config["batch_size"]))
        rows.append(
            {
                "protocol": "cross_dataset_transfer",
                "sourceDataset": source,
                "targetDataset": target,
                "kind": model_kind,
                "featureCount": feature_count,
                "sourceValRmse": train_metrics["best"].get("rmse"),
                **transfer_metrics,
            }
        )
    return rows


def run_streaming_benchmark() -> list[dict[str, Any]]:
    rows = []
    enhanced_runs = [path for path in CHECKPOINTS.iterdir() if path.is_dir() and path.name.startswith("q1_")]
    for run_dir in enhanced_runs:
        checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        model = build_model(metadata)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for batch_size in [1, 16, 64]:
            args = sample_args(metadata, batch_size)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(*args)
                started = time.perf_counter()
                repeats = 200
                for _ in range(repeats):
                    _ = model(*args)
                elapsed = time.perf_counter() - started
            rows.append(
                {
                    "protocol": "streaming_benchmark",
                    "name": run_dir.name,
                    "kind": metadata["kind"],
                    "dataset": metadata["dataset"],
                    "batchSize": batch_size,
                    "meanLatencyMs": elapsed / repeats * 1000,
                    "eventsPerSecond": batch_size * repeats / elapsed,
                    "parameterCount": metadata.get("production", {}).get("parameterCount"),
                }
            )
    return rows


def sample_args(metadata: dict[str, Any], batch_size: int) -> tuple[torch.Tensor, ...]:
    seq_len = int(metadata["seqLen"])
    if metadata["kind"] == "stgcn_forecast":
        nodes = len(metadata["task"]["nodeColumns"])
        adjacency = torch.tensor(metadata["task"]["adjacency"], dtype=torch.float32)
        return torch.zeros(batch_size, nodes, seq_len), adjacency
    features = len(metadata["task"].get("featureColumns", []))
    x = torch.zeros(batch_size, seq_len, features)
    if metadata["kind"] == "saits_imputer":
        return x, torch.ones_like(x)
    return (x,)


def flatten_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = metadata.get("metrics", {})
    final = metrics.get("final", {})
    test = metrics.get("test", {})
    production = metadata.get("production", {})
    return {
        "valLoss": final.get("val_loss"),
        "valMae": final.get("mae"),
        "valRmse": final.get("rmse"),
        "valAccuracy": final.get("accuracy"),
        "valMacroF1": final.get("macro_f1"),
        "testLoss": test.get("test_loss"),
        "testMae": test.get("test_mae"),
        "testRmse": test.get("test_rmse"),
        "testAccuracy": test.get("test_accuracy"),
        "testMacroF1": test.get("test_macro_f1"),
        "testMaskedMse": test.get("test_masked_mse"),
        "parameterCount": production.get("parameterCount"),
        "meanLatencyMs": production.get("latency", {}).get("meanLatencyMs"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(MODELS_ROOT / "configs" / "phase3_multiseed_suite.yaml"),
    )
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml(Path(args.config))
    base_config["max_rows"] = {
        **{
            "astana": 30000,
            "hf_ett": 17420,
            "hf_weather": 50000,
            "hf_traffic": 17544,
            "pems_metr_la": 12000,
            "pems_bay": 12000,
        },
        **(base_config.get("max_rows") or {}),
    }
    base_config["epochs"] = int(base_config.get("epochs", 8))
    seeds = [int(seed) for seed in base_config.get("seeds", [7, 42, 2026])]

    multi_seed = run_multi_seed(base_config, seeds)
    ablation = run_ablation(base_config)
    transfer = run_cross_dataset_transfer(base_config)
    streaming = run_streaming_benchmark()

    write_csv(TABLES / "multi_seed_results.csv", multi_seed)
    write_csv(TABLES / "ablation_results.csv", ablation)
    write_csv(TABLES / "cross_dataset_transfer.csv", transfer)
    write_csv(TABLES / "streaming_benchmark.csv", streaming)

    summary = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "protocols": {
            "multiSeedRuns": len(multi_seed),
            "ablationRuns": len(ablation),
            "transferRuns": len(transfer),
            "streamingRows": len(streaming),
            "seeds": seeds,
        },
    }
    save_json(REPORTS / "paper_experiment_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
