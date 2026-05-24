from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from flowml.data import (
    GraphForecastDataset,
    MaskedImputationDataset,
    SequenceClassificationDataset,
    SequenceReconstructionDataset,
    WindowForecastDataset,
    load_named_frame,
    save_json,
    split_dataset,
)
from flowml.models import (
    DLinearForecaster,
    GRUForecaster,
    ITransformerForecaster,
    NLinearForecaster,
    PatchTSTForecaster,
    STGCNForecaster,
    SAITSLikeImputer,
    SequenceAutoencoder,
    TCNForecaster,
    TimesBlockForecaster,
    TranADLikeAnomalyDetector,
    TransformerClassifier,
    TransformerForecaster,
)
from flowml.training import (
    benchmark_latency,
    count_parameters,
    evaluate_masked_imputer,
    evaluate_supervised,
    export_torchscript,
    save_checkpoint,
    train_autoencoder,
    train_masked_imputer,
    train_supervised,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "models"
CHECKPOINT_ROOT = MODELS_ROOT / "checkpoints"
REPORT_ROOT = MODELS_ROOT / "reports"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def model_card(run_dir: Path, metadata: dict[str, Any]) -> None:
    lines = [
        f"# {metadata['name']}",
        "",
        "## Purpose",
        metadata["purpose"],
        "",
        "## Dataset",
        f"- Name: `{metadata['dataset']}`",
        f"- Source: `{metadata['source']}`",
        f"- Rows used: `{metadata['rows']}`",
        "",
        "## Architecture",
        f"`{metadata['kind']}`",
        "",
        "## Metrics",
        "```json",
        json.dumps(metadata["metrics"], indent=2),
        "```",
        "",
        "## Production Metadata",
        "```json",
        json.dumps(metadata.get("production", {}), indent=2),
        "```",
        "",
        "## Intended Use",
        "Research backbone for Flowmatic smart-city preprocessing and core processing-unit model selection.",
        "",
        "## Limitations",
        "This is a local research checkpoint. Run larger training, ablations, and external validation before publication claims.",
    ]
    (run_dir / "model_card.md").write_text("\n".join(lines), encoding="utf-8")


def train_experiment(exp: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    name = exp["name"]
    kind = exp["kind"]
    dataset_name = exp["dataset"]
    spec = load_named_frame(dataset_name, nrows=cfg["max_rows"].get(dataset_name))
    run_dir = CHECKPOINT_ROOT / name
    seq_len = int(cfg["seq_len"])
    horizon = int(cfg["horizon"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])

    graph_adjacency = None
    sample_args: tuple[torch.Tensor, ...]

    if kind in {
        "tcn_forecast",
        "gru_forecast",
        "transformer_forecast",
        "dlinear_forecast",
        "nlinear_forecast",
        "patchtst_forecast",
        "itransformer_forecast",
        "timesblock_forecast",
    }:
        target = exp.get("target") or spec.target_column
        feature_columns = spec.numeric_columns[:16]
        if target not in feature_columns:
            feature_columns = (
                feature_columns[:15] + [target]
                if len(feature_columns) >= 16
                else feature_columns + [target]
            )
        dataset = WindowForecastDataset(spec, seq_len, horizon, feature_columns=feature_columns, target_column=target)
        train, val, test = split_dataset(dataset)
        input_dim = len(dataset.feature_columns)
        if kind == "tcn_forecast":
            model = TCNForecaster(input_dim=input_dim)
        elif kind == "gru_forecast":
            model = GRUForecaster(input_dim=input_dim)
        elif kind == "dlinear_forecast":
            model = DLinearForecaster(seq_len=seq_len, input_dim=input_dim)
        elif kind == "nlinear_forecast":
            model = NLinearForecaster(seq_len=seq_len, input_dim=input_dim)
        elif kind == "patchtst_forecast":
            model = PatchTSTForecaster(input_dim=input_dim, seq_len=seq_len)
        elif kind == "itransformer_forecast":
            model = ITransformerForecaster(input_dim=input_dim, seq_len=seq_len)
        elif kind == "timesblock_forecast":
            model = TimesBlockForecaster(input_dim=input_dim, seq_len=seq_len)
        else:
            model = TransformerForecaster(input_dim=input_dim)
        result = train_supervised(model, train, val, epochs=epochs, batch_size=batch_size, lr=lr)
        result["test"] = evaluate_supervised(model, test, batch_size=batch_size)
        sample_args = (torch.zeros(1, seq_len, input_dim),)
        task_metadata = {
            "target": target,
            "featureColumns": dataset.feature_columns,
            "scaler": dataset.scaler.to_dict(),
            "targetIndex": dataset.target_index,
        }
        purpose = "Forecast smart-city sensor values for core processing-unit routing and prediction."

    elif kind == "autoencoder":
        dataset = SequenceReconstructionDataset(spec, seq_len, feature_columns=spec.numeric_columns[:16])
        train, val, test = split_dataset(dataset)
        model = SequenceAutoencoder(input_dim=len(dataset.feature_columns))
        result = train_autoencoder(model, train, val, epochs=epochs, batch_size=batch_size, lr=lr)
        result["test"] = evaluate_supervised(model, test, batch_size=batch_size)
        sample_args = (torch.zeros(1, seq_len, len(dataset.feature_columns)),)
        task_metadata = {
            "featureColumns": dataset.feature_columns,
            "scaler": dataset.scaler.to_dict(),
            "anomalyScore": "mean squared reconstruction error",
        }
        purpose = "Repair noisy/missing smart-city streams and detect reconstruction anomalies."

    elif kind == "tranad_anomaly":
        dataset = SequenceReconstructionDataset(spec, seq_len, feature_columns=spec.numeric_columns[:16])
        train, val, test = split_dataset(dataset)
        model = TranADLikeAnomalyDetector(input_dim=len(dataset.feature_columns))
        result = train_autoencoder(model, train, val, epochs=epochs, batch_size=batch_size, lr=lr)
        result["test"] = evaluate_supervised(model, test, batch_size=batch_size)
        sample_args = (torch.zeros(1, seq_len, len(dataset.feature_columns)),)
        task_metadata = {
            "featureColumns": dataset.feature_columns,
            "scaler": dataset.scaler.to_dict(),
            "anomalyScore": "residual-conditioned transformer reconstruction error",
        }
        purpose = "Transformer anomaly detection for corrupted smart-city stream windows."

    elif kind == "saits_imputer":
        dataset = MaskedImputationDataset(spec, seq_len, feature_columns=spec.numeric_columns[:16], mask_ratio=0.25)
        train, val, test = split_dataset(dataset)
        model = SAITSLikeImputer(input_dim=len(dataset.feature_columns))
        result = train_masked_imputer(model, train, val, epochs=epochs, batch_size=batch_size, lr=lr)
        result["test"] = evaluate_masked_imputer(model, test, batch_size=batch_size)
        sample_args = (
            torch.zeros(1, seq_len, len(dataset.feature_columns)),
            torch.ones(1, seq_len, len(dataset.feature_columns)),
        )
        task_metadata = {
            "featureColumns": dataset.feature_columns,
            "scaler": dataset.scaler.to_dict(),
            "maskRatio": dataset.mask_ratio,
            "objective": "masked value reconstruction",
        }
        purpose = "Self-attention imputation for missing smart-city sensor values."

    elif kind == "transformer_classifier":
        dataset = SequenceClassificationDataset(spec, seq_len, label_column=exp.get("label"))
        train, val, test = split_dataset(dataset)
        model = TransformerClassifier(input_dim=len(dataset.feature_columns), num_classes=len(dataset.class_to_id))
        result = train_supervised(
            model,
            train,
            val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            task="classification",
        )
        result["test"] = evaluate_supervised(
            model,
            test,
            batch_size=batch_size,
            task="classification",
        )
        sample_args = (torch.zeros(1, seq_len, len(dataset.feature_columns)),)
        task_metadata = {
            "featureColumns": dataset.feature_columns,
            "labelColumn": dataset.label_column,
            "classToId": dataset.class_to_id,
            "idToClass": dataset.id_to_class,
            "scaler": dataset.scaler.to_dict(),
        }
        purpose = "Classify event severity/type from streaming smart-city telemetry windows."

    elif kind == "stgcn_forecast":
        dataset = GraphForecastDataset(
            spec,
            seq_len,
            horizon,
            max_nodes=int(exp.get("max_nodes", 24)),
        )
        train, val, test = split_dataset(dataset)
        model = STGCNForecaster(nodes=len(dataset.node_columns), seq_len=seq_len)
        adjacency = torch.tensor(dataset.adjacency, dtype=torch.float32)
        graph_adjacency = adjacency
        result = train_supervised(
            model,
            train,
            val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            graph_adjacency=adjacency,
        )
        result["test"] = evaluate_supervised(
            model,
            test,
            batch_size=batch_size,
            graph_adjacency=adjacency,
        )
        sample_args = (torch.zeros(1, len(dataset.node_columns), seq_len), adjacency)
        task_metadata = {
            "nodeColumns": dataset.node_columns,
            "adjacency": dataset.adjacency.tolist(),
            "scaler": dataset.scaler.to_dict(),
        }
        purpose = "Forecast graph-structured traffic sensor states using spatial correlations."

    else:
        raise ValueError(f"Unknown experiment kind: {kind}")

    metadata = {
        "name": name,
        "kind": kind,
        "dataset": dataset_name,
        "source": spec.source,
        "rows": len(spec.frame),
        "seqLen": seq_len,
        "horizon": horizon,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "metrics": result,
        "task": task_metadata,
        "purpose": purpose,
        "production": {
            "parameterCount": count_parameters(model),
            "latency": benchmark_latency(model, sample_args),
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(run_dir / "model.pt", model, metadata)
    metadata["production"]["torchscriptExported"] = export_torchscript(
        run_dir / "model.torchscript.pt",
        model,
        sample_args,
    )
    save_json(run_dir / "metadata.json", metadata)
    model_card(run_dir, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/configs/q1_core_suite.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg["seed"]))
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    for exp in cfg["experiments"]:
        print(f"[TRAIN] {exp['name']}")
        summaries.append(train_experiment(exp, cfg))

    suite = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "experiments": [
            {
                "name": item["name"],
                "kind": item["kind"],
                "dataset": item["dataset"],
                "final": item["metrics"]["final"],
                "test": item["metrics"].get("test", {}),
            }
            for item in summaries
        ],
    }
    save_json(REPORT_ROOT / "q1_core_suite_metrics.json", suite)
    print(json.dumps(suite, indent=2))


if __name__ == "__main__":
    main()
