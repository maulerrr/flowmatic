from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flowml.models import (
    DLinearForecaster,
    GRUForecaster,
    ITransformerForecaster,
    NLinearForecaster,
    PatchTSTForecaster,
    SAITSLikeImputer,
    STGCNForecaster,
    SequenceAutoencoder,
    TCNForecaster,
    TimesBlockForecaster,
    TranADLikeAnomalyDetector,
    TransformerClassifier,
    TransformerForecaster,
)


ROOT = Path(__file__).resolve().parents[1]


def build_model(metadata: dict):
    kind = metadata["kind"]
    if kind == "tcn_forecast":
        return TCNForecaster(input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "gru_forecast":
        return GRUForecaster(input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "transformer_forecast":
        return TransformerForecaster(input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "dlinear_forecast":
        return DLinearForecaster(seq_len=metadata["seqLen"], input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "nlinear_forecast":
        return NLinearForecaster(seq_len=metadata["seqLen"], input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "patchtst_forecast":
        return PatchTSTForecaster(input_dim=len(metadata["task"]["featureColumns"]), seq_len=metadata["seqLen"])
    if kind == "itransformer_forecast":
        return ITransformerForecaster(input_dim=len(metadata["task"]["featureColumns"]), seq_len=metadata["seqLen"])
    if kind == "timesblock_forecast":
        return TimesBlockForecaster(input_dim=len(metadata["task"]["featureColumns"]), seq_len=metadata["seqLen"])
    if kind == "autoencoder":
        return SequenceAutoencoder(input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "tranad_anomaly":
        return TranADLikeAnomalyDetector(input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "saits_imputer":
        return SAITSLikeImputer(input_dim=len(metadata["task"]["featureColumns"]))
    if kind == "transformer_classifier":
        return TransformerClassifier(
            input_dim=len(metadata["task"]["featureColumns"]),
            num_classes=len(metadata["task"]["classToId"]),
        )
    if kind == "stgcn_forecast":
        return STGCNForecaster(nodes=len(metadata["task"]["nodeColumns"]), seq_len=metadata["seqLen"])
    raise ValueError(f"Unsupported kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="astana_tcn_density_forecaster")
    args = parser.parse_args()
    run_dir = ROOT / "models" / "checkpoints" / args.run
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    model = build_model(metadata)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    seq_len = metadata["seqLen"]
    if metadata["kind"] == "stgcn_forecast":
        nodes = len(metadata["task"]["nodeColumns"])
        sample = torch.zeros(1, nodes, seq_len)
        adjacency = torch.tensor(metadata["task"]["adjacency"], dtype=torch.float32)
        with torch.no_grad():
            output = model(sample, adjacency).numpy().tolist()
    elif metadata["kind"] == "saits_imputer":
        features = metadata["task"].get("featureColumns", [])
        sample = torch.zeros(1, seq_len, len(features))
        mask = torch.ones_like(sample)
        with torch.no_grad():
            output = model(sample, mask).numpy().tolist()
    else:
        features = metadata["task"].get("featureColumns", [])
        sample = torch.zeros(1, seq_len, len(features))
        with torch.no_grad():
            raw = model(sample)
            output = raw.numpy().tolist()

    print(json.dumps({"run": args.run, "kind": metadata["kind"], "output": output}, indent=2))


if __name__ == "__main__":
    main()
