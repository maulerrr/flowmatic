from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PAPER = MODELS / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
REPORTS = PAPER / "reports"
CHECKPOINTS = MODELS / "checkpoints"


sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_rows() -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(CHECKPOINTS.iterdir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        test = metadata.get("metrics", {}).get("test", {})
        final = metadata.get("metrics", {}).get("final", {})
        production = metadata.get("production", {})
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
                "val_loss": final.get("val_loss"),
                "parameters": production.get("parameterCount"),
                "latency_ms": production.get("latency", {}).get("meanLatencyMs"),
                "torchscript": production.get("torchscriptExported"),
            }
        )
    return rows


def savefig(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.savefig(FIGURES / f"{name}.svg")
    plt.close()


def plot_forecast_leaderboard(df: pd.DataFrame) -> None:
    forecast = df[df["test_rmse"].notna()].copy()
    forecast = forecast.sort_values("test_rmse").head(16)
    plt.figure(figsize=(11, 7))
    sns.barplot(data=forecast, y="name", x="test_rmse", hue="dataset", dodge=False)
    plt.title("Forecasting and Reconstruction Test RMSE")
    plt.xlabel("Test RMSE, lower is better")
    plt.ylabel("")
    savefig("forecast_rmse_leaderboard")


def plot_classification(df: pd.DataFrame) -> None:
    cls = df[df["test_macro_f1"].notna()].copy().sort_values("test_macro_f1", ascending=False)
    if cls.empty:
        return
    plt.figure(figsize=(9, 4))
    sns.barplot(data=cls, x="test_macro_f1", y="name", color="#2a9d8f")
    plt.title("Classification Macro-F1")
    plt.xlabel("Test macro-F1, higher is better")
    plt.ylabel("")
    savefig("classification_macro_f1")


def plot_latency(df: pd.DataFrame) -> None:
    prod = df[df["latency_ms"].notna() & df["parameters"].notna()].copy()
    if prod.empty:
        return
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=prod, x="parameters", y="latency_ms", hue="kind", s=120)
    for _, row in prod.iterrows():
        plt.text(row["parameters"], row["latency_ms"], row["name"].replace("q1_", "")[:16], fontsize=7)
    plt.xscale("log")
    plt.title("Production Latency vs Parameter Count")
    plt.xlabel("Trainable parameters, log scale")
    plt.ylabel("Mean latency, ms")
    savefig("latency_vs_parameters")


def plot_training_curves() -> None:
    rows = []
    for run_dir in sorted(CHECKPOINTS.iterdir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists() or not run_dir.name.startswith("q1_"):
            continue
        metadata = load_json(metadata_path)
        for item in metadata.get("metrics", {}).get("history", []):
            rows.append(
                {
                    "name": run_dir.name,
                    "epoch": item["epoch"],
                    "train_loss": item["train_loss"],
                    "val_loss": item["val_loss"],
                }
            )
    if not rows:
        return
    curves = pd.DataFrame(rows)
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=curves, x="epoch", y="val_loss", hue="name", linewidth=1.7)
    plt.title("Enhanced Q1 Suite Validation Curves")
    plt.ylabel("Validation loss")
    savefig("validation_curves_q1_suite")


def plot_multi_seed() -> None:
    path = TABLES / "multi_seed_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metric = "testRmse"
    subset = df[df[metric].notna()].copy()
    if subset.empty:
        return
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=subset, x="kind", y=metric)
    plt.xticks(rotation=30, ha="right")
    plt.title("Multi-Seed Robustness by Architecture")
    plt.ylabel("Test RMSE")
    savefig("multi_seed_rmse_boxplot")


def plot_ablation() -> None:
    path = TABLES / "ablation_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    subset = df[df["testRmse"].notna()].copy()
    if subset.empty:
        return
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=subset, x="seqLen", y="testRmse", hue="kind", marker="o")
    plt.title("Sequence Length Ablation")
    plt.ylabel("Test RMSE")
    savefig("sequence_length_ablation")


def plot_transfer() -> None:
    path = TABLES / "cross_dataset_transfer.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df["pair"] = df["sourceDataset"] + " -> " + df["targetDataset"]
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="pair", y="test_rmse", hue="kind")
    plt.xticks(rotation=20, ha="right")
    plt.title("Cross-Dataset Zero-Shot Transfer")
    plt.ylabel("Target test RMSE")
    savefig("cross_dataset_transfer")


def plot_streaming() -> None:
    path = TABLES / "streaming_benchmark.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=df, x="batchSize", y="eventsPerSecond", hue="kind", marker="o")
    plt.xscale("log", base=2)
    plt.title("Streaming Throughput Benchmark")
    plt.ylabel("Events per second")
    savefig("streaming_throughput")

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=df, x="batchSize", y="meanLatencyMs", hue="kind", marker="o")
    plt.xscale("log", base=2)
    plt.title("Streaming Latency Benchmark")
    plt.ylabel("Mean latency, ms")
    savefig("streaming_latency")


def write_tables(df: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "checkpoint_leaderboard.csv", index=False)
    metric_columns = [
        "name",
        "kind",
        "dataset",
        "test_rmse",
        "test_mae",
        "test_accuracy",
        "test_macro_f1",
        "test_masked_mse",
        "parameters",
        "latency_ms",
        "torchscript",
    ]
    markdown = markdown_table(df[metric_columns].sort_values(["dataset", "kind", "name"]))
    (TABLES / "checkpoint_leaderboard.md").write_text(markdown, encoding="utf-8")
    write_multi_seed_aggregate()


def write_multi_seed_aggregate() -> None:
    path = TABLES / "multi_seed_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metric = "testRmse"
    rows = []
    for (base_name, kind, dataset), group in df.groupby(["baseName", "kind", "dataset"]):
        values = group[metric].dropna()
        if values.empty:
            values = group["testMacroF1"].dropna()
            used_metric = "testMacroF1"
        else:
            used_metric = metric
        if values.empty:
            values = group["testMaskedMse"].dropna()
            used_metric = "testMaskedMse"
        if values.empty:
            continue
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(
            {
                "baseName": base_name,
                "kind": kind,
                "dataset": dataset,
                "metric": used_metric,
                "n": len(values),
                "mean": values.mean(),
                "std": std,
                "ci95": 1.96 * std / (len(values) ** 0.5),
                "min": values.min(),
                "max": values.max(),
            }
        )
    out = pd.DataFrame(rows).sort_values(["dataset", "kind", "baseName"])
    out.to_csv(TABLES / "multi_seed_aggregate.csv", index=False)
    (TABLES / "multi_seed_aggregate.md").write_text(markdown_table(out), encoding="utf-8")


def write_report(df: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    best_forecast = df[df["test_rmse"].notna()].sort_values("test_rmse").head(5)
    best_cls = df[df["test_macro_f1"].notna()].sort_values("test_macro_f1", ascending=False).head(3)
    content = [
        "# Flowmatic Q1 Experiment Report",
        "",
        "## Scope",
        "This report summarizes neural forecasting, imputation, anomaly detection, classification, graph forecasting, multi-seed robustness, ablations, cross-dataset transfer, and streaming benchmarks.",
        "",
        "## Best Forecasting/Reconstruction Runs",
        markdown_table(best_forecast[["name", "kind", "dataset", "test_rmse", "test_mae"]]),
        "",
        "## Best Classification Runs",
        markdown_table(best_cls[["name", "kind", "dataset", "test_accuracy", "test_macro_f1"]]),
        "",
        "## Figures",
        "- `figures/forecast_rmse_leaderboard.png`",
        "- `figures/classification_macro_f1.png`",
        "- `figures/latency_vs_parameters.png`",
        "- `figures/validation_curves_q1_suite.png`",
        "- `figures/multi_seed_rmse_boxplot.png`",
        "- `figures/sequence_length_ablation.png`",
        "- `figures/cross_dataset_transfer.png`",
        "- `figures/streaming_throughput.png`",
        "- `figures/streaming_latency.png`",
        "",
        "## Publication Caveat",
        "These are now reproducible research artifacts. For submission, run exact official repositories with matched dataset splits where licensing/dependencies allow, then cite the local implementation as the production backbone.",
    ]
    (REPORTS / "Q1_EXPERIMENT_REPORT.md").write_text("\n".join(content), encoding="utf-8")
    write_protocol_report()


def write_protocol_report() -> None:
    content = [
        "# Research Protocol and Official Baseline Parity",
        "",
        "## Completed Locally",
        "- Full-row enhanced training protocol over Astana, ETT, Weather, and Traffic datasets.",
        "- Three-seed robustness protocol with seeds `7`, `42`, and `2026`.",
        "- Sequence-length ablations for PatchTST-style and DLinear-style models.",
        "- Cross-dataset zero-shot transfer tests for DLinear-style and PatchTST-style models.",
        "- Streaming throughput and latency benchmarks over batch sizes `1`, `16`, and `64`.",
        "- Production bundles with `.pt`, `.safetensors`, TorchScript, metadata, checksums, and model cards.",
        "",
        "## Local Parity Implementations",
        "| Family | Local run kind | Paper/reference intent | Status |",
        "| --- | --- | --- | --- |",
        "| PatchTST | `patchtst_forecast` | Channel-independent patch transformer | Local reproduction implemented |",
        "| iTransformer | `itransformer_forecast` | Variables as tokens, temporal history as features | Local reproduction implemented |",
        "| TimesNet | `timesblock_forecast` | Multi-period 2D temporal variation modeling | Local inspired reproduction implemented |",
        "| DLinear/NLinear | `dlinear_forecast`, `nlinear_forecast` | Linear decomposition and normalization baselines | Local reproduction implemented |",
        "| SAITS | `saits_imputer` | Self-attention masked time-series imputation | Local inspired reproduction implemented |",
        "| TranAD | `tranad_anomaly` | Transformer residual-conditioned anomaly reconstruction | Local inspired reproduction implemented |",
        "| STGCN | `stgcn_forecast` | Spatio-temporal graph convolution forecasting | Local inspired reproduction implemented |",
        "",
        "## Exact Official Baseline Step Before Submission",
        "For a Q1 submission, exact official upstream repositories should be run with identical splits and budgets, then included as a separate comparison block. The current codebase intentionally keeps local implementations small enough for production use and backend integration; that is not the same as claiming official SOTA parity.",
        "",
        "Recommended official checks:",
        "- PatchTST official implementation against ETT/Weather/Traffic splits.",
        "- Time-Series-Library implementations of TimesNet, iTransformer, DLinear, and NLinear.",
        "- SAITS/BRITS official imputation baselines on masked smart-city streams.",
        "- TranAD/Anomaly Transformer/USAD anomaly baselines with injected and real anomalies.",
        "- STGCN/DCRNN/Graph WaveNet/STAEformer on traffic graph datasets.",
        "",
        "## Paper Figures",
        "Use SVG files for vector publication where possible, PNG files for quick manuscript previews.",
    ]
    (REPORTS / "RESEARCH_PROTOCOL_AND_BASELINE_PARITY.md").write_text("\n".join(content), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append([format_cell(row[column]) for column in columns])
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rows)) if rows else len(str(column))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(checkpoint_rows())
    write_tables(df)
    plot_forecast_leaderboard(df)
    plot_classification(df)
    plot_latency(df)
    plot_training_curves()
    plot_multi_seed()
    plot_ablation()
    plot_transfer()
    plot_streaming()
    write_report(df)
    print(json.dumps({"figures": len(list(FIGURES.glob("*.png"))), "tables": len(list(TABLES.glob("*")))}, indent=2))


if __name__ == "__main__":
    main()
