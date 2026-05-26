from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch


MODELS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODELS_ROOT.parent
sys.path.insert(0, str(MODELS_ROOT))

from paper_experiments import (  # noqa: E402
    CHECKPOINTS,
    REPORTS,
    TABLES,
    flatten_metrics,
    run_streaming_benchmark,
    save_json,
    write_csv,
)


PORTFOLIO_PATH = MODELS_ROOT / "reports" / "production_portfolio.json"
DEFAULT_SEEDS = [42, 7, 2026]


def load_portfolio() -> list[dict[str, Any]]:
    portfolio = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    return portfolio.get("models", [])


def harvest_multi_seed_rows(portfolio: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in portfolio:
        base_name = item["run"]
        for seed in seeds:
            run_name = f"{base_name}_seed{seed}"
            metadata_path = CHECKPOINTS / run_name / "metadata.json"
            if not metadata_path.exists():
                print(f"[WARN] Missing seed checkpoint: {run_name}")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "protocol": "multi_seed",
                    "seed": seed,
                    "name": run_name,
                    "baseName": base_name,
                    "kind": metadata["kind"],
                    "dataset": metadata["dataset"],
                    "productionSlot": item.get("slot"),
                    **flatten_metrics(metadata),
                }
            )
    return rows


def metric_for_row(row: dict[str, Any]) -> tuple[str, float] | None:
    if row.get("testRmse") is not None:
        return ("testRmse", float(row["testRmse"]))
    if row.get("testMacroF1") is not None:
        return ("testMacroF1", float(row["testMacroF1"]))
    if row.get("testMaskedMse") is not None:
        return ("testMaskedMse", float(row["testMaskedMse"]))
    return None


def aggregate_multi_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["baseName"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for base_name, group in sorted(grouped.items()):
        metric_name = None
        values: list[float] = []
        for row in group:
            parsed = metric_for_row(row)
            if not parsed:
                continue
            metric_name, value = parsed
            values.append(value)
        if not values or not metric_name:
            continue
        sample = group[0]
        std = stdev(values) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        aggregates.append(
            {
                "baseName": base_name,
                "kind": sample["kind"],
                "dataset": sample["dataset"],
                "productionSlot": sample.get("productionSlot"),
                "metric": metric_name,
                "n": len(values),
                "mean": mean(values),
                "std": std,
                "ci95": ci95,
                "min": min(values),
                "max": max(values),
            }
        )
    return aggregates


def write_aggregate_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "baseName",
        "productionSlot",
        "kind",
        "dataset",
        "metric",
        "n",
        "mean",
        "std",
        "ci95",
        "min",
        "max",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("baseName", "")),
                    str(row.get("productionSlot", "")),
                    str(row.get("kind", "")),
                    str(row.get("dataset", "")),
                    str(row.get("metric", "")),
                    str(row.get("n", "")),
                    f"{row.get('mean', 0):.4f}",
                    f"{row.get('std', 0):.4f}",
                    f"{row.get('ci95', 0):.4f}",
                    f"{row.get('min', 0):.4f}",
                    f"{row.get('max', 0):.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_streaming_rows(rows: list[dict[str, Any]], portfolio: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {item["run"] for item in portfolio}
    return [row for row in rows if row.get("name") in allowed]


def upload_portfolio_to_hf() -> dict[str, Any]:
    script = MODELS_ROOT / "upload_checkpoints_to_hf.py"
    result = subprocess.run(
        [sys.executable, str(script), "--portfolio"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return {"exitCode": result.returncode, "stdout": result.stdout[-4000:]}


def main() -> None:
    if not PORTFOLIO_PATH.exists():
        raise SystemExit(f"Missing {PORTFOLIO_PATH}. Run Phase 2 prep first.")

    portfolio = load_portfolio()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3 core: harvest multi-seed metrics from checkpoints ===")
    multi_seed = harvest_multi_seed_rows(portfolio, DEFAULT_SEEDS)
    aggregates = aggregate_multi_seed(multi_seed)
    write_csv(TABLES / "multi_seed_results.csv", multi_seed)
    write_csv(TABLES / "multi_seed_aggregate.csv", aggregates)
    write_aggregate_markdown(TABLES / "multi_seed_aggregate.md", aggregates)

    print("=== Phase 3 core: streaming latency benchmark (production models) ===")
    streaming = filter_streaming_rows(run_streaming_benchmark(), portfolio)
    write_csv(TABLES / "streaming_benchmark.csv", streaming)

    print("=== Phase 3 core: upload production portfolio to Hugging Face ===")
    upload_result = upload_portfolio_to_hf()

    summary = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "phase": 3,
        "portfolioModels": len(portfolio),
        "multiSeedRows": len(multi_seed),
        "aggregateRows": len(aggregates),
        "streamingRows": len(streaming),
        "cudaAvailable": torch.cuda.is_available(),
        "artifacts": {
            "multiSeedResults": str(TABLES / "multi_seed_results.csv"),
            "multiSeedAggregate": str(TABLES / "multi_seed_aggregate.csv"),
            "streamingBenchmark": str(TABLES / "streaming_benchmark.csv"),
            "hfManifest": str(MODELS_ROOT / "reports" / "huggingface_model_manifest.json"),
        },
        "upload": upload_result,
    }
    save_json(REPORTS / "phase3_core_summary.json", summary)
    print(json.dumps(summary, indent=2))

    if upload_result["exitCode"] != 0:
        raise SystemExit("Hugging Face upload failed")


if __name__ == "__main__":
    main()
