"""Upload best Flowmatic checkpoints to Hugging Face Hub with clean repo names."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "models" / "checkpoints"
MANIFEST_PATH = ROOT / "models" / "reports" / "huggingface_model_manifest.json"
PORTFOLIO_PATH = ROOT / "models" / "reports" / "production_portfolio.json"

KIND_SLUGS: dict[str, str] = {
    "tranad_anomaly": "tranad-anomaly-detector",
    "transformer_classifier": "transformer-severity-classifier",
    "patchtst_forecast": "patchtst-density-forecaster",
    "itransformer_forecast": "itransformer-speed-forecaster",
    "saits_imputer": "saits-imputer",
    "dlinear_forecast": "dlinear-energy-forecaster",
    "nlinear_forecast": "nlinear-energy-forecaster",
    "stgcn_forecast": "stgcn-forecaster",
    "timesblock_forecast": "timesblock-forecaster",
    "tcn_forecast": "tcn-density-forecaster",
    "gru_forecast": "gru-speed-forecaster",
    "autoencoder": "autoencoder-anomaly-repair",
}

DATASET_SLUGS: dict[str, str] = {
    "astana": "astana",
    "hf_ett": "ett",
    "hf_weather": "weather",
    "hf_traffic": "traffic",
}


def load_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in environment or root .env")


def score(metadata: dict) -> float:
    test = metadata.get("metrics", {}).get("test", {})
    if "test_macro_f1" in test:
        return float(test["test_macro_f1"])
    if "test_accuracy" in test:
        return float(test["test_accuracy"])
    if "test_rmse" in test:
        return -float(test["test_rmse"])
    if "test_masked_mse" in test:
        return -float(test["test_masked_mse"])
    return 0.0


def repo_slug(metadata: dict) -> str:
    kind = metadata.get("kind", "model")
    dataset = metadata.get("dataset", "unknown")
    dataset_part = DATASET_SLUGS.get(dataset, dataset.replace("_", "-"))
    kind_part = KIND_SLUGS.get(kind, kind.replace("_", "-"))
    return f"flowmatic-{dataset_part}-{kind_part}"


def pick_best_runs(use_portfolio: bool = False) -> list[tuple[Path, dict, str]]:
    if use_portfolio and PORTFOLIO_PATH.exists():
        portfolio = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
        selected: list[tuple[Path, dict, str]] = []
        for item in portfolio.get("models", []):
            run_dir = CHECKPOINTS / item["run"]
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                print(f"[WARN] Portfolio run missing: {item['run']}")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not (run_dir / "model.safetensors").exists() and not (run_dir / "model.pt").exists():
                print(f"[WARN] Portfolio run has no weights: {item['run']}")
                continue
            selected.append((run_dir, metadata, repo_slug(metadata)))
        return selected

    groups: dict[tuple[str, str], list[tuple[Path, dict, float]]] = {}
    for run_dir in sorted(CHECKPOINTS.iterdir()):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not (run_dir / "model.safetensors").exists() and not (run_dir / "model.pt").exists():
            continue
        key = (metadata.get("kind", run_dir.name), metadata.get("dataset", "unknown"))
        groups.setdefault(key, []).append((run_dir, metadata, score(metadata)))

    selected: list[tuple[Path, dict, str]] = []
    for (_kind, _dataset), items in sorted(groups.items()):
        run_dir, metadata, _ = max(items, key=lambda item: item[2])
        selected.append((run_dir, metadata, repo_slug(metadata)))
    return selected


def build_readme(repo_slug_name: str, metadata: dict, source_run: str) -> str:
    kind = metadata.get("kind", "unknown")
    dataset = metadata.get("dataset", "unknown")
    purpose = metadata.get("purpose", "Flowmatic smart-city time-series model.")
    test = metadata.get("metrics", {}).get("test", {})
    return f"""---
library_name: flowmatic
tags:
- flowmatic
- smart-city
- time-series
- {dataset}
- {kind}
license: apache-2.0
---

# {repo_slug_name}

{purpose}

## Model summary

| Field | Value |
| --- | --- |
| Architecture | `{kind}` |
| Dataset | `{dataset}` |
| Source checkpoint | `{source_run}` |
| Test metrics | `{json.dumps(test)}` |

## Files

- `model.safetensors` — primary weights
- `model.torchscript.pt` — TorchScript export (when present)
- `metadata.json` — schema, scaler, metrics
- `model_card.md` — training card

## Usage in Flowmatic

Deploy in the Core Unit with model ID `{repo_slug_name}` (prefixed by your Hugging Face username in the UI).

Inference is routed through Flowmatic's model-inference service using your organization Hugging Face token.
"""


def upload_run(api: HfApi, username: str, token: str, run_dir: Path, metadata: dict, slug: str) -> dict:
    repo_id = f"{username}/{slug}"
    create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True, private=False)

    staging = run_dir
    readme = build_readme(slug, metadata, run_dir.name)
    config = {
        "model_type": "flowmatic-timeseries",
        "architecture": metadata.get("kind"),
        "dataset": metadata.get("dataset"),
        "flowmatic_checkpoint": run_dir.name,
        "task": metadata.get("task", {}),
    }

    temp_files = run_dir / ".hf_upload"
    temp_files.mkdir(exist_ok=True)
    (temp_files / "README.md").write_text(readme, encoding="utf-8")
    (temp_files / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    upload_paths = []
    for name in ["model.safetensors", "model.torchscript.pt", "model.pt", "metadata.json", "model_card.md"]:
        path = run_dir / name
        if path.exists():
            upload_paths.append(path)

    for path in upload_paths:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Upload {slug} from {run_dir.name}",
        )

    api.upload_file(
        path_or_fileobj=str(temp_files / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Add README for {slug}",
    )
    api.upload_file(
        path_or_fileobj=str(temp_files / "config.json"),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Add config for {slug}",
    )

    return {
        "repoId": repo_id,
        "slug": slug,
        "sourceRun": run_dir.name,
        "kind": metadata.get("kind"),
        "dataset": metadata.get("dataset"),
        "hubUrl": f"https://huggingface.co/{repo_id}",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Upload only the official Phase 2 production portfolio checkpoints",
    )
    args = parser.parse_args()

    token = load_token()
    api = HfApi(token=token)
    profile = api.whoami(token=token)
    username = profile["name"]
    print(f"Authenticated as @{username}")

    runs = pick_best_runs(use_portfolio=args.portfolio)
    label = "production portfolio" if args.portfolio else "best checkpoint per kind+dataset"
    print(f"Uploading {len(runs)} model families ({label})")

    manifest = {
        "uploadedAt": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "models": [],
    }

    for run_dir, metadata, slug in runs:
        print(f"  -> {slug}  ({run_dir.name})")
        try:
            entry = upload_run(api, username, token, run_dir, metadata, slug)
            manifest["models"].append(entry)
            print(f"     OK  https://huggingface.co/{entry['repoId']}")
        except Exception as exc:
            print(f"     FAIL {slug}: {exc}")
            manifest["models"].append(
                {"slug": slug, "sourceRun": run_dir.name, "error": str(exc)},
            )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
