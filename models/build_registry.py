from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "models" / "checkpoints"
REPORTS = ROOT / "models" / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(metadata: dict[str, Any]) -> float:
    test = metadata.get("metrics", {}).get("test", {})
    if "test_macro_f1" in test:
        return float(test["test_macro_f1"])
    if "test_rmse" in test:
        return -float(test["test_rmse"])
    if "test_masked_mse" in test:
        return -float(test["test_masked_mse"])
    return 0.0


def main() -> None:
    entries = []
    for run_dir in sorted(CHECKPOINTS.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "metadata.json").exists():
            continue
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        files = {}
        for name in ["model.pt", "model.safetensors", "model.torchscript.pt", "metadata.json", "model_card.md"]:
            path = run_dir / name
            if path.exists():
                files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        entries.append(
            {
                "name": run_dir.name,
                "kind": metadata["kind"],
                "dataset": metadata["dataset"],
                "score": score(metadata),
                "metrics": metadata.get("metrics", {}),
                "production": metadata.get("production", {}),
                "files": files,
            }
        )
    leaderboard = sorted(entries, key=lambda item: (item["kind"], -item["score"]))
    registry = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "checkpointRoot": str(CHECKPOINTS),
        "count": len(entries),
        "entries": leaderboard,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "production_model_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    (CHECKPOINTS / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(entries), "registry": str(REPORTS / "production_model_registry.json")}, indent=2))


if __name__ == "__main__":
    main()
