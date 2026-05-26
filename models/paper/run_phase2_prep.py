from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
PAPER = MODELS / "paper"
REPORTS = MODELS / "reports"


def run_step(label: str, script: Path) -> dict[str, str]:
    print(f"\n=== {label} ===")
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=False)
    return {"label": label, "script": str(script), "exitCode": str(result.returncode)}


def main() -> None:
    steps = [
        run_step("Prepare benchmark datasets", PAPER / "prepare_benchmark_datasets.py"),
        run_step("Select production portfolio", PAPER / "select_production_portfolio.py"),
        run_step("Enrich checkpoint capabilities", PAPER / "enrich_production_capabilities.py"),
    ]

    registry_script = MODELS / "build_registry.py"
    if registry_script.exists():
        steps.append(run_step("Rebuild model registry", registry_script))

    summary = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "phase": 2,
        "steps": steps,
        "artifacts": {
            "datasetManifest": str(PAPER / "reports" / "dataset_manifest.json"),
            "productionPortfolio": str(REPORTS / "production_portfolio.json"),
            "productionRegistry": str(REPORTS / "production_model_registry.json"),
            "phase3Config": str(MODELS / "configs" / "phase3_multiseed_suite.yaml"),
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase2_prep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    failed = [step for step in steps if step["exitCode"] != "0"]
    if failed:
        raise SystemExit(f"Phase 2 prep completed with failures: {[step['label'] for step in failed]}")


if __name__ == "__main__":
    main()
