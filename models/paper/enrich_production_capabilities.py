from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
PORTFOLIO_PATH = MODELS / "reports" / "production_portfolio.json"


def merge_capabilities(metadata: dict[str, Any], portfolio_item: dict[str, Any]) -> dict[str, Any]:
    capabilities = portfolio_item.get("capabilities", {})
    enriched = dict(metadata)
    enriched["production"] = {
        **(metadata.get("production") or {}),
        "official": True,
        "slot": portfolio_item["slot"],
        "priority": portfolio_item["priority"],
        "portfolioVersion": "phase2",
    }
    enriched["capabilities"] = {
        **capabilities,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    return enriched


def main() -> None:
    if not PORTFOLIO_PATH.exists():
        raise SystemExit(f"Missing portfolio: {PORTFOLIO_PATH}. Run select_production_portfolio.py first.")

    portfolio = json.loads(PORTFOLIO_PATH.read_text(encoding="utf-8"))
    updated: list[str] = []
    for item in portfolio.get("models", []):
        run_name = item["run"]
        metadata_path = CHECKPOINTS / run_name / "metadata.json"
        if not metadata_path.exists():
            print(f"[WARN] Missing checkpoint metadata for {run_name}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        enriched = merge_capabilities(metadata, item)
        metadata_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        updated.append(run_name)

    print(json.dumps({"updated": len(updated), "runs": updated}, indent=2))


if __name__ == "__main__":
    main()
