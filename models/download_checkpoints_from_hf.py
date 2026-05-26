"""Download production portfolio checkpoints from Hugging Face into models/checkpoints/."""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "models" / "checkpoints"
MANIFEST_PATH = ROOT / "models" / "reports" / "huggingface_model_manifest.json"


def load_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if token:
        return token
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("HF_TOKEN=", "HUGGINGFACE_TOKEN=")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to huggingface_model_manifest.json",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have model.safetensors or model.pt",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    token = load_token()
    models = manifest.get("models", [])
    print(f"Downloading {len(models)} models from @{manifest.get('username', 'hub')}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    for item in models:
        if item.get("error"):
            print(f"[SKIP] {item.get('slug')}: prior upload error")
            continue

        repo_id = item["repoId"]
        source_run = item["sourceRun"]
        target_dir = CHECKPOINTS / source_run

        if args.skip_existing and target_dir.exists():
            if (target_dir / "model.safetensors").exists() or (target_dir / "model.pt").exists():
                print(f"[SKIP] {source_run} (weights present)")
                continue

        print(f"  -> {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(target_dir),
            token=token,
        )
        print(f"     OK  {target_dir}")

    print(f"\nCheckpoints root: {CHECKPOINTS}")


if __name__ == "__main__":
    main()
