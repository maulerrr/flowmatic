from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

def _resolve_root() -> Path:
    env_root = os.environ.get("FLOWMATIC_ROOT")
    if env_root:
        return Path(env_root)
    here = Path(__file__).resolve()
    docker_root = here.parents[1]
    if (docker_root / "models").is_dir():
        return docker_root
    return here.parents[3]


ROOT = _resolve_root()
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from inference_engine import load_checkpoint_bundle, predict_payload, resolve_device  # noqa: E402

CHECKPOINTS_DIR = Path(os.environ.get("CHECKPOINTS_DIR", str(MODELS_DIR / "checkpoints")))
HF_HOME = Path(os.environ.get("HF_HOME", "/cache/huggingface"))
FLOWMATIC_CACHE = HF_HOME / "flowmatic-models"
INFER_DEVICE = os.environ.get("INFER_DEVICE", "auto")


@dataclass
class LoadedModel:
    model_id: str
    source: str
    run_dir: Path
    model: object
    metadata: dict
    device: object
    loaded_at: float


class ModelEngine:
    def __init__(self) -> None:
        self._loaded: dict[str, LoadedModel] = {}
        FLOWMATIC_CACHE.mkdir(parents=True, exist_ok=True)

    def device_info(self) -> dict:
        import torch

        device = resolve_device(INFER_DEVICE)
        info = {
            "device": str(device),
            "cudaAvailable": torch.cuda.is_available(),
            "cacheDir": str(FLOWMATIC_CACHE),
            "checkpointsDir": str(CHECKPOINTS_DIR),
        }
        if torch.cuda.is_available():
            info["gpuName"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpuMemoryGb"] = round(props.total_memory / 1_000_000_000, 2)
        return info

    def _local_checkpoint_for_model(self, model_id: str) -> Path | None:
        manifest_path = MODELS_DIR / "reports" / "huggingface_model_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest.get("models", []):
                if entry.get("repoId") == model_id:
                    source_run = entry.get("sourceRun")
                    if source_run:
                        candidate = CHECKPOINTS_DIR / source_run
                        if (candidate / "model.pt").exists() or (candidate / "model.safetensors").exists():
                            return candidate

        repo_slug = model_id.split("/")[-1].lower()
        if not CHECKPOINTS_DIR.exists():
            return None
        tokens = repo_slug.replace("flowmatic-", "").split("-")
        best: tuple[Path, int] | None = None
        for run_dir in CHECKPOINTS_DIR.iterdir():
            if not run_dir.is_dir():
                continue
            if not (run_dir / "model.pt").exists() and not (run_dir / "model.safetensors").exists():
                continue
            normalized = run_dir.name.lower().replace("_", "-")
            score = sum(1 for token in tokens if token and token in normalized)
            if score >= 2 and (best is None or score > best[1]):
                best = (run_dir, score)
        return best[0] if best else None

    def _download_model(self, model_id: str, token: str | None) -> Path:
        target = FLOWMATIC_CACHE / model_id.replace("/", "__")
        if (target / "model.pt").exists() or (target / "model.safetensors").exists():
            return target

        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            token=token,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "model.pt",
                "model.safetensors",
                "metadata.json",
                "config.json",
                "model_card.md",
                "README.md",
            ],
        )
        return target

    def prefetch(self, model_id: str, token: str | None = None, local_run: str | None = None) -> dict:
        if local_run:
            loaded = self.ensure_loaded("", token, local_run)
            return {
                "modelId": f"local:{local_run}",
                "localRun": local_run,
                "source": loaded.source,
                "path": str(loaded.run_dir),
                "cached": True,
            }
        local = self._local_checkpoint_for_model(model_id)
        if local:
            self.ensure_loaded(model_id, token)
            return {"modelId": model_id, "source": "local-checkpoint", "path": str(local), "cached": True}
        run_dir = self._download_model(model_id, token)
        self.ensure_loaded(model_id, token)
        return {"modelId": model_id, "source": "huggingface-cache", "path": str(run_dir), "cached": True}

    def ensure_loaded(self, model_id: str, token: str | None = None, local_run: str | None = None) -> LoadedModel:
        cache_key = f"local:{local_run}" if local_run else model_id
        if cache_key in self._loaded:
            return self._loaded[cache_key]

        if local_run:
            run_dir = CHECKPOINTS_DIR / local_run
            if not run_dir.exists():
                raise FileNotFoundError(f"Local checkpoint run not found: {local_run}")
            source = "local-checkpoint"
            resolved_model_id = cache_key
        else:
            local = self._local_checkpoint_for_model(model_id)
            if local:
                run_dir = local
                source = "local-checkpoint"
            else:
                run_dir = self._download_model(model_id, token)
                source = "huggingface-cache"
            resolved_model_id = model_id

        device = resolve_device(INFER_DEVICE)
        model, metadata = load_checkpoint_bundle(run_dir, device)
        loaded = LoadedModel(
            model_id=resolved_model_id,
            source=source,
            run_dir=run_dir,
            model=model,
            metadata=metadata,
            device=device,
            loaded_at=time.time(),
        )
        self._loaded[cache_key] = loaded
        return loaded

    def infer(
        self,
        payload: dict,
        token: str | None = None,
        model_id: str | None = None,
        local_run: str | None = None,
    ) -> dict:
        if not model_id and not local_run:
            raise ValueError("modelId or localRun is required")
        started = time.perf_counter()
        loaded = self.ensure_loaded(model_id or "", token, local_run)
        result = predict_payload(loaded.model, loaded.metadata, payload, loaded.device)
        return {
            "provider": "flowmatic-local",
            "modelId": model_id or f"local:{local_run}",
            "localRun": local_run,
            "source": loaded.source,
            "cachePath": str(loaded.run_dir),
            "device": str(loaded.device),
            "latencyMs": round((time.perf_counter() - started) * 1000),
            "kind": loaded.metadata.get("kind"),
            "result": result,
        }

    def list_cache(self) -> dict:
        cached = []
        if FLOWMATIC_CACHE.exists():
            for path in sorted(FLOWMATIC_CACHE.iterdir()):
                if path.is_dir():
                    cached.append(
                        {
                            "modelId": path.name.replace("__", "/"),
                            "path": str(path),
                            "hasWeights": (path / "model.pt").exists() or (path / "model.safetensors").exists(),
                        }
                    )
        loaded = [
            {
                "modelId": item.model_id,
                "source": item.source,
                "kind": item.metadata.get("kind"),
                "device": str(item.device),
            }
            for item in self._loaded.values()
        ]
        return {"cachedRepos": cached, "loadedInMemory": loaded, **self.device_info()}
