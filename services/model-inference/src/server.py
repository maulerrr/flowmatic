from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import ModelEngine

app = FastAPI(title="Flowmatic Local Model Inference", version="2.0.0")
engine = ModelEngine()


class InferRequest(BaseModel):
    modelId: str | None = None
    localRun: str | None = None
    token: str | None = None
    inputs: dict = Field(default_factory=dict)


class PrefetchRequest(BaseModel):
    modelId: str | None = None
    localRun: str | None = None
    token: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "model-inference", "mode": "local-gpu", **engine.device_info()}


@app.get("/v1/cache")
def cache_status() -> dict:
    return engine.list_cache()


@app.post("/v1/prefetch")
def prefetch(body: PrefetchRequest) -> dict:
    try:
        if not body.modelId and not body.localRun:
            raise HTTPException(status_code=400, detail="modelId or localRun is required")
        return engine.prefetch(body.modelId.strip() if body.modelId else "", body.token, body.localRun)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/v1/infer")
def infer(body: InferRequest) -> dict:
    try:
        if not body.modelId and not body.localRun:
            raise HTTPException(status_code=400, detail="modelId or localRun is required")
        return engine.infer(
            body.inputs,
            body.token,
            body.modelId.strip() if body.modelId else None,
            body.localRun,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8093"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
