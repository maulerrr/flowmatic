import os
import uuid
import json
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

import pandas as pd

from .ingestion import ingest
from .quality_check import quality_report
from .cleaning import clean
from .store import create_run, finalize_run


@dataclass
class Step:
    name: str
    func: Callable[[Dict[str, Any]], Dict[str, Any]]


class Pipeline:
    def __init__(self, name: str, steps: List[Step]):
        self.name = name
        self.steps = steps

    def run(self, params: Dict[str, Any], artifacts_dir: str, dataset_version_id: int) -> Dict[str, Any]:
        os.makedirs(artifacts_dir, exist_ok=True)
        run_id = str(uuid.uuid4())
        create_run(run_id, dataset_version_id, self.name, params, artifacts_dir)
        ctx: Dict[str, Any] = {"params": params, "artifacts_dir": artifacts_dir}
        try:
            for step in self.steps:
                ctx = step.func(ctx)
            metrics = ctx.get("metrics", {})
            finalize_run(run_id, "succeeded", metrics=metrics)
            return {"run_id": run_id, "status": "succeeded", "metrics": metrics}
        except Exception as e:
            finalize_run(run_id, "failed", error=str(e))
            raise


def step_ingest(ctx: Dict[str, Any]) -> Dict[str, Any]:
    p = ctx["params"]
    if p.get("source_type") == "hf":
        df = ingest(p["hf_dataset"], split=p.get("hf_split", "train"), token=p.get("hf_token"))
    else:
        path = p["path"]
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path, parse_dates=True, index_col=0)
        elif ext == ".json":
            df = pd.read_json(path)
            dt_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
            if not dt_col:
                raise ValueError("No datetime-like column found in JSON")
            df[dt_col] = pd.to_datetime(df[dt_col], errors="raise")
            df = df.set_index(dt_col)
        else:
            raise ValueError(f"Unsupported extension: {ext}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="raise")
    ctx["df"] = df
    return ctx


def step_quality(ctx: Dict[str, Any]) -> Dict[str, Any]:
    df = ctx["df"]
    qr = quality_report(df)
    metrics = {
        "duplicates": int(qr["duplicates"]),
        "outliers": int(len(qr["outliers"])),
        "missing_total": int(sum(qr["missing"].values)),
    }
    out_path = os.path.join(ctx["artifacts_dir"], "quality_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f)
    ctx["metrics"] = metrics
    ctx["quality_report"] = qr
    return ctx


def step_clean(ctx: Dict[str, Any]) -> Dict[str, Any]:
    df = ctx["df"]
    df_clean = clean(df)
    out_csv = os.path.join(ctx["artifacts_dir"], "cleaned.csv")
    df_clean.to_csv(out_csv)
    ctx["df_clean"] = df_clean
    ctx["cleaned_path"] = out_csv
    return ctx


def default_pipeline() -> Pipeline:
    return Pipeline(
        name="default",
        steps=[
            Step("ingest", step_ingest),
            Step("quality", step_quality),
            Step("clean", step_clean),
        ],
    )
