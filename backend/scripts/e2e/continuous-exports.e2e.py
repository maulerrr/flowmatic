"""Test continuous/manual exports to Postgres, MongoDB, and optional Hugging Face.

Requires export-test profile databases:
  docker compose --profile export-test up -d postgres-export-test mongo-export-test
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import requests

BASE = "http://localhost/api/v1"
OUT = Path(tempfile.gettempdir()) / "flowmatic_continuous_export_test_report.json"
RUN_ID = uuid.uuid4().hex[:8]

session = requests.Session()


def login() -> None:
    creds = {"email": "thesis.demo@flowmatic.local", "password": "ThesisDemo2025!"}
    r = session.post(f"{BASE}/auth/login", json=creds, timeout=30)
    if r.status_code == 401:
        session.post(
            f"{BASE}/auth/register",
            json={
                **creds,
                "displayName": "Thesis Demo",
                "organizationName": "Astana Smart City Lab",
            },
            timeout=30,
        )
        r = session.post(f"{BASE}/auth/login", json=creds, timeout=30)
    r.raise_for_status()


def ensure_pipeline() -> str:
    pipelines = session.get(f"{BASE}/smart-city/pipelines", timeout=30).json().get("data") or []
    pipeline = next((p for p in pipelines if p.get("name") == "Export Adapter Test"), None)
    if pipeline:
        return pipeline["id"]
    r = session.post(
        f"{BASE}/smart-city/pipelines",
        json={"name": "Export Adapter Test", "description": "Postgres/Mongo/HF export tests"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]["id"]


def ensure_ws_source(pipeline_id: str) -> str:
    sources = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/sources", timeout=30).json().get("data") or []
    source = next((s for s in sources if s.get("type") == "WEBSOCKET"), None)
    if source:
        return source["id"]
    r = session.post(
        f"{BASE}/smart-city/pipelines/{pipeline_id}/sources",
        json={
            "name": "Export Test WS Source",
            "type": "WEBSOCKET",
            "mode": "SIMULATED",
            "sensorKind": "traffic",
            "pollIntervalMs": 2000,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]["id"]


def start_pipeline(pipeline_id: str) -> None:
    session.patch(
        f"{BASE}/smart-city/pipelines/{pipeline_id}/runtime",
        json={"lakeWriteMode": "append", "sourcePollIntervalMs": 2000, "exportCadenceSeconds": 20},
        timeout=30,
    ).raise_for_status()
    session.post(f"{BASE}/smart-city/pipelines/{pipeline_id}/start", json={}, timeout=60).raise_for_status()


def create_target(pipeline_id: str, name: str, adapter_type: str, settings: dict) -> str:
    r = session.post(
        f"{BASE}/smart-city/pipelines/{pipeline_id}/export-targets",
        json={
            "name": name,
            "stage": "cleaned",
            "adapterType": adapter_type,
            "isContinuous": True,
            "cadenceSeconds": 20,
            "saveCredentials": True,
            "settings": settings,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]["id"]


def run_target(target_id: str) -> dict:
    session.post(f"{BASE}/smart-city/export-targets/{target_id}/run", timeout=120).raise_for_status()
    time.sleep(3)
    return {}


def latest_run(pipeline_id: str, adapter_type: str) -> dict | None:
    runs = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/export-runs", timeout=30).json().get("data") or []
    for run in runs:
        if run.get("adapterType") == adapter_type:
            return run
    return None


def pg_count(table: str) -> int:
    if not table_exists_pg(table):
        return 0
    cmd = [
        "docker",
        "exec",
        "flowmatic-postgres-export-test",
        "psql",
        "-U",
        "export_test",
        "-d",
        "export_test",
        "-tAc",
        f'SELECT COUNT(*) FROM "{table}"',
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out or 0)


def table_exists_pg(table: str) -> bool:
    cmd = [
        "docker",
        "exec",
        "flowmatic-postgres-export-test",
        "psql",
        "-U",
        "export_test",
        "-d",
        "export_test",
        "-tAc",
        f"SELECT to_regclass('public.\"{table}\"') IS NOT NULL",
    ]
    out = subprocess.check_output(cmd, text=True).strip().lower()
    return out in {"t", "true", "1"}


def mongo_count(collection: str) -> int:
    script = f"db.getSiblingDB('export_test').getCollection('{collection}').countDocuments({{}})"
    cmd = ["docker", "exec", "flowmatic-mongo-export-test", "mongosh", "--quiet", "--eval", script]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out or 0)


def wait_for_events(pipeline_id: str, minimum: int = 3, timeout_s: int = 45) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        events = session.get(
            f"{BASE}/smart-city/pipelines/{pipeline_id}/events",
            params={"limit": 25},
            timeout=30,
        ).json().get("data") or []
        if len(events) >= minimum:
            return len(events)
        time.sleep(2)
    return len(events)


def main() -> None:
    login()
    pipeline_id = ensure_pipeline()
    ensure_ws_source(pipeline_id)

    pg_table = f"export_test_{RUN_ID}"
    mongo_collection = f"export_test_{RUN_ID}"
    hf_repo = f"flowmatic-export-test-{RUN_ID}"

    pg_target = create_target(
        pipeline_id,
        f"Postgres {RUN_ID}",
        "postgres",
        {
            "host": "postgres-export-test",
            "port": 5432,
            "username": "export_test",
            "password": "export_test",
            "database": "export_test",
            "table": pg_table,
            "ifExists": "append",
        },
    )
    mongo_target = create_target(
        pipeline_id,
        f"Mongo {RUN_ID}",
        "mongodb",
        {
            "uri": "mongodb://mongo-export-test:27017",
            "database": "export_test",
            "collection": mongo_collection,
            "ifExists": "append",
        },
    )

    hf_target = None
    if os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN"):
        hf_target = create_target(
            pipeline_id,
            f"HF {RUN_ID}",
            "huggingface",
            {
                "repoName": hf_repo,
                "fileName": f"{hf_repo}.csv",
                "ifExists": "append",
            },
        )

    report: dict = {
        "runId": RUN_ID,
        "pipelineId": pipeline_id,
        "cases": {},
    }

    # Case 1: empty dataset / no table
    run_target(pg_target)
    run_target(mongo_target)
    report["cases"]["empty_postgres_rows"] = pg_count(pg_table)
    report["cases"]["empty_mongo_rows"] = mongo_count(mongo_collection)
    pg_empty_run = latest_run(pipeline_id, "postgres")
    report["cases"]["empty_postgres_run_status"] = pg_empty_run.get("status") if pg_empty_run else None

    # Case 2: generate events and export
    start_pipeline(pipeline_id)
    event_count = wait_for_events(pipeline_id, minimum=5)
    report["cases"]["events_generated"] = event_count

    run_target(pg_target)
    run_target(mongo_target)
    if hf_target:
        run_target(hf_target)

    pg_after_first = pg_count(pg_table)
    mongo_after_first = mongo_count(mongo_collection)
    report["cases"]["postgres_after_first_export"] = pg_after_first
    report["cases"]["mongo_after_first_export"] = mongo_after_first

    # Case 3: append second batch
    time.sleep(8)
    run_target(pg_target)
    run_target(mongo_target)
    pg_after_second = pg_count(pg_table)
    mongo_after_second = mongo_count(mongo_collection)
    report["cases"]["postgres_after_second_export"] = pg_after_second
    report["cases"]["mongo_after_second_export"] = mongo_after_second

    preview = session.get(
        f"{BASE}/smart-city/pipelines/{pipeline_id}/export-preview",
        params={"stage": "cleaned", "limit": 5},
        timeout=30,
    ).json().get("data")
    report["exportPreview"] = preview

    runs = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/export-runs", timeout=30).json().get("data") or []
    report["recentRuns"] = runs[:8]

    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {OUT}")

    if event_count < 3:
        raise SystemExit("Not enough sensor events generated")
    if pg_after_first <= 0:
        raise SystemExit("Postgres export did not insert rows")
    if mongo_after_first <= 0:
        raise SystemExit("MongoDB export did not insert rows")
    if pg_after_second < pg_after_first:
        raise SystemExit("Postgres append failed (row count did not grow)")
    if mongo_after_second < mongo_after_first:
        raise SystemExit("Mongo append failed (row count did not grow)")
    if hf_target:
        hf_run = latest_run(pipeline_id, "huggingface")
        if not hf_run or hf_run.get("status") != "SUCCEEDED":
            raise SystemExit(f"Hugging Face export failed: {hf_run}")
        report["cases"]["hf_destination"] = hf_run.get("destination")

    print("Continuous export adapter tests passed")


if __name__ == "__main__":
    main()
