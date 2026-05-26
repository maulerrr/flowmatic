"""End-to-end demo: simulated WS ingest -> core processing -> MinIO medallion lake + S3 export."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "http://localhost/api/v1"
OUT = Path(__file__).resolve().parent / "minio_pipeline_demo_report.json"

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
    pipeline = next((p for p in pipelines if p.get("name") == "Astana Live Demo"), None)
    if not pipeline:
        r = session.post(
            f"{BASE}/smart-city/pipelines",
            json={
                "name": "Astana Live Demo",
                "description": "MinIO E2E: WS simulated source -> lake -> export",
            },
            timeout=30,
        )
        r.raise_for_status()
        pipeline = r.json()["data"]
    return pipeline["id"]


def ensure_minio_lake(pipeline_id: str) -> str:
    lakes = session.get(f"{BASE}/smart-city/data-lakes", timeout=30).json().get("data") or []
    lake = next((item for item in lakes if item.get("name") == "Thesis MinIO Lake"), None)
    if not lake:
        r = session.post(
            f"{BASE}/smart-city/data-lakes",
            json={
                "name": "Thesis MinIO Lake",
                "provider": "MINIO",
                "bucket": "flowmatic-media",
                "region": "us-east-1",
                "endpoint": "http://minio:9000",
                "basePrefix": "smart-city-thesis",
                "accessKey": "minio",
                "secretKey": "minio123",
                "isDefault": True,
            },
            timeout=30,
        )
        r.raise_for_status()
        lake = r.json()["data"]
    session.patch(
        f"{BASE}/smart-city/pipelines/{pipeline_id}",
        json={"dataLakeConnectionId": lake["id"]},
        timeout=30,
    ).raise_for_status()
    test = session.post(f"{BASE}/smart-city/data-lakes/{lake['id']}/test", timeout=60)
    test.raise_for_status()
    return lake["id"]


def ensure_ws_source(pipeline_id: str) -> str:
    sources = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/sources", timeout=30).json().get("data") or []
    source = next((s for s in sources if s.get("type") == "WEBSOCKET"), None)
    if not source:
        r = session.post(
            f"{BASE}/smart-city/pipelines/{pipeline_id}/sources",
            json={
                "name": "IoT WebSocket Simulator",
                "type": "WEBSOCKET",
                "mode": "SIMULATED",
                "sensorKind": "iot",
                "pollIntervalMs": 2000,
            },
            timeout=30,
        )
        r.raise_for_status()
        source = r.json()["data"]
    return source["id"]


def ensure_json_export_target(pipeline_id: str) -> str:
    targets = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/export-targets", timeout=30).json().get("data") or []
    target = next((t for t in targets if t.get("name") == "MinIO cleaned NDJSON"), None)
    if target:
        return target["id"]
    r = session.post(
        f"{BASE}/smart-city/pipelines/{pipeline_id}/export-targets",
        json={
            "name": "MinIO cleaned NDJSON",
            "stage": "cleaned",
            "adapterType": "json",
            "isContinuous": True,
            "cadenceSeconds": 30,
            "settings": {"ifExists": "append"},
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]["id"]


def main() -> None:
    login()
    pipeline_id = ensure_pipeline()
    lake_id = ensure_minio_lake(pipeline_id)
    source_id = ensure_ws_source(pipeline_id)
    target_id = ensure_json_export_target(pipeline_id)

    session.patch(
        f"{BASE}/smart-city/pipelines/{pipeline_id}/runtime",
        json={"lakeWriteMode": "append", "sourcePollIntervalMs": 2000, "exportCadenceSeconds": 30},
        timeout=30,
    ).raise_for_status()

    start = session.post(
        f"{BASE}/smart-city/pipelines/{pipeline_id}/start",
        json={"sourcePollIntervalMs": 2000},
        timeout=60,
    )
    start.raise_for_status()

    report: dict = {
        "pipelineId": pipeline_id,
        "lakeId": lake_id,
        "sourceId": source_id,
        "exportTargetId": target_id,
        "eventsBefore": 0,
        "eventsAfter": 0,
        "lakeObjects": [],
        "exportRuns": [],
    }

    events = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/events", params={"limit": 5}, timeout=30).json()
    report["eventsBefore"] = len(events.get("data") or [])

    print("Waiting 35s for WS ingest + lake writes + export cadence…")
    time.sleep(35)

    events_after = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/events", params={"limit": 25}, timeout=30).json()
    report["eventsAfter"] = len(events_after.get("data") or [])

    lake_objects = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/data-lake-objects", timeout=30).json()
    report["lakeObjects"] = lake_objects.get("data", {}).get("objects", [])

    export_runs = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/export-runs", timeout=30).json()
    report["exportRuns"] = (export_runs.get("data") or [])[:5]

    session.post(f"{BASE}/smart-city/export-targets/{target_id}/run", timeout=120).raise_for_status()
    export_runs = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/export-runs", timeout=30).json()
    report["exportRuns"] = (export_runs.get("data") or [])[:5]

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {OUT}")

    raw_count = sum(len(g.get("objects") or []) for g in report["lakeObjects"] if g.get("stage") == "raw")
    cleaned_count = sum(len(g.get("objects") or []) for g in report["lakeObjects"] if g.get("stage") == "cleaned")
    if report["eventsAfter"] <= report["eventsBefore"]:
        raise SystemExit("No new sensor events observed")
    if raw_count == 0 and cleaned_count == 0:
        raise SystemExit("No lake objects listed (check MinIO connection and pipeline ACTIVE status)")
    print("E2E demo passed")


if __name__ == "__main__":
    main()
