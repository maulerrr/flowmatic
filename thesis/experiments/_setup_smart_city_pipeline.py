"""Create Astana geospatial smart-city pipeline for thesis defence screenshots."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "http://localhost/api/v1"
OUT = Path(__file__).resolve().parent / "smart_city_pipeline_setup.json"

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
    print("login", r.status_code)


def main() -> None:
    login()

    pipelines = session.get(f"{BASE}/smart-city/pipelines", timeout=30).json()
    existing = pipelines.get("data") or []
    pipeline = next((p for p in existing if p.get("name") == "Astana Geospatial Demo"), None)

    if not pipeline:
        r = session.post(
            f"{BASE}/smart-city/pipelines",
            json={
                "name": "Astana Geospatial Demo",
                "description": "Thesis defence demo: Astana geospatial traffic simulator (latitude/longitude)",
            },
            timeout=30,
        )
        r.raise_for_status()
        pipeline = r.json()["data"]
        print("created pipeline", pipeline["id"])
    else:
        print("reusing pipeline", pipeline["id"])

    pipeline_id = pipeline["id"]
    sources = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/sources", timeout=30).json()
    source_list = sources.get("data") or []

    desired = [
        ("traffic", "Astana Geospatial Traffic", "WEBSOCKET"),
        ("weather", "Astana Weather Stations", "HTTP_POLLING"),
    ]

    existing_kinds = {s.get("sensorKind"): s for s in source_list}
    for kind, name, transport in desired:
        if kind in existing_kinds:
            source_list.append(existing_kinds[kind])
            continue
        r = session.post(
            f"{BASE}/smart-city/pipelines/{pipeline_id}/sources",
            json={
                "name": name,
                "type": transport,
                "mode": "SIMULATED",
                "sensorKind": kind,
                "pollIntervalMs": 2000 if kind == "traffic" else 3000,
            },
            timeout=30,
        )
        r.raise_for_status()
        src = r.json()["data"]
        print("created source", kind, src["id"])
        source_list.append(src)

    for src in source_list:
        if src.get("status") != "RUNNING":
            r = session.post(f"{BASE}/smart-city/sources/{src['id']}/start", timeout=30)
            if r.ok:
                print("started source", src["id"], src.get("name"), src.get("sensorKind"))
            else:
                print("start failed", src["id"], r.status_code, r.text[:200])

    for i in range(8):
        events = session.get(
            f"{BASE}/smart-city/pipelines/{pipeline_id}/events",
            params={"limit": 20},
            timeout=30,
        ).json()
        data = events.get("data") or []
        print(f"poll {i+1}: {len(data)} recent events")
        if data:
            sample = data[0].get("payload") or data[0]
            lat = sample.get("latitude") or sample.get("lat")
            lng = sample.get("longitude") or sample.get("lng")
            print(" sample coords", lat, lng)
            break
        time.sleep(2)

    obs = session.get(f"{BASE}/smart-city/pipelines/{pipeline_id}/observability", timeout=30).json()

    result = {
        "pipelineId": pipeline_id,
        "pipelineName": pipeline.get("name"),
        "sources": [
            {
                "id": s["id"],
                "name": s.get("name"),
                "status": s.get("status"),
                "sensorKind": s.get("sensorKind"),
                "type": s.get("type"),
            }
            for s in source_list
        ],
        "recentEventCount": len(
            (
                session.get(
                    f"{BASE}/smart-city/pipelines/{pipeline_id}/events",
                    params={"limit": 20},
                    timeout=30,
                ).json().get("data")
                or []
            )
        ),
        "observability": obs.get("data"),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
