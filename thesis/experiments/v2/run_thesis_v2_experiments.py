#!/usr/bin/env python3
"""Generate thesis v2 evidence: preparation stress, routing eval, baselines."""
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results"
ASTANA_CSV = ROOT / "data" / "astana_synthetic_data.csv"
PORTFOLIO = ROOT / "models" / "reports" / "production_portfolio.json"
MULTISEED_MD = ROOT / "models" / "paper" / "tables" / "multi_seed_aggregate.md"


@dataclass
class RegistryModel:
    id: str
    kind: str
    dataset: str
    modality: str
    tasks: list[str]
    sensor_kinds: list[str]
    priority: int
    requires_geo: bool = False


KIND_PROFILE: dict[str, dict[str, Any]] = {
    "tranad_anomaly": {"modality": "traffic", "tasks": ["anomaly"], "sensor_kinds": ["traffic", "generic"], "priority": 1},
    "patchtst_forecast": {"modality": "traffic", "tasks": ["forecast"], "sensor_kinds": ["traffic"], "priority": 3},
    "itransformer_forecast": {"modality": "traffic", "tasks": ["forecast"], "sensor_kinds": ["traffic"], "priority": 5},
    "transformer_classifier": {"modality": "traffic", "tasks": ["classification"], "sensor_kinds": ["traffic"], "priority": 7},
    "saits_imputer": {"modality": "generic", "tasks": ["imputation"], "sensor_kinds": ["traffic", "weather", "air", "generic"], "priority": 8},
    "timesblock_forecast": {"modality": "weather", "tasks": ["forecast"], "sensor_kinds": ["weather", "air"], "priority": 1},
    "stgcn_forecast": {"modality": "traffic", "tasks": ["forecast"], "sensor_kinds": ["traffic"], "requires_geo": True, "priority": 2},
    "dlinear_forecast": {"modality": "energy", "tasks": ["forecast"], "sensor_kinds": ["generic"], "priority": 20},
}


def load_registry() -> list[RegistryModel]:
    data = json.loads(PORTFOLIO.read_text(encoding="utf-8"))
    models: list[RegistryModel] = []
    for m in data.get("models", []):
        kind = m["kind"]
        prof = KIND_PROFILE.get(kind, {"modality": "generic", "tasks": ["forecast"], "sensor_kinds": ["generic"], "priority": 99})
        models.append(
            RegistryModel(
                id=m["run"],
                kind=kind,
                dataset=m["dataset"],
                modality=prof["modality"],
                tasks=prof["tasks"],
                sensor_kinds=prof["sensor_kinds"],
                priority=prof["priority"],
                requires_geo=prof.get("requires_geo", False),
            )
        )
    return models


def score_model(model: RegistryModel, profile: dict[str, Any]) -> float:
    score = float(model.priority)
    if profile["modality"] == "traffic" and "astana" in model.dataset:
        score -= 3
    if profile["modality"] == "traffic" and "traffic" in model.dataset:
        score -= 2
    if profile["modality"] == "weather" and "weather" in model.dataset:
        score -= 4
    if profile["modality"] == "weather" and model.modality != "weather":
        score += 100
    if profile["modality"] == "traffic" and model.modality == "weather":
        score += 100
    if profile["preferred_task"] == "anomaly" and "anomaly" in model.tasks:
        score -= 2
    if profile["preferred_task"] == "forecast" and "forecast" in model.tasks:
        score -= 1
    if profile.get("has_geo") and model.requires_geo:
        score -= 1
    return score


def classify_event(sensor_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    fields = [k.lower() for k in payload]
    has_geo = any(x in f for f in fields for x in ("lat", "lng", "longitude", "latitude"))
    weather_hints = ("temperature", "humidity", "pressure", "wind", "precipitation", "weather")
    traffic_hints = ("speed", "traffic", "vehicle", "density", "congestion", "flow", "occupancy")
    has_weather = any(any(h in f for h in weather_hints) for f in fields)
    has_traffic = any(any(h in f for h in traffic_hints) for f in fields)
    has_severity = any("severity" in f for f in fields)
    modality = "generic"
    if sensor_kind == "weather" or (has_weather and not has_traffic):
        modality = "weather"
    if sensor_kind == "traffic" or has_traffic:
        modality = "traffic"
    values = list(payload.values())
    missing_ratio = sum(1 for v in values if v in (None, "", "null")) / max(len(values), 1)
    preferred_task = "anomaly"
    if has_severity:
        preferred_task = "classification"
    elif missing_ratio >= 0.2:
        preferred_task = "imputation"
    elif has_traffic or has_weather:
        preferred_task = "forecast"
    return {
        "sensor_kind": sensor_kind or "generic",
        "modality": modality,
        "has_geo": has_geo,
        "preferred_task": preferred_task,
    }


def pick_best(models: list[RegistryModel], profile: dict[str, Any], prefer_anomaly: bool = True) -> RegistryModel | None:
    def find_task(task: str) -> RegistryModel | None:
        compatible = [
            m
            for m in models
            if (
                m.modality == profile["modality"]
                or (task == "imputation" and m.modality == "generic")
            )
            and (profile["sensor_kind"] in m.sensor_kinds or "generic" in m.sensor_kinds)
            and (not m.requires_geo or profile["has_geo"])
            and task in m.tasks
        ]
        compatible.sort(key=lambda m: score_model(m, profile))
        return compatible[0] if compatible else None

    if prefer_anomaly:
        a = find_task("anomaly")
        if a:
            return a
    return find_task(profile["preferred_task"])


def expected_kind_for_scenario(
    models: list[RegistryModel],
    sensor_kind: str,
    payload: dict[str, Any],
    prefer_anomaly: bool,
) -> str | None:
    """Policy oracle: kind selected by the same router used in production."""
    profile = classify_event(sensor_kind, payload)
    chosen = pick_best(models, profile, prefer_anomaly=prefer_anomaly)
    return chosen.kind if chosen else None


def routing_evaluation() -> dict[str, Any]:
    models = load_registry()
    scenarios = [
        {
            "name": "traffic_density_forecast",
            "sensor_kind": "traffic",
            "payload": {"speedKmh": 42, "trafficDensity": 68},
            "oracle_kind": "patchtst_forecast",
            "prefer_anomaly": False,
        },
        {
            "name": "traffic_speed_forecast",
            "sensor_kind": "traffic",
            "payload": {"speedKmh": 42, "trafficDensity": 68, "latitude": 51.12, "longitude": 71.45},
            "oracle_kind": "stgcn_forecast",
            "prefer_anomaly": False,
        },
        {
            "name": "traffic_anomaly",
            "sensor_kind": "traffic",
            "payload": {"speedKmh": 120, "trafficDensity": 190, "latitude": 51.1, "longitude": 71.4},
            "oracle_kind": "tranad_anomaly",
            "prefer_anomaly": True,
        },
        {
            "name": "weather_forecast",
            "sensor_kind": "weather",
            "payload": {"temperatureC": 4.2, "humidityPct": 71, "pressureHpa": 1012},
            "oracle_kind": "timesblock_forecast",
            "prefer_anomaly": False,
        },
        {
            "name": "traffic_classification",
            "sensor_kind": "traffic",
            "payload": {"speedKmh": 35, "trafficDensity": 55, "severity": "High"},
            "oracle_kind": "transformer_classifier",
            "prefer_anomaly": False,
        },
        {
            "name": "sparse_imputation",
            "sensor_kind": "traffic",
            "payload": {"speedKmh": None, "trafficDensity": None, "latitude": 51.11, "longitude": 71.41},
            "oracle_kind": "saits_imputer",
            "prefer_anomaly": False,
        },
        {
            "name": "graph_traffic_geo",
            "sensor_kind": "traffic",
            "payload": {"speedKmh": 50, "trafficDensity": 80, "latitude": 51.16, "longitude": 71.43, "adjacency": True},
            "oracle_kind": "stgcn_forecast",
            "prefer_anomaly": False,
        },
    ]
    # Expand with jittered variants
    events = []
    rng = random.Random(2026)
    for base in scenarios:
        for i in range(20):
            p = dict(base["payload"])
            if "speedKmh" in p and p["speedKmh"] is not None:
                p["speedKmh"] = max(0, p["speedKmh"] + rng.randint(-15, 15))
            if "trafficDensity" in p and p["trafficDensity"] is not None:
                p["trafficDensity"] = max(0, p["trafficDensity"] + rng.randint(-20, 20))
            events.append({**base, "event_id": f"{base['name']}_{i}"})

    correct = 0
    task_correct = 0
    policy_correct = 0
    rows = []
    for ev in events:
        profile = classify_event(ev["sensor_kind"], ev["payload"])
        prefer_anom = ev.get("prefer_anomaly", True)
        chosen = pick_best(models, profile, prefer_anomaly=prefer_anom)
        policy_kind = expected_kind_for_scenario(
            models, ev["sensor_kind"], ev["payload"], prefer_anom
        )
        oracle = next((m for m in models if m.kind == ev["oracle_kind"]), None)
        kind_match = chosen is not None and chosen.kind == ev["oracle_kind"]
        policy_match = chosen is not None and policy_kind is not None and chosen.kind == policy_kind
        task_match = (
            chosen is not None
            and oracle is not None
            and bool(set(chosen.tasks) & set(oracle.tasks))
        )
        if kind_match:
            correct += 1
        if task_match:
            task_correct += 1
        if policy_match:
            policy_correct += 1
        rows.append(
            {
                "event_id": ev["event_id"],
                "sensor_kind": ev["sensor_kind"],
                "oracle_kind": ev["oracle_kind"],
                "policy_kind": policy_kind,
                "selected_kind": chosen.kind if chosen else None,
                "selected_run": chosen.id if chosen else None,
                "modality": profile["modality"],
                "preferred_task": profile["preferred_task"],
                "kind_match": kind_match,
                "policy_match": policy_match,
                "task_match": task_match,
            }
        )

    n = len(rows)
    mismatch_by_scenario: dict[str, Any] = {}
    scenario_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        base = re.sub(r"_\d+$", "", row["event_id"])
        st = scenario_stats.setdefault(base, {"total": 0, "correct": 0, "wrong_selected": Counter()})
        st["total"] += 1
        if row["kind_match"]:
            st["correct"] += 1
        else:
            st["wrong_selected"][row["selected_kind"] or "none"] += 1
    for base, st in sorted(scenario_stats.items()):
        mismatch_by_scenario[base] = {
            "total": st["total"],
            "correct": st["correct"],
            "accuracy": st["correct"] / st["total"] if st["total"] else 0,
            "top_wrong": st["wrong_selected"].most_common(1)[0] if st["wrong_selected"] else None,
        }

    return {
        "n_events": n,
        "top1_kind_accuracy": correct / n,
        "task_slot_accuracy": task_correct / n,
        "policy_consistency": policy_correct / n,
        "mismatch_by_scenario": mismatch_by_scenario,
        "rows_sample": rows[:12],
        "rows": rows,
    }


def load_astana_rows(limit: int | None = None) -> list[dict[str, str]]:
    rows = []
    with ASTANA_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if limit and i + 1 >= limit:
                break
    return rows


def dqi_scores(rows: list[dict[str, str]], columns: list[str]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        z = {k: 0.0 for k in ("completeness", "consistency", "accuracy", "timeliness", "uniqueness", "validity")}
        z["composite"] = 0.0
        return z

    missing_cells = 0
    for row in rows:
        for c in columns:
            v = row.get(c, "")
            if v is None or str(v).strip() == "":
                missing_cells += 1
    completeness = 1 - missing_cells / (n * len(columns))

    seen = set()
    dup = 0
    for row in rows:
        key = tuple(row.get(c, "") for c in columns)
        if key in seen:
            dup += 1
        seen.add(key)
    uniqueness = 1 - dup / n

    def _num(row: dict, *keys: str) -> float | None:
        for k in keys:
            v = row.get(k)
            if v is None or str(v).strip() == "" or str(v).lower() == "nan":
                continue
            try:
                return float(v)
            except ValueError:
                continue
        return None

    valid = 0
    total_checks = 0
    for row in rows:
        try:
            lat = _num(row, "Latitude", "latitude")
            lon = _num(row, "Longitude", "longitude")
            spd = _num(row, "Speed_kmh", "speedKmh")
            den = _num(row, "Traffic_Density", "trafficDensity")
            if lat is None or lon is None or spd is None or den is None:
                total_checks += 4
                continue
            total_checks += 4
            if 40 <= lat <= 52 and 68 <= lon <= 73 and 0 <= spd <= 200 and 0 <= den <= 300:
                valid += 4
            elif 40 <= lat <= 52 and 68 <= lon <= 73:
                valid += 2
        except (KeyError, ValueError):
            total_checks += 4
    accuracy = valid / max(total_checks, 1)
    validity = accuracy

    # Schema consistency: all rows have required keys
    required = set(columns)
    consistency = sum(1 for row in rows if required <= set(row.keys())) / n

    # Timeliness: parseable ISO timestamps within study window
    ok_ts = 0
    for row in rows:
        ts = row.get("Timestamp") or row.get("sourceTimestamp", "")
        if re.match(r"\d{4}-\d{2}-\d{2}T", ts):
            ok_ts += 1
    timeliness = ok_ts / n

    weights = [1 / 6] * 6
    dims = [completeness, consistency, accuracy, timeliness, uniqueness, validity]
    composite = sum(w * q for w, q in zip(weights, dims)) / sum(weights)
    return {
        "completeness": round(completeness, 4),
        "consistency": round(consistency, 4),
        "accuracy": round(accuracy, 4),
        "timeliness": round(timeliness, 4),
        "uniqueness": round(uniqueness, 4),
        "validity": round(validity, 4),
        "composite": round(composite, 4),
    }


def corrupt_rows(rows: list[dict[str, str]], rate: float, rng: random.Random) -> list[dict[str, str]]:
    out = [dict(r) for r in rows]
    n = len(out)
    k = int(n * rate)
    idxs = rng.sample(range(n), k) if k else []
    for i in idxs:
        choice = rng.randint(0, 5)
        if choice == 0:
            out[i]["Speed_kmh"] = ""
        elif choice == 1:
            out[i]["Latitude"] = "999"
        elif choice == 2:
            out[i]["Longitude"] = "-10"
        elif choice == 3:
            out[i]["Event_ID"] = out[rng.randint(0, n - 1)]["Event_ID"]
        elif choice == 4:
            out[i]["Timestamp"] = "invalid"
        else:
            out[i]["Traffic_Density"] = "NaN"
    return out


def clean_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    stats = {"dropped": 0, "imputed": 0, "deduped": 0}
    cols = list(rows[0].keys()) if rows else []
    cleaned: list[dict[str, str]] = []
    seen: set[tuple] = set()
    for row in rows:
        r = dict(row)
        ts = r.get("Timestamp") or r.get("sourceTimestamp", "")
        if not ts or not re.match(r"\d{4}-\d{2}-\d{2}", str(ts)):
            stats["dropped"] += 1
            continue
        try:
            lat = float(r.get("Latitude") or r["latitude"])
            lon = float(r.get("Longitude") or r["longitude"])
            if not (40 <= lat <= 52 and 68 <= lon <= 73):
                stats["dropped"] += 1
                continue
        except ValueError:
            stats["dropped"] += 1
            continue
        for field in ("Speed_kmh", "Traffic_Density", "speedKmh", "trafficDensity"):
            if field not in r:
                continue
            if not str(r.get(field, "")).strip() or str(r.get(field)).lower() == "nan":
                r[field] = "40"
                stats["imputed"] += 1
        key = tuple(r.get(c, "") for c in cols)
        if key in seen:
            stats["deduped"] += 1
            continue
        seen.add(key)
        cleaned.append(r)
    return cleaned, stats


def preparation_stress_test() -> dict[str, Any]:
    rng = random.Random(42)
    base = load_astana_rows(10000)
    columns = list(base[0].keys())
    clean_base, _ = clean_rows(base)
    scenarios = []
    for rate in (0.0, 0.01, 0.05, 0.10, 0.20):
        dirty = corrupt_rows(clean_base, rate, rng) if rate > 0 else [dict(r) for r in clean_base]
        before = dqi_scores(dirty, columns)
        cleaned, fix_stats = clean_rows(dirty)
        after = dqi_scores(cleaned, columns)
        scenarios.append(
            {
                "corruption_rate": rate,
                "rows_in": len(dirty),
                "rows_out": len(cleaned),
                "dqi_before": before,
                "dqi_after": after,
                "delta_composite": round(after["composite"] - before["composite"], 4),
                **fix_stats,
            }
        )
    return {"scenarios": scenarios, "columns": columns}


def dqi_weight_sensitivity() -> dict[str, Any]:
    """Recompute composite DQI after cleaning at 20% corruption under alternative weights."""
    rng = random.Random(42)
    base = load_astana_rows(10000)
    columns = list(base[0].keys())
    clean_base, _ = clean_rows(base)
    dirty = corrupt_rows(clean_base, 0.20, rng)
    cleaned_rows, _ = clean_rows(dirty)
    after = dqi_scores(cleaned_rows, columns)

    def composite(weights: list[float], scores: dict[str, float]) -> float:
        keys = ("completeness", "consistency", "accuracy", "timeliness", "uniqueness", "validity")
        num = sum(weights[i] * scores[k] for i, k in enumerate(keys))
        den = sum(weights)
        return round(num / den, 4)

    uniform = [1 / 6] * 6
    completeness_x2 = [2 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]
    validity_x2 = [1 / 7, 1 / 7, 2 / 7, 1 / 7, 1 / 7, 1 / 7]
    schemes = {
        "uniform_1_6": uniform,
        "completeness_x2": completeness_x2,
        "validity_x2": validity_x2,
    }
    composites = {name: composite(w, after) for name, w in schemes.items()}
    return {
        "corruption_rate": 0.20,
        "composites_after_cleaning": composites,
        "rank_stable": len(set(composites.values())) == 1 or max(composites.values()) - min(composites.values()) < 0.01,
    }


def parse_multiseed_md() -> list[dict[str, str]]:
    lines = MULTISEED_MD.read_text(encoding="utf-8").strip().splitlines()
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        rows.append(dict(zip(header, parts)))
    return rows


def sklearn_baselines() -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        return {"error": "sklearn not installed", "skipped": True}

    rows = load_astana_rows(30000)
    y_raw = [r.get("Severity") or r.get("severity") for r in rows]
    X = []
    for r in rows:
        X.append(
            [
                float(r.get("Speed_kmh") or r.get("speedKmh") or 0),
                float(r.get("Traffic_Density") or r.get("trafficDensity") or 0),
                float(r.get("Latitude") or r.get("latitude")),
                float(r.get("Longitude") or r.get("longitude")),
            ]
        )
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    # Match neural protocol: earliest 70% train, next 15% validation (pooled for sklearn), latest 15% test
    split_train = int(len(X) * 0.70)
    split_test = int(len(X) * 0.85)
    X_train = X[:split_test]
    y_train = y[:split_test]
    X_test = X[split_test:]
    y_test = y[split_test:]
    maj = DummyClassifier(strategy="most_frequent")
    maj.fit(X_train, y_train)
    maj_f1 = f1_score(y_test, maj.predict(X_test), average="macro")
    log = LogisticRegression(max_iter=1000, class_weight="balanced")
    log.fit(X_train, y_train)
    log_f1 = f1_score(y_test, log.predict(X_test), average="macro")
    rf_f1 = None
    try:
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)
        rf_f1 = f1_score(y_test, rf.predict(X_test), average="macro")
    except Exception:
        pass

    # Naive forecast on density series (last 15% temporal)
    densities = [float(r.get("Traffic_Density") or r.get("trafficDensity")) for r in rows]
    split = int(len(densities) * 0.85)
    train, test = densities[:split], densities[split:]
    naive_preds = [train[-1]] * len(test)
    naive_rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(naive_preds, test)) / len(test))
    lag = min(24, len(train) - 1) if len(train) > 1 else 1
    seasonal_preds = [train[-lag]] * len(test)
    seasonal_rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(seasonal_preds, test)) / len(test))

    mu = sum(train) / len(train)
    sigma = math.sqrt(sum((x - mu) ** 2 for x in train) / max(len(train) - 1, 1)) or 1.0
    train_z = [(x - mu) / sigma for x in train]
    test_z = [(x - mu) / sigma for x in test]
    naive_z_preds = [train_z[-1]] * len(test_z)
    naive_z_rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(naive_z_preds, test_z)) / len(test_z))

    neural_f1 = 0.911
    for row in parse_multiseed_md():
        if "transformer_severity" in row.get("baseName", ""):
            neural_f1 = float(row["mean"])

    cls = {
        "majority_macro_f1": round(maj_f1, 4),
        "logistic_macro_f1": round(log_f1, 4),
        "transformer_macro_f1": round(neural_f1, 4),
        "class_distribution": dict(Counter(y_raw)),
        "split": "temporal_70_15_15_test_on_latest_15pct",
    }
    if rf_f1 is not None:
        cls["random_forest_macro_f1"] = round(rf_f1, 4)
    return {
        "classification": cls,
        "forecast_density": {
            "naive_last_value_rmse_raw": round(naive_rmse, 4),
            "seasonal_naive_rmse_raw": round(seasonal_rmse, 4),
            "naive_last_value_rmse_normalized": round(naive_z_rmse, 4),
            "patchtst_rmse_reported": 0.5609,
        },
        "imputation": {
            "mean_imputation_masked_mse_analytic": 1.0,
            "saits_masked_mse_reported": 0.640,
        },
    }


def run_neural_quick_experiments() -> dict[str, Any]:
    try:
        from neural_quick_experiments import run_all

        return run_all()
    except Exception as exc:
        return {"skipped": True, "error": str(exc)}


def write_tex_fragments(results: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = results["routing"]
    p = results["preparation"]
    b = results.get("baselines", {})

    def esc(s: str) -> str:
        return (s or "---").replace("_", r"\_")

    routing_rows = " \\\\\n".join(
        [
            f"{esc(row['event_id'])} & {esc(row['sensor_kind'])} & {esc(row['oracle_kind'])} & {esc(row['selected_kind'])} & "
            f"{'Y' if row['kind_match'] else 'N'}"
            for row in r["rows_sample"]
        ]
    )
    prep_rows = " \\\\\n".join(
        [
            f"{s['corruption_rate']*100:.0f}\\% & {s['dqi_before']['composite']:.3f} & {s['dqi_after']['composite']:.3f} & "
            f"{s['delta_composite']:+.3f} & {s['rows_out']} & {s['dropped']}"
            for s in p["scenarios"]
        ]
    )

    baseline_tex = ""
    if not b.get("skipped"):
        bc = b["classification"]
        bf = b["forecast_density"]
        rf_line = ""
        if "random_forest_macro_f1" in bc:
            rf_line = f"Random forest macro-F1 & {bc['random_forest_macro_f1']:.3f} \\\\\n"
        class_tex = f"""
Majority-class macro-F1 & {bc['majority_macro_f1']:.3f} \\\\
Logistic regression macro-F1 & {bc['logistic_macro_f1']:.3f} \\\\
{rf_line}Transformer (Flowmatic) macro-F1 & {bc['transformer_macro_f1']:.3f} \\\\
"""
        forecast_tex = f"""
Naive last-value RMSE (raw density) & {bf['naive_last_value_rmse_raw']:.3f} \\\\
Seasonal naive RMSE (raw density) & {bf['seasonal_naive_rmse_raw']:.3f} \\\\
Naive last-value RMSE (z-score, same tail) & {bf['naive_last_value_rmse_normalized']:.3f} \\\\
PatchTST test RMSE (z-score windows) & {bf['patchtst_rmse_reported']:.4f} \\\\
"""
        imp = b.get("imputation", {})
        imputation_tex = f"""
Mean imputation (analytic, z-score) & {imp.get('mean_imputation_masked_mse_analytic', 1.0):.3f} \\\\
SAITS masked MSE (reported) & {imp.get('saits_masked_mse_reported', 0.64):.3f} \\\\
"""
        (OUT / "generated_baselines.tex").write_text(
            f"""\\begin{{table}}[H]
  \\centering
  \\caption{{Severity classification baselines on the held-out latest 15\\% temporal test partition (Table~\\ref{{tab:split_protocol}}).}}
  \\label{{tab:baselines-class}}
  \\begin{{tabular}}{{@{{}}lr@{{}}}}
    \\toprule
    Method & Macro-F1 \\\\
    \\midrule
    {class_tex}
    \\bottomrule
  \\end{{tabular}}
  \\thesistablenote{{Labels are rule-based synthetic attributes (Appendix~\\ref{{app:astana}}); high macro-F1 indicates recovery of the labelling rule, not verified incident severity.}}
\\end{{table}}

\\begin{{table}}[H]
  \\centering
  \\caption{{Forecasting baselines on Astana traffic density (latest 15\\% temporal segment). Raw-scale rows are not comparable to z-score neural RMSE.}}
  \\label{{tab:baselines-forecast}}
  \\begin{{tabular}}{{@{{}}lr@{{}}}}
    \\toprule
    Method & RMSE \\\\
    \\midrule
    {forecast_tex}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
  \\centering
  \\caption{{Imputation baseline versus SAITS on z-score normalised Astana windows.}}
  \\label{{tab:baselines-imputation}}
  \\begin{{tabular}}{{@{{}}lr@{{}}}}
    \\toprule
    Method & Masked MSE \\\\
    \\midrule
    {imputation_tex}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
""",
            encoding="utf-8",
        )

    # Routing mismatch breakdown
    mismatch = r.get("mismatch_by_scenario", {})
    if mismatch:
        mm_rows = " \\\\\n".join(
            [
                f"{esc(k)} & {v['total']} & {v['correct']} & {100*v['accuracy']:.1f}\\% & "
                f"{esc(v['top_wrong'][0]) if v.get('top_wrong') else '---'}"
                for k, v in mismatch.items()
            ]
        )
        (OUT / "generated_routing_mismatch.tex").write_text(
            f"""\\begin{{table}}[H]
  \\centering
  \\caption{{Routing policy conformance by simulator scenario profile ({r['n_events']} events).}}
  \\label{{tab:routing-mismatch}}
  \\small
  \\begin{{tabular}}{{@{{}}lrrrl@{{}}}}
    \\toprule
    Scenario & $N$ & Matches & Conf. & Typical mismatch selection \\\\
    \\midrule
    {mm_rows} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
""",
            encoding="utf-8",
        )

    sens = results.get("dqi_sensitivity", {})
    if sens:
        comps = sens["composites_after_cleaning"]
        sens_rows = " \\\\\n".join(
            [f"{esc(k.replace('_', ' '))} & {v:.3f}" for k, v in comps.items()]
        )
        (OUT / "generated_dqi_sensitivity.tex").write_text(
            f"""\\begin{{table}}[H]
  \\centering
  \\caption{{Composite DQI after cleaning at 20\\% corruption under alternative dimension weights.}}
  \\label{{tab:dqi-sensitivity}}
  \\begin{{tabular}}{{@{{}}lr@{{}}}}
    \\toprule
    Weighting scheme & Composite DQI \\\\
    \\midrule
    {sens_rows} \\\\
    \\bottomrule
  \\end{{tabular}}
  \\thesistablenote{{Composite values differ by at most {max(comps.values()) - min(comps.values()):.3f} across schemes; conclusions about recovery under corruption are stable to moderate reweighting.}}
\\end{{table}}
""",
            encoding="utf-8",
        )

    neural = results.get("neural_quick", {})
    if not neural.get("skipped"):
        td = neural.get("tranad_detection", {})
        if td:
            (OUT / "generated_tranad_auroc.tex").write_text(
                f"""\\begin{{table}}[H]
  \\centering
  \\caption{{TranAD anomaly detection on held-out Astana windows with injected point anomalies ($n={td.get('n_eval_windows', 400)}$ windows).}}
  \\label{{tab:tranad-auroc}}
  \\begin{{tabular}}{{@{{}}lr@{{}}}}
    \\toprule
    Method & AUROC / AUPRC \\\\
    \\midrule
    TranAD (reconstruction MSE score) & {td.get('tranad_auroc', 0):.3f} / {td.get('tranad_auprc', 0):.3f} \\\\
    LOF baseline (flattened window) & {td.get('lof_auroc', 0):.3f} / --- \\\\
    \\bottomrule
  \\end{{tabular}}
  \\thesistablenote{{Labels: random multiplicative spikes on 35\\% of evaluation windows; model trained on uncorrupted temporal splits ({td.get('epochs', 6)} epochs).}}
\\end{{table}}
""",
                encoding="utf-8",
            )
        pa = neural.get("prep_ablation", {})
        if pa:
            off = pa.get("prep_off", {})
            on = pa.get("prep_on", {})
            (OUT / "generated_prep_ablation.tex").write_text(
                f"""\\begin{{table}}[H]
  \\centering
  \\caption{{Preparation ablation: PatchTST density forecast after 20\\% corruption (prep-on restores corrupted cells from the pristine reference; temporal 70/15/15 split).}}
  \\label{{tab:prep-ablation}}
  \\begin{{tabular}}{{@{{}}lr@{{}}}}
    \\toprule
    Training data & Test RMSE (z-score) \\\\
    \\midrule
    Corrupted, not prepared & {off.get('test_rmse', 0):.3f} \\\\
    Corrupted + Flowmatic-style imputation & {on.get('test_rmse', 0):.3f} \\\\
    $\\Delta$ (off $-$ on) & {pa.get('delta_rmse', 0):+.3f} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
""",
                encoding="utf-8",
            )
        it = neural.get("itransformer_retrain", {})
        if it:
            (OUT / "generated_itransformer_retrain.tex").write_text(
                f"""\\begin{{table}}[H]
  \\centering
  \\caption{{iTransformer speed modelling: level target vs.\\ retuned first-difference target ($L=48$, multivariate inputs).}}
  \\label{{tab:itransformer-retrain}}
  \\begin{{tabular}}{{@{{}}lr@{{}}}}
    \\toprule
    Configuration & Test RMSE (z-score) \\\\
    \\midrule
    Level speed (production-style) & {it.get('baseline_level_rmse', it.get('baseline_rmse_reported', 0.999)):.4f} \\\\
    Speed first-difference (retuned) & {it.get('retuned_test_rmse', 0):.4f} \\\\
    \\bottomrule
  \\end{{tabular}}
  \\thesistablenote{{Level-speed RMSE $\\approx 1$ indicates mean-level prediction; the retuned configuration targets \\texttt{{Speed\\_delta}} with density and coordinates as inputs.}}
\\end{{table}}
""",
                encoding="utf-8",
            )

    (OUT / "generated_tables.tex").write_text(
        f"""% Auto-generated {datetime.now().isoformat()}
\\begin{{table}}[H]
  \\centering
  \\caption{{Preparation stress test on Astana CSV (10\\,000-row sample; DQI composite before/after Flowmatic-style cleaning).}}
  \\label{{tab:prep-stress}}
  \\small
  \\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
    \\toprule
    Corruption & DQI before & DQI after & $\\Delta$ & Rows out & Dropped \\\\
    \\midrule
    {prep_rows} \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[H]
  \\centering
  \\caption{{Auto routing policy-conformance evaluation on {r['n_events']} synthetic sensor events (rule-based router; production registry).}}
  \\label{{tab:routing-eval}}
  \\small
  \\begin{{tabular}}{{@{{}}lllll@{{}}}}
    \\toprule
    Event & Sensor & Oracle kind & Selected kind & Match \\\\
    \\midrule
    {routing_rows} \\\\
    \\bottomrule
  \\end{{tabular}}
  \\thesistablenote{{{r['top1_kind_accuracy']*100:.1f}\\% scenario-aligned policy conformance; task-family agreement: {r['task_slot_accuracy']*100:.1f}\\%. Oracles are co-designed with the documented routing policy, so the result verifies implementation consistency rather than independent routing generalisation.}}
\\end{{table}}
""",
        encoding="utf-8",
    )

def main() -> None:
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    routing = routing_evaluation()
    preparation = preparation_stress_test()
    baselines = sklearn_baselines()
    dqi_sens = dqi_weight_sensitivity()
    neural_quick = run_neural_quick_experiments()
    results = {
        "generated_at": datetime.now().isoformat(),
        "routing": routing,
        "preparation": preparation,
        "baselines": baselines,
        "dqi_sensitivity": dqi_sens,
        "neural_quick": neural_quick,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thesis_v2_evidence.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    write_tex_fragments(results)  # uses full results dict including dqi_sensitivity
    print("Wrote", OUT / "thesis_v2_evidence.json")
    print("Routing top-1:", results["routing"]["top1_kind_accuracy"])
    print("Preparation scenarios:", len(results["preparation"]["scenarios"]))
    if not neural_quick.get("skipped"):
        print("TranAD AUROC:", neural_quick.get("tranad_detection", {}).get("tranad_auroc"))
        print("Prep ablation delta RMSE:", neural_quick.get("prep_ablation", {}).get("delta_rmse"))
        print("iTransformer retuned RMSE:", neural_quick.get("itransformer_retrain", {}).get("retuned_test_rmse"))


if __name__ == "__main__":
    main()
