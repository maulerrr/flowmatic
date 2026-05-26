import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "http://localhost/api/v1"
CSV_PATH = Path(r"c:\Users\BG\Desktop\flowmatic\data\astana_synthetic_data.csv")
OUT_JSON = Path(r"c:\Users\BG\Desktop\flowmatic\thesis\experiments\upload_pipeline_results.json")
OUT_MD = Path(r"c:\Users\BG\Desktop\flowmatic\thesis\experiments\UPLOAD_PIPELINE_REPORT.md")

EMAIL = "thesis.demo@flowmatic.local"
PASSWORD = "ThesisDemo2025!"
DISPLAY = "Thesis Demo"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    result: dict = {
        "experiment": "upload_pipeline",
        "apiBase": BASE,
        "startedAt": utc_now_iso(),
        "timingsSeconds": {},
        "auth": {},
        "upload": {},
        "polling": {},
        "finalRun": None,
        "preview": None,
        "exportHistory": None,
        "parsedSummary": None,
        "errors": [],
        "completedAt": None,
    }

    session = requests.Session()
    session.headers.setdefault("Accept", "application/json")

    t0 = time.perf_counter()

    reg_payload = {
        "email": EMAIL,
        "password": PASSWORD,
        "displayName": DISPLAY,
        "organizationName": "Thesis Demo Org",
    }

    try:
        reg = session.post(f"{BASE}/auth/register", json=reg_payload, timeout=120)
        result["auth"]["register_status"] = reg.status_code
        result["auth"]["register_body_preview"] = reg.text[:2000]

        if reg.status_code >= 400:
            login = session.post(
                f"{BASE}/auth/login",
                json={"email": EMAIL, "password": PASSWORD},
                timeout=120,
            )
            result["auth"]["login_status"] = login.status_code
            result["auth"]["login_body_preview"] = login.text[:2000]
            if login.status_code >= 400:
                result["errors"].append(f"login failed: {login.status_code} {login.text[:800]}")
        else:
            result["auth"]["path"] = "register_ok"
    except Exception as e:
        result["errors"].append(f"auth exception: {e}")

    result["timingsSeconds"]["auth"] = round(time.perf_counter() - t0, 3)
    ck = session.cookies.get("flowmatic_session")
    result["auth"]["session_cookie_present"] = bool(ck)

    if not ck:
        result["completedAt"] = utc_now_iso()
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text("# Upload pipeline experiment\n\nAuth failed.\n", encoding="utf-8")
        print(json.dumps({"ok": False, "step": "auth"}, indent=2))
        return

    t_upload_start = time.perf_counter()
    try:
        with CSV_PATH.open("rb") as f:
            files = {"file": (CSV_PATH.name, f, "text/csv")}
            upload_resp = session.post(f"{BASE}/ingestion/upload", files=files, timeout=600)
        result["upload"]["status"] = upload_resp.status_code
        result["upload"]["elapsedSeconds"] = round(time.perf_counter() - t_upload_start, 3)
        try:
            result["upload"]["body"] = upload_resp.json()
        except json.JSONDecodeError:
            result["upload"]["body"] = upload_resp.text[:8000]
    except Exception as e:
        result["errors"].append(f"upload exception: {e}")

    body = result["upload"].get("body") or {}
    run_id = None
    if isinstance(body, dict):
        data = body.get("data") or {}
        run_id = data.get("runId")
        preview = data.get("preview") or {}
        result["upload"]["runId"] = run_id
        result["upload"]["uploadPreviewRowCount"] = preview.get("rowCount")

    result["timingsSeconds"]["upload"] = result["upload"].get("elapsedSeconds")

    poll_interval = 1.5
    max_wait = 600
    t_poll_start = time.perf_counter()

    def fetch_run():
        return session.get(f"{BASE}/pipelines/runs/{run_id}", timeout=120) if run_id else None

    last_status = None
    polls: list = []

    if run_id:
        deadline = time.perf_counter() + max_wait
        while time.perf_counter() < deadline:
            elapsed = round(time.perf_counter() - t_poll_start, 3)
            gr = fetch_run()
            entry: dict = {"elapsedSincePollStart": elapsed}
            if gr is None:
                entry["error"] = "no run id"
                polls.append(entry)
                break

            entry["status_code"] = gr.status_code
            try:
                payload = gr.json()
            except json.JSONDecodeError:
                payload = {"raw": gr.text[:2000]}
            entry["payload"] = payload

            d = payload.get("data") if isinstance(payload, dict) else None
            st = d.get("status") if isinstance(d, dict) else None
            last_status = st
            entry["run_status"] = st
            polls.append(entry)

            if st in {"completed", "failed"}:
                break
            time.sleep(poll_interval)

        result["polling"]["polls"] = polls
        result["polling"]["elapsedSeconds"] = round(time.perf_counter() - t_poll_start, 3)
        result["polling"]["finalStatus"] = last_status

        gr_final = fetch_run()
        if gr_final and gr_final.ok:
            try:
                jr = gr_final.json()
                fr = jr.get("data")
                result["finalRun"] = fr
                if isinstance(fr, dict):
                    summ = fr.get("summary")
                    if isinstance(summ, str) and summ.strip().startswith("{"):
                        try:
                            result["parsedSummary"] = json.loads(summ)
                        except json.JSONDecodeError:
                            result["parsedSummary_truncated"] = summ[:4000]
            except json.JSONDecodeError:
                pass

        pr = session.get(f"{BASE}/pipelines/runs/{run_id}/preview", timeout=120)
        result["preview"] = {"status": pr.status_code}
        try:
            result["preview"]["body"] = pr.json()
        except json.JSONDecodeError:
            result["preview"]["body"] = pr.text[:8000]

        eh = session.get(f"{BASE}/exports/runs/{run_id}/history", timeout=60)
        result["exportHistory"] = {"status": eh.status_code}
        try:
            result["exportHistory"]["body"] = eh.json()
        except json.JSONDecodeError:
            result["exportHistory"]["body"] = eh.text[:4000]

    result["completedAt"] = utc_now_iso()
    result["timingsSeconds"]["polling"] = result["polling"].get("elapsedSeconds")
    result["timingsSeconds"]["total"] = round(time.perf_counter() - t0, 3)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    fr = result.get("finalRun")
    prv = result.get("preview") or {}
    eh = result.get("exportHistory") or {}

    md = []
    md.append("# Upload Pipeline Experiment Report")
    md.append("")
    md.append(f"- **Completed at**: `{result['completedAt']}`")
    md.append(f"- **API**: `{BASE}`")
    md.append("")
    md.append("## Backend flow")
    md.append("")
    md.append("1. Upload validates CSV, stores source, queues pipeline job.")
    md.append("2. Worker parses CSV, runs quality + cleaning, stores metrics and summary on the run.")
    md.append("")
    md.append("## Outcome snapshot")
    md.append("")
    auth_ok = result["auth"].get("session_cookie_present")
    md.append("| Step | OK | Detail |")
    md.append("| --- | --- | --- |")
    md.append(f"| Session | {'yes' if auth_ok else 'no'} | register {result['auth'].get('register_status')} login {result['auth'].get('login_status', 'na')} |")
    upl = result.get("upload", {}).get("status"); upl_ok = isinstance(upl, int) and 200 <= upl < 300
    md.append(f"| Upload | {'yes' if upl_ok else 'no'} | HTTP {result['upload'].get('status')} in {result['upload'].get('elapsedSeconds')}s |")
    md.append(f"| Pipeline | `{result['polling'].get('finalStatus', 'n/a')}` | polled {result['polling'].get('elapsedSeconds')}s |")

    md.append("")
    md.append("## Metrics")
    md.append("")
    if isinstance(fr, dict):
        for key in ("status", "rowsIngested", "rowsCleaned", "rowsErrors", "processingTimeMs"):
            if key in fr:
                md.append(f"- **{key}**: {fr[key]}")
        rf = fr.get("resultFile")
        if isinstance(rf, dict) and rf:
            md.append(f"- **resultFile**: `{rf.get('fileName')}` id `{rf.get('id')}` size `{rf.get('fileSize')}`")
        summ = fr.get("summary")
        if isinstance(summ, str):
            md.append("")
            md.append("### Stored pipeline summary (snippet)")
            md.append("")
            md.append("```json")
            md.append((summ[:4000] + "...(truncated)") if len(summ) > 4000 else summ)
            md.append("```")
    else:
        md.append("(No final run payload.)")

    prv_body = prv.get("body")
    if isinstance(prv_body, dict) and prv_body.get("success") is True:
        pdata = prv_body.get("data")
        if isinstance(pdata, dict) and pdata.get("stats"):
            md.append("")
            md.append("### Preview stats")
            md.append("")
            md.append("```json")
            md.append(json.dumps(pdata.get("stats"), indent=2))
            md.append("```")

    ps = result.get("parsedSummary")
    if isinstance(ps, dict):
        md.append("")
        md.append("### Parsed summary.scores")
        scores = ps.get("scores")
        if isinstance(scores, dict):
            md.append("")
            md.append("```json")
            md.append(json.dumps(scores, indent=2))
            md.append("```")

    md.append("")
    md.append("## Export")
    md.append("")
    md.append(f"- History endpoint HTTP `{eh.get('status')}` (explicit POST export required).")
    if isinstance(eh.get("body"), dict):
        md.append("")
        md.append("```json")
        md.append(json.dumps(eh["body"], indent=2, default=str)[:6000])
        md.append("```")

    md.append("")
    md.append("## Timing (seconds)")
    md.append("")
    md.append("```json")
    md.append(json.dumps(result["timingsSeconds"], indent=2))
    md.append("```")

    errs = result.get("errors") or []
    if errs:
        md.append("")
        md.append("## Errors")
        for msg in errs:
            md.append(f"- {msg}")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    succeeded = upl_ok and result["polling"].get("finalStatus") == "completed"
    print(json.dumps({"ok": succeeded, "run_status": result["polling"].get("finalStatus")}, indent=2))


if __name__ == "__main__":
    main()

