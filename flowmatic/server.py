# flowmatic/server.py

import os
import uuid
import tempfile
import traceback
import urllib.parse

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import openai
import uvicorn
from flowmatic.ingestion import ingest
from flowmatic.quality_check import quality_report, detect_outliers_zscore
from flowmatic.cleaning import clean
from flowmatic.hf_push import push_df_to_hf
from flowmatic.db_upload import build_postgres_url, upload_df_to_postgres

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In‐memory storage for cleaned DataFrames & quality reports
CLEANED_DATA = {}
QUALITY_REPORTS = {}

try:
    openai_key = os.environ.get("OPENAI_API_KEY") or ""
    openai.api_key = openai_key
except Exception:
    openai_key = ""

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "openai_available": bool(openai_key), "initial": True},
    )

@app.post("/process", response_class=HTMLResponse)
async def post_process(
    request: Request,
    upload_file: UploadFile = File(None),
    hf_dataset: str = Form(""),
    hf_split: str = Form("train"),
    hf_token: str = Form(""),
):
    try:
        # Only treat it as a real file if upload_file.filename is non-empty
        if upload_file is not None and upload_file.filename:
            ext = os.path.splitext(upload_file.filename)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(upload_file.file, parse_dates=True, index_col=0)
            elif ext == ".json":
                df = pd.read_json(upload_file.file)
                # Find a datetime-like column
                datetime_cols = [
                    c for c in df.columns if "date" in c.lower() or "time" in c.lower()
                ]
                if datetime_cols:
                    dt_col = datetime_cols[0]
                    df[dt_col] = pd.to_datetime(df[dt_col], errors="raise")
                    df = df.set_index(dt_col)
                else:
                    return HTMLResponse(
                        content="<h3>Error: No column containing 'date' or 'time' found in JSON.</h3>",
                        status_code=400,
                    )
            else:
                return HTMLResponse(
                    content=f"<h3>Unsupported file type: {ext}</h3>", status_code=400
                )

        elif hf_dataset:
            # Only ingest from HF if the dataset field is non-empty
            df = ingest(hf_dataset, split=hf_split, token=hf_token or None)

        else:
            # Neither a valid file was uploaded nor an HF dataset was provided
            return HTMLResponse(
                content="<h3>No file uploaded or HF dataset ID provided.</h3>",
                status_code=400,
            )

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="raise")

    except Exception:
        tb = traceback.format_exc()
        return HTMLResponse(content=f"<pre>Error loading data:\n{tb}</pre>", status_code=500)


   # ─── Data Quality Check ───────────────────────────────────────────────
    try:
        qr = quality_report(df)
        # Extract metrics:
        missing_dict = qr["missing"].to_dict()            # { column_name: missing_count, … }
        duplicates_count = qr["duplicates"]                # int
        outlier_count = len(qr["outliers"])                # int
    except Exception:
        tb = traceback.format_exc()
        return HTMLResponse(content=f"<pre>Error during quality check:\n{tb}</pre>", status_code=500)

    # Store these metrics in QUALITY_REPORTS (keyed by data_id).
    data_id = str(uuid.uuid4())
    QUALITY_REPORTS[data_id] = {
        "missing": missing_dict,
        "duplicates": duplicates_count,
        "outliers": outlier_count,
    }

    # ─── Cleaning ───────────────────────────────────────────────────────
    try:
        df_clean = clean(df)
    except Exception:
        tb = traceback.format_exc()
        return HTMLResponse(content=f"<pre>Error during cleaning:\n{tb}</pre>", status_code=500)

    CLEANED_DATA[data_id] = df_clean
    # Redirect to results
    return RedirectResponse(url=f"/results/{data_id}", status_code=302)

@app.get("/results/{data_id}", response_class=HTMLResponse)
async def get_results(request: Request, data_id: str):
    if data_id not in CLEANED_DATA or data_id not in QUALITY_REPORTS:
        return HTMLResponse(content="<h3>Data not found.</h3>", status_code=404)

    df_clean = CLEANED_DATA[data_id]
    qr_metrics = QUALITY_REPORTS[data_id]
    missing_dict   = qr_metrics["missing"]      # {col: count, …}
    duplicates_cnt = qr_metrics["duplicates"]   # int
    outliers_cnt   = qr_metrics["outliers"]     # int

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "openai_available": bool(openai_key),
            "initial": False,
            "data_id": data_id,
            "cleaned_head": df_clean.head(50).to_dict(orient="records"),
            "columns": list(df_clean.columns),
            # Pass the QA metrics into the template:
            "missing_dict": missing_dict,
            "duplicates_cnt": duplicates_cnt,
            "outliers_cnt": outliers_cnt,
        },
    )

@app.get("/download/{data_id}")
async def download_file(data_id: str, fmt: str = "csv"):
    if data_id not in CLEANED_DATA:
        return HTMLResponse(content="<h3>Data not found.</h3>", status_code=404)

    df_clean = CLEANED_DATA[data_id]
    suffix = ".csv" if fmt == "csv" else ".json"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    if fmt == "csv":
        df_clean.to_csv(tmp_path, index=True)
    else:
        df_clean.to_json(tmp_path, date_format="iso", orient="records")
    tmp.close()
    filename = f"flowmatic_cleaned_{data_id}{suffix}"
    return FileResponse(path=tmp_path, filename=filename, media_type="text/csv" if fmt == "csv" else "application/json")

@app.post("/push_hf")
async def post_push_hf(
    request: Request,
    data_id: str = Form(...),
    hf_token: str = Form(...),
    hf_repo_name: str = Form(...),
):
    if data_id not in CLEANED_DATA:
        return HTMLResponse(content="<h3>Data not found.</h3>", status_code=404)

    df_clean = CLEANED_DATA[data_id]
    try:
        push_df_to_hf(
            df=df_clean,
            repo_name=hf_repo_name,
            token=hf_token,
            path_in_repo="flowmatic_cleaned.csv",
            commit_message="Add cleaned data via Flowmatic",
            branch="main",
        )
        # Redirect back with hf_status=success
        params = urllib.parse.urlencode({"hf_status": "success"})
        return RedirectResponse(url=f"/results/{data_id}?{params}", status_code=302)
    except Exception as e:
        msg = urllib.parse.quote(str(e))
        params = urllib.parse.urlencode({"hf_status": "error", "hf_msg": msg})
        return RedirectResponse(url=f"/results/{data_id}?{params}", status_code=302)

@app.post("/upload_db")
async def post_upload_db(
    request: Request,
    data_id: str = Form(...),
    pg_host: str = Form("localhost"),
    pg_port: int = Form(5432),
    pg_db: str = Form("flowmatic"),
    pg_user: str = Form("postgres"),
    pg_pass: str = Form(""),
    pg_table: str = Form("test"),
):
    if data_id not in CLEANED_DATA:
        return HTMLResponse(content="<h3>Data not found.</h3>", status_code=404)

    df_clean = CLEANED_DATA[data_id]
    try:
        db_url = build_postgres_url(
            username=pg_user or "postgres",
            password=pg_pass or "",
            host=pg_host or "localhost",
            port=pg_port or 5432,
            database=pg_db or "flowmatic",
        )
        upload_df_to_postgres(
            df=df_clean,
            table_name=pg_table or "test",
            db_url=db_url,
            if_exists="append",
            index=False,
        )
        params = urllib.parse.urlencode({"db_status": "success"})
        return RedirectResponse(url=f"/results/{data_id}?{params}", status_code=302)
    except Exception as e:
        msg = urllib.parse.quote(str(e))
        params = urllib.parse.urlencode({"db_status": "error", "db_msg": msg})
        return RedirectResponse(url=f"/results/{data_id}?{params}", status_code=302)


if __name__ == "__main__":
    uvicorn.run(
        "flowmatic.server:app",  # module:app
        host="0.0.0.0",
        port=5124,
        reload=True,            
    )