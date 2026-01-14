# Flowmatic - Intelligent Data Preparation Platform

An end-to-end data preparation platform with NestJS backend and Vue 3 frontend.

## 🏗️ Architecture

- **Backend**: NestJS (TypeScript)
- **Frontend**: Vue 3 + TypeScript + Vite
- **Database**: SQLite (development) / PostgreSQL (production)
- **Data Processing**: Native TypeScript with statistical analysis
- **Runtime**: Bun (fast JavaScript runtime)

## 📋 Prerequisites

- [Bun](https://bun.sh) 1.0+
- Git

## 🚀 Quick Start

### Install Bun (if not already installed)

```bash
# Windows
powershell -c "irm bun.sh/install.ps1|iex"

# macOS/Linux
curl -fsSL https://bun.sh/install | bash
```

### Backend Setup

```bash
cd backend
bun install
cp .env.example .env
bun run start:dev
```

Backend runs on `http://localhost:3000`
API docs available at `http://localhost:3000/api`

### Frontend Setup

```bash
cd frontend
bun install
bun run dev
```

Frontend runs on `http://localhost:5173`

---

## Flowmatic Pipeline Overview

Here’s a high-level diagram of the Flowmatic preprocessing workflow:

![Flowmatic Pipeline](docs/images/flowmatic-pipeline.jpg)

---

## Features

* **Data Ingestion**
  – Load local CSV/JSON files or pull from public/private Hugging Face datasets
* **Data Quality Checking**
  – Report missing values, count duplicates, and detect statistical outliers via Z-score
* **Data Cleaning**
  – Remove duplicates, interpolate or forward-fill missing values, winsorize outliers
* **FastAPI Web Interface**
  – Upload or load data, view quality reports, preview cleaned data, download CSV/JSON
* **Export Options**
  – Push cleaned data to a Hugging Face Hub dataset or upload to PostgreSQL
* **OpenAI Anomaly Explanations (Optional)**
  – Ask the model to interpret potential real-world causes of outliers (if API key is configured)

---

## Platform Extensions (New)

Flowmatic now includes a lightweight platform layer for managing multiple datasets, reproducible pipelines, and model artifacts:

* **Dataset Registry (SQLite)** – Track datasets, their versions, and provenance (`flowmatic/store.py`).
* **Pipeline Orchestration** – Define and run composable steps (ingest → quality → clean) with artifacts and metrics saved per run (`flowmatic/pipeline.py`).
* **Artifacts & Runs** – Artifacts saved under `data/runs/<version>/` and status recorded as succeeded/failed with metrics.
* **Model Artifacts** – Train baseline models on cleaned data and save under `models/` with metadata (`flowmatic/models/anomaly.py`).
* **API Endpoints** – List datasets and trigger pipeline runs from the browser or programmatically.

Quick endpoints:

```
GET  /datasets                 # List discovered cleaned dataset files
POST /pipelines/run            # Start default pipeline on local path or HF dataset
    Form fields: path | hf_dataset, hf_split, hf_token, dataset_name, version
GET  /download_artifacts?dir=… # Browse run artifacts (served from /data)
```

---

## Project Structure

```flowmatic/
├── README.md
├── requirements.txt
├── docs/
│   ├── images/
│   │   └── flowmatic-pipeline.jpg    # Pipeline visualization
│   └── sample_traffic_data.csv       # Example dataset
├── static/
│   └── icons/
│       ├── upload.svg
│       ├── huggingface.svg
│       ├── postgres.svg
│       └── psql.svg
├── templates/
│   └── index.html                    # Jinja2 template for FastAPI
├── flowmatic/
│   ├── __init__.py
│   ├── streamlit-demo/
│   │   └── app.py                    # (Optional) Streamlit demo entrypoint
│   ├── ingestion.py                  # Loading CSV/JSON or HF datasets
│   ├── quality_check.py              # Reporting missing values, duplicates, outliers
│   ├── cleaning.py                   # Imputation, duplicate removal, outlier capping
│   ├── hf_push.py                    # Helpers to push DataFrame to HF Hub
│   ├── db_upload.py                  # Helpers to upload DataFrame to PostgreSQL
│   └── server.py                     # FastAPI server exposing Flowmatic functionality
├── .env.example                      # Example environment variables
└── .gitignore
```

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/flowmatic.git
   cd flowmatic
   ```

2. **Create a virtual environment & install dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate         # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
flowmatic/
├── backend/                    # NestJS backend
│   ├── src/
│   │   ├── entities/          # TypeORM entities
│   │   ├── modules/
│   │   │   ├── ingestion/     # Data loading (CSV, JSON, HF, URLs)
│   │   │   ├── quality/       # Quality analysis
│   │   │   ├── cleaning/      # Data cleaning
│   │   │   ├── pipeline/      # Pipeline orchestration
│   │   │   ├── export/        # Data export
│   │   │   └── storage/       # Dataset registry
│   │   ├── app.module.ts
│   │   └── main.ts
│   ├── package.json
│   └── tsconfig.json
├── frontend/                   # Vue 3 frontend
│   ├── src/
│   │   ├── assets/            # Styles
│   │   ├── components/        # Vue components
│   │   ├── views/             # Page views
│   │   ├── router/            # Vue router
│   │   ├── services/          # API client
│   │   ├── App.vue
│   │   └── main.ts
│   ├── package.json
│   └── vite.config.ts
├── archive-python/             # Original Python implementation
├── docs/                       # Documentation
└── README.md
```

## 🔌 API Endpoints

### Ingestion
- `POST /ingestion/upload` - Upload file
- `POST /ingestion/url` - Ingest from URL
- `POST /ingestion/huggingface` - Ingest from HuggingFace

### Pipelines
- `POST /pipelines/run` - Run data preparation pipeline
- `GET /pipelines/runs` - List all pipeline runs
- `GET /pipelines/runs/:id` - Get pipeline run details

### Export
- `POST /export/csv` - Export as CSV
- `POST /export/json` - Export as JSON
- `POST /export/huggingface` - Push to HuggingFace
- `POST /export/database` - Export to database

## 🎯 Features

### Data Ingestion
- CSV/JSON file upload
- URL-based ingestion
- HuggingFace datasets
- Automatic datetime detection
- Schema inference

### Quality Analysis
- Missing value detection
- Duplicate identification
- Outlier detection (Z-score)
- Column type classification
- Statistical summaries

### Data Cleaning
- Duplicate removal
- Missing value imputation
- Outlier handling (winsorization)
- Adaptive strategies

### Pipeline Orchestration
- Reproducible workflows
- Artifact storage
- Metrics tracking
- Dataset versioning

## 🛠️ Development

### Backend Commands

```bash
bun run start:dev    # Development with hot reload
bun run build        # Build for production
bun run start:prod   # Run production build
bun run lint         # Lint code
bun run test         # Run tests
```

### Frontend Commands

```bash
bun run dev          # Development server
bun run build        # Build for production
bun run preview      # Preview production build
bun run lint         # Lint code
```

## 🌐 Environment Variables

### Backend (.env)

```env
PORT=3000
DB_TYPE=sqlite
DB_DATABASE=flowmatic.db
UPLOAD_DIR=./uploads
ARTIFACTS_DIR=./data/runs
HF_TOKEN=your_huggingface_token
CORS_ORIGIN=http://localhost:5173
```

## 📊 Usage Workflow

1. **Upload Data**: Navigate to Upload page and select your data source
2. **Process**: Pipeline automatically runs quality checks and cleaning
3. **Review**: View quality metrics and cleaning statistics
4. **Export**: Download cleaned data or export to external services

## 🔄 Migration from Python

The original Python implementation has been archived in `archive-python/`. All functionality has been migrated to TypeScript with:

- Improved type safety
- Better async handling
- Cleaner architecture
- Modern web stack
- Bun runtime for blazing fast performance

## 📝 License

MIT

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

### Environment Variables

Create a copy of `.env.example` and fill in any values you need. For example:

```bash
cp .env.example .env
```

Open `.env` in your editor and set:

```
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
```

* `HF_TOKEN` (optional)
  Required if you plan to ingest private Hugging Face datasets or push to a private dataset repo.

* `OPENAI_API_KEY` (optional)
  Required if you want to enable automatic anomaly explanations via the OpenAI API.

---

## Usage

### 1. FastAPI Web Server

This is the primary interface for Flowmatic. It lets you upload or load data, view quality metrics, preview the cleaned results, and export to either Hugging Face or PostgreSQL—all from the browser.

1. **Run the FastAPI server**

   ```bash
   uvicorn flowmatic.server:app --reload
   ```

2. **Open your browser** at `http://127.0.0.1:8000/`.

3. **Workflow**:

   * **Upload or Hugging Face**
     Choose “Upload File” or “Hugging Face” via the radio buttons. If you select “Upload File,” pick a local CSV/JSON; if you select “Hugging Face,” enter the dataset ID, split, and optionally your HF token.

   * **Data Quality Insights**
     After processing, Flowmatic displays:

     * Number of duplicate rows
     * Number of outliers (|Z-score| > 3)
     * A table of missing-value counts per column

   * **Cleaned Data Preview**
     View the first 50 rows of the cleaned DataFrame in a table.

   * **Download Cleaned Data**
     Download as CSV or JSON.

   * **Export Options**
     Select either “Push to Hugging Face” (enter your HF token & repo name) or “Upload to PostgreSQL” (enter or accept defaults for host, port, db, user, password, table). Only the relevant input fields appear based on your selection.

   * **Toast Notifications**
     Success or failure of an HF push or DB upload shows a small “toast” message in the top right, which automatically hides after 5 seconds.

#### Pipelines & Models

To run the default pipeline on a file without using the form:

```
curl -X POST -F path="data/raw/your_file.csv" http://127.0.0.1:8000/pipelines/run
```

To train a baseline anomaly detector on a cleaned dataset:

```
C:/Users/BG/Desktop/flowmatic/.venv/Scripts/python.exe data/model/train.py --path data/cleaned/cleaned.csv --outdir models
```

---

### 2. (Optional) Streamlit Demo

If you prefer an interactive Streamlit app instead, there is a minimal demo in `flowmatic/app.py` (and `flowmatic/streamlit_demo/` if present). This provides similar ingestion → quality check → cleaning → export workflows inside a Streamlit interface.

To run the Streamlit demo:

```bash
streamlit run flowmatic/streamlit-demo/app.py
```

> **Note:** The Streamlit demo is optional. The FastAPI server is the recommended production interface.

---

## Module Overview

### flowmatic/ingestion.py

* **`ingest(source: str, split: str=None, token: str=None) → pd.DataFrame`**
  Loads local CSV/JSON or Hugging Face datasets, automatically detects a datetime‐like column to set as the index, and returns a `DataFrame` with a `DatetimeIndex`.

### flowmatic/quality\_check.py

* **`report_missing(df: pd.DataFrame) → pd.Series`**
  Returns the count of missing values per column.
* **`report_duplicates(df: pd.DataFrame) → int`**
  Returns the number of duplicate rows.
* **`detect_outliers_zscore(df: pd.DataFrame, threshold: float=3.0) → pd.DataFrame`**
  Returns a sub-DataFrame of rows whose numeric columns exceed the Z-score threshold.
* **`quality_report(df: pd.DataFrame) → dict`**
  Prints a summary to console and returns a dictionary containing:

  ```
  {
    "missing": pd.Series,       # missing count per column
    "duplicates": int,          # total duplicate rows
    "outliers": pd.DataFrame,   # rows flagged as outliers
  }
  ```

### flowmatic/cleaning.py

* **`clean(df: pd.DataFrame) → pd.DataFrame`**
  Runs a three-stage cleaning pipeline:

  1. **Remove duplicates**
  2. **Impute missing values** (time-based interpolation or forward/backward fill)
  3. **Cap outliers** using winsorization (clipping to specified quantiles)
     Returns a cleaned `DataFrame`.

### flowmatic/hf\_push.py

* **`ensure_hf_repo(repo_name: str, token: str, private: bool=False) → str`**
  Checks if a Hugging Face Hub *dataset* repo exists for your username. If not, creates it. Returns the full repo ID (e.g. `username/repo_name`).
* **`push_df_to_hf(df: pd.DataFrame, repo_name: str, token: str, path_in_repo: str="cleaned.csv", commit_message: str="Add cleaned data", branch: str="main") → None`**
  Exports `df` to a temporary CSV and `upload_file(...)` to the HF Hub dataset under `path_in_repo`.

### flowmatic/db\_upload.py

* **`build_postgres_url(username: str, password: str, host: str, port: int, database: str) → str`**
  Constructs a SQLAlchemy database URL for PostgreSQL (e.g. `postgresql+psycopg2://user:pw@host:port/db`).
* **`infer_sqlalchemy_types(df: pd.DataFrame) → dict`**
  Infers an appropriate SQLAlchemy dtype (Integer, Float, DateTime, Boolean, Text) for each column.
* **`upload_df_to_postgres(df: pd.DataFrame, table_name: str, db_url: str, if_exists: str="append", index: bool=False, custom_dtypes: dict=None) → None`**
  Uses `df.to_sql(...)` to create or append to the specified table in PostgreSQL. If the table does not exist, it’s created with the DataFrame’s schema.

### flowmatic/server.py

* Defines FastAPI endpoints to support the above:

  * **`GET /`** → Renders `index.html` initial form
  * **`POST /process`** → Ingest, run `quality_report`, run `clean`, store results under a UUID, redirect to `/results/{data_id}`
  * **`GET /results/{data_id}`** → Render `index.html` with quality insights, cleaned table preview, download links, and export‐option forms
  * **`GET /download/{data_id}`** → Stream cleaned data as CSV or JSON
  * **`POST /push_hf`** → Push cleaned data to HF, then redirect back with `?hf_status=…`
  * **`POST /upload_db`** → Upload cleaned data to PostgreSQL, then redirect back with `?db_status=…`

---

## Contributing

Contributions and issue reports are welcome! Please open a pull request or GitHub issue and adhere to the existing code style. Any major new feature should come with updated documentation and tests.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
