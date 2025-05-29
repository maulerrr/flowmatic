# Flowmatic

**Flowmatic** is an end-to-end, modular pipeline for time-series data preprocessing in smart-city environments. It provides tools to ingest raw sensor streams (CSV/JSON or Hugging Face datasets), automatically detect data-quality issues (missing values, duplicates, outliers), clean or impute those anomalies, and preview the results in an interactive Streamlit app. Optional integration with the OpenAI API enables automated explanations of detected anomalies.

## Flowmatic Pipeline Overview

Here’s a high-level diagram of the Flowmatic preprocessing workflow:

![Flowmatic Pipeline](docs\images\flowmatic-pipeline.jpg)

## Features

- **Data Ingestion**  
  – Load local CSV/JSON files or pull from public/private Hugging Face datasets  
- **Data Quality Checking**  
  – Report missing values, count duplicates, and detect statistical outliers via Z-score  
- **Data Cleaning**  
  – Remove duplicates, interpolate or forward-fill missing values, winsorize outliers  
- **Streamlit Preview App**  
  – Upload or load data, view quality reports and raw vs. cleaned data side by side  
- **OpenAI Anomaly Explanations (Optional)**  
  – Ask the model to interpret potential real-world causes of outliers  

---

## Project Structure

```
flowmatic/
├── README.md
├── requirements.txt
├── data/
│ ├── raw/ # Place your raw CSV/JSON files here
│ └── cleaned/ # Cleaned output is saved here
└── flowmatic/
├── init.py
├── ingestion.py # Loading CSV/JSON or HF datasets
├── quality_check.py # Reporting missing values, duplicates, outliers
├── cleaning.py # Imputation, duplicate removal, outlier capping
├── main.py # CLI-driven end-to-end processing
└── streamlit_app.py # Interactive Streamlit GUI
```

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/flowmatic.git
   cd flowmatic
   ```

2. Create a virtual environment & install dependencies

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate      # macOS/Linux
    .venv\Scripts\activate         # Windows
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Configuration
### Hugging Face Datasets

To load private HF datasets, set HF_TOKEN in your environment:

```bash
export HF_TOKEN="your_hf_token"    # macOS/Linux
set HF_TOKEN="your_hf_token"       # Windows
```

### OpenAI Anomaly Explanations (optional)

To enable anomaly explanations, provide an API key via Streamlit secrets or environment variable:

```bash
# Option A: ~/.streamlit/secrets.toml
[general]
OPENAI_API_KEY = "your_openai_key"

# Option B: environment variable
export OPENAI_API_KEY="your_openai_key"
```

## Usage
### 1. CLI-Based Batch Processing

Make sure your raw data is in `data/raw/`
```bash
python -m flowmatic.main
```
Each file in `data/raw/` will be:

1. Ingested (CSV/JSON or HF)

2. Quality-checked (missing, duplicates, outliers)

3. Cleaned (imputed, deduplicated, winsorized)

4. Saved as *_cleaned.csv under data/cleaned/

### 2. Interactive Streamlit App
```bash
streamlit run flowmatic/streamlit_app.py
```
1. Upload a local CSV/JSON or enter a Hugging Face dataset ID

2. View raw data preview and automated quality report

3. (Optional) Ask OpenAI to explain detected outliers

4. Run the cleaning pipeline and download the cleaned CSV

## Module Overview
### flowmatic/ingestion.py
`ingest(source, split=None, token=None)` → `DataFrame`
Loads local CSV/JSON or HF datasets, ensures a DatetimeIndex.

### flowmatic/quality_check.py
`quality_report(df)` → `dict`
Reports missing counts, duplicate rows, and extracts Z-score outliers.

### flowmatic/cleaning.py
`clean(df)` → `DataFrame`
Removes duplicates, imputes missing data (time-based or ffill), and caps outliers via winsorization.

### flowmatic/main.py
Command-line script to batch-process all files in `data/raw/` and output cleaned versions to `data/cleaned/`.

### flowmatic/streamlit_app.py
Streamlit GUI with optional OpenAI integration for anomaly explanations.

## Contributing
Contributions and issue reports are welcome! Please open a pull request or GitHub issue.

## License
This project is licensed under the MIT License. See LICENSE for details.
