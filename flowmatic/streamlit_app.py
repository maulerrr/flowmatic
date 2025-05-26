import os
import pandas as pd
import streamlit as st
from flowmatic.ingestion import ingest
from flowmatic.quality_check import quality_report, detect_outliers_zscore
from flowmatic.cleaning import clean
import openai

# —————————————————————————————————————————————————————————
# Page config
# —————————————————————————————————————————————————————————
st.set_page_config(page_title="flowmatic Preview", layout="wide")

# —————————————————————————————————————————————————————————
# OpenAI key (optional, safe fallback if no secrets.toml)
# —————————————————————————————————————————————————————————
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    openai_key = os.getenv("OPENAI_API_KEY", None)

if openai_key:
    openai.api_key = openai_key

# —————————————————————————————————————————————————————————
# Sidebar: Data Ingestion
# —————————————————————————————————————————————————————————
st.sidebar.header("1) Data Ingestion")

upload = st.sidebar.file_uploader(
    "Upload CSV or JSON", type=["csv", "json"]
)

hf_dataset = st.sidebar.text_input(
    "Or HuggingFace dataset ID", placeholder="e.g. user/smart-city-traffic"
)
hf_split = st.sidebar.selectbox("HF split", ["train", "test", "validation"])

if st.sidebar.button("Load Data"):
    if upload:
        # Directly read the uploaded file
        ext = os.path.splitext(upload.name)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(upload, parse_dates=True, index_col=0)
        elif ext == ".json":
            df = pd.read_json(upload).set_index("datetime")
        else:
            st.sidebar.error(f"Unsupported upload type: {ext}")
            st.stop()
    elif hf_dataset:
        # HF ingest; token is optional
        df = ingest(hf_dataset, split=hf_split, token=os.getenv("HF_TOKEN", None))
    else:
        st.sidebar.error("Please upload a file _or_ enter an HF dataset ID.")
        st.stop()

    st.session_state["df"] = df
    st.success(f"✅ Data loaded: {df.shape}")

# —————————————————————————————————————————————————————————
# Main: Preview & Quality Report
# —————————————————————————————————————————————————————————
if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(200))

    st.subheader("Quality Report")
    qr = quality_report(df)
    st.markdown("**Missing values per column:**")
    st.table(qr["missing"])
    st.markdown(f"**Duplicate rows:** {qr['duplicates']}")

    outliers = detect_outliers_zscore(df)
    st.markdown(f"**Outliers detected:** {len(outliers)} rows")
    if not outliers.empty:
        st.dataframe(outliers)

    # —————————————————————————————————————————————————————————
    # 2) Explain anomalies via OpenAI (optional)
    # —————————————————————————————————————————————————————————
    if openai_key:
        st.subheader("Explain Anomalies (OpenAI)")
        if not outliers.empty:
            # build a minimal summary
            summary = outliers.describe().to_dict()
            prompt = (
                "Detected these outlier summary stats in a time-series:\n"
                f"{summary}\n"
                "Explain potential causes in an urban traffic/passenger-flow context."
            )
            if st.button("Ask OpenAI to Explain"):
                with st.spinner("🔍 Generating explanation…"):
                    resp = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        max_tokens=256,
                        temperature=0.4,
                    )
                st.write(resp.choices[0].text.strip())
        else:
            st.info("No outliers to explain.")
    else:
        st.info("🔒 OpenAI key not configured; anomaly explanations are disabled.")

    # —————————————————————————————————————————————————————————
    # 3) Clean & Export Data
    # —————————————————————————————————————————————————————————
    st.subheader("Clean & Export Data")
    if st.button("Run Cleaning Pipeline"):
        df_clean = clean(df)
        st.success(f"✅ Cleaned data: {df_clean.shape}")
        st.download_button(
            "Download cleaned CSV",
            df_clean.to_csv().encode("utf-8"),
            file_name="flowmatic_cleaned.csv",
            mime="text/csv",
        )
