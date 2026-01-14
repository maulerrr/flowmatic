import os
import pandas as pd
import streamlit as st
import openai

from flowmatic.ingestion import ingest
from flowmatic.quality_check import quality_report, detect_outliers_zscore
from flowmatic.cleaning import clean
from flowmatic.hf_push import push_df_to_hf
from flowmatic.db_upload import build_postgres_url, upload_df_to_postgres

# —————————————————————————————————————————————————————————
# Page config
# —————————————————————————————————————————————————————————
st.set_page_config(page_title="Flowmatic Preview", layout="wide")

# —————————————————————————————————————————————————————————
# OpenAI key (optional)
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
upload = st.sidebar.file_uploader("Upload CSV or JSON", type=["csv", "json"])
hf_dataset = st.sidebar.text_input("Or Hugging Face dataset ID", placeholder="e.g. user/smart-city-traffic")
hf_split = st.sidebar.selectbox("HF split", ["train", "test", "validation"])

if st.sidebar.button("Load Data"):
    if upload:
        ext = os.path.splitext(upload.name)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(upload, parse_dates=True, index_col=0)
        elif ext == ".json":
            df = pd.read_json(upload).set_index("datetime")
        else:
            st.sidebar.error(f"Unsupported upload type: {ext}")
            st.stop()
    elif hf_dataset:
        df = ingest(hf_dataset, split=hf_split, token=os.getenv("HF_TOKEN", None))
    else:
        st.sidebar.error("Please upload a file _or_ enter an HF dataset ID.")
        st.stop()

    st.session_state["df"] = df
    # Clear any previously cleaned data
    if "df_clean" in st.session_state:
        del st.session_state["df_clean"]
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
    # 3) Clean Data (single button, stores df_clean)
    # —————————————————————————————————————————————————————————
    st.subheader("Clean Data")
    if st.button("Run Cleaning Pipeline"):
        df_clean = clean(df)
        st.session_state["df_clean"] = df_clean
        st.success(f"✅ Cleaned data: {df_clean.shape}")

    # —————————————————————————————————————————————————————————
    # 4) Download cleaned data (always visible once df_clean exists)
    # —————————————————————————————————————————————————————————
    if "df_clean" in st.session_state:
        df_clean = st.session_state["df_clean"]

        st.subheader("Download Cleaned Data")
        fmt = st.radio("Select format:", ("CSV", "JSON"), index=0, key="download_fmt")
        if fmt == "CSV":
            st.download_button(
                "Download cleaned CSV",
                df_clean.to_csv().encode("utf-8"),
                file_name="flowmatic_cleaned.csv",
                mime="text/csv",
                key="dl_csv",
            )
        else:
            st.download_button(
                "Download cleaned JSON",
                df_clean.to_json(date_format="iso", orient="records").encode("utf-8"),
                file_name="flowmatic_cleaned.json",
                mime="application/json",
                key="dl_json",
            )

        # —————————————————————————————————————————————————————————
        # 5) Export Options (only if cleaned data is available)
        # —————————————————————————————————————————————————————————
        st.subheader("Export Options")

        col1, col2 = st.columns(2)

        # —Push to HF —
        with col1:
            if st.button("Push to Hugging Face"):
                st.session_state["push_to_hf_active"] = True

            if st.session_state.get("push_to_hf_active", False):
                st.markdown("**Enter HF credentials**")
                hf_token_input = st.text_input(
                    "HF_TOKEN:", type="password", key="hf_token"
                )
                hf_repo_name = st.text_input(
                    "Repo Name:",
                    placeholder="your-username/flowmatic_dataset",
                    key="hf_repo",
                )

                if hf_token_input and hf_repo_name:
                    if st.button("Confirm Push to HF", key="confirm_hf"):
                        try:
                            push_df_to_hf(
                                df=df_clean,
                                repo_name=hf_repo_name,
                                token=hf_token_input,
                                path_in_repo="flowmatic_cleaned.csv",
                                commit_message="Add cleaned data via Flowmatic",
                                branch="main",
                            )
                            st.success(
                                f"✅ Successfully pushed to HF repo `{hf_repo_name}`."
                            )
                            st.session_state["push_to_hf_active"] = False
                        except Exception as e:
                            st.error(f"Failed to push to HF: {e}")

        # —Upload to PostgreSQL—
        with col2:
            if st.button("Upload to PostgreSQL"):
                st.session_state["upload_to_db_active"] = True

            if st.session_state.get("upload_to_db_active", False):
                st.markdown("**Enter PostgreSQL credentials (press Enter to use defaults):**")
                pg_host = st.text_input("Host:", value="localhost", key="pg_host")
                pg_port = st.text_input("Port:", value="5432", key="pg_port")
                pg_db = st.text_input("Database Name:", value="flowmatic", key="pg_db")
                pg_user = st.text_input("Username:", value="postgres", key="pg_user")
                pg_pass = st.text_input("Password:", type="password", key="pg_pass")
                pg_table = st.text_input("Table Name:", value="test", key="pg_table")

                if st.button("Confirm Upload to PostgreSQL", key="confirm_pg"):
                    try:
                        # Build URL (defaults applied if empty)
                        db_url = build_postgres_url(
                            username=pg_user or "postgres",
                            password=pg_pass or "",
                            host=pg_host or "localhost",
                            port=int(pg_port or 5432),
                            database=pg_db or "flowmatic",
                        )
                        # Use 'append' mode so that if table doesn't exist, it will be created with correct schema
                        upload_df_to_postgres(
                            df=df_clean,
                            table_name=pg_table or "test",
                            db_url=db_url,
                            if_exists="append",
                            index=False,
                        )
                        st.success(
                            f"✅ Data uploaded to `{pg_table or 'test'}` in PostgreSQL at {pg_host}:{pg_port}/{pg_db}."
                        )
                        st.session_state["upload_to_db_active"] = False
                    except Exception as e:
                        st.error(f"Failed to upload to PostgreSQL: {e}")
