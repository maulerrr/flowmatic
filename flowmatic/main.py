import os
from flowmatic.ingestion import ingest
from flowmatic.quality_check import quality_report
from flowmatic.cleaning import clean
from tqdm import tqdm

DATA_RAW_DIR   = "../data/raw"
DATA_CLEAN_DIR = "../data/cleaned"

def ensure_dirs():
    os.makedirs(DATA_CLEAN_DIR, exist_ok=True)

def process_file(filepath: str, hf_token=None):
    print(f"\n▶ Ingesting: {filepath}")
    df = ingest(filepath, token=hf_token)
    print("✔ Loaded data:", df.shape)

    print("\n▶ Quality report:")
    qr = quality_report(df)

    print("\n▶ Cleaning data...")
    df_clean = clean(df)
    print("✔ Cleaned data:", df_clean.shape)

    out_path = os.path.join(DATA_CLEAN_DIR, os.path.basename(filepath).replace('.csv','_cleaned.csv'))
    df_clean.to_csv(out_path)
    print("✔ Saved cleaned data to", out_path)

if __name__ == "__main__":
    ensure_dirs()
    # Example sources: local CSVs or HF dataset identifiers
    sources = [
        os.path.join(DATA_RAW_DIR, "Astana_90days_routes30.csv"),
        # or a Hugging Face dataset:
        # "your-username/smart-city-traffic",
    ]
    for src in tqdm(sources):
        process_file(src, hf_token=os.getenv("HF_TOKEN"))
