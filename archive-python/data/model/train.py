import argparse
import os
import uuid

import pandas as pd

from flowmatic.models.anomaly import train_isolation_forest, save_model


def main():
	parser = argparse.ArgumentParser(description="Train anomaly detector on a CSV or JSON time-series.")
	parser.add_argument("--path", required=True, help="Path to cleaned CSV/JSON")
	parser.add_argument("--outdir", default="models", help="Directory to save the model")
	parser.add_argument("--name", default=None, help="Model name file stem (optional)")
	parser.add_argument("--contamination", type=float, default=0.01)
	args = parser.parse_args()

	ext = os.path.splitext(args.path)[1].lower()
	if ext == ".csv":
		df = pd.read_csv(args.path, index_col=0, parse_dates=True)
	elif ext == ".json":
		df = pd.read_json(args.path)
		dt_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
		if not dt_col:
			raise SystemExit("No datetime-like column found in JSON")
		df[dt_col] = pd.to_datetime(df[dt_col], errors="raise")
		df = df.set_index(dt_col)
	else:
		raise SystemExit(f"Unsupported extension: {ext}")

	model, metrics = train_isolation_forest(df, contamination=args.contamination)
	stem = args.name or f"isoforest_{uuid.uuid4().hex[:8]}"
	out_path = os.path.join(args.outdir, f"{stem}.joblib")
	save_model(model, out_path)
	print({"path": out_path, "metrics": metrics})


if __name__ == "__main__":
	main()

