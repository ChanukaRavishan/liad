from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

NUMERIC_COLS = [
    "unique_location_ids",
    "avg_distance_from_home_km",
    "avg_speed_kmh",
    "max_stay_duration",
    "transformations",
    "max_distance_from_home",
]

def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".parquet")[0])

def fit_global_robust_scaler(
    train_files: dict,
    numeric_cols=NUMERIC_COLS,
    fit_on_label0: bool = True,
    sample_per_bucket: int | None = 200_000,   # set None to use all rows (may be huge)
    random_state: int = 42,
) -> RobustScaler:
    """
    Fit ONE RobustScaler on (optionally) normal train rows across ALL buckets.
    Uses optional sampling per bucket to avoid OOM.
    """
    rng = np.random.default_rng(random_state)
    chunks = []

    print('reading all train: memory heavy so wait yeah')

    for b, path in sorted(train_files.items()):
        df = pd.read_parquet(path, columns=(numeric_cols + (["label"] if fit_on_label0 else [])))

        if fit_on_label0 and "label" in df.columns:
            df = df[df["label"] == 0]

        if df.empty:
            continue

        # Optional per-bucket sampling for memory control
        if sample_per_bucket is not None and len(df) > sample_per_bucket:
            idx = rng.choice(len(df), size=sample_per_bucket, replace=False)
            df = df.iloc[idx]

        chunks.append(df[numeric_cols])

        print(f"Fit collect bucket {b}: {len(df):,} rows")

    if not chunks:
        raise RuntimeError("No data collected to fit scaler (train may be empty or no label==0 rows).")

    fit_df = pd.concat(chunks, ignore_index=True)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(fit_df)

    print("\n=== Global scaler fit done ===")
    med = pd.Series(scaler.center_, index=numeric_cols)
    iqr = pd.Series(scaler.scale_, index=numeric_cols)
    print("Median:\n", med.round(4).to_string())
    print("IQR:\n", iqr.round(4).to_string())

    return scaler


def transform_and_write_all(
    scaler: RobustScaler,
    files: dict,
    out_dir: Path,
    numeric_cols=NUMERIC_COLS,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for b, path in sorted(files.items()):
        df = pd.read_parquet(path)
        missing = [c for c in numeric_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        df[numeric_cols] = scaler.transform(df[numeric_cols]).astype("float32")

        out_path = out_dir / path.name
        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path}")



TRAIN_DIR = Path("../../processed/trial5/2m/train_weekly")
TEST_DIR  = Path("../../processed/trial5/2m/test_weekly")

OUT_TRAIN_DIR = Path("../processed/trial5/2m/scaled_global/train_weekly")
OUT_TEST_DIR  = Path("../processed/trial5/2m/scaled_global/test_weekly")

SCALER_PATH = Path("../processed/trial5/2m/scaled_global/robust_scaler.joblib")

train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.parquet")}
test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.parquet")}

common_buckets = sorted(set(train_files).intersection(test_files))
if not common_buckets:
    raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

train_files = {b: train_files[b] for b in common_buckets}
test_files  = {b: test_files[b] for b in common_buckets}

print('global scaling start')

scaler = fit_global_robust_scaler(
    train_files,
    numeric_cols=NUMERIC_COLS,
    fit_on_label0=False,
    sample_per_bucket=None,   # tune this (None = use all rows)
    random_state=42,
)

print('global scaling done')

SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"scaler": scaler, "numeric_cols": NUMERIC_COLS}, SCALER_PATH)
print(f"\nSaved scaler to: {SCALER_PATH}")

transform_and_write_all(scaler, train_files, OUT_TRAIN_DIR, numeric_cols=NUMERIC_COLS)
transform_and_write_all(scaler, test_files,  OUT_TEST_DIR,  numeric_cols=NUMERIC_COLS)

print("\nDone.")
