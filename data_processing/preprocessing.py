from pathlib import Path
import pandas as pd

def agent_type_filter(train_df: pd.DataFrame):
    """
    Compute total duration per agent on the train month and keep agents
    above the 16th percentile (this needs to be customized depending on the data).
    Works on an in-memory DataFrame (no extra csv read).
    """
    tmp = train_df[['agent', 'started_at', 'finished_at']].copy()

    tmp['started_at'] = pd.to_datetime(tmp['started_at'])
    tmp['finished_at'] = pd.to_datetime(tmp['finished_at'])

    tmp['duration_min'] = (tmp['finished_at'] - tmp['started_at']).dt.total_seconds() / 60.0
    tmp['duration'] = tmp['duration_min'].clip(lower=0).fillna(0)

    train_agent_dur = tmp.groupby('agent')['duration'].sum()

    df = pd.DataFrame({'train_duration': train_agent_dur}).fillna(0)

    q1_value = df['train_duration'].quantile(0.16)
    df_top_q1 = df[df['train_duration'] >= q1_value]

    return df_top_q1.index

TRAIN_DIR = Path("../data/trail5/stop_past")
TEST_DIR  = Path("../data/trail5/stop_future")

OUT_TRAIN_DIR = Path("../processed/trial5/stop_past")
OUT_TEST_DIR  = Path("../processed/trial5/stop_future")

OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

PREC = 5  # keep this, otherwise float equality is a lie

def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".parquet")[0])

train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.parquet")}
test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.parquet")}
common_buckets = sorted(set(train_files).intersection(test_files))

if not common_buckets:
    raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

for b in common_buckets:
    train_path = train_files[b]
    test_path  = test_files[b]

    print(f"\n=== Processing bucket {b} ===")
    train_data = pd.read_parquet(train_path)
    test_data  = pd.read_parquet(test_path)

    # ---- your resident + duration filters (kept as-is) ----
    residents = agent_type_filter(train_data)
    train_data = train_data[train_data["agent"].isin(residents)].copy()
    test_data  = test_data[test_data["agent"].isin(residents)].copy()

    train_data["duration_min"] = (
        pd.to_datetime(train_data["finished_at"]) - pd.to_datetime(train_data["started_at"])
    ).dt.total_seconds() / 60.0
    test_data["duration_min"] = (
        pd.to_datetime(test_data["finished_at"]) - pd.to_datetime(test_data["started_at"])
    ).dt.total_seconds() / 60.0

    train_data = train_data[train_data["duration_min"] > 15].copy()
    test_data  = test_data[test_data["duration_min"] > 15].copy()

    # ---- Drop old IDs (the contaminated column) ----
    train_data = train_data.drop(columns=["location_id"], errors="ignore")
    test_data  = test_data.drop(columns=["location_id"], errors="ignore")

    # ---- Quantize coords for stable matching ----
    for df in (train_data, test_data):
        df["lat_q"] = df["latitude"].round(PREC)
        df["lon_q"] = df["longitude"].round(PREC)

    # ---- Build a GLOBAL mapping across train+test (for this bucket) ----
    all_coords = pd.concat(
        [
            train_data[["lat_q", "lon_q"]],
            test_data[["lat_q", "lon_q"]],
        ],
        ignore_index=True
    ).dropna().drop_duplicates()

    # Deterministic ID assignment: sort coords, then enumerate
    all_coords = all_coords.sort_values(["lat_q", "lon_q"], kind="mergesort").reset_index(drop=True)
    all_coords["location_id"] = all_coords.index.astype(int)

    # ---- Merge mapping back into train/test ----
    train_data = train_data.merge(all_coords, on=["lat_q", "lon_q"], how="left")
    test_data  = test_data.merge(all_coords, on=["lat_q", "lon_q"], how="left")

    # Sanity: location_id should never be missing unless lat/lon missing
    # If you want to drop missing coords entirely:
    # train_data = train_data.dropna(subset=["location_id"]).copy()
    # test_data  = test_data.dropna(subset=["location_id"]).copy()

    train_data["location_id"] = train_data["location_id"].astype("int")
    test_data["location_id"]  = test_data["location_id"].astype("int")

    # ---- Rename / drop ----
    train_data = train_data.rename(columns={"category": "poi_category"})
    test_data  = test_data.rename(columns={"category": "poi_category"})

    drop_cols = ["distance_meters", "duration_min", "poi_id", "lat_q", "lon_q"]
    train_data = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns])
    test_data  = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])

    # ---- Save ----
    out_train_path = OUT_TRAIN_DIR / f"agent_bucket={b}.parquet"
    out_test_path  = OUT_TEST_DIR  / f"agent_bucket={b}.parquet"

    train_data.to_parquet(out_train_path, index=False)
    test_data.to_parquet(out_test_path, index=False)

    print(f"Saved: {out_train_path}")
    print(f"Saved: {out_test_path}")