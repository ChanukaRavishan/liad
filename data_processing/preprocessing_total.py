from pathlib import Path
import pandas as pd
from helper_functions import agent_type_filter

#uncomment the agent filter later
TRAIN_DIR = Path("../data/trail5/sim2/10k/stop_past")
TEST_DIR  = Path("../data/trail5/sim2/10k/stop_future")

OUT_TRAIN_DIR = Path("../processed/trial5/sim2/10k/whole/stop_past")
OUT_TEST_DIR  = Path("../processed/trial5/sim2/10k/whole/stop_future")

OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

PREC = 5

def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket_")[1].split(".parquet")[0])

train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket_*.parquet")}
test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket_*.parquet")}
common_buckets = sorted(set(train_files).intersection(test_files))

if not common_buckets:
    raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

for b in common_buckets:
    train_path = train_files[b]
    test_path  = test_files[b]

    print(f"\n=== Processing bucket {b} ===")
    train_data = pd.read_parquet(train_path)
    train_data.rename(columns={"user_id": "agent", "category": "poi_category"}, inplace=True)
    
    test_data  = pd.read_parquet(test_path)
    test_data.rename(columns={"user_id": "agent", "category": "poi_category"}, inplace=True)

    # residents = agent_type_filter(train_data)
    # train_data = train_data[train_data["agent"].isin(residents)].copy()
    # test_data  = test_data[test_data["agent"].isin(residents)].copy()

    train_data["duration_min"] = (
        pd.to_datetime(train_data["finished_at"]) - pd.to_datetime(train_data["started_at"])
    ).dt.total_seconds() / 60.0
    test_data["duration_min"] = (
        pd.to_datetime(test_data["finished_at"]) - pd.to_datetime(test_data["started_at"])
    ).dt.total_seconds() / 60.0

    train_data = train_data[train_data["duration_min"] > 15].copy()
    test_data  = test_data[test_data["duration_min"] > 15].copy()

    train_data = train_data.drop(columns=["location_id"], errors="ignore")
    test_data  = test_data.drop(columns=["location_id"], errors="ignore")

    for df in (train_data, test_data):
        df["lat_q"] = df["latitude"].round(PREC)
        df["lon_q"] = df["longitude"].round(PREC)

    all_coords = pd.concat(
        [
            train_data[["lat_q", "lon_q"]],
            test_data[["lat_q", "lon_q"]],
        ],
        ignore_index=True
    ).dropna().drop_duplicates()

    all_coords = all_coords.sort_values(["lat_q", "lon_q"], kind="mergesort").reset_index(drop=True)
    all_coords["location_id"] = all_coords.index.astype(int)

    train_data = train_data.merge(all_coords, on=["lat_q", "lon_q"], how="left")
    test_data  = test_data.merge(all_coords, on=["lat_q", "lon_q"], how="left")

    train_data["location_id"] = train_data["location_id"].astype("int")
    test_data["location_id"]  = test_data["location_id"].astype("int")

    drop_cols = ["distance_meters", "duration_min", "poi_id", "lat_q", "lon_q"]
    train_data = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns])
    test_data  = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])

    out_train_path = OUT_TRAIN_DIR / f"agent_bucket={b}.parquet"
    out_test_path  = OUT_TEST_DIR  / f"agent_bucket={b}.parquet"

    train_data.to_parquet(out_train_path, index=False)
    test_data.to_parquet(out_test_path, index=False)

    print(f"Saved: {out_train_path}")
    print(f"Saved: {out_test_path}")