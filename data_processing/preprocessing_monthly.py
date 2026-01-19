from pathlib import Path
import pandas as pd
from helper_functions import agent_type_filter



TRAIN_DIR = Path("../data/trail5/10k/stop_past")
TEST_DIR  = Path("../data/trail5/10k/stop_future")

# Output structure:
# ../processed/trial5/month1/{stop_past, stop_future}/agent_bucket=*.parquet
# ../processed/trial5/month2/{stop_past, stop_future}/agent_bucket=*.parquet
BASE_OUT = Path("../processed/trial5")
OUT_M1_PAST   = BASE_OUT / "month1" / "stop_past"
OUT_M1_FUTURE = BASE_OUT / "month1" / "stop_future"
OUT_M2_PAST   = BASE_OUT / "month2" / "stop_past"
OUT_M2_FUTURE = BASE_OUT / "month2" / "stop_future"

for d in (OUT_M1_PAST, OUT_M1_FUTURE, OUT_M2_PAST, OUT_M2_FUTURE):
    d.mkdir(parents=True, exist_ok=True)

PREC = 5

def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".parquet")[0])

def split_into_two_halves_by_time(df: pd.DataFrame, time_col: str = "started_at"):
    """
    Split into two equal halves by sorting on time_col.
    If odd number of rows, month1 gets the extra row (ceil(n/2)).
    """
    if df.empty:
        return df.copy(), df.copy()

    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    # Put NaT at the end deterministically
    d = d.sort_values([time_col], kind="mergesort", na_position="last").reset_index(drop=True)

    n = len(d)
    cut = (n + 1) // 2  # month1 gets the extra if odd
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()

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

    train_data = train_data.drop(columns=["location_id"], errors="ignore")
    test_data  = test_data.drop(columns=["location_id"], errors="ignore")

    for df in (train_data, test_data):
        df["lat_q"] = df["latitude"].round(PREC)
        df["lon_q"] = df["longitude"].round(PREC)

    all_coords = pd.concat(
        [train_data[["lat_q", "lon_q"]], test_data[["lat_q", "lon_q"]]],
        ignore_index=True
    ).dropna().drop_duplicates()

    all_coords = all_coords.sort_values(["lat_q", "lon_q"], kind="mergesort").reset_index(drop=True)
    all_coords["location_id"] = all_coords.index.astype(int)

    train_data = train_data.merge(all_coords, on=["lat_q", "lon_q"], how="left")
    test_data  = test_data.merge(all_coords, on=["lat_q", "lon_q"], how="left")

    for df_name, df in (("train", train_data), ("test", test_data)):
        if df["location_id"].isna().any():
            df["location_id"] = df["location_id"].astype("Int64")
        else:
            df["location_id"] = df["location_id"].astype("int")

    train_data = train_data.rename(columns={"category": "poi_category"})
    test_data  = test_data.rename(columns={"category": "poi_category"})

    drop_cols = ["distance_meters", "duration_min", "poi_id", "lat_q", "lon_q"]
    train_data = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns])
    test_data  = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])

    # ---- Split each (train/test) into two time-ordered halves ----
    train_m1, train_m2 = split_into_two_halves_by_time(train_data, time_col="started_at")
    test_m1,  test_m2  = split_into_two_halves_by_time(test_data,  time_col="started_at")

    # ---- Save ----
    out_train_m1 = OUT_M1_PAST   / f"agent_bucket={b}.parquet"
    out_train_m2 = OUT_M2_PAST   / f"agent_bucket={b}.parquet"
    out_test_m1  = OUT_M1_FUTURE / f"agent_bucket={b}.parquet"
    out_test_m2  = OUT_M2_FUTURE / f"agent_bucket={b}.parquet"

    train_m1.to_parquet(out_train_m1, index=False)
    train_m2.to_parquet(out_train_m2, index=False)
    test_m1.to_parquet(out_test_m1, index=False)
    test_m2.to_parquet(out_test_m2, index=False)

    print(f"Saved: {out_train_m1}")
    print(f"Saved: {out_train_m2}")
    print(f"Saved: {out_test_m1}")
    print(f"Saved: {out_test_m2}")