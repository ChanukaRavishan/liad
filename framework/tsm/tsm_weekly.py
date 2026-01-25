import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import math
import os
import ast
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path
from typing import Tuple, List


def append_df(df: pd.DataFrame, path: str):
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)



def score_partition(train_p: pd.DataFrame, test_p: pd.DataFrame) -> pd.DataFrame:
    
    if test_p.empty:
        return pd.DataFrame(columns=["agent", "anomaly_score"])

    # weights pinned as plain floats
    alpha = 0.280890
    beta = 0.146024
    gamma = 0.146024
    delta = 0.203904
    a = 0.024093
    b = 0.331588
    c = 0.933864
    d = 0.928097
    e = -0.638238
    
    KEYS = ["agent", "day_type", "time_segment"]

    test_p = test_p.reset_index(drop=True)
    test_p["test_row_id"] = np.arange(len(test_p), dtype=np.int64)

    pairs = test_p.merge(train_p, on=KEYS, how="inner", suffixes=("_test", "_train"))

    if pairs.empty:
        # no matching train slots -> could return inf or NaN; here: 0 so it doesn’t explode rankings
        out = test_p.groupby("agent", sort=False).size().reset_index()[["agent"]]
        out["anomaly_score"] = 0.0
        return out

    # vector numeric diffs
    score_count = (pairs["unique_location_ids_test"] - pairs["unique_location_ids_train"]).abs().to_numpy()
    score_dist  = (pairs["avg_distance_from_home_km_test"] - pairs["avg_distance_from_home_km_train"]).abs().to_numpy()
    score_speed = (pairs["avg_speed_kmh_test"] - pairs["avg_speed_kmh_train"]).abs().to_numpy()

    max_stay        = (pairs["max_stay_duration_test"] - pairs["max_stay_duration_train"]).abs().to_numpy()
    transformations = (pairs["transformations_test"] - pairs["transformations_train"]).abs().to_numpy()
    max_distance    = (pairs["max_distance_from_home_test"] - pairs["max_distance_from_home_train"]).abs().to_numpy()

    dom_changed = (
        pairs["dominent_poi_test"].astype(str).to_numpy() !=
        pairs["dominent_poi_train"].astype(str).to_numpy()
    ).astype(np.float32)

    # set diffs (tight, but only on this partition’s merged pairs)
    t_loc = pairs["unique_locs_test"].to_list()
    r_loc = pairs["unique_locs_train"].to_list()
    new_locs = np.fromiter(
    (len(set(ast.literal_eval(t)) - set(ast.literal_eval(r))) 
     for t, r in zip(t_loc, r_loc)),
    dtype=np.float32, 
    count=len(pairs))

    t_poi = pairs["poi_dict_test"].to_list()
    r_poi = pairs["poi_dict_train"].to_list()
    new_pois = np.fromiter(
    (len(set(ast.literal_eval(t)) - set(ast.literal_eval(r))) 
     for t, r in zip(t_poi, r_poi)),
    dtype=np.float32, 
    count=len(pairs))

    total = (
        (alpha * score_count) +
        (beta  * score_dist) +
        (gamma * score_speed) +
        (delta * new_locs) +
        (a * max_stay) +
        (b * transformations) +
        (c * max_distance) +
        (d * dom_changed) +
        (e * new_pois)
    ).astype(np.float32)

    pairs["pair_score"] = total

    # min per test row
    min_per_test = pairs.groupby("test_row_id", sort=False)["pair_score"].min()

    # map test_row_id -> agent
    test_agents = test_p.set_index("test_row_id")["agent"]
    min_df = min_per_test.to_frame("min_score").join(test_agents, how="left")

    # max per agent
    out = min_df.groupby("agent", sort=False)["min_score"].sum().reset_index()
    out.rename(columns={"min_score": "anomaly_score"}, inplace=True)

    return out


# def score_partition(train_p: pd.DataFrame, test_p: pd.DataFrame) -> pd.DataFrame:
#     if test_p.empty:
#         return pd.DataFrame(columns=["agent", "anomaly_score"])

#     # Weights
#     alpha, beta, gamma, delta = 0.280890, 0.146024, 0.146024, 0.203904
#     a, b, c, d, e = 0.024093, 0.331588, 0.933864, 0.928097, -0.638238
    
#     KEYS = ["agent", "day_type", "time_segment"]
#     # Metrics to Z-score
#     METRICS = [
#         "unique_location_ids", "avg_distance_from_home_km", 
#         "avg_speed_kmh", "max_stay_duration", "max_distance_from_home", "transformations"
#     ]

#     # 1. Group Training Stats
#     train_stats = train_p.groupby(KEYS)[METRICS].agg(["mean", "std"]).reset_index()
#     train_stats.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in train_stats.columns]

#     # 2. Preparation
#     test_p = test_p.reset_index(drop=True)
#     test_p["test_row_id"] = np.arange(len(test_p), dtype=np.int64)

#     # Left merge to keep test rows that have NO match in training (The "New Routine" Anomaly)
#     scored_pairs = test_p.merge(train_stats, on=KEYS, how="left")
    
#     # Identify rows with no historical match in this specific time/day segment
#     discovery_mask = scored_pairs["unique_location_ids_mean"].isna()

#     # 3. Vectorized Z-Scoring
#     def calculate_z(df, col):
#         # Use a small epsilon to avoid division by zero
#         # clip(0) handles the log1p warning if data was somehow negative
#         val = df[col].clip(lower=0)
#         mu = df[f"{col}_mean"]
#         sigma = df[f"{col}_std"].fillna(0) + 0.001
#         return (val - mu).abs() / sigma

#     z_count = calculate_z(scored_pairs, "unique_location_ids")
#     z_dist  = calculate_z(scored_pairs, "avg_distance_from_home_km")
#     z_speed = calculate_z(scored_pairs, "avg_speed_kmh")
#     z_max_d = calculate_z(scored_pairs, "max_distance_from_home")
#     z_trans = calculate_z(scored_pairs, "transformations")
    
#     # Log-transform for stay duration to handle heavy tails
#     # log1p(|test - mean|) / sigma_log
#     z_stay = (np.log1p(scored_pairs["max_stay_duration"].clip(0)) - 
#               np.log1p(scored_pairs["max_stay_duration_mean"].clip(0))).abs()

#     # 4. Set Differences (Inner Join required to compare specific lists)
#     pairs_inner = test_p.merge(train_p[KEYS + ["unique_locs", "poi_dict", "dominent_poi"]], on=KEYS, how="inner")
    
#     if not pairs_inner.empty:
#         # Vectorized Set Diffs
#         new_locs_arr = np.fromiter(
#             (len(set(ast.literal_eval(t)) - set(ast.literal_eval(r))) 
#              for t, r in zip(pairs_inner["unique_locs_x"], pairs_inner["unique_locs_y"])),
#             dtype=np.float32, count=len(pairs_inner))
        
#         new_pois_arr = np.fromiter(
#             (len(set(ast.literal_eval(t)) - set(ast.literal_eval(r))) 
#              for t, r in zip(pairs_inner["poi_dict_x"], pairs_inner["poi_dict_y"])),
#             dtype=np.float32, count=len(pairs_inner))

#         dom_changed_arr = (pairs_inner["dominent_poi_x"] != pairs_inner["dominent_poi_y"]).astype(np.float32)

#         # Map these back to the main scored_pairs dataframe (min aggregation for best-match)
#         pairs_inner["tmp_score"] = (delta * new_locs_arr) + (e * new_pois_arr) + (d * dom_changed_arr)
#         set_scores = pairs_inner.groupby("test_row_id")["tmp_score"].min()
#         scored_pairs = scored_pairs.join(set_scores, on="test_row_id")
#     else:
#         scored_pairs["tmp_score"] = 0.0

#     # 5. Combine Components
#     # Fillna(2.0) assumes that if we don't know the variance, we assume a moderate deviation
#     final_score = (
#         (alpha * z_count.fillna(2.0)) +
#         (beta  * z_dist.fillna(2.0)) +
#         (gamma * z_speed.fillna(2.0)) +
#         (c * z_max_d.fillna(2.0)) +
#         (b * z_trans.fillna(2.0)) +
#         (a * z_stay.fillna(2.0)) +
#         scored_pairs["tmp_score"].fillna(0.0)
#     )

#     # Apply Discovery Penalty: If the agent has NO records for this time_segment/day_type
#     # we assign it the "Max" potential anomaly for that row.
#     final_score[discovery_mask] = final_score.max() if not final_score.empty else 10.0

#     scored_pairs["anomaly_score"] = final_score

#     # 6. Final Aggregation
#     # Per-row min (best match to history), then per-agent max (strongest signal)
#     out = scored_pairs.groupby("agent", sort=False)["anomaly_score"].max().reset_index()
#     return out





# ---------------- PARTITION DRIVER ----------------

def score_weekly_partitioned(train, test,
                             out_path: str,
                             n_parts: int = 100):
    """
    n_parts=100 means ~1% per shard. Use 200/500 if merge still heavy.
    """

    # partition id
    train["pid"] = (train["agent"].values % n_parts).astype(np.int16)
    test["pid"]  = (test["agent"].values  % n_parts).astype(np.int16)

    # process partitions
    for pid in range(n_parts):
        train_p = train[train["pid"] == pid].drop(columns=["pid"])
        test_p  = test[test["pid"] == pid].drop(columns=["pid"])

        if test_p.empty:
            continue

        print(f"Partition {pid}/{n_parts-1}: train_rows={len(train_p):,} test_rows={len(test_p):,}")

        out = score_partition(train_p, test_p)
        append_df(out, out_path)

        # free per-partition temp
        del train_p, test_p, out
        gc.collect()

    print("Done. Saved:", out_path)



TRAIN_DIR = Path("../../processed/trial5/sim2/10k/whole/scaled_global/train_weekly")
TEST_DIR  = Path("../../processed/trial5/sim2/10k/whole/scaled_global/test_weekly")

OUT_PATH = "../../processed/trial5/sim2/10k/weekly_sum.csv"
TMP_DIR  = Path("../../processed/trial5/sim2/10k/_tmp_weekly_parts")
TMP_DIR.mkdir(parents=True, exist_ok=True)



def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".csv")[0])

train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.csv")}
test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.csv")}
common_buckets = sorted(set(train_files).intersection(test_files))

if not common_buckets:
    raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

def process_bucket(b: int, n_parts: int = 100) -> str:
    train_path = train_files[b]
    test_path  = test_files[b]

    out_tmp = TMP_DIR / f"weekly_bucket={b}.csv"
    if out_tmp.exists():
        out_tmp.unlink()

    print(f"=== Processing bucket {b} ===")
    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    score_weekly_partitioned(train_data, test_data, str(out_tmp), n_parts=n_parts)

    del train_data, test_data
    gc.collect()
    return str(out_tmp)

N_JOBS = 30

tmp_files = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
    delayed(process_bucket)(b, 100) for b in common_buckets
)
if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)

first = True
for f in tmp_files:
    df = pd.read_csv(f)
    df.to_csv(OUT_PATH, mode="a", index=False, header=first)
    first = False

print("Done. Saved:", OUT_PATH)