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
    """
    Compute anomaly score for agents present in this partition.
    Rule:
      For each test row, compare to all train rows with same (agent, day_type, time_segment),
      keep min score; then agent score = max over its test rows of these mins.
    """
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



TRAIN_DIR = Path("../../processed/trial5/2m/scaled_global/train_weekly")
TEST_DIR  = Path("../../processed/trial5/2m/scaled_global/test_weekly")

OUT_PATH = "../../processed/trial5/2m/weekly.csv"
TMP_DIR  = Path("../../processed/trial5/2m/_tmp_weekly_parts")
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

# ----- run buckets in parallel -----
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