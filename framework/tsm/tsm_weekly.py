import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import math
import os
import ast
import gc
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path

def to_fset(x):
    return frozenset(x) if isinstance(x, list) else frozenset()

def append_df(df: pd.DataFrame, path: str):
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)

def load_weekly_csv(file):
    """
    Load weekly profile CSV with memory-friendly dtypes.
    Adjust dtypes if your columns differ.
    """
    dtypes = {
        "agent": "int64",
        "day_type": "category",
        "time_segment": "category",
        "dominent_poi": "category",
        # chunk (week index) might exist:
        "chunk": "int16",
    }

    # numeric columns we need
    num_cols = [
        "unique_location_ids", "avg_distance_from_home_km", "avg_speed_kmh",
        "max_stay_duration", "transformations", "max_distance_from_home",
    ]

    df = file.copy()

    # cast / clean
    if "agent" in df.columns:
        df["agent"] = pd.to_numeric(df["agent"], errors="coerce").fillna(-1).astype("int64")

    for c, t in dtypes.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(t)
            except Exception:
                pass

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")


    df["unique_locs_set"] = df["unique_locs"].apply(to_fset)
    df["poi_dict_set"] = df["poi_dict"].apply(to_fset)

    # keep only needed columns to cut RAM
    keep = ["agent", "day_type", "time_segment", "dominent_poi",
            "unique_locs_set", "poi_dict_set"] + num_cols
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    return df


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
    alpha = 0.12843882159674627
    beta = 0.1277477026006959
    gamma = 0.27669311145381176
    delta = 0.16113361696495052
    a = 0.12838810029558317
    b = 0.3045173365002024
    c = 0.760066526759694
    d = 0.12433512484214806
    e = -0.14401261257898515

    KEYS = ["agent", "day_type", "time_segment"]
    NUM = ["unique_location_ids","avg_distance_from_home_km","avg_speed_kmh",
           "max_stay_duration","transformations","max_distance_from_home"]

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
    t_loc = pairs["unique_locs_set_test"].to_list()
    r_loc = pairs["unique_locs_set_train"].to_list()
    new_locs = np.fromiter((len(t - r) for t, r in zip(t_loc, r_loc)),
                           dtype=np.float32, count=len(pairs))

    t_poi = pairs["poi_dict_set_test"].to_list()
    r_poi = pairs["poi_dict_set_train"].to_list()
    new_pois = np.fromiter((len(t - r) for t, r in zip(t_poi, r_poi)),
                           dtype=np.float32, count=len(pairs))

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
    out = min_df.groupby("agent", sort=False)["min_score"].max().reset_index()
    out.rename(columns={"min_score": "anomaly_score"}, inplace=True)

    return out


# ---------------- PARTITION DRIVER ----------------

def score_weekly_partitioned(train, test,
                             out_path: str,
                             n_parts: int = 200):
    """
    n_parts=100 means ~1% per shard. Use 200/500 if merge still heavy.
    """
    # remove old output (we append)
    if os.path.exists(out_path):
        os.remove(out_path)

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




train = pd.read_csv('')
test = pd.read_csv('')

train = load_weekly_csv(train)
test  = load_weekly_csv(test)

score_weekly_partitioned(train, test, '../../processed/trial5/10k/anomaly_scores/weekly/weekly.csv')