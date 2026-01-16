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
from concurrent.futures import ProcessPoolExecutor
import os

def process_single_anomaly_bucket(b, train_files, test_files):
    train_path = train_files[b]
    test_path  = test_files[b]

    train_data = pd.read_csv(train_path)
    test_data  = pd.read_csv(test_path)

    for c in ["unique_locs", "poi_dict"]:
        train_data[c] = train_data[c].apply(parse_list_col)
        test_data[c]  = test_data[c].apply(parse_list_col)

    # Intersection of agents
    common_agents = np.intersect1d(train_data["agent"].unique(), test_data["agent"].unique())
    train_data = train_data[train_data["agent"].isin(common_agents)]
    test_data  = test_data[test_data["agent"].isin(common_agents)]

    KEYS = ["agent", "day_type", "time_segment"]
    merged = test_data.merge(train_data, on=KEYS, how="left", suffixes=("_test", "_train"))

    # Cleanup list columns
    for c in ["unique_locs_train", "poi_dict_train", "unique_locs_test", "poi_dict_test"]:
        if c in merged.columns:
            merged[c] = merged[c].apply(lambda x: x if isinstance(x, list) else [])

    # Compute scores
    rows = []
    for agent, g in merged.groupby("agent", sort=False):
        # Using .get() or checking for 'new' score logic
        score = score_agent_group(g)
        rows.append({"agent": agent, "anomaly_score": score})

    return pd.DataFrame(rows)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def score_agent_group(g):
    # Fill missing train numeric with 0
    for col in ["unique_location_ids_train", "avg_distance_from_home_km_train", "avg_speed_kmh_train",
                "max_stay_duration_train", "transformations_train", "max_distance_from_home_train"]:
        if col in g.columns:
            g[col] = g[col].fillna(0)

    # numeric components
    score_count = (g["unique_location_ids_test"] - g["unique_location_ids_train"]).abs().to_numpy()
    score_dist  = (g["avg_distance_from_home_km_test"] - g["avg_distance_from_home_km_train"]).abs().to_numpy()
    score_speed = (g["avg_speed_kmh_test"] - g["avg_speed_kmh_train"]).abs().to_numpy()

    max_stay        = (g["max_stay_duration_test"] - g["max_stay_duration_train"]).abs().to_numpy()
    transformations = (g["transformations_test"] - g["transformations_train"]).abs().to_numpy()
    max_distance    = (g["max_distance_from_home_test"] - g["max_distance_from_home_train"]).abs().to_numpy()

    dominent_poi_changed = (g["dominent_poi_test"] != g["dominent_poi_train"]).astype(int).to_numpy()

    # set diffs (small per-agent rows; do python lists)
    tl = g["unique_locs_train"].tolist()
    te = g["unique_locs_test"].tolist()
    score_new_locs = np.array([len(set(b) - set(a)) for a, b in zip(tl, te)], dtype=float)

    pt = g["poi_dict_train"].tolist()
    pe = g["poi_dict_test"].tolist()
    score_pois_locs = np.array([len(set(b) - set(a)) for a, b in zip(pt, pe)], dtype=float)

    # weights 1
    alpha= 0.15248
    beta= -0.0961
    gamma= 0.056233
    delta= 0.01
    a = -0.01
    b = 0.111
    c = 0.6357
    d = 0.043
    e = 0.001

    #weights 2
    # alpha = -0.058183330468428024
    # beta = -0.2213398142336268
    # gamma = 0.22550177075110456
    # delta = -0.7087388288942588
    # a = -0.20564926276067405
    # b = 0.006309287255213067
    # c = 0.5215565484850472
    # d = 0.0463382656166072
    # e = -0.14526639342286488

    total = (
        (alpha * score_count) +
        (beta  * score_dist) +
        (gamma * score_speed) +
        (delta * score_new_locs) +
        (a * max_stay) +
        (b * transformations) +
        (c * max_distance) +
        (d * dominent_poi_changed) +
        (e * score_pois_locs)
    )

    return float(np.sum(total)) if len(total) else 0.0

LIST_COLS = ["unique_locs", "poi_dict"]

def parse_list_col(s):
    if pd.isna(s) or s == 0:
        return []
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, (list, tuple, set)) else []
        except Exception:
            return []
    return []



def main():

    TRAIN_DIR = Path("../../processed/trial5/train_monthly")
    TEST_DIR  = Path("../../processed/trial5/test_monthly")

    OUT_DIR = Path("../../processed/trial5/anomaly_scores")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_ALL = OUT_DIR / "anomaly_scores_all_buckets.csv"

    def bucket_id_from_path(p: Path) -> int:
        return int(p.name.split("agent_bucket=")[1].split(".csv")[0])

    train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.csv")}
    test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.csv")}
    common_buckets = sorted(set(train_files).intersection(test_files))
    max_workers = min(len(common_buckets), os.cpu_count() - 1)
    
    print(f"Parallelizing anomaly scoring for {len(common_buckets)} buckets...")
    
    all_dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the function to buckets
        futures = [executor.submit(process_single_anomaly_bucket, b, train_files, test_files) 
                   for b in common_buckets]
        
        for future in futures:
            result_df = future.result()
            all_dfs.append(result_df)
            print(f"Completed a bucket. Results gathered: {len(all_dfs)}")

    # 2. Combine all bucket results into one final file
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # If an agent appeared in multiple buckets, you might want to aggregate
        # e.g., final_df = final_df.groupby('agent').mean().reset_index()
        
        final_df.to_csv(OUT_ALL, index=False)
        print(f"\nSUCCESS: All scores saved to {OUT_ALL}")

if __name__ == "__main__":
    main()


# TRAIN_DIR = Path("../../processed/trial5/train_monthly")
# TEST_DIR  = Path("../../processed/trial5/test_monthly")

# OUT_DIR = Path("../../processed/trial5/anomaly_scores")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# OUT_ALL = OUT_DIR / "anomaly_scores_all_buckets.csv"

# def bucket_id_from_path(p: Path) -> int:
#     return int(p.name.split("agent_bucket=")[1].split(".csv")[0])

# train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.csv")}
# test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.csv")}
# common_buckets = sorted(set(train_files).intersection(test_files))

# if not common_buckets:
#     raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

# if OUT_ALL.exists():
#     OUT_ALL.unlink()

# for b in common_buckets:
#     train_path = train_files[b]
#     test_path  = test_files[b]

#     print(f"\n=== Processing bucket {b} ===")
#     train_data = pd.read_csv(train_path)
#     test_data  = pd.read_csv(test_path)

#     for c in ["unique_locs", "poi_dict"]:
#         train_data[c] = train_data[c].apply(parse_list_col)
#         test_data[c]  = test_data[c].apply(parse_list_col)

#     # Keep only agents present in both
#     common_agents = np.intersect1d(train_data["agent"].unique(), test_data["agent"].unique())
#     train_data = train_data[train_data["agent"].isin(common_agents)]
#     test_data  = test_data[test_data["agent"].isin(common_agents)]


#     KEYS = ["agent", "day_type", "time_segment"]
#     merged = test_data.merge(train_data, on=KEYS, how="left", suffixes=("_test", "_train"))


#     for c in ["unique_locs_train", "poi_dict_train"]:
#         if c in merged.columns:
#             merged[c] = merged[c].apply(lambda x: x if isinstance(x, list) else [])
#     for c in ["unique_locs_test", "poi_dict_test"]:
#         if c in merged.columns:
#             merged[c] = merged[c].apply(lambda x: x if isinstance(x, list) else [])


#     rows = []
#     for agent, g in merged.groupby("agent", sort=False):
#         rows.append({"agent": agent, "anomaly_score": score_agent_group(g)})

#     new_df = pd.DataFrame(rows).dropna(subset=["new"])

#     if OUT_ALL.exists():
#         existing = pd.read_csv(OUT_ALL)

#         if "new" in existing.columns:
#             existing = existing.drop(columns=["new"])

#         updated = existing.merge(
#             new_df,
#             on="agent",
#             how="left"
#         )

#     else:
        
#         updated = new_df

#     updated.to_csv(OUT_ALL, index=False)

#     print(f"Saved updated results: {OUT_ALL}")
