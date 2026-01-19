import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import os
import ast
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
    # alpha= 0.15248
    # beta= -0.0961
    # gamma= 0.056233
    # delta= 0.01
    # a = -0.01
    # b = 0.111
    # c = 0.6357
    # d = 0.043
    # e = 0.001

    #weights 2
    alpha = 0.12843882159674627
    beta = 0.1277477026006959
    gamma = 0.27669311145381176
    delta = 0.16113361696495052
    a = 0.12838810029558317
    b = 0.3045173365002024
    c = 0.760066526759694
    d = 0.12433512484214806
    e = -0.14401261257898515

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

    return float(np.max(total)) if len(total) else 0.0

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

    TRAIN_DIR = Path("../../processed/trial5/2m/monthly/month1_train")
    TEST_DIR  = Path("../../processed/trial5/2m/monthly/month2_test")

    OUT_DIR = Path("../../processed/trial5/2m/anomaly_scores")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_ALL = OUT_DIR / "1vs2.csv"

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