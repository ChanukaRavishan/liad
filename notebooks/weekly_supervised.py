#!/usr/bin/env python3
import argparse
import ast
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ["agent", "day_type", "time_segment"]

FEATURES = [
    "count_diff",
    "dist_diff",
    "speed_diff",
    "max_stay_diff",
    "transformations_diff",
    "max_distance_diff",
    "dom_changed",
    "new_locs",
    "new_pois",
]


def safe_list_from_literal(x):
    """Parse string like "[1,2,3]" into list; return [] on failure."""
    if pd.isna(x):
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    s = str(x)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return list(v)
        # sometimes dict keys are what you want; treat dict as keys
        if isinstance(v, dict):
            return list(v.keys())
        return []
    except Exception:
        return []


def compute_pair_components(pairs: pd.DataFrame) -> pd.DataFrame:
    """Given merged pairs (test+train) compute component columns."""
    out = pd.DataFrame(index=pairs.index)

    out["count_diff"] = (pairs["unique_location_ids_test"] - pairs["unique_location_ids_train"]).abs().astype("float32")
    out["dist_diff"]  = (pairs["avg_distance_from_home_km_test"] - pairs["avg_distance_from_home_km_train"]).abs().astype("float32")
    out["speed_diff"] = (pairs["avg_speed_kmh_test"] - pairs["avg_speed_kmh_train"]).abs().astype("float32")

    out["max_stay_diff"] = (pairs["max_stay_duration_test"] - pairs["max_stay_duration_train"]).abs().astype("float32")
    out["transformations_diff"] = (pairs["transformations_test"] - pairs["transformations_train"]).abs().astype("float32")
    out["max_distance_diff"] = (pairs["max_distance_from_home_test"] - pairs["max_distance_from_home_train"]).abs().astype("float32")

    # binary mismatch: min over pairs will be 0 if ANY train match has same dom poi
    out["dom_changed"] = (
        pairs["dominent_poi_test"].astype(str).to_numpy() !=
        pairs["dominent_poi_train"].astype(str).to_numpy()
    ).astype("float32")

    # "new_*" computed as set difference sizes
    t_loc = pairs["unique_locs_test"].map(safe_list_from_literal).to_list()
    r_loc = pairs["unique_locs_train"].map(safe_list_from_literal).to_list()
    out["new_locs"] = np.fromiter(
        (len(set(t) - set(r)) for t, r in zip(t_loc, r_loc)),
        dtype=np.float32,
        count=len(pairs)
    )

    t_poi = pairs["poi_dict_test"].map(safe_list_from_literal).to_list()
    r_poi = pairs["poi_dict_train"].map(safe_list_from_literal).to_list()
    out["new_pois"] = np.fromiter(
        (len(set(t) - set(r)) for t, r in zip(t_poi, r_poi)),
        dtype=np.float32,
        count=len(pairs)
    )

    return out


def build_min_component_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each test row, match against NORMAL train rows with same KEYS,
    compute per-component mins across pairs -> features for supervised learning.
    """
    train_norm = train_df[train_df["label"] == 0].copy()
    if train_norm.empty:
        raise ValueError("No normal rows in train (label==0). Cannot build normal reference pool.")

    test = test_df.copy().reset_index(drop=True)
    test["test_row_id"] = np.arange(len(test), dtype=np.int64)

    pairs = test.merge(train_norm, on=KEYS, how="inner", suffixes=("_test", "_train"))

    if pairs.empty:
        # No matches => you cannot learn from "compare to normal in same slot".
        # Returning NaNs so you can decide how to handle (drop or fallback).
        cols = [f"min_{c}" for c in FEATURES]
        out = test[["test_row_id", "label"]].copy()
        for c in cols:
            out[c] = np.nan
        return out

    comps = compute_pair_components(pairs)
    comps["test_row_id"] = pairs["test_row_id"].to_numpy()

    # min per test row for each component
    min_comps = comps.groupby("test_row_id", sort=False)[FEATURES].min()
    min_comps.columns = [f"min_{c}" for c in min_comps.columns]

    out = test[["test_row_id", "label"]].set_index("test_row_id").join(min_comps, how="left").reset_index()
    return out


def fit_logreg_weights(feat_df: pd.DataFrame, cv_folds: int = 5, seed: int = 42):
    feat_cols = [f"min_{c}" for c in FEATURES]
    df = feat_df.dropna(subset=feat_cols + ["label"]).copy()

    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)

    # standardize -> stable training; later we convert back to raw-feature weights
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        n_jobs=None
    )
    model.fit(Xs, y)

    # Quick CV sanity check (optional but useful)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    pr_aucs, roc_aucs = [], []
    for tr, va in skf.split(Xs, y):
        m = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
            class_weight="balanced"
        )
        m.fit(Xs[tr], y[tr])
        p = m.predict_proba(Xs[va])[:, 1]
        pr_aucs.append(average_precision_score(y[va], p))
        roc_aucs.append(roc_auc_score(y[va], p))

    # Convert weights back to RAW scale:
    # score = w_scaled · ((x - mean)/std) + b
    # => score = (w_scaled/std) · x + (b - sum(w_scaled*mean/std))
    w_scaled = model.coef_.ravel()
    b_scaled = float(model.intercept_[0])

    mean_ = scaler.mean_
    std_ = scaler.scale_

    w_raw = w_scaled / std_
    b_raw = b_scaled - float(np.sum((w_scaled * mean_) / std_))

    weights_raw = {feat_cols[i]: float(w_raw[i]) for i in range(len(feat_cols))}
    weights_scaled = {feat_cols[i]: float(w_scaled[i]) for i in range(len(feat_cols))}

    metrics = {
        "cv_pr_auc_mean": float(np.mean(pr_aucs)),
        "cv_pr_auc_std": float(np.std(pr_aucs)),
        "cv_roc_auc_mean": float(np.mean(roc_aucs)),
        "cv_roc_auc_std": float(np.std(roc_aucs)),
    }

    return weights_raw, b_raw, weights_scaled, b_scaled, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train CSV/Parquet")
    ap.add_argument("--test", required=True, help="Path to test CSV/Parquet")
    ap.add_argument("--out_weights", default="learned_weights.json", help="Output JSON file")
    args = ap.parse_args()

    def read_any(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    train_df = read_any(args.train)
    test_df = read_any(args.test)

    # Basic checks
    for col in KEYS + ["label"]:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing required column: {col}")

    feat_df = build_min_component_features(train_df, test_df)

    weights_raw, bias_raw, weights_scaled, bias_scaled, metrics = fit_logreg_weights(feat_df)

    payload = {
        "description": "LogReg learned weights on per-test-row min-components vs normal train rows (label==0).",
        "features": [f"min_{c}" for c in FEATURES],
        "weights_raw": weights_raw,
        "bias_raw": bias_raw,
        "weights_scaled": weights_scaled,
        "bias_scaled": bias_scaled,
        "metrics": metrics,
    }

    with open(args.out_weights, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", args.out_weights)
    print(json.dumps(metrics, indent=2))
    print("\nRAW weights (drop into a linear score on min-components):")
    for k, v in weights_raw.items():
        print(f"  {k:>22s}: {v:+.6f}")
    print(f"  {'bias_raw':>22s}: {bias_raw:+.6f}")


if __name__ == "__main__":
    main()
