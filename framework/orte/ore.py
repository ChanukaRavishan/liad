import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import ast
from typing import Optional, Dict, Tuple


def _to_locset(x):
    """
    Convert transformation / transformation_str into a Python set of tokens.
    Handles:
      - list of ints/strings
      - stringified list like "[1,2,3]"
      - comma-separated strings "1,2,3"
      - None/NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    if isinstance(x, (list, tuple)):
        return set(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return set()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    return set(v)
            except Exception:
                pass

        if "," in s:
            return set(t.strip() for t in s.split(",") if t.strip())
        return {s}

    return set([str(x)])


def score_max_outlier_by_poi(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    use_transformation_jaccard: bool = True,
    weights: Optional[Dict[str, float]] = None,
    return_row_level: bool = False,
) -> pd.DataFrame:
    """
    For each test row, find the most similar train row among SAME agent + SAME dominant_poi.
    Score(test_row) = min distance across those candidates.
    Agent score = max Score(test_row) across that agent's test rows.

    If no train rows exist for (agent, dominant_poi), fallback to all train rows for that agent.

    Returns:
      - by default: DataFrame [agent, anomaly_score]
      - if return_row_level: DataFrame with per-test-row min_score + chosen match source
    """

    if weights is None:
        weights = {
            "duration_min": 0.9954,
            "max_distance_from_home": 1.0792,
            "avg_speed_first_half": 0.0466,
            "avg_speed_second_half": 0.3712,
            # optional transformation dissimilarity
            "trans_jaccard": -0.3132,  # this is added ON TOP
        }

    train = train_df.copy()
    test = test_df.copy()

    for col in ["agent", "dominant_poi"]:
        if col not in train.columns or col not in test.columns:
            raise ValueError(f"Missing required column: {col}")

    train["dominant_poi"] = train["dominant_poi"].astype(str)
    test["dominant_poi"] = test["dominant_poi"].astype(str)

    test = test.reset_index(drop=True)
    test["test_row_id"] = np.arange(len(test), dtype=np.int64)

    numeric_cols = [
        c for c in ["duration_min", "max_distance_from_home", "avg_speed_first_half", "avg_speed_second_half"]
        if c in train.columns and c in test.columns
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found in both train and test.")

    pairs_poi = test.merge(
        train,
        on=["agent", "dominant_poi"],
        how="inner",
        suffixes=("_test", "_train"),
    )

    matched_ids = set(pairs_poi["test_row_id"].unique()) if not pairs_poi.empty else set()
    missing_test = test[~test["test_row_id"].isin(matched_ids)]

    pairs_agent = pd.DataFrame()
    if not missing_test.empty:
        pairs_agent = missing_test.merge(
            train,
            on=["agent"],
            how="inner",
            suffixes=("_test", "_train"),
        )
    
        if not pairs_agent.empty:
            pairs_agent["match_source"] = "agent_fallback"

    if not pairs_poi.empty:
        pairs_poi["match_source"] = "poi_match"

    # Combine
    if pairs_poi.empty and pairs_agent.empty:
        # no train overlap at all (weird)
        out = test[["agent"]].drop_duplicates().copy()
        out["anomaly_score"] = np.nan
        return out

    pairs = pd.concat([pairs_poi, pairs_agent], ignore_index=True)

    # Numeric distance (weighted L1)
    total = np.zeros(len(pairs), dtype=np.float32)
    for c in numeric_cols:
        w = float(weights.get(c, 0.0))
        if w == 0.0:
            continue
        a = pd.to_numeric(pairs[f"{c}_test"], errors="coerce").to_numpy(dtype=np.float32)
        b = pd.to_numeric(pairs[f"{c}_train"], errors="coerce").to_numpy(dtype=np.float32)
        diff = np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0))
        total += (w * diff).astype(np.float32)

    # Optional transformation similarity penalty via (1 - Jaccard)
    # NOTE: This part is Python-loop-ish, but only over merged candidate pairs.
    # If your merge explodes (huge pairs), you must cap candidates (see note below).
    if use_transformation_jaccard:
        # pick a transformation column
        tcol = "transformation"
        if tcol not in pairs.columns:
            tcol = "transformation_str" if "transformation_str" in pairs.columns else None

        if tcol is not None:
            wj = float(weights.get("trans_jaccard", 0.0))
            if wj != 0.0:
                t_test = pairs[f"{tcol}_test"].tolist()
                t_train = pairs[f"{tcol}_train"].tolist()

                jac_pen = np.empty(len(pairs), dtype=np.float32)
                for i, (xt, xr) in enumerate(zip(t_test, t_train)):
                    st = _to_locset(xt)
                    sr = _to_locset(xr)
                    if not st and not sr:
                        jac_pen[i] = 0.0
                        continue
                    inter = len(st & sr)
                    uni = len(st | sr)
                    jac = inter / uni if uni else 0.0
                    jac_pen[i] = 1.0 - jac  # penalty: higher = more different

                total += (wj * jac_pen).astype(np.float32)

    pairs["pair_score"] = total

    # For each test row: keep min (most similar train row)
    min_per_test = pairs.groupby("test_row_id", sort=False)["pair_score"].min()

    # If you want row-level details, return those
    if return_row_level:
        row_level = (
            min_per_test.to_frame("min_score")
            .join(test.set_index("test_row_id")[["agent", "dominant_poi"]], how="left")
            .reset_index()
        )
        # Agent max outlier too
        agent_level = row_level.groupby("agent", sort=False)["min_score"].max().reset_index()
        agent_level.rename(columns={"min_score": "anomaly_score"}, inplace=True)
        return row_level, agent_level

    # Agent score = max outlier across its test rows
    tmp = (
        min_per_test.to_frame("min_score")
        .join(test.set_index("test_row_id")["agent"], how="left")
    )
    out = tmp.groupby("agent", sort=False)["min_score"].max().reset_index()
    out.rename(columns={"min_score": "anomaly_score"}, inplace=True)
    return out







def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".csv")[0])


def main():
    TRAIN_DIR = Path("../../processed/trial5/sim1/2m/ore/stop_past")
    TEST_DIR  = Path("../../processed/trial5/sim1/2m/ore/stop_future")
    OUT_RESULTS_DIR = Path("../../processed/trial5/sim1/2m")
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    OUT_CSV = OUT_RESULTS_DIR / "ore_comparison.csv"

    train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.csv")}
    test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.csv")}
    common_buckets = sorted(set(train_files).intersection(test_files))
    if not common_buckets:
        raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

    write_header = not OUT_CSV.exists()

    for b in common_buckets:
        print(f"\n=== Processing bucket {b} ===")

        train_path = train_files[b]
        train_data = pd.read_csv(train_path)

        test_path = test_files[b]
        test_data = pd.read_csv(test_path)

        scores = score_max_outlier_by_poi(train_df=train_data, test_df=test_data,)

        scores.to_csv(
            OUT_CSV,
            mode="a",
            index=False,
            header=write_header
        )
        write_header = False

    print(f"\n✅ Saved results to {OUT_CSV}")

if __name__ == "__main__":
    main()