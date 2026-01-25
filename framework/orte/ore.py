import pandas as pd
import joblib
from pathlib import Path



def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".parquet")[0])


def main():
    TRAIN_DIR = Path("../processed/trial5/sim1/2m/ore/stop_past")
    TEST_DIR  = Path("../processed/trial5/sim1/2m/ore/stop_future")
    OUT_RESULTS_DIR = Path("../processed/trial5/sim1/2m")
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    OUT_CSV = OUT_RESULTS_DIR / "ore.csv"

    train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.parquet")}
    test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.parquet")}
    common_buckets = sorted(set(train_files).intersection(test_files))
    if not common_buckets:
        raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

    bundle = joblib.load("gb_anomaly_bundle.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    write_header = not OUT_CSV.exists()

    for b in common_buckets:
        print(f"\n=== Processing bucket {b} ===")

        test_path = test_files[b]
        test_data = pd.read_parquet(test_path)

        # --- feature prep ---
        X_new = test_data[feature_cols].fillna(0.0)
        X_new = scaler.transform(X_new)

        # --- row-level scores ---
        test_data["score"] = model.predict_proba(X_new)[:, 1]

        # --- agent-level aggregation ---
        agent_scores = (
            test_data
            .groupby("agent", as_index=False)["score"]
            .max()
        )

        agent_scores.to_csv(
            OUT_CSV,
            mode="a",
            index=False,
            header=write_header
        )
        write_header = False

    print(f"\n✅ Saved results to {OUT_CSV}")

if __name__ == "__main__":
    main()