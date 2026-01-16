import os
import ast
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def parse_maybe_list(x):
    """
    Safely parse list-like values that may be stored as strings in CSV.
    Returns a Python list.
    Handles: NaN/None, list, tuple, set, np.ndarray, strings like "[1,2]".
    """
    # Fast path for common nulls
    if x is None:
        return []

    # If it's already list-like, return it as a list
    if isinstance(x, list):
        return x
    if isinstance(x, (set, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()

    # Handle scalar NaN (only safe for scalars)
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return []

    # Strings: try literal_eval
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, set, tuple)):
                return list(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                # choose keys (adjust if you want values instead)
                return list(v.keys())
            return []
        except Exception:
            return []

    # For anything else, try a safe pandas scalar-null check
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    return []



def ensure_columns_exist(df, cols, fill_value=np.nan):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def build_anomaly_features(train_profiles: pd.DataFrame, test_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-row anomaly features by comparing test slot vs train slot
    for the SAME agent + day_type + time_segment.

    Prevents many-to-many merges by aggregating to unique keys first.
    """
    keys = ['agent', 'day_type', 'time_segment']

    base_numeric = [
        'unique_location_ids',
        'avg_distance_from_home_km',
        'avg_speed_kmh',
        'max_stay_duration',
        'transformations',
        'max_distance_from_home',
        'label'
    ]
    base_misc = ['unique_locs', 'poi_dict', 'dominent_poi']

    train_profiles = ensure_columns_exist(train_profiles.copy(), base_numeric + base_misc)
    test_profiles  = ensure_columns_exist(test_profiles.copy(),  base_numeric + base_misc)

    # Aggregate to ensure 1 row per key in each split
    agg_num = {
        'unique_location_ids': 'mean',
        'avg_distance_from_home_km': 'mean',
        'avg_speed_kmh': 'mean',
        'max_stay_duration': 'max',
        'transformations': 'mean',
        'max_distance_from_home': 'max',
        'label': 'max'
    }
    # For list-like / categorical fields: take first non-null occurrence
    agg_misc = {
        'unique_locs': 'first',
        'poi_dict': 'first',
        'dominent_poi': 'first'
    }

    train_agg = train_profiles.groupby(keys, as_index=False).agg({**agg_num, **agg_misc})
    test_agg  = test_profiles.groupby(keys, as_index=False).agg({**agg_num, **agg_misc})

    # Merge on agent+slot (THE actual intended join)
    merged = pd.merge(
        test_agg,
        train_agg,
        on=keys,
        suffixes=('_test', '_train'),
        how='left'
    )

    numeric_train_cols = [
        'unique_location_ids_train',
        'avg_distance_from_home_km_train',
        'avg_speed_kmh_train',
        'max_stay_duration_train',
        'transformations_train',
        'max_distance_from_home_train'
    ]
    for c in numeric_train_cols:
        if c not in merged.columns:
            merged[c] = 0.0
    merged[numeric_train_cols] = merged[numeric_train_cols].fillna(0)


    for c in ['unique_locs_train', 'unique_locs_test', 'poi_dict_train', 'poi_dict_test']:
        if c not in merged.columns:
            merged[c] = [[]] * len(merged)
        merged[c] = merged[c].apply(parse_maybe_list)

    merged['f_count_diff'] = (merged['unique_location_ids_test'] - merged['unique_location_ids_train']).abs()
    merged['f_dist_diff']  = (merged['avg_distance_from_home_km_test'] - merged['avg_distance_from_home_km_train']).abs()
    merged['f_speed_diff'] = (merged['avg_speed_kmh_test'] - merged['avg_speed_kmh_train']).abs()

    def get_new_loc_count(row):
        set_train = set(row['unique_locs_train']) if isinstance(row['unique_locs_train'], list) else set()
        set_test  = set(row['unique_locs_test'])  if isinstance(row['unique_locs_test'], list)  else set()
        return len(set_test - set_train)

    merged['f_new_locs'] = merged.apply(get_new_loc_count, axis=1)

    merged['f_max_stay_diff'] = (merged['max_stay_duration_test'] - merged['max_stay_duration_train']).abs()
    merged['f_transforms_diff'] = (merged['transformations_test'] - merged['transformations_train']).abs()
    merged['f_max_dist_diff'] = (merged['max_distance_from_home_test'] - merged['max_distance_from_home_train']).abs()

    if 'dominent_poi_test' not in merged.columns:
        merged['dominent_poi_test'] = np.nan
    if 'dominent_poi_train' not in merged.columns:
        merged['dominent_poi_train'] = np.nan

    merged['f_dom_poi_changed'] = (merged['dominent_poi_test'] != merged['dominent_poi_train']).astype(int)

    def get_new_poi_count(row):
        set_train = set(row['poi_dict_train']) if isinstance(row['poi_dict_train'], list) else set()
        set_test  = set(row['poi_dict_test'])  if isinstance(row['poi_dict_test'], list)  else set()
        return len(set_test - set_train)

    merged['f_new_pois'] = merged.apply(get_new_poi_count, axis=1)

    return merged


def fit_anomaly_weight_model(train_profiles: pd.DataFrame, test_profiles: pd.DataFrame):
    merged = build_anomaly_features(train_profiles, test_profiles)

    feature_cols = [
        'f_count_diff',
        'f_dist_diff',
        'f_speed_diff',
        'f_new_locs',
        'f_max_stay_diff',
        'f_transforms_diff',
        'f_max_dist_diff',
        'f_dom_poi_changed',
        'f_new_pois',
    ]

    X = merged[feature_cols]
    y = merged['label_test']  # 0/1 anomalous row in test

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            n_jobs=1
        ))
    ])

    model.fit(X, y)
    return model, feature_cols


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    train = pd.read_csv('../processed/trial5/train_monthly_subsampled.csv')
    test  = pd.read_csv('../processed/trial5/test_monthly_subsampled.csv')

    for col in ['unique_locs', 'poi_dict']:
        if col in train.columns:
            train[col] = train[col].apply(parse_maybe_list)
        if col in test.columns:
            test[col] = test[col].apply(parse_maybe_list)


    print('fitting the model')
    model, feature_cols = fit_anomaly_weight_model(train, test)

    clf = model.named_steps['clf']
    weights = clf.coef_[0]

    # print weights
    for name, w in zip(feature_cols, weights):
        print(name, w)

    # save weights
    weights_df = pd.DataFrame({
        "feature": feature_cols,
        "weight": weights
    }).sort_values("weight", key=abs, ascending=False)

    out_path = "sim2_evalb_model_weights.csv"
    weights_df.to_csv(out_path, index=False)
    print(f"\nSaved weights to: {out_path}")


if __name__ == "__main__":
    main()
