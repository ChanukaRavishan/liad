import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import math
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_anomaly_features(train_profiles, test_profiles):
    # Merge train/test profiles (don't filter by agent here; keep everything)
    merged = pd.merge(
        test_profiles,
        train_profiles,
        on=['day_type', 'time_segment'],
        suffixes=('_test', '_train'),
        how='left'
    )

    # Fill numeric training columns when no history exists
    numeric_cols = [
        'unique_location_ids_train',
        'avg_distance_from_home_km_train',
        'avg_speed_kmh_train',
        'max_stay_duration_train',
        'transformations_train',
        'max_distance_from_home_train'
    ]
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    # Component 1: Count difference
    merged['f_count_diff'] = (merged['unique_location_ids_test'] -
                              merged['unique_location_ids_train']).abs()

    # Component 2: Distance difference
    merged['f_dist_diff'] = (merged['avg_distance_from_home_km_test'] -
                             merged['avg_distance_from_home_km_train']).abs()

    # Component 3: Speed difference
    merged['f_speed_diff'] = (merged['avg_speed_kmh_test'] -
                              merged['avg_speed_kmh_train']).abs()

    # Component 4: New locations
    def get_new_loc_count(row):
        locs_train = row['unique_locs_train']
        locs_test = row['unique_locs_test']
        set_train = set(locs_train) if isinstance(locs_train, list) else set()
        set_test = set(locs_test) if isinstance(locs_test, list) else set()
        return len(set_test - set_train)

    merged['f_new_locs'] = merged.apply(get_new_loc_count, axis=1)

    # Component 5: max stay duration
    merged['f_max_stay_diff'] = (
        merged['max_stay_duration_test'] -
        merged['max_stay_duration_train']
    ).abs()

    # Component 6: number of transformations
    merged['f_transforms_diff'] = (
        merged['transformations_test'] -
        merged['transformations_train']
    ).abs()

    # Component 7: max distance from home
    merged['f_max_dist_diff'] = (
        merged['max_distance_from_home_test'] -
        merged['max_distance_from_home_train']
    ).abs()

    # Component 8: dominant poi changed
    merged['f_dom_poi_changed'] = (
        merged['dominent_poi_test'] != merged['dominent_poi_train']
    ).astype(int)

    # Component 9: new POI categories
    def get_new_poi_count(row):
        pois_train = row['poi_dict_train']
        pois_test = row['poi_dict_test']
        set_train = set(pois_train) if isinstance(pois_train, list) else set()
        set_test = set(pois_test) if isinstance(pois_test, list) else set()
        return len(set_test - set_train)

    merged['f_new_pois'] = merged.apply(get_new_poi_count, axis=1)

    return merged

def fit_anomaly_weight_model(train_profiles, test_profiles):
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
    y = merged['label_test']  # 0/1 anomalous row

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',  # you likely have few anomalies
            max_iter=1000
        ))
    ])

    model.fit(X, y)
    return model, feature_cols



os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

train = pd.read_csv('../processed/train_monthly.csv')
test = pd.read_csv('../processed/test_monthly.csv')

gt = pd.read_csv('../processed/anomalous_segmented.csv')
residents = pd.read_csv('../processed/residents.csv')
train = train[train.agent.isin(residents.agent.unique())]

train['label'] = 0
test['label'] = 0

gt_agents = set(gt['agent'].unique())
train_agents = set(train['agent'].unique())
normal_agents = np.array(list(train_agents - gt_agents))

print("GT agents:", len(gt_agents))
print("Available normal agents:", len(normal_agents))
np.random.seed(42)
sampled_normals = np.random.choice(normal_agents, size=100000, replace=False)

train = pd.concat([
    train[train['agent'].isin(gt_agents)],          # anomalous agents
    train[train['agent'].isin(sampled_normals)]     # clean agents
]).reset_index(drop=True)

test = test[test.agent.isin(train.agent.unique())]


for agent, gt_agent in gt.groupby('agent'):
    agent_mask = test['agent'] == agent

    if not agent_mask.any():
        continue

    for _, row in gt_agent.iterrows():
        anomaly_time_segment = row['time_segment']
        anomaly_day_type = row['day_type']

        overlap_mask = (
            agent_mask &
            (test['day_type'] == anomaly_day_type) &
            (test['time_segment'] == anomaly_time_segment)
        )

        test.loc[overlap_mask, 'label'] = 1


def fit_anomaly_weight_model(train_profiles, test_profiles):
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
    y = merged['label_test']  # 0/1 anomalous row

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',  # you likely have few anomalies
            max_iter=1000
        ))
    ])

    model.fit(X, y)
    return model, feature_cols


model, feature_cols = fit_anomaly_weight_model(train, test)
scaler = model.named_steps['scaler']
clf = model.named_steps['clf']

weights = clf.coef_[0]
for name, w in zip(feature_cols, weights):
    print(name, w)



weights_df = pd.DataFrame({
    "feature": feature_cols,
    "weight": weights
})

weights_df = weights_df.sort_values("weight", key=abs, ascending=False)

weights_df.to_csv("sim2_evalb_model_weights.csv", index=False)