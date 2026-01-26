import numpy as np
import pandas as pd
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
import re
from collections import Counter
from scipy.stats import zscore
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.ensemble import IsolationForest
from collections import defaultdict, Counter
import hashlib


def add_features(df):
    df = df.copy()
    
    df['start_minute'] = df['started_at'].dt.hour * 60 + df['started_at'].dt.minute
    
    def _len_trans(x):
        if isinstance(x, (list, tuple)):
            return len(x)
        try:
            return len(eval(x))
        except:
            return np.nan
    
    df['len_trans'] = df['transformation'].apply(_len_trans)
    df['speed_imbalance'] = (df['avg_speed_first_half'] - df['avg_speed_second_half']).abs()
    
    return df


def supervised_anomaly_scoring(test_df):
    test = add_features(test_df)

    test = pd.get_dummies(test, columns=['dominant_poi'], prefix='poi')
    
    # Dynamically find the new POI columns
    poi_cols = [c for c in test.columns if c.startswith('poi_')]
    
    base_features = [
        'duration_min', 'start_minute', 'len_trans',
        'max_distance_from_home', 'avg_speed_first_half',
        'avg_speed_second_half', 'speed_imbalance'
    ]

    feature_cols = base_features + poi_cols

    X = test[feature_cols].fillna(0.0)
    y = test['label'].values
    groups = test['agent'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=5)

    preds = np.zeros(len(test))
    model = GradientBoostingClassifier()

    model.fit(X, y)

    preds[:] = model.predict_proba(X)[:, 1]

    test['anomaly_score_row'] = preds

    ap = average_precision_score(y, preds)
    print(f"Out-of-fold Average Precision = {ap:.4f}")

    agent_scores = (
        test.groupby('agent')['anomaly_score_row']
            .max()
            .reset_index(name='agent_anomaly_score')
    )

    return agent_scores, test, model, scaler, feature_cols







gt = pd.read_csv('../../processed/trial5/sim1/gt/anomalous_temporal.csv')

print('loading gt')

TRAIN_DIR = Path("../../processed/trial5/sim1/2m/ore/stop_past")
TEST_DIR  = Path("../../processed/trial5/sim1/2m/ore/stop_future")


def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".csv")[0])

train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.csv")}
test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.csv")}
common_buckets = sorted(set(train_files).intersection(test_files))


all_test_list = []

for b in common_buckets:
    
    test_path  = test_files[b]

    print(f"=== Processing bucket {b} ===")
    
    test_data  = pd.read_csv(test_path)
    
    all_test_list.append(test_data)

test = pd.concat(all_test_list, ignore_index=True)

test['started_at']  = pd.to_datetime(test['started_at'])
test['finished_at'] = pd.to_datetime(test['finished_at'])

gt['started_at'] = pd.to_datetime(gt['started_at']) 
gt['started_at'] = gt['started_at'].dt.tz_convert('Asia/Tokyo')
gt['finished_at'] = pd.to_datetime(gt['finished_at']) 
gt['finished_at'] = gt['finished_at'].dt.tz_convert('Asia/Tokyo')

test['label'] = 0

print('processing label')
for agent, gt_agent in gt.groupby('agent'):
    agent_mask = test['agent'] == agent

    if not agent_mask.any():
        continue

    for _, row in gt_agent.iterrows():
        anomaly_start_time = row['started_at']
        anomaly_end_time   = row['finished_at']

        overlap_mask = (
            agent_mask &
            (test['started_at'] < anomaly_end_time) &
            (test['finished_at'] > anomaly_start_time)
        )

        test.loc[overlap_mask, 'label'] = 1


print('fitting the model')
agent_scores, test_scored, model, scaler, feature_cols = supervised_anomaly_scoring(test)

print('model fitting done')

joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
    },
    "gb_anomaly_bundle.joblib"
)

agent_scores.to_csv('../../processed/trial5/sim1/2m/ore.csv', index=False)