from pathlib import Path
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import math
from argparse import ArgumentParser
import os
from concurrent.futures import ProcessPoolExecutor
import functools
import math
from pathlib import Path


def agent_type_filter(train_df: pd.DataFrame):
    """
    Compute total duration per agent on the train month and keep agents
    above the 16th percentile (this needs to be customized depending on the data).
    Works on an in-memory DataFrame (no extra csv read).
    """
    tmp = train_df[['agent', 'started_at', 'finished_at']].copy()

    tmp['started_at'] = pd.to_datetime(tmp['started_at'])
    tmp['finished_at'] = pd.to_datetime(tmp['finished_at'])

    tmp['duration_min'] = (tmp['finished_at'] - tmp['started_at']).dt.total_seconds() / 60.0
    tmp['duration'] = tmp['duration_min'].clip(lower=0).fillna(0)

    train_agent_dur = tmp.groupby('agent')['duration'].sum()

    df = pd.DataFrame({'train_duration': train_agent_dur}).fillna(0)

    q1_value = df['train_duration'].quantile(0.16)
    df_top_q1 = df[df['train_duration'] >= q1_value]

    return df_top_q1.index

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in km between (lat1, lon1) and (lat2, lon2).
    All inputs are pandas Series / numpy arrays.
    """
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def assign_time_segment(dt):
    """
    Assign time segment based on hour and minute.
    Segments: 0-5.59, 6-8.59, 9-13.59, 14-17.29, 17.30-21.29, 21.30-23.59
    """
    hour = dt.hour
    minute = dt.minute

    if hour < 6:
        return '0-5.59'
    elif hour < 9:
        return '6-8.59'
    elif hour < 14:
        return '9-13.59'
    elif hour < 17 or (hour == 17 and minute < 30):
        return '14-17.29'
    elif hour < 21 or (hour == 21 and minute < 30):
        return '17.30-21.29'
    else:
        return '21.30-23.59'


def split_by_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split staypoints that span multiple days and then slice by fixed time bins.
    This is still heavy, but we avoid pointless extra copies.
    """
    df = df.copy()

    df['num_days'] = (df['finished_at'].dt.normalize() - df['started_at'].dt.normalize()).dt.days + 1

    df_exploded = df.loc[df.index.repeat(df['num_days'])].copy()
    df_exploded['day_offset'] = df_exploded.groupby(level=0).cumcount()

    df_exploded['current_day_midnight'] = (
        df_exploded['started_at'].dt.normalize() +
        pd.to_timedelta(df_exploded['day_offset'], unit='D')
    )

    df_exploded['started_at'] = df_exploded[['started_at', 'current_day_midnight']].max(axis=1)
    df_exploded['day_end_boundary'] = df_exploded['current_day_midnight'] + pd.to_timedelta(1, unit='D')
    df_exploded['finished_at'] = df_exploded[['finished_at', 'day_end_boundary']].min(axis=1)

    df_daily = df_exploded[df_exploded['started_at'] < df_exploded['finished_at']].reset_index(drop=True)

    time_bins = [
        ('00:00:00', '06:00:00', 'Early Morning'),
        ('06:00:00', '09:00:00', 'Morning Rush'),
        ('09:00:00', '14:00:00', 'Mid Day'),
        ('14:00:00', '17:30:00', 'Afternoon'),
        ('17:30:00', '21:30:00', 'Evening'),
        ('21:30:00', '1 day', 'Night')
    ]

    final_segments = []

    for start_str, end_str, label in time_bins:
        temp_df = df_daily.copy()

        bin_start_delta = pd.to_timedelta(start_str)
        bin_end_delta = pd.to_timedelta(end_str)

        bin_abs_start = temp_df['current_day_midnight'] + bin_start_delta
        bin_abs_end = temp_df['current_day_midnight'] + bin_end_delta

        temp_df['started_at'] = pd.concat([temp_df['started_at'], bin_abs_start], axis=1).max(axis=1)
        temp_df['finished_at'] = pd.concat([temp_df['finished_at'], bin_abs_end], axis=1).min(axis=1)

        valid_segments = temp_df[temp_df['started_at'] < temp_df['finished_at']]
        final_segments.append(valid_segments)

    df_split = pd.concat(final_segments).sort_values(by=['started_at']).reset_index(drop=True)

    cols_to_drop = ['num_days', 'day_offset', 'current_day_midnight', 'day_end_boundary']
    df_split = df_split.drop(columns=[c for c in cols_to_drop if c in df_split.columns])

    return df_split


def data_processing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full processing pipeline for a month (train or test):
    - timezone handling
    - split by days + time bins
    - compute distance_from_home
    - compute speeds vectorized
    - add day_type, time_segment, etc.
    """
    print("Starting data_processing...")
    
    # Timezone handling: we assume the timestamps are UTC, then we convert to Asia/Tokyo
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True).dt.tz_convert('Asia/Tokyo')
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True).dt.tz_convert('Asia/Tokyo')

    '''
    Warning: This approach is highly Timezone sensitive, we assume the data you provide are UTC, then we convert to
    Asia/Tokyo.

    Below is couple of lines, for you to refer if your data are already in Asia/Tokyo
    timezone and you just want to localize it.

    df['started_at'] = pd.to_datetime(df['started_at']).dt.tz_localize('Asia/Tokyo')
    df['finished_at'] = pd.to_datetime(df['finished_at']).dt.tz_localize('Asia/Tokyo')
    '''

    print("Splitting by time bins...")
    df = split_by_time_bins(df)
    print("After split_by_time_bins:", df.shape)

    df['duration_min'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60.0
    df['duration'] = df['duration_min'].clip(lower=0).fillna(0)

    print("Computing homes...")
    dur = df.groupby(['agent', 'location_id'])['duration'].sum()
    homes = dur.groupby('agent').idxmax()
    homes = homes.apply(lambda x: x[1])

    homes_df = homes.reset_index()
    homes_df.columns = ['agent', 'home_location_id']


    home_coords_unique = (
        df.groupby(['agent', 'location_id'], as_index=False)
          .agg(
              home_latitude=('latitude', 'first'),
              home_longitude=('longitude', 'first')
          )
    )

    homes_df = homes_df.merge(
        home_coords_unique,
        left_on=['agent', 'home_location_id'],
        right_on=['agent', 'location_id'],
        how='left',
        validate='one_to_one'
    )

    homes_df = homes_df[['agent', 'home_location_id', 'home_latitude', 'home_longitude']]

    df = df.merge(homes_df, on='agent', how='left', validate='many_to_one')

    print("Computing distance_from_home...")
    df['distance_from_home'] = haversine_vectorized(
        df['latitude'].to_numpy(),
        df['longitude'].to_numpy(),
        df['home_latitude'].to_numpy(),
        df['home_longitude'].to_numpy()
    )

    df['time_segment'] = df['started_at'].apply(assign_time_segment)
    df['day_of_week'] = df['started_at'].dt.dayofweek
    df['day_type'] = df['day_of_week'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')

    df = df.sort_values(['agent', 'started_at']).reset_index(drop=True)

    print("Computing speeds...")
    df = compute_speeds_vectorized(df)
    print("Speed computation completed.")

    df.fillna(0, inplace=True)

    return df


def compute_speeds_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute speed between consecutive staypoints for each agent,
    only when they are in the same time_segment.
    No explicit Python loops.
    """
    df = df.copy()
    df = df.sort_values(['agent', 'started_at'])

    df['prev_lat'] = df.groupby('agent')['latitude'].shift()
    df['prev_lon'] = df.groupby('agent')['longitude'].shift()
    df['prev_finished_at'] = df.groupby('agent')['finished_at'].shift()
    df['prev_time_segment'] = df.groupby('agent')['time_segment'].shift()

    same_segment = df['time_segment'] == df['prev_time_segment']
    valid = same_segment & df['prev_lat'].notna()

    n = len(df)
    dist = np.zeros(n, dtype='float64')
    dt_hours = np.ones(n, dtype='float64')

    idx = np.where(valid)[0]

    dist[idx] = haversine_vectorized(
        df['prev_lat'].values[idx],
        df['prev_lon'].values[idx],
        df['latitude'].values[idx],
        df['longitude'].values[idx]
    )

    dt_hours[idx] = (
        (df['started_at'].values[idx] - df['prev_finished_at'].values[idx]) /
        np.timedelta64(1, 'h')
    )

    speed = np.zeros(n, dtype='float64')
    nonzero = (dt_hours > 0) & valid.values
    speed[nonzero] = dist[nonzero] / dt_hours[nonzero]

    df['speed'] = speed

    df.drop(columns=['prev_lat', 'prev_lon', 'prev_finished_at', 'prev_time_segment'], inplace=True)

    return df
