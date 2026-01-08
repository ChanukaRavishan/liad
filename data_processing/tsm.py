import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import math
import argparse
import ArgumentParser
import os

# ---------- CONFIG ----------

# how many parallel workers to use when building profiles
# start low (1–4), especially with 10GB files
N_JOBS_PROFILES = 1


# ---------- UTILS ----------

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


# ---------- FILTER AGENTS ----------

def agent_type_filter(train_df: pd.DataFrame):
    """
    Compute total duration per agent on the train month and keep agents
    above the 16th percentile (like your original).
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


# ---------- SPLIT BY DAY + TIME BINS ----------

def split_by_time_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split staypoints that span multiple days and then slice by fixed time bins.
    This is still heavy, but we avoid pointless extra copies.
    """
    df = df.copy()

    # number of days spanned by each row
    df['num_days'] = (df['finished_at'].dt.normalize() - df['started_at'].dt.normalize()).dt.days + 1

    # explode over days
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

    # Note: we still copy per bin because we're modifying started/finished in-place per bin.
    # This is less horrific than before because we killed the other bottlenecks.
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


# ---------- MAIN DATA PROCESSING PER MONTH ----------

def data_processing(df: pd.DataFrame, residents) -> pd.DataFrame:
    """
    Full processing pipeline for a month (train or test):
    - filter agents
    - timezone handling
    - split by days + time bins
    - compute distance_from_home
    - compute speeds vectorized
    - add day_type, time_segment, etc.
    """
    print("Starting data_processing...")
    df = df[df['agent'].isin(residents)].copy()
    print("After resident filter:", df.shape)

    # Timezone handling: assume timestamps are UTC, convert to Asia/Tokyo
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True).dt.tz_convert('Asia/Tokyo')
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True).dt.tz_convert('Asia/Tokyo')

    # Split into day segments and time bins
    print("Splitting by time bins...")
    df = split_by_time_bins(df)
    print("After split_by_time_bins:", df.shape)

    # Duration in minutes
    df['duration_min'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60.0
    df['duration'] = df['duration_min'].clip(lower=0).fillna(0)

    # Find home per agent as location with max total duration
    print("Computing homes...")
    dur = df.groupby(['agent', 'location_id'])['duration'].sum()
    homes = dur.groupby('agent').idxmax()
    homes = homes.apply(lambda x: x[1])  # extract location_id

    homes_df = homes.reset_index()
    homes_df.columns = ['agent', 'home_location_id']

    # Unique coords per (agent, location_id)
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

    # Distance from home (vectorized)
    print("Computing distance_from_home...")
    df['distance_from_home'] = haversine_vectorized(
        df['latitude'].to_numpy(),
        df['longitude'].to_numpy(),
        df['home_latitude'].to_numpy(),
        df['home_longitude'].to_numpy()
    )

    # Time segment, day type
    df['time_segment'] = df['started_at'].apply(assign_time_segment)
    df['day_of_week'] = df['started_at'].dt.dayofweek
    df['day_type'] = df['day_of_week'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')

    # Sort
    df = df.sort_values(['agent', 'started_at']).reset_index(drop=True)

    # Compute speeds between staypoints, vectorized
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

    # previous row within agent
    df['prev_lat'] = df.groupby('agent')['latitude'].shift()
    df['prev_lon'] = df.groupby('agent')['longitude'].shift()
    df['prev_finished_at'] = df.groupby('agent')['finished_at'].shift()
    df['prev_time_segment'] = df.groupby('agent')['time_segment'].shift()

    same_segment = df['time_segment'] == df['prev_time_segment']
    valid = same_segment & df['prev_lat'].notna()

    # init arrays
    n = len(df)
    dist = np.zeros(n, dtype='float64')
    dt_hours = np.ones(n, dtype='float64')  # avoid div by zero default

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


# ---------- BUILD PROFILES (OPTIONALLY PARALLEL) ----------

def _build_profiles_single_chunk(chunk_df: pd.DataFrame):
    """
    Build profiles for a subset of agents (single process, no parallel).
    """
    profiles = []

    segment_required_minutes = {
        '0-5.59': 6 * 60,           # 360
        '6-8.59': 3 * 60,           # 180
        '9-13.59': 5 * 60,          # 300
        '14-17.29': int(3.5 * 60),  # 210
        '17.30-21.29': 4 * 60,      # 240
        '21.30-23.59': int(2.5 * 60)  # 150
    }

    for agent, agent_data in chunk_df.groupby('agent'):
        for day_type in ['weekday', 'weekend']:
            day_data = agent_data[agent_data['day_type'] == day_type]

            for segment in ['0-5.59', '6-8.59', '9-13.59',
                            '14-17.29', '17.30-21.29', '21.30-23.59']:
                segment_data = day_data[day_data['time_segment'] == segment]

                if len(segment_data) == 0:
                    continue

                period_start = segment_data['started_at'].min()
                period_end = segment_data['started_at'].max()

                # 1. unique location IDs
                unique_locations = segment_data['location_id'].nunique()

                # 2. avg distance from home
                avg_distance_from_home = round(segment_data['distance_from_home'].mean(), 2)

                # 3. avg speed
                avg_speed = round(segment_data['speed'].mean(), 2)

                # 4. array of unique locations
                unique_loc = segment_data['location_id'].unique()

                # 5. mean of daily max stay duration
                mean_of_daily_max = round(
                    segment_data.groupby(segment_data['started_at'].dt.date)['duration'].max().mean(),
                    2
                )

                # 6. mean transformations per day (rounded up)
                transformations = segment_data.groupby(
                    segment_data['started_at'].dt.date
                )['location_id'].count().mean()
                transformations = math.ceil(transformations)

                # 7. max distance from home
                max_distance_from_home = round(segment_data['distance_from_home'].max(), 2)

                # 8. dominant poi category
                if 'poi_category' in segment_data.columns and segment_data['poi_category'].notna().any():
                    dominent_poi = segment_data['poi_category'].value_counts().idxmax()
                    poi_dict = segment_data['poi_category'].dropna().unique()
                else:
                    dominent_poi = None
                    poi_dict = []

                profiles.append({
                    's_date': period_start,
                    'e_date': period_end,
                    'agent': agent,
                    'day_type': day_type,
                    'time_segment': segment,
                    'unique_location_ids': unique_locations,
                    'avg_distance_from_home_km': avg_distance_from_home,
                    'avg_speed_kmh': avg_speed,
                    'unique_locs': unique_loc,
                    'max_stay_duration': mean_of_daily_max,
                    'transformations': transformations,
                    'max_distance_from_home': max_distance_from_home,
                    'dominent_poi': dominent_poi,
                    'poi_dict': poi_dict
                })

    return profiles


def build_profiles(data: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    """
    Build agent profiles with optional parallelization over agent chunks.
    """
    print("Building agent profiles...")

    if n_jobs is None or n_jobs < 2:
        # single process
        profiles = _build_profiles_single_chunk(data)
    else:
        agents = data['agent'].unique()
        agent_chunks = np.array_split(agents, n_jobs)

        chunks = []
        for chunk_ids in agent_chunks:
            chunk_df = data[data['agent'].isin(chunk_ids)].copy()
            chunks.append(chunk_df)

        results = Parallel(n_jobs=n_jobs)(
            delayed(_build_profiles_single_chunk)(chunk_df)
            for chunk_df in chunks
        )

        profiles = []
        for r in results:
            profiles.extend(r)

    agent_profiles = pd.DataFrame(profiles)
    print(f"Profiles created for {agent_profiles['agent'].nunique()} agents")
    print(f"Total profile entries: {len(agent_profiles)}")

    agent_profiles.fillna(0, inplace=True)
    return agent_profiles


# ---------- MAIN SCRIPT ----------

if __name__ == "__main__":
    train_path = '../processed/train.csv'
    test_path = '../processed/test.csv'
    residents_path = '../processed/residents.csv'   # cache

    parser = ArgumentParser(
        description='Data Processing for LIAD (Warning: Timezone Sensitive)'
    )
    parser.add_argument(
        'datatype',
        type=int,
        choices=[1, 2],
        help='1 = process TRAIN, 2 = process TEST'
    )

    args = parser.parse_args()

    # --------- Load or compute residents ---------
    if os.path.exists(residents_path):
        print(f"Loading cached residents from {residents_path} ...")
        residents_df = pd.read_csv(residents_path)
        residents = residents_df['agent'].values
        print("Number of residents (cached):", len(residents))
    else:
        print("Reading train CSV to compute residents...")
        train_df_for_residents = pd.read_csv(train_path)
        print("Train shape (for residents):", train_df_for_residents.shape)

        print("Selecting residents from train...")
        residents = agent_type_filter(train_df_for_residents)
        print("Number of residents:", len(residents))

        # cache to disk so test runs don't need to re-read 10GB train
        residents_df = pd.DataFrame({'agent': residents})
        residents_df.to_csv(residents_path, index=False)
        print(f"Residents cached to {residents_path}")

        # free memory
        del train_df_for_residents

    # --------- Helper: weekly profiles ---------
    def build_weekly_profiles(df, monthly_profiles_path, weekly_profiles_path, label: str):
        """
        df: already processed (data_processing), with started_at tz-aware
        """
        # monthly
        print(f"Building {label} monthly profiles...")
        month_profiles = build_profiles(df, n_jobs=N_JOBS_PROFILES)
        month_profiles.to_csv(monthly_profiles_path, index=False)
        print(f"Saved {label} monthly profiles to {monthly_profiles_path}")

        # weekly
        print(f"Building {label} weekly profiles...")
        df['week'] = df['started_at'].dt.to_period('W').astype(str)
        weekly_list = []
        i = 0
        for wk, chunk in df.groupby('week'):
            print(f"  Processing {label} week: {wk}, rows: {len(chunk)}")
            prof = build_profiles(chunk, n_jobs=N_JOBS_PROFILES)
            prof['chunk'] = i
            weekly_list.append(prof)
            i += 1
        weekly_profiles = pd.concat(weekly_list, ignore_index=True)
        weekly_profiles.to_csv(weekly_profiles_path, index=False)
        print(f"Saved {label} weekly profiles to {weekly_profiles_path}")

    # --------- Branch: train / test ---------

    if args.datatype == 1:
        # TRAIN
        print("Reading full TRAIN CSV...")
        train_df = pd.read_csv(train_path)
        print("Train shape:", train_df.shape)

        print("Running data_processing on TRAIN...")
        train_data = data_processing(train_df, residents)
        del train_df   # free raw

        build_weekly_profiles(
            df=train_data,
            monthly_profiles_path='../processed/train_monthly.csv',
            weekly_profiles_path='../processed/train_weekly.csv',
            label='TRAIN'
        )

        del train_data
        print("TRAIN processing complete.")

    elif args.datatype == 2:
        # TEST
        print("Reading full TEST CSV...")
        test_df = pd.read_csv(test_path)
        print("Test shape:", test_df.shape)

        print("Running data_processing on TEST...")
        test_data = data_processing(test_df, residents)
        del test_df   # free raw

        build_weekly_profiles(
            df=test_data,
            monthly_profiles_path='../processed/test_monthly.csv',
            weekly_profiles_path='../processed/test_weekly.csv',
            label='TEST'
        )

        del test_data
        print("TEST processing complete.")
