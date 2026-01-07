import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import math


def agent_type_filter(train):

    train_data = pd.read_csv(train)

    train_data['started_at'] = pd.to_datetime(train_data['started_at'])
    train_data['finished_at'] = pd.to_datetime(train_data['finished_at'])

    train_data['duration_min'] = (train_data['finished_at'] - train_data['started_at']).dt.total_seconds() / 60.0
    train_data['duration'] = train_data['duration_min'].clip(lower=0).fillna(0)

    train_agent_dur = train_data.groupby('agent')['duration'].sum()

    df = pd.DataFrame({
    'train_duration': train_agent_dur,
        }).fillna(0)

    q1_value = df['train_duration'].quantile(0.16)

    df_top_q1 = df[df['train_duration'] >= q1_value]

    return df_top_q1.index


def split_by_time_bins(df):
    
    df = df.copy()
    
    df['num_days'] = (df['finished_at'].dt.normalize() - df['started_at'].dt.normalize()).dt.days + 1
    
    df_exploded = df.loc[df.index.repeat(df['num_days'])].copy()
    df_exploded['day_offset'] = df_exploded.groupby(level=0).cumcount()
    
    df_exploded['current_day_midnight'] = df_exploded['started_at'].dt.normalize() + pd.to_timedelta(df_exploded['day_offset'], unit='D')
    
    df_exploded['started_at'] = df_exploded[['started_at', 'current_day_midnight']].max(axis=1)
    df_exploded['day_end_boundary'] = df_exploded['current_day_midnight'] + pd.to_timedelta(1, unit='D')
    df_exploded['finished_at'] = df_exploded[['finished_at', 'day_end_boundary']].min(axis=1)
    
    df_daily = df_exploded[df_exploded['started_at'] < df_exploded['finished_at']].reset_index(drop=True)

    time_bins = [
        ('00:00:00', '06:00:00', 'Early Morning'),  # 0:00 - 5:59
        ('06:00:00', '09:00:00', 'Morning Rush'),   # 6:00 - 8:59
        ('09:00:00', '14:00:00', 'Mid Day'),        # 9:00 - 13:59
        ('14:00:00', '17:30:00', 'Afternoon'),      # 14:00 - 17:29
        ('17:30:00', '21:30:00', 'Evening'),        # 17:30 - 21:29
        ('21:30:00', '1 day',    'Night')           # 21:30 - 23:59
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


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth (in km)
    using the Haversine formula.
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance

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

def data_processing(path, residents):
    
    data = pd.read_csv(path)
    
    data = data[data.agent.isin(residents)]

    data['started_at'] = pd.to_datetime(data['started_at'])
    data['started_at'] = data['started_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')
    data['finished_at'] = pd.to_datetime(data['finished_at'])
    data['finished_at'] = data['finished_at'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')
    
    data = split_by_time_bins(data.copy())

    data['duration_min'] = (data['finished_at'] - data['started_at']).dt.total_seconds() / 60.0
    data['duration'] = data['duration_min'].clip(lower=0).fillna(0)


    dur = data.groupby(['agent', 'location_id'])['duration'].sum()
    homes = dur.groupby('agent').idxmax()
    homes = homes.apply(lambda x: x[1])
    homes_df = homes.reset_index()
    homes_df.columns = ['agent', 'home_location_id']



    home_coords_unique = (
    data.groupby(['agent', 'location_id'], as_index=False)
        .agg(
            home_latitude=('latitude', 'first'),
            home_longitude=('longitude', 'first')
        )
    )

    # Merge homes_df (one row per agent) with unique coordinates
    homes_df = homes_df.merge(
    home_coords_unique,
    left_on=['agent', 'home_location_id'],
    right_on=['agent', 'location_id'],
    how='left',
    validate='one_to_one'  # <-- will raise if you mess this up again
    )

    homes_df = homes_df[['agent', 'home_location_id', 'home_latitude', 'home_longitude']]

    # Now this is safely one-to-one on agent:
    data = data.merge(homes_df, on='agent', how='left', validate='many_to_one')

    # Calculate distance from home for each staypoint
    data['distance_from_home'] = data.apply(
    lambda row: haversine_distance(
        row['latitude'], row['longitude'],
        row['home_latitude'], row['home_longitude']
    ), axis=1
    )

    # Assign time segments
    data['time_segment'] = data['started_at'].apply(assign_time_segment)

    data['day_of_week'] = data['started_at'].dt.dayofweek
    data['day_type'] = data['day_of_week'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')

    data = data.sort_values(['agent', 'started_at']).reset_index(drop=True)

    print("Calculating speeds between staypoints...")
    data['speed'] = np.nan

    for agent in data['agent'].unique():
        agent_mask = data['agent'] == agent
        agent_indices = data[agent_mask].index.tolist()
        agent_data = data.loc[agent_indices].copy()
        
        for i in range(1, len(agent_data)):
            prev_idx = agent_indices[i-1]
            curr_idx = agent_indices[i]
            
            prev_row = data.loc[prev_idx]
            curr_row = data.loc[curr_idx]
            
            if prev_row['time_segment'] == curr_row['time_segment']:
                # Calculate distance between locations
                distance = haversine_distance(
                    prev_row['latitude'], prev_row['longitude'],
                    curr_row['latitude'], curr_row['longitude']
                )
                
                time_diff = (curr_row['started_at'] - prev_row['finished_at']).total_seconds() / 3600.0
                
                if time_diff > 0:
                    speed = distance / time_diff
                    data.at[curr_idx, 'speed'] = speed

    print("Speed calculation completed.")
    data.fillna(0, inplace=True)

    return data

def build_profiles(data):

    print("Building agent profiles...")

    profiles = []

    segment_required_minutes = {
    '0-5.59': 6 * 60,          # 360
    '6-8.59': 3 * 60,          # 180
    '9-13.59': 5 * 60,         # 300
    '14-17.29': int(3.5 * 60), # 210
    '17.30-21.29': 4 * 60,     # 240
    '21.30-23.59': int(2.5 * 60)  # 150
}

    for agent in data['agent'].unique():
        agent_data = data[data['agent'] == agent]
        
        for day_type in ['weekday', 'weekend']:
            day_data = agent_data[agent_data['day_type'] == day_type]
            
            for segment in ['0-5.59', '6-8.59', '9-13.59', '14-17.29', '17.30-21.29', '21.30-23.59']:
                segment_data = day_data[day_data['time_segment'] == segment]
                
                if len(segment_data) > 0:

                    # segment_data['date'] = segment_data['started_at'].dt.date
                    # duration_per_day = segment_data.groupby('date')['duration'].sum()

                    # required = segment_required_minutes[segment]
                    # threshold = 0.5 * required  # 60%

                    # # days that have enough coverage
                    # valid_days = duration_per_day[duration_per_day >= threshold].index

                    # # keep only rows from valid days
                    # segment_data = segment_data[segment_data['date'].isin(valid_days)]

                    # if after filtering nothing remains, skip
                    if len(segment_data) == 0:
                        continue

                    period_start = segment_data['started_at'].min()
                    period_end = segment_data['started_at'].max()

                    # 1. Unique location IDs
                    unique_locations = segment_data['location_id'].nunique()
                    
                    # 2. Average distance from home
                    avg_distance_from_home = round(segment_data['distance_from_home'].mean(), 2)
                    
                    # 3. Average speed between staypoints (only for rows with speed calculated)
                    avg_speed = round(segment_data['speed'].mean(),2)

                    # 4. Array of unique locations
                    unique_loc = segment_data['location_id'].unique()

                    # 5. Max Stay duration in one staypoint
                    mean_of_daily_max = round(segment_data.groupby(segment_data['started_at'].dt.date)['duration'].max().mean(), 2)

                    # 6. mean transformations of staypoints within a day
                    transformations = segment_data.groupby(segment_data['started_at'].dt.date)['location_id'].count().mean()
                    transformations = math.ceil(transformations)

                    # 7. Max distance from home
                    max_distance_from_home = round(segment_data['distance_from_home'].max(), 2)
                    
                    # 8. Domanent staypoint Category
                    dominent_poi = segment_data['poi_category'].value_counts().idxmax()

                    # 9. Dictionary of POIs visited
                    poi_dict = segment_data['poi_category'].unique()

                    
                    
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

    agent_profiles = pd.DataFrame(profiles)
    print(f"Profiles created for {agent_profiles['agent'].nunique()} agents")
    print(f"Total profile entries: {len(agent_profiles)}")

    agent_profiles.fillna(0, inplace=True)

    return agent_profiles


train = '../processed/train.csv'
test = '../processed/test.csv'

residents = agent_type_filter(train)
train_data = data_processing(train, residents)
train_agent_profiles = build_profiles(train_data)
train_agent_profiles.to_csv('../processed/train_monthly.csv', index=False)

test_data = data_processing(test, residents)
test_agent_profiles = build_profiles(test_data)
test_agent_profiles.to_csv('../processed/test_monthly.csv', index=False)



test_data = data_processing(test, residents)

test_data['week'] = test_data['started_at'].dt.to_period('W').astype(str)

test_profiles = []
i = 0

for wk, chunk in test_data.groupby('week'):

    print(f"Processing week: {wk}, rows: {len(chunk)}")
    prof = build_profiles(chunk)
    prof['chunk'] = i
    test_profiles.append(prof)
    i = i + 1

test_agent_profiles = pd.concat(test_profiles, ignore_index=True)
test_agent_profiles.to_csv('../processed/test_weekly.csv', index=False)



train_data = data_processing(train, residents)

train_data['week'] = train_data['started_at'].dt.to_period('W').astype(str)

train_profiles = []
i = 0

for wk, chunk in train_data.groupby('week'):

    print(f"Processing week: {wk}, rows: {len(chunk)}")
    prof = build_profiles(chunk)
    prof['chunk'] = i
    train_profiles.append(prof)
    i = i + 1

train_agent_profiles = pd.concat(train_profiles, ignore_index=True)
train_agent_profiles.to_csv('../processed/train_weekly.csv', index=False)