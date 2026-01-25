import numpy as np
import pandas as pd
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
from helper_functions import *
import re
from collections import Counter
from scipy.stats import zscore
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def calculate_transformation_metrics(sequence_rows, home_coord, haversine_distance, home_location):
    """
    Calculate metrics for a transformation sequence.
    
    Args:
        sequence_rows: List of DataFrame rows representing the transformation sequence
        home_coord: Dict with 'latitude' and 'longitude' of home
        haversine_distance: Function to calculate distance
    
    Returns:
        Dict with metrics: avg_speed_first_half, avg_speed_second_half, max_distance_from_home
    """
    if len(sequence_rows) < 2:
        return {
            'avg_speed_first_half': np.nan,
            'avg_speed_second_half': np.nan,
            'max_distance_from_home': 0.0
        }
    
    # Calculate speeds between consecutive locations
    speeds = []
    for i in range(len(sequence_rows) - 1):
        prev_row = sequence_rows[i]
        curr_row = sequence_rows[i + 1]
        
        # Calculate distance between consecutive locations
        distance_km = haversine_distance(
            prev_row['latitude'], prev_row['longitude'],
            curr_row['latitude'], curr_row['longitude']
        )
        
        # Calculate time difference in hours
        time_diff_hours = (curr_row['started_at'] - prev_row['finished_at']).total_seconds() / 3600.0
        
        # Calculate speed (km/h), avoid division by zero
        if time_diff_hours > 0:
            speed = distance_km / time_diff_hours
            speeds.append(speed)
    
    # Split speeds into first half and second half
    num_transitions = len(speeds)
    if num_transitions == 0:
        return {
            'avg_speed_first_half': np.nan,
            'avg_speed_second_half': np.nan,
            'max_distance_from_home': 0.0
        }
    
    # First half: from start to midpoint
    # Second half: from midpoint to end
    midpoint = num_transitions // 2
    
    first_half_speeds = speeds[:midpoint] if midpoint > 0 else []
    second_half_speeds = speeds[midpoint:] if midpoint < num_transitions else []
    
    avg_speed_first_half = np.mean(first_half_speeds) if first_half_speeds else np.nan
    avg_speed_second_half = np.mean(second_half_speeds) if second_half_speeds else np.nan
    
    # Calculate distance from home for each location
    distances_from_home = []
    for row in sequence_rows:
        distance = haversine_distance(
            row['latitude'], row['longitude'],
            home_coord['latitude'], home_coord['longitude']
        )
        distances_from_home.append(distance)
    
    max_distance_from_home = max(distances_from_home) if distances_from_home else 0.0

    # ---------- Dominant POI category within the sequence ----------
    dominant_poi = None
    max_dur = -np.inf

    for row in sequence_rows:
        poi = row.get('poi_category', None)
        if pd.isna(poi):
            continue
        if row.get('location_id') == home_location:
            continue

        # Prefer precomputed duration_min; otherwise compute from timestamps
        if 'duration_min' in row:
            dur = row['duration_min']
        else:
            dur = (row['finished_at'] - row['started_at']).total_seconds() / 60.0

        if pd.isna(dur):
            continue

        if dur > max_dur:
            max_dur = dur
            dominant_poi = poi
    return {
        'avg_speed_first_half': avg_speed_first_half,
        'avg_speed_second_half': avg_speed_second_half,
        'max_distance_from_home': max_distance_from_home,
        'dominant_poi': dominant_poi
    }



def extract_home_to_home_transformations(data, homes):
    """
    Extract location transformations for each agent where each transformation
    starts from home and ends at home, with additional metrics.
    
    Args:
        data: DataFrame with columns ['agent', 'started_at', 'finished_at', 'location_id', 
              'latitude', 'longitude']
        homes: Series or DataFrame mapping agent to home_location_id
              (can be a Series with agent as index, or DataFrame with 'agent' and 'home_location_id' columns)
    
    Returns:
        DataFrame with columns: ['agent', 'transformation', 'started_at', 'finished_at',
                                'avg_speed_first_half', 'avg_speed_second_half', 'max_distance_from_home']
        where 'transformation' is a list of location_ids in sequence (e.g., [home, X, Y, Z, home])
    """
    
    if isinstance(homes, pd.DataFrame):
        if 'home_location_id' in homes.columns:
            homes_dict = dict(zip(homes['agent'], homes['home_location_id']))
        else:
            # Assume first column is agent, second is home_location_id
            homes_dict = dict(zip(homes.iloc[:, 0], homes.iloc[:, 1]))
    elif isinstance(homes, pd.Series):
        homes_dict = homes.to_dict()
    else:
        raise ValueError("homes must be a DataFrame or Series")
    
    # Get home coordinates for each agent
    home_coords = {}
    for agent in data['agent'].unique():
        home_location = homes_dict.get(agent)
        if home_location is not None:
            home_row = data[(data['agent'] == agent) & 
                           (data['location_id'] == home_location)].iloc[0]
            home_coords[agent] = {
                'latitude': home_row['latitude'],
                'longitude': home_row['longitude']
            }
    
    data = data.sort_values(['agent', 'started_at']).reset_index(drop=True)
    
    transformations = []
    
    for agent in data['agent'].unique():
        agent_data = data[data['agent'] == agent].copy().reset_index(drop=True)
        home_location = homes_dict.get(agent)
        home_coord = home_coords.get(agent)
        
        if home_location is None or home_coord is None:
            continue  
        
        current_sequence = []
        current_sequence_rows = []  
        sequence_start_time = None
        
        for idx, row in agent_data.iterrows():
            location_id = row['location_id']
            
            if location_id == home_location:
                if len(current_sequence) > 0:
                    
                    # Complete the transformation by adding home at the end
                    current_sequence.append(location_id)
                    current_sequence_rows.append(row)
                    sequence_end_time = row['started_at']
                    
                    # Only add if we have at least home->something->home
                    if len(current_sequence) >= 2:
                        # Calculate metrics for this transformation
                        transformation_metrics = calculate_transformation_metrics(
                            current_sequence_rows, home_coord, haversine_distance, home_location
                        )
                        
                        transformations.append({
                            'agent': agent,
                            'transformation': current_sequence.copy(),
                            'started_at': sequence_start_time,
                            'finished_at': sequence_end_time,
                            **transformation_metrics
                        })
                    
                    # Start a new sequence
                    current_sequence = [location_id]
                    current_sequence_rows = [row]
                    sequence_start_time = row['finished_at']
                else:
                    # Start a new sequence from home
                    current_sequence = [location_id]
                    current_sequence_rows = [row]
                    sequence_start_time = row['finished_at']
            else:
                # We're not at home, add to current sequence
                if len(current_sequence) > 0:
                    current_sequence.append(location_id)
                    current_sequence_rows.append(row)
    
    result_df = pd.DataFrame(transformations)
    
    if len(result_df) > 0:
        # Convert transformation list to string representation for easier viewing
        result_df['transformation_str'] = result_df['transformation'].apply(
            lambda x: '->'.join(map(str, x))
        )
    
    return result_df


def processing_transformations(path):

    data = pd.read_parquet(path)

    data['started_at'] = pd.to_datetime(data['started_at'])
    data['finished_at'] = pd.to_datetime(data['finished_at'])

    data['duration_min'] = (data['finished_at'] - data['started_at']).dt.total_seconds() / 60.0
    data['duration'] = data['duration_min'].clip(lower=0).fillna(0)

    dur = data.groupby(['agent', 'location_id'])['duration'].sum()
    homes = dur.groupby('agent').idxmax()
    homes = homes.apply(lambda x: x[1])
    homes_df = homes.reset_index()
    homes_df.columns = ['agent', 'home_location_id']


    results = extract_home_to_home_transformations(data, homes_df)
    results['started_at'] = results['started_at'].dt.round('30min')
    results['finished_at'] = results['finished_at'].dt.round('30min')
    results['duration_min'] = (results['finished_at'] - results['started_at']).dt.total_seconds() / 60.0
    results['duration'] = results['duration_min'].clip(lower=0).fillna(0)


    #results['dominant_poi'] = results.apply(dominant_poi, axis=1, args=(data,))

    results = results.fillna(0)

    return results



def bucket_id_from_path(p: Path) -> int:
    return int(p.name.split("agent_bucket=")[1].split(".parquet")[0])

def process_one_bucket(b: int,
                       train_path: str,
                       test_path: str,
                       out_train: str,
                       out_test: str,
                       resident_agents: set):
    """Run one bucket end-to-end. Designed to be picklable for multiprocessing."""
    train_profile = processing_transformations(Path(train_path))
    test_profile  = processing_transformations(Path(test_path))

    # filter
    train_profile = train_profile[train_profile["agent"].isin(resident_agents)]
    test_profile  = test_profile[test_profile["agent"].isin(resident_agents)]

    # write
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(out_test).parent.mkdir(parents=True, exist_ok=True)
    train_profile.to_csv(out_train, index=False)
    test_profile.to_csv(out_test, index=False)

    return b, len(train_profile), len(test_profile)

def main():
    TRAIN_DIR = Path("../processed/trial5/sim1/2m/whole/stop_past")
    TEST_DIR  = Path("../processed/trial5/sim1/2m/whole/stop_future")

    OUT_TRAIN_DIR = Path("../processed/trial5/sim1/2m/ore/stop_past")
    OUT_TEST_DIR  = Path("../processed/trial5/sim1/2m/ore/stop_future")
    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    residents = pd.read_csv(
        "../processed/trial5/sim1/Sim1_Trial5_Agent_Classification.csv",
        low_memory=False
    )
    residents = residents[
        (residents["past_classification"] == "residents") &
        (residents["future_classification"] == "residents")
    ]
    resident_agents = set(residents["agent"].values.tolist())

    train_files = {bucket_id_from_path(p): p for p in TRAIN_DIR.glob("agent_bucket=*.parquet")}
    test_files  = {bucket_id_from_path(p): p for p in TEST_DIR.glob("agent_bucket=*.parquet")}
    common_buckets = sorted(set(train_files).intersection(test_files))
    if not common_buckets:
        raise RuntimeError("No matching agent_bucket files found between train and test dirs.")

    # Choose workers: don’t be an HPC menace. Start with min(CPU, #buckets).
    max_workers = min(os.cpu_count() or 4, len(common_buckets))

    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for b in common_buckets:
            train_path = str(train_files[b])
            test_path  = str(test_files[b])
            out_train  = str(OUT_TRAIN_DIR / f"agent_bucket={b}.csv")
            out_test   = str(OUT_TEST_DIR  / f"agent_bucket={b}.csv")

            tasks.append(
                ex.submit(process_one_bucket, b, train_path, test_path, out_train, out_test, resident_agents)
            )

        for fut in as_completed(tasks):
            b, n_train, n_test = fut.result()
            print(f"=== Done bucket {b}: train_rows={n_train:,} test_rows={n_test:,} ===")

if __name__ == "__main__":
    main()