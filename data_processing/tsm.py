import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from joblib import Parallel, delayed
import math
from argparse import ArgumentParser
import os
from concurrent.futures import ProcessPoolExecutor
import functools
from pathlib import Path
from helper_functions import *
# ---------- CONFIG ----------

# how many parallel workers to use when building profiles
N_JOBS_PROFILES = 40



# ---------- BUILD PROFILES ----------

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
                unique_loc = segment_data['location_id'].unique().tolist()

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

                    poi_durations = segment_data.dropna(subset=['poi_category']).groupby('poi_category')['duration'].sum()
                    dominant_poi = poi_durations.idxmax()

                    poi_dict = segment_data['poi_category'].dropna().unique().tolist()
                else:
                    dominant_poi = None
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
                    'dominent_poi': dominant_poi,
                    'poi_dict': poi_dict
                })

    return pd.DataFrame(profiles)


def build_profiles(data: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    """
    Build agent profiles with optional parallelization over agent chunks.
    """
    print("Building agent profiles...")

    if n_jobs is None or n_jobs < 2:
        agent_profiles = _build_profiles_single_chunk(data)
        if not isinstance(agent_profiles, pd.DataFrame):
            raise TypeError("_build_profiles_single_chunk must return a pd.DataFrame")
    else:
        agents = data["agent"].unique()
        agent_chunks = np.array_split(agents, n_jobs)

        # Build chunks (still copies; see note below for better)
        chunks = [data[data["agent"].isin(chunk_ids)] for chunk_ids in agent_chunks if len(chunk_ids) > 0]

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_build_profiles_single_chunk)(chunk_df)
            for chunk_df in chunks
        )

        # Validate + concat
        for r in results:
            if not isinstance(r, pd.DataFrame):
                raise TypeError("_build_profiles_single_chunk must return a pd.DataFrame")
        agent_profiles = pd.concat(results, ignore_index=True)

    if "agent" in agent_profiles.columns:
        print(f"Profiles created for {agent_profiles['agent'].nunique()} agents")
    print(f"Total profile entries: {len(agent_profiles)}")

    return agent_profiles.fillna(0)




def append_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)

def bucket_id_from_path(p: Path) -> int:

    return int(p.name.split("agent_bucket=")[1].split(".parquet")[0])

def process_one_file(in_path: Path, out_path: Path, repartition: bool):
    """
    Read one parquet, process it, build monthly profiles, and append/save to out_path.
    If repartition=False and out_path exists, skip.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not repartition:
        print(f"SKIP (exists): {out_path}")
        return

    print(f"READ : {in_path}")
    df = pd.read_parquet(in_path)

    processed = data_processing(df)
    #monthly_profiles = build_profiles(processed, n_jobs=N_JOBS_PROFILES)
    #append_df(monthly_profiles, out_path)

    # weekly profiles
    processed["week"] = processed["started_at"].dt.to_period("W").astype(str)
    weekly_list = []
    i = 0
    for wk, chunk in processed.groupby("week"):
       prof = build_profiles(chunk, n_jobs=N_JOBS_PROFILES)
       prof["chunk"] = i
       weekly_list.append(prof)
       i += 1
    weekly_profiles = pd.concat(weekly_list, ignore_index=True)

    append_df(weekly_profiles, out_path)

    del df, processed, weekly_profiles
    print(f"WROTE: {out_path}")


def process_bucket(b, train_files, test_files, out_train_dir, out_test_dir, repartition):
    """Helper function to process a single bucket pair."""
    in_train = train_files[b]
    in_test  = test_files[b]

    out_train = out_train_dir / f"agent_bucket={b}.csv"
    out_test  = out_test_dir  / f"agent_bucket={b}.csv"

    print(f"Starting Bucket {b}...")
    process_one_file(in_train, out_train, repartition=repartition)
    process_one_file(in_test,  out_test,  repartition=repartition)
    return f"Finished Bucket {b}"

def main():
    train_dir = Path("../processed/trial5/2m/stop_past")
    test_dir  = Path("../processed/trial5/2m/stop_future")

    out_train_dir = Path("../processed/trial5/2m/train_weekly")
    out_test_dir  = Path("../processed/trial5/2m/test_weekly")

    parser = ArgumentParser(description="Build monthly profiles per agent_bucket for train & test")
    parser.add_argument("--repartition", action="store_true",
                        help="Force rebuild output files even if they exist")
    args = parser.parse_args()

    train_files = {bucket_id_from_path(p): p for p in train_dir.glob("agent_bucket=*.parquet")}
    test_files  = {bucket_id_from_path(p): p for p in test_dir.glob("agent_bucket=*.parquet")}

    common = sorted(set(train_files).intersection(test_files))
    
    max_workers = os.cpu_count() - 2
    
    print(f"Parallelizing {len(common)} buckets across {max_workers} cores...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        func = functools.partial(
            process_bucket, 
            train_files=train_files, 
            test_files=test_files, 
            out_train_dir=out_train_dir, 
            out_test_dir=out_test_dir, 
            repartition=args.repartition
        )
    
        results = list(executor.map(func, common))

    for res in results:
        print(res)

if __name__ == "__main__":
    main()