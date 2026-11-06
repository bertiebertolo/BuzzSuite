"""
Flight Center Distance Analysis Module

This module provides functions to analyze mosquito flight distances to cage center,
stratified by sex category. It calculates the average distance from the cage center
for each flight fragment (>20 frames), exports results to CSV, and creates visualizations.

Author: BuzzWatch Analysis Team
"""

import numpy as np
import pandas as pd
import os
from scipy import stats


def calculate_cage_centroid(cage_border_points):
    """
    Calculate the centroid of the cage polygon (any number of vertices >= 3).
    Uses the standard polygon centroid (area-weighted, shoelace formula). If the
    polygon is degenerate (zero area), falls back to the mean of vertices.
    """
    if cage_border_points is None or len(cage_border_points) < 3:
        return None

    points = np.array([tuple(cage_border_points[i]) for i in np.arange(len(cage_border_points))])

    # Ensure polygon is closed for centroid calculation
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    x = points[:, 0]
    y = points[:, 1]

    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * np.sum(cross)

    if abs(area) < 1e-9:  # Degenerate polygon, fallback to vertex mean
        return (float(np.mean(x[:-1])), float(np.mean(y[:-1])))

    cx = (1 / (6 * area)) * np.sum((x[:-1] + x[1:]) * cross)
    cy = (1 / (6 * area)) * np.sum((y[:-1] + y[1:]) * cross)

    return (float(cx), float(cy))


def analyze_flight_center_distances(mosquito_tracks, cage_border_points, min_flight_frames=20):
    """
    Analyze distance from each flight point to cage center.
    
    For each continuous flight segment (state==1) with duration >= min_flight_frames:
    - Calculate Euclidean distance from each point to cage centroid
    - Compute average distance for the segment
    
    Parameters
    ----------
    mosquito_tracks : mosquito_obj_tracker
        Object containing tracked mosquito trajectories
    cage_border_points : list of tuples
        4-point polygon defining cage borders
    min_flight_frames : int, default=20
        Minimum flight duration (frames) to include in analysis
    
    Returns
    -------
    list of dict
        Each dict contains:
        - mosquito_id: tracking ID
        - fragment_id: unique ID for this flight segment
        - avg_distance: average distance (pixels) to cage center
        - min_distance: minimum distance in segment
        - max_distance: maximum distance in segment
        - fragment_duration: number of frames in flight
        - start_frame: absolute frame index of segment start
        - end_frame: absolute frame index of segment end
        - num_points: number of points in segment
    """
    
    centroid = calculate_cage_centroid(cage_border_points)
    if centroid is None:
        raise ValueError("Cannot calculate cage centroid - cage_border_points not set")
    
    results = []
    
    if not hasattr(mosquito_tracks, 'objects') or mosquito_tracks.objects is None:
        raise ValueError("mosquito_tracks.objects not available")
    
    # Iterate through all mosquito tracks
    for mosquito_id, obj in mosquito_tracks.objects.items():
        coordinates = obj.get('coordinates', [])
        state = obj.get('state', [])
        start_frame = obj.get('start', 0)
        
        if len(coordinates) == 0 or len(state) == 0:
            continue
        
        # Find all continuous flight segments (state == 1)
        i = 0
        fragment_id = 0
        while i < len(state):
            # Skip non-flying states
            if state[i] != 1:
                i += 1
                continue
            
            # Found start of flight segment
            j = i
            while j < len(state) and state[j] == 1:
                j += 1
            
            # Extract flight segment
            flight_segment = coordinates[i:j]
            flight_duration = j - i
            
            # Only analyze segments >= min_flight_frames
            if flight_duration >= min_flight_frames:
                # Calculate distance from each point to centroid
                distances = []
                for point in flight_segment:
                    if point is not None and len(point) == 2:
                        try:
                            dx = point[0] - centroid[0]
                            dy = point[1] - centroid[1]
                            distance = np.sqrt(dx**2 + dy**2)
                            distances.append(distance)
                        except (TypeError, ValueError):
                            continue
                
                if len(distances) > 0:
                    results.append({
                        'mosquito_id': mosquito_id,
                        'fragment_id': fragment_id,
                        'avg_distance': np.mean(distances),
                        'min_distance': np.min(distances),
                        'max_distance': np.max(distances),
                        'fragment_duration': flight_duration,
                        'start_frame': start_frame + i,
                        'end_frame': start_frame + j - 1,
                        'num_points': len(distances)
                    })
                    fragment_id += 1
            
            i = j
    
    return results


def extract_sex_from_filename(filename):
    """
    Extract sex category from tracking filename.
    
    Parameters
    ----------
    filename : str
        Tracking filename (e.g., 'forward_mosq_tracks_CagePenzanceFruM_251104_Dusk_160410_v05')
    
    Returns
    -------
    str
        Sex category: 'FruM', 'Female', 'Male', or 'Unknown'
    """
    filename_upper = filename.upper()
    
    if 'FRUM' in filename_upper:
        return 'FruM'
    elif 'F_' in filename_upper or 'F__' in filename_upper:
        return 'Female'
    elif 'M_' in filename_upper or 'M__' in filename_upper:
        return 'Male'
    else:
        return 'Unknown'


def aggregate_results_by_individual(fragment_results, sex):
    """
    Aggregate fragment-level results to individual mosquito level.
    
    Parameters
    ----------
    fragment_results : list of dict
        Fragment-level results from analyze_flight_center_distances()
    sex : str
        Sex category for this video
    
    Returns
    -------
    list of dict
        Individual-level aggregation with columns:
        - sex: sex category
        - mosquito_id: tracking ID
        - num_fragments: total flight fragments for this individual
        - avg_distance_mean: mean of fragment average distances
        - avg_distance_std: std dev across fragments
        - min_distance_overall: minimum distance across all fragments
        - max_distance_overall: maximum distance across all fragments
        - total_flight_frames: sum of all flight frames
    """
    
    results_by_individual = []
    
    # Group by mosquito_id
    mosquito_groups = {}
    for frag in fragment_results:
        mosq_id = frag['mosquito_id']
        if mosq_id not in mosquito_groups:
            mosquito_groups[mosq_id] = []
        mosquito_groups[mosq_id].append(frag)
    
    # Aggregate each mosquito
    for mosquito_id, fragments in mosquito_groups.items():
        avg_distances = [f['avg_distance'] for f in fragments]
        
        # Get min/max distances with fallback for mock data
        min_distances = [f.get('min_distance', f['avg_distance']) for f in fragments]
        max_distances = [f.get('max_distance', f['avg_distance']) for f in fragments]
        flight_durations = [f.get('fragment_duration', 0) for f in fragments]
        
        results_by_individual.append({
            'sex': sex,
            'mosquito_id': mosquito_id,
            'num_fragments': len(fragments),
            'avg_distance_mean': np.mean(avg_distances),
            'avg_distance_std': np.std(avg_distances),
            'min_distance_overall': min(min_distances),
            'max_distance_overall': max(max_distances),
            'total_flight_frames': sum(flight_durations)
        })
    
    return results_by_individual


def save_results_to_csv(fragment_results, output_path):
    """
    Save fragment-level results to CSV file.
    
    Parameters
    ----------
    fragment_results : list of dict
        Fragment-level results from analyze_flight_center_distances()
    output_path : str
        Path to output CSV file
    """
    df = pd.DataFrame(fragment_results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} fragment results to {output_path}")


def perform_statistical_comparison(fragment_results_list, video_names):
    """
    Perform ANOVA comparing avg_distance across sex categories.
    
    Parameters
    ----------
    fragment_results_list : list of list of dict
        Fragment results for each video
    video_names : list of str
        Video names for context
    
    Returns
    -------
    dict
        Statistical test results with keys:
        - f_statistic: F-value from ANOVA
        - p_value: p-value from ANOVA
        - group_means: dict of mean distances by sex
        - group_stds: dict of std devs by sex
    """
    
    # Flatten and organize by sex
    sex_groups = {}
    for video_results, video_name in zip(fragment_results_list, video_names):
        for frag in video_results:
            sex = frag.get('sex', 'Unknown')
            if sex not in sex_groups:
                sex_groups[sex] = []
            sex_groups[sex].append(frag['avg_distance'])
    
    # Perform ANOVA
    if len(sex_groups) < 2:
        return {
            'f_statistic': np.nan,
            'p_value': np.nan,
            'group_means': {sex: np.mean(vals) for sex, vals in sex_groups.items()},
            'group_stds': {sex: np.std(vals) for sex, vals in sex_groups.items()}
        }
    
    groups = list(sex_groups.values())
    f_stat, p_val = stats.f_oneway(*groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_val,
        'group_means': {sex: np.mean(vals) for sex, vals in sex_groups.items()},
        'group_stds': {sex: np.std(vals) for sex, vals in sex_groups.items()},
        'group_ns': {sex: len(vals) for sex, vals in sex_groups.items()}
    }
