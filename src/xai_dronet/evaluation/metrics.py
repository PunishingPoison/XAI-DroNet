import numpy as np
from typing import List, Tuple

def calculate_collision_rate(collisions: int, total_steps: int) -> float:
    """
    Calculates the collision rate based on number of collisions per total steps.
    Returns 0.0 if total_steps is 0.
    """
    if total_steps == 0:
        return 0.0
    return float(collisions) / float(total_steps)

def calculate_total_distance(positions: List[Tuple[float, float, float]]) -> float:
    """
    Calculates the total Euclidean distance traveled based on a list of 3D positions.
    """
    if len(positions) < 2:
        return 0.0
        
    pos_array = np.array(positions)
    diffs = np.diff(pos_array, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))

def calculate_steering_variance(steering_angles: List[float]) -> float:
    """
    Calculates the variance of steering angles to measure flight smoothness.
    """
    if len(steering_angles) < 2:
        return 0.0
    return float(np.var(steering_angles))

def calculate_attention_consistency(heatmaps: List[np.ndarray]) -> float:
    """
    Calculates the temporal consistency of Grad-CAM heatmaps.
    A lower variance between consecutive frames indicates higher consistency.
    Returns a score where 1.0 is perfectly consistent and 0.0 is completely inconsistent.
    """
    if len(heatmaps) < 2:
        return 1.0
        
    diffs = []
    for i in range(1, len(heatmaps)):
        # Normalize heatmaps to 0-1 range for stable comparison
        prev = heatmaps[i-1] / (np.max(heatmaps[i-1]) + 1e-8)
        curr = heatmaps[i] / (np.max(heatmaps[i]) + 1e-8)
        
        # Mean Absolute Error between consecutive heatmaps
        mae = np.mean(np.abs(curr - prev))
        diffs.append(mae)
        
    avg_diff = np.mean(diffs)
    # Convert MAE to a consistency score [0, 1]
    consistency_score = max(0.0, 1.0 - avg_diff)
    return float(consistency_score)
