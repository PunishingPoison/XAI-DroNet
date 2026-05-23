import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

from xai_dronet.evaluation.metrics import (
    calculate_collision_rate,
    calculate_total_distance,
    calculate_steering_variance,
    calculate_attention_consistency
)

class DroneEvaluator:
    """
    Evaluator class to track and calculate metrics over a simulation episode.
    """
    def __init__(self, environment_name: str = "Unknown"):
        self.environment_name = environment_name
        self.reset()
        
    def reset(self):
        self.total_steps = 0
        self.collisions = 0
        self.positions: List[Tuple[float, float, float]] = []
        self.steering_angles: List[float] = []
        self.heatmaps: List[np.ndarray] = []
        self.frame_times: List[float] = [] # Optional for FPS calculation
        
    def log_step(
        self, 
        position: Tuple[float, float, float],
        steering_angle: float,
        has_collided: bool,
        heatmap: np.ndarray = None,
        frame_time: float = None
    ):
        """
        Log data for a single step in the simulation.
        """
        self.total_steps += 1
        self.positions.append(position)
        self.steering_angles.append(steering_angle)
        
        if has_collided:
            self.collisions += 1
            
        if heatmap is not None:
            self.heatmaps.append(heatmap)
            
        if frame_time is not None:
            self.frame_times.append(frame_time)
            
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Computes all metrics based on the logged data.
        """
        metrics = {
            "environment": self.environment_name,
            "total_steps": self.total_steps,
            "collisions": self.collisions,
            "collision_rate": calculate_collision_rate(self.collisions, self.total_steps),
            "total_distance": calculate_total_distance(self.positions),
            "steering_variance": calculate_steering_variance(self.steering_angles),
            "attention_consistency": calculate_attention_consistency(self.heatmaps)
        }
        
        if len(self.frame_times) > 1:
            total_time = sum(self.frame_times)
            metrics["avg_fps"] = self.total_steps / total_time if total_time > 0 else 0.0
        else:
            metrics["avg_fps"] = 0.0
            
        return metrics

    def save_report(self, output_path: str):
        """
        Save the computed metrics to a JSON file.
        """
        metrics = self.compute_metrics()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Evaluation report saved to {output_path}")
