"""
Exercise 4: ML Experiment Tracking System

Build a simple experiment tracking system similar to MLflow. Track:
- Hyperparameters
- Metrics
- Artifacts (models, plots)
- Code versions
- Environment info
"""

import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class RunInfo:
    """Information about a single experiment run"""
    run_id: str
    experiment_name: str
    status: str  # 'running', 'completed', 'failed'
    start_time: datetime
    end_time: Optional[datetime]
    params: Dict[str, Any]
    metrics: Dict[str, List[float]]  # Metric name -> list of values over time
    artifacts: List[str]  # Paths to artifact files
    tags: Dict[str, str]


class ExperimentTracker:
    """
    Experiment tracking system for ML experiments.
    
    Tracks parameters, metrics, and artifacts for reproducibility.
    """
    
    def __init__(self, tracking_dir: Path = Path("experiments")):
        self.tracking_dir = Path(tracking_dir)
        # TODO: Create tracking directory if it doesn't exist
        self.current_run: Optional[RunInfo] = None
        self.runs: Dict[str, RunInfo] = {}
        self._load_runs()
    
    def _load_runs(self):
        """Load existing runs from disk"""
        # TODO: Load runs from metadata file if it exists
        pass
    
    def _save_runs(self):
        """Save runs metadata to disk"""
        # TODO: Save all runs metadata to JSON file
        pass
    
    def start_run(self, experiment_name: str, 
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new experiment run.
        
        Args:
            experiment_name: Name of the experiment
            tags: Optional tags for the run
        
        Returns:
            Run ID
        """
        # TODO: Generate unique run ID
        # Create RunInfo object
        # Set as current_run
        # Return run_id
        pass
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        # TODO: Update current_run.params with new params
        # Raise error if no active run
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics (can be called multiple times)"""
        # TODO: Update current_run.metrics
        # Handle step-based logging (for time series metrics)
        # Raise error if no active run
        pass
    
    def log_artifact(self, file_path: Path, artifact_name: Optional[str] = None):
        """Save an artifact (model, plot, etc.)"""
        # TODO: Copy file to run's artifact directory
        # Update current_run.artifacts
        # Raise error if no active run
        pass
    
    def end_run(self, status: str = 'completed'):
        """End the current run"""
        # TODO: Update run status and end_time
        # Save run to disk
        # Clear current_run
        pass
    
    def _save_run(self, run: RunInfo):
        """Save individual run to disk"""
        # TODO: Create run directory
        # Save run metadata as JSON
        pass
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        # TODO: Generate unique ID (e.g., using timestamp + hash)
        pass
    
    def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Retrieve a run by ID"""
        # TODO: Return run from self.runs dict
        pass
    
    def search_runs(self, experiment_name: Optional[str] = None,
                   filter_params: Optional[Dict] = None) -> List[RunInfo]:
        """Search runs by criteria"""
        # TODO: Filter runs by experiment_name and/or params
        # Return matching runs
        pass
    
    def compare_runs(self, run_ids: List[str]) -> Dict:
        """Compare multiple runs"""
        # TODO: Compare parameters and metrics across runs
        # Return comparison dictionary
        pass


# Usage example
if __name__ == "__main__":
    tracker = ExperimentTracker()
    
    # Start experiment
    run_id = tracker.start_run("hyperparameter_tuning", tags={"model": "random_forest"})
    
    # Log parameters
    tracker.log_params({
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1
    })
    
    # Simulate training (log metrics over time)
    for epoch in range(10):
        tracker.log_metrics({
            'accuracy': 0.85 + epoch * 0.01,
            'loss': 0.5 - epoch * 0.05
        }, step=epoch)
    
    # End run
    tracker.end_run()
    
    # Search runs
    runs = tracker.search_runs(experiment_name="hyperparameter_tuning")
    print(f"Found {len(runs)} runs")
    
    # Compare runs
    if len(runs) > 1:
        comparison = tracker.compare_runs([r.run_id for r in runs[:2]])
        print("\nComparison:")
        print(json.dumps(comparison, indent=2))

