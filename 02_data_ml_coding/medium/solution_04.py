"""
Solution for Exercise 4: ML Experiment Tracking System

This file contains the reference solution.
"""

import json
import pickle
import hashlib
import shutil
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
        self.tracking_dir.mkdir(exist_ok=True)
        self.current_run: Optional[RunInfo] = None
        self.runs: Dict[str, RunInfo] = {}
        self._load_runs()
    
    def _load_runs(self):
        """Load existing runs from disk"""
        metadata_file = self.tracking_dir / "runs_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                for run_id, run_data in data.items():
                    run_data['start_time'] = datetime.fromisoformat(run_data['start_time'])
                    run_data['end_time'] = datetime.fromisoformat(run_data['end_time']) if run_data['end_time'] else None
                    self.runs[run_id] = RunInfo(**run_data)
    
    def _save_runs(self):
        """Save runs metadata to disk"""
        metadata_file = self.tracking_dir / "runs_metadata.json"
        data = {}
        for run_id, run in self.runs.items():
            data[run_id] = {
                'run_id': run.run_id,
                'experiment_name': run.experiment_name,
                'status': run.status,
                'start_time': run.start_time.isoformat(),
                'end_time': run.end_time.isoformat() if run.end_time else None,
                'params': run.params,
                'metrics': run.metrics,
                'artifacts': run.artifacts,
                'tags': run.tags
            }
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
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
        run_id = self._generate_run_id()
        
        self.current_run = RunInfo(
            run_id=run_id,
            experiment_name=experiment_name,
            status='running',
            start_time=datetime.now(),
            end_time=None,
            params={},
            metrics={},
            artifacts=[],
            tags=tags or {}
        )
        
        return run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        self.current_run.params.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics (can be called multiple times)"""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        for metric_name, value in metrics.items():
            if metric_name not in self.current_run.metrics:
                self.current_run.metrics[metric_name] = []
            
            # If step provided, extend list to that step
            if step is not None:
                while len(self.current_run.metrics[metric_name]) <= step:
                    self.current_run.metrics[metric_name].append(None)
                self.current_run.metrics[metric_name][step] = value
            else:
                self.current_run.metrics[metric_name].append(value)
    
    def log_artifact(self, file_path: Path, artifact_name: Optional[str] = None):
        """Save an artifact (model, plot, etc.)"""
        if self.current_run is None:
            raise ValueError("No active run. Call start_run() first.")
        
        run_artifact_dir = self.tracking_dir / self.current_run.run_id / "artifacts"
        run_artifact_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_name = artifact_name or file_path.name
        dest_path = run_artifact_dir / artifact_name
        
        # Copy file
        shutil.copy(file_path, dest_path)
        
        self.current_run.artifacts.append(str(dest_path.relative_to(self.tracking_dir)))
    
    def end_run(self, status: str = 'completed'):
        """End the current run"""
        if self.current_run is None:
            raise ValueError("No active run.")
        
        self.current_run.status = status
        self.current_run.end_time = datetime.now()
        
        # Save run
        self.runs[self.current_run.run_id] = self.current_run
        self._save_run(self.current_run)
        self._save_runs()
        
        self.current_run = None
    
    def _save_run(self, run: RunInfo):
        """Save individual run to disk"""
        run_dir = self.tracking_dir / run.run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_file = run_dir / "run_info.json"
        with open(metadata_file, 'w') as f:
            data = asdict(run)
            data['start_time'] = run.start_time.isoformat()
            data['end_time'] = run.end_time.isoformat() if run.end_time else None
            json.dump(data, f, indent=2)
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Retrieve a run by ID"""
        return self.runs.get(run_id)
    
    def search_runs(self, experiment_name: Optional[str] = None,
                   filter_params: Optional[Dict] = None) -> List[RunInfo]:
        """Search runs by criteria"""
        results = []
        
        for run in self.runs.values():
            if experiment_name and run.experiment_name != experiment_name:
                continue
            
            if filter_params:
                # Check if run params match filter
                match = all(run.params.get(k) == v 
                           for k, v in filter_params.items())
                if not match:
                    continue
            
            results.append(run)
        
        return results
    
    def compare_runs(self, run_ids: List[str]) -> Dict:
        """Compare multiple runs"""
        runs = [self.runs[rid] for rid in run_ids if rid in self.runs]
        
        comparison = {
            'run_ids': run_ids,
            'params': {},
            'final_metrics': {}
        }
        
        for run in runs:
            # Compare params
            for param, value in run.params.items():
                if param not in comparison['params']:
                    comparison['params'][param] = {}
                comparison['params'][param][run.run_id] = value
            
            # Compare final metrics
            for metric, values in run.metrics.items():
                if metric not in comparison['final_metrics']:
                    comparison['final_metrics'][metric] = {}
                final_value = values[-1] if values else None
                comparison['final_metrics'][metric][run.run_id] = final_value
        
        return comparison

