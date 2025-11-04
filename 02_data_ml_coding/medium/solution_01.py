"""
Solution for Exercise 1: Model Versioning and Registry

This file contains the reference solution.
"""

import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ModelStage(Enum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ModelMetadata:
    """Model metadata container"""
    version: str
    stage: ModelStage
    trained_at: datetime
    metrics: Dict[str, float]
    features: List[str]
    model_type: str
    hyperparameters: Dict[str, Any]
    path: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['stage'] = self.stage.value
        data['trained_at'] = self.trained_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        data['stage'] = ModelStage(data['stage'])
        data['trained_at'] = datetime.fromisoformat(data['trained_at'])
        return cls(**data)


class ModelRegistry:
    """
    Model registry for versioning and lifecycle management.
    
    Stores models on disk and tracks metadata in JSON.
    """
    
    def __init__(self, registry_path: Path = Path("model_registry")):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.models: Dict[str, ModelMetadata] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, ModelMetadata]:
        """Load existing metadata"""
        if not self.metadata_file.exists():
            return {}
        
        with open(self.metadata_file, 'r') as f:
            data = json.load(f)
        
        return {version: ModelMetadata.from_dict(meta) 
                for version, meta in data.items()}
    
    def _save_metadata(self):
        """Save metadata to disk"""
        data = {version: meta.to_dict() 
                 for version, meta in self.models.items()}
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """
        Register a new model version.
        
        Args:
            model: The trained model object
            metadata: Model metadata
        
        Returns:
            Version string
        """
        # Save model to disk
        model_path = self.registry_path / f"model_v{metadata.version}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        metadata.path = str(model_path)
        
        # Store metadata
        self.models[metadata.version] = metadata
        self._save_metadata()
        
        return metadata.version
    
    def get_model(self, version: Optional[str] = None, 
                  stage: Optional[ModelStage] = None) -> tuple:
        """
        Retrieve model and metadata.
        
        Args:
            version: Specific version to retrieve
            stage: Retrieve current model in stage (dev/staging/production)
        
        Returns:
            (model, metadata) tuple
        """
        if version:
            if version not in self.models:
                raise ValueError(f"Model version {version} not found")
            metadata = self.models[version]
        elif stage:
            # Find latest model in stage
            models_in_stage = [m for v, m in self.models.items() if m.stage == stage]
            if not models_in_stage:
                raise ValueError(f"No model found in stage {stage.value}")
            # Get most recent by trained_at
            metadata = max(models_in_stage, key=lambda m: m.trained_at)
        else:
            raise ValueError("Must provide either version or stage")
        
        # Load model
        if not metadata.path or not Path(metadata.path).exists():
            raise ValueError(f"Model file not found: {metadata.path}")
        
        with open(metadata.path, 'rb') as f:
            model = pickle.load(f)
        
        return model, metadata
    
    def promote_model(self, version: str, target_stage: ModelStage):
        """
        Promote model to target stage.
        
        Args:
            version: Model version to promote
            target_stage: Target stage (dev/staging/production)
        """
        if version not in self.models:
            raise ValueError(f"Model version {version} not found")
        
        # Update stage
        self.models[version].stage = target_stage
        
        # Optionally demote previous model in that stage
        for v, m in self.models.items():
            if v != version and m.stage == target_stage:
                # Demote to dev if it was production/staging
                if target_stage == ModelStage.PRODUCTION:
                    m.stage = ModelStage.STAGING
                elif target_stage == ModelStage.STAGING:
                    m.stage = ModelStage.DEV
        
        self._save_metadata()
    
    def list_models(self, stage: Optional[ModelStage] = None,
                   min_metric: Optional[Dict[str, float]] = None) -> List[ModelMetadata]:
        """
        List models matching criteria.
        
        Args:
            stage: Filter by stage
            min_metric: Filter by minimum metric values (e.g., {'accuracy': 0.9})
        
        Returns:
            List of matching ModelMetadata objects
        """
        results = []
        
        for metadata in self.models.values():
            # Filter by stage
            if stage and metadata.stage != stage:
                continue
            
            # Filter by metrics
            if min_metric:
                match = all(
                    metadata.metrics.get(metric, 0) >= threshold
                    for metric, threshold in min_metric.items()
                )
                if not match:
                    continue
            
            results.append(metadata)
        
        return results

