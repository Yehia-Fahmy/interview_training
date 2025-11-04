"""
Exercise 1: Model Versioning and Registry

Build a simple model registry system that can:
1. Store and version models
2. Track metadata (metrics, training date, features)
3. Promote models (dev → staging → production)
4. Rollback to previous versions
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
        # TODO: Implement serialization
        # Convert ModelStage enum to string value
        # Convert datetime to ISO format string
        pass
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        # TODO: Implement deserialization
        # Convert string back to ModelStage enum
        # Convert ISO string back to datetime
        pass


class ModelRegistry:
    """
    Model registry for versioning and lifecycle management.
    
    Stores models on disk and tracks metadata in JSON.
    """
    
    def __init__(self, registry_path: Path = Path("model_registry")):
        self.registry_path = Path(registry_path)
        # TODO: Create registry directory if it doesn't exist
        self.metadata_file = self.registry_path / "metadata.json"
        self.models: Dict[str, ModelMetadata] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, ModelMetadata]:
        """Load existing metadata"""
        # TODO: Load metadata from JSON file if it exists
        # Return empty dict if file doesn't exist
        pass
    
    def _save_metadata(self):
        """Save metadata to disk"""
        # TODO: Save all metadata to JSON file
        pass
    
    def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """
        Register a new model version.
        
        Args:
            model: The trained model object
            metadata: Model metadata
        
        Returns:
            Version string
        """
        # TODO: Implement model registration
        # 1. Save model to disk as pickle file
        # 2. Update metadata with model path
        # 3. Store metadata in self.models dict
        # 4. Save metadata to disk
        # 5. Return version string
        pass
    
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
        # TODO: Implement model retrieval
        # If version provided, get that version
        # If stage provided, find latest model in that stage
        # Load model from disk using pickle
        # Return (model, metadata) tuple
        pass
    
    def promote_model(self, version: str, target_stage: ModelStage):
        """
        Promote model to target stage.
        
        Args:
            version: Model version to promote
            target_stage: Target stage (dev/staging/production)
        """
        # TODO: Implement model promotion
        # 1. Find model by version
        # 2. Update its stage to target_stage
        # 3. (Optional) Demote previous model in that stage
        # 4. Save updated metadata
        pass
    
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
        # TODO: Implement model listing
        # Filter by stage if provided
        # Filter by metrics if provided
        # Return list of matching models
        pass


# Usage example
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create registry
    registry = ModelRegistry()
    
    # Train a model
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Register model
    metadata = ModelMetadata(
        version="1.0.0",
        stage=ModelStage.DEV,
        trained_at=datetime.now(),
        metrics={"accuracy": 0.95, "f1": 0.93},
        features=[f"feature_{i}" for i in range(10)],
        model_type="RandomForest",
        hyperparameters={"n_estimators": 100}
    )
    
    version = registry.register_model(model, metadata)
    print(f"Registered model version: {version}")
    
    # Promote to production
    registry.promote_model(version, ModelStage.PRODUCTION)
    
    # Retrieve production model
    prod_model, prod_meta = registry.get_model(stage=ModelStage.PRODUCTION)
    print(f"Production model: {prod_meta.version}")

