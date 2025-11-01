# Exercise 1: Model Versioning and Registry

**Difficulty:** Medium  
**Time Limit:** 60 minutes  
**Focus:** Production ML infrastructure, model management

## Problem

Build a simple model registry system that can:
1. Store and version models
2. Track metadata (metrics, training date, features)
3. Promote models (dev → staging → production)
4. Rollback to previous versions

This is critical for the "Improvement Engineer" role which involves managing the ML lifecycle.

## Requirements

1. Create a `ModelRegistry` class with:
   - `register_model()` - Store model with metadata
   - `get_model()` - Retrieve model by version or stage
   - `promote_model()` - Move model between stages
   - `list_models()` - Query models by criteria

2. Track metadata:
   - Model version
   - Training timestamp
   - Performance metrics
   - Feature list
   - Model type and hyperparameters
   - Stage (dev, staging, production)

3. Support persistence (file-based or database)

## Solution Template

```python
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
        # Your implementation
        pass
    
    def promote_model(self, version: str, target_stage: ModelStage):
        """Promote model to target stage."""
        # Your implementation
        # Should demote previous model in that stage?
        pass
    
    def list_models(self, stage: Optional[ModelStage] = None,
                   min_metric: Optional[Dict[str, float]] = None) -> List[ModelMetadata]:
        """List models matching criteria"""
        # Your implementation
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
```

## Key Learning Points

1. **Model Lifecycle:** Managing versions and stages
2. **Metadata Tracking:** What information is important
3. **Production Practices:** How real ML systems work

## Design Considerations

- How to handle concurrent model deployments?
- Should you automatically demote previous models?
- How to handle model dependencies (feature pipelines)?
- Should registry support remote storage (S3, etc.)?

