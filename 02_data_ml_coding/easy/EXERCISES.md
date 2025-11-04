# Data/ML Coding - Easy Exercises

These exercises focus on fundamental ML concepts, model implementation, and basic evaluation.

---

## Exercise 1: Classification Model from Scratch

**Difficulty:** Easy  
**Time Limit:** 45 minutes (with AI assistance allowed)  
**Focus:** Implementing ML algorithms, code quality, design rationale

### Problem

Implement a logistic regression classifier from scratch (without using sklearn's LogisticRegression, but you can use NumPy). The focus is on:

1. **Clean, readable code** with proper structure
2. **Explaining design choices**
3. **Proper documentation**

### Requirements

1. Implement logistic regression with:
   - Gradient descent optimization
   - L2 regularization (optional)
   - Convergence detection
   
2. Structure your code with:
   - Clear class structure
   - Well-documented methods
   - Separation of concerns

3. Include:
   - Training method
   - Prediction method
   - Probability estimation

### Solution Template

```python
import numpy as np
from typing import Optional

class LogisticRegression:
    """
    Logistic Regression classifier from scratch.
    
    This implementation uses gradient descent for optimization.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 regularization: float = 0.0):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            max_iter: Maximum number of iterations
            regularization: L2 regularization strength
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid activation function."""
        # Your implementation
        pass
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        # Your implementation
        pass
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Compute gradients for weights and bias."""
        # Your implementation
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,), binary {0, 1}
        
        Returns:
            self for method chaining
        """
        # Initialize weights
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Compute gradients and update
            # Your implementation
            
            # Track loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Check convergence (optional)
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                    break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates."""
        # Your implementation
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        # Your implementation
        pass

# Test your implementation
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")
```

### Key Learning Points

1. **Understanding ML Fundamentals:** Implement core algorithm logic
2. **Code Organization:** Classes, methods, documentation
3. **Design Decisions:** Explain why you made certain choices

### Design Considerations to Explain

- Why use sigmoid for binary classification?
- Why gradient descent vs other optimizers?
- How does learning rate affect convergence?
- Why include regularization?

---

## Exercise 2: Model Evaluation Metrics

**Difficulty:** Easy  
**Time Limit:** 30 minutes  
**Focus:** Understanding evaluation metrics, implementation quality

### Problem

Implement comprehensive evaluation metrics for classification models. Create a clean, reusable evaluation module.

### Requirements

1. Implement the following metrics from scratch:
   - Accuracy
   - Precision (per class and macro-averaged)
   - Recall (per class and macro-averaged)
   - F1-score (per class and macro-averaged)
   - Confusion matrix
   - ROC-AUC (for binary classification)

2. Create a class that:
   - Computes all metrics
   - Formats results nicely
   - Handles binary and multiclass scenarios

3. Write unit tests to validate your implementation

### Solution Template

```python
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator.
    
    This implementation focuses on clarity and correctness over using
    sklearn directly, though sklearn can be used for validation.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: Optional[np.ndarray] = None):
        """
        Initialize with true labels and predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = y_proba if y_proba is not None else None
        self.classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        self.n_classes = len(self.classes)
    
    def confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix."""
        # Your implementation from scratch
        pass
    
    def accuracy(self) -> float:
        """Compute accuracy."""
        # Your implementation
        pass
    
    def precision_per_class(self) -> Dict[int, float]:
        """Compute precision for each class."""
        # Your implementation
        pass
    
    def recall_per_class(self) -> Dict[int, float]:
        """Compute recall for each class."""
        # Your implementation
        pass
    
    def f1_per_class(self) -> Dict[int, float]:
        """Compute F1-score for each class."""
        # Your implementation
        pass
    
    def macro_averaged_metrics(self) -> Dict[str, float]:
        """Compute macro-averaged precision, recall, F1."""
        # Your implementation
        pass
    
    def roc_auc(self) -> Optional[float]:
        """Compute ROC-AUC (binary classification only)."""
        if self.n_classes != 2 or self.y_proba is None:
            return None
        # Your implementation
        pass
    
    def summary(self) -> Dict:
        """Return comprehensive metrics summary."""
        summary = {
            'accuracy': self.accuracy(),
            'per_class': {
                'precision': self.precision_per_class(),
                'recall': self.recall_per_class(),
                'f1': self.f1_per_class()
            },
            'macro_averaged': self.macro_averaged_metrics(),
            'confusion_matrix': self.confusion_matrix().tolist()
        }
        
        if self.n_classes == 2:
            roc_auc = self.roc_auc()
            if roc_auc is not None:
                summary['roc_auc'] = roc_auc
        
        return summary

# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
    
    # Evaluate
    metrics = ClassificationMetrics(y_test, y_pred, y_proba)
    summary = metrics.summary()
    
    print("Classification Metrics Summary:")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro-averaged F1: {summary['macro_averaged']['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix())
```

### Key Learning Points

1. **Metric Understanding:** Know what each metric means
2. **Implementation Quality:** Clean, tested code
3. **Edge Cases:** Handle binary vs multiclass, missing probabilities

### Design Considerations

- Why implement from scratch vs using sklearn? (Understanding)
- How to handle edge cases (single class, perfect predictions)?
- Should metrics handle class imbalance?

---

## Exercise 3: Feature Engineering Pipeline

**Difficulty:** Easy  
**Time Limit:** 40 minutes  
**Focus:** Clean pipeline design, feature engineering best practices

### Problem

Create a reusable feature engineering pipeline that handles:
1. Missing value imputation
2. Categorical encoding
3. Numerical scaling
4. Feature selection (optional)

The pipeline should be:
- Easy to use (sklearn-style fit/transform)
- Well-documented
- Testable

### Requirements

1. Create a `FeaturePipeline` class with:
   - `fit()` method to learn transformations
   - `transform()` method to apply transformations
   - Support for both numerical and categorical features

2. Implement common transformations:
   - Numerical: Standardization, normalization, imputation
   - Categorical: One-hot encoding, label encoding, target encoding

3. Write example usage with real-world data

### Solution Template

```python
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class FeaturePipeline:
    """
    Reusable feature engineering pipeline.
    
    Handles missing values, encoding, and scaling in a clean pipeline.
    """
    
    def __init__(self, 
                 numerical_features: List[str],
                 categorical_features: List[str],
                 imputation_strategy: str = 'mean',
                 scaling: bool = True):
        """
        Initialize feature pipeline.
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            imputation_strategy: Strategy for missing values ('mean', 'median', 'mode')
            scaling: Whether to scale numerical features
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.imputation_strategy = imputation_strategy
        self.scaling = scaling
        
        # Initialize transformers
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoders = {}  # One encoder per categorical feature
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features (DataFrame)
            y: Training labels (optional, for target encoding)
        """
        # Fit numerical imputer
        if self.numerical_features:
            self.numerical_imputer = SimpleImputer(strategy=self.imputation_strategy)
            self.numerical_imputer.fit(X[self.numerical_features])
            
            # Fit scaler
            if self.scaling:
                self.scaler = StandardScaler()
                imputed = self.numerical_imputer.transform(X[self.numerical_features])
                self.scaler.fit(imputed)
        
        # Fit categorical imputer and encoders
        if self.categorical_features:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.categorical_imputer.fit(X[self.categorical_features])
            
            for col in self.categorical_features:
                self.encoders[col] = LabelEncoder()
                imputed_col = self.categorical_imputer.transform(X[[col]])[:, 0]
                # Only fit on non-null values
                non_null_mask = pd.notna(X[col])
                if non_null_mask.any():
                    self.encoders[col].fit(imputed_col[non_null_mask])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform (DataFrame)
        
        Returns:
            Transformed features (numpy array)
        """
        transformed_parts = []
        
        # Transform numerical features
        if self.numerical_features:
            imputed = self.numerical_imputer.transform(X[self.numerical_features])
            if self.scaling:
                imputed = self.scaler.transform(imputed)
            transformed_parts.append(imputed)
        
        # Transform categorical features
        if self.categorical_features:
            imputed = self.categorical_imputer.transform(X[self.categorical_features])
            
            # Encode each categorical feature
            encoded_features = []
            for i, col in enumerate(self.categorical_features):
                encoded = self.encoders[col].transform(imputed[:, i])
                # One-hot encode
                n_classes = len(self.encoders[col].classes_)
                one_hot = np.zeros((len(encoded), n_classes))
                one_hot[np.arange(len(encoded)), encoded] = 1
                encoded_features.append(one_hot)
            
            if encoded_features:
                transformed_parts.append(np.hstack(encoded_features))
        
        return np.hstack(transformed_parts) if transformed_parts else np.array([])

# Test
if __name__ == "__main__":
    # Create sample data with missing values
    data = pd.DataFrame({
        'age': [25, 30, None, 35, 40, None],
        'salary': [50000, 60000, 70000, None, 80000, 90000],
        'city': ['NYC', 'SF', None, 'NYC', 'SF', 'NYC'],
        'department': ['Engineering', 'Sales', 'Engineering', None, 'Sales', 'Engineering']
    })
    
    pipeline = FeaturePipeline(
        numerical_features=['age', 'salary'],
        categorical_features=['city', 'department'],
        scaling=True
    )
    
    pipeline.fit(data)
    transformed = pipeline.transform(data)
    print("Transformed shape:", transformed.shape)
    print("Sample transformed features:\n", transformed[:3])
```

### Key Learning Points

1. **Pipeline Design:** sklearn-style API for consistency
2. **Separation of Concerns:** Fit learns, transform applies
3. **Documentation:** Clear docstrings and type hints

### Design Considerations

- Why separate fit and transform?
- How to handle unseen categories in transform?
- Should you support different encoding strategies?

---

## Exercise 4: End-to-End ML Pipeline

**Difficulty:** Easy  
**Time Limit:** 50 minutes  
**Focus:** Complete ML workflow, code organization

### Problem

Create a complete, production-ready ML pipeline that:
1. Loads and splits data
2. Engineers features
3. Trains a model
4. Evaluates performance
5. Saves the model

The code should be:
- Well-organized (separate functions/classes)
- Documented
- Easy to modify and extend

### Requirements

1. Create a pipeline class that orchestrates the entire workflow
2. Support for different model types (configurable)
3. Save/load functionality for models and pipelines
4. Comprehensive logging and progress tracking

### Solution Template

```python
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json

class MLPipeline:
    """
    End-to-end machine learning pipeline.
    
    Handles data loading, preprocessing, training, evaluation, and model saving.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary with model settings, paths, etc.
        """
        self.config = config
        self.model = None
        self.feature_pipeline = None  # From previous exercise
        self.train_history = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        # Your implementation - support CSV, JSON, etc.
        pass
    
    def split_data(self, df: pd.DataFrame, target_col: str,
                   test_size: float = 0.2) -> tuple:
        """Split data into train/test sets."""
        # Your implementation
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Training history/metrics
        """
        # Your implementation
        # 1. Engineer features
        # 2. Train model
        # 3. Evaluate on validation set if provided
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on given data."""
        # Your implementation
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        # Your implementation
        pass
    
    def save(self, model_dir: Path):
        """Save model and pipeline to disk."""
        # Save model
        # Save feature pipeline
        # Save config
        pass
    
    @classmethod
    def load(cls, model_dir: Path) -> 'MLPipeline':
        """Load saved pipeline."""
        # Load model, pipeline, config
        pass

# Example usage
if __name__ == "__main__":
    config = {
        'model_type': 'random_forest',
        'model_params': {'n_estimators': 100, 'random_state': 42},
        'test_size': 0.2,
        'random_state': 42
    }
    
    pipeline = MLPipeline(config)
    
    # Load and split data
    df = pipeline.load_data('data.csv')
    X_train, X_test, y_train, y_test = pipeline.split_data(df, 'target')
    
    # Train
    history = pipeline.train(X_train, y_train)
    
    # Evaluate
    metrics = pipeline.evaluate(X_test, y_test)
    print("Test Metrics:", metrics)
    
    # Save
    pipeline.save(Path('models/my_model'))
    
    # Load later
    loaded_pipeline = MLPipeline.load(Path('models/my_model'))
```

### Key Learning Points

1. **Pipeline Architecture:** Organizing ML workflows
2. **Configuration Management:** Externalizing settings
3. **Model Persistence:** Saving and loading models
4. **Code Reusability:** Making components reusable

### Design Considerations

- How to structure configuration?
- Should preprocessing be part of pipeline or separate?
- How to version models and pipelines?
- What metadata should be saved?

