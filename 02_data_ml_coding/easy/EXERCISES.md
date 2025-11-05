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

---

## Exercise 5: Data Cleaning and Preprocessing with Pandas

**Difficulty:** Easy  
**Time Limit:** 45 minutes  
**Focus:** Pandas data manipulation, cleaning, and preprocessing

### Problem

Given a messy dataset with various data quality issues, create a robust data cleaning function that handles:
1. Missing values detection and handling
2. Duplicate detection and removal
3. Data type corrections
4. Outlier detection and treatment
5. String normalization
6. Date parsing and validation

The function should return a cleaned DataFrame and a summary report of all cleaning operations performed.

### Requirements

1. Create a `DataCleaner` class with:
   - `clean()` method that performs all cleaning operations
   - `get_report()` method that returns a summary of cleaning operations
   - Configurable cleaning strategies

2. Handle common data quality issues:
   - Missing values (NaN, None, empty strings, 'N/A', etc.)
   - Duplicate rows
   - Incorrect data types
   - Outliers in numerical columns
   - Inconsistent string formatting
   - Invalid date formats

3. Make the cleaning process robust and configurable

### Solution Template

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

class DataCleaner:
    """
    Comprehensive data cleaning utility for pandas DataFrames.
    
    Handles missing values, duplicates, data types, outliers, and string normalization.
    """
    
    def __init__(self, 
                 handle_missing: str = 'drop',
                 handle_duplicates: bool = True,
                 detect_outliers: bool = True,
                 normalize_strings: bool = True):
        """
        Initialize data cleaner.
        
        Args:
            handle_missing: Strategy for missing values ('drop', 'forward_fill', 'backward_fill', 'mean', 'median', 'mode')
            handle_duplicates: Whether to remove duplicate rows
            detect_outliers: Whether to detect and handle outliers
            normalize_strings: Whether to normalize string columns
        """
        self.handle_missing = handle_missing
        self.handle_duplicates = handle_duplicates
        self.detect_outliers = detect_outliers
        self.normalize_strings = normalize_strings
        self.report = {}
    
    def clean(self, df: pd.DataFrame, 
              date_columns: Optional[List[str]] = None,
              numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean the dataframe.
        
        Args:
            df: Input dataframe to clean
            date_columns: List of column names that should be dates
            numeric_columns: List of column names that should be numeric
        
        Returns:
            Cleaned dataframe
        """
        df_cleaned = df.copy()
        self.report = {'original_shape': df.shape}
        
        # TODO: Implement cleaning steps
        # 1. Normalize missing value representations
        # 2. Handle missing values according to strategy
        # 3. Remove duplicates if enabled
        # 4. Fix data types (dates, numeric)
        # 5. Detect and handle outliers
        # 6. Normalize strings
        
        self.report['final_shape'] = df_cleaned.shape
        return df_cleaned
    
    def get_report(self) -> Dict:
        """Return summary report of cleaning operations."""
        return self.report

# Test
if __name__ == "__main__":
    # Create messy sample data
    messy_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, None, 5],
        'name': ['Alice', 'Bob', 'bob', 'Charlie', 'DAVE', '', 'Eve'],
        'age': [25, 30, 30, None, 150, 28, 35],
        'salary': [50000, '60000', 60000, None, 80000, 55000, 'invalid'],
        'date_joined': ['2020-01-15', '2020/02/20', 'invalid', '2020-03-10', None, '2020-04-01', '2020-05-15']
    })
    
    cleaner = DataCleaner(handle_missing='mean', handle_duplicates=True)
    cleaned = cleaner.clean(messy_data, 
                            date_columns=['date_joined'],
                            numeric_columns=['age', 'salary'])
    
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\nCleaning Report:")
    print(cleaner.get_report())
```

### Key Learning Points

1. **Pandas Proficiency:** Mastering DataFrame operations
2. **Data Quality:** Understanding common data issues
3. **Robustness:** Handling edge cases and errors gracefully

---

## Exercise 6: Advanced Feature Engineering with Pandas and Scikit-learn

**Difficulty:** Easy  
**Time Limit:** 50 minutes  
**Focus:** Feature engineering, pandas groupby, aggregations, sklearn transformers

### Problem

Create a feature engineering module that generates advanced features from raw data:
1. Temporal features (extract from dates)
2. Aggregated features (groupby operations)
3. Interaction features
4. Binning and discretization
5. Polynomial features
6. Target encoding for categorical variables

### Requirements

1. Create a `FeatureEngineer` class that:
   - Generates temporal features from date columns
   - Creates aggregated features using groupby
   - Generates interaction and polynomial features
   - Performs binning and discretization
   - Implements target encoding

2. Use both pandas and scikit-learn transformers appropriately

3. Ensure features are properly handled for train/test splits

### Solution Template

```python
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline.
    
    Generates temporal, aggregated, interaction, and encoded features.
    """
    
    def __init__(self,
                 date_columns: Optional[List[str]] = None,
                 groupby_columns: Optional[List[str]] = None,
                 aggregate_columns: Optional[List[str]] = None,
                 create_interactions: bool = True,
                 create_polynomials: bool = False,
                 n_bins: int = 5):
        """
        Initialize feature engineer.
        
        Args:
            date_columns: Columns to extract temporal features from
            groupby_columns: Columns to use for groupby aggregations
            aggregate_columns: Columns to aggregate
            create_interactions: Whether to create interaction features
            create_polynomials: Whether to create polynomial features
            n_bins: Number of bins for discretization
        """
        self.date_columns = date_columns or []
        self.groupby_columns = groupby_columns or []
        self.aggregate_columns = aggregate_columns or []
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.n_bins = n_bins
        
        self.polynomial_transformer = None
        self.discretizer = None
        self.target_encodings = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the feature engineer on training data.
        
        Args:
            X: Training features
            y: Training target (optional, needed for target encoding)
        """
        # TODO: Fit transformers and compute encodings
        # 1. Fit polynomial transformer if enabled
        # 2. Fit discretizer if binning is needed
        # 3. Compute target encodings if y is provided
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        """
        X_transformed = X.copy()
        
        # TODO: Apply all transformations
        # 1. Extract temporal features
        # 2. Add aggregated features
        # 3. Create interaction features
        # 4. Apply polynomial transformation
        # 5. Apply discretization
        # 6. Apply target encoding
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

# Test
if __name__ == "__main__":
    # Create sample data
    data = pd.DataFrame({
        'transaction_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'customer_id': np.random.randint(1, 10, 100),
        'product_category': np.random.choice(['A', 'B', 'C'], 100),
        'amount': np.random.uniform(10, 1000, 100),
        'quantity': np.random.randint(1, 10, 100)
    })
    
    target = pd.Series(np.random.randint(0, 2, 100))
    
    engineer = FeatureEngineer(
        date_columns=['transaction_date'],
        groupby_columns=['customer_id'],
        aggregate_columns=['amount'],
        create_interactions=True
    )
    
    features = engineer.fit_transform(data, target)
    print(f"Original shape: {data.shape}")
    print(f"Engineered shape: {features.shape}")
    print(f"\nNew columns: {set(features.columns) - set(data.columns)}")
```

### Key Learning Points

1. **Feature Engineering:** Creating meaningful features from raw data
2. **Pandas Groupby:** Advanced aggregations and transformations
3. **Scikit-learn Integration:** Using transformers correctly

---

## Exercise 7: Model Training Pipeline with Scikit-learn

**Difficulty:** Easy  
**Time Limit:** 40 minutes  
**Focus:** Scikit-learn pipelines, model selection, evaluation

### Problem

Create a comprehensive model training pipeline using scikit-learn that:
1. Supports multiple model types (classification and regression)
2. Uses sklearn Pipeline for preprocessing and modeling
3. Implements proper train/validation/test splits
4. Evaluates models with appropriate metrics
5. Handles both classification and regression tasks

### Requirements

1. Create a `ModelTrainer` class that:
   - Accepts different model types (RandomForest, GradientBoosting, LogisticRegression, etc.)
   - Uses sklearn Pipeline for end-to-end modeling
   - Implements proper data splitting
   - Evaluates with appropriate metrics
   - Supports both classification and regression

2. Use sklearn's Pipeline, GridSearchCV, and evaluation metrics

3. Return training history and evaluation results

### Solution Template

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

class ModelTrainer:
    """
    Comprehensive model training pipeline using scikit-learn.
    
    Supports both classification and regression tasks.
    """
    
    def __init__(self, 
                 task_type: str = 'classification',
                 model_type: str = 'random_forest',
                 model_params: Optional[Dict] = None,
                 use_scaling: bool = True):
        """
        Initialize model trainer.
        
        Args:
            task_type: 'classification' or 'regression'
            model_type: Type of model to use
            model_params: Hyperparameters for the model
            use_scaling: Whether to scale features
        """
        self.task_type = task_type
        self.model_type = model_type
        self.model_params = model_params or {}
        self.use_scaling = use_scaling
        
        self.model = None
        self.pipeline = None
        self.history = {}
    
    def _create_model(self):
        """Create the model based on task type and model type."""
        # TODO: Create appropriate model based on task_type and model_type
        # Classification: RandomForestClassifier, LogisticRegression
        # Regression: RandomForestRegressor, LinearRegression
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2, 
              val_size: float = 0.2,
              random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model with proper train/val/test splits.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            val_size: Proportion of validation set (from training set)
            random_state: Random seed
        
        Returns:
            Dictionary with training history and metrics
        """
        # TODO: Implement training pipeline
        # 1. Split data (train/test, then train/val)
        # 2. Create preprocessing pipeline
        # 3. Create model
        # 4. Train model
        # 5. Evaluate on validation and test sets
        # 6. Return metrics
        
        return self.history
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on given data."""
        # TODO: Make predictions and compute appropriate metrics
        # Classification: accuracy, precision, recall, f1
        # Regression: mse, mae, r2
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.pipeline.predict(X)

# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    
    # Test classification
    X_clf, y_clf = make_classification(n_samples=1000, n_features=10, 
                                       n_informative=5, random_state=42)
    X_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(10)])
    
    trainer_clf = ModelTrainer(task_type='classification', 
                               model_type='random_forest',
                               model_params={'n_estimators': 100, 'random_state': 42})
    results_clf = trainer_clf.train(X_clf, pd.Series(y_clf))
    
    print("Classification Results:")
    print(results_clf)
    
    # Test regression
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                                   n_informative=5, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
    
    trainer_reg = ModelTrainer(task_type='regression',
                               model_type='random_forest',
                               model_params={'n_estimators': 100, 'random_state': 42})
    results_reg = trainer_reg.train(X_reg, pd.Series(y_reg))
    
    print("\nRegression Results:")
    print(results_reg)
```

### Key Learning Points

1. **Scikit-learn Pipelines:** Chaining preprocessing and modeling
2. **Model Selection:** Choosing appropriate models for tasks
3. **Evaluation:** Using correct metrics for classification vs regression

---

## Exercise 8: Cross-validation and Hyperparameter Tuning

**Difficulty:** Easy  
**Time Limit:** 45 minutes  
**Focus:** Cross-validation, hyperparameter tuning, model selection

### Problem

Implement a comprehensive hyperparameter tuning system that:
1. Performs cross-validation for model evaluation
2. Implements grid search and random search
3. Handles nested cross-validation
4. Tracks and compares multiple models
5. Returns best model and hyperparameters

### Requirements

1. Create a `HyperparameterTuner` class that:
   - Implements k-fold cross-validation
   - Supports grid search and random search
   - Handles both classification and regression
   - Tracks multiple models and their performance
   - Returns best model configuration

2. Use sklearn's GridSearchCV, RandomizedSearchCV, and cross_val_score

3. Include proper handling of preprocessing in cross-validation

### Solution Template

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning with cross-validation.
    
    Supports grid search, random search, and model comparison.
    """
    
    def __init__(self,
                 task_type: str = 'classification',
                 cv_folds: int = 5,
                 scoring: Optional[str] = None,
                 random_state: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            task_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric (None for default)
            random_state: Random seed
        """
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_model = None
        self.best_params = None
        self.cv_results = {}
    
    def grid_search(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   model,
                   param_grid: Dict[str, List],
                   use_pipeline: bool = True) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            model: Model instance or class
            param_grid: Dictionary of parameter grids
            use_pipeline: Whether to use pipeline with preprocessing
        
        Returns:
            Dictionary with best model and results
        """
        # TODO: Implement grid search
        # 1. Create pipeline if needed
        # 2. Set up GridSearchCV
        # 3. Fit and evaluate
        # 4. Return best model and results
        pass
    
    def random_search(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     model,
                     param_distributions: Dict[str, List],
                     n_iter: int = 50,
                     use_pipeline: bool = True) -> Dict[str, Any]:
        """
        Perform random search for hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            model: Model instance or class
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of iterations
            use_pipeline: Whether to use pipeline with preprocessing
        
        Returns:
            Dictionary with best model and results
        """
        # TODO: Implement random search
        # Similar to grid_search but using RandomizedSearchCV
        pass
    
    def compare_models(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      models: Dict[str, Any],
                      use_pipeline: bool = True) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Args:
            X: Features
            y: Target
            models: Dictionary of model_name -> (model_class, param_grid)
            use_pipeline: Whether to use pipeline with preprocessing
        
        Returns:
            DataFrame with comparison results
        """
        # TODO: Compare multiple models
        # 1. For each model, perform cross-validation
        # 2. Collect results
        # 3. Return comparison DataFrame
        pass

# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=10, 
                               n_informative=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    tuner = HyperparameterTuner(task_type='classification', cv_folds=5)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    results = tuner.grid_search(X, pd.Series(y), 
                                 RandomForestClassifier(random_state=42),
                                 param_grid)
    
    print("Best Parameters:", results['best_params'])
    print("Best CV Score:", results['best_score'])
    print("\nCV Results Summary:")
    print(results['cv_results'])
```

### Key Learning Points

1. **Cross-validation:** Understanding k-fold CV and its importance
2. **Hyperparameter Tuning:** Grid search vs random search
3. **Model Selection:** Comparing multiple models systematically
