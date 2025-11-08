"""
Exercise 4: Model Interpretability & Explainability

Build a comprehensive model interpretability system with SHAP values,
feature importance, and explainable AI techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")


class ModelInterpreter:
    """
    Comprehensive model interpretability system.
    
    Computes feature importance, SHAP values, and generates explanations.
    """
    
    def __init__(self, 
                 model: Any,
                 method: str = 'shap',  # 'shap', 'permutation', 'builtin'
                 sample_size: Optional[int] = None):
        """
        Initialize interpreter.
        
        Args:
            model: Trained model to interpret
            method: Interpretation method
            sample_size: Sample size for SHAP computation (for efficiency)
        """
        self.model = model
        self.method = method
        self.sample_size = sample_size
        
        self.shap_explainer_ = None
        self.shap_values_ = None
        self.feature_importances_ = {}
        self.pdp_data_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit interpreter (prepare explainers).
        
        Args:
            X: Training data
            y: Training target (optional)
        
        Returns:
            self
        """
        # TODO: Initialize explainers
        # For SHAP: Create appropriate explainer (TreeExplainer for trees, KernelExplainer for others)
        # For permutation: Will compute during explain
        # Store sample of data for efficient computation
        pass
    
    def explain_global(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate global explanations (overall model behavior).
        
        Args:
            X: Data to explain
        
        Returns:
            Dictionary with feature importance and global insights
        """
        # TODO: Compute global explanations
        # Feature importance (SHAP, permutation, or built-in)
        # Partial dependence plots data
        # Feature interactions
        pass
    
    def explain_local(self, X: pd.DataFrame, instance_idx: int) -> Dict[str, Any]:
        """
        Generate local explanation for a single prediction.
        
        Args:
            X: Data
            instance_idx: Index of instance to explain
        
        Returns:
            Dictionary with local explanation
        """
        # TODO: Compute local explanation
        # SHAP values for this instance
        # Feature contributions
        # Prediction breakdown
        pass
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            DataFrame with features and importance scores
        """
        # TODO: Compute feature importance
        # Use method specified (SHAP, permutation, or built-in)
        # Return sorted DataFrame
        pass
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values"""
        # TODO: Compute SHAP values
        # Use appropriate explainer based on model type
        # Handle sampling for large datasets
        pass
    
    def compute_permutation_importance(self, X: pd.DataFrame, y: pd.Series,
                                       scoring: str = 'accuracy') -> Dict[str, float]:
        """Compute permutation importance"""
        # TODO: Compute permutation importance
        # Use sklearn's permutation_importance
        # Return dictionary mapping features to importance scores
        pass
    
    def compute_partial_dependence(self, X: pd.DataFrame, features: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute partial dependence for features.
        
        Args:
            X: Features
            features: List of feature names to compute PD for
        
        Returns:
            Dictionary mapping feature names to PD values
        """
        # TODO: Compute partial dependence
        # Use sklearn's partial_dependence
        # Return dictionary with PD data
        pass


class ExplanationVisualizer:
    """
    Visualizer for model explanations.
    """
    
    def __init__(self, interpreter: ModelInterpreter):
        """
        Initialize visualizer.
        
        Args:
            interpreter: Fitted ModelInterpreter
        """
        self.interpreter = interpreter
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_k: int = 20, ax: Optional[plt.Axes] = None):
        """Plot feature importance"""
        # TODO: Create bar plot of feature importance
        pass
    
    def plot_shap_summary(self, shap_values: np.ndarray, X: pd.DataFrame,
                         top_k: int = 20, ax: Optional[plt.Axes] = None):
        """Plot SHAP summary plot"""
        # TODO: Create SHAP summary plot
        # If SHAP available, use shap.summary_plot
        # Otherwise, create custom visualization
        pass
    
    def plot_shap_waterfall(self, shap_values: np.ndarray, X: pd.DataFrame,
                           instance_idx: int, ax: Optional[plt.Axes] = None):
        """Plot SHAP waterfall for single instance"""
        # TODO: Create waterfall plot showing contribution of each feature
        pass
    
    def plot_partial_dependence(self, pdp_data: Dict[str, np.ndarray],
                               feature_name: str, ax: Optional[plt.Axes] = None):
        """Plot partial dependence"""
        # TODO: Create partial dependence plot
        pass
    
    def plot_feature_interaction(self, X: pd.DataFrame, feature1: str, feature2: str,
                                shap_values: np.ndarray, ax: Optional[plt.Axes] = None):
        """Plot feature interaction"""
        # TODO: Visualize interaction between two features
        pass


class ModelAuditor:
    """
    Auditor for detecting model issues like bias and problematic patterns.
    """
    
    def __init__(self, model: Any, sensitive_features: Optional[List[str]] = None):
        """
        Initialize auditor.
        
        Args:
            model: Trained model
            sensitive_features: List of sensitive feature names (for bias detection)
        """
        self.model = model
        self.sensitive_features = sensitive_features or []
    
    def detect_bias(self, X: pd.DataFrame, y: pd.Series, 
                   protected_group: str) -> Dict[str, float]:
        """
        Detect potential bias against protected groups.
        
        Args:
            X: Features
            y: True labels
            protected_group: Name of protected group feature
        
        Returns:
            Dictionary with bias metrics
        """
        # TODO: Implement bias detection
        # Demographic parity: P(pred=1 | protected) vs P(pred=1 | not protected)
        # Equalized odds: P(pred=1 | y=1, protected) vs P(pred=1 | y=1, not protected)
        # Return metrics
        pass
    
    def detect_feature_interactions(self, X: pd.DataFrame, 
                                   shap_values: np.ndarray) -> List[Tuple[str, str, float]]:
        """
        Detect significant feature interactions.
        
        Args:
            X: Features
            shap_values: SHAP values
        
        Returns:
            List of (feature1, feature2, interaction_strength) tuples
        """
        # TODO: Detect feature interactions
        # Analyze SHAP values to find features that interact
        # Return top interactions
        pass
    
    def flag_high_uncertainty(self, X: pd.DataFrame, 
                             uncertainty_threshold: float = 0.3) -> List[int]:
        """
        Flag predictions with high uncertainty.
        
        Args:
            X: Features
            uncertainty_threshold: Threshold for flagging (based on prediction confidence)
        
        Returns:
            List of instance indices with high uncertainty
        """
        # TODO: Flag uncertain predictions
        # Use prediction probabilities to identify low-confidence predictions
        # Return indices
        pass
    
    def validate_against_domain_knowledge(self, feature_importance: pd.DataFrame,
                                         domain_rules: Dict[str, Any]) -> List[str]:
        """
        Validate model behavior against domain knowledge.
        
        Args:
            feature_importance: Feature importance DataFrame
            domain_rules: Dictionary of domain rules (e.g., {'feature_A': 'should_be_positive'})
        
        Returns:
            List of validation warnings
        """
        # TODO: Validate against domain knowledge
        # Check if feature importance aligns with domain expectations
        # Return warnings
        pass


# Usage example
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    
    # Interpret model
    interpreter = ModelInterpreter(model, method='shap' if SHAP_AVAILABLE else 'permutation')
    interpreter.fit(X_train, y_train)
    
    # Global explanation
    print("\n=== Global Explanation ===")
    importance_df = interpreter.get_feature_importance(X_test, y_test)
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Local explanation
    print("\n=== Local Explanation (First Test Instance) ===")
    local_explanation = interpreter.explain_local(X_test, instance_idx=0)
    print(f"Prediction: {model.predict(X_test.iloc[[0]])[0]}")
    print(f"Prediction Probability: {model.predict_proba(X_test.iloc[[0]])[0][1]:.4f}")
    
    # Visualize
    if SHAP_AVAILABLE:
        visualizer = ExplanationVisualizer(interpreter)
        print("\nGenerating visualizations...")
        # visualizer.plot_feature_importance(importance_df, top_k=10)
        # visualizer.plot_shap_summary(interpreter.shap_values_, X_test, top_k=10)
    
    # Audit model
    print("\n=== Model Audit ===")
    auditor = ModelAuditor(model)
    
    # Check for high uncertainty predictions
    uncertain_indices = auditor.flag_high_uncertainty(X_test, uncertainty_threshold=0.3)
    print(f"Found {len(uncertain_indices)} predictions with high uncertainty")
    
    # Validate against domain knowledge
    domain_rules = {
        'feature_0': 'should_be_important',  # Example rule
    }
    warnings = auditor.validate_against_domain_knowledge(importance_df, domain_rules)
    if warnings:
        print("Domain validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")
