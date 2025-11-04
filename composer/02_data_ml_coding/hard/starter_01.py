"""
Exercise 1: LLM Evaluation Framework

Build a comprehensive evaluation framework for Large Language Models.
"""

import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class EvaluationResult:
    """Result of LLM evaluation"""
    metric_name: str
    score: float
    details: Dict


class LLMEvaluator(ABC):
    """Base class for LLM evaluators"""
    
    @abstractmethod
    def evaluate(self, predictions: List[str], 
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        """Evaluate predictions"""
        pass


class BLEUEvaluator(LLMEvaluator):
    """BLEU score evaluator"""
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        # TODO: Implement BLEU score calculation
        # Use n-gram precision with brevity penalty
        # Return EvaluationResult with score and details
        pass


class ROUGEEvaluator(LLMEvaluator):
    """ROUGE score evaluator"""
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        # TODO: Implement ROUGE score calculation
        # Calculate ROUGE-1, ROUGE-2, ROUGE-L
        # Return average or individual scores
        pass


class SemanticSimilarityEvaluator(LLMEvaluator):
    """Semantic similarity using embeddings"""
    
    def __init__(self, embedding_model=None):
        # TODO: Initialize embedding model (e.g., sentence-transformers)
        # Can use a simple implementation or load a model
        self.embedding_model = embedding_model
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        # TODO: Compute embeddings and cosine similarity
        # Return average similarity score
        pass


class LLMEvaluationFramework:
    """
    Comprehensive framework for evaluating LLMs.
    
    Supports multiple metrics, model comparison, and reporting.
    """
    
    def __init__(self):
        self.evaluators: Dict[str, LLMEvaluator] = {}
        self.results_history: List[Dict] = []
    
    def register_evaluator(self, name: str, evaluator: LLMEvaluator):
        """Register an evaluator"""
        # TODO: Add evaluator to self.evaluators
        pass
    
    def evaluate(self, model_name: str, 
                predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None,
                metrics: Optional[List[str]] = None) -> Dict[str, EvaluationResult]:
        """
        Evaluate model predictions.
        
        Args:
            model_name: Name/ID of the model being evaluated
            predictions: Model predictions
            references: Ground truth references (optional)
            inputs: Input prompts (optional)
            metrics: List of metric names to compute (None = all)
        
        Returns:
            Dictionary of metric_name -> EvaluationResult
        """
        # TODO: Implement evaluation
        # Loop through requested metrics
        # Call each evaluator
        # Store results in history
        # Return results dictionary
        pass
    
    def compare_models(self, model_results: Dict[str, Dict[str, EvaluationResult]]) -> Dict:
        """
        Compare multiple models.
        
        Args:
            model_results: Dict of model_name -> metric_results
        
        Returns:
            Comparison summary
        """
        # TODO: Implement model comparison
        # Aggregate metrics across models
        # Return comparison dictionary
        pass
    
    def generate_report(self, results: Dict[str, EvaluationResult]) -> str:
        """Generate human-readable evaluation report"""
        # TODO: Implement report generation
        # Format results nicely
        pass


# Usage example
if __name__ == "__main__":
    framework = LLMEvaluationFramework()
    
    # Register evaluators
    framework.register_evaluator("bleu", BLEUEvaluator())
    framework.register_evaluator("rouge", ROUGEEvaluator())
    
    # Example predictions and references
    predictions = [
        "The cat sat on the mat.",
        "It was a sunny day."
    ]
    references = [
        "A cat was sitting on the mat.",
        "The weather was sunny."
    ]
    
    # Evaluate
    results = framework.evaluate(
        model_name="gpt-4",
        predictions=predictions,
        references=references,
        metrics=["bleu", "rouge"]
    )
    
    # Generate report
    print(framework.generate_report(results))

