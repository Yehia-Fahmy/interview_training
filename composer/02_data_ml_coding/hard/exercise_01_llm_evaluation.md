# Exercise 1: LLM Evaluation Framework

**Difficulty:** Hard  
**Time Limit:** 90 minutes  
**Focus:** LLM evaluation, production ML for language models (critical for Improvement Engineer role)

## Problem

Build a comprehensive evaluation framework for Large Language Models. The role involves deploying LLMs, so understanding how to evaluate them is crucial.

## Requirements

1. Implement evaluation for:
   - **Automated metrics**: BLEU, ROUGE, perplexity
   - **Semantic similarity**: Embedding-based evaluation
   - **Task-specific metrics**: Accuracy for classification tasks
   - **Custom evaluators**: Extensible framework for new metrics

2. Support batch and streaming evaluation

3. Create a framework that can:
   - Compare multiple LLMs
   - Track evaluation over time
   - Generate comprehensive reports

## Solution Template

```python
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
        # Your implementation
        # Use nltk.translate.bleu_score or implement from scratch
        pass

class ROUGEEvaluator(LLMEvaluator):
    """ROUGE score evaluator"""
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        # Your implementation
        # Use rouge-score library or implement from scratch
        pass

class SemanticSimilarityEvaluator(LLMEvaluator):
    """Semantic similarity using embeddings"""
    
    def __init__(self, embedding_model=None):
        # Initialize embedding model (e.g., sentence-transformers)
        self.embedding_model = embedding_model
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        # Compute embeddings and cosine similarity
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
        self.evaluators[name] = evaluator
    
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
        results = {}
        
        metrics_to_compute = metrics or list(self.evaluators.keys())
        
        for metric_name in metrics_to_compute:
            if metric_name not in self.evaluators:
                continue
            
            evaluator = self.evaluators[metric_name]
            result = evaluator.evaluate(predictions, references, inputs)
            results[metric_name] = result
        
        # Store in history
        self.results_history.append({
            'model_name': model_name,
            'results': {k: {'score': v.score, 'details': v.details} 
                       for k, v in results.items()},
            'timestamp': None  # Add timestamp
        })
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict[str, EvaluationResult]]) -> Dict:
        """
        Compare multiple models.
        
        Args:
            model_results: Dict of model_name -> metric_results
        
        Returns:
            Comparison summary
        """
        comparison = {}
        
        # Aggregate metrics across models
        all_metrics = set()
        for results in model_results.values():
            all_metrics.update(results.keys())
        
        for metric in all_metrics:
            comparison[metric] = {
                model: results[metric].score 
                for model, results in model_results.items() 
                if metric in results
            }
        
        return comparison
    
    def generate_report(self, results: Dict[str, EvaluationResult]) -> str:
        """Generate human-readable evaluation report"""
        report = "LLM Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        for metric_name, result in results.items():
            report += f"{metric_name}:\n"
            report += f"  Score: {result.score:.4f}\n"
            if result.details:
                report += f"  Details: {json.dumps(result.details, indent=4)}\n"
            report += "\n"
        
        return report

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
```

## Key Learning Points

1. **LLM Evaluation:** Understanding different evaluation metrics
2. **Framework Design:** Extensible, modular architecture
3. **Production Evaluation:** What metrics matter for production LLMs

## Design Considerations

- How to handle reference-free evaluation (when no ground truth)?
- Should evaluation support streaming (for large datasets)?
- How to weight different metrics?
- What about cost/latency evaluation alongside quality?

