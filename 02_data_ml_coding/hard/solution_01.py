"""
Solution for Exercise 1: LLM Evaluation Framework

This file contains the reference solution.
"""

import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from collections import Counter
from datetime import datetime


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
    
    def _get_ngrams(self, text: str, n: int) -> List[tuple]:
        """Get n-grams from text"""
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        if references is None or len(predictions) != len(references):
            return EvaluationResult("bleu", 0.0, {"error": "References required and must match predictions length"})
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            # Calculate BLEU for this pair
            pred_ngrams = {}
            ref_ngrams = {}
            
            # Calculate precision for n-grams 1-4
            precisions = []
            for n in range(1, 5):
                pred_ng = self._get_ngrams(pred, n)
                ref_ng = self._get_ngrams(ref, n)
                
                if len(pred_ng) == 0:
                    precisions.append(0.0)
                    continue
                
                pred_counts = Counter(pred_ng)
                ref_counts = Counter(ref_ng)
                
                matches = sum(min(pred_counts[ng], ref_counts[ng]) for ng in pred_counts)
                precisions.append(matches / len(pred_ng))
            
            # Geometric mean of precisions
            if any(p == 0 for p in precisions):
                bleu = 0.0
            else:
                bleu = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
            
            # Brevity penalty
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            if pred_len < ref_len:
                bp = np.exp(1 - ref_len / pred_len)
            else:
                bp = 1.0
            
            bleu_scores.append(bleu * bp)
        
        avg_bleu = np.mean(bleu_scores)
        return EvaluationResult("bleu", avg_bleu, {
            "individual_scores": bleu_scores,
            "per_ngram_precision": {
                f"p{i+1}": np.mean([p[i] for p in [precisions for _ in range(len(predictions))]]) 
                if len(predictions) > 0 else 0.0
                for i in range(4)
            }
        })


class ROUGEEvaluator(LLMEvaluator):
    """ROUGE score evaluator"""
    
    def _get_ngrams(self, text: str, n: int) -> Counter:
        """Get n-grams from text"""
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        return Counter(ngrams)
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        if references is None or len(predictions) != len(references):
            return EvaluationResult("rouge", 0.0, {"error": "References required"})
        
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            # ROUGE-1 (unigram overlap)
            pred_1grams = self._get_ngrams(pred, 1)
            ref_1grams = self._get_ngrams(ref, 1)
            matches_1 = sum(min(pred_1grams[ng], ref_1grams[ng]) for ng in pred_1grams)
            rouge_1 = matches_1 / len(ref.split()) if len(ref.split()) > 0 else 0.0
            rouge_1_scores.append(rouge_1)
            
            # ROUGE-2 (bigram overlap)
            pred_2grams = self._get_ngrams(pred, 2)
            ref_2grams = self._get_ngrams(ref, 2)
            matches_2 = sum(min(pred_2grams[ng], ref_2grams[ng]) for ng in pred_2grams)
            rouge_2 = matches_2 / len(self._get_ngrams(ref, 2)) if len(self._get_ngrams(ref, 2)) > 0 else 0.0
            rouge_2_scores.append(rouge_2)
            
            # ROUGE-L (longest common subsequence, simplified)
            # Simplified: use word-level LCS
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            lcs_len = self._lcs_length(pred_words, ref_words)
            rouge_l = lcs_len / len(ref_words) if len(ref_words) > 0 else 0.0
            rouge_l_scores.append(rouge_l)
        
        avg_rouge_1 = np.mean(rouge_1_scores)
        avg_rouge_2 = np.mean(rouge_2_scores)
        avg_rouge_l = np.mean(rouge_l_scores)
        
        return EvaluationResult("rouge", (avg_rouge_1 + avg_rouge_2 + avg_rouge_l) / 3, {
            "rouge_1": avg_rouge_1,
            "rouge_2": avg_rouge_2,
            "rouge_l": avg_rouge_l,
            "individual_scores": {
                "rouge_1": rouge_1_scores,
                "rouge_2": rouge_2_scores,
                "rouge_l": rouge_l_scores
            }
        })
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class SemanticSimilarityEvaluator(LLMEvaluator):
    """Semantic similarity using embeddings (simplified)"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple word-based embedding (for demonstration)"""
        words = text.lower().split()
        # Simple: average of word hashes (in practice, use proper embeddings)
        vec = np.zeros(100)
        for word in words:
            hash_val = hash(word) % 100
            vec[hash_val] += 1
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def evaluate(self, predictions: List[str],
                references: Optional[List[str]] = None,
                inputs: Optional[List[str]] = None) -> EvaluationResult:
        if references is None:
            return EvaluationResult("semantic_similarity", 0.0, {"error": "References required"})
        
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_emb = self._simple_embedding(pred)
            ref_emb = self._simple_embedding(ref)
            similarity = np.dot(pred_emb, ref_emb)
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        return EvaluationResult("semantic_similarity", avg_similarity, {
            "individual_scores": similarities
        })


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
            'timestamp': datetime.now().isoformat()
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

