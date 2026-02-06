"""
================================================================================
PINTEREST ML PRACTITIONER INTERVIEW CHALLENGE
================================================================================

PROBLEM: Personalized Pin Recommendation System
TIME: 60 minutes

You are building a personalized recommendation system for Pinterest's home feed.
Given a user visiting Pinterest, recommend the most relevant Pins from a large
catalog to maximize user engagement (saves, clicks, closeups).

This challenge covers all ML Practitioner evaluation axes:
- Problem Exploration
- Training Data / Dataset Generation  
- Model Selection
- Feature Engineering
- Evaluation

================================================================================
SUGGESTED TIME ALLOCATION
================================================================================

| Section                        | Time    |
|--------------------------------|---------|
| Problem Discussion (verbal)    | 10 min  |
| PinDataset implementation      | 10 min  |
| Two-Tower Model implementation | 20 min  |
| Training Loop                  | 10 min  |
| Evaluation Metrics             | 10 min  |

================================================================================
PROBLEM EXPLORATION (Discuss with interviewer before coding)
================================================================================

Before writing any code, think through these questions:

1. PROBLEM FRAMING:
   - How would you frame this as an ML problem? (classification, ranking, retrieval?)
   - What is the prediction target? What does a positive label mean?
   - Why is a Two-Tower architecture well-suited for this problem?

2. TRAINING DATA:
   - What user-pin interactions would you use as training signals?
   - How would you handle implicit feedback (views) vs explicit feedback (saves)?
   - What are the risks of training on historical data? (popularity bias, position bias)

3. NEGATIVE SAMPLING:
   - How do you generate negative examples for training?
   - What are the tradeoffs of random negatives vs hard negatives?

4. SCALE CONSIDERATIONS:
   - Pinterest has 500M+ users and billions of Pins. How does Two-Tower help here?
   - How would you serve recommendations at <100ms latency?

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ==============================================================================
# SYNTHETIC DATA GENERATION (PROVIDED - DO NOT MODIFY)
# ==============================================================================

def generate_synthetic_data(
    num_users: int = 1000,
    num_pins: int = 5000,
    num_interactions: int = 50000,
    user_feature_dim: int = 16,
    pin_feature_dim: int = 32,
) -> Dict:
    """
    Generates synthetic Pinterest-like interaction data.
    
    This simulates:
    - User features: demographics, interests, activity level
    - Pin features: visual embeddings, category, freshness
    - Interactions: user-pin pairs with engagement labels
    
    Returns:
        Dictionary containing:
        - user_features: (num_users, user_feature_dim) tensor
        - pin_features: (num_pins, pin_feature_dim) tensor  
        - interactions: List of (user_id, pin_id, label) tuples
        - num_users, num_pins, user_feature_dim, pin_feature_dim
    """
    # Generate user feature vectors (e.g., interest embeddings from past behavior)
    user_features = torch.randn(num_users, user_feature_dim)
    
    # Generate pin feature vectors (e.g., visual embeddings from image model)
    pin_features = torch.randn(num_pins, pin_feature_dim)
    
    # Create some latent "interest clusters" to make the data learnable
    num_clusters = 10
    user_clusters = torch.randint(0, num_clusters, (num_users,))
    pin_clusters = torch.randint(0, num_clusters, (num_pins,))
    
    # Add cluster signal to features
    for i in range(num_users):
        user_features[i, user_clusters[i].item() % user_feature_dim] += 2.0
    for i in range(num_pins):
        pin_features[i, pin_clusters[i].item() % pin_feature_dim] += 2.0
    
    # Generate interactions with label based on cluster match
    interactions = []
    for _ in range(num_interactions):
        user_id = random.randint(0, num_users - 1)
        pin_id = random.randint(0, num_pins - 1)
        
        # Higher engagement probability if clusters match
        cluster_match = user_clusters[user_id] == pin_clusters[pin_id]
        base_prob = 0.7 if cluster_match else 0.1
        
        # Add some noise
        label = 1 if random.random() < base_prob else 0
        interactions.append((user_id, pin_id, label))
    
    return {
        "user_features": user_features,
        "pin_features": pin_features,
        "interactions": interactions,
        "num_users": num_users,
        "num_pins": num_pins,
        "user_feature_dim": user_feature_dim,
        "pin_feature_dim": pin_feature_dim,
    }


# ==============================================================================
# SECTION 1: DATASET CLASS
# Evaluation Axis: Training Data / Feature Engineering
# Time: ~10 minutes
# ==============================================================================

class PinRecommendationDataset(Dataset):
    """
    PyTorch Dataset for Pin recommendation training.
    
    TODO: Implement the following methods:
    
    1. __init__: Store the data and precompute anything needed
    2. __len__: Return the number of samples
    3. __getitem__: Return a single training sample as a dictionary with keys:
       - 'user_features': tensor of shape (user_feature_dim,)
       - 'pin_features': tensor of shape (pin_feature_dim,)
       - 'label': tensor of shape (1,) with value 0 or 1
    
    HINTS:
    - Each sample is one user-pin interaction
    - Look up user_id and pin_id in the feature matrices
    - Consider: should you normalize features? (discuss tradeoffs with interviewer)
    """
    
    def __init__(self, data: Dict):
        """
        Args:
            data: Dictionary from generate_synthetic_data()
        """
        # TODO: Implement initialization
        # Store references to feature matrices and interactions
        pass
    
    def __len__(self) -> int:
        # TODO: Return number of samples
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # TODO: Return a single sample as a dictionary
        # Keys: 'user_features', 'pin_features', 'label'
        pass


# ==============================================================================
# SECTION 2: TWO-TOWER MODEL ARCHITECTURE
# Evaluation Axis: Model Selection
# Time: ~20 minutes
# ==============================================================================

class UserTower(nn.Module):
    """
    User Tower: Encodes user features into a dense embedding.
    
    TODO: Implement a neural network that:
    1. Takes user features as input (shape: batch_size x user_feature_dim)
    2. Outputs a user embedding (shape: batch_size x embedding_dim)
    
    ARCHITECTURE SUGGESTIONS:
    - 2-3 fully connected layers with ReLU activation
    - Consider: dropout for regularization
    - Consider: batch normalization
    - Final layer should NOT have activation (we'll normalize later)
    
    DISCUSSION POINTS:
    - Why do we want user and pin embeddings in the same vector space?
    - What embedding_dim would you choose and why?
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        # TODO: Define layers
        pass
    
    def forward(self, user_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_features: (batch_size, user_feature_dim)
        Returns:
            user_embedding: (batch_size, embedding_dim)
        """
        # TODO: Implement forward pass
        pass


class PinTower(nn.Module):
    """
    Pin Tower: Encodes pin features into a dense embedding.
    
    TODO: Implement a neural network similar to UserTower.
    
    DISCUSSION POINTS:
    - Should PinTower have the same architecture as UserTower?
    - Pin features often include visual embeddings - how might this affect design?
    - At Pinterest scale, pin embeddings are precomputed. Why?
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        # TODO: Define layers
        pass
    
    def forward(self, pin_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pin_features: (batch_size, pin_feature_dim)
        Returns:
            pin_embedding: (batch_size, embedding_dim)
        """
        # TODO: Implement forward pass
        pass


class TwoTowerModel(nn.Module):
    """
    Two-Tower Retrieval Model for Pin Recommendations.
    
    This model:
    1. Encodes users and pins into the same embedding space
    2. Computes similarity scores using dot product or cosine similarity
    3. Outputs a score indicating how likely the user is to engage with the pin
    
    TODO: Implement:
    1. __init__: Create UserTower and PinTower
    2. forward: 
       - Get user embedding from UserTower
       - Get pin embedding from PinTower
       - L2 normalize both embeddings (for cosine similarity)
       - Compute similarity score
       - Return logits (can apply sigmoid during training/eval)
    
    DISCUSSION POINTS:
    - Dot product vs cosine similarity - what are the tradeoffs?
    - How does L2 normalization affect training dynamics?
    - Why is this architecture efficient for retrieval at scale?
    """
    
    def __init__(
        self,
        user_feature_dim: int,
        pin_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # TODO: Initialize towers
        pass
    
    def forward(
        self,
        user_features: torch.Tensor,
        pin_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_features: (batch_size, user_feature_dim)
            pin_features: (batch_size, pin_feature_dim)
        Returns:
            scores: (batch_size,) similarity scores
        """
        # TODO: Implement forward pass
        # 1. Get embeddings from both towers
        # 2. L2 normalize embeddings
        # 3. Compute dot product similarity
        # 4. Optional: scale by temperature parameter
        pass
    
    def get_user_embedding(self, user_features: torch.Tensor) -> torch.Tensor:
        """Get normalized user embedding for inference/indexing."""
        # TODO: Implement - useful for production where we precompute embeddings
        pass
    
    def get_pin_embedding(self, pin_features: torch.Tensor) -> torch.Tensor:
        """Get normalized pin embedding for inference/indexing."""
        # TODO: Implement - useful for production where we precompute embeddings
        pass


# ==============================================================================
# SECTION 3: TRAINING LOOP
# Evaluation Axis: Model Selection (loss function, optimization)
# Time: ~10 minutes
# ==============================================================================

def train_model(
    model: TwoTowerModel,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> List[float]:
    """
    Train the Two-Tower model.
    
    TODO: Implement the training loop with:
    1. Binary Cross-Entropy loss (BCEWithLogitsLoss)
    2. Adam optimizer
    3. Training loop that iterates over epochs and batches
    4. Return list of epoch losses for plotting
    
    DISCUSSION POINTS:
    - Why BCE loss for this problem? What about contrastive losses?
    - What batch size would you use and why?
    - How would you handle class imbalance (more negatives than positives)?
    - What regularization techniques would you consider?
    
    HINTS:
    - BCEWithLogitsLoss expects raw scores (applies sigmoid internally)
    - Don't forget model.train() and optimizer.zero_grad()
    - Move tensors to the correct device
    """
    # TODO: Implement training loop
    # 1. Set up loss function and optimizer
    # 2. Loop over epochs
    # 3. Loop over batches
    # 4. Forward pass, compute loss, backward pass, optimizer step
    # 5. Track and return losses
    
    epoch_losses = []
    
    # YOUR CODE HERE
    
    return epoch_losses


# ==============================================================================
# SECTION 4: EVALUATION METRICS
# Evaluation Axis: Evaluation
# Time: ~10 minutes
# ==============================================================================

def evaluate_model(
    model: TwoTowerModel,
    eval_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate the Two-Tower model with retrieval metrics.
    
    TODO: Implement the following metrics:
    
    1. AUC (Area Under ROC Curve):
       - Measures ranking quality
       - Use torch to compute or implement manually
    
    2. Recall@K (K=10):
       - For each user, what fraction of relevant items appear in top K?
       - This is the key metric for retrieval systems
    
    DISCUSSION POINTS:
    - Why is AUC important for this problem?
    - What's the difference between offline metrics (AUC, Recall@K) and 
      online metrics (CTR, engagement rate)?
    - How would you A/B test a new model at Pinterest?
    - What about NDCG? When would you use it over Recall@K?
    
    HINTS:
    - Don't forget model.eval() and torch.no_grad()
    - Collect all predictions and labels first, then compute metrics
    """
    # TODO: Implement evaluation
    # 1. Set model to eval mode
    # 2. Collect predictions and labels
    # 3. Compute AUC
    # 4. Compute Recall@K
    
    metrics = {
        "auc": 0.0,
        "recall_at_10": 0.0,
    }
    
    # YOUR CODE HERE
    
    return metrics


def compute_auc(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute AUC (Area Under ROC Curve) manually.
    
    TODO: Implement AUC computation.
    
    HINT: AUC = probability that a random positive is ranked higher than a random negative
    
    Simple approach:
    1. Get indices of positive and negative examples
    2. For each positive, count how many negatives it's ranked above
    3. AUC = (sum of counts) / (num_positives * num_negatives)
    """
    # YOUR CODE HERE
    pass


# ==============================================================================
# SECTION 5: PUTTING IT ALL TOGETHER
# ==============================================================================

def main():
    """
    Main function to run the full pipeline.
    
    This is provided as a test harness - once you implement the TODOs above,
    this should run end-to-end and print training loss and evaluation metrics.
    """
    print("=" * 60)
    print("Pinterest Pin Recommendation - Two Tower Model")
    print("=" * 60)
    
    # Generate data
    print("\n[1/5] Generating synthetic data...")
    data = generate_synthetic_data(
        num_users=1000,
        num_pins=5000,
        num_interactions=50000,
    )
    print(f"  Users: {data['num_users']}, Pins: {data['num_pins']}")
    print(f"  Interactions: {len(data['interactions'])}")
    
    # Create dataset and split
    print("\n[2/5] Creating datasets...")
    dataset = PinRecommendationDataset(data)
    
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False)
    print(f"  Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Initialize model
    print("\n[3/5] Initializing Two-Tower model...")
    model = TwoTowerModel(
        user_feature_dim=data["user_feature_dim"],
        pin_feature_dim=data["pin_feature_dim"],
        embedding_dim=64,
        hidden_dim=128,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Train
    print("\n[4/5] Training...")
    losses = train_model(model, train_loader, num_epochs=10, learning_rate=1e-3)
    if losses:
        print(f"  Final training loss: {losses[-1]:.4f}")
    
    # Evaluate
    print("\n[5/5] Evaluating...")
    metrics = evaluate_model(model, eval_loader)
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Recall@10: {metrics['recall_at_10']:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


# ==============================================================================
# FOLLOW-UP DISCUSSION QUESTIONS
# ==============================================================================

"""
After completing the implementation, be prepared to discuss:

1. COLD START PROBLEM:
   - How would you handle new users with no interaction history?
   - How would you handle new pins that were just uploaded?
   - What features would you use as fallbacks?

2. PRODUCTION CONSIDERATIONS:
   - How would you serve this model at Pinterest scale (500M+ users)?
   - Describe the two-stage retrieval + ranking architecture
   - How often would you retrain? Update embeddings?
   - How would you use approximate nearest neighbor (ANN) search? (HNSW, etc.)

3. ADVANCED ARCHITECTURES:
   - How would you incorporate sequential user behavior? (see: PinnerFormer)
   - How would you add real-time features? (see: TransAct)
   - Multi-task learning: what other objectives might you optimize jointly?

4. NEGATIVE SAMPLING STRATEGIES:
   - Random negatives vs in-batch negatives vs hard negatives
   - How does negative sampling affect training dynamics?
   - What is the "sampling bias" problem and how would you correct it?

5. ONLINE EVALUATION:
   - What online metrics would you track? (CTR, save rate, session length)
   - How would you design an A/B test for this model?
   - What guardrail metrics would you monitor? (diversity, freshness)

6. FAIRNESS AND SAFETY:
   - How would you ensure diverse recommendations?
   - How would you prevent filter bubbles?
   - How would you handle potentially harmful content?
"""


if __name__ == "__main__":
    main()
