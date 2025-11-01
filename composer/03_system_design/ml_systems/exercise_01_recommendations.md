# Exercise 1: Design a Real-time Recommendation System

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** ML systems, real-time serving, feature pipelines

## Problem

Design a recommendation system for an e-commerce platform that:
- Serves personalized recommendations in real-time (< 100ms latency)
- Handles millions of users and products
- Updates recommendations as user behavior changes
- Supports multiple recommendation strategies (collaborative, content-based, etc.)
- Handles cold start for new users/products

## Requirements to Discuss

1. **System Architecture**
   - Online vs offline components?
   - How to integrate real-time signals?
   - Caching strategy?

2. **Feature Pipeline**
   - Real-time features (user session, clicks)?
   - Batch features (user history, product stats)?
   - Feature store design?

3. **Model Serving**
   - How to serve multiple models?
   - Model ensemble?
   - A/B testing framework?
   - How to handle model updates?

4. **Personalization**
   - User embedding/cold start?
   - How to rank candidates?
   - Diversification?

5. **Scalability**
   - How to pre-compute vs compute-on-demand?
   - Candidate generation vs ranking?
   - How to scale to millions of users?

6. **Data Pipeline**
   - How to collect user behavior?
   - How to update user profiles?
   - Training data pipeline?

## Key Topics to Cover

- **Two-Stage Architecture**: Candidate generation + ranking
- **Feature Stores**: Online and offline features
- **Embeddings**: User and item embeddings
- **Real-time Serving**: Caching, pre-computation
- **Model Registry**: Versioning, A/B testing

## Sample Discussion Points

1. "I'd use a two-stage architecture: candidate generation using collaborative filtering embeddings (pre-computed), then ranking with a real-time model that uses session features."

2. "For features, I'd maintain a feature store. Batch features (user purchase history) updated daily, real-time features (current session clicks) computed on-the-fly."

3. "Model serving would use a model server with caching. Pre-computed embeddings stored in a vector database like FAISS or Pinecone for fast candidate retrieval."

4. "For cold start, I'd use content-based recommendations (similar items) until enough interaction data exists for collaborative filtering."

5. "Real-time updates would use a lightweight model that can incorporate recent behavior without full retraining. Batch models retrain daily on full history."

## Additional Considerations

- How to handle explainability (why these recommendations)?
- How to prevent filter bubbles?
- How to balance exploration vs exploitation?
- How to handle privacy (differential privacy, federated learning)?

