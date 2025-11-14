# Challenge 04: Recommendation System

## Problem Statement

Design an end-to-end recommendation system that:
- Provides personalized recommendations for users
- Handles cold start problem (new users, new items)
- Supports multiple recommendation strategies (collaborative filtering, content-based, hybrid)
- Scales to millions of users and items
- Delivers recommendations in real-time (< 100ms latency)
- Continuously learns from user interactions

## Requirements to Clarify

**Ask these questions before designing:**

### Functional Requirements
- What's being recommended? (products, content, connections)
- What are the recommendation use cases? (homepage, related items, search)
- Do we need explanations for recommendations?
- What's the recommendation format? (ranked list, top-N items)

### Non-Functional Requirements
- How many users? (current and expected)
- How many items? (current and expected)
- What's the latency requirement? (< 100ms, < 500ms?)
- What's the throughput requirement? (requests/sec)
- How often should recommendations update? (real-time, hourly, daily)

### Constraints
- What data do we have? (user interactions, item metadata, user profiles)
- Are there any business rules? (diversity, freshness, business constraints)
- What's the budget?
- What's the acceptable recommendation quality?

## Design Considerations

### Core Components

1. **Data Collection**
   - User interactions (clicks, views, purchases)
   - User profiles
   - Item metadata
   - Context (time, location, device)

2. **Feature Engineering**
   - User features (preferences, behavior)
   - Item features (content, popularity)
   - Interaction features (implicit, explicit)
   - Context features

3. **Model Training**
   - Collaborative filtering models
   - Content-based models
   - Deep learning models
   - Hybrid models

4. **Candidate Generation**
   - Generate candidate items (thousands)
   - Fast, approximate methods
   - Multiple strategies

5. **Ranking**
   - Rank candidates (top-N)
   - Precise, slower methods
   - Personalization

6. **Serving**
   - Real-time recommendation API
   - Caching
   - A/B testing

### Key Design Decisions

#### 1. Recommendation Architecture

**Two-Stage Architecture (Most Common):**
1. **Candidate Generation**: Fast, approximate (thousands of candidates)
2. **Ranking**: Slower, precise (top-N items)

**Benefits:**
- Balances latency and quality
- Scalable to large catalogs
- Allows multiple candidate sources

**Recommendation**: Use two-stage architecture for production

#### 2. Recommendation Strategies

**Collaborative Filtering:**
- User-based: Find similar users
- Item-based: Find similar items
- Matrix factorization: Learn latent factors
- Good for: Rich interaction data

**Content-Based:**
- Match user preferences to item features
- Good for: Cold start, explainability

**Deep Learning:**
- Neural collaborative filtering
- Wide & Deep models
- Good for: Complex patterns, large scale

**Hybrid:**
- Combine multiple strategies
- Good for: Best of all worlds

**Recommendation**: Start with collaborative filtering, add content-based for cold start, use deep learning for scale

#### 3. Candidate Generation

**Approaches:**
- **Matrix Factorization**: Pre-compute item similarities
- **Two-Tower Models**: Embed users and items separately
- **Approximate Nearest Neighbors**: Fast similarity search (FAISS, Annoy)
- **Rule-Based**: Popular items, trending items

**Storage:**
- Pre-computed embeddings (vector database)
- Item similarity matrix (key-value store)
- Popular/trending lists (cache)

**Recommendation**: Use two-tower model + ANN for scalability

#### 4. Ranking

**Approaches:**
- **Learning-to-Rank**: Train ranking model
- **Deep Learning**: Wide & Deep, DCN
- **Feature Engineering**: User-item interaction features

**Features:**
- User features (preferences, history)
- Item features (popularity, quality)
- Interaction features (user-item similarity)
- Context features (time, location)

**Recommendation**: Use learning-to-rank with deep learning

#### 5. Cold Start Problem

**New Users:**
- Use popular/trending items
- Ask for preferences
- Use demographic features
- Use content-based recommendations

**New Items:**
- Use content-based recommendations
- Boost new items in ranking
- Use item metadata

**Recommendation**: Hybrid approach (popular + content-based)

## High-Level Architecture

```
[User Interactions] (Clicks, Views, Purchases)
    ↓
[Data Pipeline] (ETL, Feature Engineering)
    ↓
[Training Pipeline]
    ├── [Candidate Generation Model] (Two-Tower, Matrix Factorization)
    └── [Ranking Model] (Learning-to-Rank, Deep Learning)
         ↓
[Model Storage] (Model Registry, Embeddings)
    ↓
[Recommendation Service]
    ├── [Candidate Generation] (ANN Search, Pre-computed)
    └── [Ranking] (Real-time Scoring)
         ↓
[Post-Processing] (Diversity, Business Rules, Filtering)
    ↓
[API] (Recommendation Endpoint)
    ↓
[Users]
```

## Detailed Design

### Data Collection

**User Interactions:**
- Implicit feedback (clicks, views, time spent)
- Explicit feedback (ratings, likes)
- Purchase data
- Search queries

**Storage:**
- Event stream (Kafka) for real-time
- Data warehouse for batch processing
- Key-value store for recent interactions

### Feature Engineering

**User Features:**
- Historical preferences
- Behavior patterns
- Demographic information
- Context (time, location, device)

**Item Features:**
- Content features (category, tags, description)
- Popularity metrics (views, purchases)
- Quality metrics (ratings, reviews)
- Freshness (new items)

**Interaction Features:**
- User-item similarity
- Interaction history
- Time since last interaction

**Storage:**
- Feature store (offline + online)
- Real-time feature computation

### Model Training

**Candidate Generation Model:**
- **Two-Tower Architecture**:
  - User tower: Embed user features
  - Item tower: Embed item features
  - Similarity: Cosine similarity between embeddings
- **Training**: Contrastive learning (positive vs negative pairs)
- **Output**: User and item embeddings

**Ranking Model:**
- **Wide & Deep Architecture**:
  - Wide: Memorization (linear model)
  - Deep: Generalization (neural network)
- **Features**: User features, item features, interaction features
- **Training**: Pointwise/pairwise/listwise learning-to-rank
- **Output**: Relevance scores

### Candidate Generation (Serving)

**Process:**
1. Get user embedding (from user ID or compute on-the-fly)
2. Search for similar items using ANN (FAISS, Annoy)
3. Apply filters (availability, business rules)
4. Return top-K candidates (e.g., 1000)

**Optimization:**
- Pre-compute user embeddings for active users
- Use approximate nearest neighbors (FAISS)
- Cache popular candidates
- Parallel search across item categories

**Latency**: < 50ms for candidate generation

### Ranking (Serving)

**Process:**
1. Get candidate items (from candidate generation)
2. Compute features for each candidate
3. Score using ranking model
4. Apply post-processing (diversity, business rules)
5. Return top-N items (e.g., 20)

**Optimization:**
- Batch scoring
- Feature caching
- Model optimization (quantization)
- Pre-compute common features

**Latency**: < 50ms for ranking

### Post-Processing

**Diversity:**
- Ensure variety in recommendations
- Avoid too many similar items
- Category diversity

**Business Rules:**
- Boost certain items (promotions)
- Filter inappropriate content
- Ensure availability

**Freshness:**
- Include new items
- Avoid stale recommendations
- Time-based decay

### Serving API

**Endpoint:**
```
GET /recommendations?user_id=123&context=homepage&limit=20
```

**Response:**
```json
{
  "user_id": "123",
  "recommendations": [
    {
      "item_id": "item_456",
      "score": 0.95,
      "reason": "Based on your preferences"
    },
    ...
  ],
  "candidates_generated": 1000,
  "latency_ms": 85
}
```

### Real-time Learning

**Approach:**
- Collect user interactions in real-time
- Update user embeddings incrementally
- Retrain ranking model periodically
- Use online learning for fast adaptation

**Technologies:**
- Stream processing (Flink, Kafka Streams)
- Online learning algorithms
- Incremental model updates

## Scaling Strategy

### Start Small
- Single recommendation model
- Batch training (daily)
- Simple candidate generation
- ~100K users, ~10K items

### Scale Gradually

**Phase 1: Two-Stage Architecture**
- Separate candidate generation and ranking
- Batch training
- ~1M users, ~100K items

**Phase 2: Distributed Training**
- Distributed model training
- Real-time feature computation
- ~10M users, ~1M items

**Phase 3: Real-time Learning**
- Online learning
- Real-time candidate updates
- Multi-region deployment
- ~100M users, ~10M items

## Trade-offs

### Latency vs Quality
- **Lower latency**: Fewer candidates, simpler ranking, lower quality
- **Higher quality**: More candidates, complex ranking, higher latency
- **Recommendation**: Balance with two-stage architecture

### Personalization vs Diversity
- **More personalization**: Better relevance, less diversity
- **More diversity**: Better exploration, potentially lower relevance
- **Recommendation**: Post-processing for diversity

### Real-time vs Batch
- **Real-time**: Better freshness, more complex, higher cost
- **Batch**: Simpler, lower cost, potentially stale
- **Recommendation**: Hybrid (batch training + real-time updates)

### Accuracy vs Explainability
- **Deep learning**: Higher accuracy, less explainable
- **Content-based**: Lower accuracy, more explainable
- **Recommendation**: Use both, explain content-based recommendations

## Operational Concerns

### Monitoring
- Recommendation quality (CTR, conversion rate)
- Latency (p50, p95, p99)
- Model performance (AUC, NDCG)
- User engagement metrics

### A/B Testing
- Test different models
- Test different strategies
- Measure impact on business metrics
- Gradual rollouts

### Data Quality
- Handle missing data
- Detect data drift
- Validate user interactions
- Monitor feature quality

### Cost Optimization
- Cache recommendations
- Optimize model inference
- Use efficient ANN algorithms
- Batch processing where possible

## Follow-up Questions

**Be prepared to answer:**
1. "How do you handle the cold start problem?"
2. "How would you ensure recommendation diversity?"
3. "How do you update recommendations in real-time?"
4. "How would you handle a new user with no history?"
5. "How do you measure recommendation quality?"
6. "How would you implement A/B testing?"
7. "How do you handle bias in recommendations?"

## Example Solutions

### Simple Solution (Start Here)
- Single collaborative filtering model
- Batch training
- Simple candidate generation
- ~100K users, ~10K items
- ~200ms latency

### Optimized Solution
- Two-stage architecture
- Deep learning models
- ANN for candidate generation
- Feature store
- ~10M users, ~1M items
- ~100ms latency

### Production Solution
- Two-stage architecture with multiple strategies
- Real-time learning
- Multi-region deployment
- Advanced post-processing
- Comprehensive monitoring
- ~100M users, ~10M items
- ~50ms latency

## Practice Exercise

**Design this system in 45-60 minutes:**
1. Clarify requirements (5-10 min)
2. High-level design (10-15 min)
3. Detailed design (15-20 min)
4. Scaling and optimization (10-15 min)
5. Trade-offs discussion (5 min)

**Focus on:**
- Two-stage architecture
- Candidate generation
- Ranking
- Cold start handling
- Real-time updates

