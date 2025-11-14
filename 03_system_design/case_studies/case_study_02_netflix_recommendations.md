# Case Study 02: Netflix Recommendation System

## Overview

Netflix's recommendation system serves personalized content to 200M+ subscribers, driving 80% of content watched. This case study explores how Netflix likely architected their recommendation system.

## Scale Requirements

- **Users**: 200M+ subscribers
- **Items**: 10,000+ titles (movies, TV shows)
- **Requests**: Billions of recommendations per day
- **Latency**: < 100ms for homepage recommendations
- **Personalization**: Highly personalized per user

## Architecture Overview

### High-Level Architecture

```
[User] → [Netflix App/Web]
            ↓
    [Recommendation API]
            ↓
    [Two-Stage System]
    ├── [Candidate Generation] (1000s of candidates)
    └── [Ranking] (Top-N items)
            ↓
    [Post-Processing] (Diversity, Business Rules)
            ↓
    [Homepage/UI]
```

## Key Components

### 1. Two-Stage Architecture

**Stage 1: Candidate Generation**
- Generate 1000-5000 candidate items
- Fast, approximate methods
- Multiple strategies combined
- Latency: < 50ms

**Stage 2: Ranking**
- Rank candidates to top-N (typically 20-40)
- Precise, slower methods
- Personalization
- Latency: < 50ms

**Why Two-Stage?**
- Can't rank all items in real-time (too expensive)
- Candidate generation filters to relevant items
- Ranking personalizes from candidates
- Balances quality and latency

### 2. Candidate Generation Strategies

**Multiple Strategies Combined:**

1. **Collaborative Filtering**:
   - Matrix factorization (learn user/item embeddings)
   - Find similar users/items
   - Generate candidates based on similarity

2. **Content-Based**:
   - Match user preferences to item features
   - Use item metadata (genre, actors, etc.)
   - Good for cold start

3. **Trending/Popular**:
   - Currently popular items
   - Trending items
   - New releases

4. **Contextual**:
   - Time of day
   - Device type
   - Location

**Implementation:**
- Pre-compute embeddings offline
- Store in vector database (FAISS, Annoy)
- Real-time ANN search for candidates
- Combine results from multiple strategies

### 3. Ranking Model

**Wide & Deep Architecture:**

**Wide Component (Memorization):**
- Linear model
- Captures specific user-item interactions
- Good for: "Users who watched X also watched Y"

**Deep Component (Generalization):**
- Neural network
- Learns complex patterns
- Good for: Discovering new preferences

**Features:**
- User features (history, preferences, demographics)
- Item features (genre, rating, popularity)
- Interaction features (user-item similarity)
- Context features (time, device)

**Training:**
- Train on user interaction data (views, ratings, skips)
- Optimize for engagement (watch time, completion rate)
- A/B test different models

### 4. Feature Store

**Offline Features:**
- User historical preferences
- Item popularity metrics
- User-item interaction history
- Computed in batch, stored in data warehouse

**Online Features:**
- Real-time user context
- Current session data
- Recent interactions
- Computed on-demand, stored in Redis

**Feature Consistency:**
- Ensure training and serving use same features
- Point-in-time correctness
- Feature versioning

### 5. Post-Processing

**Diversity:**
- Ensure variety in recommendations
- Avoid too many similar items
- Category diversity (different genres)

**Business Rules:**
- Boost certain content (new releases, originals)
- Filter inappropriate content
- Ensure availability (not removed from catalog)

**Freshness:**
- Include new releases
- Avoid stale recommendations
- Time-based decay

### 6. Real-time Learning

**Online Learning:**
- Update user preferences in real-time
- Adapt to recent interactions
- Improve recommendations quickly

**Implementation:**
- Stream user interactions (Kafka)
- Update user embeddings incrementally
- Retrain ranking model periodically (daily/weekly)

## Design Decisions & Trade-offs

### 1. Two-Stage vs Single-Stage

**Trade-off**: Quality vs Latency
- **Single-stage**: Rank all items, too slow
- **Two-stage**: Fast candidate generation + precise ranking
- **Decision**: Two-stage architecture for scalability

### 2. Collaborative Filtering vs Content-Based

**Trade-off**: Accuracy vs Cold Start
- **Collaborative filtering**: Better accuracy, cold start problem
- **Content-based**: Handles cold start, lower accuracy
- **Decision**: Hybrid approach, combine both

### 3. Real-time vs Batch Updates

**Trade-off**: Freshness vs Complexity
- **Real-time**: Better freshness, more complex
- **Batch**: Simpler, potentially stale
- **Decision**: Hybrid (real-time for user preferences, batch for model training)

### 4. Personalization vs Diversity

**Trade-off**: Relevance vs Exploration
- **More personalization**: Better relevance, less diversity
- **More diversity**: Better exploration, potentially lower relevance
- **Decision**: Balance with post-processing (diversity filters)

## Scaling Strategy

### Phase 1: Initial System
- Simple collaborative filtering
- Batch recommendations
- ~1M users

### Phase 2: Scale
- Two-stage architecture
- Multiple candidate strategies
- Real-time features
- ~10M users

### Phase 3: Advanced
- Deep learning ranking
- Real-time learning
- Advanced post-processing
- ~100M users

### Phase 4: Current Scale
- Multi-strategy candidate generation
- Advanced ranking models
- Comprehensive personalization
- 200M+ users

## Key Learnings

1. **Two-Stage Architecture**: Essential for scalability and quality
2. **Multiple Strategies**: Combining strategies improves coverage
3. **Feature Store**: Critical for consistency between training and serving
4. **Post-Processing**: Important for business goals and user experience
5. **Real-time Learning**: Improves personalization and freshness
6. **A/B Testing**: Critical for measuring impact and optimizing

## Interview Takeaways

When designing recommendation systems:
- Use two-stage architecture (candidate generation + ranking)
- Combine multiple candidate strategies
- Implement feature store for consistency
- Consider cold start problem
- Balance personalization and diversity
- Design for real-time updates
- Always discuss A/B testing

## References

- Netflix Tech Blog
- "The Netflix Recommender System" research papers
- Industry best practices for recommendation systems

