# System Design - ML Systems Exercises

These exercises focus specifically on ML/AI system design, which is crucial for the Improvement Engineer role at 8090.

---

## Exercise 1: Design a Real-time Recommendation System

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** ML systems, real-time serving, feature pipelines

### Problem

Design a recommendation system for an e-commerce platform that:
- Serves personalized recommendations in real-time (< 100ms latency)
- Handles millions of users and products
- Updates recommendations as user behavior changes
- Supports multiple recommendation strategies (collaborative, content-based, etc.)
- Handles cold start for new users/products

### Key Topics to Cover

- **Candidate Generation**: Two-stage approach (retrieval + ranking)
- **Feature Pipeline**: Real-time vs batch features
- **Model Serving**: Low-latency inference, caching
- **Personalization**: User embeddings, cold start handling
- **A/B Testing**: Multi-armed bandit, exploration vs exploitation

### Discussion Points

1. Online vs offline components
2. Real-time feature pipeline design
3. Model serving architecture
4. Personalization strategies
5. Scalability and pre-computation

---

## Exercise 2: Design an ML Model Serving Platform

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Model serving, versioning, monitoring, scalability

### Problem

Design a platform for serving ML models that:
- Serves multiple model types (sklearn, PyTorch, TensorFlow)
- Handles versioning and A/B testing
- Provides low-latency inference
- Monitors model performance and drift
- Scales to handle traffic spikes

### Key Topics to Cover

- **Model Registry**: Versioning, stage promotion
- **Serving Infrastructure**: Batching, GPU acceleration
- **Traffic Management**: Load balancing, routing
- **Monitoring**: Latency, throughput, drift detection
- **A/B Testing**: Traffic splitting, gradual rollouts

### Discussion Points

1. Model registry and versioning
2. Serving architecture and batching
3. Traffic routing and load balancing
4. Monitoring and alerting
5. A/B testing framework

---

## Exercise 3: Design a Distributed ML Training Infrastructure

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Distributed training, data pipelines, resource management

### Problem

Design a system for training large ML models that:
- Distributes training across multiple machines
- Handles large datasets efficiently
- Manages GPU resources
- Supports experiment tracking
- Handles failures and checkpoints

### Key Topics to Cover

- **Data Pipeline**: Distributed data loading, preprocessing
- **Training Orchestration**: Parameter servers, all-reduce
- **Resource Management**: GPU scheduling, multi-tenancy
- **Experiment Tracking**: Hyperparameter tuning, metrics
- **Fault Tolerance**: Checkpointing, failure recovery

### Discussion Points

1. Distributed training architecture
2. Data pipeline and preprocessing
3. Resource management and scheduling
4. Experiment tracking and hyperparameter tuning
5. Fault tolerance and recovery

---

## Exercise 4: Design a Feature Store System

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Feature engineering, storage, serving, real-time features

### Problem

Design a feature store that:
- Stores features for training and serving
- Serves features with low latency
- Supports both batch and real-time features
- Handles feature versioning
- Provides feature discovery and lineage

### Key Topics to Cover

- **Storage**: Batch features (data warehouse), online features (Redis/DynamoDB)
- **Feature Serving**: Low-latency lookups, batch serving
- **Real-time Features**: Stream processing, windowed aggregations
- **Versioning**: Feature schemas, backward compatibility
- **Discovery**: Feature catalog, documentation

### Discussion Points

1. Storage architecture (batch vs online)
2. Feature serving and caching
3. Real-time feature computation
4. Feature versioning and schema evolution
5. Feature discovery and governance

---

## Focus Areas (8090 Context)

Given that 8090 is building Software Factory with AI-first approach:
- Production ML systems
- LLM deployment and management
- Model reliability and monitoring
- Experimentation and A/B testing
- Agentic systems (mentioned in job posting)

## How to Practice

1. **Understand the ML lifecycle** - Training, serving, monitoring
2. **Consider both offline and online** - Batch vs real-time
3. **Think about feature engineering** - Feature stores, pipelines
4. **Design for reliability** - Monitoring, drift detection, rollbacks
5. **Plan for experimentation** - A/B testing, gradual rollouts

## Evaluation Criteria (8090 Interview)

- Understanding of ML system patterns
- Trade-offs between different approaches
- Real-time vs batch considerations
- Monitoring and observability
- Scalability and reliability

