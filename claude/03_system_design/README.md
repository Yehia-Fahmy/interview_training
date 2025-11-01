# System Design Interview

## 🎯 Overview

The System Design interview is **discussion-based** (not hands-on coding). It focuses on:

- **Distributed systems**: Scalability, reliability, fault tolerance
- **ML systems architecture**: Training, serving, monitoring
- **Data-intensive applications**: Data pipelines, feature stores, data lakes
- **Production ML**: Deployment strategies, A/B testing, monitoring
- **Trade-offs**: No perfect solution, only appropriate ones

**Duration**: 45-60 minutes

**Format**: Whiteboard/diagram discussion (virtual or in-person)

## 🎓 What 8090 AI is Looking For

### Technical Depth
- Understanding of distributed systems concepts
- Knowledge of ML system components
- Awareness of production challenges
- Familiarity with modern ML infrastructure

### Problem-Solving Approach
- Clarifying requirements before designing
- Considering multiple approaches
- Discussing trade-offs explicitly
- Thinking about edge cases and failure modes

### Communication
- Clear explanations of complex concepts
- Good use of diagrams
- Asking clarifying questions
- Explaining reasoning behind decisions

### Production Mindset
- Monitoring and observability
- Cost considerations
- Scalability planning
- Operational concerns

## 📋 Interview Structure

### Phase 1: Requirements Gathering (10-15 min)
**Your Goal**: Understand the problem fully before designing

Questions to ask:
- **Scale**: How many users? Requests per second? Data volume?
- **Latency**: What are the latency requirements? (p50, p95, p99)
- **Availability**: What's the uptime requirement? (99.9%? 99.99%?)
- **Consistency**: Strong vs eventual consistency needs?
- **Budget**: Cost constraints? Infrastructure preferences?
- **Team**: Team size? Expertise level?
- **Timeline**: MVP vs full system? Phased rollout?

### Phase 2: High-Level Design (15-20 min)
**Your Goal**: Present overall architecture

Components to consider:
- Client/API layer
- Load balancing
- Application servers
- Databases (SQL, NoSQL, vector DBs)
- Caching layers
- Message queues
- ML model serving
- Monitoring and logging

### Phase 3: Deep Dive (15-20 min)
**Your Goal**: Drill into specific components

Interviewer will pick areas to explore:
- Data pipeline architecture
- Model training workflow
- Model serving strategy
- Feature engineering
- Monitoring and alerting
- A/B testing framework
- Failure handling

### Phase 4: Discussion (5-10 min)
**Your Goal**: Address trade-offs and improvements

Topics:
- Bottlenecks and how to address them
- Cost optimization
- Security considerations
- Future enhancements
- Operational challenges

## 🔑 Key Concepts

### Distributed Systems

#### Scalability
- **Horizontal vs Vertical Scaling**
  - Horizontal: Add more machines (preferred for web scale)
  - Vertical: Add more resources to existing machines (limited)
- **Load Balancing**: Round-robin, least connections, consistent hashing
- **Sharding**: Partition data across multiple databases
- **Replication**: Master-slave, master-master, quorum-based

#### Reliability
- **Fault Tolerance**: System continues working despite failures
- **Redundancy**: Eliminate single points of failure
- **Circuit Breakers**: Prevent cascading failures
- **Graceful Degradation**: Reduced functionality vs complete failure

#### Consistency
- **CAP Theorem**: Consistency, Availability, Partition Tolerance (pick 2)
- **Eventual Consistency**: Updates propagate eventually
- **Strong Consistency**: All reads see latest write
- **Trade-offs**: Consistency vs latency vs availability

### ML System Components

#### Training Pipeline
```
Data Collection → Data Validation → Feature Engineering →
Model Training → Model Evaluation → Model Versioning →
Model Registry
```

Components:
- **Data Lake/Warehouse**: Store raw and processed data
- **Feature Store**: Centralized feature computation and storage
- **Training Infrastructure**: GPU/TPU clusters, distributed training
- **Experiment Tracking**: MLflow, Weights & Biases
- **Model Registry**: Versioned model storage

#### Serving Pipeline
```
Request → Load Balancer → Model Server → Model →
Response → Logging
```

Strategies:
- **Online Serving**: Real-time predictions (low latency)
- **Batch Serving**: Precompute predictions (high throughput)
- **Hybrid**: Combination of both

Considerations:
- Model loading and caching
- Request batching for efficiency
- A/B testing infrastructure
- Canary deployments

#### Monitoring Pipeline
```
Metrics Collection → Aggregation → Alerting → Dashboards
```

What to monitor:
- **System Metrics**: CPU, memory, disk, network
- **Application Metrics**: Request rate, latency, errors
- **Model Metrics**: Prediction distribution, accuracy, drift
- **Business Metrics**: User engagement, revenue impact

### ML-Specific Challenges

#### Model Serving
- **Latency Requirements**: Online vs batch serving
- **Model Size**: Large models (LLMs) require special handling
- **Throughput**: Requests per second capacity
- **Cost**: GPU costs for inference

Solutions:
- Model compression (quantization, distillation)
- Caching predictions
- Request batching
- Model serving frameworks (TensorFlow Serving, TorchServe, Triton)

#### Feature Engineering
- **Online Features**: Computed at request time (low latency)
- **Offline Features**: Precomputed (batch processing)
- **Feature Store**: Centralized feature management

Challenges:
- Training-serving skew
- Feature freshness
- Feature versioning
- Feature reuse across models

#### Model Training at Scale
- **Data Parallelism**: Split data across workers
- **Model Parallelism**: Split model across workers
- **Distributed Training**: Frameworks like Horovod, DeepSpeed
- **Resource Management**: Kubernetes, Kubeflow

#### Model Monitoring
- **Data Drift**: Input distribution changes
- **Concept Drift**: Relationship between inputs and outputs changes
- **Performance Degradation**: Model accuracy decreases over time

Solutions:
- Statistical tests for drift detection
- Automated retraining pipelines
- Shadow deployments for testing
- Human-in-the-loop validation

## 💡 Design Patterns

### Pattern 1: Lambda Architecture
For systems needing both real-time and batch processing

```
Data Source → Batch Layer (Hadoop/Spark) → Serving Layer
           → Speed Layer (Storm/Flink) → Serving Layer
```

Use when: Need both historical analysis and real-time processing

### Pattern 2: Kappa Architecture
Simplified alternative to Lambda (stream processing only)

```
Data Source → Stream Processing (Kafka/Flink) → Serving Layer
```

Use when: Can handle all processing in streaming fashion

### Pattern 3: Microservices for ML
Separate services for different ML tasks

```
API Gateway → Model Service A (Recommendation)
           → Model Service B (Ranking)
           → Model Service C (Personalization)
```

Use when: Multiple models, independent scaling needs

### Pattern 4: Feature Store Architecture
Centralized feature management

```
Raw Data → Feature Engineering → Feature Store → Training
                                              → Serving
```

Use when: Multiple models share features, need consistency

## 📝 Practice Scenarios

### Scenario 1: Design a Recommendation System
**Scale**: 100M users, 1M items, 1000 req/sec
**Requirements**: Personalized recommendations, <100ms latency

Key considerations:
- Collaborative filtering vs content-based
- Online vs offline computation
- Cold start problem
- Real-time updates

### Scenario 2: Design an LLM-Powered Chatbot
**Scale**: 1M daily active users
**Requirements**: Context-aware, <2s response time

Key considerations:
- LLM selection and hosting
- Context management (RAG)
- Cost optimization
- Rate limiting

### Scenario 3: Design a Real-Time Fraud Detection System
**Scale**: 10K transactions/sec
**Requirements**: <50ms latency, 99.99% availability

Key considerations:
- Real-time feature computation
- Model serving latency
- False positive handling
- Feedback loop for model improvement

### Scenario 4: Design an ML Training Platform
**Scale**: 100 data scientists, 1000 experiments/month
**Requirements**: Support distributed training, experiment tracking

Key considerations:
- Resource allocation
- Experiment management
- Model versioning
- Cost optimization

### Scenario 5: Design a Feature Store
**Scale**: 100K features, 1000 req/sec
**Requirements**: Low latency, consistency between training and serving

Key considerations:
- Storage backend
- Feature computation (online vs offline)
- Versioning
- Monitoring

### Scenario 6: Design an A/B Testing Platform for ML Models
**Scale**: 10 concurrent experiments, 1M users
**Requirements**: Statistical significance, minimal bias

Key considerations:
- Traffic splitting
- Metrics collection
- Statistical testing
- Experiment isolation

## 🎯 Design Framework (CIRCLES Method)

### C - Clarify
- Understand the problem
- Ask clarifying questions
- Define scope

### I - Identify
- Identify users
- Identify use cases
- Identify constraints

### R - Requirements
- Functional requirements
- Non-functional requirements (scale, latency, availability)
- Out of scope

### C - Components
- High-level architecture
- Major components
- Data flow

### L - List Solutions
- Multiple approaches
- Trade-offs
- Recommendation

### E - Evaluate
- Deep dive into components
- Discuss trade-offs
- Address concerns

### S - Summary
- Recap design
- Highlight key decisions
- Discuss next steps

## 💪 Interview Tips

### Do's
✅ Start with requirements clarification
✅ Draw diagrams (boxes and arrows)
✅ Think out loud
✅ Discuss trade-offs explicitly
✅ Consider failure modes
✅ Ask for feedback
✅ Be honest about what you don't know
✅ Relate to your experience when relevant

### Don'ts
❌ Jump into design without clarifying
❌ Design in silence
❌ Ignore non-functional requirements
❌ Over-engineer for current scale
❌ Dismiss interviewer's suggestions
❌ Get defensive about your design
❌ Forget about monitoring and operations

### Common Mistakes
1. **Not asking questions**: Always clarify requirements first
2. **Over-complicating**: Start simple, add complexity as needed
3. **Ignoring trade-offs**: Every decision has trade-offs
4. **Forgetting monitoring**: Production systems need observability
5. **Not considering costs**: Especially important for ML systems
6. **Ignoring data**: ML systems are data-intensive

## 📚 Study Resources

### Books
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Machine Learning Systems Design" by Chip Huyen
- "System Design Interview" by Alex Xu

### Online Resources
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [ML System Design](https://github.com/chiphuyen/machine-learning-systems-design)
- [AWS Architecture Center](https://aws.amazon.com/architecture/)

### Practice
- Draw diagrams for systems you use daily
- Read engineering blogs (Netflix, Uber, Airbnb)
- Review open-source ML platforms (Kubeflow, MLflow)

## 🎓 Key Takeaways

1. **Requirements First**: Always clarify before designing
2. **Start Simple**: Begin with basic design, add complexity
3. **Think Scale**: Consider how system grows
4. **Trade-offs Matter**: No perfect solution, only appropriate ones
5. **ML is Different**: ML systems have unique challenges
6. **Production Focus**: Monitoring, cost, operations matter
7. **Communication**: Clear explanation is as important as good design

---

Ready to practice? Start with Scenario 1 and work through each one!

