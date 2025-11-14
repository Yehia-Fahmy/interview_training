# System Design Interview Guide

## How to Approach the Interview

This guide provides a structured framework for handling system design interviews, with specific focus on ML/data-intensive systems.

---

## The Interview Structure

A typical system design interview follows this flow:

1. **Problem Clarification** (5-10 min)
2. **High-Level Design** (10-15 min)
3. **Detailed Design** (15-20 min)
4. **Scale & Optimization** (10-15 min)
5. **Trade-offs & Wrap-up** (5 min)

---

## Step-by-Step Framework

### Step 1: Problem Clarification (CRITICAL!)

**Never jump into solutions immediately!** Always clarify requirements first.

#### Questions to Ask:

**Functional Requirements:**
- What is the core functionality?
- What are the main use cases?
- What are the input/output formats?
- Are there any specific features required?

**Non-Functional Requirements:**
- What's the expected scale? (users, requests/sec, data volume)
- What are the latency requirements?
- What are the availability requirements? (99.9%, 99.99%?)
- What are the consistency requirements?
- Are there any cost constraints?

**Constraints & Assumptions:**
- What technologies can/can't we use?
- Are there any compliance requirements?
- What's the expected growth rate?
- Are there any geographic constraints?

**Example Questions:**
- "What's the expected read/write ratio?"
- "What's the acceptable latency for this operation?"
- "Do we need strong consistency or is eventual consistency okay?"
- "What's the expected data growth rate?"
- "Are there any specific compliance requirements (GDPR, HIPAA)?"

#### Common ML-Specific Questions:
- "What's the model update frequency?"
- "Do we need real-time or batch inference?"
- "What's the acceptable model accuracy vs latency trade-off?"
- "How do we handle model versioning?"
- "What monitoring metrics are most important?"

---

### Step 2: High-Level Design

Start with a **simple, working solution**, then iterate.

#### Components to Identify:

1. **User Interface/API Layer**
   - REST APIs, GraphQL, gRPC
   - API Gateway
   - Load balancer

2. **Application Layer**
   - Business logic
   - Microservices
   - Message queues

3. **Data Layer**
   - Databases (SQL, NoSQL)
   - Caches
   - Object storage

4. **External Services**
   - Third-party APIs
   - CDN
   - Monitoring services

#### For ML Systems, Also Consider:

- **Training Pipeline**
  - Data ingestion
  - Feature engineering
  - Model training
  - Model validation

- **Serving Infrastructure**
  - Model server
  - Feature store
  - Inference API

- **Monitoring**
  - Model performance metrics
  - Data drift detection
  - System health

#### Drawing the Diagram:

```
[Users] → [Load Balancer] → [API Gateway] → [Application Servers]
                                                      ↓
                                              [Message Queue]
                                                      ↓
                                              [ML Service]
                                                      ↓
                                    [Feature Store] ← [Model Server]
                                                      ↓
                                              [Database]
                                                      ↓
                                              [Cache]
```

**Tips:**
- Use clear boxes and arrows
- Label components
- Show data flow
- Indicate read/write paths
- Show caching layers

---

### Step 3: Detailed Design

Deep dive into each component.

#### Database Design:

**Questions to Address:**
- What data model? (relational, document, key-value)
- What's the schema?
- What are the indexes?
- How do we partition/shard?
- What's the replication strategy?

**For ML Systems:**
- Feature storage schema
- Model metadata storage
- Experiment tracking data
- Prediction logs

#### API Design:

**Questions to Address:**
- What are the endpoints?
- What's the request/response format?
- What's the authentication/authorization?
- Rate limiting strategy?

**For ML Systems:**
- Inference endpoint design
- Batch vs real-time endpoints
- Model versioning in API
- Feature serving endpoints

#### Caching Strategy:

**Questions to Address:**
- What to cache?
- Cache eviction policy (LRU, LFU, TTL)
- Cache invalidation strategy
- Multi-level caching?

**For ML Systems:**
- Model prediction caching
- Feature caching
- Prompt caching (for LLMs)

#### Message Queue/Event Streaming:

**Questions to Address:**
- What events to publish?
- Consumer groups?
- Message ordering requirements?
- Dead letter queues?

---

### Step 4: Scale & Optimization

Discuss how the system scales and handles bottlenecks.

#### Identify Bottlenecks:

1. **Database**
   - Read replicas
   - Sharding
   - Caching
   - Denormalization

2. **Application Servers**
   - Horizontal scaling
   - Load balancing
   - Stateless design

3. **Network**
   - CDN
   - Edge caching
   - Compression

4. **ML-Specific:**
   - Model serving optimization (batching, quantization)
   - Feature computation optimization
   - Distributed training

#### Scaling Strategy:

**Start Small:**
- Single server
- Single database
- No caching

**Scale Gradually:**
- Add load balancer
- Add read replicas
- Add caching
- Add CDN
- Shard database
- Add message queues

**For ML Systems:**
- Scale inference servers
- Optimize model serving (batching)
- Use GPU clusters for training
- Implement feature caching

#### Capacity Estimation:

**Back-of-envelope calculations:**

- **Traffic estimates:**
  - Users: 1M DAU
  - Requests per user: 10/day
  - Total requests: 10M/day = ~115 requests/sec
  - Peak traffic: 3x = ~350 requests/sec

- **Storage estimates:**
  - Data per user: 1KB
  - Total users: 1M
  - Total storage: 1GB
  - With replication (3x): 3GB

- **Bandwidth estimates:**
  - Request size: 1KB
  - Response size: 10KB
  - Requests/sec: 350
  - Bandwidth: 350 * (1KB + 10KB) = ~3.85 MB/sec

**Always state your assumptions!**

---

### Step 5: Trade-offs & Wrap-up

Discuss pros/cons and operational concerns.

#### Common Trade-offs:

1. **Consistency vs Availability**
   - Strong consistency: Lower availability, simpler logic
   - Eventual consistency: Higher availability, more complex

2. **Latency vs Throughput**
   - Lower latency: More servers, higher cost
   - Higher throughput: Batching, higher latency

3. **SQL vs NoSQL**
   - SQL: ACID, complex queries, harder to scale
   - NoSQL: Flexible schema, easier to scale, eventual consistency

4. **Monolith vs Microservices**
   - Monolith: Simpler, faster development, harder to scale
   - Microservices: Scalable, complex, network overhead

5. **Batch vs Real-time**
   - Batch: Simpler, lower cost, higher latency
   - Real-time: Complex, higher cost, lower latency

#### Operational Concerns:

**Always Discuss:**
- **Monitoring**: What metrics to track?
- **Logging**: What to log? Where?
- **Alerting**: What alerts? Who gets notified?
- **Debugging**: How to trace issues?
- **Disaster Recovery**: Backup strategy? Failover?
- **Security**: Authentication? Authorization? Encryption?
- **Cost**: How to optimize costs?

**For ML Systems:**
- Model performance monitoring
- Data drift detection
- Model versioning and rollback
- A/B testing infrastructure
- Cost of model inference

---

## Communication Tips

### Do's:

✅ **Think Out Loud**
- Explain your reasoning
- "I'm considering X because..."
- "The trade-off here is..."

✅ **Ask Questions**
- Clarify requirements
- "Can you clarify..."
- "I'm assuming..."

✅ **Start Simple**
- "Let's start with a simple design..."
- "We can optimize later..."

✅ **Use Examples**
- "For example, if a user..."
- "In this scenario..."

✅ **Acknowledge Limitations**
- "This design has limitations..."
- "We could improve this by..."

### Don'ts:

❌ **Don't Jump to Solutions**
- Always clarify first

❌ **Don't Over-Engineer**
- Start simple, iterate

❌ **Don't Ignore Scale**
- Always consider growth

❌ **Don't Forget Operations**
- Monitoring, debugging, failures

❌ **Don't Be Silent**
- Keep talking, explain your thinking

---

## Common Interview Scenarios

### Scenario 1: Scalable API Design

**Approach:**
1. Clarify: Read/write ratio? Latency requirements?
2. Design: API → Load balancer → App servers → Database
3. Scale: Add caching, read replicas, CDN
4. Discuss: Consistency, availability trade-offs

### Scenario 2: ML Model Serving

**Approach:**
1. Clarify: Real-time or batch? Latency requirements? Model size?
2. Design: API → Model server → Feature store → Database
3. Scale: Batching, model optimization, horizontal scaling
4. Discuss: Model versioning, A/B testing, monitoring

### Scenario 3: Real-time Analytics

**Approach:**
1. Clarify: Data volume? Latency requirements? Query patterns?
2. Design: Data source → Stream processor → Storage → Query API
3. Scale: Partitioning, indexing, caching
4. Discuss: Lambda vs Kappa architecture, data quality

### Scenario 4: Recommendation System

**Approach:**
1. Clarify: Real-time or batch? Personalization level? Data sources?
2. Design: User → API → Recommendation service → Feature store → Models
3. Scale: Caching, pre-computation, two-stage retrieval
4. Discuss: Cold start problem, freshness vs accuracy

---

## Handling Follow-up Questions

### "What if X happens?"

**Framework:**
1. Acknowledge the scenario
2. Explain the impact
3. Propose a solution
4. Discuss trade-offs

**Example:**
- **Q**: "What if the database goes down?"
- **A**: "We'd lose availability for writes. We can mitigate this by:
  1. Having read replicas for read availability
  2. Implementing a write queue to buffer writes
  3. Using a multi-region setup for disaster recovery
  The trade-off is increased complexity and cost."

### "How would you optimize X?"

**Framework:**
1. Identify the bottleneck
2. Propose optimization
3. Explain the impact
4. Discuss trade-offs

**Example:**
- **Q**: "How would you optimize model inference latency?"
- **A**: "We can optimize in several ways:
  1. Model quantization to reduce size
  2. Batching requests to improve throughput
  3. Using GPUs for faster inference
  4. Caching frequent predictions
  The trade-off is between cost (GPUs) and latency."

---

## Red Flags to Avoid

1. **Not asking questions** - Shows lack of critical thinking
2. **Over-engineering** - Shows poor judgment
3. **Ignoring scale** - Shows lack of experience
4. **Forgetting operations** - Shows lack of production experience
5. **Not discussing trade-offs** - Shows lack of depth
6. **Being too quiet** - Makes it hard to evaluate

---

## Final Checklist

Before the interview:
- [ ] Review common system design patterns
- [ ] Practice drawing diagrams
- [ ] Practice explaining designs out loud
- [ ] Review ML system architectures
- [ ] Prepare questions to ask

During the interview:
- [ ] Clarify requirements first
- [ ] Start with simple design
- [ ] Think out loud
- [ ] Discuss trade-offs
- [ ] Consider operational concerns
- [ ] Ask follow-up questions

After the interview:
- [ ] Reflect on what went well
- [ ] Identify areas for improvement
- [ ] Review the design you proposed
- [ ] Research better solutions if needed

---

## Practice Exercises

1. **Design a URL shortener** (classic starter)
2. **Design a chat system** (real-time, scalability)
3. **Design a search engine** (indexing, ranking)
4. **Design a video streaming service** (CDN, encoding)
5. **Design an LLM serving system** (ML-specific)
6. **Design a recommendation system** (ML-specific)
7. **Design a real-time analytics dashboard** (data pipeline)

Practice each one:
- Time yourself (45-60 min)
- Draw diagrams
- Explain out loud
- Discuss trade-offs
- Consider scale

---

## Remember

The goal isn't to design a perfect system - it's to demonstrate:
- **Problem-solving skills**
- **Technical depth**
- **Communication ability**
- **Operational thinking**
- **Trade-off analysis**

Good luck! 🚀

