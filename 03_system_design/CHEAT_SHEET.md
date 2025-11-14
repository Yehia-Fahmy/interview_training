# System Design Cheat Sheet

## 🎯 5-Step Interview Framework

1. **Clarify** (5-10 min): Ask about scale, latency, availability, constraints
2. **High-Level** (10-15 min): Draw simple boxes (API → App → DB)
3. **Detailed** (15-20 min): Database, caching, scaling strategies
4. **Scale** (10-15 min): Bottlenecks, optimization, multi-region
5. **Trade-offs** (5 min): Consistency vs Availability, Latency vs Cost

---

## 📊 Key Concepts

### CAP Theorem
- **Consistency**: All nodes see same data
- **Availability**: System responds to requests
- **Partition Tolerance**: System works despite network failures
- **Pick 2**: Usually choose Availability + Partition Tolerance (eventual consistency)

### Scaling Strategies
- **Horizontal**: Add more servers (preferred)
- **Vertical**: Bigger servers (limited)
- **Read Replicas**: Scale reads
- **Sharding**: Scale writes (partition data)

### Caching Layers
1. **CDN**: Static content, edge caching
2. **Application Cache**: Redis, frequently accessed data
3. **Database Cache**: Query results

### Database Types
- **SQL**: ACID, complex queries, harder to scale (PostgreSQL, MySQL)
- **NoSQL**: Flexible, easier to scale, eventual consistency
  - **Document**: MongoDB (flexible schemas)
  - **Key-Value**: Redis (caching, sessions)
  - **Column**: Cassandra (wide tables)
  - **Time-Series**: InfluxDB (metrics)

---

## 🤖 ML System Patterns

### Two-Stage Architecture
```
Candidate Generation (fast, 1000s) → Ranking (slow, top-N)
```

### Feature Store
- **Offline**: Batch features for training (Data Warehouse)
- **Online**: Real-time features for serving (Redis/DynamoDB)
- **Key**: Ensure consistency between training and serving

### Model Serving
- **Batching**: Group requests to maximize GPU utilization
- **Caching**: Cache predictions, prompts, features
- **Optimization**: Quantization (FP16, INT8), model pruning

### Training Pipeline
- **Distributed**: Data parallelism (split data), Model parallelism (split model)
- **Tracking**: MLflow, experiment tracking, model registry
- **Failure**: Checkpointing, retry logic

---

## 🔄 Data Pipeline Patterns

### ETL vs ELT
- **ETL**: Transform before loading (complex transformations)
- **ELT**: Load first, transform in destination (large datasets)

### Lambda Architecture
- **Batch Layer**: Historical data processing
- **Speed Layer**: Real-time processing
- **Serving Layer**: Merge results

### Kappa Architecture
- **Single Stream**: Process everything as stream
- **Reprocess**: Reprocess for historical queries

---

## 🏗️ Common Architectures

### Scalable API
```
Users → CDN → Load Balancer → API Gateway → App Servers → Cache → DB
                                                                    ↓
                                                              Read Replicas
```

### ML Serving
```
Users → API Gateway → Model Server → Feature Store → Database
                              ↓
                         GPU Cluster
                              ↓
                         Response Cache
```

### Recommendation System
```
Request → Candidate Gen (ANN) → Ranking (ML Model) → Post-process → Response
```

---

## 💬 Common Trade-offs

| Decision | Option A | Option B |
|----------|----------|----------|
| **Consistency** | Strong (slower, simpler) | Eventual (faster, complex) |
| **Scaling** | Horizontal (preferred) | Vertical (limited) |
| **Database** | SQL (ACID, complex queries) | NoSQL (flexible, scalable) |
| **Caching** | More (higher memory, lower compute) | Less (lower memory, higher compute) |
| **Latency** | Lower (more servers, higher cost) | Higher (fewer servers, lower cost) |
| **ML Serving** | Real-time (lower latency, higher cost) | Batch (higher latency, lower cost) |

---

## ❓ Must-Ask Questions

### General
- What's the expected scale? (users, QPS, data volume)
- What are latency requirements? (p50, p95, p99)
- What's the read/write ratio?
- Strong consistency or eventual consistency?
- Availability requirement? (99.9%, 99.99%)

### ML-Specific
- Real-time or batch inference?
- Model update frequency?
- How ensure feature consistency?
- Acceptable latency vs cost trade-off?

---

## 🎯 Quick Patterns

### Load Balancing
- **Round-robin**: Even distribution
- **Least connections**: Better for long-lived connections
- **Consistent hashing**: For sticky sessions

### Caching Strategies
- **Cache-aside**: App checks cache, loads from DB if miss
- **Write-through**: Write to cache and DB simultaneously
- **Write-behind**: Write to cache, async to DB

### Database Sharding
- **Hash-based**: Hash key → shard (even distribution)
- **Range-based**: Range of keys → shard (natural partitioning)
- **Directory-based**: Lookup table (flexible, but SPOF)

### Failure Handling
- **Retry**: Exponential backoff
- **Circuit Breaker**: Stop calling failing service
- **Graceful Degradation**: Fallback to simpler functionality

---

## 📈 Capacity Estimation (Back-of-Envelope)

### Example: 1M DAU, 10 requests/user/day
- **Requests/day**: 1M × 10 = 10M
- **Requests/sec**: 10M / 86400 ≈ 115 req/sec
- **Peak (3x)**: ~350 req/sec

### Storage: 1M users, 1KB/user
- **Total**: 1M × 1KB = 1GB
- **With replication (3x)**: 3GB

### Bandwidth: 350 req/sec, 1KB request, 10KB response
- **Bandwidth**: 350 × (1KB + 10KB) ≈ 3.85 MB/sec

---

## 🚨 Red Flags to Avoid

1. ❌ Not asking clarifying questions
2. ❌ Jumping to solutions immediately
3. ❌ Over-engineering (start simple!)
4. ❌ Ignoring scale (think from 1 to millions)
5. ❌ Forgetting operations (monitoring, debugging)
6. ❌ Not discussing trade-offs

---

## ✅ Interview Success Formula

1. **Clarify** requirements thoroughly (5-10 min)
2. **Start simple**, then iterate
3. **Think out loud** - explain reasoning
4. **Discuss trade-offs** - every decision has pros/cons
5. **Consider operations** - monitoring, debugging, failures
6. **Scale gradually** - from 1 user to millions

**Remember**: Demonstrate problem-solving, technical depth, and communication - not perfection!

