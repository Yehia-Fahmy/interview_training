# System Design Quick Start Guide

**For limited preparation time - focus on these essentials**

## 🎯 Interview Framework (5 Steps - 45-60 min)

### Step 1: Clarify Requirements (5-10 min)
**CRITICAL - Never skip this!**

Ask:
- **Functional**: What does it do? Main use cases?
- **Scale**: Users? Requests/sec? Data volume?
- **Non-functional**: Latency? Availability? Consistency?
- **Constraints**: Budget? Technologies? Compliance?

### Step 2: High-Level Design (10-15 min)
Draw simple boxes:
- **API/Load Balancer** → **App Servers** → **Database/Cache**
- For ML: Add **Model Server**, **Feature Store**, **Training Pipeline**

### Step 3: Detailed Design (15-20 min)
Deep dive:
- **Database**: SQL vs NoSQL? Sharding? Replication?
- **Caching**: What to cache? Where? (CDN, Redis, DB cache)
- **Scaling**: Horizontal vs vertical? Load balancing?

### Step 4: Scale & Optimize (10-15 min)
Discuss bottlenecks:
- **Database**: Read replicas, sharding, caching
- **Application**: Horizontal scaling, stateless design
- **ML-specific**: Batching, model optimization, feature caching

### Step 5: Trade-offs (5 min)
Always discuss:
- Consistency vs Availability
- Latency vs Cost
- SQL vs NoSQL
- Batch vs Real-time

---

## 📚 Must-Know Concepts (Cheat Sheet)

### Distributed Systems
- **CAP Theorem**: Can only have 2 of 3 (Consistency, Availability, Partition tolerance)
- **Scaling**: Horizontal (add servers) vs Vertical (bigger servers)
- **Load Balancing**: Round-robin, least connections, consistent hashing
- **Caching**: Multi-level (CDN → Redis → DB cache)
- **Database**: Read replicas (scale reads), Sharding (scale writes)

### ML Systems
- **Two-Stage**: Candidate generation (fast, approximate) → Ranking (slow, precise)
- **Feature Store**: Offline (batch) + Online (real-time) for consistency
- **Model Serving**: Batching, quantization, caching
- **Training Pipeline**: Distributed training, experiment tracking, model registry

### Data Pipelines
- **ETL**: Extract → Transform → Load (transform before loading)
- **ELT**: Extract → Load → Transform (transform in destination)
- **Lambda**: Batch + Streaming layers
- **Kappa**: Streaming-only (reprocess for history)

---

## 🎯 Priority Challenges (Focus on These)

### ML Systems (Most Important for Your Role)
1. **LLM Serving** (`ml_systems/challenge_01_llm_serving.md`)
   - Key: Batching, caching, tiered serving, cost optimization
   
2. **Feature Store** (`ml_systems/challenge_03_feature_store.md`)
   - Key: Offline vs online, consistency, real-time computation

3. **Model Training Pipeline** (`ml_systems/challenge_02_model_training_pipeline.md`)
   - Key: Distributed training, experiment tracking, failure handling

### Distributed Systems (Quick Practice)
1. **Scalable API** (`distributed_systems/challenge_01_scalable_api.md`)
   - Key: Load balancing, caching, database scaling

---

## 💡 Quick Tips

### Do's ✅
- **Always clarify requirements first** (5-10 min)
- **Start simple**, then iterate
- **Think out loud** - explain your reasoning
- **Discuss trade-offs** - every decision has pros/cons
- **Consider operations** - monitoring, debugging, failures

### Don'ts ❌
- Don't jump to solutions immediately
- Don't over-engineer (start simple)
- Don't ignore scale (think from 1 user to millions)
- Don't forget operations (monitoring, debugging)

---

## 📖 Quick Reference: Common Patterns

### Scalable API Pattern
```
Users → Load Balancer → API Gateway → App Servers → Cache → Database
                                                          ↓
                                                    Read Replicas
```

### ML Serving Pattern
```
Users → API Gateway → Model Server → Feature Store → Database
                              ↓
                         GPU Cluster
                              ↓
                         Response Cache
```

### Two-Stage Recommendation
```
Request → Candidate Generation (1000s) → Ranking (Top-N) → Response
```

---

## ⏱️ 1-Week Crash Course

### Day 1-2: Learn Framework
- Read `INTERVIEW_GUIDE.md` (focus on 5-step framework)
- Practice clarifying requirements (ask questions)

### Day 3-4: Practice ML Challenges
- **Day 3**: LLM Serving (most relevant)
- **Day 4**: Feature Store + Model Training Pipeline

### Day 5: Practice Distributed Systems
- Scalable API challenge

### Day 6-7: Mock Practice
- Time yourself (45-60 min)
- Practice explaining out loud
- Review case studies (LLM Serving, Recommendation System)

---

## 🎯 Interview Day Checklist

- [ ] Review 5-step framework
- [ ] Practice clarifying questions
- [ ] Review common patterns (above)
- [ ] Practice drawing simple diagrams
- [ ] Prepare to discuss trade-offs

---

## 📝 Common Questions to Ask

**Always start with these:**
1. "What's the expected scale?" (users, QPS, data volume)
2. "What are the latency requirements?" (p95, p99)
3. "What's the read/write ratio?"
4. "Do we need strong consistency or is eventual consistency okay?"
5. "What's the availability requirement?" (99.9%, 99.99%)

**For ML systems:**
1. "Real-time or batch inference?"
2. "What's the model update frequency?"
3. "How do we ensure feature consistency between training and serving?"
4. "What's the acceptable latency vs cost trade-off?"

---

**Remember**: The goal isn't a perfect design - it's demonstrating problem-solving, technical depth, and communication skills. Focus on the framework and practice explaining your reasoning!

