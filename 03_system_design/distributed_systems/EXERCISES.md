# System Design - Distributed Systems Exercises

These exercises are **discussion-based** (no coding). Practice explaining your design decisions, trade-offs, and architecture choices.

---

## Exercise 1: Design a Scalable Web Crawler

**Difficulty:** Medium  
**Time Limit:** 45-60 minutes (discussion)  
**Focus:** Distributed systems, scaling, reliability

### Problem

Design a web crawler system that can:
- Crawl billions of web pages
- Handle rate limiting and politeness policies
- Avoid duplicates
- Scale horizontally
- Handle failures gracefully

### Key Topics to Cover

- **Distributed Queue**: RabbitMQ, Kafka, or custom
- **Deduplication**: Bloom filters, distributed hash tables
- **Partitioning**: Domain-based, hash-based
- **Consistency**: Eventual consistency patterns
- **Monitoring**: Health checks, metrics

### Discussion Points

1. URL Frontier/Queue management
2. Distributed crawling architecture
3. Politeness & rate limiting per domain
4. Storage for billions of pages
5. Fault tolerance and recovery

---

## Exercise 2: Design a Distributed Cache System

**Difficulty:** Medium-Hard  
**Time Limit:** 45-60 minutes  
**Focus:** Caching strategies, consistency, scalability

### Problem

Design a distributed cache system (similar to Redis or Memcached) that:
- Handles millions of requests per second
- Scales horizontally
- Maintains high availability
- Handles cache invalidation
- Supports different eviction policies

### Key Topics to Cover

- **Consistent Hashing**: Virtual nodes, ring structure
- **Replication**: Master-slave, master-master strategies
- **Eviction Policies**: LRU, LFU, FIFO implementation
- **High Availability**: Failure handling, network partitions
- **Performance**: Latency optimization, serialization

### Discussion Points

1. Architecture (client-server vs peer-to-peer)
2. Data partitioning and sharding
3. Replication strategies
4. Eviction policies
5. High availability and fault tolerance

---

## Exercise 3: Design a Real-time Analytics System

**Difficulty:** Medium-Hard  
**Time Limit:** 45-60 minutes  
**Focus:** Stream processing, real-time computation, scalability

### Problem

Design a real-time analytics system that:
- Processes millions of events per second
- Computes aggregations in real-time
- Supports time-windowed queries
- Scales to handle traffic spikes
- Provides low-latency queries

### Key Topics to Cover

- **Stream Processing**: Kafka, Flink, Spark Streaming
- **Time Windows**: Tumbling, sliding, session windows
- **Storage**: Time-series databases, columnar storage
- **Aggregation**: Pre-computation vs on-demand
- **Query Layer**: Real-time dashboards, alerting

### Discussion Points

1. Event ingestion and processing
2. Real-time aggregation architecture
3. Storage for time-series data
4. Query optimization
5. Scaling and fault tolerance

---

## Exercise 4: Design a Message Queue System

**Difficulty:** Medium-Hard  
**Time Limit:** 45-60 minutes  
**Focus:** Message queues, pub-sub, reliability

### Problem

Design a distributed message queue system that:
- Handles millions of messages per second
- Supports multiple consumers
- Guarantees message delivery
- Handles failures gracefully
- Supports different delivery semantics

### Key Topics to Cover

- **Message Storage**: Durable storage, replication
- **Delivery Guarantees**: At-least-once, exactly-once, at-most-once
- **Consumer Groups**: Load balancing, parallelism
- **Ordering**: Per-partition ordering, global ordering
- **Dead Letter Queues**: Error handling

### Discussion Points

1. Message storage and persistence
2. Producer and consumer coordination
3. Delivery guarantees and ordering
4. Scaling and partitioning
5. Failure handling and recovery

---

## How to Practice

1. **Read the problem** carefully
2. **Design the system** on paper/whiteboard (no code)
3. **Explain your decisions** out loud (as in interview)
4. **Discuss trade-offs** for each design choice
5. **Consider scale** (1M, 10M, 1B users/data points)
6. **Review** common patterns and solutions

## Evaluation Criteria (8090 Interview)

- Technical depth of your design
- Ability to clarify technical details
- Rationale behind design choices
- Understanding of distributed systems concepts
- Scalability and reliability considerations

