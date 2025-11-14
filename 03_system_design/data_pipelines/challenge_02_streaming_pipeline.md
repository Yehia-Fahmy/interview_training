# Challenge 02: Streaming Pipeline

## Problem Statement

Design a real-time streaming data pipeline that can:
- Process millions of events per second
- Handle multiple event sources
- Transform and enrich events in real-time
- Support multiple downstream consumers
- Handle backpressure and failures
- Maintain exactly-once or at-least-once semantics

## Requirements to Clarify

**Ask these questions before designing:**

### Functional Requirements
- What types of events? (user actions, system metrics, business events)
- What transformations are needed? (filtering, enrichment, aggregation)
- What are the downstream consumers? (databases, data warehouses, real-time dashboards)
- Do we need exactly-once processing?

### Non-Functional Requirements
- What's the event volume? (events/sec)
- What's the latency requirement? (< 100ms, < 1s?)
- What's the throughput requirement?
- What's the acceptable data loss? (zero, minimal)

### Constraints
- What's the event format? (structured, semi-structured)
- Are there ordering requirements?
- What's the budget?
- What's the event size? (small, large)

## Design Considerations

### Core Components

1. **Event Sources**
   - Multiple producers
   - Event ingestion
   - Schema management

2. **Message Queue/Stream**
   - Buffer events
   - Handle backpressure
   - Durability guarantees

3. **Stream Processing**
   - Real-time transformations
   - Windowing and aggregations
   - State management

4. **Downstream Sinks**
   - Multiple consumers
   - Different formats
   - Different latencies

5. **Monitoring**
   - Processing lag
   - Error tracking
   - Throughput monitoring

### Key Design Decisions

#### 1. Message Queue/Stream

**Apache Kafka:**
- High throughput
- Durability
- Exactly-once semantics
- Good for: High-volume, multiple consumers

**AWS Kinesis:**
- Managed service
- Auto-scaling
- Good for: AWS ecosystem

**RabbitMQ:**
- Message routing
- Lower throughput
- Good for: Complex routing, lower volume

**Recommendation**: Kafka for high-volume, Kinesis for managed service

#### 2. Stream Processing

**Apache Flink:**
- Low latency
- Stateful processing
- Exactly-once semantics
- Good for: Complex processing, low latency

**Apache Kafka Streams:**
- Simple, Kafka-native
- Good for: Simple transformations, Kafka ecosystem

**Apache Storm:**
- Mature, but lower-level
- Good for: Custom processing

**Recommendation**: Flink for complex processing, Kafka Streams for simplicity

#### 3. Processing Semantics

**At-Least-Once:**
- Events processed at least once
- May have duplicates
- Simpler, faster
- Good for: Idempotent operations

**Exactly-Once:**
- Events processed exactly once
- No duplicates
- More complex, slower
- Good for: Financial transactions, critical operations

**Recommendation**: At-least-once for most cases, exactly-once for critical operations

#### 4. Windowing

**Tumbling Windows:**
- Fixed-size, non-overlapping
- Simple, efficient
- Good for: Regular aggregations

**Sliding Windows:**
- Fixed-size, overlapping
- More complex
- Good for: Smooth aggregations

**Session Windows:**
- Variable-size based on activity
- Complex
- Good for: User sessions

**Recommendation**: Tumbling windows for most cases

#### 5. State Management

**In-Memory State:**
- Fast, but limited
- Lost on failure
- Good for: Small state

**External State Store:**
- RocksDB, Redis
- Persistent, scalable
- Good for: Large state, fault tolerance

**Recommendation**: External state store for production

## High-Level Architecture

```
[Event Sources]
    ├── [User Actions]
    ├── [System Metrics]
    └── [Business Events]
         ↓
[Message Queue/Stream] (Kafka, Kinesis)
    ↓
[Stream Processing] (Flink, Kafka Streams)
    ├── [Filtering]
    ├── [Enrichment]
    ├── [Aggregation]
    └── [Windowing]
         ↓
[Downstream Sinks]
    ├── [Real-time Database]
    ├── [Data Warehouse]
    ├── [Alerting System]
    └── [Dashboard]
```

## Detailed Design

### Event Ingestion

**Event Format:**
```json
{
  "event_id": "evt_123",
  "event_type": "page_view",
  "timestamp": "2024-01-01T00:00:00Z",
  "user_id": "user_456",
  "properties": {
    "page": "/home",
    "duration": 30
  }
}
```

**Ingestion:**
- Multiple producers
- Schema validation
- Rate limiting
- Partitioning strategy

**Partitioning:**
- Partition by key (user_id, event_type)
- Ensures ordering per partition
- Enables parallel processing

### Stream Processing

**Processing Tasks:**

1. **Filtering:**
   - Filter irrelevant events
   - Reduce downstream load

2. **Enrichment:**
   - Join with reference data
   - Add context
   - Lookup external data

3. **Transformation:**
   - Format conversion
   - Field mapping
   - Data cleaning

4. **Aggregation:**
   - Count events
   - Sum metrics
   - Calculate averages
   - Windowed aggregations

**State Management:**
- Maintain state for aggregations
- Checkpoint state periodically
- Recover from failures

**Windowing:**
- Tumbling windows (1min, 5min, 1hour)
- Sliding windows
- Session windows

### Downstream Sinks

**Types:**

1. **Real-time Database:**
   - Update user profiles
   - Update counters
   - Low latency (< 100ms)

2. **Data Warehouse:**
   - Batch loading
   - Analytics
   - Higher latency (minutes)

3. **Alerting System:**
   - Real-time alerts
   - Threshold-based
   - Low latency (< 1s)

4. **Dashboard:**
   - Real-time metrics
   - WebSocket updates
   - Low latency (< 1s)

### Backpressure Handling

**Problem:**
- Downstream slower than upstream
- Buffer fills up
- Need to slow down processing

**Solutions:**

1. **Backpressure Propagation:**
   - Slow down upstream
   - Reduce processing rate

2. **Buffering:**
   - Buffer events
   - Handle temporary spikes
   - Risk of memory issues

3. **Dropping Events:**
   - Drop low-priority events
   - Last resort

**Recommendation**: Backpressure propagation + buffering

### Failure Handling

**Failure Scenarios:**

1. **Processing Failure:**
   - Retry processing
   - Dead letter queue
   - Alert on failures

2. **Node Failure:**
   - Automatic failover
   - State recovery
   - Continue processing

3. **Downstream Failure:**
   - Retry sending
   - Buffer events
   - Dead letter queue

**Recovery:**
- Checkpoint state
- Replay from checkpoint
- Exactly-once processing

## Scaling Strategy

### Start Small
- Single stream processor
- Single consumer
- Basic processing
- ~10K events/sec

### Scale Gradually

**Phase 1: Add Parallelism**
- Multiple processing tasks
- Parallel consumers
- ~100K events/sec

**Phase 2: Distributed Processing**
- Distributed stream processing
- Multiple partitions
- ~1M events/sec

**Phase 3: Optimize**
- Optimize processing logic
- Efficient state management
- Multi-region deployment
- ~10M events/sec

## Trade-offs

### Latency vs Throughput
- **Lower latency**: Smaller batches, more resources
- **Higher throughput**: Larger batches, fewer resources
- **Recommendation**: Balance based on requirements

### Exactly-Once vs At-Least-Once
- **Exactly-once**: More complex, slower, no duplicates
- **At-least-once**: Simpler, faster, may have duplicates
- **Recommendation**: At-least-once for most cases, exactly-once for critical

### Stateful vs Stateless
- **Stateful**: More complex, enables aggregations
- **Stateless**: Simpler, limited functionality
- **Recommendation**: Stateful for aggregations, stateless for simple transformations

## Operational Concerns

### Monitoring
- Processing lag
- Throughput (events/sec)
- Error rate
- Latency
- State size

### Debugging
- Event tracing
- Processing logs
- Error logs
- State inspection

### Data Quality
- Schema validation
- Data completeness
- Anomaly detection

## Follow-up Questions

**Be prepared to answer:**
1. "How do you handle backpressure?"
2. "How would you ensure exactly-once processing?"
3. "How do you handle late-arriving events?"
4. "How would you optimize processing latency?"
5. "How do you handle state recovery?"
6. "How would you handle schema evolution?"
7. "How do you handle duplicate events?"

## Example Solutions

### Simple Solution (Start Here)
- Single stream processor
- Basic transformations
- Single consumer
- ~10K events/sec

### Optimized Solution
- Distributed processing
- Stateful aggregations
- Multiple consumers
- Backpressure handling
- ~1M events/sec

### Production Solution
- Advanced stream processing
- Exactly-once semantics
- Comprehensive monitoring
- Multi-region deployment
- Fault tolerance
- ~10M events/sec

## Practice Exercise

**Design this system in 45-60 minutes:**
1. Clarify requirements (5-10 min)
2. High-level design (10-15 min)
3. Detailed design (15-20 min)
4. Scaling and optimization (10-15 min)
5. Trade-offs discussion (5 min)

**Focus on:**
- Stream processing architecture
- Message queue selection
- Processing semantics
- State management
- Failure handling

