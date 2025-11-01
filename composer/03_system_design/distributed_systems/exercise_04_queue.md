# Exercise 4: Design a Message Queue System

**Difficulty:** Medium-Hard  
**Time Limit:** 45-60 minutes  
**Focus:** Message queues, ordering guarantees, durability

## Problem

Design a distributed message queue system (similar to RabbitMQ or Kafka) that:
- Handles millions of messages per second
- Supports multiple producers and consumers
- Provides ordering guarantees
- Ensures message durability
- Supports different delivery semantics

## Requirements to Discuss

1. **Message Model**
   - Topic-based? Queue-based?
   - Pub-sub vs point-to-point?
   - How to organize messages?

2. **Ordering Guarantees**
   - Global ordering vs per-partition ordering?
   - How to maintain order under failures?
   - Idempotency?

3. **Durability**
   - Where to store messages?
   - Replication strategy?
   - Write-ahead logs?

4. **Consumer Groups**
   - How to distribute messages to consumers?
   - Load balancing?
   - How to handle consumer failures?

5. **Scalability**
   - How to partition topics/queues?
   - Horizontal scaling?
   - Rebalancing when consumers join/leave?

6. **Delivery Semantics**
   - At-least-once, at-most-once, exactly-once?
   - How to implement each?
   - Trade-offs?

## Key Topics to Cover

- **Partitioning**: Hash-based, range-based
- **Replication**: Leader-follower, quorum
- **Consensus**: Raft, Paxos for leader election
- **Log Structure**: Append-only logs, compaction
- **Consumer Coordination**: ZooKeeper, custom coordinator

## Sample Discussion Points

1. "I'd use a partitioned topic model. Messages are assigned to partitions based on a key. This allows parallelism while maintaining order per partition."

2. "For durability, I'd use write-ahead logs. Messages are appended to logs on multiple replicas before acknowledging to producer."

3. "Consumer groups would subscribe to partitions. Each partition is consumed by one consumer in the group. If a consumer fails, its partitions are rebalanced to other consumers."

4. "For ordering, I'd maintain order per partition. Global ordering would be too expensive. If global ordering is needed, use a single partition."

5. "Exactly-once delivery is tricky. I'd use idempotent producers (with deduplication) and transactional consumers with commit logs."

## Additional Considerations

- How to handle message retention and cleanup?
- How to support priority queues?
- How to handle dead letter queues?
- How to implement message routing/filtering?

