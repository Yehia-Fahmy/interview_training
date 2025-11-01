# Exercise 2: Design a Distributed Cache System

**Difficulty:** Medium-Hard  
**Time Limit:** 45-60 minutes  
**Focus:** Caching strategies, consistency, scalability

## Problem

Design a distributed cache system (similar to Redis or Memcached) that:
- Handles millions of requests per second
- Scales horizontally
- Maintains high availability
- Handles cache invalidation
- Supports different eviction policies

## Requirements to Discuss

1. **Architecture**
   - Client-server model?
   - Peer-to-peer?
   - Proxy layer?

2. **Data Partitioning**
   - How to shard data across nodes?
   - Consistent hashing?
   - How to handle node additions/removals?

3. **Replication**
   - Master-slave or master-master?
   - How many replicas?
   - Read/write consistency?

4. **Eviction Policies**
   - LRU, LFU, FIFO?
   - How to implement efficiently?
   - Distributed eviction?

5. **High Availability**
   - What if a node fails?
   - How to handle network partitions?
   - Split-brain scenarios?

6. **Performance**
   - How to minimize latency?
   - Memory vs network trade-offs?
   - Serialization format?

## Key Topics to Cover

- **Consistent Hashing**: Virtual nodes, ring structure
- **Replication**: Primary-secondary, quorum reads
- **Consistency Models**: Eventual, strong, causal
- **Partition Tolerance**: CAP theorem trade-offs
- **Failure Handling**: Failover, recovery

## Sample Discussion Points

1. "I'd use consistent hashing to distribute keys across nodes. Each node would be responsible for a range of hash space."

2. "For replication, I'd use a primary-secondary model with 3 replicas. Writes go to primary, reads can go to any replica for read scaling."

3. "For eviction, I'd implement LRU with approximate algorithms to handle distributed nature. Each node maintains local LRU."

4. "Node failures would trigger rehashing. Data from failed node's replicas would be promoted. New nodes can join and take over part of the hash space."

5. "I'd use async replication to avoid blocking writes. This provides eventual consistency but better write performance."

## Additional Considerations

- How to handle cache warming after failures?
- How to implement distributed locking?
- How to handle cache stampede (thundering herd)?
- Hot partition problem and solutions?

