# Challenge 03: Distributed Cache

## Problem Statement

Design a distributed caching system that can:
- Handle millions of requests per second
- Provide sub-millisecond latency
- Scale horizontally
- Handle node failures gracefully
- Support cache eviction policies
- Maintain consistency across nodes

## Requirements to Clarify

**Ask these questions before designing:**

### Functional Requirements
- What types of data? (key-value, objects, query results)
- What's the data size? (small values, large values)
- What operations? (get, set, delete, increment)
- Do we need TTL (time-to-live)?

### Non-Functional Requirements
- What's the expected QPS? (requests/sec)
- What's the latency requirement? (< 1ms, < 10ms?)
- What's the availability requirement? (99.9%, 99.99%?)
- What's the storage requirement? (current and growth)

### Constraints
- What's the data access pattern? (random, sequential)
- Are there consistency requirements? (strong, eventual)
- What's the budget?
- What's the network latency? (same datacenter, multi-region)

## Design Considerations

### Core Components

1. **Cache Nodes**
   - Store key-value pairs
   - Handle get/set/delete operations
   - Implement eviction policies

2. **Load Balancer**
   - Distribute requests across nodes
   - Health checks
   - Request routing

3. **Consistent Hashing**
   - Map keys to nodes
   - Handle node additions/removals
   - Minimize data movement

4. **Replication**
   - Replicate data across nodes
   - Handle node failures
   - Consistency management

5. **Eviction Policy**
   - LRU (Least Recently Used)
   - LFU (Least Frequently Used)
   - TTL-based expiration

### Key Design Decisions

#### 1. Architecture Pattern

**Option A: Client-Side Hashing**
- Client determines which node to contact
- No load balancer needed
- Client needs node list
- Good for: Simple, low latency

**Option B: Server-Side Routing**
- Load balancer routes requests
- Client doesn't need node list
- Single point of failure (load balancer)
- Good for: Simpler clients, better load balancing

**Recommendation**: Server-side routing for production (better load balancing)

#### 2. Consistent Hashing

**Purpose:**
- Map keys to nodes consistently
- Minimize data movement on node changes
- Distribute load evenly

**Algorithm:**
1. Create hash ring (0 to 2^64-1)
2. Hash nodes onto ring
3. Hash keys onto ring
4. Key belongs to first node clockwise

**Virtual Nodes:**
- Each physical node has multiple virtual nodes
- Better load distribution
- More even data distribution

**Recommendation**: Use consistent hashing with virtual nodes

#### 3. Replication Strategy

**Master-Slave:**
- One master, multiple replicas
- Reads from replicas, writes to master
- Simple, but master is bottleneck

**Multi-Master:**
- Multiple masters
- Writes to any master
- More complex, better availability

**Recommendation**: Master-slave for simplicity, multi-master for high availability

#### 4. Consistency Model

**Strong Consistency:**
- All nodes see same value
- Slower writes
- More complex

**Eventual Consistency:**
- Nodes eventually converge
- Faster writes
- Simpler

**Recommendation**: Eventual consistency for cache (acceptable for most use cases)

#### 5. Eviction Policy

**LRU (Least Recently Used):**
- Evict least recently used items
- Good for: Temporal locality

**LFU (Least Frequently Used):**
- Evict least frequently used items
- Good for: Frequency-based access

**TTL (Time-To-Live):**
- Evict after expiration
- Good for: Time-sensitive data

**Recommendation**: LRU for general purpose, TTL for time-sensitive data

## High-Level Architecture

```
[Clients]
    ↓
[Load Balancer] (Request Routing)
    ↓
[Consistent Hashing] (Key → Node Mapping)
    ↓
[Cache Nodes]
    ├── [Node 1] (Master) ← → [Replica 1]
    ├── [Node 2] (Master) ← → [Replica 2]
    └── [Node N] (Master) ← → [Replica N]
         ↓
    [Hash Ring] (Consistent Hashing)
```

## Detailed Design

### Cache Node Design

**Data Structure:**
- **In-Memory Hash Table**: Fast lookups (O(1))
- **LRU List**: Track access order (for LRU eviction)
- **TTL Index**: Track expiration times (for TTL eviction)

**Operations:**
- **GET**: O(1) lookup, update LRU
- **SET**: O(1) insert, update LRU, set TTL
- **DELETE**: O(1) delete, remove from LRU

**Memory Management:**
- Monitor memory usage
- Evict when memory limit reached
- Background eviction thread

### Consistent Hashing

**Hash Ring:**
```
0 ──────────────────────────────────────── 2^64-1
    │     │     │     │     │     │
   Node1 Node2 Node3 Node4 Node5 Node6
```

**Key Lookup:**
1. Hash key: `hash(key) % 2^64`
2. Find first node clockwise
3. Route request to that node

**Virtual Nodes:**
- Each physical node = 100-200 virtual nodes
- Better load distribution
- More even data distribution

**Node Addition:**
1. Add virtual nodes to ring
2. Migrate keys from adjacent nodes
3. Minimal data movement (~1/N of data)

**Node Removal:**
1. Remove virtual nodes from ring
2. Migrate keys to adjacent nodes
3. Replicate to maintain replication factor

### Replication

**Replication Factor: 3**
- Each key stored on 3 nodes
- 1 master, 2 replicas
- Reads from any replica
- Writes to master, then replicate

**Replication Process:**
1. Write to master
2. Master replicates to replicas
3. Wait for quorum (2/3 nodes)
4. Return success

**Failure Handling:**
- If master fails: Promote replica to master
- If replica fails: Replicate from master to new replica
- Maintain replication factor

### Eviction Policy

**LRU Implementation:**
- Doubly-linked list for access order
- Hash map for O(1) lookup
- Move to front on access
- Evict from tail

**TTL Implementation:**
- Store expiration time with each key
- Background thread checks expiration
- Evict expired keys

**Memory-Based Eviction:**
- Monitor memory usage
- Evict when threshold reached (e.g., 90%)
- Use eviction policy (LRU, LFU)

### API Design

**Operations:**
```
GET /cache/{key}
SET /cache/{key} (body: value, ttl)
DELETE /cache/{key}
INCREMENT /cache/{key} (body: delta)
```

**Response:**
```json
{
  "key": "user_123",
  "value": "{\"name\": \"John\"}",
  "ttl": 3600
}
```

## Scaling Strategy

### Start Small
- Single cache node
- No replication
- Simple eviction
- ~10K requests/sec

### Scale Gradually

**Phase 1: Add Replication**
- 3-node cluster
- Master-slave replication
- ~100K requests/sec

**Phase 2: Consistent Hashing**
- Multiple nodes
- Consistent hashing
- ~1M requests/sec

**Phase 3: Multi-Region**
- Replicate across regions
- Regional routing
- ~10M requests/sec

## Trade-offs

### Consistency vs Availability
- **Strong consistency**: Lower availability, slower writes
- **Eventual consistency**: Higher availability, faster writes
- **Recommendation**: Eventual consistency for cache (acceptable)

### Memory vs Disk
- **In-memory**: Very fast, limited capacity, expensive
- **Disk-backed**: Slower, larger capacity, cheaper
- **Recommendation**: In-memory for hot data, disk for cold data

### Latency vs Consistency
- **Lower latency**: Read from any replica, eventual consistency
- **Strong consistency**: Read from master, higher latency
- **Recommendation**: Read from replicas for lower latency

## Operational Concerns

### Monitoring
- Request rate (QPS)
- Latency (p50, p95, p99)
- Cache hit rate
- Memory usage
- Node health

### Failure Handling
- Node failure detection
- Automatic failover
- Data replication
- Rebalancing

### Performance
- Optimize hash function
- Minimize network hops
- Batch operations
- Connection pooling

## Follow-up Questions

**Be prepared to answer:**
1. "How do you handle a node failure?"
2. "How do you ensure data consistency?"
3. "How would you optimize cache hit rate?"
4. "How do you handle cache warming?"
5. "How would you implement cache invalidation?"
6. "How do you handle hot keys?"
7. "How would you optimize memory usage?"

## Example Solutions

### Simple Solution (Start Here)
- Single cache node
- Simple hash table
- LRU eviction
- ~10K requests/sec

### Optimized Solution
- Multi-node cluster
- Consistent hashing
- Replication
- Advanced eviction
- ~1M requests/sec

### Production Solution
- Multi-region deployment
- Consistent hashing with virtual nodes
- Multi-master replication
- Comprehensive monitoring
- Auto-scaling
- ~10M requests/sec

## Practice Exercise

**Design this system in 45-60 minutes:**
1. Clarify requirements (5-10 min)
2. High-level design (10-15 min)
3. Detailed design (15-20 min)
4. Scaling and optimization (10-15 min)
5. Trade-offs discussion (5 min)

**Focus on:**
- Consistent hashing
- Replication strategy
- Eviction policies
- Failure handling

