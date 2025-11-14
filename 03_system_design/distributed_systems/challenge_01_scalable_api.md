# Challenge 01: Scalable API

## Problem Statement

Design a REST API system that can handle:
- 10 million requests per second
- Sub-100ms latency (p95)
- 99.99% availability
- Support for both read and write operations
- Handle traffic spikes (10x normal load)

## Requirements to Clarify

**Ask these questions before designing:**

### Functional Requirements
- What does the API do? (CRUD operations, search, analytics)
- What's the read/write ratio? (90/10, 50/50?)
- What are the data access patterns? (random access, range queries)
- Do we need transactions? (ACID requirements)

### Non-Functional Requirements
- What's the expected QPS? (average and peak)
- What are latency requirements? (p50, p95, p99)
- What's the availability requirement? (99.9%, 99.99%?)
- What's the data size? (current and growth)

### Constraints
- What's the data model? (structured, unstructured)
- Are there any compliance requirements?
- What's the budget?
- What geographic distribution? (single region, multi-region)

## Design Considerations

### Core Components

1. **Load Balancer**
   - Distribute traffic across servers
   - Health checks
   - SSL termination

2. **API Gateway**
   - Request routing
   - Authentication/authorization
   - Rate limiting
   - Request/response transformation

3. **Application Servers**
   - Business logic
   - Stateless design
   - Horizontal scaling

4. **Database Layer**
   - Primary database (writes)
   - Read replicas (reads)
   - Caching layer
   - Sharding/partitioning

5. **Caching**
   - CDN (static content)
   - Application cache (Redis)
   - Database query cache

### Key Design Decisions

#### 1. Database Strategy

**SQL Database:**
- ACID transactions
- Complex queries
- Strong consistency
- Harder to scale horizontally

**NoSQL Database:**
- Flexible schema
- Easier horizontal scaling
- Eventual consistency
- Limited query capabilities

**Recommendation**: Use SQL for transactional data, NoSQL for scalable reads

#### 2. Caching Strategy

**Multi-Level Caching:**
1. **CDN**: Static content, edge caching
2. **Application Cache**: Frequently accessed data (Redis)
3. **Database Cache**: Query results

**Cache-Aside Pattern:**
- Check cache first
- If miss, query database
- Store in cache for future requests

**Recommendation**: Multi-level caching with cache-aside pattern

#### 3. Database Scaling

**Read Replicas:**
- Scale reads horizontally
- Eventual consistency
- Lower latency for reads

**Sharding:**
- Partition data across databases
- Scale writes horizontally
- Need sharding strategy (hash-based, range-based)

**Recommendation**: Start with read replicas, add sharding when needed

#### 4. Load Balancing

**Strategies:**
- Round-robin: Even distribution
- Least connections: Better for long-lived connections
- Consistent hashing: For sticky sessions

**Recommendation**: Use least connections for general traffic, consistent hashing for stateful services

## High-Level Architecture

```
[Users]
    ↓
[CDN] (Static Content)
    ↓
[Load Balancer] (Traffic Distribution)
    ↓
[API Gateway] (Auth, Rate Limiting, Routing)
    ↓
[Application Servers] (Stateless, Horizontally Scalable)
    ├── [App Server 1]
    ├── [App Server 2]
    └── [App Server N]
         ↓
    [Cache Layer] (Redis Cluster)
         ↓
    ├── [Read Replicas] (Read Operations)
    └── [Primary Database] (Write Operations)
         ↓
    [Object Storage] (Files, Blobs)
```

## Detailed Design

### API Gateway

**Functions:**
- Authentication (JWT, OAuth)
- Authorization (RBAC)
- Rate limiting (per user, per IP)
- Request validation
- Request/response transformation
- API versioning

**Rate Limiting:**
- Token bucket algorithm
- Per-user limits
- Per-endpoint limits
- Distributed rate limiting (Redis)

### Application Servers

**Design Principles:**
- **Stateless**: No session state on server
- **Horizontal Scaling**: Add more servers as needed
- **Health Checks**: Monitor server health
- **Graceful Shutdown**: Handle in-flight requests

**Scaling:**
- Auto-scaling based on CPU/memory
- Scale up during peak hours
- Scale down during off-peak

### Database Design

**Primary Database (Writes):**
- Handle all write operations
- Strong consistency
- Replication for durability
- Backup strategy

**Read Replicas:**
- Handle read operations
- Eventual consistency (acceptable)
- Multiple replicas for scale
- Geographic distribution for latency

**Sharding Strategy:**
- **Hash-based**: Distribute evenly
- **Range-based**: Natural partitioning
- **Directory-based**: Flexible, but single point of failure

**Example Sharding:**
```
Shard 1: user_id % 4 == 0
Shard 2: user_id % 4 == 1
Shard 3: user_id % 4 == 2
Shard 4: user_id % 4 == 3
```

### Caching Strategy

**Cache Layers:**

1. **CDN** (Edge Caching)
   - Static content (images, CSS, JS)
   - TTL: Hours to days
   - Geographic distribution

2. **Application Cache** (Redis)
   - Frequently accessed data
   - TTL: Minutes to hours
   - Cache-aside pattern
   - Cache invalidation strategy

3. **Database Query Cache**
   - Query results
   - TTL: Seconds to minutes
   - Automatic invalidation on writes

**Cache Invalidation:**
- **Write-through**: Update cache on write
- **Write-behind**: Update cache asynchronously
- **TTL-based**: Expire after time
- **Event-based**: Invalidate on data change

### API Design

**RESTful Design:**
```
GET    /api/v1/users/{id}        # Get user
POST   /api/v1/users             # Create user
PUT    /api/v1/users/{id}         # Update user
DELETE /api/v1/users/{id}        # Delete user
GET    /api/v1/users/{id}/posts  # Get user posts
```

**Response Format:**
```json
{
  "data": {...},
  "meta": {
    "timestamp": "2024-01-01T00:00:00Z",
    "version": "v1"
  }
}
```

## Scaling Strategy

### Start Small
- Single application server
- Single database
- No caching
- ~1K requests/sec

### Scale Gradually

**Phase 1: Add Caching**
- Implement Redis cache
- Add CDN for static content
- ~10K requests/sec

**Phase 2: Horizontal Scaling**
- Add load balancer
- Add more application servers
- Add read replicas
- ~100K requests/sec

**Phase 3: Database Optimization**
- Implement database sharding
- Optimize queries
- Add connection pooling
- ~1M requests/sec

**Phase 4: Advanced Optimization**
- Multi-region deployment
- Advanced caching strategies
- Database optimization
- ~10M requests/sec

## Trade-offs

### Consistency vs Availability
- **Strong consistency**: Lower availability, simpler logic
- **Eventual consistency**: Higher availability, more complex
- **Recommendation**: Strong consistency for writes, eventual for reads

### Latency vs Cost
- **Lower latency**: More servers, caching, higher cost
- **Lower cost**: Fewer servers, less caching, higher latency
- **Recommendation**: Balance based on requirements

### SQL vs NoSQL
- **SQL**: ACID, complex queries, harder to scale
- **NoSQL**: Flexible, easier to scale, limited queries
- **Recommendation**: Use both (SQL for transactional, NoSQL for scalable reads)

## Operational Concerns

### Monitoring
- Request rate (QPS)
- Latency (p50, p95, p99)
- Error rate
- Server health (CPU, memory)
- Database performance
- Cache hit rate

### Debugging
- Distributed tracing (OpenTelemetry)
- Request logs
- Error tracking
- Performance profiling

### Disaster Recovery
- Multi-region deployment
- Automatic failover
- Database backups
- Disaster recovery plan

### Security
- Authentication and authorization
- Rate limiting
- Input validation
- SQL injection prevention
- DDoS protection

## Follow-up Questions

**Be prepared to answer:**
1. "How would you handle a database failure?"
2. "How do you ensure data consistency across shards?"
3. "How would you handle a traffic spike?"
4. "How do you implement rate limiting?"
5. "How would you optimize for cost?"
6. "How do you handle cache invalidation?"
7. "How would you implement multi-region deployment?"

## Example Solutions

### Simple Solution (Start Here)
- Single application server
- Single database
- Basic caching
- ~1K requests/sec

### Optimized Solution
- Multiple application servers
- Read replicas
- Multi-level caching
- Database optimization
- ~1M requests/sec

### Production Solution
- Multi-region deployment
- Database sharding
- Advanced caching
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
- Scalability patterns
- Database design
- Caching strategy
- Load balancing
- Monitoring and operations

