# Challenge 03: Feature Store

## Problem Statement

Design a feature store that serves features for both:
- **Training**: Batch features for model training
- **Serving**: Real-time features for model inference

The system should:
- Support online and offline feature computation
- Ensure feature consistency between training and serving
- Handle feature versioning and lineage
- Scale to millions of features and billions of feature values
- Support low-latency feature serving (< 10ms p95)

## Requirements to Clarify

**Ask these questions before designing:**

### Functional Requirements
- What types of features? (numerical, categorical, embeddings, time-series)
- What's the feature update frequency? (real-time, hourly, daily)
- Do we need point-in-time correctness? (time travel queries)
- What's the feature serving pattern? (single features, feature sets)

### Non-Functional Requirements
- What's the feature serving latency requirement? (< 10ms, < 100ms?)
- What's the throughput requirement? (requests/sec)
- What's the storage requirement? (current and growth)
- What's the availability requirement? (99.9%, 99.99%?)

### Constraints
- What data sources? (databases, streams, APIs)
- What's the feature computation complexity? (simple aggregations, ML models)
- Are there any compliance requirements?
- What's the budget?

## Design Considerations

### Core Components

1. **Feature Registry**
   - Feature metadata
   - Feature schemas
   - Feature versioning
   - Feature lineage

2. **Offline Feature Store**
   - Batch feature computation
   - Historical features
   - Training dataset generation
   - Time-travel queries

3. **Online Feature Store**
   - Real-time feature computation
   - Low-latency feature serving
   - Feature caching
   - Point-in-time feature values

4. **Feature Computation Engine**
   - Batch computation (Spark, Flink)
   - Real-time computation (Flink, Kafka Streams)
   - Feature transformations
   - Feature validation

5. **Feature Serving API**
   - REST/GraphQL API
   - Batch and single feature requests
   - Feature versioning in requests
   - Low-latency responses

### Key Design Decisions

#### 1. Architecture Pattern

**Option A: Dual Store (Offline + Online)**
- Separate stores for batch and real-time
- Offline: Data warehouse (BigQuery, Snowflake)
- Online: Key-value store (Redis, DynamoDB)
- Good for: Clear separation, different SLAs

**Option B: Unified Store**
- Single store for both
- Use different access patterns
- Good for: Simpler architecture, consistency

**Recommendation**: Dual store for production systems (most common pattern)

#### 2. Offline Feature Store

**Storage Options:**
- **Data Warehouse**: BigQuery, Snowflake, Redshift
  - Good for: Large-scale analytics, SQL queries
- **Object Storage**: S3, GCS with Parquet
  - Good for: Cost-effective, flexible
- **HDFS**: For on-premise
  - Good for: Large-scale, existing infrastructure

**Recommendation**: Data warehouse for ease of use, object storage for cost

#### 3. Online Feature Store

**Storage Options:**
- **Redis**: In-memory, very fast
  - Good for: High throughput, low latency
- **DynamoDB**: Managed, scalable
  - Good for: AWS ecosystem, auto-scaling
- **Cassandra**: Distributed, scalable
  - Good for: Very large scale, multi-region

**Recommendation**: Redis for low latency, DynamoDB/Cassandra for scale

#### 4. Feature Computation

**Batch Computation:**
- Spark, Flink Batch
- Scheduled jobs (hourly, daily)
- Compute features for all entities
- Store in offline store

**Real-time Computation:**
- Flink, Kafka Streams
- Compute on-demand or pre-compute
- Store in online store
- Low-latency requirements

**Recommendation**: Batch for historical, real-time for current values

#### 5. Feature Versioning

**Approaches:**
- **Timestamp-based**: Features tagged with timestamp
- **Version numbers**: Features have version numbers
- **Schema evolution**: Features evolve over time

**Recommendation**: Timestamp-based for time-travel, version numbers for schema changes

## High-Level Architecture

```
[Data Sources]
    ├── [Databases]
    ├── [Streams]
    └── [APIs]
         ↓
[Feature Computation Engine]
    ├── [Batch Computation] (Spark/Flink)
    └── [Real-time Computation] (Flink/Kafka Streams)
         ↓
    ├── [Offline Feature Store] (Data Warehouse/Object Storage)
    └── [Online Feature Store] (Redis/DynamoDB)
         ↓
[Feature Registry] (Metadata, Schema, Lineage)
    ↓
[Feature Serving API] (REST/GraphQL)
    ↓
[Consumers]
    ├── [Training Pipeline]
    └── [Model Serving]
```

## Detailed Design

### Feature Registry

**Stored Information:**
- Feature definitions (name, type, description)
- Feature schemas (data types, constraints)
- Feature versions
- Feature lineage (data sources, transformations)
- Feature owners and documentation

**Storage:**
- SQL database (PostgreSQL) or feature store metadata service

### Offline Feature Store

**Data Model:**
```
Entity ID | Feature 1 | Feature 2 | ... | Timestamp
----------|-----------|-----------|-----|------------
user_123  | 100       | "active"  | ... | 2024-01-01
user_123  | 150       | "active"  | ... | 2024-01-02
```

**Key Features:**
- Time-series data (point-in-time correctness)
- Partitioned by time (daily/hourly partitions)
- Supports time-travel queries
- Used for training dataset generation

**Query Pattern:**
```sql
SELECT features
FROM feature_store
WHERE entity_id = 'user_123'
  AND timestamp <= '2024-01-01'
```

### Online Feature Store

**Data Model:**
```
Key: entity_id:feature_name
Value: feature_value
TTL: based on feature freshness
```

**Key Features:**
- Key-value storage
- Low-latency access (< 10ms)
- TTL-based expiration
- Supports batch gets

**Access Pattern:**
```
GET /features?entity_id=user_123&features=feature1,feature2
```

### Feature Computation

**Batch Computation Pipeline:**
1. **Ingest**: Pull data from sources
2. **Transform**: Apply feature transformations
3. **Validate**: Check feature quality
4. **Store**: Write to offline store
5. **Backfill**: Update online store (if needed)

**Real-time Computation:**
1. **Stream Processing**: Process events in real-time
2. **Compute**: Apply transformations
3. **Store**: Update online store
4. **Invalidate**: Remove stale features

### Feature Serving API

**Endpoints:**
- `GET /features/{entity_id}` - Get all features for entity
- `GET /features/{entity_id}?features=f1,f2` - Get specific features
- `POST /features/batch` - Batch get features
- `GET /features/{entity_id}?timestamp=2024-01-01` - Time-travel query

**Response Format:**
```json
{
  "entity_id": "user_123",
  "features": {
    "feature1": 100,
    "feature2": "active"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Feature Consistency

**Challenge**: Ensuring training and serving use same features

**Solutions:**
1. **Point-in-time correctness**: Use same timestamp for training and serving
2. **Feature versioning**: Explicitly version features
3. **Feature validation**: Validate features match schema
4. **Monitoring**: Track feature drift between training and serving

## Scaling Strategy

### Start Small
- Single database for offline
- Redis for online
- Simple batch computation
- ~1M features, ~100K requests/sec

### Scale Gradually

**Phase 1: Separate Offline/Online**
- Data warehouse for offline
- Redis cluster for online
- ~10M features, ~500K requests/sec

**Phase 2: Distributed Computation**
- Spark cluster for batch
- Flink for real-time
- ~100M features, ~1M requests/sec

**Phase 3: Multi-Region**
- Replicate online store
- Regional offline stores
- ~1B features, ~10M requests/sec

## Trade-offs

### Consistency vs Latency
- **Strong consistency**: Slower, simpler
- **Eventual consistency**: Faster, more complex
- **Recommendation**: Eventual consistency for online store, strong for offline

### Storage Cost vs Compute Cost
- **Pre-compute**: Higher storage, lower compute
- **On-demand**: Lower storage, higher compute
- **Recommendation**: Pre-compute common features, on-demand for rare ones

### Feature Freshness vs Latency
- **Fresher features**: More computation, higher latency
- **Staler features**: Less computation, lower latency
- **Recommendation**: Balance based on use case (real-time vs batch)

### Offline vs Online Separation
- **Separate stores**: More complex, better optimization
- **Unified store**: Simpler, less optimized
- **Recommendation**: Separate stores for production scale

## Operational Concerns

### Monitoring
- Feature serving latency
- Feature freshness
- Feature computation lag
- Storage usage
- Cache hit rates

### Debugging
- Feature lineage tracking
- Feature value inspection
- Computation logs
- Serving request logs

### Data Quality
- Feature validation
- Schema enforcement
- Missing value handling
- Outlier detection

### Cost Optimization
- Cache frequently accessed features
- Compress historical features
- Use cost-effective storage (object storage for offline)
- Optimize computation (incremental updates)

## Follow-up Questions

**Be prepared to answer:**
1. "How do you ensure feature consistency between training and serving?"
2. "How would you handle feature schema changes?"
3. "How do you compute features in real-time?"
4. "How would you optimize feature serving latency?"
5. "How do you handle feature backfills?"
6. "How would you implement time-travel queries?"
7. "How do you monitor feature quality?"

## Example Solutions

### Simple Solution (Start Here)
- Single database for offline/online
- Simple batch computation
- Basic feature serving API
- ~1M features, ~10K requests/sec

### Optimized Solution
- Separate offline/online stores
- Distributed batch computation
- Redis cluster for online
- Feature caching
- ~100M features, ~1M requests/sec

### Production Solution
- Dual-store architecture
- Distributed batch and real-time computation
- Multi-region deployment
- Advanced feature versioning and lineage
- Comprehensive monitoring
- ~1B features, ~10M requests/sec

## Practice Exercise

**Design this system in 45-60 minutes:**
1. Clarify requirements (5-10 min)
2. High-level design (10-15 min)
3. Detailed design (15-20 min)
4. Scaling and optimization (10-15 min)
5. Trade-offs discussion (5 min)

**Focus on:**
- Offline vs online architecture
- Feature consistency
- Feature computation
- Feature serving
- Feature versioning

