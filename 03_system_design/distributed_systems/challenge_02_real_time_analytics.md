# Challenge 02: Real-time Analytics

## Problem Statement

Design a real-time analytics system that can:
- Process millions of events per second
- Provide real-time dashboards (< 1 second latency)
- Support ad-hoc queries
- Handle historical data (years of data)
- Scale to handle traffic spikes

## Requirements to Clarify

**Ask these questions before designing:**

### Functional Requirements
- What types of events? (user actions, system metrics, business events)
- What analytics are needed? (aggregations, time-series, ad-hoc queries)
- What's the query pattern? (pre-defined dashboards, ad-hoc queries)
- Do we need real-time alerts?

### Non-Functional Requirements
- What's the event volume? (events/sec)
- What's the data retention? (days, months, years)
- What's the query latency requirement? (< 1s, < 10s?)
- What's the data accuracy requirement? (exact, approximate)

### Constraints
- What's the data format? (structured, semi-structured)
- Are there any compliance requirements?
- What's the budget?
- What's the query complexity? (simple aggregations, complex joins)

## Design Considerations

### Core Components

1. **Data Ingestion**
   - Event collection
   - Data validation
   - Schema management

2. **Stream Processing**
   - Real-time processing
   - Windowing and aggregations
   - State management

3. **Storage**
   - Real-time storage (for dashboards)
   - Historical storage (for queries)
   - Time-series database

4. **Query Layer**
   - Real-time query API
   - Ad-hoc query engine
   - Caching layer

5. **Visualization**
   - Dashboard service
   - Real-time updates
   - Alerting

### Key Design Decisions

#### 1. Architecture Pattern

**Lambda Architecture:**
- Batch layer: Historical data processing
- Speed layer: Real-time processing
- Serving layer: Merge batch and real-time results
- Good for: Exact results, complex processing

**Kappa Architecture:**
- Single stream processing layer
- Reprocess data for historical queries
- Good for: Simpler architecture, eventual consistency

**Recommendation**: Start with Lambda, consider Kappa for simpler use cases

#### 2. Stream Processing

**Technologies:**
- **Apache Flink**: Low latency, stateful processing
- **Apache Kafka Streams**: Simple, Kafka-native
- **Apache Storm**: Mature, but lower-level
- **AWS Kinesis**: Managed, AWS ecosystem

**Recommendation**: Flink for complex processing, Kafka Streams for simplicity

#### 3. Storage Strategy

**Real-time Storage:**
- **Redis**: In-memory, very fast
- **Apache Druid**: Time-series, optimized for analytics
- **ClickHouse**: Columnar, fast aggregations

**Historical Storage:**
- **Data Warehouse**: BigQuery, Snowflake, Redshift
- **Object Storage**: S3, GCS (with query engines)
- **Time-series DB**: InfluxDB, TimescaleDB

**Recommendation**: Druid/ClickHouse for real-time, data warehouse for historical

#### 4. Query Pattern

**Pre-computed Aggregations:**
- Compute aggregations in stream processing
- Store in fast storage (Redis, Druid)
- Very fast queries (< 100ms)
- Limited flexibility

**On-demand Queries:**
- Query raw data on-demand
- More flexible
- Slower queries (seconds)
- Higher cost

**Recommendation**: Hybrid (pre-compute common queries, on-demand for ad-hoc)

## High-Level Architecture

```
[Event Sources]
    ├── [User Actions]
    ├── [System Metrics]
    └── [Business Events]
         ↓
[Event Ingestion] (Kafka, Kinesis)
    ↓
[Stream Processing] (Flink, Kafka Streams)
    ├── [Real-time Aggregations]
    └── [Windowed Processing]
         ↓
    ├── [Real-time Storage] (Druid, ClickHouse, Redis)
    └── [Historical Storage] (Data Warehouse, Object Storage)
         ↓
[Query Layer] (API, Query Engine)
    ↓
[Dashboard Service] (Visualization, Alerts)
    ↓
[Users]
```

## Detailed Design

### Data Ingestion

**Event Format:**
```json
{
  "event_id": "evt_123",
  "event_type": "page_view",
  "user_id": "user_456",
  "timestamp": "2024-01-01T00:00:00Z",
  "properties": {
    "page": "/home",
    "duration": 30
  }
}
```

**Ingestion Pipeline:**
1. Collect events from sources
2. Validate schema
3. Enrich with metadata (user info, context)
4. Publish to message queue (Kafka)

**Throughput**: Millions of events per second

### Stream Processing

**Processing Tasks:**
1. **Filtering**: Filter relevant events
2. **Enrichment**: Add context data
3. **Aggregation**: Compute metrics (counts, sums, averages)
4. **Windowing**: Time-based windows (1min, 5min, 1hour)
5. **State Management**: Maintain state for aggregations

**Example Aggregations:**
- Page views per minute
- Unique users per hour
- Revenue per day
- Error rate per 5 minutes

**Technologies:**
- Apache Flink (low latency, stateful)
- Apache Kafka Streams (simple, Kafka-native)

### Storage Design

**Real-time Storage (Druid/ClickHouse):**

**Data Model:**
```
Timestamp | Metric | Value | Dimensions
----------|--------|-------|------------
2024-01-01 00:00 | page_views | 1000 | page=/home
2024-01-01 00:01 | page_views | 1200 | page=/home
```

**Features:**
- Columnar storage (fast aggregations)
- Time-based partitioning
- Pre-aggregated metrics
- Fast queries (< 100ms)

**Historical Storage (Data Warehouse):**

**Data Model:**
- Raw events (for ad-hoc queries)
- Aggregated data (for faster queries)
- Partitioned by time

**Features:**
- SQL queries
- Complex joins
- Ad-hoc analysis

### Query Layer

**API Endpoints:**
```
GET /api/v1/metrics?metric=page_views&start=2024-01-01&end=2024-01-02&granularity=1h
GET /api/v1/dashboard?dashboard_id=homepage
GET /api/v1/query (ad-hoc SQL queries)
```

**Query Types:**

1. **Pre-computed Metrics** (Fast)
   - Query from real-time storage
   - < 100ms latency
   - Limited to pre-computed metrics

2. **Ad-hoc Queries** (Slower)
   - Query from historical storage
   - Seconds to minutes
   - Full flexibility

**Caching:**
- Cache common queries
- TTL based on data freshness
- Invalidate on data updates

### Dashboard Service

**Features:**
- Real-time dashboards
- Auto-refresh (every few seconds)
- Historical views
- Alerting (threshold-based)

**Real-time Updates:**
- WebSocket connections
- Push updates to clients
- Efficient data transfer (deltas only)

## Scaling Strategy

### Start Small
- Single stream processor
- Single storage system
- Basic dashboards
- ~10K events/sec

### Scale Gradually

**Phase 1: Separate Real-time/Historical**
- Stream processing for real-time
- Data warehouse for historical
- ~100K events/sec

**Phase 2: Optimize Storage**
- Time-series database for real-time
- Optimized data warehouse
- ~1M events/sec

**Phase 3: Distributed Processing**
- Distributed stream processing
- Multi-region storage
- ~10M events/sec

## Trade-offs

### Latency vs Accuracy
- **Lower latency**: Approximate results, pre-aggregation
- **Higher accuracy**: Exact results, slower queries
- **Recommendation**: Approximate for dashboards, exact for reports

### Storage Cost vs Query Speed
- **More storage**: Pre-aggregate more, faster queries
- **Less storage**: Store raw data, slower queries
- **Recommendation**: Pre-aggregate common queries, store raw for ad-hoc

### Real-time vs Batch
- **Real-time**: Lower latency, higher cost
- **Batch**: Higher latency, lower cost
- **Recommendation**: Real-time for dashboards, batch for reports

## Operational Concerns

### Monitoring
- Event ingestion rate
- Processing lag
- Query latency
- Storage usage
- Data freshness

### Data Quality
- Schema validation
- Data completeness
- Duplicate detection
- Data accuracy checks

### Cost Optimization
- Compress historical data
- Use cost-effective storage (object storage)
- Optimize query patterns
- Auto-scale processing

## Follow-up Questions

**Be prepared to answer:**
1. "How do you handle late-arriving events?"
2. "How would you ensure data accuracy?"
3. "How do you handle schema evolution?"
4. "How would you optimize query performance?"
5. "How do you handle data retention?"
6. "How would you implement real-time alerts?"
7. "How do you handle backpressure?"

## Example Solutions

### Simple Solution (Start Here)
- Single stream processor
- Single storage system
- Basic dashboards
- ~10K events/sec

### Optimized Solution
- Lambda architecture
- Time-series database for real-time
- Data warehouse for historical
- Optimized queries
- ~1M events/sec

### Production Solution
- Kappa or Lambda architecture
- Distributed stream processing
- Multi-level storage
- Advanced query optimization
- Real-time dashboards
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
- Storage strategy
- Query layer design
- Real-time vs batch trade-offs

