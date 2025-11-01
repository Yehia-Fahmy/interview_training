# Exercise 4: Design a Feature Store System

**Difficulty:** Medium-Hard  
**Time Limit:** 60 minutes  
**Focus:** Feature engineering, data infrastructure, serving features

## Problem

Design a feature store that:
- Manages features for training and serving
- Supports both batch and real-time features
- Handles feature versioning
- Serves features with low latency (< 10ms p99)
- Supports feature discovery and governance
- Handles billions of feature values

## Requirements to Discuss

1. **Architecture**
   - Online vs offline feature stores?
   - How to sync between them?
   - Storage backend (Redis, Cassandra, etc.)?

2. **Feature Definition**
   - How to define features (DSL, code)?
   - Feature transformations?
   - Feature lineage?

3. **Batch Features**
   - How to compute batch features?
   - When to recompute?
   - Storage format?

4. **Real-time Features**
   - How to compute on-the-fly?
   - Windowed aggregations?
   - State management?

5. **Serving Layer**
   - How to serve features quickly?
   - Caching strategy?
   - Batch feature retrieval?

6. **Governance**
   - Feature discovery?
   - Access control?
   - Feature quality monitoring?
   - Schema validation?

## Key Topics to Cover

- **Feature Stores**: Tecton, Feast, custom implementations
- **Storage**: Online (Redis, DynamoDB) vs Offline (S3, Data Warehouse)
- **Feature Engineering**: Transformations, aggregations
- **Serving**: Low-latency APIs, caching
- **Data Quality**: Validation, monitoring, drift detection

## Sample Discussion Points

1. "I'd implement a dual-store architecture: offline store (data warehouse) for training, online store (Redis/Cassandra) for serving. Batch pipeline syncs features from offline to online."

2. "Features are defined as code with transformations. Feature definitions are versioned. Supports both point-in-time features (snapshot) and time-series features."

3. "Batch features computed via Spark/Flink jobs. Features stored in parquet format in data warehouse. Computed daily or on-demand."

4. "Real-time features computed using stream processing. State maintained in Redis or embedded state store. Supports windowed aggregations (last 1 hour, last 24 hours)."

5. "Serving layer provides REST API. Features cached in Redis for hot paths. Supports batch retrieval (all features for an entity) and individual lookups."

6. "Governance includes feature registry with metadata, access control, data quality checks (null rates, value ranges), and monitoring for feature drift."

## Additional Considerations

- How to handle feature backfilling (computing historical features)?
- How to support feature joins across entities?
- How to handle feature dependencies?
- How to ensure consistency between training and serving features?

