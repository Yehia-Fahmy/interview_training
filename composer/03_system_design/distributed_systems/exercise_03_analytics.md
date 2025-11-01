# Exercise 3: Design a Real-time Analytics System

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Stream processing, real-time systems, data pipelines

## Problem

Design a system to process and analyze billions of events per day in real-time:
- User click events
- Metrics aggregation (counts, averages, percentiles)
- Real-time dashboards
- Historical data retention
- Support for ad-hoc queries

## Requirements to Discuss

1. **Data Ingestion**
   - How to handle high-throughput ingestion?
   - Message queue (Kafka)?
   - How to handle backpressure?

2. **Stream Processing**
   - Apache Flink, Spark Streaming, or custom?
   - Windowing (tumbling, sliding)?
   - State management?

3. **Storage Architecture**
   - Hot storage for real-time queries?
   - Cold storage for historical data?
   - Time-series database?

4. **Query Layer**
   - How to serve real-time dashboards?
   - How to handle ad-hoc queries?
   - Caching strategy?

5. **Scalability**
   - How to partition streams?
   - How to handle skew?
   - Auto-scaling?

6. **Reliability**
   - Exactly-once processing?
   - Handling late events?
   - Failure recovery?

## Key Topics to Cover

- **Lambda Architecture**: Batch + stream processing
- **Kappa Architecture**: Stream-only processing
- **Time Windows**: Tumbling, sliding, session windows
- **State Stores**: RocksDB, embedded databases
- **Query Optimization**: Pre-aggregation, materialized views

## Sample Discussion Points

1. "I'd use Kafka for ingestion. Events would be partitioned by user_id for ordering guarantees. Multiple consumers would process different partitions."

2. "For processing, I'd use a stream processing framework like Flink. It provides windowing, state management, and exactly-once guarantees."

3. "I'd implement a lambda architecture: stream processing for real-time metrics (stored in Redis or time-series DB), batch processing for historical accuracy (stored in data warehouse)."

4. "Real-time dashboards would query pre-aggregated metrics from a time-series database like InfluxDB or TimescaleDB."

5. "For late events, I'd use watermarks in the stream processor. Events within watermark tolerance are processed normally, beyond that goes to a separate late stream."

## Additional Considerations

- How to handle event ordering across partitions?
- How to compute complex aggregations (e.g., distinct count)?
- How to handle schema evolution?
- How to debug and monitor stream processing?

