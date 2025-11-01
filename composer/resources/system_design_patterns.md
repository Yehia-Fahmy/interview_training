# System Design Patterns

Common patterns for the System Design interview.

## Scalability Patterns

- **Horizontal Scaling**: Add more servers
- **Vertical Scaling**: Increase server capacity
- **Load Balancing**: Distribute traffic
- **Caching**: Reduce database load
- **CDN**: Reduce latency

## Data Storage

- **SQL Databases**: Relational data, ACID transactions
- **NoSQL Databases**: 
  - Document (MongoDB): Flexible schemas
  - Key-Value (Redis): Caching, sessions
  - Column-family (Cassandra): Wide tables
  - Graph (Neo4j): Relationships
- **Time-Series**: InfluxDB, TimescaleDB
- **Object Stores**: S3, Azure Blob

## Distributed Systems

- **Consistent Hashing**: Distributed caching
- **Leader Election**: Raft, Paxos
- **Replication**: Master-slave, master-master
- **Partitioning**: Hash-based, range-based
- **Quorum**: Read/write consistency

## ML System Patterns

- **Lambda Architecture**: Batch + stream
- **Kappa Architecture**: Stream-only
- **Two-Stage Retrieval**: Candidate generation + ranking
- **Feature Stores**: Online/offline features
- **Model Serving**: APIs, batching, optimization

## Reliability Patterns

- **Circuit Breakers**: Prevent cascade failures
- **Retries**: Exponential backoff
- **Timeouts**: Prevent hanging requests
- **Graceful Degradation**: Fallback mechanisms
- **Health Checks**: Monitor system status

