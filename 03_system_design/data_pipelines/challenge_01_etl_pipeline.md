# Challenge 01: ETL Pipeline

## Problem Statement

Design a scalable ETL (Extract, Transform, Load) pipeline that can:
- Process terabytes of data daily
- Handle multiple data sources (databases, APIs, files)
- Transform data with complex business logic
- Load data into data warehouse
- Handle failures gracefully
- Support data quality checks
- Enable data lineage tracking

## Requirements to Clarify

**Ask these questions before designing:**

### Functional Requirements
- What are the data sources? (databases, APIs, files, streams)
- What transformations are needed? (cleaning, aggregation, enrichment)
- What's the destination? (data warehouse, data lake, databases)
- What's the data format? (structured, semi-structured, unstructured)

### Non-Functional Requirements
- What's the data volume? (GB/day, TB/day)
- What's the processing frequency? (hourly, daily, real-time)
- What's the acceptable latency? (hours, minutes?)
- What's the data retention? (days, months, years)

### Constraints
- What's the data schema? (fixed, evolving)
- Are there any compliance requirements?
- What's the budget?
- What's the acceptable data loss? (zero tolerance, some tolerance)

## Design Considerations

### Core Components

1. **Extraction**
   - Pull data from sources
   - Handle different source types
   - Incremental vs full extraction

2. **Transformation**
   - Data cleaning
   - Data validation
   - Business logic
   - Data enrichment

3. **Loading**
   - Load into destination
   - Handle schema changes
   - Optimize loading performance

4. **Orchestration**
   - Schedule jobs
   - Handle dependencies
   - Retry on failures
   - Monitor execution

5. **Data Quality**
   - Validation rules
   - Data profiling
   - Anomaly detection
   - Quality metrics

### Key Design Decisions

#### 1. ETL vs ELT Pattern

**ETL (Extract-Transform-Load):**
- Transform before loading
- Good for: Complex transformations, smaller datasets
- Tools: Apache Airflow, Luigi, custom scripts

**ELT (Extract-Load-Transform):**
- Load first, transform in destination
- Good for: Large datasets, simple transformations
- Tools: dbt, SQL-based transformations

**Recommendation**: ETL for complex transformations, ELT for large datasets

#### 2. Batch vs Streaming

**Batch Processing:**
- Process data in batches (hourly, daily)
- Simpler, lower cost
- Higher latency
- Good for: Historical analysis, reporting

**Streaming Processing:**
- Process data in real-time
- More complex, higher cost
- Lower latency
- Good for: Real-time analytics, alerts

**Recommendation**: Batch for most use cases, streaming for real-time needs

#### 3. Incremental vs Full Load

**Full Load:**
- Extract all data every time
- Simple, but expensive
- Good for: Small datasets, initial load

**Incremental Load:**
- Extract only changed data
- More complex, but efficient
- Good for: Large datasets, frequent updates

**Recommendation**: Incremental for large datasets, full for small or when needed

#### 4. Orchestration

**Scheduling:**
- Cron-based: Simple, but limited
- Workflow engines: Airflow, Prefect, Dagster
- Managed services: AWS Glue, Azure Data Factory

**Dependencies:**
- Define job dependencies
- Handle failures
- Retry logic

**Recommendation**: Use workflow engine (Airflow) for complex pipelines

#### 5. Data Quality

**Validation:**
- Schema validation
- Data type checks
- Range checks
- Uniqueness checks
- Referential integrity

**Monitoring:**
- Track data quality metrics
- Alert on quality issues
- Data profiling

**Recommendation**: Implement comprehensive data quality checks

## High-Level Architecture

```
[Data Sources]
    ├── [Databases]
    ├── [APIs]
    ├── [Files] (S3, GCS)
    └── [Streams]
         ↓
[Extraction Layer]
    ├── [Extractors] (Source-specific)
    └── [Staging Area] (Raw Data Storage)
         ↓
[Transformation Layer]
    ├── [Transformers] (Business Logic)
    ├── [Data Quality] (Validation)
    └── [Enrichment] (Join, Lookup)
         ↓
[Loading Layer]
    ├── [Loaders] (Destination-specific)
    └── [Data Warehouse] (Final Destination)
         ↓
[Orchestration] (Scheduling, Monitoring)
```

## Detailed Design

### Extraction Layer

**Extractors:**
- **Database Extractors**: JDBC/ODBC connectors
- **API Extractors**: REST API clients
- **File Extractors**: S3/GCS readers
- **Stream Extractors**: Kafka consumers

**Extraction Strategies:**

1. **Full Extraction:**
   - Extract all data
   - Simple, but expensive
   - Use for: Initial load, small datasets

2. **Incremental Extraction:**
   - Extract only changed data
   - Use timestamps or change data capture (CDC)
   - More efficient for large datasets

3. **Change Data Capture (CDC):**
   - Capture database changes in real-time
   - Use database logs (binlog, WAL)
   - Most efficient for frequent updates

**Staging Area:**
- Store raw extracted data
- Preserve original format
- Enable reprocessing
- Use object storage (S3, GCS)

### Transformation Layer

**Transformation Types:**

1. **Data Cleaning:**
   - Remove duplicates
   - Handle missing values
   - Standardize formats
   - Normalize data

2. **Data Validation:**
   - Schema validation
   - Data type checks
   - Business rule validation
   - Data quality checks

3. **Data Enrichment:**
   - Join with reference data
   - Lookup external data
   - Calculate derived fields
   - Add metadata

4. **Data Aggregation:**
   - Group by dimensions
   - Calculate metrics
   - Summarize data

**Transformation Engine:**
- **Spark**: Distributed processing, large datasets
- **Pandas**: Simple transformations, smaller datasets
- **SQL**: For ELT pattern, data warehouse

**Data Quality:**
- Validate schema
- Check data types
- Validate business rules
- Detect anomalies
- Track quality metrics

### Loading Layer

**Loading Strategies:**

1. **Full Load:**
   - Replace entire table
   - Simple, but expensive
   - Use for: Small tables, initial load

2. **Incremental Load:**
   - Append new data
   - Update changed data
   - More efficient

3. **Upsert (Merge):**
   - Insert new, update existing
   - Handle duplicates
   - Most flexible

**Destination:**
- **Data Warehouse**: BigQuery, Snowflake, Redshift
- **Data Lake**: S3, GCS (with query engines)
- **Databases**: PostgreSQL, MySQL

**Optimization:**
- Partition by date
- Use columnar format (Parquet)
- Compress data
- Optimize loading (bulk insert)

### Orchestration

**Workflow Engine (Apache Airflow):**

**DAG (Directed Acyclic Graph):**
```python
extract_task >> transform_task >> load_task
extract_task >> quality_check_task >> load_task
```

**Features:**
- Schedule jobs (cron, interval)
- Handle dependencies
- Retry on failures
- Monitor execution
- Alert on failures

**Job Management:**
- Queue jobs
- Allocate resources
- Handle concurrency
- Monitor progress

### Data Lineage

**Track:**
- Data sources
- Transformations applied
- Destination
- Data flow
- Dependencies

**Storage:**
- Metadata store
- Graph database (for lineage)
- Documentation

## Scaling Strategy

### Start Small
- Single server
- Simple scripts
- Manual scheduling
- ~10GB/day

### Scale Gradually

**Phase 1: Add Orchestration**
- Workflow engine (Airflow)
- Automated scheduling
- ~100GB/day

**Phase 2: Distributed Processing**
- Spark cluster
- Parallel processing
- ~1TB/day

**Phase 3: Optimize**
- Incremental processing
- Optimized transformations
- Parallel extraction/loading
- ~10TB/day

## Trade-offs

### ETL vs ELT
- **ETL**: More control, complex, slower
- **ELT**: Simpler, faster, less control
- **Recommendation**: ETL for complex logic, ELT for large datasets

### Batch vs Streaming
- **Batch**: Simpler, lower cost, higher latency
- **Streaming**: More complex, higher cost, lower latency
- **Recommendation**: Batch for most cases, streaming for real-time

### Full vs Incremental
- **Full**: Simpler, expensive, slower
- **Incremental**: More complex, efficient, faster
- **Recommendation**: Incremental for large datasets

## Operational Concerns

### Monitoring
- Job execution status
- Processing time
- Data volume processed
- Error rates
- Data quality metrics

### Failure Handling
- Automatic retry
- Error handling
- Dead letter queue
- Alerting

### Data Quality
- Validation rules
- Quality metrics
- Anomaly detection
- Data profiling

### Cost Optimization
- Use spot instances
- Optimize transformations
- Incremental processing
- Compress data

## Follow-up Questions

**Be prepared to answer:**
1. "How do you handle schema changes?"
2. "How would you ensure data quality?"
3. "How do you handle failures?"
4. "How would you optimize processing time?"
5. "How do you track data lineage?"
6. "How would you handle late-arriving data?"
7. "How do you handle data backfills?"

## Example Solutions

### Simple Solution (Start Here)
- Single server
- Simple ETL scripts
- Manual scheduling
- ~10GB/day

### Optimized Solution
- Workflow engine (Airflow)
- Distributed processing (Spark)
- Incremental processing
- Data quality checks
- ~1TB/day

### Production Solution
- Advanced orchestration
- Distributed processing
- Incremental + CDC
- Comprehensive data quality
- Data lineage tracking
- ~10TB+/day

## Practice Exercise

**Design this system in 45-60 minutes:**
1. Clarify requirements (5-10 min)
2. High-level design (10-15 min)
3. Detailed design (15-20 min)
4. Scaling and optimization (10-15 min)
5. Trade-offs discussion (5 min)

**Focus on:**
- ETL vs ELT pattern
- Extraction strategies
- Transformation design
- Loading optimization
- Orchestration

