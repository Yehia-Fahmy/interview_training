# Foundational Prompt – Agent Telemetry Ingestion

**Scenario**: Software Factory wants near-real-time visibility into agent executions across customer projects. Design a telemetry ingestion pipeline that collects structured events, stores them efficiently, and powers dashboards within 60 seconds of an execution.

## Key Requirements
- Ingest up to 50k events/sec from distributed agent runtimes.
- Guarantee at-least-once delivery; duplicates must be deduplicated within five minutes.
- Provide queryable aggregates (per customer, per workflow, per agent version).
- Support retention for 7 days hot, 90 days warm.

## Discussion Topics
- Event schema design and versioning.
- Choice of transport (gRPC vs. HTTPS), ingestion gateways, and authentication.
- Streaming vs. micro-batch processing (Kafka, Pulsar, Flink, Beam, etc.).
- Storage tiering (OLAP vs. time-series) and cost optimizations.
- Alerting strategy when telemetry falls behind or gaps appear.
- Backpressure and replay handling when downstream systems lag.

## Deliverable
- 10–15 minute walkthrough covering architecture diagram, scaling levers, failure recovery, and observability plan.

