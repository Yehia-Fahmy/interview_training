# Case Study 03: Real-time Matching System

## Overview

This case study explores how a real-time matching system connecting users with service providers might be architected, handling millions of requests per day across hundreds of cities.

## Scale Requirements

- **Riders**: Millions of active riders
- **Drivers**: Millions of active drivers
- **Requests**: Millions of ride requests per day
- **Latency**: < 2 seconds for matching
- **Geographic Scale**: Global, hundreds of cities
- **Real-time**: Must be real-time (not batch)

## Architecture Overview

### High-Level Architecture

```
[Client App] → [Request Service] → [Matching Engine]
                                    ├── [Provider Location Service]
                                    ├── [ETA Service]
                                    └── [Pricing Service]
                                         ↓
                                    [Notification Service]
                                         ↓
                                    [Provider App]
```

## Key Components

### 1. Request Service

**Functions:**
- Receive service requests from users
- Validate requests
- Route to appropriate matching engine (by city/region)
- Handle request lifecycle

**Request Flow:**
1. User requests service
2. Request validated (location, payment, etc.)
3. Route to matching engine for city
4. Start matching process
5. Return match result

### 2. Provider Location Service

**Functions:**
- Track provider locations in real-time
- Maintain provider availability
- Update provider status (available, on-service, offline)
- Geospatial indexing for fast lookups

**Implementation:**
- **Geospatial Database**: Redis Geo, MongoDB Geospatial
- **Indexing**: R-tree or similar for spatial queries
- **Updates**: Real-time location updates (every few seconds)
- **Queries**: Find drivers within radius of pickup location

**Data Model:**
```
Provider {
  provider_id: "123",
  location: {lat: 37.7749, lng: -122.4194},
  status: "available",
  service_type: "standard",
  last_update: timestamp
}
```

### 3. Matching Engine

**Matching Algorithm:**

1. **Find Candidates**:
   - Query providers within radius (e.g., 5 miles)
   - Filter by availability, service type
   - Consider provider preferences

2. **Score Providers**:
   - ETA to pickup location
   - Provider rating
   - Provider distance
   - Historical performance

3. **Select Best Match**:
   - Rank by score
   - Consider provider preferences
   - Assign service

**Optimization:**
- Pre-compute provider clusters
- Cache common queries
- Parallel processing
- Batching (match multiple requests together)

### 4. ETA Service

**Functions:**
- Calculate estimated time of arrival
- Consider traffic, distance, route
- Update ETAs in real-time

**Implementation:**
- **Route Planning**: Mapping API, internal routing
- **Traffic Data**: Real-time traffic information
- **Historical Data**: Historical travel times
- **Machine Learning**: Predict ETAs using ML models

**Caching:**
- Cache ETAs for common routes
- Update based on traffic changes
- Invalidate stale ETAs

### 5. Pricing Service

**Dynamic Pricing (Surge Pricing):**

**Factors:**
- Supply (available drivers)
- Demand (ride requests)
- Time of day
- Weather conditions
- Events

**Algorithm:**
- Monitor supply/demand ratio
- Calculate surge multiplier
- Update prices dynamically
- Display to riders

**Implementation:**
- Real-time monitoring
- Machine learning models
- A/B testing
- Price optimization

### 6. Notification Service

**Functions:**
- Notify providers of service requests
- Notify users of provider assignment
- Send updates (provider ETA, arrival, etc.)

**Implementation:**
- Push notifications (APNs, FCM)
- In-app notifications
- SMS (fallback)
- Real-time updates via WebSocket

### 7. Real-time Data Pipeline

**Event Stream:**
- Provider location updates
- Service requests
- Service status changes
- Pricing updates

**Processing:**
- Apache Kafka for event streaming
- Real-time processing (Flink, Storm)
- Update provider locations
- Trigger matching
- Update pricing

## Design Decisions & Trade-offs

### 1. Geospatial Database

**Trade-off**: Query Speed vs Update Speed
- **R-tree Index**: Fast queries, slower updates
- **Grid-based**: Fast updates, more memory
- **Decision**: R-tree for queries, optimize updates with batching

### 2. Matching Strategy

**Trade-off**: Optimality vs Latency
- **Optimal matching**: Best match, slower (seconds)
- **Greedy matching**: Good match, faster (< 1s)
- **Decision**: Greedy matching for real-time, optimize for common cases

### 3. ETA Calculation

**Trade-off**: Accuracy vs Latency
- **Detailed routing**: More accurate, slower
- **Simplified routing**: Less accurate, faster
- **Decision**: Use ML models for fast, accurate ETAs

### 4. Pricing Strategy

**Trade-off**: Revenue vs User Experience
- **Higher prices**: More revenue, lower demand
- **Lower prices**: Less revenue, higher demand
- **Decision**: Dynamic pricing to balance supply/demand

## Scaling Strategy

### Phase 1: Single City
- Simple matching algorithm
- Basic geospatial queries
- Manual pricing
- ~10K requests/day

### Phase 2: Multiple Cities
- City-specific matching engines
- Optimized geospatial indexing
- Automated pricing
- ~100K requests/day

### Phase 3: Global Scale
- Multi-region deployment
- Advanced matching algorithms
- ML-based pricing
- ~1M requests/day

### Phase 4: Large Scale
- Real-time matching at scale
- Advanced ML models
- Comprehensive optimization
- Millions of requests/day

## Key Learnings

1. **Geospatial Indexing**: Critical for fast driver lookups
2. **Real-time Processing**: Essential for matching and pricing
3. **Greedy Matching**: Good balance of quality and latency
4. **Dynamic Pricing**: Balances supply and demand effectively
5. **Event-Driven Architecture**: Enables real-time updates
6. **Multi-Region**: Necessary for global scale

## Interview Takeaways

When designing real-time matching systems:
- Use geospatial databases for location queries
- Implement greedy matching for low latency
- Consider supply/demand balance
- Design for real-time updates
- Handle high write loads (location updates)
- Optimize for common queries
- Consider multi-region deployment

## References

- Industry best practices for real-time systems
- Technical talks on real-time matching systems
- Research papers on geospatial matching algorithms

