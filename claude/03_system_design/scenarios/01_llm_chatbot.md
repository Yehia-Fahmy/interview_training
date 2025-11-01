# System Design: LLM-Powered Chatbot Platform

## Problem Statement

Design a production-ready LLM-powered chatbot platform that can handle customer support queries for multiple companies. The system should provide context-aware responses, integrate with company knowledge bases, and scale to millions of users.

**This is highly relevant to 8090's Software Factory**, which uses LLMs and agentic systems for software development tasks.

## Requirements Clarification

### Functional Requirements
1. **Core Functionality**
   - Accept user queries in natural language
   - Provide relevant, context-aware responses
   - Support multi-turn conversations with memory
   - Integrate with company-specific knowledge bases (documents, FAQs, APIs)
   - Handle multiple companies (multi-tenancy)

2. **Advanced Features**
   - Escalate to human agents when needed
   - Support multiple languages
   - Provide suggested responses for common queries
   - Track conversation history
   - Analytics and insights dashboard

3. **Integration**
   - REST API for third-party integration
   - Webhooks for notifications
   - Support for popular chat platforms (Slack, Discord, web widget)

### Non-Functional Requirements

#### Scale
- **Users**: 1M daily active users across all companies
- **Companies**: 1000 companies using the platform
- **Conversations**: 100K concurrent conversations
- **Messages**: 10K messages per second peak
- **Knowledge Base**: 100GB of documents per company

#### Performance
- **Latency**: 
  - p50: < 1 second
  - p95: < 2 seconds
  - p99: < 3 seconds
- **Availability**: 99.9% uptime (8.76 hours downtime/year)
- **Throughput**: Handle 10K messages/second

#### Other
- **Cost**: Optimize LLM API costs
- **Security**: Encrypt data at rest and in transit, GDPR compliant
- **Reliability**: No message loss, graceful degradation

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│  (Web Widget, Mobile App, Slack, Discord, API Clients)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway                                 │
│  (Rate Limiting, Authentication, Request Routing)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Application Layer                              │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Chatbot    │  │  Knowledge   │  │   Analytics  │          │
│  │   Service    │  │    Service   │  │   Service    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                  │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PostgreSQL  │  │    Redis     │  │   Vector DB  │          │
│  │ (Metadata)   │  │   (Cache)    │  │ (Embeddings) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   External Services                              │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  LLM API     │  │  Monitoring  │  │  Message     │          │
│  │(GPT/Claude)  │  │(DataDog/New  │  │  Queue       │          │
│  │              │  │  Relic)      │  │  (Kafka)     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Design

### 1. API Gateway

**Responsibilities**:
- Request routing
- Rate limiting (per company, per user)
- Authentication and authorization
- Request/response logging
- SSL termination

**Technology Choices**:
- **Option 1**: NGINX + Kong (open source, highly configurable)
- **Option 2**: AWS API Gateway (managed, scales automatically)
- **Option 3**: Envoy (service mesh, advanced routing)

**Recommendation**: Kong for flexibility and cost control

**Rate Limiting Strategy**:
```
- Free tier: 100 messages/day per user
- Pro tier: 1000 messages/day per user
- Enterprise: Custom limits
- Burst handling: Token bucket algorithm
```

### 2. Chatbot Service

**Core Components**:

#### a) Conversation Manager
```python
class ConversationManager:
    """
    Manages conversation state and context
    """
    - load_conversation(conversation_id)
    - save_message(message)
    - get_context(max_messages=10)
    - summarize_history()  # For long conversations
```

**Context Window Management**:
- Keep last N messages in context (N=10 default)
- Summarize older messages to save tokens
- Store full history in database

#### b) Intent Classifier
```python
class IntentClassifier:
    """
    Determines user intent to route appropriately
    """
    - classify(message) -> Intent
    - confidence_score() -> float
    - should_escalate() -> bool
```

**Intents**:
- `question`: General question (use RAG)
- `complaint`: Escalate to human
- `transaction`: Needs API call
- `chitchat`: Simple response, no RAG needed

#### c) Response Generator
```python
class ResponseGenerator:
    """
    Generates responses using LLM
    """
    - generate_response(query, context, knowledge)
    - validate_response()
    - add_citations()
```

**Prompt Structure**:
```
System: You are a helpful customer support assistant for {company_name}.
Use the provided context to answer questions accurately.
If you don't know, say so - don't make up information.

Context: {retrieved_knowledge}

Conversation History:
{conversation_history}

User: {current_query}
```

### 3. Knowledge Service (RAG System)

**Components**:

#### a) Document Ingestion Pipeline
```
Upload → Parse (PDF/HTML/Markdown) → Chunk → Embed → Store
```

**Chunking Strategy**:
- Chunk size: 512 tokens (balance between context and granularity)
- Overlap: 50 tokens (preserve context across chunks)
- Respect document structure (don't split mid-sentence)

#### b) Vector Database
**Options**:
- **Pinecone**: Managed, easy to use, expensive at scale
- **Weaviate**: Open source, self-hosted, flexible
- **Qdrant**: Fast, Rust-based, good for production

**Recommendation**: Weaviate for cost control and flexibility

**Schema**:
```json
{
  "id": "doc_chunk_id",
  "company_id": "company_123",
  "content": "chunk text",
  "embedding": [0.1, 0.2, ...],
  "metadata": {
    "source": "faq.pdf",
    "page": 5,
    "timestamp": "2024-01-01"
  }
}
```

#### c) Retrieval Strategy
```python
def retrieve_relevant_context(query, company_id, top_k=5):
    # 1. Embed query
    query_embedding = embed_model.encode(query)
    
    # 2. Vector search
    results = vector_db.search(
        embedding=query_embedding,
        filter={"company_id": company_id},
        limit=top_k
    )
    
    # 3. Rerank (optional but recommended)
    reranked = reranker.rerank(query, results)
    
    return reranked[:3]  # Return top 3 after reranking
```

**Optimization**: Cache frequent queries in Redis

### 4. Data Layer

#### PostgreSQL Schema
```sql
-- Companies
CREATE TABLE companies (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    tier VARCHAR(50),  -- free, pro, enterprise
    settings JSONB,
    created_at TIMESTAMP
);

-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    company_id UUID REFERENCES companies(id),
    user_id VARCHAR(255),
    status VARCHAR(50),  -- active, closed, escalated
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id),
    role VARCHAR(50),  -- user, assistant, system
    content TEXT,
    metadata JSONB,  -- tokens_used, latency, etc.
    created_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_conversations_company ON conversations(company_id);
CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_messages_created ON messages(created_at);
```

#### Redis Caching Strategy
```
Cache Keys:
- conversation:{id}:context  (TTL: 1 hour)
- company:{id}:config  (TTL: 24 hours)
- query:{hash}:response  (TTL: 1 hour, for common queries)
- user:{id}:rate_limit  (TTL: 1 day)
```

## Scalability Considerations

### Horizontal Scaling

**Stateless Services**:
- Chatbot Service: Scale based on CPU/memory
- Knowledge Service: Scale based on query volume
- Auto-scaling: Scale up at 70% CPU, scale down at 30%

**Stateful Services**:
- PostgreSQL: Read replicas for read-heavy workloads
- Redis: Redis Cluster for horizontal scaling
- Vector DB: Sharding by company_id

### Load Balancing

**Strategy**: Least connections (better for long-running LLM requests)

**Health Checks**:
```python
@app.get("/health")
def health_check():
    checks = {
        "database": check_db_connection(),
        "redis": check_redis_connection(),
        "llm_api": check_llm_api(),
    }
    return {"status": "healthy" if all(checks.values()) else "unhealthy"}
```

### Caching Strategy

**Multi-Level Caching**:
1. **L1 (Application)**: In-memory LRU cache (10K items)
2. **L2 (Redis)**: Distributed cache
3. **L3 (CDN)**: Static assets and common responses

**What to Cache**:
- Conversation context (hot data)
- Company configurations
- Common query responses
- Embedding vectors for frequent queries

## Cost Optimization

### LLM API Costs

**Strategies**:
1. **Intent-based routing**: Use smaller models for simple queries
2. **Response caching**: Cache responses for similar queries
3. **Prompt optimization**: Minimize token usage
4. **Batch processing**: Batch non-urgent requests

**Cost Breakdown** (estimated for 1M messages/day):
```
- GPT-4: $0.03/1K tokens * 500 tokens/message * 1M = $15K/day
- GPT-3.5: $0.002/1K tokens * 500 tokens/message * 1M = $1K/day
- Hybrid (80% GPT-3.5, 20% GPT-4): ~$4K/day

Savings with caching (30% hit rate): ~$2.8K/day
```

### Infrastructure Costs

**Monthly Estimate**:
- Compute (Kubernetes): $5K
- Database (PostgreSQL + Redis): $2K
- Vector DB (Weaviate): $3K
- Monitoring & Logging: $1K
- **Total**: ~$11K/month + LLM costs

## Monitoring and Observability

### Key Metrics

**System Metrics**:
- Request rate (messages/second)
- Response latency (p50, p95, p99)
- Error rate (%)
- Cache hit rate (%)

**ML Metrics**:
- LLM token usage
- Retrieval quality (relevance score)
- Response quality (user feedback)
- Escalation rate (%)

**Business Metrics**:
- Active conversations
- User satisfaction (CSAT)
- Resolution rate
- Cost per conversation

### Alerting Rules

```yaml
alerts:
  - name: HighLatency
    condition: p95_latency > 3s for 5 minutes
    action: page_oncall
  
  - name: HighErrorRate
    condition: error_rate > 5% for 5 minutes
    action: page_oncall
  
  - name: LLMAPIDown
    condition: llm_api_success_rate < 95% for 2 minutes
    action: page_oncall
  
  - name: HighCost
    condition: daily_llm_cost > $20K
    action: notify_team
```

### Logging Strategy

```python
# Structured logging
logger.info(
    "message_processed",
    extra={
        "conversation_id": conv_id,
        "company_id": company_id,
        "latency_ms": latency,
        "tokens_used": tokens,
        "cache_hit": cache_hit,
        "model": model_name
    }
)
```

## Security Considerations

### Data Protection
- **Encryption at rest**: AES-256 for database
- **Encryption in transit**: TLS 1.3
- **PII handling**: Tokenize sensitive data
- **Data retention**: Configurable per company (GDPR compliance)

### Access Control
- **Authentication**: OAuth 2.0 / JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **API keys**: Rotate every 90 days
- **Audit logging**: Log all data access

### LLM Safety
- **Content filtering**: Block harmful content
- **Prompt injection protection**: Validate and sanitize inputs
- **Output validation**: Check for sensitive data leakage
- **Rate limiting**: Prevent abuse

## Failure Modes and Mitigation

### LLM API Failure
**Impact**: Cannot generate responses
**Mitigation**:
- Fallback to cached responses
- Use backup LLM provider
- Queue requests for retry
- Graceful degradation (canned responses)

### Database Failure
**Impact**: Cannot access conversation history
**Mitigation**:
- Database replication (master-slave)
- Automatic failover
- Regular backups (point-in-time recovery)

### Vector DB Failure
**Impact**: Cannot retrieve knowledge
**Mitigation**:
- Replica sets
- Fallback to keyword search
- Cached popular queries

### High Load
**Impact**: Increased latency, potential timeouts
**Mitigation**:
- Auto-scaling
- Request queuing
- Circuit breakers
- Load shedding (reject low-priority requests)

## Trade-offs and Alternatives

### LLM Selection
**GPT-4 vs GPT-3.5**:
- GPT-4: Better quality, 10x cost, slower
- GPT-3.5: Good quality, cheaper, faster
- **Decision**: Use GPT-3.5 by default, GPT-4 for complex queries

### Vector DB vs Traditional Search
**Vector DB**:
- Pros: Semantic search, better relevance
- Cons: More expensive, requires embeddings
**Traditional Search (Elasticsearch)**:
- Pros: Cheaper, faster for exact matches
- Cons: Misses semantic similarity
- **Decision**: Hybrid approach (vector + keyword)

### Synchronous vs Asynchronous Processing
**Synchronous**:
- Pros: Immediate response, simpler
- Cons: Blocks on slow LLM calls
**Asynchronous**:
- Pros: Better throughput, non-blocking
- Cons: More complex, requires polling/websockets
- **Decision**: Async with streaming for better UX

## Future Enhancements

1. **Multi-modal support**: Images, voice, video
2. **Proactive messaging**: Anticipate user needs
3. **Advanced analytics**: Sentiment analysis, topic modeling
4. **Fine-tuned models**: Company-specific fine-tuning
5. **Agent capabilities**: Book appointments, process refunds
6. **Multi-language**: Automatic translation
7. **Voice interface**: Speech-to-text, text-to-speech

## Interview Discussion Points

### Questions to Expect
1. How do you handle context window limits?
2. How do you prevent prompt injection attacks?
3. How do you measure response quality?
4. How do you optimize costs?
5. How do you handle multi-tenancy?
6. How do you ensure low latency?
7. How do you handle model updates?

### Key Points to Emphasize
- **RAG architecture** for knowledge grounding
- **Caching strategy** for cost and latency
- **Monitoring** for production reliability
- **Multi-tenancy** for SaaS model
- **Scalability** through stateless services
- **Cost optimization** through smart routing

## Summary

This design provides:
- ✅ Scalable architecture (1M+ users)
- ✅ Low latency (<2s p95)
- ✅ Cost-effective (smart caching and routing)
- ✅ Reliable (redundancy and failover)
- ✅ Secure (encryption and access control)
- ✅ Observable (comprehensive monitoring)

**Key Design Decisions**:
1. RAG for knowledge grounding
2. Multi-level caching for performance
3. Hybrid LLM usage for cost optimization
4. Microservices for independent scaling
5. Comprehensive monitoring for reliability
