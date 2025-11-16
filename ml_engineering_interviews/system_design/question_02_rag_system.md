# System Design: RAG Serving System

**Target Role:** EvenUp  
**Difficulty:** ⭐⭐⭐  
**Time:** 45 minutes

## Problem Statement

Design a Retrieval-Augmented Generation (RAG) system for document-based question answering. The system should be able to ingest legal documents, answer questions about them, and scale to handle millions of documents and thousands of queries per second.

## Initial Requirements to Clarify

1. **Scale**: How many documents? Query rate? Document size?
2. **Document types**: What formats? (PDF, Word, HTML, etc.)
3. **Query types**: Simple questions? Complex multi-hop reasoning?
4. **Latency requirements**: P50, P95, P99 latency?
5. **Accuracy requirements**: How important is correctness?
6. **Update frequency**: How often do documents change?

**Assumed Requirements:**
- Millions of documents
- Thousands of queries per second
- Documents range from 1 page to 1000+ pages
- P95 latency < 2 seconds
- Support for multi-document queries
- Real-time document updates

## High-Level Architecture

```
┌─────────────┐
│   Users     │
│  (Queries)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│      Query Router                   │
│  (Query Understanding, Routing)     │
└──────┬──────────────────────────────┘
       │
       ├─────────────────┬─────────────┐
       ▼                 ▼             ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Embedding  │  │   Vector    │  │    LLM      │
│  Generation │  │   Search    │  │  Inference  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┴────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Response       │
              │  Generation    │
              └─────────────────┘
```

## Core Components

### 1. Document Ingestion Pipeline

**Responsibilities:**
- Parse documents (PDF, Word, HTML)
- Extract text and structure
- Chunk documents appropriately
- Generate embeddings
- Store in vector database

**Design:**
- **Parsing**: Use libraries like PyPDF2, python-docx, BeautifulSoup
- **Chunking**: Sliding window with overlap to preserve context
- **Embedding**: Batch embedding generation using sentence transformers
- **Storage**: Vector database (Pinecone, Weaviate, or FAISS) + metadata DB

**Chunking Strategy:**
```python
def chunk_document(text, chunk_size=512, overlap=50):
    # Split into chunks with overlap
    # Preserve sentence boundaries
    # Add metadata (document_id, chunk_index, page_number)
    pass
```

### 2. Embedding Generation

**Responsibilities:**
- Generate embeddings for document chunks
- Generate embeddings for queries
- Handle batch processing
- Update embeddings when documents change

**Design:**
- **Model**: Use sentence transformers (all-MiniLM-L6-v2 or similar)
- **Batch processing**: Process chunks in batches for efficiency
- **Caching**: Cache embeddings to avoid regeneration
- **Async processing**: Generate embeddings asynchronously for new documents

**Optimization:**
- Use GPU for batch embedding generation
- Cache frequently accessed embeddings
- Use quantization for faster inference

### 3. Vector Search

**Responsibilities:**
- Store document embeddings
- Perform similarity search
- Return top-k relevant chunks
- Handle filtering and metadata queries

**Design:**
- **Vector DB**: Pinecone (managed) or Weaviate/FAISS (self-hosted)
- **Indexing**: HNSW or IVF for fast approximate search
- **Metadata filtering**: Support filtering by document type, date, etc.
- **Hybrid search**: Combine vector search with keyword search

**Search Strategy:**
```python
def search(query_embedding, top_k=5, filters=None):
    # Vector similarity search
    # Apply metadata filters
    # Rerank results (optional)
    # Return top-k chunks with scores
    pass
```

### 4. LLM Inference

**Responsibilities:**
- Generate answers from retrieved context
- Handle prompt engineering
- Manage context window limits
- Reduce hallucinations

**Design:**
- **Model**: Use GPT-4, Claude, or open-source LLMs (Llama, Mistral)
- **Prompting**: Few-shot examples, chain-of-thought
- **Context management**: Truncate if exceeds context limit
- **Caching**: Cache common queries

**Prompt Template:**
```
Context: {retrieved_chunks}

Question: {user_query}

Answer the question based on the context above. 
If the answer is not in the context, say "I don't know."
```

### 5. Response Generation

**Responsibilities:**
- Combine retrieved chunks
- Generate final answer
- Add citations
- Format response

**Design:**
- **Reranking**: Rerank retrieved chunks by relevance
- **Context aggregation**: Combine multiple chunks intelligently
- **Citation**: Include source document and chunk references
- **Formatting**: Structure response (answer, sources, confidence)

## Scalability Considerations

### Horizontal Scaling
- **Embedding workers**: Scale based on document ingestion rate
- **Vector DB**: Shard by document type or date
- **LLM serving**: Multiple inference instances with load balancing
- **Query processing**: Stateless workers, scale with query rate

### Caching Strategy
- **Query cache**: Cache common queries and answers
- **Embedding cache**: Cache document embeddings
- **Retrieval cache**: Cache retrieval results for similar queries
- **LLM cache**: Cache LLM responses for identical queries

### Database Optimization
- **Vector DB**: Use managed service (Pinecone) or optimize FAISS
- **Metadata DB**: PostgreSQL with proper indexing
- **Read replicas**: For metadata queries

## Reliability and Failure Handling

### Failure Modes
1. **Document parsing fails**: Log error, skip document, notify
2. **Embedding generation fails**: Retry with exponential backoff
3. **Vector search fails**: Fallback to keyword search
4. **LLM inference fails**: Retry or use fallback model
5. **Timeout**: Set timeouts, return partial results

### Data Consistency
- **Eventual consistency**: Embeddings generated asynchronously
- **Versioning**: Track document versions
- **Updates**: Invalidate cache when documents update

### Monitoring
- **Metrics**: Query latency, retrieval quality, answer accuracy
- **Alerts**: High latency, low retrieval quality, LLM errors
- **Logging**: Log all queries and responses for analysis

## Follow-up Questions

### Q1: How do you handle long documents that exceed context limits?

**Answer:**
- **Chunking**: Split documents into smaller chunks with overlap
- **Hierarchical retrieval**: Retrieve relevant sections first, then chunks
- **Multi-stage retrieval**: Coarse-grained then fine-grained
- **Summarization**: Summarize long documents before embedding

**Follow-up:**
- "What if a single chunk is too long?"
- "How do you preserve context across chunks?"

### Q2: How do you ensure retrieval quality?

**Answer:**
- **Hybrid search**: Combine vector and keyword search
- **Reranking**: Use cross-encoder to rerank top results
- **Query expansion**: Expand queries with synonyms
- **Feedback loop**: Use user feedback to improve retrieval

**Follow-up:**
- "How do you handle ambiguous queries?"
- "What if no relevant documents exist?"

### Q3: How do you scale embedding generation?

**Answer:**
- **Batch processing**: Process multiple chunks in batches
- **GPU acceleration**: Use GPUs for faster embedding
- **Async processing**: Generate embeddings asynchronously
- **Caching**: Cache embeddings to avoid regeneration

**Follow-up:**
- "What if you need to regenerate all embeddings?"
- "How do you handle embedding model updates?"

### Q4: How do you handle multi-document queries?

**Answer:**
- **Multi-vector search**: Search across multiple document collections
- **Aggregation**: Combine results from multiple searches
- **Deduplication**: Remove duplicate chunks
- **Context prioritization**: Prioritize more relevant documents

**Follow-up:**
- "What if documents contradict each other?"
- "How do you handle conflicting information?"

### Q5: How do you reduce hallucinations?

**Answer:**
- **Grounding**: Only use retrieved context, don't use model knowledge
- **Citation**: Require citations for all claims
- **Verification**: Cross-check answers against source documents
- **Confidence scores**: Provide confidence scores for answers
- **Prompt engineering**: Explicitly instruct model to say "I don't know"

**Follow-up:**
- "What if the model is confident but wrong?"
- "How do you detect hallucinations automatically?"

### Q6: How do you evaluate RAG system performance?

**Answer:**
- **Retrieval metrics**: Precision, recall, MRR (Mean Reciprocal Rank)
- **Answer quality**: BLEU, ROUGE, semantic similarity
- **Human evaluation**: Expert review of answers
- **A/B testing**: Compare different retrieval/LLM strategies

**Follow-up:**
- "How do you automate evaluation?"
- "What metrics matter most for your use case?"

## Trade-offs

### Accuracy vs Latency
- **More chunks**: Better accuracy but higher latency
- **Reranking**: Better results but slower
- **Solution**: Configurable retrieval parameters, use faster models for low-latency queries

### Cost vs Quality
- **Better LLM**: Higher cost but better quality
- **More chunks**: Higher embedding cost but better context
- **Solution**: Tiered service levels, use cheaper models for simple queries

### Consistency vs Performance
- **Synchronous updates**: Consistent but slower
- **Asynchronous updates**: Faster but eventual consistency
- **Solution**: Async updates with versioning and cache invalidation

## Implementation Considerations

### Technology Stack
- **Document parsing**: PyPDF2, python-docx, BeautifulSoup
- **Embeddings**: sentence-transformers, HuggingFace
- **Vector DB**: Pinecone (managed) or Weaviate/FAISS (self-hosted)
- **LLM**: OpenAI API, Anthropic API, or self-hosted (Llama, Mistral)
- **API**: FastAPI or Flask
- **Queue**: Redis Queue or RabbitMQ for async processing

### Security
- **Authentication**: API keys or OAuth
- **Authorization**: Role-based access to documents
- **Data privacy**: Encrypt documents at rest
- **Input validation**: Validate queries and documents

## Summary

This RAG system provides:
1. **Scalable document processing** for millions of documents
2. **Fast retrieval** with vector search
3. **Accurate answers** with LLM generation
4. **Reliability** with error handling and monitoring
5. **Quality assurance** with evaluation and feedback

Key success factors:
- Efficient chunking and embedding strategy
- High-quality retrieval with hybrid search
- Proper prompt engineering to reduce hallucinations
- Comprehensive evaluation and monitoring

