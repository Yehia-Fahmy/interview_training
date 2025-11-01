# Exercise 1: Design a Scalable Web Crawler

**Difficulty:** Medium  
**Time Limit:** 45-60 minutes (discussion)  
**Focus:** Distributed systems, scaling, reliability

## Problem

Design a web crawler system that can:
- Crawl billions of web pages
- Handle rate limiting and politeness policies
- Avoid duplicates
- Scale horizontally
- Handle failures gracefully

## Requirements to Discuss

1. **Architecture**
   - How would you structure the system?
   - What components do you need?
   - How to distribute work?

2. **URL Frontier/Queue**
   - How to manage URLs to crawl?
   - How to prioritize URLs?
   - How to handle duplicates?

3. **Distributed Crawling**
   - How to partition work across workers?
   - How to coordinate between crawlers?

4. **Politeness & Rate Limiting**
   - How to respect robots.txt?
   - How to implement rate limiting per domain?

5. **Storage**
   - Where to store crawled content?
   - How to handle billions of pages?

6. **Fault Tolerance**
   - What if a crawler crashes?
   - How to handle network failures?

## Key Topics to Cover

- **Distributed Queue**: RabbitMQ, Kafka, or custom
- **Deduplication**: Bloom filters, distributed hash tables
- **Partitioning**: Domain-based, hash-based
- **Consistency**: Eventual consistency patterns
- **Monitoring**: Health checks, metrics

## Sample Discussion Points

1. "I would use a distributed queue like Kafka to manage URLs. Each crawler worker would consume URLs from partitions assigned to it."

2. "For deduplication, I'd use a distributed Bloom filter. URLs are hashed and checked before crawling."

3. "Rate limiting should be per-domain. Each crawler maintains a per-domain rate limiter with a token bucket algorithm."

4. "For storage, I'd use a distributed object store like S3. Crawled pages would be stored with compression."

5. "Workers would checkpoint their progress. If a worker fails, another can resume from the checkpoint."

## Additional Considerations

- How to handle JavaScript-heavy sites (headless browsers)?
- How to respect robots.txt efficiently?
- How to update already-crawled pages (refresh strategy)?
- How to handle CAPTCHAs and anti-bot measures?

