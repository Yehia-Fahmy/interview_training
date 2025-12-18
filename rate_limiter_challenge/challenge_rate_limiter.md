# Challenge: API Rate Limiter

## Problem Statement

You are building a rate limiting system for a ride-sharing API. The system needs to protect backend services from being overwhelmed by limiting the number of requests a client can make within a given time window.

Your task is to implement a **rate limiter** that supports multiple algorithms and can handle high-throughput scenarios.

---

## Business Context

- **Goal**: Protect API infrastructure from abuse and ensure fair resource allocation
- **Use Cases**:
  - Limit ride requests per user to prevent spam
  - Throttle third-party API integrations
  - Implement tiered rate limits (free vs premium users)
  - Protect against DDoS attacks
- **Constraints**: 
  - Must handle 100K+ requests per second
  - Decision latency < 1ms
  - Memory efficient (millions of unique clients)

---

## Requirements

### Core Requirements

1. **Basic Rate Limiter**
   - Implement `is_allowed(client_id: str) -> bool`
   - Return `True` if request is allowed, `False` if rate limited
   - Support configurable limits (e.g., 100 requests per minute)

2. **Algorithm Implementation**
   Choose and implement at least ONE of these algorithms:
   
   | Algorithm | Description | Pros | Cons |
   |-----------|-------------|------|------|
   | **Fixed Window** | Count requests in fixed time buckets | Simple, memory efficient | Burst at window boundaries |
   | **Sliding Window Log** | Track timestamp of each request | Accurate | Memory intensive |
   | **Sliding Window Counter** | Weighted average of current + previous window | Good balance | Approximate |
   | **Token Bucket** | Tokens refill at constant rate | Allows bursts | More complex |
   | **Leaky Bucket** | Requests processed at constant rate | Smooth output | May drop requests |

3. **Rate Limit Info**
   - Return remaining requests allowed
   - Return time until limit resets
   - Return whether client is currently limited

### Advanced Requirements (Stretch Goals)

4. **Multiple Limits**
   - Support multiple rate limits per client (e.g., 10/second AND 1000/hour)

5. **Tiered Limits**
   - Different limits for different client tiers (free, premium, enterprise)

6. **Distributed Rate Limiting**
   - Design for multiple server instances (discuss approach)

---

## Interface Specification

```python
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        pass
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a request from client_id should be allowed.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            True if request is allowed, False if rate limited
        """
        pass
    
    def get_limit_info(self, client_id: str) -> dict:
        """
        Get rate limit information for a client.
        
        Returns:
            {
                "allowed": bool,
                "remaining": int,
                "reset_at": float,  # Unix timestamp
                "retry_after": float  # Seconds until next allowed request
            }
        """
        pass
```

---

## Algorithm Deep Dive

### Fixed Window Counter

```
Window 1 (0:00-0:59)    Window 2 (1:00-1:59)
[|||||||||||]           [||||]
 11 requests             4 requests

Limit: 10/minute
Result: Window 1 exceeded, Window 2 OK
```

**Implementation Hint**: Use `int(time.time() // window_seconds)` as window key.

### Sliding Window Log

```
Timeline: ----[--r--r--r--r--]----
                 ^window^
                 
Keep list of timestamps, remove old ones, count remaining.
```

**Implementation Hint**: Use `collections.deque` for efficient removal from front.

### Token Bucket

```
Bucket capacity: 10 tokens
Refill rate: 1 token/second

Time 0: [##########] 10 tokens, request → 9 tokens
Time 1: [##########] 10 tokens (refilled), request → 9 tokens
Time 5: [##########] 10 tokens (capped at max)
```

**Implementation Hint**: Calculate tokens based on time elapsed since last request.

---

## Time Budget (90 minutes)

| Phase | Time | Focus |
|-------|------|-------|
| Problem Review | 10 min | Understand requirements, choose algorithm |
| Core Implementation | 50 min | Implement chosen algorithm |
| Testing & Edge Cases | 15 min | Test with various scenarios |
| Discussion | 15 min | Trade-offs, distributed systems, production |

---

## Evaluation Rubric

| Dimension | Weight | What We Look For |
|-----------|--------|------------------|
| Algorithm Choice | 20% | Appropriate algorithm for requirements |
| Correctness | 30% | Accurate rate limiting, handles edge cases |
| Code Quality | 25% | Clean, readable, well-structured |
| Performance | 15% | O(1) operations, memory efficient |
| Understanding | 10% | Can explain trade-offs and alternatives |

---

## Test Scenarios

Your implementation should handle:

1. **Basic limiting**: 10 requests allowed, 11th blocked
2. **Window reset**: After window expires, requests allowed again
3. **Multiple clients**: Each client has independent limits
4. **Burst traffic**: Many requests in short time
5. **Edge cases**: Empty client_id, negative limits, etc.

---

## Production Discussion Topics

Be prepared to discuss:

1. **Distributed Rate Limiting**: How would you implement this across multiple servers?
   - Redis-based approach?
   - Consistent hashing?
   - Eventual consistency trade-offs?

2. **Race Conditions**: How do you handle concurrent requests from same client?

3. **Monitoring**: What metrics would you track?

4. **Graceful Degradation**: What happens if rate limiter fails?

5. **Client Communication**: How do you inform clients they're rate limited?
   - HTTP 429 status code
   - Retry-After header
   - X-RateLimit-* headers

---

## Example Usage

```python
# Create limiter: 10 requests per 60 seconds
limiter = RateLimiter(max_requests=10, window_seconds=60)

# Simulate requests
for i in range(15):
    client = "user_123"
    allowed = limiter.is_allowed(client)
    print(f"Request {i+1}: {'ALLOWED' if allowed else 'BLOCKED'}")

# Check limit info
info = limiter.get_limit_info("user_123")
print(f"Remaining: {info['remaining']}, Reset in: {info['retry_after']:.1f}s")
```

Expected output:
```
Request 1: ALLOWED
Request 2: ALLOWED
...
Request 10: ALLOWED
Request 11: BLOCKED
Request 12: BLOCKED
...
Remaining: 0, Reset in: 58.2s
```

---

## Hints

- Start with the simplest algorithm (Fixed Window) to get something working
- Use `time.time()` for current timestamp
- Consider using `defaultdict` for per-client tracking
- Think about memory cleanup for inactive clients

---

Good luck! Remember: a working simple solution beats a broken complex one.
