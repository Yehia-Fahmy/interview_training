# Lyft Laptop Interview Challenge: API Rate Limiter

## Overview

Implement a rate limiter that controls API request rates per client.

## Challenge Structure

| File | Description |
|------|-------------|
| `challenge_rate_limiter.md` | Full problem description and algorithm options |
| `starter_rate_limiter.py` | Minimal starter code with test harness |
| `generate_traffic_data.py` | Generate realistic traffic patterns for testing |

## Quick Start

```bash
# Read the challenge
cat challenge_rate_limiter.md

# Implement your solution in starter_rate_limiter.py
# Then run tests
python starter_rate_limiter.py
```

## Time Breakdown (90 minutes)

| Phase | Time | Focus |
|-------|------|-------|
| Problem Review | 10 min | Choose algorithm, clarify requirements |
| Core Implementation | 50 min | Implement rate limiter |
| Testing | 15 min | Edge cases, performance |
| Discussion | 15 min | Distributed systems, production concerns |

## Algorithm Options

| Algorithm | Complexity | Memory | Accuracy |
|-----------|------------|--------|----------|
| Fixed Window | O(1) | Low | Approximate |
| Sliding Window Log | O(n) | High | Exact |
| Sliding Window Counter | O(1) | Low | Good |
| Token Bucket | O(1) | Low | Exact |

## What We Look For

- **Algorithm Choice**: Pick appropriate algorithm and justify
- **Correctness**: Accurate limiting, handles edge cases
- **Code Quality**: Clean, readable implementation
- **Performance**: O(1) operations, memory efficient
- **Understanding**: Trade-offs, distributed considerations

## Key Concepts

- Time-based windowing
- Per-client state management
- Atomic operations
- Memory efficiency for many clients
- Distributed synchronization

## Production Topics to Prepare

1. How to implement across multiple servers?
2. What happens during clock skew?
3. How to handle rate limiter failures?
4. What HTTP headers to return?
5. How to monitor rate limiting?

---

Good luck!
