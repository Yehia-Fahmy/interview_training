# Lyft Laptop Interview Challenge: Image Caching System

## Overview

Build an LRU (Least Recently Used) image caching system for a ride-sharing/mapping application.

## Challenge Structure

| File | Description |
|------|-------------|
| `challenge_image_caching.md` | Full problem description, requirements, and evaluation criteria |
| `starter_image_caching.py` | Starter code with ImageCache class stubs |
| `generate_image_data.py` | Script to generate test images |
| `images/` | Generated test images (after running data generator) |

## Quick Start

1. **Generate test data**:
   ```bash
   python generate_image_data.py
   ```

2. **Read the challenge description**:
   ```bash
   cat challenge_image_caching.md
   ```

3. **Start coding**:
   ```bash
   # Open starter_image_caching.py and implement your solution
   python starter_image_caching.py
   ```

## Time Breakdown (90 minutes total)

| Phase | Time | Focus |
|-------|------|-------|
| Problem Review | 15 min | Read problem, explore data, ask clarifying questions |
| Core Implementation | 45 min | Implement LRU cache with basic operations |
| Advanced Features | 15 min | Add size-based eviction, TTL, or other features |
| Discussion | 15 min | Discuss approach, trade-offs, production considerations |

## What We Look For

- **Intuition**: Can you translate an ill-defined problem into a working solution?
- **Clarity**: Is your code clean, well-organized, and easy to understand?
- **Correctness**: Does your solution work and handle edge cases?
- **Completeness**: Do you identify limitations and corner cases?
- **Understanding**: Can you explain your decisions and trade-offs?

## Core Requirements

1. **LRU Cache Implementation**
   - O(1) get and put operations
   - Evict least recently used items when cache is full
   - Track cache statistics (hits, misses, hit rate)

2. **Basic Operations**
   - `get(image_id)` - Retrieve cached image
   - `put(image_id, image_data)` - Store image
   - `contains(image_id)` - Check if exists
   - `remove(image_id)` - Remove specific image
   - `clear()` - Clear all cached images

## Advanced Features (If Time Permits)

- Size-based eviction (bytes, not just count)
- TTL (Time-To-Live) support
- Thread safety for concurrent access
- Image loading integration

## Key Data Structures

Consider using:
- `collections.OrderedDict` - Maintains order, supports `move_to_end()`
- Doubly linked list + hash map - Classic O(1) LRU implementation
- `heapq` - For TTL-based expiration (priority queue)

## Interview Tips

1. **Clarify requirements first** - Ask about size limits, eviction policies, concurrency
2. **Start simple** - A working OrderedDict solution beats a broken custom implementation
3. **Think out loud** - Explain your reasoning as you code
4. **Manage time** - Don't spend too long on any single step
5. **Consider production** - Think about deployment, monitoring, scaling

## Production Discussion Topics

Be prepared to discuss:

- How would you deploy this cache in production?
- What metrics would you monitor?
- How would you handle 10x more traffic?
- How would you extend this to multiple servers (distributed caching)?
- How would you handle cache invalidation when images update?
- What happens after a server restart (cold start)?

## Example Usage

```python
# Create cache with 100MB limit
cache = ImageCache(max_size_bytes=100 * 1024 * 1024)

# Store an image
with open("images/map_tile_001.png", "rb") as f:
    image_data = f.read()
cache.put("map_tile_001", image_data)

# Retrieve an image (cache hit)
data = cache.get("map_tile_001")

# Check statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

## Test Your Implementation

The starter file includes a test harness:

```bash
python starter_image_caching.py
```

This will run:
- Basic functionality tests (put, get, eviction)
- Performance tests with simulated workload
- Real image tests (if images are generated)

---

Good luck! Remember: we want to see how you think and approach problems, not just the final answer.
