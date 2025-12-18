# Challenge: Image Caching System

## Problem Statement

You are building an image caching system for a ride-sharing/mapping application. The system needs to efficiently cache map tiles, driver photos, vehicle images, and promotional banners to provide a smooth user experience while minimizing network bandwidth and reducing latency.

Your task is to implement an **LRU (Least Recently Used) image cache** that can store images in memory with configurable size limits and eviction policies.

---

## Business Context

- **Goal**: Reduce image loading latency and minimize redundant network requests
- **Constraint**: Limited memory budget (cache size must be bounded)
- **Success Metrics**:
  - Cache hit rate > 80% for frequently accessed images
  - Image retrieval latency < 5ms for cached images
  - Memory usage stays within configured limits
- **Production Consideration**: System will handle thousands of image requests per second

---

## Requirements

### Core Requirements (Must Have)

1. **LRU Cache Implementation**
   - Implement a cache with configurable maximum size (in bytes or number of items)
   - Evict least recently used items when cache is full
   - O(1) time complexity for get and put operations

2. **Basic Operations**
   - `get(image_id)` - Retrieve an image from cache, return None if not present
   - `put(image_id, image_data)` - Store an image in cache
   - `contains(image_id)` - Check if image exists in cache
   - `remove(image_id)` - Remove specific image from cache
   - `clear()` - Clear all cached images

3. **Cache Statistics**
   - Track cache hits and misses
   - Track total memory usage
   - Calculate hit rate

### Advanced Requirements (Nice to Have)

4. **Size-Based Eviction**
   - Track actual image sizes in bytes
   - Evict based on total memory usage, not just item count

5. **TTL (Time-To-Live) Support**
   - Allow setting expiration time for cached images
   - Automatically evict expired images

6. **Image Loading Integration**
   - Load images from disk if not in cache
   - Support common image formats (PNG, JPEG, etc.)

### Stretch Goals (If Time Permits)

7. **Thread Safety**
   - Make cache operations thread-safe for concurrent access

8. **Cache Warming**
   - Pre-load frequently used images on startup

9. **Compression**
   - Store compressed versions to save memory

---

## Dataset / Test Data

The `generate_image_data.py` script creates test images and metadata:

| File | Description |
|------|-------------|
| `images/` | Directory with sample PNG/JPEG images |
| `image_metadata.csv` | Image IDs, sizes, formats, access patterns |
| `access_log.csv` | Simulated access sequence for testing |

### Image Metadata Schema

| Column | Type | Description |
|--------|------|-------------|
| image_id | string | Unique identifier (e.g., "map_tile_123") |
| filename | string | File path relative to images directory |
| format | string | Image format (png, jpeg, webp) |
| width | int | Image width in pixels |
| height | int | Image height in pixels |
| size_bytes | int | File size in bytes |
| category | string | Type: map_tile, driver_photo, vehicle, banner |

---

## Time Budget (90 minutes total)

| Phase | Time | Focus |
|-------|------|-------|
| Problem Review | 15 min | Read problem, explore data, ask clarifying questions |
| Core Implementation | 45 min | Implement LRU cache with basic operations |
| Advanced Features | 15 min | Add size-based eviction, TTL, or stats |
| Discussion | 15 min | Discuss approach, trade-offs, production considerations |

---

## Evaluation Rubric

| Dimension | Weight | What We Look For |
|-----------|--------|------------------|
| Intuition | 25% | Translating ill-defined problem into coherent solution |
| Clarity | 25% | Clean code, good naming, follows best practices |
| Correctness | 25% | Working solution, handles edge cases, proper LRU behavior |
| Completeness | 15% | Identifies corner cases, discusses limitations |
| Understanding | 10% | Clear explanation of approach and trade-offs |

---

## Technical Hints

### Data Structures to Consider

- **OrderedDict**: Python's `collections.OrderedDict` maintains insertion order and supports `move_to_end()`
- **Doubly Linked List + Hash Map**: Classic LRU implementation with O(1) operations
- **heapq**: For TTL-based expiration (priority queue by expiration time)

### Key Edge Cases

- Cache is full when adding new item
- Image larger than max cache size
- Concurrent access (if implementing thread safety)
- Image already exists (update vs replace)
- Empty cache operations

### Memory Considerations

- Images can be large (100KB - 5MB each)
- Consider storing image bytes vs PIL Image objects
- Track actual memory usage, not just file sizes

---

## Production Discussion Topics

Be prepared to discuss these topics in the final 15 minutes:

1. **Deployment**: How would you deploy this cache in production?
2. **Monitoring**: What metrics would you track? How would you alert on issues?
3. **Scaling**: How would you handle 10x, 100x more traffic?
4. **Distributed Caching**: How would you extend this to multiple servers?
5. **Cache Invalidation**: How would you handle image updates?
6. **Cold Start**: What happens when the cache is empty after restart?
7. **Memory Pressure**: How would you handle system memory pressure?

---

## Getting Started

1. Run the data generator to create test images:
   ```bash
   python generate_image_data.py
   ```

2. Open `starter_image_caching.py` and implement the `ImageCache` class

3. Test your implementation with the provided test harness

4. Be prepared to discuss your approach and potential improvements

---

## Example Usage

```python
# Create cache with 100MB limit
cache = ImageCache(max_size_bytes=100 * 1024 * 1024)

# Store an image
with open("images/map_tile_001.png", "rb") as f:
    image_data = f.read()
cache.put("map_tile_001", image_data)

# Retrieve an image
data = cache.get("map_tile_001")
if data:
    print(f"Cache hit! Image size: {len(data)} bytes")
else:
    print("Cache miss")

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory used: {stats['memory_used_bytes'] / 1024 / 1024:.2f} MB")
```

---

Good luck!
