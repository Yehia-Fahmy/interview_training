"""
Starter Code: Image Caching System

Your task is to implement an LRU (Least Recently Used) image cache that can
store images in memory with configurable size limits and eviction policies.

See challenge_image_caching.md for full problem description.

Time Budget:
- Problem Review: 15 minutes
- Core Implementation: 45 minutes
- Advanced Features: 15 minutes
- Discussion with interviewer: 15 minutes

Tips:
- Start with a simple working implementation
- Use collections.OrderedDict for easy LRU tracking
- Prioritize correctness over optimization
- Think about edge cases as you code
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field

# Optional imports - use if needed
# from PIL import Image
# import threading


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CacheEntry:
    """Represents a cached image entry."""
    image_id: str
    data: bytes
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if this entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return time.time() > self.created_at + self.ttl_seconds


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_used_bytes: int = 0
    item_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# Image Cache Implementation
# =============================================================================

class ImageCache:
    """
    LRU Image Cache with configurable size limits.
    
    Your implementation should support:
    - O(1) get and put operations
    - LRU eviction when cache is full
    - Tracking cache statistics
    
    Advanced features (if time permits):
    - Size-based eviction (bytes, not just count)
    - TTL support for automatic expiration
    - Thread safety
    """
    
    def __init__(
        self,
        max_items: int = 100,
        max_size_bytes: int = None,
        default_ttl_seconds: Optional[float] = None
    ):
        """
        Initialize the image cache.
        
        Args:
            max_items: Maximum number of items in cache (default: 100)
            max_size_bytes: Maximum total size in bytes (optional)
            default_ttl_seconds: Default TTL for cached items (optional)
        """
        self.max_items = max_items
        self.max_size_bytes = max_size_bytes
        self.default_ttl_seconds = default_ttl_seconds
        
        # TODO: Initialize your data structures here
        # Hint: Consider using OrderedDict for LRU tracking
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
    
    def get(self, image_id: str) -> Optional[bytes]:
        """
        Retrieve an image from the cache.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            Image data as bytes if found, None otherwise
            
        Note: This should update the LRU order (most recently used)
        """
        image = None
        if image_id in self._cache:
            if self._cache[image_id].is_expired():
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.item_count -= 1
                self._stats.memory_used_bytes -= self._cache[image_id].size_bytes
                del self._cache[image_id]
                return image
            self._stats.hits += 1
            image = self._cache[image_id].data
            self._cache[image_id].last_accessed = time.time()
            self._cache.move_to_end(image_id)
        else:
            self._stats.misses += 1
        return image
        
    
    def put(
        self,
        image_id: str,
        image_data: bytes,
        ttl_seconds: Optional[float] = None
    ) -> bool:
        """
        Store an image in the cache.
        
        Args:
            image_id: Unique identifier for the image
            image_data: Image data as bytes
            ttl_seconds: Optional TTL override for this item
            
        Returns:
            True if successfully stored, False otherwise
            
        Note: Should evict LRU items if cache is full
        """
        if len(image_data) > self.max_size_bytes: return False

        if image_id in self._cache:
            old_entry = self._cache.pop(image_id)
            self._stats.memory_used_bytes -= old_entry.size_bytes
            self._stats.item_count -= 1
        
        while self._stats.item_count >= self.max_items or self._stats.memory_used_bytes >= self.max_size_bytes:
            popped_image_id, popped_image = self._cache.popitem(last=False)
            self._stats.memory_used_bytes -= popped_image.size_bytes
            self._stats.item_count -= 1
            self._stats.evictions += 1
        
        image_object = CacheEntry(image_id, image_data, len(image_data), time.time(), time.time(), ttl_seconds)
        self._stats.item_count += 1
        self._stats.memory_used_bytes += image_object.size_bytes
        self._cache[image_id] = image_object
        return True
    
    def contains(self, image_id: str) -> bool:
        """
        Check if an image exists in the cache.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            True if image is in cache (and not expired), False otherwise
            
        Note: This should NOT update LRU order
        """
        return image_id in self._cache and not self._cache[image_id].is_expired()
    
    def remove(self, image_id: str) -> bool:
        """
        Remove a specific image from the cache.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            True if image was removed, False if not found
        """
        if not self.contains(image_id):
            return False
        image = self._cache.pop(image_id)
        self._stats.memory_used_bytes -= image.size_bytes
        self._stats.item_count -= 1
        return True
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
    
    def _evict_one(self) -> Optional[str]:
        """
        Evict the least recently used item from the cache.
        
        Returns:
            The image_id of the evicted item, or None if cache is empty
        """
        if self._stats.item_count < 1:
            return None
        id, image = self._cache.popitem(last=False)
        self._stats.memory_used_bytes -= image.size_bytes
        self._stats.item_count -= 1
        self._stats.evictions += 1  # Add this
        return id
    
    def _evict_expired(self) -> int:
        """
        Remove all expired items from the cache.
        
        Returns:
            Number of items evicted
        """
        keys_expired = 0
        for key in self._cache.keys():
            if self._cache[key].is_expired():
                self.remove(key)
                keys_expired += 1
        return keys_expired
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # TODO: Return cache statistics
        # Should include: hits, misses, hit_rate, evictions,
        # memory_used_bytes, item_count
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate,
            "evictions": self._stats.evictions,
            "memory_used_bytes": self._stats.memory_used_bytes,
            "item_count": self._stats.item_count,
        }
    
    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)
    
    def __contains__(self, image_id: str) -> bool:
        """Support 'in' operator."""
        return self.contains(image_id)


# =============================================================================
# Helper Functions
# =============================================================================

def load_image_from_disk(filepath: str) -> Optional[bytes]:
    """
    Load an image from disk as bytes.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Image data as bytes, or None if file not found
    """
    try:
        with open(filepath, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except IOError as e:
        print(f"Error loading image {filepath}: {e}")
        return None


def get_image_size(image_data: bytes) -> Tuple[int, int]:
    """
    Get image dimensions from bytes (requires PIL).
    
    Args:
        image_data: Image data as bytes
        
    Returns:
        Tuple of (width, height)
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_data))
        return img.size
    except ImportError:
        print("PIL not installed, cannot get image dimensions")
        return (0, 0)
    except Exception as e:
        print(f"Error getting image size: {e}")
        return (0, 0)


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


# =============================================================================
# Cached Image Loader (Advanced)
# =============================================================================

class CachedImageLoader:
    """
    Image loader with integrated caching.
    
    This class combines the cache with disk/network loading,
    implementing the full caching pattern.
    """
    
    def __init__(
        self,
        cache: ImageCache,
        image_directory: str = "images"
    ):
        """
        Initialize the cached image loader.
        
        Args:
            cache: ImageCache instance to use
            image_directory: Base directory for image files
        """
        self.cache = cache
        self.image_directory = Path(image_directory)
    
    def get_image(self, image_id: str, filename: Optional[str] = None) -> Optional[bytes]:
        """
        Get an image, loading from disk if not in cache.
        
        Args:
            image_id: Unique identifier for the image
            filename: Optional filename (defaults to image_id)
            
        Returns:
            Image data as bytes, or None if not found
        """
        # TODO: Implement cache-through loading
        # 1. Try to get from cache
        # 2. If cache miss, load from disk
        # 3. If loaded, store in cache
        # 4. Return image data
        pass
    
    def preload_images(self, image_ids: list) -> int:
        """
        Preload multiple images into cache.
        
        Args:
            image_ids: List of image IDs to preload
            
        Returns:
            Number of images successfully loaded
        """
        # TODO: Implement cache warming
        pass


# =============================================================================
# Test Harness
# =============================================================================

def run_basic_tests():
    """Run basic tests to verify implementation."""
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60)
    
    # Test 1: Basic put and get
    print("\nTest 1: Basic put and get")
    cache = ImageCache(max_items=10)
    test_data = b"fake image data 12345"
    cache.put("test_image_1", test_data)
    result = cache.get("test_image_1")
    
    if result == test_data:
        print("  PASS: Basic put/get works")
    else:
        print("  FAIL: Basic put/get failed")
    
    # Test 2: Cache miss
    print("\nTest 2: Cache miss")
    result = cache.get("nonexistent_image")
    if result is None:
        print("  PASS: Cache miss returns None")
    else:
        print("  FAIL: Cache miss should return None")
    
    # Test 3: LRU eviction
    print("\nTest 3: LRU eviction")
    small_cache = ImageCache(max_items=3)
    small_cache.put("img_1", b"data1")
    small_cache.put("img_2", b"data2")
    small_cache.put("img_3", b"data3")
    
    # Access img_1 to make it recently used
    small_cache.get("img_1")
    
    # Add img_4, should evict img_2 (least recently used)
    small_cache.put("img_4", b"data4")
    
    if small_cache.get("img_2") is None and small_cache.get("img_1") is not None:
        print("  PASS: LRU eviction works correctly")
    else:
        print("  FAIL: LRU eviction not working")
    
    # Test 4: Cache statistics
    print("\nTest 4: Cache statistics")
    stats = cache.get_stats()
    print(f"  Hits: {stats.get('hits', 'N/A')}")
    print(f"  Misses: {stats.get('misses', 'N/A')}")
    print(f"  Hit Rate: {stats.get('hit_rate', 'N/A')}")
    print(f"  Items: {stats.get('item_count', 'N/A')}")
    
    # Test 5: Contains check
    print("\nTest 5: Contains check")
    if cache.contains("test_image_1") and not cache.contains("nonexistent"):
        print("  PASS: Contains check works")
    else:
        print("  FAIL: Contains check failed")
    
    # Test 6: Remove
    print("\nTest 6: Remove")
    cache.put("to_remove", b"temporary data")
    removed = cache.remove("to_remove")
    still_exists = cache.contains("to_remove")
    if removed and not still_exists:
        print("  PASS: Remove works correctly")
    else:
        print("  FAIL: Remove failed")
    
    # Test 7: Clear
    print("\nTest 7: Clear")
    cache.clear()
    if len(cache) == 0:
        print("  PASS: Clear works correctly")
    else:
        print("  FAIL: Clear failed")
    
    print("\n" + "=" * 60)
    print("Basic Tests Complete")
    print("=" * 60)


def run_performance_test():
    """Run performance tests with simulated workload."""
    print("\n" + "=" * 60)
    print("Running Performance Test")
    print("=" * 60)
    
    import random
    
    # Create cache with 100 items
    cache = ImageCache(max_items=100)
    
    # Generate test data
    num_images = 200
    test_images = {
        f"img_{i}": os.urandom(1024 * random.randint(10, 100))  # 10KB - 100KB
        for i in range(num_images)
    }
    
    # Simulate access pattern (Zipf-like distribution)
    num_accesses = 10000
    popular_images = list(test_images.keys())[:20]  # First 20 are "popular"
    
    start_time = time.time()
    
    for _ in range(num_accesses):
        # 80% of accesses go to popular images
        if random.random() < 0.8:
            image_id = random.choice(popular_images)
        else:
            image_id = random.choice(list(test_images.keys()))
        
        # Try to get from cache
        data = cache.get(image_id)
        
        # If miss, load and store
        if data is None:
            cache.put(image_id, test_images[image_id])
    
    elapsed = time.time() - start_time
    stats = cache.get_stats()
    
    print(f"\nPerformance Results:")
    print(f"  Total accesses: {num_accesses}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Accesses/second: {num_accesses / elapsed:.0f}")
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats.get('hits', 'N/A')}")
    print(f"  Misses: {stats.get('misses', 'N/A')}")
    print(f"  Hit Rate: {stats.get('hit_rate', 0) * 100:.1f}%")
    print(f"  Evictions: {stats.get('evictions', 'N/A')}")
    print(f"  Memory Used: {format_bytes(stats.get('memory_used_bytes', 0))}")
    
    print("\n" + "=" * 60)


def run_real_image_test():
    """Run test with real images from generated data."""
    print("\n" + "=" * 60)
    print("Running Real Image Test")
    print("=" * 60)
    
    images_dir = Path("images")
    if not images_dir.exists():
        print("  No images directory found. Run generate_image_data.py first.")
        print("  Skipping real image test.")
        return
    
    # Find all images
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    
    if not image_files:
        print("  No images found in images directory.")
        return
    
    print(f"  Found {len(image_files)} images")
    
    # Create cache with 50MB limit
    cache = ImageCache(max_size_bytes=50 * 1024 * 1024)
    
    # Load all images
    total_loaded = 0
    for img_path in image_files:
        image_data = load_image_from_disk(str(img_path))
        if image_data:
            cache.put(img_path.stem, image_data)
            total_loaded += 1
    
    print(f"  Loaded {total_loaded} images into cache")
    
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Items: {stats.get('item_count', 'N/A')}")
    print(f"  Memory Used: {format_bytes(stats.get('memory_used_bytes', 0))}")
    print(f"  Evictions: {stats.get('evictions', 'N/A')}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Image Caching System - Test Suite")
    print("=" * 60)
    
    # Run tests
    run_basic_tests()
    run_performance_test()
    run_real_image_test()
    
    print("\nDone! Review the test output and implement missing functionality.")
