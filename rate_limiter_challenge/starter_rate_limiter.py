"""
API Rate Limiter Challenge

Implement a rate limiter that controls the number of requests a client can make
within a given time window.

See challenge_rate_limiter.md for full problem description.

Time Budget: 90 minutes total
- Problem Review: 10 min
- Core Implementation: 50 min
- Testing & Edge Cases: 15 min
- Discussion: 15 min
"""

import time
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass


# =============================================================================
# Your Implementation Here
# =============================================================================

class RateLimiter:
    """
    Rate limiter implementation.
    
    Choose and implement one of these algorithms:
    - Fixed Window Counter
    - Sliding Window Log
    - Sliding Window Counter
    - Token Bucket
    - Leaky Bucket
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize rate limiter with given limits."""
        pass
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request from client_id should be allowed."""
        pass
    
    def get_limit_info(self, client_id: str) -> Dict[str, Any]:
        """
        Get rate limit information for a client.
        
        Returns dict with: allowed, remaining, reset_at, retry_after
        """
        pass


# =============================================================================
# Test Harness
# =============================================================================

def test_basic_limiting():
    """Test that requests are limited after max is reached."""
    print("\n" + "=" * 50)
    print("Test 1: Basic Rate Limiting")
    print("=" * 50)
    
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    client = "user_123"
    
    results = []
    for i in range(8):
        allowed = limiter.is_allowed(client)
        results.append(allowed)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  Request {i+1}: {status}")
    
    # First 5 should be allowed, rest blocked
    expected = [True, True, True, True, True, False, False, False]
    if results == expected:
        print("  PASS: Basic limiting works correctly")
    else:
        print(f"  FAIL: Expected {expected}, got {results}")


def test_multiple_clients():
    """Test that each client has independent limits."""
    print("\n" + "=" * 50)
    print("Test 2: Multiple Clients")
    print("=" * 50)
    
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    
    # Client A makes 3 requests
    for i in range(3):
        limiter.is_allowed("client_a")
    
    # Client B should still be allowed
    allowed = limiter.is_allowed("client_b")
    
    if allowed:
        print("  PASS: Clients have independent limits")
    else:
        print("  FAIL: Client B should be allowed")
    
    # Client A should be blocked
    blocked = not limiter.is_allowed("client_a")
    if blocked:
        print("  PASS: Client A correctly blocked")
    else:
        print("  FAIL: Client A should be blocked")


def test_window_reset():
    """Test that limits reset after window expires."""
    print("\n" + "=" * 50)
    print("Test 3: Window Reset (using 2-second window)")
    print("=" * 50)
    
    limiter = RateLimiter(max_requests=3, window_seconds=2)
    client = "user_456"
    
    # Exhaust limit
    for _ in range(3):
        limiter.is_allowed(client)
    
    # Should be blocked
    blocked = not limiter.is_allowed(client)
    print(f"  After 3 requests: {'BLOCKED' if blocked else 'ALLOWED'}")
    
    # Wait for window to reset
    print("  Waiting 2.5 seconds for window reset...")
    time.sleep(2.5)
    
    # Should be allowed again
    allowed = limiter.is_allowed(client)
    print(f"  After reset: {'ALLOWED' if allowed else 'BLOCKED'}")
    
    if blocked and allowed:
        print("  PASS: Window reset works correctly")
    else:
        print("  FAIL: Window reset not working")


def test_limit_info():
    """Test get_limit_info returns correct information."""
    print("\n" + "=" * 50)
    print("Test 4: Limit Info")
    print("=" * 50)
    
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    client = "user_789"
    
    # Make 3 requests
    for _ in range(3):
        limiter.is_allowed(client)
    
    info = limiter.get_limit_info(client)
    
    if info is None:
        print("  SKIP: get_limit_info not implemented")
        return
    
    print(f"  Remaining: {info.get('remaining', 'N/A')}")
    print(f"  Allowed: {info.get('allowed', 'N/A')}")
    print(f"  Retry After: {info.get('retry_after', 'N/A')}s")
    
    if info.get('remaining') == 2 and info.get('allowed') == True:
        print("  PASS: Limit info is accurate")
    else:
        print("  INFO: Check remaining count (expected 2)")


def test_burst_traffic():
    """Test handling of burst traffic."""
    print("\n" + "=" * 50)
    print("Test 5: Burst Traffic (100 rapid requests)")
    print("=" * 50)
    
    limiter = RateLimiter(max_requests=10, window_seconds=60)
    client = "burst_client"
    
    allowed_count = 0
    blocked_count = 0
    
    for _ in range(100):
        if limiter.is_allowed(client):
            allowed_count += 1
        else:
            blocked_count += 1
    
    print(f"  Allowed: {allowed_count}")
    print(f"  Blocked: {blocked_count}")
    
    if allowed_count == 10 and blocked_count == 90:
        print("  PASS: Burst traffic handled correctly")
    else:
        print(f"  INFO: Expected 10 allowed, 90 blocked")


def run_performance_test():
    """Test performance under high load."""
    print("\n" + "=" * 50)
    print("Performance Test: 100K requests")
    print("=" * 50)
    
    limiter = RateLimiter(max_requests=1000, window_seconds=60)
    num_clients = 1000
    requests_per_client = 100
    
    start = time.time()
    
    for i in range(num_clients):
        client = f"client_{i}"
        for _ in range(requests_per_client):
            limiter.is_allowed(client)
    
    elapsed = time.time() - start
    total_requests = num_clients * requests_per_client
    
    print(f"  Total requests: {total_requests:,}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Requests/second: {total_requests / elapsed:,.0f}")
    print(f"  Latency per request: {elapsed / total_requests * 1000:.3f}ms")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Rate Limiter Challenge - Test Suite")
    print("=" * 50)
    
    try:
        test_basic_limiting()
        test_multiple_clients()
        test_window_reset()
        test_limit_info()
        test_burst_traffic()
        run_performance_test()
    except Exception as e:
        print(f"\nError during tests: {e}")
        print("Make sure RateLimiter is fully implemented.")
    
    print("\n" + "=" * 50)
    print("Tests Complete")
    print("=" * 50)
