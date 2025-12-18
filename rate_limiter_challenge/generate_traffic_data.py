"""
Traffic Data Generator for Rate Limiter Challenge

Generates realistic API traffic patterns for testing rate limiters.
"""

import csv
import random
import time
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class TrafficEvent:
    """A single API request event."""
    timestamp: float
    client_id: str
    endpoint: str
    client_tier: str  # free, premium, enterprise


def generate_client_id(tier: str, index: int) -> str:
    """Generate a client ID."""
    prefixes = {"free": "free", "premium": "prem", "enterprise": "ent"}
    return f"{prefixes[tier]}_{index:04d}"


def generate_traffic(
    duration_seconds: int = 300,
    num_clients: int = 100,
    base_rate: float = 10.0,  # requests per second total
) -> List[TrafficEvent]:
    """
    Generate realistic API traffic.
    
    Traffic patterns:
    - 70% free tier (lower limits)
    - 25% premium tier (medium limits)
    - 5% enterprise tier (high limits)
    
    Includes:
    - Normal steady traffic
    - Burst patterns (some clients spike)
    - Abusive clients (try to exceed limits)
    """
    events = []
    
    # Client distribution
    tier_distribution = {
        "free": int(num_clients * 0.70),
        "premium": int(num_clients * 0.25),
        "enterprise": int(num_clients * 0.05) or 1,
    }
    
    # Generate clients
    clients = []
    for tier, count in tier_distribution.items():
        for i in range(count):
            clients.append({
                "id": generate_client_id(tier, i),
                "tier": tier,
                "behavior": random.choice(["normal", "normal", "normal", "bursty", "abusive"]),
            })
    
    # Endpoints
    endpoints = [
        "/api/v1/rides/request",
        "/api/v1/rides/estimate",
        "/api/v1/user/profile",
        "/api/v1/driver/location",
        "/api/v1/payment/methods",
    ]
    
    # Generate events
    current_time = 0.0
    while current_time < duration_seconds:
        # Pick a client based on activity level
        client = random.choice(clients)
        
        # Determine inter-arrival time based on behavior
        if client["behavior"] == "normal":
            interval = random.expovariate(base_rate / num_clients)
        elif client["behavior"] == "bursty":
            # Sometimes burst, sometimes quiet
            if random.random() < 0.1:  # 10% chance of burst
                interval = random.uniform(0.001, 0.01)  # Very fast
            else:
                interval = random.expovariate(base_rate / num_clients)
        else:  # abusive
            interval = random.uniform(0.001, 0.05)  # Constant high rate
        
        current_time += interval
        
        if current_time < duration_seconds:
            events.append(TrafficEvent(
                timestamp=current_time,
                client_id=client["id"],
                endpoint=random.choice(endpoints),
                client_tier=client["tier"],
            ))
    
    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)
    
    return events


def save_traffic_csv(events: List[TrafficEvent], filepath: str) -> None:
    """Save traffic events to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "client_id", "endpoint", "client_tier"])
        for event in events:
            writer.writerow([
                f"{event.timestamp:.6f}",
                event.client_id,
                event.endpoint,
                event.client_tier,
            ])


def analyze_traffic(events: List[TrafficEvent]) -> None:
    """Print traffic statistics."""
    if not events:
        print("No events generated.")
        return
    
    duration = events[-1].timestamp - events[0].timestamp
    
    # Count by client
    client_counts = {}
    tier_counts = {"free": 0, "premium": 0, "enterprise": 0}
    
    for event in events:
        client_counts[event.client_id] = client_counts.get(event.client_id, 0) + 1
        tier_counts[event.client_tier] += 1
    
    print("\n" + "=" * 50)
    print("Traffic Analysis")
    print("=" * 50)
    print(f"Total events: {len(events):,}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Overall rate: {len(events) / duration:.1f} req/sec")
    print(f"Unique clients: {len(client_counts)}")
    
    print("\nRequests by tier:")
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count:,} ({count/len(events)*100:.1f}%)")
    
    # Top clients
    top_clients = sorted(client_counts.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 clients by request count:")
    for client_id, count in top_clients:
        rate = count / duration
        print(f"  {client_id}: {count} requests ({rate:.2f}/sec)")
    
    # Check for potential abusers (> 1 req/sec average)
    abusers = [(c, n) for c, n in client_counts.items() if n / duration > 1.0]
    if abusers:
        print(f"\nPotential rate limit violators: {len(abusers)}")


def main():
    print("=" * 50)
    print("Traffic Data Generator")
    print("=" * 50)
    
    # Generate traffic
    print("\nGenerating 5 minutes of API traffic...")
    events = generate_traffic(
        duration_seconds=300,
        num_clients=100,
        base_rate=50.0,
    )
    
    # Analyze
    analyze_traffic(events)
    
    # Save
    output_file = "traffic_data.csv"
    save_traffic_csv(events, output_file)
    print(f"\nSaved to: {output_file}")
    
    # Create rate limit config suggestion
    print("\n" + "=" * 50)
    print("Suggested Rate Limits")
    print("=" * 50)
    print("  free:       10 requests / 60 seconds")
    print("  premium:    100 requests / 60 seconds")
    print("  enterprise: 1000 requests / 60 seconds")


if __name__ == "__main__":
    main()
