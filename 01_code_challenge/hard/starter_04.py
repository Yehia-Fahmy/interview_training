"""
Exercise 4: Concurrent Producer-Consumer Pattern

Implement a high-performance producer-consumer system with:
- Multiple producers generating data items
- Multiple consumers processing items
- Bounded queue to prevent memory overflow
- Thread-safe operations
- Graceful shutdown handling

Requirements:
- Handle 10 producers and 5 consumers
- Queue should be bounded (max 1000 items)
- Ensure all items are processed
- Measure throughput (items/second)
"""

import threading
import queue
import time
import random


class ProducerConsumerSystem:
    """
    Thread-safe producer-consumer system.
    
    TODO: Implement complete system with:
    - Producer threads
    - Consumer threads
    - Bounded queue
    - Graceful shutdown
    - Throughput measurement
    """
    def __init__(self, num_producers=10, num_consumers=5, queue_size=1000):
        # TODO: Initialize queue, threads, synchronization primitives
        # self.queue = queue.Queue(maxsize=queue_size)
        # self.shutdown_event = threading.Event()
        # self.processed_count = 0
        # self.lock = threading.Lock()
        pass
    
    def producer(self, producer_id):
        """
        Producer thread function.
        
        Args:
            producer_id: Unique ID for this producer
        """
        # TODO: Implement producer logic
        # Generate items and add to queue
        # Handle queue.Full exceptions
        # Respect shutdown signal
        items_produced = 0
        
        # Example item generation:
        # item = f"item_{producer_id}_{items_produced}"
        
        print(f"Producer {producer_id} produced {items_produced} items")
    
    def consumer(self, consumer_id):
        """
        Consumer thread function.
        
        Args:
            consumer_id: Unique ID for this consumer
        """
        # TODO: Implement consumer logic
        # Get items from queue
        # Process items using process_item()
        # Handle queue.Empty exceptions
        # Respect shutdown signal and drain queue
        items_processed = 0
        
        print(f"Consumer {consumer_id} processed {items_processed} items")
    
    def process_item(self, item):
        """
        Process a single item.
        
        Args:
            item: Item to process
        """
        # TODO: Implement item processing
        # Simulate work with time.sleep()
        # Adjust sleep duration for realistic testing
        pass
    
    def start(self):
        """Start producers and consumers."""
        # TODO: Create and start producer threads
        # TODO: Create and start consumer threads
        pass
    
    def stop(self):
        """Gracefully shutdown the system."""
        # TODO: Signal shutdown
        # TODO: Wait for producers to finish
        # TODO: Wait for queue to empty
        # TODO: Wait for consumers to finish
        # TODO: Print final statistics
        pass


if __name__ == "__main__":
    system = ProducerConsumerSystem(num_producers=10, num_consumers=5)
    
    start_time = time.time()
    system.start()
    
    # Run for 10 seconds
    time.sleep(10)
    
    system.stop()
    elapsed = time.time() - start_time
    
    print(f"\nTotal time: {elapsed:.2f}s")
    # print(f"Throughput: {system.processed_count / elapsed:.2f} items/sec")

