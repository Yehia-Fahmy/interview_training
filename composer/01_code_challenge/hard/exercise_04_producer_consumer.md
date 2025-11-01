# Exercise 4: Concurrent Producer-Consumer Pattern

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Concurrency, thread safety, queue optimization

## Problem

Implement a high-performance producer-consumer system where:

1. **Multiple producers** generate data items
2. **Multiple consumers** process items
3. **Bounded queue** to prevent memory overflow
4. **Thread-safe** operations
5. **Graceful shutdown** handling

**Requirements:**
- Handle 10 producers and 5 consumers
- Queue should be bounded (max 1000 items)
- Ensure all items are processed
- Measure throughput (items/second)

## Solution Template

```python
import threading
import queue
import time
import random

class ProducerConsumerSystem:
    """Thread-safe producer-consumer system"""
    def __init__(self, num_producers=10, num_consumers=5, queue_size=1000):
        self.queue = queue.Queue(maxsize=queue_size)
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.producers = []
        self.consumers = []
        self.shutdown_event = threading.Event()
        self.processed_count = 0
        self.lock = threading.Lock()
    
    def producer(self, producer_id):
        """Producer thread function"""
        items_produced = 0
        while not self.shutdown_event.is_set():
            try:
                # Generate item
                item = f"item_{producer_id}_{items_produced}"
                
                # Add to queue with timeout
                self.queue.put(item, timeout=1)
                items_produced += 1
                
                # Simulate work
                time.sleep(random.uniform(0.001, 0.01))
            except queue.Full:
                continue
        
        print(f"Producer {producer_id} produced {items_produced} items")
    
    def consumer(self, consumer_id):
        """Consumer thread function"""
        items_processed = 0
        while not self.shutdown_event.is_set() or not self.queue.empty():
            try:
                # Get item with timeout
                item = self.queue.get(timeout=1)
                
                # Process item (simulate work)
                self.process_item(item)
                
                self.queue.task_done()
                items_processed += 1
                
                with self.lock:
                    self.processed_count += 1
            except queue.Empty:
                continue
        
        print(f"Consumer {consumer_id} processed {items_processed} items")
    
    def process_item(self, item):
        """Process a single item"""
        # Simulate processing
        time.sleep(random.uniform(0.005, 0.02))
    
    def start(self):
        """Start producers and consumers"""
        # Create and start producer threads
        for i in range(self.num_producers):
            p = threading.Thread(target=self.producer, args=(i,))
            p.start()
            self.producers.append(p)
        
        # Create and start consumer threads
        for i in range(self.num_consumers):
            c = threading.Thread(target=self.consumer, args=(i,))
            c.start()
            self.consumers.append(c)
    
    def stop(self):
        """Gracefully shutdown the system"""
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for producers
        for p in self.producers:
            p.join()
        
        # Wait for queue to empty
        self.queue.join()
        
        # Wait for consumers
        for c in self.consumers:
            c.join()
        
        print(f"Total items processed: {self.processed_count}")

if __name__ == "__main__":
    system = ProducerConsumerSystem(num_producers=10, num_consumers=5)
    
    start_time = time.time()
    system.start()
    
    # Run for 10 seconds
    time.sleep(10)
    
    system.stop()
    elapsed = time.time() - start_time
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {system.processed_count / elapsed:.2f} items/sec")
```

## Key Learning Points

1. **Thread Safety:** Using `queue.Queue` for safe communication
2. **Bounded Queues:** Preventing memory issues with `maxsize`
3. **Graceful Shutdown:** Coordinating thread termination
4. **Synchronization:** Using events and locks appropriately

## Advanced Considerations

- Compare with `multiprocessing.Queue` for CPU-bound tasks
- Implement priority queues for prioritized processing
- Add monitoring/metrics collection
- Handle exceptions gracefully

