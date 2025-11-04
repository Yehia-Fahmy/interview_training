"""
Solution for Exercise 2: Threading vs Multiprocessing

This file contains the reference solution.
"""

import time
import math
from threading import Thread
from multiprocessing import Pool


def cpu_intensive_task(n):
    """A CPU-intensive operation"""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def sequential_process(data):
    """Sequential processing"""
    results = []
    for item in data:
        results.append(cpu_intensive_task(item))
    return results


def threaded_process(data, num_threads=4):
    """
    Threading approach - splits data into chunks and processes in parallel threads.
    Note: Due to Python's GIL, this may not provide speedup for CPU-bound tasks.
    """
    def process_chunk(chunk):
        return [cpu_intensive_task(item) for item in chunk]
    
    chunk_size = len(data) // num_threads
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    threads = []
    results = [None] * len(chunks)
    
    def worker(chunk_idx, chunk):
        results[chunk_idx] = process_chunk(chunk)
    
    for i, chunk in enumerate(chunks):
        thread = Thread(target=worker, args=(i, chunk))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Flatten results
    return [item for sublist in results for item in sublist]


def multiprocessed_process(data, num_processes=4):
    """
    Multiprocessing approach - uses process pool for true parallelization.
    This bypasses the GIL and should show significant speedup for CPU-bound tasks.
    """
    with Pool(processes=num_processes) as pool:
        results = pool.map(cpu_intensive_task, data)
    return results

