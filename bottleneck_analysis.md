# Multiprocessing Implementation Guide for Resampling Optimization

## Current System Architecture Understanding

### Resampling Pipeline Structure
```
ResampleStage
  └── ResamplingProcessor (src/processors/data_preparation/resampling_processor.py)
      └── process_chunked_dataset()
          └── for each chunk:
              ├── Load chunk: src.isel(spatial_chunk).compute()  [0.3-0.5s I/O]
              └── NumpyResampler.resample_windowed()            [0.1-0.2s CPU]
                  └── BlockSumAggregationStrategy.resample_direct()
```

### Current Performance Profile
- **Sequential processing**: 1.5s per chunk (800 chunks × 2 datasets = 40 minutes)
- **CPU utilization**: 1.5/256 cores (0.6%)
- **Memory**: 1.6GB (excellent due to chunking)
- **Bottleneck breakdown**:
  - 30% I/O (chunk loading from rioxarray)
  - 15% CPU (block aggregation computation)
  - 55% Overhead (object creation, logging, coordination)

## Proposed Multiprocessing Implementation

### Architecture Design Pattern: Producer-Consumer Pipeline

```python
# Three-stage pipeline with bounded queues
[Chunk Producer] → Queue(maxsize=100) → [Resampling Workers] → Queue(maxsize=50) → [Result Collector]
     (1 thread)                            (50 processes)                              (1 thread)
```

### Implementation Details

#### 1. Modify `ResamplingProcessor.process_chunked_dataset()`

```python
def process_chunked_dataset(self, src, dataset_info, output_table):
    """Enhanced with multiprocessing support."""
    
    # Setup queues
    chunk_queue = Queue(maxsize=100)  # Prevents memory overflow
    result_queue = Queue(maxsize=50)
    
    # Start producer thread
    producer = Thread(
        target=self._chunk_producer,
        args=(src, dataset_info, chunk_queue)
    )
    producer.start()
    
    # Start worker pool
    with ProcessPoolExecutor(max_workers=50) as executor:
        # Submit resampling jobs
        futures = []
        for _ in range(800):  # Total chunks
            chunk_data = chunk_queue.get()
            if chunk_data is None:  # Sentinel
                break
            
            future = executor.submit(
                self._resample_chunk_worker,
                chunk_data,
                self.config
            )
            futures.append(future)
        
        # Collect results in order
        for i, future in enumerate(futures):
            result = future.result()
            result_queue.put((i, result))
            
            # Update progress
            if i % 10 == 0:
                self._update_progress(i, len(futures))
```

#### 2. Create Worker Functions

```python
@staticmethod
def _resample_chunk_worker(chunk_data, config):
    """Stateless worker function for multiprocessing."""
    # Runs in separate process - no DB access, no shared state
    
    # Unpack chunk data
    chunk_array = chunk_data['array']
    chunk_bounds = chunk_data['bounds']
    target_shape = chunk_data['target_shape']
    
    # Create resampler (lightweight)
    from src.domain.resampling.strategies.block_sum_aggregation import BlockSumAggregationStrategy
    strategy = BlockSumAggregationStrategy()
    
    # Perform resampling
    result = strategy.resample_direct(
        chunk_array,
        chunk_bounds,
        target_shape,
        chunk_data['target_bounds'],
        config,
        progress_callback=None  # No progress in worker
    )
    
    return {
        'data': result,
        'chunk_idx': chunk_data['chunk_idx'],
        'bounds': chunk_data['target_bounds']
    }
```

#### 3. Optimize Chunk Loading with Prefetching

```python
def _chunk_producer(self, src, dataset_info, queue):
    """Prefetch chunks to hide I/O latency."""
    with ThreadPoolExecutor(max_workers=4) as io_executor:
        # Prefetch next 4 chunks while processing current ones
        prefetch_futures = []
        
        for chunk_info in self._generate_chunks(src.shape):
            # Start prefetching
            if len(prefetch_futures) < 4:
                future = io_executor.submit(
                    self._load_chunk_data,
                    src, chunk_info
                )
                prefetch_futures.append(future)
            
            # Get oldest prefetched chunk
            if prefetch_futures:
                chunk_data = prefetch_futures.pop(0).result()
                queue.put(chunk_data)
            
            # Keep prefetch pipeline full
            if chunk_info['index'] < total_chunks - 4:
                future = io_executor.submit(
                    self._load_chunk_data,
                    src, self._generate_chunks(src.shape)[chunk_info['index'] + 4]
                )
                prefetch_futures.append(future)
        
        # Sentinel to signal completion
        queue.put(None)
```

### Configuration and Resource Management

#### 1. Add Configuration Options
```yaml
# config.yml
resampling:
  multiprocessing:
    enabled: true
    max_workers: 50
    chunk_prefetch: 4
    memory_limit_per_worker: 512  # MB
    queue_sizes:
      chunk_queue: 100
      result_queue: 50
```

#### 2. Resource Monitoring
```python
class ProcessPoolMonitor:
    """Monitor and manage process pool resources."""
    
    def __init__(self, max_workers=50):
        self.max_workers = min(max_workers, cpu_count() - 1)
        self.active_workers = 0
        self.memory_per_worker = self._calculate_memory_limit()
    
    def _calculate_memory_limit(self):
        # Leave 20% system memory free
        available_memory = psutil.virtual_memory().available * 0.8
        return int(available_memory / self.max_workers / 1024 / 1024)  # MB
```

### Integration Points

#### 1. Minimal Changes to Existing Code
- Keep `NumpyResampler` and `BlockSumAggregationStrategy` unchanged
- Only modify `ResamplingProcessor.process_chunked_dataset()`
- Add new worker functions as static methods
- Maintain backward compatibility with `multiprocessing.enabled` flag

#### 2. Error Handling
```python
def _handle_worker_failure(self, future, chunk_idx):
    """Fallback to sequential processing on worker failure."""
    try:
        return future.result(timeout=30)
    except Exception as e:
        logger.warning(f"Worker failed for chunk {chunk_idx}: {e}")
        # Fallback to sequential processing
        return self._resample_chunk_sequential(chunk_idx)
```

### Expected Performance Improvements

| Metric | Current | With Multiprocessing | Improvement |
|--------|---------|---------------------|-------------|
| Chunk processing time | 1.5s | 0.03s (amortized) | 50x |
| Total resampling time | 40 min | ~1.5 min | 27x |
| CPU utilization | 0.6% | 20% | 33x |
| Memory usage | 1.6GB | ~25GB (50 × 0.5GB) | Acceptable |

### Implementation Priority

1. **Phase 1**: Basic multiprocessing for chunk computation
   - Implement `_resample_chunk_worker()`
   - Add ProcessPoolExecutor to `process_chunked_dataset()`
   - Test with small dataset

2. **Phase 2**: Add I/O prefetching
   - Implement `_chunk_producer()` with prefetch
   - Add queue-based coordination
   - Monitor memory usage

3. **Phase 3**: Production hardening
   - Add resource monitoring
   - Implement fallback mechanisms
   - Add configuration options

This approach maintains the current architecture's elegance while providing massive speedup through parallelization.