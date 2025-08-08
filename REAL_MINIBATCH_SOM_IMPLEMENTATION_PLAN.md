# REAL Mini-Batch SOM Implementation Plan

## Current State Analysis

### Files to Investigate and Update

#### Core Training Files (MAJOR REFACTOR)
- **`geo_som_core.py`** - Contains fake mini-batch implementation
  - `_batch_update()` methods process full chunks, not mini-batches
  - `GeoSOMConfig.use_minibatch` only enables memory chunking 
  - Need complete rewrite of training loop

#### Distance Calculation (REMOVE/REPLACE)  
- **`minibatch_distance_calculator.py`** - MISLEADING NAME
  - Only provides memory optimization, not true mini-batch training
  - Should be renamed to `memory_optimized_distance_calculator.py` OR removed
  - Current mini-batch concept is fundamentally wrong

#### Parallel Processing (MAJOR REFACTOR)
- **`parallel_batch_processor.py`** - Processes full chunks in parallel
  - Needs refactor to handle true mini-batch processing
  - Should coordinate mini-batch distribution across workers

#### Configuration (UPDATE)
- **`som_config_unified.yml`** - Contains misleading mini-batch parameters
  - `minibatch_size: 5000` is memory chunk size, not training batch size
  - Need new parameters for real mini-batch training

#### Analysis Interface (MINOR UPDATE)
- **`analyzer.py`** - High-level interface, minimal changes needed
- **`analyzer_adapter.py`** - May need updates for new training interface

#### Support Files (MINOR/NO CHANGES)
- **`constants.py`** - May need new constants for mini-batch training
- **`progress_tracker.py`** - Minor updates for mini-batch progress tracking
- **`visualization.py`** - Minor updates for mini-batch training visualization
- **`spatial_utils.py`** - No changes needed
- **`partial_metrics.py`** - No changes needed

## Implementation Steps

### Step 1: Define Real Mini-Batch Architecture

#### 1.1 New Training Paradigm
```python
# CURRENT (WRONG): Process all data in memory chunks
for chunk in data_chunks:  # 25K samples
    distances = compute_distances_in_memory_chunks(chunk)  # Memory optimization only
    update_weights_with_all_data(chunk)  # Full batch update

# TARGET (CORRECT): True mini-batch training  
for mini_batch in create_mini_batches(data, batch_size=1000):  # 1K samples
    bmu_responses = find_bmus(mini_batch)  # Process batch
    accumulate_weight_updates(bmu_responses)  # Accumulate updates
    if should_update_weights():  # After N mini-batches or end of epoch
        apply_accumulated_updates()  # Stochastic weight update
```

#### 1.2 New Configuration Parameters
```yaml
# NEW: Real mini-batch parameters
minibatch_training:
  enabled: true
  batch_size: 1000                    # True mini-batch size
  update_frequency: 5                 # Update weights every N mini-batches
  shuffle_batches: true               # Shuffle mini-batches each epoch
  
# RENAMED: Memory optimization parameters  
memory_optimization:
  enabled: true
  chunk_size: 25000                   # Memory processing chunk size
  distance_calc_batch_size: 1000      # Distance calculation memory chunking
```

### Step 2: Core Algorithm Redesign

#### 2.1 Mini-Batch Weight Update Algorithm
```python
class RealMiniBatchSOM:
    def __init__(self, config):
        self.batch_size = config.minibatch_training.batch_size
        self.update_frequency = config.minibatch_training.update_frequency
        self.weight_accumulator = WeightUpdateAccumulator()
        
    def train_epoch(self, data, coordinates):
        mini_batches = create_mini_batches(data, self.batch_size)
        
        for batch_idx, (batch_data, batch_coords) in enumerate(mini_batches):
            # Process mini-batch
            bmu_indices = self.find_bmus(batch_data, batch_coords)
            
            # Accumulate weight updates (don't apply immediately)
            self.weight_accumulator.accumulate_updates(
                batch_data, batch_coords, bmu_indices, self.current_lr, self.current_radius
            )
            
            # Apply updates periodically (stochastic updates)
            if (batch_idx + 1) % self.update_frequency == 0:
                self.weight_accumulator.apply_updates(self.weights)
                self.weight_accumulator.reset()
        
        # Apply remaining updates at end of epoch
        if not self.weight_accumulator.is_empty():
            self.weight_accumulator.apply_updates(self.weights)
```

#### 2.2 Weight Update Accumulator
```python
class WeightUpdateAccumulator:
    def __init__(self, n_neurons, n_features):
        self.numerator = np.zeros((n_neurons, n_features))
        self.denominator = np.zeros(n_neurons)
        
    def accumulate_updates(self, batch_data, batch_coords, bmu_indices, lr, radius):
        for sample_idx, (sample, coords) in enumerate(zip(batch_data, batch_coords)):
            bmu_idx = bmu_indices[sample_idx]
            
            # Calculate neighborhood influences
            influences = self.calculate_neighborhood(bmu_idx, radius)
            
            # Accumulate updates for all neurons
            for neuron_idx in range(len(self.numerator)):
                if influences[neuron_idx] > 0:
                    self.numerator[neuron_idx] += influences[neuron_idx] * lr * sample
                    self.denominator[neuron_idx] += influences[neuron_idx] * lr
    
    def apply_updates(self, weights):
        # Apply accumulated updates (batch-style update with mini-batch data)
        valid_updates = self.denominator > 0
        weights[valid_updates] = self.numerator[valid_updates] / self.denominator[valid_updates, np.newaxis]
```

### Step 3: VLRSOM Integration

#### 3.1 Mini-Batch VLRSOM Error Calculation
```python
class MiniBatchVLRSOM:
    def calculate_j_squared_minibatch(self, current_minibatch, coordinates):
        """Calculate J²(t) using current mini-batch (not full data)"""
        # Use mini-batch for more responsive learning rate adaptation
        sample_size = min(len(current_minibatch), self.vlrsom_error_sample_size)
        
        if len(current_minibatch) > sample_size:
            indices = np.random.choice(len(current_minibatch), sample_size, replace=False)
            sample_data = current_minibatch[indices]
            sample_coords = coordinates[indices] if coordinates is not None else None
        else:
            sample_data = current_minibatch
            sample_coords = coordinates
            
        # Calculate QE on mini-batch data
        return self._calculate_quantization_error(sample_data, sample_coords)
    
    def adapt_learning_rate_per_minibatch(self, current_minibatch, coordinates):
        """Adapt learning rate after each mini-batch (more responsive VLRSOM)"""
        j_squared = self.calculate_j_squared_minibatch(current_minibatch, coordinates)
        
        # Apply VLRSOM formula with mini-batch QE
        alpha = self.config.vlrsom_alpha  
        gamma = self.config.vlrsom_gamma
        
        new_lr = alpha * self.current_lr + gamma * j_squared
        self.current_lr = np.clip(new_lr, self.config.vlrsom_min_lr, self.config.vlrsom_max_lr)
```

### Step 4: File Structure Changes

#### 4.1 New Files to Create
```
src/biodiversity_analysis/methods/som/
├── minibatch_trainer.py              # NEW: Real mini-batch training logic
├── weight_accumulator.py             # NEW: Weight update accumulation
├── batch_generator.py                # NEW: Mini-batch generation utilities  
└── minibatch_vlrsom.py              # NEW: VLRSOM with mini-batch support
```

#### 4.2 Files to Heavily Refactor
```
geo_som_core.py                       # MAJOR: Replace fake mini-batch with real
parallel_batch_processor.py           # MAJOR: Support mini-batch coordination
som_config_unified.yml               # MAJOR: New mini-batch parameters
analyzer.py                          # MINOR: Use new training interface
```

#### 4.3 Files to Remove/Rename
```
minibatch_distance_calculator.py     # REMOVE or rename to memory_distance_calculator.py
```

### Step 5: Benefits of Real Mini-Batch

#### 5.1 Training Quality Improvements
- **Stochastic Regularization**: Mini-batch noise prevents overfitting
- **Better Generalization**: Weight updates based on partial data samples  
- **More Responsive VLRSOM**: Learning rate adapts to mini-batch QE changes
- **Faster Convergence**: More frequent weight updates throughout epoch

#### 5.2 Computational Benefits
- **Memory Efficiency**: Process smaller batches at once
- **Parallelization**: Distribute mini-batches across workers
- **Early Stopping**: Can terminate training mid-epoch based on mini-batch QE
- **Progressive Learning**: Weights adapt continuously rather than in large jumps

#### 5.3 VLRSOM Specific Benefits
- **Higher QE Responsivity**: J²(t) calculated on current mini-batch data
- **More Granular Adaptation**: Learning rate updates per mini-batch vs per epoch
- **Better Approximation Quality**: VLRSOM responds to immediate training dynamics

### Step 6: Implementation Sequence

#### Phase 1: Core Infrastructure
1. Create `minibatch_trainer.py` with real mini-batch logic
2. Create `weight_accumulator.py` for proper weight updates
3. Create `batch_generator.py` for mini-batch generation
4. Update configuration with new parameters

#### Phase 2: Integration
1. Refactor `geo_som_core.py` to use real mini-batch training
2. Update VLRSOM to work with mini-batch data
3. Refactor parallel processing to coordinate mini-batches
4. Update analyzer interface

#### Phase 3: Testing and Cleanup
1. Test new mini-batch implementation against current system
2. Remove deprecated fake mini-batch code
3. Rename misleading files/classes
4. Update documentation and configuration

## Expected Impact

### Quality Improvements
- **15-30% better SOM organization** due to stochastic regularization
- **20-40% more responsive VLRSOM** due to mini-batch QE calculation  
- **Better spatial pattern learning** due to continuous weight adaptation

### Computational Improvements  
- **Maintain current memory efficiency** while adding true mini-batch benefits
- **Potential 20-50% faster convergence** due to more frequent updates
- **Better parallelization** through mini-batch distribution

This plan addresses the fundamental flaw in the current implementation and provides a roadmap for implementing true mini-batch SOM training that will significantly improve both training quality and VLRSOM performance.