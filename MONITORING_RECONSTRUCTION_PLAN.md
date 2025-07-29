# Monitoring System Reconstruction Plan

## Current State Analysis

### 1. Existing Components

#### ProgressManager (`src/core/progress_manager.py`)
- **Architecture**: Hierarchical progress tracking (pipeline → phase → step → substep)
- **Storage**: In-memory only with singleton pattern
- **Features**:
  - Node-based hierarchy with parent-child relationships
  - Progress aggregation from children to parents
  - Callback system for events (started, progress, completed)
  - Rate limiting for updates
  - History tracking to file (JSON)
- **Issues**:
  - No database persistence
  - Lost on SSH disconnect/process restart
  - Singleton pattern makes testing harder

#### MonitorManager (`src/pipelines/orchestrator/monitor_manager.py`)
- **Architecture**: Thread-based resource monitoring
- **Features**:
  - Memory and CPU tracking
  - System metrics collection
  - Stage-specific memory contexts
- **Issues**:
  - Not integrated with ProgressManager
  - No persistence
  - Limited usefulness for remote monitoring

#### Current Usage Patterns
- ProgressManager: Used in `scripts/process_manager.py` and tests
- MonitorManager: Used in `PipelineOrchestrator`
- TileProcessor: Has its own `_tile_progress` tracking
- Database: Stores results but not real-time progress

### 2. Key Problems
1. **No unified monitoring** - Multiple separate systems
2. **No persistence** - Everything lost on disconnect
3. **No remote visibility** - Can't monitor SSH/tmux sessions effectively
4. **Inconsistent APIs** - Each component tracks differently

## Reconstruction Plan

### Phase 1: Create Monitoring Abstractions (Week 1)

#### 1.1 Create Base Abstractions
```
src/abstractions/monitoring/
├── __init__.py
├── base.py              # Core abstractions
├── types.py             # Type definitions
└── exceptions.py        # Monitoring exceptions
```

**Key Classes**:
- `MonitoringBackend(ABC)` - Storage backend interface
- `ProgressTracker(ABC)` - Progress tracking interface
- `ResourceMonitor(ABC)` - Resource monitoring interface
- `MonitoringContext` - Shared context for monitoring

#### 1.2 Define Types
- `ProgressNode` - Node in hierarchy
- `ProgressData` - Progress update data
- `ResourceMetrics` - CPU/Memory/Disk metrics
- `MonitoringEvent` - Events (started, progress, completed, etc.)

### Phase 2: Implement Storage Backends (Week 1-2)

#### 2.1 Memory Backend
```
src/base/monitoring/backends/memory_backend.py
```
- Fast, in-memory storage
- Good for development/testing
- Implements existing ProgressManager logic

#### 2.2 Database Backend
```
src/base/monitoring/backends/database_backend.py
```
- PostgreSQL persistence
- Survives disconnections
- Queryable from anywhere

#### 2.3 Hybrid Backend
```
src/base/monitoring/backends/hybrid_backend.py
```
- Memory for fast local access
- Async writes to database
- Best of both worlds

### Phase 3: Database Schema (Week 2)

#### 3.1 Add Monitoring Tables
```sql
-- Progress tracking table
CREATE TABLE pipeline_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id),
    node_id VARCHAR(255) NOT NULL,
    node_level VARCHAR(50) NOT NULL CHECK (node_level IN ('pipeline', 'phase', 'step', 'substep')),
    parent_id VARCHAR(255),
    node_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress_percent FLOAT DEFAULT 0,
    completed_units INTEGER DEFAULT 0,
    total_units INTEGER DEFAULT 100,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, node_id)
);

-- Resource metrics table
CREATE TABLE pipeline_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES experiments(id),
    node_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    memory_mb FLOAT,
    cpu_percent FLOAT,
    disk_usage_mb FLOAT,
    throughput_per_sec FLOAT,
    custom_metrics JSONB DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX idx_progress_experiment_node ON pipeline_progress(experiment_id, node_id);
CREATE INDEX idx_progress_status ON pipeline_progress(status) WHERE status = 'running';
CREATE INDEX idx_metrics_experiment_time ON pipeline_metrics(experiment_id, timestamp DESC);
```

### Phase 4: Unified Implementation (Week 2-3)

#### 4.1 Create Unified Tracker
```
src/base/monitoring/unified_tracker.py
```
- Combines progress and resource tracking
- Supports all backends
- Migration wrapper for old ProgressManager API

#### 4.2 Update Pipeline Integration
```python
# src/pipelines/orchestrator/monitor_manager.py
class UnifiedMonitor:
    """Replaces MonitorManager with unified system."""
    
    def __init__(self, config, db):
        # Backend selection based on config
        backend = self._create_backend(config, db)
        self.tracker = UnifiedTracker(backend)
```

### Phase 5: Monitoring CLI Tools (Week 3)

#### 5.1 Create Monitor Script
```
scripts/monitor.py
```
Commands:
- `monitor status <experiment>` - Current status
- `monitor watch <experiment>` - Live updates
- `monitor history <experiment>` - Historical data
- `monitor list` - All running experiments
- `monitor resources` - System resources

#### 5.2 tmux Integration
Update tmux scripts to use new monitor:
- Dedicated monitoring pane
- Auto-refresh with `watch`
- Database queries for remote monitoring

### Phase 6: Migration Strategy (Week 4)

#### 6.1 Compatibility Layer
- Keep old ProgressManager API working
- Internally redirect to new system
- Deprecation warnings

#### 6.2 Staged Rollout
1. Deploy new system alongside old
2. Update PipelineOrchestrator to use new system
3. Migrate other components gradually
4. Remove old system after validation

## Implementation Priority

### High Priority (Do First)
1. Database schema creation
2. Database backend implementation
3. Basic monitoring CLI
4. Update PipelineOrchestrator

### Medium Priority
1. Hybrid backend
2. Full CLI features
3. tmux integration
4. Migration wrappers

### Low Priority
1. Advanced metrics
2. Web dashboard
3. Alerting system

## Benefits

1. **Unified System** - Single monitoring API for everything
2. **Persistence** - Survives SSH disconnects
3. **Remote Monitoring** - Check progress from anywhere
4. **Clean Architecture** - Follows established patterns
5. **Backward Compatible** - No breaking changes
6. **Testable** - Easy to mock backends
7. **Extensible** - Easy to add new backends/metrics

## Success Criteria

1. Can monitor pipeline progress after SSH disconnect
2. Single source of truth for all monitoring data
3. No performance regression vs current system
4. Clean separation of concerns
5. All existing tests pass
6. New monitoring accurately reflects pipeline state

## Next Steps

1. Review and approve this plan
2. Create database schema migration
3. Implement database backend
4. Create basic CLI tool
5. Test with real pipeline runs