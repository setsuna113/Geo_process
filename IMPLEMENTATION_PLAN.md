Geo Project - Comprehensive Fix Implementation Plan
Root Issues Identified
1. ✅ Critical 7-Hour Hang Issue (RESOLVED)
Location: src/raster_data/loaders/metadata_extractor.py:20 - gdal.Open() call
Cause: GDAL synchronously loading massive raster files (GB+ size) with no timeout/progress
Impact: Process hangs before any database records are created
Evidence: Database shows 1 raster_source with pending status, no cache/grid/feature data
✅ FIXED: Lightweight metadata extraction with GDAL timeout protection implemented
2. ⚠️ Memory Pressure During Data Resampling (NEW CRITICAL ISSUE)
Location: src/processors/data_preparation/resampling_processor.py - _load_raster_data_with_timeout() and resampling engines
Cause: Loading entire large raster files (100+ MB) into memory without chunking
Impact: Pipeline hangs with critical memory pressure, no progress updates
Evidence: Pipeline starts successfully, registers metadata, but hangs during actual data processing
Symptoms: 100% CPU usage, continuous memory pressure warnings, process becomes unresponsive
3. Missing Progress Monitoring System
Current: Only high-level workflow progress in resampling_workflow.py
Missing: File I/O, tile-level, engine-level progress tracking
Gap: Progress reports exist but never update during actual processing
Impact: Cannot monitor resampling progress or identify bottlenecks
4. Incomplete Checkpointing System
Exists: BaseTileProcessor has tile-level checkpointing
Missing: No checkpointing in main pipeline stages (raster loading, resampling, merging)
Issue: Database experiments track only high-level state, not processing steps
5. Poor Process Management
Current: Basic bash → Python → tmux chain
Problem: No pause/resume, no graceful shutdown, no signal handling
Missing: Process lifecycle management, state persistence
Implementation Plan - Systematic Step-by-Step Approach
Phase 1: Fix Critical Hang Issue (Priority: URGENT) - ✅ COMPLETED
Step 1.1: Create Lightweight Raster Registration System - ✅ COMPLETED
 File: src/raster_data/loaders/lightweight_metadata.py
 Create LightweightMetadataExtractor class
 Extract only essential metadata (bounds, resolution, CRS) without full file scan
 Use GDAL with read timeouts and minimal data reading
 Add progress callbacks for file operations
 Store basic metadata first, defer heavy extraction
Step 1.2: Update RasterCatalog for Lazy Loading - ✅ COMPLETED
 File: src/raster_data/catalog.py
 Add add_raster_lightweight() method using lightweight extractor
 Update scan_directory() to use lightweight registration by default
 Add --full-metadata option for complete metadata extraction
 Implement deferred metadata loading with extract_full_metadata_async()
Step 1.3: Update ResamplingProcessor - ✅ COMPLETED
 File: src/processors/data_preparation/resampling_processor.py
 Replace heavy catalog registration with lightweight version
 Add file existence check before processing
 Implement timeout mechanisms for file operations
 Add progress callbacks for each dataset processing step
Step 1.4: Database Configuration Fixes - ✅ COMPLETED
 Files: src/config/config.py, src/config/defaults.py, src/database/connection.py
 Fix test mode detection to honor FORCE_TEST_MODE=true
 Update database connection to use Unix socket (/var/run/postgresql)
 Proper handling of DB_NAME environment variable in test mode
 Schema compatibility fixes for raster_sources and processing_queue tables
Phase 1.5: Fix Memory Pressure During Resampling (Priority: URGENT) - 🟡 IN PROGRESS
Step 1.5.1: Implement Memory-Aware Pipeline Infrastructure - ✅ COMPLETED
 File: src/config/processing_config.py (NEW)

 Create ProcessingConfig dataclass for memory-aware processing
 Add ChunkInfo helper class for chunk tracking
 Include memory adjustment methods
 Support serialization for checkpointing
 File: src/pipelines/stages/base_stage.py

 Add ProcessingConfig support to base stage
 Implement pause/resume/cancel functionality
 Add memory-aware processing hooks
 Support checkpoint data methods
 File: src/pipelines/stages/resample_stage.py

 Implement memory-aware resampling stage
 Add chunked processing configuration
 Sort datasets by size (smallest first)
 Memory pressure handling with cleanup triggers
 Progress callbacks for dataset processing
 File: src/pipelines/stages/merge_stage.py

 Add chunked merging for large datasets
 Memory-efficient xarray operations
 Temporary file handling for chunk processing
 Lazy loading for merged results
 File: src/pipelines/orchestrator.py

 Enhanced memory pressure handling
 Automatic retry with chunking on memory errors
 Memory monitoring context during execution
 Stage progress callback integration
Step 1.5.2: Implement Chunked Raster Data Loading - ⏳ PENDING
 File: src/processors/data_preparation/resampling_processor.py
 Replace _load_raster_data_with_timeout() with chunked loading approach
 Implement tile-based or windowed reading for large rasters
 Add memory monitoring and automatic chunk size adjustment
 Use rioxarray with explicit chunking parameters
 Add progress callbacks for chunk loading
Step 1.5.3: Optimize Resampling Engines for Memory Efficiency - ⏳ PENDING
 Files: src/resampling/engines/numpy_resampler.py, src/resampling/engines/gdal_resampler.py
 Implement streaming/chunked resampling instead of full-array processing
 Add memory-aware processing with configurable chunk sizes
 Implement intermediate result caching to disk for large datasets
 Add garbage collection triggers between chunks
 Support resume from partial resampling operations
Step 1.5.4: Add Memory Management Infrastructure - ⏳ PENDING
 File: src/base/memory_manager.py
 Create MemoryManager class for monitoring and controlling memory usage
 Implement automatic garbage collection when memory pressure is detected
 Add memory profiling and reporting capabilities
 Support memory-aware processing with dynamic chunk sizing
 Integration with existing memory_tracker.py
Phase 2: Core Progress Monitoring Infrastructure
Step 2.1: Create Progress Monitoring Hub
 File: src/core/progress_manager.py
 Create ProgressManager singleton class
 Implement hierarchical progress tracking (pipeline → phase → step → substep)
 Add progress aggregation and reporting methods
 Support multiple progress callback types (console, database, file)
 Thread-safe progress updates
Step 2.2: Create Progress Event System
 File: src/core/progress_events.py
 Define ProgressEvent data classes for different event types
 Create event types: FileIOProgress, ProcessingProgress, MemoryProgress
 Implement event bus for progress communication
 Add progress event validation and filtering
Step 2.3: Update Configuration for Progress Settings
 File: src/config/defaults.py
 Add progress monitoring configuration section
 Define checkpoint intervals, progress reporting frequencies
 Add timeout settings for various operations
 Process management configuration (signals, graceful shutdown)
Phase 3: Robust Checkpointing and Resume System
Step 3.1: Create Central Checkpoint Manager
 File: src/core/checkpoint_manager.py
 Create CheckpointManager class for unified checkpointing
 Implement checkpoint serialization (JSON + binary data)
 Add checkpoint validation and corruption detection
 Support hierarchical checkpoints (pipeline → phase → step)
 Automatic checkpoint cleanup and retention policies
Step 3.2: Extend Database Schema for Checkpoints
 File: src/database/schema.py
 Add pipeline_checkpoints table
 Add processing_steps table for fine-grained progress
 Add file_processing_status table for individual file progress
 Update existing tables to support resume operations
 Add checkpoint metadata and validation columns
Step 3.3: Update Database Connection for Checkpoint Operations
 File: src/database/connection.py
 Add checkpoint-specific database methods
 Implement checkpoint data compression
 Add checkpoint recovery methods
 Support transactional checkpoint operations
Phase 4: Process-Aware Base Classes
Step 4.1: Create Process-Aware Base Processor
 File: src/base/process_aware_processor.py
 Extend BaseProcessor with progress and checkpoint support
 Add signal handling (SIGTERM, SIGINT, SIGUSR1 for pause)
 Implement graceful shutdown with state saving
 Add automatic progress reporting
 Support resume-from-checkpoint functionality
Step 4.2: Update Existing Base Classes
 File: src/base/processor.py
 Add progress callback hooks to base methods
 Implement checkpoint-aware batch processing
 Add timeout and cancellation support
 Update error handling to save state before failing
Step 4.3: Create Signal Handler System
 File: src/core/signal_handler.py
 Implement process signal management
 Add pause/resume signal handling (SIGUSR1/SIGUSR2)
 Graceful shutdown on SIGTERM/SIGINT
 Progress report trigger on SIGUSR1
 Integration with checkpoint manager
Phase 5: Update Processor Modules
Step 5.1: Update ResamplingProcessor with Progress/Checkpoints
 File: src/processors/data_preparation/resampling_processor.py
 Inherit from ProcessAwareProcessor
 Add progress reporting for each resampling step
 Implement checkpoint save/restore for partial datasets
 Add file-level processing status tracking
 Support resume from interrupted resampling operations
Step 5.2: Update RasterMerger with Progress Support
 File: src/processors/data_preparation/raster_merger.py
 Add progress callbacks for merge operations
 Implement checkpoint support for partial merges
 Track merge progress at tile/chunk level
 Support resume from interrupted merge operations
Step 5.3: Update Pipeline Orchestrator
 File: src/pipelines/unified_resampling/pipeline_orchestrator.py
 Integrate with ProgressManager and CheckpointManager
 Add phase-level checkpointing (resampling, merging, analysis)
 Implement pipeline resume logic
 Add detailed progress reporting for each phase
 Support selective phase execution (resume from specific phase)
Phase 6: Update Resampling Engines
Step 6.1: Update NumpyResampler with Progress
 File: src/resampling/engines/numpy_resampler.py
 Add progress callbacks for chunk-based resampling
 Implement intermediate result caching
 Add memory usage monitoring and reporting
 Support cancellation and cleanup
Step 6.2: Update GDALResampler with Progress
 File: src/resampling/engines/gdal_resampler.py
 Add GDAL progress callbacks integration
 Implement timeout mechanisms for GDAL operations
 Add memory monitoring for GDAL operations
 Support operation cancellation
Step 6.3: Update ResamplingCacheManager
 File: src/resampling/cache_manager.py
 Add progress reporting for cache operations
 Implement cache validation and recovery
 Add cache statistics and monitoring
 Support partial cache restoration
Phase 7: Better Process Management Solution
Step 7.1: Create Process Controller
 File: src/core/process_controller.py
 Replace tmux with Python-based process management
 Implement daemon mode with PID file management
 Add pause/resume functionality (not just kill)
 Support background execution with log rotation
 Process health monitoring and auto-restart
Step 7.2: Create CLI Process Management
 File: scripts/process_manager.py
 CLI tool for start/stop/pause/resume/status operations
 Integration with checkpoint system for resume
 Process monitoring and resource usage display
 Log viewing and tail functionality
Step 7.3: Update Main Execution Scripts
 File: scripts/production/run_unified_resampling.sh
 Replace tmux with process controller
 Add process management options (daemon, foreground, etc.)
 Support resume from checkpoint via command line
 Add signal forwarding for process control
Phase 8: Integration and Testing
Step 8.1: Integration Testing
 File: tests/integration/test_progress_checkpointing.py
 Test full pipeline with progress monitoring
 Test checkpoint save/restore at each phase
 Test signal handling and graceful shutdown
 Test resume from various checkpoint states
Step 8.2: Performance Testing
 File: tests/performance/test_large_dataset_processing.py
 Test with large raster files (GB+ size)
 Measure progress reporting overhead
 Test checkpoint performance impact
 Memory usage monitoring during processing
Step 8.3: Recovery Testing
 File: tests/recovery/test_failure_scenarios.py
 Test recovery from various failure points
 Test corrupted checkpoint handling
 Test partial processing resume
 Test signal-based process control
Implementation Order and Dependencies
Phase 1 (URGENT - Fix Hang): Days 1-2 - ✅ COMPLETED
Step 1.1 → Step 1.2 → Step 1.3 → Step 1.4
Goal: Process can start without hanging on large files
Status: ✅ 100% Complete
Phase 1.5 (URGENT - Fix Memory): Days 3-4 - 🟡 IN PROGRESS
Step 1.5.1 (✅ Complete) → Step 1.5.2 → Step 1.5.3 → Step 1.5.4
Goal: Process large files without memory pressure
Status: 25% Complete (Pipeline infrastructure done)
Phase 2 (Progress Infrastructure): Days 5-6
Step 2.1 → Step 2.2 → Step 2.3
Goal: Comprehensive progress monitoring system
Phase 3 (Checkpointing): Days 7-8
Step 3.2 → Step 3.1 → Step 3.3
Goal: Save/restore processing state at any point
Phase 4 (Base Classes): Day 9
Step 4.1 → Step 4.2 → Step 4.3
Goal: All processors support progress/checkpoints
Phase 5 (Processor Updates): Days 10-11
Step 5.1 → Step 5.2 → Step 5.3
Goal: Main processing pipeline fully instrumented
Phase 6 (Engine Updates): Day 12
Step 6.1 → Step 6.2 → Step 6.3
Goal: Resampling engines report progress
Phase 7 (Process Management): Day 13
Step 7.1 → Step 7.2 → Step 7.3
Goal: Replace tmux with proper process control
Phase 8 (Integration): Days 14-15
Step 8.1 → Step 8.2 → Step 8.3
Goal: Fully tested, production-ready system
Success Criteria
Immediate Success (After Phase 1) - ✅ ACHIEVED:
 Process starts without hanging on large raster files
 Basic progress reporting shows file processing status
 Database gets populated with raster metadata
Phase 1.5 Success (After Memory Fix) - 🎯 TARGET:
 Pipeline processes large raster files without memory pressure hangs
 Chunked data loading supports files 100+ MB
 Memory usage stays below configured limits during processing
 Progress reporting works during all resampling phases
Intermediate Success (After Phase 5):
 Full progress monitoring throughout pipeline
 Can resume from any processing phase
 Graceful shutdown preserves processing state
Final Success (After Phase 8):
 Replace tmux with pause/resume capable process management
 Complete progress visibility from file I/O to final results
 Robust recovery from any failure point
 Production-ready with comprehensive monitoring