#!/usr/bin/env python3
# scripts/process_manager.py
"""CLI tool for managing biodiversity pipeline processes."""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.process_controller import ProcessController
from src.core.progress_manager import get_progress_manager
from src.checkpoints import get_checkpoint_manager
from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.schema import schema
from tabulate import tabulate


class PipelineCLI:
    """Command-line interface for pipeline management."""
    
    def __init__(self):
        self.config = Config()
        self.db = DatabaseManager()
        self.process_controller = ProcessController()
        self.progress_manager = get_progress_manager()
        self.checkpoint_manager = get_checkpoint_manager()
        
        # Clean up stale processes on startup
        self.process_controller.registry.cleanup_stale_processes()
    
    def start(self, args):
        """Start a pipeline process."""
        
        # Get values from args with defaults
        experiment_name = args.experiment_name or args.name or "production_test"
        analysis_method = args.analysis_method or "som"
        
        # Build command - run the pipeline orchestrator directly
        command = [
            sys.executable,
            "-c",
            f"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Preserve environment variables
if os.getenv('FORCE_TEST_MODE'):
    os.environ['FORCE_TEST_MODE'] = os.getenv('FORCE_TEST_MODE')
if os.getenv('DB_NAME'):
    os.environ['DB_NAME'] = os.getenv('DB_NAME')

from src.pipelines.orchestrator import PipelineOrchestrator
from src.pipelines.stages.load_stage import DataLoadStage
from src.pipelines.stages.resample_stage import ResampleStage
from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.export_stage import ExportStage
from src.pipelines.stages.analysis_stage import AnalysisStage
from src.config.config import Config
from src.database.connection import DatabaseManager

config = Config()
db = DatabaseManager()

# Initialize signal handler first
from src.core.signal_handler import get_signal_handler
signal_handler = get_signal_handler()

# Create pipeline orchestrator
orchestrator = PipelineOrchestrator(config, db)

# Configure pipeline stages
analysis_method = '{analysis_method}'
stages = [
    DataLoadStage,
    ResampleStage, 
    MergeStage,
    ExportStage,
    lambda: AnalysisStage(analysis_method)  # Use specified analysis method
]

# Instantiate stages (handle lambda for AnalysisStage)
stage_instances = []
for stage_class in stages:
    if callable(stage_class) and not isinstance(stage_class, type):
        # Handle lambda case
        stage_instances.append(stage_class())
    else:
        # Handle normal class case
        stage_instances.append(stage_class())

for stage in stage_instances:
    orchestrator.register_stage(stage)

# Use the experiment name passed from command line
experiment_name = '{experiment_name}'
description = 'Production pipeline with skip-resampling functionality'

print(f'🚀 Starting pipeline: {{experiment_name}}')

# Run complete pipeline
results = orchestrator.run_pipeline(
    experiment_name=experiment_name,
    description=description,
    resume_from_checkpoint=True
)

print('✅ Pipeline completed successfully!')
"""
        ]
            
        # Note: Arguments are already embedded in the Python code string above
        # No need to add them to the command list
        
        # Start process
        try:
            pid = self.process_controller.start_process(
                name=args.name or f"pipeline_{int(time.time())}",
                command=command,
                daemon_mode=args.daemon,
                auto_restart=args.auto_restart,
                working_dir=project_root
            )
            
            print(f"✅ Started process with PID: {pid}")
            
            if not args.daemon:
                # Follow logs if not daemon
                self._follow_logs(args.name or f"pipeline_{int(time.time())}")
            else:
                print(f"Running in daemon mode. Use 'process_manager.py logs {args.name}' to view logs")
                
        except Exception as e:
            print(f"❌ Failed to start process: {e}")
            sys.exit(1)
    
    def stop(self, args):
        """Stop a pipeline process."""
        try:
            if self.process_controller.stop_process(args.name, timeout=args.timeout):
                print(f"✅ Stopped process: {args.name}")
            else:
                print(f"❌ Failed to stop process: {args.name}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def pause(self, args):
        """Pause a pipeline process."""
        try:
            if self.process_controller.pause_process(args.name):
                print(f"⏸️  Paused process: {args.name}")
            else:
                print(f"❌ Failed to pause process: {args.name}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def resume(self, args):
        """Resume a paused process."""
        try:
            if self.process_controller.resume_process(args.name):
                print(f"▶️  Resumed process: {args.name}")
            else:
                print(f"❌ Failed to resume process: {args.name}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def status(self, args):
        """Show process status."""
        try:
            if args.name:
                # Show specific process
                info = self.process_controller.get_process_status(args.name)
                self._display_process_status(info)
                
                # Show pipeline progress if available
                self._show_pipeline_progress(args.name)
            else:
                # Show all processes
                processes = self.process_controller.list_processes()
                if not processes:
                    print("No managed processes running")
                else:
                    self._display_process_list(processes)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def logs(self, args):
        """Show process logs."""
        try:
            if args.follow:
                self._follow_logs(args.name)
            else:
                lines = self.process_controller.tail_log(args.name, args.lines)
                for line in lines:
                    print(line)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def checkpoints(self, args):
        """Manage checkpoints."""
        if args.action == 'list':
            self._list_checkpoints(args)
        elif args.action == 'info':
            self._checkpoint_info(args)
        elif args.action == 'cleanup':
            self._cleanup_checkpoints(args)
    
    def experiments(self, args):
        """List experiments and their status."""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT id, name, status, started_at, completed_at,
                           config->>'target_resolution' as resolution
                    FROM experiments
                    ORDER BY started_at DESC
                    LIMIT %s
                """, (args.limit,))
                
                experiments = cursor.fetchall()
                
                if not experiments:
                    print("No experiments found")
                    return
                
                # Format for display
                data = []
                for exp in experiments:
                    duration = None
                    if exp['completed_at'] and exp['started_at']:
                        duration = (exp['completed_at'] - exp['started_at']).total_seconds()
                        duration = f"{duration/60:.1f}m"
                    
                    data.append([
                        exp['id'][:8],
                        exp['name'][:30],
                        exp['status'],
                        exp['resolution'],
                        exp['started_at'].strftime('%Y-%m-%d %H:%M'),
                        duration or '-'
                    ])
                
                headers = ['ID', 'Name', 'Status', 'Resolution', 'Started', 'Duration']
                print(tabulate(data, headers=headers, tablefmt='grid'))
                
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    def resources(self, args):
        """Show resource usage."""
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            print("System Resources:")
            print(f"  CPU Usage: {cpu_percent}%")
            print(f"  Memory: {memory.percent}% ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
            print(f"  Disk: {disk.percent}% ({disk.used/1024/1024/1024:.1f}GB / {disk.total/1024/1024/1024:.1f}GB)")
            print()
            
            # Process-specific resources
            processes = self.process_controller.list_processes()
            if processes:
                print("Process Resources:")
                data = []
                for proc in processes:
                    data.append([
                        proc.name,
                        proc.pid,
                        f"{proc.cpu_percent:.1f}%",
                        f"{proc.memory_mb:.1f}MB"
                    ])
                
                headers = ['Process', 'PID', 'CPU', 'Memory']
                print(tabulate(data, headers=headers, tablefmt='simple'))
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _display_process_status(self, info: Dict[str, Any]):
        """Display detailed process status."""
        print(f"\nProcess: {info.name}")
        print(f"PID: {info.pid}")
        print(f"Status: {info.status}")
        print(f"CPU: {info.cpu_percent:.1f}%")
        print(f"Memory: {info.memory_mb:.1f} MB")
        print(f"Uptime: {self._format_duration(time.time() - info.start_time)}")
        
        if info.log_file:
            print(f"Log file: {info.log_file}")
    
    def _display_process_list(self, processes: list):
        """Display process list in table format."""
        data = []
        for proc in processes:
            uptime = self._format_duration(time.time() - proc.start_time)
            data.append([
                proc.name,
                proc.pid,
                proc.status,
                f"{proc.cpu_percent:.1f}%",
                f"{proc.memory_mb:.1f}MB",
                uptime
            ])
        
        headers = ['Name', 'PID', 'Status', 'CPU', 'Memory', 'Uptime']
        print(tabulate(data, headers=headers, tablefmt='grid'))
    
    def _show_pipeline_progress(self, process_name: str):
        """Show pipeline progress if available."""
        # Try to find active pipeline
        active_nodes = self.progress_manager.get_active_nodes()
        
        for node in active_nodes:
            if process_name in node.get('name', ''):
                print(f"\nPipeline Progress:")
                self._display_progress_tree(node)
                break
    
    def _display_progress_tree(self, node: Dict[str, Any], indent: int = 0):
        """Display progress tree recursively."""
        prefix = "  " * indent
        status_icon = {
            'running': '🔄',
            'completed': '✅',
            'failed': '❌',
            'pending': '⏳'
        }.get(node['status'], '❓')
        
        print(f"{prefix}{status_icon} {node['name']}: {node['progress_percent']:.1f}%")
        
        if 'children' in node:
            for child in node['children']:
                self._display_progress_tree(child, indent + 1)
    
    def _list_checkpoints(self, args):
        """List available checkpoints."""
        checkpoints = self.checkpoint_manager.list_checkpoints(
            processor_name=args.processor,
            level=args.level,
            status='valid'
        )
        
        if not checkpoints:
            print("No checkpoints found")
            return
        
        data = []
        for cp in checkpoints[:args.limit]:
            data.append([
                cp['checkpoint_id'][:20],
                cp['level'],
                cp['processor_name'],
                self._format_size(cp['file_size_bytes']),
                cp['created_at'].strftime('%Y-%m-%d %H:%M')
            ])
        
        headers = ['Checkpoint ID', 'Level', 'Processor', 'Size', 'Created']
        print(tabulate(data, headers=headers, tablefmt='grid'))
    
    def _checkpoint_info(self, args):
        """Show checkpoint details."""
        checkpoint = self.checkpoint_manager.get_checkpoint_info(args.checkpoint_id)
        
        if not checkpoint:
            print(f"Checkpoint not found: {args.checkpoint_id}")
            return
        
        print(f"\nCheckpoint: {checkpoint.checkpoint_id}")
        print(f"Level: {checkpoint.level}")
        print(f"Processor: {checkpoint.processor_name}")
        print(f"Status: {checkpoint.status}")
        print(f"File: {checkpoint.file_path}")
        print(f"Size: {self._format_size(checkpoint.file_size_bytes)}")
        print(f"Created: {checkpoint.created_at}")
        
        if checkpoint.data_summary:
            print(f"\nData Summary:")
            print(json.dumps(checkpoint.data_summary, indent=2))
    
    def _cleanup_checkpoints(self, args):
        """Clean up old checkpoints."""
        count = self.checkpoint_manager.cleanup_old_checkpoints(
            days_old=args.days,
            keep_minimum={
                'pipeline': args.keep_pipeline,
                'phase': args.keep_phase,
                'step': args.keep_step,
                'substep': args.keep_substep
            }
        )
        print(f"Cleaned up {count} checkpoints")
    
    def _follow_logs(self, process_name: str):
        """Follow process logs in real-time."""
        print(f"Following logs for {process_name} (Ctrl+C to stop)...")
        
        last_size = 0
        try:
            while True:
                lines = self.process_controller.tail_log(process_name, 100)
                
                # Print new lines
                for line in lines[last_size:]:
                    print(line)
                
                last_size = len(lines)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopped following logs")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _format_size(self, bytes_size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f}{unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f}TB"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Biodiversity Pipeline Process Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a pipeline process')
    start_parser.add_argument('--name', help='Process name')
    start_parser.add_argument('--experiment-name', help='Experiment name')
    start_parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    start_parser.add_argument('--auto-restart', action='store_true', help='Auto-restart on crash')
    start_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    start_parser.add_argument('--target-resolution', type=float, help='Target resolution')
    start_parser.add_argument('--max-samples', type=int, help='Maximum samples')
    start_parser.add_argument('--analysis-method', choices=['som', 'maxp_regions', 'gwpca'], help='Analysis method')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop a process')
    stop_parser.add_argument('name', help='Process name')
    stop_parser.add_argument('--timeout', type=float, default=30, help='Shutdown timeout')
    
    # Pause command
    pause_parser = subparsers.add_parser('pause', help='Pause a process')
    pause_parser.add_argument('name', help='Process name')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume a paused process')
    resume_parser.add_argument('name', help='Process name')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show process status')
    status_parser.add_argument('name', nargs='?', help='Process name (optional)')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show process logs')
    logs_parser.add_argument('name', help='Process name')
    logs_parser.add_argument('--lines', type=int, default=100, help='Number of lines')
    logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow logs')
    
    # Checkpoints command
    cp_parser = subparsers.add_parser('checkpoints', help='Manage checkpoints')
    cp_subparsers = cp_parser.add_subparsers(dest='action')
    
    cp_list = cp_subparsers.add_parser('list', help='List checkpoints')
    cp_list.add_argument('--processor', help='Filter by processor')
    cp_list.add_argument('--level', help='Filter by level')
    cp_list.add_argument('--limit', type=int, default=20, help='Number to show')
    
    cp_info = cp_subparsers.add_parser('info', help='Show checkpoint details')
    cp_info.add_argument('checkpoint_id', help='Checkpoint ID')
    
    cp_cleanup = cp_subparsers.add_parser('cleanup', help='Clean up old checkpoints')
    cp_cleanup.add_argument('--days', type=int, default=7, help='Days to keep')
    cp_cleanup.add_argument('--keep-pipeline', type=int, default=5, help='Min pipeline checkpoints')
    cp_cleanup.add_argument('--keep-phase', type=int, default=3, help='Min phase checkpoints')
    cp_cleanup.add_argument('--keep-step', type=int, default=2, help='Min step checkpoints')
    cp_cleanup.add_argument('--keep-substep', type=int, default=1, help='Min substep checkpoints')
    
    # Experiments command
    exp_parser = subparsers.add_parser('experiments', help='List experiments')
    exp_parser.add_argument('--limit', type=int, default=10, help='Number to show')
    
    # Resources command
    res_parser = subparsers.add_parser('resources', help='Show resource usage')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    cli = PipelineCLI()
    
    commands = {
        'start': cli.start,
        'stop': cli.stop,
        'pause': cli.pause,
        'resume': cli.resume,
        'status': cli.status,
        'logs': cli.logs,
        'checkpoints': cli.checkpoints,
        'experiments': cli.experiments,
        'resources': cli.resources
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()