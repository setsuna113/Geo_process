#!/usr/bin/env python3
"""Unified monitoring and logging CLI for pipeline execution."""

import argparse
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import DatabaseManager
from src.infrastructure.monitoring import MonitoringClient
from src.config import config
from tabulate import tabulate
import logging

# Setup basic logging for the CLI
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MonitorCLI:
    """CLI for monitoring and log access."""
    
    def __init__(self):
        """Initialize monitor CLI."""
        self.db = DatabaseManager()
        self.monitor = MonitoringClient(self.db)
    
    def status(self, args):
        """Show experiment status with progress."""
        try:
            status = self.monitor.get_experiment_status(args.experiment)
            
            # Display header
            print(f"\n{'='*60}")
            print(f"Experiment: {status['name']} ({status['id'][:8]})")
            print(f"Status: {self._format_status(status['status'])}")
            print(f"Started: {self._format_time(status['started_at'])}")
            if status['completed_at']:
                print(f"Completed: {self._format_time(status['completed_at'])}")
                duration = (status['completed_at'] - status['started_at']).total_seconds()
                print(f"Duration: {self._format_duration(duration)}")
            print(f"{'='*60}")
            
            # Display progress tree
            print("\nProgress:")
            self._display_progress_tree(status['progress_tree'])
            
            # Show error summary if any
            if status['error_count'] > 0:
                print(f"\n⚠️  {status['error_count']} errors found.")
                print(f"Run 'monitor.py logs {args.experiment} --level ERROR' to view")
            
            # Show current resource usage if running
            if status['status'] == 'running':
                self._show_current_metrics(status['id'])
                
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            sys.exit(1)
    
    def logs(self, args):
        """Query and display logs."""
        try:
            # Resolve experiment name to ID
            experiment_id = self.monitor._resolve_experiment_id(args.experiment)
            
            logs = self.monitor.query_logs(
                experiment_id=experiment_id,
                level=args.level,
                search=args.search,
                start_time=args.since,
                limit=args.limit
            )
            
            if not logs:
                print("No logs found matching criteria")
                return
            
            # Format based on output type
            if args.json:
                # JSON output for machine parsing
                for log in logs:
                    print(json.dumps(log, default=str))
            else:
                # Human-readable output
                for log in logs:
                    # Color based on level
                    level_colors = {
                        'ERROR': '\033[91m',
                        'CRITICAL': '\033[95m',
                        'WARNING': '\033[93m',
                        'INFO': '\033[0m',
                        'DEBUG': '\033[90m'
                    }
                    color = level_colors.get(log['level'], '')
                    reset = '\033[0m' if color else ''
                    
                    # Format timestamp
                    timestamp = log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Build output line
                    line = f"{timestamp} {color}{log['level']:8}{reset}"
                    
                    # Add context if available
                    if log.get('stage') and log['stage'] != 'unknown':
                        line += f" [{log['stage']}]"
                    
                    line += f" {log['message']}"
                    print(line)
                    
                    # Show traceback if requested and available
                    if args.traceback and log.get('traceback'):
                        print(f"{color}{'='*60}")
                        print(log['traceback'])
                        print(f"{'='*60}{reset}")
                        
        except Exception as e:
            logger.error(f"Error querying logs: {e}")
            sys.exit(1)
    
    def watch(self, args):
        """Live monitoring of experiment."""
        print(f"Watching experiment {args.experiment} (Ctrl+C to stop)")
        print("Press 'q' to quit, 'e' to show errors, 'm' to show metrics\n")
        
        last_log_id = None
        show_mode = 'logs'  # logs, errors, metrics
        
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end='')
                
                # Get current status
                status = self.monitor.get_experiment_status(args.experiment)
                
                # Show header
                print(f"Experiment: {status['name']} - Status: {self._format_status(status['status'])}")
                print(f"Running for: {self._format_duration((datetime.utcnow() - status['started_at']).total_seconds())}")
                print("="*80)
                
                # Show progress summary
                self._show_progress_summary(status['progress_tree'])
                print("="*80)
                
                if show_mode == 'logs':
                    # Show recent logs
                    logs = self.monitor.query_logs(
                        experiment_id=status['id'],
                        after_id=last_log_id,
                        limit=15
                    )
                    
                    if logs:
                        print("\nRecent logs:")
                        for log in logs[-10:]:  # Show last 10
                            level_indicator = {
                                'ERROR': '❌', 'WARNING': '⚠️ ', 
                                'INFO': '✓ ', 'DEBUG': '🔍'
                            }.get(log['level'], '  ')
                            
                            print(f"{log['timestamp'].strftime('%H:%M:%S')} {level_indicator} "
                                  f"{log['message'][:100]}")
                        
                        last_log_id = logs[-1]['id']
                
                elif show_mode == 'errors':
                    # Show recent errors
                    errors = self.monitor.query_logs(
                        experiment_id=status['id'],
                        level='ERROR',
                        limit=10
                    )
                    
                    if errors:
                        print("\nRecent errors:")
                        for err in errors:
                            print(f"{err['timestamp'].strftime('%H:%M:%S')} "
                                  f"[{err.get('stage', 'unknown')}] {err['message']}")
                    else:
                        print("\nNo errors found ✅")
                
                elif show_mode == 'metrics':
                    # Show metrics
                    self._show_current_metrics(status['id'])
                
                # Check if complete
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    print(f"\n\nExperiment {status['status']}!")
                    break
                
                # Wait and check for input
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\nStopped watching")
    
    def metrics(self, args):
        """Show performance metrics."""
        try:
            metrics = self.monitor.get_metrics(
                experiment_id=args.experiment,
                node_id=args.node,
                metric_type=args.type
            )
            
            if not metrics:
                print("No metrics found")
                return
            
            # Group by node for display
            by_node = {}
            for m in metrics:
                node = m.get('node_id', 'system')
                if node not in by_node:
                    by_node[node] = []
                by_node[node].append(m)
            
            # Display metrics
            for node, node_metrics in by_node.items():
                print(f"\nNode: {node}")
                print("-" * 60)
                
                # Create table data
                data = []
                for m in node_metrics[:20]:  # Limit display
                    row = [m['timestamp'].strftime('%H:%M:%S')]
                    
                    if 'memory_mb' in m:
                        row.append(f"{m['memory_mb']:.1f}")
                    else:
                        row.append('-')
                        
                    if 'cpu_percent' in m:
                        row.append(f"{m['cpu_percent']:.1f}")
                    else:
                        row.append('-')
                        
                    if 'throughput_per_sec' in m:
                        row.append(f"{m['throughput_per_sec']:.2f}")
                    else:
                        row.append('-')
                    
                    data.append(row)
                
                headers = ['Time', 'Memory(MB)', 'CPU%', 'Items/sec']
                print(tabulate(data, headers=headers, tablefmt='simple'))
                
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            sys.exit(1)
    
    def errors(self, args):
        """Show error summary."""
        try:
            summary = self.monitor.get_error_summary(args.experiment)
            
            print(f"\nError Summary for {args.experiment}")
            print("="*60)
            print(f"Total Errors: {summary['total_count']}")
            
            if summary['by_level']:
                print("\nErrors by Level:")
                for level, count in summary['by_level'].items():
                    print(f"  {level}: {count}")
            
            if summary['by_stage']:
                print("\nErrors by Stage:")
                for stage, count in summary['by_stage'].items():
                    print(f"  {stage}: {count}")
            
            if summary['recent_errors']:
                print("\nRecent Errors:")
                print("-"*60)
                
                for i, err in enumerate(summary['recent_errors'][:10]):
                    print(f"\n{i+1}. [{err['timestamp']}] Stage: {err.get('stage', 'unknown')}")
                    print(f"   {err['message']}")
                    
                    if args.traceback and 'traceback' in err:
                        print("\n   Traceback:")
                        for line in err['traceback'].split('\n')[:5]:
                            print(f"   {line}")
                        print("   ...")
                        
        except Exception as e:
            logger.error(f"Error getting error summary: {e}")
            sys.exit(1)
    
    def list_experiments(self, args):
        """List recent experiments."""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        e.id, e.name, e.status, e.started_at, e.completed_at,
                        COUNT(DISTINCT l.id) FILTER (WHERE l.level IN ('ERROR', 'CRITICAL')) as error_count,
                        COUNT(DISTINCT p.node_id) FILTER (WHERE p.status = 'completed') as completed_nodes,
                        COUNT(DISTINCT p.node_id) as total_nodes
                    FROM experiments e
                    LEFT JOIN pipeline_logs l ON e.id = l.experiment_id
                    LEFT JOIN pipeline_progress p ON e.id = p.experiment_id
                    GROUP BY e.id, e.name, e.status, e.started_at, e.completed_at
                    ORDER BY e.started_at DESC
                    LIMIT %s
                """, (args.limit,))
                
                experiments = cursor.fetchall()
                
                if not experiments:
                    print("No experiments found")
                    return
                
                # Format data
                data = []
                for exp in experiments:
                    duration = '-'
                    if exp['completed_at'] and exp['started_at']:
                        dur_sec = (exp['completed_at'] - exp['started_at']).total_seconds()
                        duration = self._format_duration(dur_sec)
                    
                    progress = f"{exp['completed_nodes']}/{exp['total_nodes']}"
                    
                    data.append([
                        exp['id'][:8],
                        exp['name'][:30],
                        self._format_status(exp['status']),
                        exp['started_at'].strftime('%Y-%m-%d %H:%M'),
                        duration,
                        progress,
                        exp['error_count'] or 0
                    ])
                
                headers = ['ID', 'Name', 'Status', 'Started', 'Duration', 'Progress', 'Errors']
                print(tabulate(data, headers=headers, tablefmt='grid'))
                
        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            sys.exit(1)
    
    def _display_progress_tree(self, nodes: List[Dict], indent: int = 0):
        """Display progress tree recursively."""
        for node in nodes:
            prefix = "  " * indent
            
            # Status icon
            status_icon = {
                'completed': '✅',
                'running': '🔄',
                'failed': '❌',
                'pending': '⏳',
                'cancelled': '🚫'
            }.get(node['status'], '❓')
            
            # Progress bar
            progress = node['progress_percent']
            bar_width = 20
            filled = int(bar_width * progress / 100)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            print(f"{prefix}{status_icon} {node['name']}: [{bar}] {progress:.1f}%")
            
            # Recurse for children
            if node.get('children'):
                self._display_progress_tree(node['children'], indent + 1)
    
    def _show_progress_summary(self, nodes: List[Dict]):
        """Show compact progress summary."""
        # Flatten tree and count by status
        status_counts = {'completed': 0, 'running': 0, 'failed': 0, 'pending': 0}
        
        def count_nodes(nodes):
            for node in nodes:
                status_counts[node['status']] += 1
                if node.get('children'):
                    count_nodes(node['children'])
        
        count_nodes(nodes)
        
        # Display summary
        total = sum(status_counts.values())
        if total > 0:
            completed_pct = (status_counts['completed'] / total) * 100
            print(f"Overall Progress: {completed_pct:.1f}% "
                  f"(✅ {status_counts['completed']} | "
                  f"🔄 {status_counts['running']} | "
                  f"❌ {status_counts['failed']} | "
                  f"⏳ {status_counts['pending']})")
    
    def _show_current_metrics(self, experiment_id: str):
        """Show current resource metrics."""
        metrics = self.monitor.get_metrics(experiment_id)
        
        if metrics:
            latest = metrics[0]
            print(f"\nCurrent Resources:")
            if 'memory_mb' in latest:
                print(f"  Memory: {latest['memory_mb']:.1f} MB")
            if 'cpu_percent' in latest:
                print(f"  CPU: {latest['cpu_percent']:.1f}%")
            if 'throughput_per_sec' in latest:
                print(f"  Throughput: {latest['throughput_per_sec']:.2f} items/sec")
    
    def _format_status(self, status: str) -> str:
        """Format status with color."""
        colors = {
            'running': '\033[92m',  # Green
            'completed': '\033[94m',  # Blue
            'failed': '\033[91m',  # Red
            'pending': '\033[93m',  # Yellow
            'cancelled': '\033[95m'  # Magenta
        }
        color = colors.get(status, '')
        reset = '\033[0m' if color else ''
        return f"{color}{status.upper()}{reset}"
    
    def _format_time(self, dt: Optional[datetime]) -> str:
        """Format datetime for display."""
        if not dt:
            return '-'
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Unified monitoring and logging for pipeline execution"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show experiment status')
    status_parser.add_argument('experiment', help='Experiment name or ID')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Query experiment logs')
    logs_parser.add_argument('experiment', help='Experiment name or ID')
    logs_parser.add_argument('--level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                            help='Filter by log level')
    logs_parser.add_argument('--search', help='Search in log messages')
    logs_parser.add_argument('--since', help='Time filter (e.g., "1h", "30m", "1d")')
    logs_parser.add_argument('--limit', type=int, default=100, help='Max logs to show')
    logs_parser.add_argument('--json', action='store_true', help='Output as JSON')
    logs_parser.add_argument('--traceback', action='store_true', help='Show full tracebacks')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Live monitoring')
    watch_parser.add_argument('experiment', help='Experiment name or ID')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show performance metrics')
    metrics_parser.add_argument('experiment', help='Experiment name or ID')
    metrics_parser.add_argument('--node', help='Filter by node ID')
    metrics_parser.add_argument('--type', choices=['memory', 'cpu', 'throughput'],
                               help='Metric type to show')
    
    # Errors command
    errors_parser = subparsers.add_parser('errors', help='Show error summary')
    errors_parser.add_argument('experiment', help='Experiment name or ID')
    errors_parser.add_argument('--traceback', action='store_true', help='Show tracebacks')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--limit', type=int, default=20, help='Number to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    cli = MonitorCLI()
    
    commands = {
        'status': cli.status,
        'logs': cli.logs,
        'watch': cli.watch,
        'metrics': cli.metrics,
        'errors': cli.errors,
        'list': cli.list_experiments
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()