#!/usr/bin/env python3
"""
Resource checker for CP-WOPT and other analysis methods.
Warns if configured resources exceed available resources.
"""

import subprocess
import os
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Tuple, List

def get_system_resources():
    """Get current system resource availability."""
    
    # Get CPU info - SECURE: No shell=True
    cpu_info = subprocess.check_output(["nproc"]).decode().strip()
    total_cpus = int(cpu_info)
    
    # Get current CPU usage - SECURE: Using proper subprocess pipeline
    ps_proc = subprocess.Popen(["ps", "aux"], stdout=subprocess.PIPE)
    awk_proc = subprocess.Popen(
        ["awk", "$3 > 0.0 {sum+=$3} END {print sum}"],
        stdin=ps_proc.stdout,
        stdout=subprocess.PIPE
    )
    ps_proc.stdout.close()  # Allow ps_proc to receive SIGPIPE if awk_proc exits
    cpu_usage = awk_proc.communicate()[0].decode().strip()
    
    # Handle empty output
    try:
        used_cores = int(float(cpu_usage) / 100) if cpu_usage else 0
    except (ValueError, TypeError):
        used_cores = 0
    free_cores = total_cpus - used_cores
    
    # Get memory info (in GB) - SECURE: Using proper subprocess pipeline
    free_proc = subprocess.Popen(["free", "-g"], stdout=subprocess.PIPE)
    grep_proc = subprocess.Popen(
        ["grep", "Mem"],
        stdin=free_proc.stdout,
        stdout=subprocess.PIPE
    )
    free_proc.stdout.close()
    awk_proc = subprocess.Popen(
        ["awk", "{print $2,$3,$7}"],
        stdin=grep_proc.stdout,
        stdout=subprocess.PIPE
    )
    grep_proc.stdout.close()
    mem_output = awk_proc.communicate()[0].decode().strip()
    
    # Parse memory info with error handling
    try:
        mem_info = mem_output.split()
        total_mem = int(mem_info[0])
        used_mem = int(mem_info[1])
        available_mem = int(mem_info[2])
    except (IndexError, ValueError):
        # Fallback values if parsing fails
        total_mem = 16
        used_mem = 8
        available_mem = 8
    
    return {
        'cpu': {
            'total': total_cpus,
            'used': used_cores,
            'free': free_cores,
            'safe_allocation': int(free_cores * 0.7)  # 70% of free
        },
        'memory': {
            'total_gb': total_mem,
            'used_gb': used_mem,
            'available_gb': available_mem,
            'safe_allocation_gb': int(available_mem * 0.7)  # 70% of available
        }
    }

def calculate_optimal_config(resources):
    """Calculate optimal CP-WOPT configuration based on resources."""
    
    # CPU-based calculations
    n_workers = resources['cpu']['safe_allocation']
    
    # Memory-based calculations
    safe_mem = resources['memory']['safe_allocation_gb']
    
    # Tensor memory allocation (50% of safe memory)
    max_tensor_gb = int(safe_mem * 0.5)
    
    # Batch size calculation
    # Assuming 180x360 spatial grid, 4 features, 8 bytes per value
    mem_per_sample = (180 * 360 * 4 * 8) / (1024**3)  # GB per full tensor
    # Batch size based on 10% of tensor memory
    batch_size = int((max_tensor_gb * 0.1) / mem_per_sample)
    batch_size = max(1000, min(100000, batch_size))  # Clamp to reasonable range
    
    available_mem = resources['memory']['available_gb']
    
    config = {
        'algorithm': {
            'rank': 50 if safe_mem > 50 else 30,  # Adjust rank based on memory
        },
        'memory': {
            'max_tensor_gb': max_tensor_gb,
            'chunk_memory_fraction': 0.7 if available_mem < 200 else 0.8
        },
        'parallel': {
            'n_workers': n_workers
        },
        'minibatch': {
            'batch_size': batch_size
        }
    }
    
    return config

def generate_env_exports(config):
    """Generate environment variable export commands."""
    exports = []
    exports.append(f"export CPWOPT_ALGORITHM_RANK={config['algorithm']['rank']}")
    exports.append(f"export CPWOPT_MEMORY_MAX_TENSOR_GB={config['memory']['max_tensor_gb']}")
    exports.append(f"export CPWOPT_PARALLEL_N_WORKERS={config['parallel']['n_workers']}")
    exports.append(f"export CPWOPT_MINIBATCH_BATCH_SIZE={config['minibatch']['batch_size']}")
    return exports

def check_method_resources(method: str, config_path: str = None) -> Tuple[bool, List[str], Dict]:
    """Check if method configuration is compatible with available resources."""
    
    # Get current resources
    resources = get_system_resources()
    
    # Load method config
    if method == 'cpwopt':
        if not config_path:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'cpwopt' / 'cpwopt_config.yml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract resource requirements
        required_workers = config.get('parallel', {}).get('n_workers', 1)
        required_memory = config.get('memory', {}).get('max_tensor_gb', 8)
        
    elif method == 'som':
        # Add SOM checking if needed
        required_workers = 100  # Default SOM workers
        required_memory = 64    # Default SOM memory
    else:
        return True, [], {}
    
    # Check resources
    warnings = []
    suggestions = {}
    is_safe = True
    
    # CPU check
    if required_workers > resources['cpu']['safe_allocation']:
        is_safe = False
        warnings.append(
            f"❌ CPU WARNING: Config requests {required_workers} workers but only "
            f"{resources['cpu']['safe_allocation']} cores safely available"
        )
        suggestions['n_workers'] = resources['cpu']['safe_allocation']
    
    # Memory check
    if required_memory > resources['memory']['safe_allocation_gb']:
        is_safe = False
        warnings.append(
            f"❌ MEMORY WARNING: Config requests {required_memory}GB but only "
            f"{resources['memory']['safe_allocation_gb']}GB safely available"
        )
        suggestions['max_tensor_gb'] = resources['memory']['safe_allocation_gb']
    
    return is_safe, warnings, suggestions

def main():
    """Main function for standalone usage."""
    print("=== Resource Configuration Checker ===\n")
    
    # Get current resources
    resources = get_system_resources()
    
    print("Current System Resources:")
    print(f"  CPU: {resources['cpu']['used']}/{resources['cpu']['total']} cores in use")
    print(f"       {resources['cpu']['free']} cores free")
    print(f"       {resources['cpu']['safe_allocation']} cores recommended")
    print(f"  Memory: {resources['memory']['used_gb']}/{resources['memory']['total_gb']} GB in use")
    print(f"          {resources['memory']['available_gb']} GB available")
    print(f"          {resources['memory']['safe_allocation_gb']} GB recommended")
    
    # Calculate optimal config
    config = calculate_optimal_config(resources)
    
    print("\nRecommended CP-WOPT Configuration:")
    print(f"  Rank: {config['algorithm']['rank']}")
    print(f"  Max tensor memory: {config['memory']['max_tensor_gb']} GB")
    print(f"  Workers: {config['parallel']['n_workers']}")
    print(f"  Batch size: {config['minibatch']['batch_size']}")
    
    print("\nEnvironment Variables:")
    for export in generate_env_exports(config):
        print(f"  {export}")
    
    # Save to file
    with open('cpwopt_dynamic_config.json', 'w') as f:
        json.dump({
            'timestamp': subprocess.check_output(['date']).decode().strip(),
            'resources': resources,
            'config': config
        }, f, indent=2)
    
    print("\nConfiguration saved to cpwopt_dynamic_config.json")

if __name__ == "__main__":
    main()