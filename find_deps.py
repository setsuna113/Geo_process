import yaml
import subprocess
import sys
import os

# --- Configuration ---
# A list of common, low-level packages to ignore in the suggestions.
# This helps reduce noise.
IGNORE_LIST = {
    'python', 'python_abi', 'pip', '_libgcc_mutex', '_openmp_mutex',
    'bzip2', 'ca-certificates', 'ld_impl_linux-64', 'libffi', 'libgcc-ng',
    'libgomp', 'libstdcxx-ng', 'libuuid', 'ncurses', 'openssl', 'readline',
    'sqlite', 'tk', 'tzdata', 'xz', 'zlib'
}
# ---------------------

def read_yaml_file(filepath):
    """Reads and parses the environment.yml file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def get_primary_dependencies(data):
    """Extracts the list of primary dependencies from the YAML data."""
    if 'dependencies' not in data or not isinstance(data['dependencies'], list):
        print("Error: Could not find a valid 'dependencies' list in the YAML file.")
        sys.exit(1)
    
    primary_deps = set()
    for item in data['dependencies']:
        if isinstance(item, str):
            primary_deps.add(item.split('=')[0])
        elif isinstance(item, dict) and 'pip' in item:
            # Handle pip dependencies if you have them
            for pip_dep in item['pip']:
                primary_deps.add(pip_dep.split('==')[0])
    return primary_deps

def get_package_runtime_deps(package_name):
    """Runs 'conda search --info' and parses its runtime dependencies."""
    print(f"  -> Analyzing dependencies for '{package_name}'...")
    try:
        command = ["conda", "search", "--info", f"conda-forge::{package_name}"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        in_dependencies_section = False
        deps = set()
        for line in result.stdout.splitlines():
            if line.strip().startswith('dependencies:'):
                in_dependencies_section = True
                continue
            
            if in_dependencies_section:
                if ':' in line and not line.startswith('  -'):
                    break # We've moved past the dependencies section
                
                # --- THIS IS THE CORRECTED LOGIC ---
                if line.strip().startswith('-'):
                    # Use split() which handles variable whitespace better
                    parts = line.strip().split()
                    # Check if the line has at least 2 parts (e.g., ['-', 'numpy'])
                    if len(parts) > 1:
                        # The package name is the second part
                        dep_name = parts[1]
                        deps.add(dep_name)
                # --- END OF CORRECTION ---

        return deps
    except subprocess.CalledProcessError:
        print(f"Warning: Could not fetch info for '{package_name}'. It might be a pip package or from another channel.")
        return set()
    except Exception as e:
        print(f"An unexpected error occurred while fetching info for '{package_name}': {e}")
        return set()

def update_yaml_file(filepath, new_deps):
    """Appends new dependencies to the YAML file, preserving comments."""
    if not new_deps:
        return

    # Add a comment block for clarity
    comment = "\n# --- Optional dependencies added by find_deps.py ---\n"
    with open(filepath, 'a') as f:
        f.write(comment)
        # Use yaml.dump to write the new dependencies in the correct format
        yaml.dump(sorted(list(new_deps)), f, default_flow_style=False, indent=2)

    print(f"\n✅ Success! Updated '{filepath}' with {len(new_deps)} new packages.")


def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python find_deps.py <path_to_environment.yml>")
        sys.exit(1)

    yaml_filepath = sys.argv[1]
    
    print(f"Reading '{yaml_filepath}'...")
    env_data = read_yaml_file(yaml_filepath)
    primary_deps = get_primary_dependencies(env_data)
    
    all_potential_deps = set()
    
    print("\nFetching dependency information from conda-forge...")
    for package in sorted(list(primary_deps)):
        if package in IGNORE_LIST:
            continue
        runtime_deps = get_package_runtime_deps(package)
        all_potential_deps.update(runtime_deps)
        
    candidates = sorted(list(all_potential_deps - primary_deps - IGNORE_LIST))
    
    if not candidates:
        print("\nNo new optional dependencies found to suggest.")
        sys.exit(0)
        
    print("\n--- Potential Optional Dependencies Found ---")
    print("The following packages were found as dependencies of your primary packages.")
    print("Select the ones you need for your project's optional features.\n")
    
    for i, can in enumerate(candidates):
        print(f"  [{i+1}] {can}")
        
    print("\nEnter the numbers of the packages to add (e.g., '1, 4, 5'), or 'all', or press Enter to skip.")
    
    try:
        choice = input("> ").strip().lower()
        if not choice:
            print("No packages selected. Exiting.")
            sys.exit(0)
            
        chosen_deps = set()
        if choice == 'all':
            chosen_deps = set(candidates)
        else:
            indices = [int(i.strip()) - 1 for i in choice.split(',')]
            for i in indices:
                if 0 <= i < len(candidates):
                    chosen_deps.add(candidates[i])
                else:
                    print(f"Warning: Index {i+1} is out of range.")
                    
        if not chosen_deps:
            print("No valid packages selected. Exiting.")
            sys.exit(0)

        update_yaml_file(yaml_filepath, chosen_deps)

    except (ValueError, IndexError):
        print("Invalid input. Please enter numbers separated by commas.")
        sys.exit(1)

if __name__ == "__main__":
    main()