#!/usr/bin/env python3
"""Example usage of the unified configuration system.

This demonstrates how to use the new clean configuration system
while maintaining backwards compatibility with existing code.
"""

from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.machine_learning.config_loader import MLRegressionConfig
from src.machine_learning.config.unified_config import UnifiedMLConfig


def example_direct_unified_config():
    """Example of using UnifiedMLConfig directly for new code."""
    print("\n" + "=" * 60)
    print("Example 1: Direct UnifiedMLConfig Usage (New Code)")
    print("=" * 60)
    
    # Create config with defaults
    config = UnifiedMLConfig()
    
    # Access config values with type safety
    print(f"\nDefault configuration values:")
    print(f"  Preprocessing:")
    print(f"    - Missing data: {config.preprocessing.missing_data_strategy}")
    print(f"    - Standardization: {config.preprocessing.standardization_method}")
    print(f"  GLM:")
    print(f"    - Distribution: {config.glm.distribution}")
    print(f"    - Max iterations: {config.glm.max_iterations}")
    print(f"  Spatial:")
    print(f"    - MESF enabled: {config.spatial.mesf_enabled}")
    print(f"    - Max eigenvectors: {config.spatial.max_eigenvectors}")
    print(f"  Compute:")
    print(f"    - H2O memory: {config.compute.h2o_max_memory}")
    print(f"    - Parallel processing: {config.compute.use_parallel}")
    
    # Modify configuration
    print("\n\nModifying configuration:")
    config.preprocessing.missing_data_strategy = "imputation"
    config.glm.distribution = "poisson"
    config.compute.h2o_max_memory = "256G"
    
    print(f"  - Changed missing data to: {config.preprocessing.missing_data_strategy}")
    print(f"  - Changed GLM distribution to: {config.glm.distribution}")
    print(f"  - Changed H2O memory to: {config.compute.h2o_max_memory}")
    
    # Save to file
    output_path = Path("custom_config.yml")
    config.save_to_file(output_path)
    print(f"\n✓ Configuration saved to {output_path}")
    
    # Load from file
    config2 = UnifiedMLConfig(output_path)
    print(f"✓ Configuration loaded from {output_path}")
    print(f"  - GLM distribution: {config2.glm.distribution}")
    
    # Clean up
    output_path.unlink()
    print(f"✓ Cleaned up {output_path}")


def example_backwards_compatible():
    """Example of using unified config with backwards compatibility."""
    print("\n" + "=" * 60)
    print("Example 2: Backwards Compatible Usage (Existing Code)")
    print("=" * 60)
    
    # Load config using MLRegressionConfig with unified backend
    config = MLRegressionConfig(use_unified=True, validate=False)
    
    print("\nUsing unified config through legacy interface:")
    
    # All existing getter methods work
    prep_config = config.get_preprocessing_config()
    print(f"  Preprocessing config keys: {list(prep_config.keys())}")
    
    glm_config = config.get_glm_config()
    print(f"  GLM config keys: {list(glm_config.keys())}")
    
    spatial_config = config.get_spatial_config()
    print(f"  Spatial config keys: {list(spatial_config.keys())}")
    
    # New compute config also available
    compute_config = config.get_compute_config()
    print(f"  Compute config: {compute_config}")
    
    print("\n✓ All legacy methods work with unified backend!")


def example_gradual_migration():
    """Example showing gradual migration strategy."""
    print("\n" + "=" * 60)
    print("Example 3: Gradual Migration Strategy")
    print("=" * 60)
    
    print("\nMigration approach:")
    print("1. Start with legacy config (existing code works as-is)")
    print("2. Enable unified config with use_unified=True")
    print("3. Existing code continues to work via adapters")
    print("4. New code can use UnifiedMLConfig directly")
    print("5. Gradually migrate old code to use dataclasses")
    
    # Show both approaches side by side
    print("\nLegacy approach (complex nested dicts):")
    legacy_config = MLRegressionConfig(use_unified=False, validate=False)
    prep = legacy_config.get_preprocessing_config()
    print(f"  prep['missing_data']['strategy'] = {prep.get('missing_data', {}).get('strategy')}")
    
    print("\nUnified approach (clean dataclasses):")
    unified_config = MLRegressionConfig(use_unified=True, validate=False)
    if hasattr(unified_config, 'unified_config'):
        print(f"  config.preprocessing.missing_data_strategy = {unified_config.unified_config.preprocessing.missing_data_strategy}")
    
    print("\n✓ Both approaches work simultaneously!")


def example_programmatic_config():
    """Example of programmatic configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Programmatic Configuration")
    print("=" * 60)
    
    # Create config programmatically
    config = UnifiedMLConfig()
    
    # Update multiple values
    updates = {
        'preprocessing.missing_data_strategy': 'imputation',
        'preprocessing.missing_threshold': 0.2,
        'glm.max_iterations': 2000,
        'spatial.mesf_threshold': 0.3,
        'compute.h2o_max_memory': '512G'
    }
    
    print("\nUpdating configuration programmatically:")
    for key, value in updates.items():
        config.update(key, value)
        print(f"  Set {key} = {value}")
    
    # Retrieve values using dot notation
    print("\nRetrieving values:")
    print(f"  config.get('preprocessing.missing_data_strategy') = {config.get('preprocessing.missing_data_strategy')}")
    print(f"  config.get('glm.max_iterations') = {config.get('glm.max_iterations')}")
    print(f"  config.get('compute.h2o_max_memory') = {config.get('compute.h2o_max_memory')}")
    
    print("\n✓ Programmatic configuration complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Unified Configuration System Examples")
    print("=" * 60)
    print("\nThe unified configuration system provides:")
    print("  • Clean, typed dataclass structure")
    print("  • Full backwards compatibility")
    print("  • Type safety and IDE autocomplete")
    print("  • Easy programmatic manipulation")
    print("  • Gradual migration path")
    
    example_direct_unified_config()
    example_backwards_compatible()
    example_gradual_migration()
    example_programmatic_config()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()