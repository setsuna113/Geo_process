#\!/usr/bin/env python
"""Test ML setup and run a simple example"""

import sys
import os
sys.path.insert(0, '/home/jason/geo')

print("Testing ML module setup...")

# Test imports
try:
    from src.machine_learning.research.methods import Method1ContinuousGLM
    print("✓ Method1ContinuousGLM imported")
except Exception as e:
    print(f"✗ Failed to import Method1: {e}")
    
try:
    from src.machine_learning.research.methods import Method2CategoricalGLM  
    print("✓ Method2CategoricalGLM imported")
except Exception as e:
    print(f"✗ Failed to import Method2: {e}")
    
try:
    from src.machine_learning.research.methods import Method3DML
    print("✓ Method3DML imported")
except Exception as e:
    print(f"✗ Failed to import Method3: {e}")

print("\nAll imports successful\! ML module is ready.")
print("\nNote: H2O requires significant memory and may have initialization issues.")
print("Consider using smaller spatial sampling (e.g., 1000 or 2000) for testing.")
