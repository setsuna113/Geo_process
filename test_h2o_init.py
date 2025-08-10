#\!/usr/bin/env python
"""Test H2O initialization with new memory settings"""

import sys
sys.path.insert(0, '/home/jason/geo')

from src.machine_learning.utils.h2o_utils import ensure_h2o_initialized
import h2o

print("Testing H2O initialization with 4G memory...")
ensure_h2o_initialized(max_mem_size="4G", min_mem_size="1G")

# Test that it works
print("\nH2O cluster info:")
print(h2o.cluster())

# Clean shutdown
h2o.cluster().shutdown()
print("\nH2O test successful\!")
