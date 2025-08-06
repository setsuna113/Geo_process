#!/usr/bin/env python
"""Estimate coverage when combining fungi with plant/animal data."""

import numpy as np

# From the fungi analysis
fungi_coverage = 0.222  # 22.2%

# Typical coverage for other biodiversity data (estimates based on common patterns)
# Plants are usually better sampled
plants_coverage = 0.60  # Estimate: 60% coverage

# Vertebrates intermediate
vertebrates_coverage = 0.40  # Estimate: 40% coverage

# If coverage were independent (worst case)
independent_coverage = fungi_coverage * plants_coverage * vertebrates_coverage
print(f"If coverage were independent: {independent_coverage*100:.1f}%")

# If coverage were perfectly overlapping (best case)
perfect_overlap = min(fungi_coverage, plants_coverage, vertebrates_coverage)
print(f"If coverage perfectly overlaps: {perfect_overlap*100:.1f}%")

# Realistic estimate (partial overlap - well-studied areas)
# Assume 70% correlation in coverage patterns
correlation = 0.7
realistic_coverage = fungi_coverage * (1 + correlation * (min(plants_coverage/fungi_coverage, 1) - 1))
print(f"\nRealistic estimate (70% correlation): {realistic_coverage*100:.1f}%")

# By region estimates
print("\nExpected complete case coverage by region:")
print("  Temperate (30-60°N): ~15-20% (best case)")
print("  Tropics (30°S-30°N): ~5-10% (fungi limits this)")
print("  High latitudes: <1% (very sparse)")
print("  Global average: ~8-12%")

print("\nSample size estimates:")
print("  Total 1km pixels globally: ~150 million")
print("  With fungi data (22%): ~33 million")
print("  Complete cases (10%): ~15 million")
print("  After aggregation to 0.5°: ~60,000 cells")
print("  After aggregation to 1.0°: ~15,000 cells")