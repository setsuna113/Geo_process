#!/usr/bin/env python3
"""Recommend best data format for ML tasks (SOM, PCA, clustering)."""

def analyze_format_options():
    """Compare different formats for ML pipeline."""
    
    print("=== DATA FORMAT RECOMMENDATION FOR ML PIPELINE ===\n")
    
    print("REQUIREMENTS:")
    print("- SOM (Self-Organizing Maps)")
    print("- PCA (Principal Component Analysis)")  
    print("- Clustering analysis")
    print("- Maximum compatibility")
    print("- Efficient parsing")
    print("- No compression needed (plenty of space)\n")
    
    print("FORMAT COMPARISON:\n")
    
    formats = {
        "1. Apache Parquet": {
            "pros": [
                "Columnar format - extremely fast for ML (read only needed columns)",
                "Native support in pandas, dask, spark, arrow",
                "Preserves data types perfectly (int, float, categories)",
                "Supports metadata and schema",
                "Memory-mapped reading for large datasets",
                "Can handle NULL values efficiently"
            ],
            "cons": [
                "Binary format (not human readable)",
                "Requires parquet library"
            ],
            "compatibility": "Excellent - all major ML frameworks",
            "parsing_speed": "10/10",
            "recommendation": "⭐⭐⭐⭐⭐"
        },
        
        "2. HDF5": {
            "pros": [
                "Hierarchical structure (can store multiple datasets)",
                "Native numpy/scipy support",
                "Extremely fast for numerical arrays",
                "Supports chunking and partial reads",
                "Can store metadata"
            ],
            "cons": [
                "Can be complex for simple tabular data",
                "Not columnar (loads full rows)"
            ],
            "compatibility": "Good - scientific Python stack",
            "parsing_speed": "9/10",
            "recommendation": "⭐⭐⭐⭐"
        },
        
        "3. NumPy NPZ": {
            "pros": [
                "Native numpy format",
                "Direct array loading",
                "Can store multiple arrays",
                "Simple and fast"
            ],
            "cons": [
                "No built-in coordinate information",
                "Limited metadata support",
                "Memory inefficient for sparse data"
            ],
            "compatibility": "Good - Python scientific stack",
            "parsing_speed": "8/10",
            "recommendation": "⭐⭐⭐"
        },
        
        "4. CSV": {
            "pros": [
                "Universal compatibility",
                "Human readable",
                "Simple structure"
            ],
            "cons": [
                "Slow parsing for large files",
                "No data type preservation",
                "Inefficient for numerical data",
                "No metadata support"
            ],
            "compatibility": "Universal",
            "parsing_speed": "3/10",
            "recommendation": "⭐⭐"
        },
        
        "5. NetCDF": {
            "pros": [
                "Self-describing format",
                "Excellent for gridded spatial data",
                "Preserves coordinates",
                "Good compression options"
            ],
            "cons": [
                "Overkill for simple ML tasks",
                "Requires specific libraries",
                "Not optimized for row/column access"
            ],
            "compatibility": "Good - scientific community",
            "parsing_speed": "6/10",
            "recommendation": "⭐⭐⭐"
        }
    }
    
    for format_name, details in formats.items():
        print(f"{format_name} {details['recommendation']}")
        print(f"  Compatibility: {details['compatibility']}")
        print(f"  Parsing speed: {details['parsing_speed']}")
        print("  Pros:")
        for pro in details['pros']:
            print(f"    + {pro}")
        print("  Cons:")
        for con in details['cons']:
            print(f"    - {con}")
        print()
    
    print("\n🏆 RECOMMENDATION: Apache Parquet\n")
    
    print("REASONS:")
    print("1. Fastest parsing for ML workflows (columnar storage)")
    print("2. Native integration with scikit-learn, pandas, dask")
    print("3. Preserves data types (no float->string->float conversions)")
    print("4. Efficient NULL handling for sparse biodiversity data")
    print("5. Can read subset of columns (e.g., only species of interest)")
    print("6. Memory-mapped reading (no need to load full dataset)")
    
    print("\nIMPLEMENTATION:")
    print("```python")
    print("# Save")
    print("df.to_parquet('biodiversity_data.parquet', engine='pyarrow')")
    print("")
    print("# Load for ML")
    print("df = pd.read_parquet('biodiversity_data.parquet')")
    print("# Or load specific columns")
    print("df = pd.read_parquet('biodiversity_data.parquet', columns=['terrestrial_richness'])")
    print("```")
    
    print("\nFOR OUR PIPELINE:")
    print("- Keep NetCDF for spatial operations (preserves grid metadata)")
    print("- Export to Parquet for ML tasks (optimal performance)")
    print("- Skip CSV entirely (unless needed for external tools)")

if __name__ == "__main__":
    analyze_format_options()