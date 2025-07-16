### **3. ATTRIBUTIONS.md**
```markdown
# Attributions and Acknowledgments

## Architecture Inspiration

This project's modular architecture and design patterns were inspired by:

- **GeoCore Framework** 
  - Repository: [link]
  - Paper: [citation]
  - License: [their license]
  - Specific inspirations:
    - Feature caching system design
    - Modular component registry pattern
    - SQL-based feature engineering approach

## Direct Code Adaptations

The following components include adapted code:

1. **Feature caching mechanism** (`src/biodiversity_pipeline/core/cache.py`)
   - Based on GeoCore's caching implementation
   - Modified to support multiple grid types
   - Original: [link to their file]

## Dependencies

This project builds upon these excellent open-source libraries:

- PostgreSQL & PostGIS - Spatial database
- GeoPandas - Spatial operations in Python  
- H3-py - Hexagonal grid system (Uber)
- Rasterio - Raster data processing
- [Full list in requirements.txt]