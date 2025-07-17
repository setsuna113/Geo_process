"""
Export intermediate processing data (species_grid_intersections) to CSV
This is the "middle step" after processing but before final aggregation
"""

import csv
import sys
import logging
from datetime import datetime
from pathlib import Path
from .connection import db
from ..config import config

logger = logging.getLogger(__name__)

def get_output_path(filename_base: str) -> Path:
    """Get timestamped output path in data directory following config pattern."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_base}_{timestamp}.csv"
    
    # Use config to get data directory (defaults to PROJECT_ROOT/data)
    data_dir = config.get('paths.data_dir', Path.cwd() / 'data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory: {data_dir}")
    
    return data_dir / filename

def export_intersections_to_csv(limit=None):
    """
    Export the processed intersection data (middle step) to CSV.
    
    This contains:
    - Raw species range + grid cell intersections
    - Coverage percentages, areas, presence scores
    - Before aggregation into richness summaries
    """
    
    output_file = get_output_path("species_intersections")
    
    query = """
        SELECT 
            sgi.grid_id,
            sgi.cell_id,
            sgi.species_name,
            sgi.category,
            sgi.range_type,
            sgi.intersection_area_km2,
            sgi.coverage_percent,
            sgi.presence_score,
            sgi.computed_at,
            sr.source_file,
            g.name as grid_name,
            g.grid_type,
            g.resolution
        FROM species_grid_intersections sgi
        JOIN species_ranges sr ON sgi.species_range_id = sr.id
        JOIN grids g ON sgi.grid_id = g.id
        ORDER BY grid_id, cell_id, category, species_name
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    logger.info(f"Exporting intersection data to {output_file}...")
    
    with db.get_cursor() as cursor:
        cursor.execute(query)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'grid_id', 'cell_id', 'species_name', 'category', 
                'range_type', 'intersection_area_km2', 'coverage_percent',
                'presence_score', 'computed_at', 'source_file',
                'grid_name', 'grid_type', 'resolution'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            row_count = 0
            batch_size = config.get('processing.batch_size', 1000)
            
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                    
                for row in batch:
                    writer.writerow(dict(row))
                    row_count += 1
                
                if row_count % (batch_size * 10) == 0:
                    logger.info(f"Exported {row_count} records...")
    
    logger.info(f"✅ Exported {row_count} intersection records to {output_file}")
    return output_file, row_count

def export_richness_to_csv():
    """
    Export the final aggregated richness data to CSV.
    
    This is your final ML training data.
    """
    
    output_file = get_output_path("species_richness")
    
    query = """
        SELECT 
            srs.grid_id,
            srs.cell_id,
            srs.category,
            srs.range_type,
            srs.species_count,
            srs.avg_coverage,
            srs.total_intersection_area,
            srs.avg_presence_score,
            g.name as grid_name,
            g.resolution
        FROM species_richness_summary srs
        JOIN grids g ON srs.grid_id = g.id
        ORDER BY grid_id, cell_id, category
    """
    
    logger.info(f"Exporting richness summary to {output_file}...")
    
    with db.get_cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if rows:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in rows:
                    writer.writerow(dict(row))
        
        logger.info(f"✅ Exported {len(rows)} richness records to {output_file}")
        return output_file, len(rows)

def export_all_data(limit_intersections=None):
    """
    Export both intermediate and final data with timestamps.
    
    Args:
        limit_intersections: Optional limit for intersection records (useful for testing)
    
    Returns:
        dict with export information
    """
    
    logger.info("🗂️ Starting data export to timestamped CSV files...")
    
    results = {}
    
    try:
        # Export intermediate data
        intersections_file, intersections_count = export_intersections_to_csv(limit_intersections)
        results['intersections'] = {
            'file': intersections_file,
            'count': intersections_count
        }
        
        # Export final richness data
        richness_file, richness_count = export_richness_to_csv()
        results['richness'] = {
            'file': richness_file, 
            'count': richness_count
        }
        
        logger.info(f"📊 Data Export Summary:")
        logger.info(f"  Middle step (intersections): {intersections_count} records → {intersections_file}")
        logger.info(f"  Final step (richness): {richness_count} records → {richness_file}")
        logger.info(f"  Files saved to: {config.get('paths.data_dir', 'data/')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Export data with optional limit for testing
    limit = 1000 if len(sys.argv) > 1 and sys.argv[1] == '--test' else None
    
    try:
        results = export_all_data(limit_intersections=limit)
        print("✅ Export completed successfully!")
        
        for data_type, info in results.items():
            print(f"  {data_type}: {info['count']} records → {info['file']}")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)
