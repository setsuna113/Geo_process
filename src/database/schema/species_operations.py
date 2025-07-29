"""Species operations for database schema."""

from typing import Dict, Any, List, Optional
import json
import logging
from ..connection import db
from ..exceptions import handle_database_error, safe_fetch_id

logger = logging.getLogger(__name__)


class SpeciesOperations:
    """Species-related database operations."""
    
    @handle_database_error("store_species_range")
    def store_species_range(self, species_data: Dict[str, Any]) -> str:
        """Store species range data from .gpkg file."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO species_ranges 
                (species_name, scientific_name, genus, family, order_name, class_name, 
                 phylum, kingdom, category, range_type, geometry, source_file, 
                 source_dataset, confidence, area_km2, metadata)
                VALUES (%(species_name)s, %(scientific_name)s, %(genus)s, %(family)s,
                        %(order_name)s, %(class_name)s, %(phylum)s, %(kingdom)s,
                        %(category)s, %(range_type)s, ST_GeomFromText(%(geometry_wkt)s, 4326),
                        %(source_file)s, %(source_dataset)s, %(confidence)s, 
                        %(area_km2)s, %(metadata)s)
                RETURNING id
            """, {
                'species_name': species_data['species_name'],
                'scientific_name': species_data.get('scientific_name', ''),
                'genus': species_data.get('genus', ''),
                'family': species_data.get('family', ''),
                'order_name': species_data.get('order_name', ''),
                'class_name': species_data.get('class_name', ''),
                'phylum': species_data.get('phylum', ''),
                'kingdom': species_data.get('kingdom', ''),
                'category': species_data.get('category', 'unknown'),
                'range_type': species_data.get('range_type', 'distribution'),
                'geometry_wkt': species_data['geometry_wkt'],
                'source_file': species_data['source_file'],
                'source_dataset': species_data.get('source_dataset', ''),
                'confidence': species_data.get('confidence', 1.0),
                'area_km2': species_data.get('area_km2'),
                'metadata': json.dumps(species_data.get('metadata', {}))
            })
            range_id = safe_fetch_id(cursor, "store_species_range")
            logger.debug(f"✅ Stored species range: {species_data['species_name']} ({range_id})")
            return range_id
    
    @handle_database_error("store_species_intersections_batch")
    def store_species_intersections_batch(self, intersections: List[Dict]) -> int:
        """Bulk store species-grid intersections."""
        with db.get_cursor() as cursor:
            intersection_records = []
            for inter in intersections:
                intersection_records.append((
                    inter['grid_id'],
                    inter['cell_id'],
                    inter['species_range_id'],
                    inter['species_name'],
                    inter['category'],
                    inter['range_type'],
                    inter.get('intersection_area_km2'),
                    inter.get('coverage_percent'),
                    inter.get('presence_score', 1.0),
                    json.dumps(inter.get('computation_metadata', {}))
                ))
            
            cursor.executemany("""
                INSERT INTO species_grid_intersections 
                (grid_id, cell_id, species_range_id, species_name, category, range_type,
                 intersection_area_km2, coverage_percent, presence_score, computation_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, species_range_id) 
                DO UPDATE SET
                    intersection_area_km2 = EXCLUDED.intersection_area_km2,
                    coverage_percent = EXCLUDED.coverage_percent,
                    presence_score = EXCLUDED.presence_score,
                    computation_metadata = EXCLUDED.computation_metadata,
                    computed_at = CURRENT_TIMESTAMP
            """, intersection_records)
            
            inserted_count = cursor.rowcount
            logger.info(f"✅ Stored {inserted_count} species-grid intersections")
            return inserted_count
    
    def get_species_ranges(self, category: Optional[str] = None, 
                          source_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get species ranges with optional filtering."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM species_ranges WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = %s"
                params.append(category)
            
            if source_file:
                query += " AND source_file = %s"
                params.append(source_file)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_species_richness(self, grid_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get species richness summary."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM species_richness_summary WHERE grid_id = %s"
            params = [grid_id]
            
            if category:
                query += " AND category = %s"
                params.append(category)
            
            query += " ORDER BY cell_id, category"
            cursor.execute(query, params)
            return cursor.fetchall()