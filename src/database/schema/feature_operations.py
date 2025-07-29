"""Feature operations for database schema."""

from typing import Dict, Any, List, Optional
import json
import logging
from ..connection import db
from ..exceptions import handle_database_error, safe_fetch_id

logger = logging.getLogger(__name__)


class FeatureOperations:
    """Feature-related database operations."""
    
    @handle_database_error("store_feature")
    def store_feature(self, grid_id: str, cell_id: str, feature_type: str,
                     feature_name: str, feature_value: float,
                     metadata: Optional[Dict] = None) -> str:
        """Store computed feature."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO features 
                (grid_id, cell_id, feature_type, feature_name, feature_value, computation_metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, feature_type, feature_name)
                DO UPDATE SET
                    feature_value = EXCLUDED.feature_value,
                    computation_metadata = EXCLUDED.computation_metadata,
                    computed_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (grid_id, cell_id, feature_type, feature_name, feature_value,
                  json.dumps(metadata or {})))
            return safe_fetch_id(cursor, "store_feature")
    
    @handle_database_error("store_features_batch")
    def store_features_batch(self, features: List[Dict]) -> int:
        """Bulk store features."""
        with db.get_cursor() as cursor:
            feature_records = []
            for feat in features:
                feature_records.append((
                    feat['grid_id'],
                    feat['cell_id'],
                    feat['feature_type'],
                    feat['feature_name'],
                    feat['feature_value'],
                    json.dumps(feat.get('computation_metadata', {}))
                ))
            
            cursor.executemany("""
                INSERT INTO features 
                (grid_id, cell_id, feature_type, feature_name, feature_value, computation_metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, feature_type, feature_name)
                DO UPDATE SET
                    feature_value = EXCLUDED.feature_value,
                    computation_metadata = EXCLUDED.computation_metadata,
                    computed_at = CURRENT_TIMESTAMP
            """, feature_records)
            
            return cursor.rowcount
    
    @handle_database_error("store_climate_data_batch")
    def store_climate_data_batch(self, climate_data: List[Dict]) -> int:
        """Bulk store climate data."""
        with db.get_cursor() as cursor:
            climate_records = []
            for data in climate_data:
                climate_records.append((
                    data['grid_id'],
                    data['cell_id'],
                    data['variable'],
                    data['value'],
                    data.get('source'),
                    data.get('resolution')
                ))
            
            cursor.executemany("""
                INSERT INTO climate_data 
                (grid_id, cell_id, variable, value, source, resolution)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (grid_id, cell_id, variable, source, resolution)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    extracted_at = CURRENT_TIMESTAMP
            """, climate_records)
            
            return cursor.rowcount
    
    def get_features(self, grid_id: str, feature_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get features for a grid."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM features WHERE grid_id = %s"
            params = [grid_id]
            
            if feature_type:
                query += " AND feature_type = %s"
                params.append(feature_type)
            
            query += " ORDER BY cell_id, feature_type, feature_name"
            cursor.execute(query, params)
            return cursor.fetchall()