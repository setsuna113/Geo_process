# src/raster/catalog.py
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from src.database.connection import DatabaseManager
from src.database.utils import DatabaseSchemaUtils
from .loaders.geotiff_loader import GeoTIFFLoader
from .loaders.metadata_extractor import RasterMetadataExtractor
from .validators.coverage_validator import CoverageValidator
from .validators.value_validator import ValueValidator
from .loaders.base_loader import BaseRasterLoader
from src.core.registry import Registry
from src.config import config

logger = logging.getLogger(__name__)

@dataclass
class RasterEntry:
    """Entry in the raster catalog."""
    id: int
    name: str
    path: Path
    dataset_type: str
    resolution_degrees: float
    bounds: Tuple[float, float, float, float]
    data_type: str
    nodata_value: Optional[float]
    file_size_mb: float
    last_validated: Optional[datetime]
    is_active: bool
    metadata: Dict[str, Any]

class RasterCatalog:
    """Manage catalog of available raster data sources."""
    
    def __init__(self, db_connection: DatabaseManager, config: Config):
        self.db = db_connection
        self.config = config
        self.registry = Registry("raster_catalog")
        self._loader_cache = {}
        
        # Initialize components
        self.metadata_extractor = RasterMetadataExtractor(db_connection)
        self.schema_utils = DatabaseSchemaUtils(db_connection, config)
        
    def scan_directory(self, directory: Path, 
                      pattern: str = "*.tif",
                      validate: bool = True) -> List[RasterEntry]:
        """Scan directory for raster files and add to catalog."""
        discovered = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    entry = self.add_raster(file_path, validate=validate)
                    discovered.append(entry)
                    logger.info(f"Added to catalog: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to add {file_path.name}: {e}")
        
        logger.info(f"Discovered {len(discovered)} raster files")
        return discovered
    
    def add_raster(self, file_path: Path, 
                   dataset_type: Optional[str] = None,
                   validate: bool = True) -> RasterEntry:
        """Add a raster to the catalog."""
        # Get appropriate loader
        loader = self._get_loader(file_path)
        
        # Extract metadata
        full_metadata = self.metadata_extractor.extract_full_metadata(file_path)
        
        # Detect dataset type if not provided
        if dataset_type is None:
            dataset_type = self._detect_dataset_type(file_path)
        
        # Validate if requested
        validation_results = {}
        if validate:
            coverage_validator = CoverageValidator(loader)
            value_validator = ValueValidator(loader)
            
            validation_results = {
                'values': value_validator.validate_values(file_path, dataset_type),
                'timestamp': datetime.now().isoformat()
            }
        
        # Store in database
        raster_id = self.metadata_extractor.store_in_database(
            full_metadata, 
            file_path.stem
        )
        
        # Create catalog entry
        entry = RasterEntry(
            id=raster_id,
            name=file_path.stem,
            path=file_path,
            dataset_type=dataset_type,
            resolution_degrees=full_metadata['spatial_info']['pixel_size']['x'],
            bounds=tuple(full_metadata['spatial_info']['extent'].values()),
            data_type=full_metadata['data_info']['bands'][0]['data_type'],
            nodata_value=full_metadata['data_info']['bands'][0]['nodata_value'],
            file_size_mb=full_metadata['file_info']['size_mb'],
            last_validated=datetime.now() if validate else None,
            is_active=True,
            metadata={**full_metadata, 'validation': validation_results}
        )
        
        return entry
    
    def get_raster(self, name: str) -> Optional[RasterEntry]:
        """Get raster entry by name."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # Build query using schema utils
            columns = ['id', 'name', 'file_path', 'pixel_size_degrees', 'data_type', 
                      'nodata_value', 'file_size_mb', 'active', 'metadata']
            
            query, params = self.schema_utils.build_select_query(
                'raster_sources', 
                columns,
                include_geometry_bounds=True,
                where_conditions=['name = %s'],
                include_active_filter=True
            )
            
            cur.execute(query, [name] + params)
            
            row = cur.fetchone()
            if not row:
                return None
            
            return RasterEntry(
                id=row[0],
                name=row[1],
                path=Path(row[2]),
                dataset_type=self._detect_dataset_type(Path(row[2])),
                resolution_degrees=row[3],
                bounds=(row[9], row[10], row[11], row[12]),  # min_x, min_y, max_x, max_y from bounds
                data_type=row[4],
                nodata_value=row[5],
                file_size_mb=row[6],
                last_validated=None,  # TODO: Add to DB schema
                is_active=row[7],
                metadata=row[8] or {}
            )
    
    def list_rasters(self, dataset_type: Optional[str] = None, 
                    active_only: bool = True) -> List[RasterEntry]:
        """List all rasters in catalog."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # Build query using schema utils
            columns = ['id', 'name', 'file_path', 'pixel_size_degrees', 'data_type', 
                      'nodata_value', 'file_size_mb', 'active', 'metadata']
            
            where_conditions = []
            params = []
            
            if dataset_type:
                metadata_col = self.schema_utils.get_metadata_column('raster_sources') or 'metadata'
                where_conditions.append(f"{metadata_col}->>'dataset_type' = %s")
                params.append(dataset_type)
            
            query, base_params = self.schema_utils.build_select_query(
                'raster_sources',
                columns,
                include_geometry_bounds=True,
                where_conditions=where_conditions,
                include_active_filter=active_only
            )
            
            query += " ORDER BY name"
            cur.execute(query, params + base_params)
            
            entries = []
            for row in cur.fetchall():
                entries.append(RasterEntry(
                    id=row[0],
                    name=row[1],
                    path=Path(row[2]),
                    dataset_type=self._detect_dataset_type(Path(row[2])),
                    resolution_degrees=row[3],
                    bounds=(row[9], row[10], row[11], row[12]),
                    data_type=row[4],
                    nodata_value=row[5],
                    file_size_mb=row[6],
                    last_validated=None,
                    is_active=row[7],
                    metadata=row[8] or {}
                ))
            
            return entries
    
    def validate_catalog(self, fix_issues: bool = False) -> Dict[str, Any]:
        """Validate all entries in catalog."""
        results = {
            'total': 0,
            'valid': 0,
            'issues': [],
            'fixed': 0
        }
        
        entries = self.list_rasters()
        results['total'] = len(entries)
        
        for entry in entries:
            # Check file exists
            if not entry.path.exists():
                issue = {
                    'raster': entry.name,
                    'type': 'missing_file',
                    'description': f'File not found: {entry.path}'
                }
                results['issues'].append(issue)
                
                if fix_issues:
                    self.deactivate_raster(entry.id)
                    results['fixed'] += 1
                continue
            
            # Validate values
            try:
                loader = self._get_loader(entry.path)
                validator = ValueValidator(loader)
                validation = validator.validate_values(entry.path, entry.dataset_type)
                
                if not validation['validation']['valid']:
                    issue = {
                        'raster': entry.name,
                        'type': 'invalid_values',
                        'description': validation['validation']['issues']
                    }
                    results['issues'].append(issue)
                else:
                    results['valid'] += 1
                    
            except Exception as e:
                issue = {
                    'raster': entry.name,
                    'type': 'validation_error',
                    'description': str(e)
                }
                results['issues'].append(issue)
        
        return results
    
    def deactivate_raster(self, raster_id: int) -> None:
        """Deactivate a raster in the catalog."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE raster_sources SET active = false WHERE id = %s",
                (raster_id,)
            )
            conn.commit()
    
    def _get_loader(self, file_path: Path) -> GeoTIFFLoader:
        """Get appropriate loader for file."""
        # For now, assume GeoTIFF
        # In future, check file and return appropriate loader
        if 'geotiff' not in self._loader_cache:
            from src.config import config
            self._loader_cache['geotiff'] = GeoTIFFLoader(config)
        
        return self._loader_cache['geotiff']
    
    def _detect_dataset_type(self, file_path: Path) -> str:
        """Detect dataset type from filename."""
        name_lower = file_path.name.lower()
        
        if 'plant' in name_lower or 'daru' in name_lower:
            return 'plants'
        elif 'vertebrate' in name_lower or 'iucn' in name_lower:
            return 'vertebrates'
        elif 'terrestrial' in name_lower:
            return 'terrestrial'
        elif 'marine' in name_lower:
            return 'marine'
        else:
            return 'unknown'
    
    def generate_report(self, output_path: Path) -> None:
        """Generate catalog report."""
        entries = self.list_rasters()
        
        report = {
            'generated': datetime.now().isoformat(),
            'total_rasters': len(entries),
            'total_size_gb': sum(e.file_size_mb for e in entries) / 1024,
            'by_type': {},
            'rasters': []
        }
        
        # Group by type
        for entry in entries:
            dtype = entry.dataset_type
            if dtype not in report['by_type']:
                report['by_type'][dtype] = {'count': 0, 'size_mb': 0}
            
            report['by_type'][dtype]['count'] += 1
            report['by_type'][dtype]['size_mb'] += entry.file_size_mb
            
            report['rasters'].append({
                'name': entry.name,
                'type': entry.dataset_type,
                'resolution_m': entry.resolution_degrees * 111000,  # Approximate
                'bounds': entry.bounds,
                'size_mb': entry.file_size_mb,
                'validated': entry.last_validated is not None
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Catalog report written to {output_path}")
