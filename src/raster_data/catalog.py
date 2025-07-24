# src/raster_data/catalog.py
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from src.database.connection import DatabaseManager
from src.config.config import Config
from src.raster_data.loaders.base_loader import BaseRasterLoader
from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader
from src.raster_data.loaders.metadata_extractor import RasterMetadataExtractor
from src.raster_data.loaders.lightweight_metadata import LightweightMetadataExtractor, extract_metadata_with_progress
from src.raster_data.validators.coverage_validator import CoverageValidator
from src.raster_data.validators.value_validator import ValueValidator
from src.core.registry import Registry

logger = logging.getLogger(__name__)

@dataclass
class RasterEntry:
    """Entry in the raster catalog."""
    id: str  # Changed from int to str to accommodate UUID
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
        self._loader_cache: Dict[str, Any] = {}
        
        # Initialize components
        self.metadata_extractor = RasterMetadataExtractor(db_connection)
        self.lightweight_extractor = LightweightMetadataExtractor(
            db_connection, 
            timeout_seconds=config.get('raster_processing.gdal_timeout', 30)
        )
        
    def scan_directory(self, directory: Path, 
                      pattern: str = "*.tif",
                      validate: bool = True,
                      lightweight: bool = True) -> List[RasterEntry]:
        """Scan directory for raster files and add to catalog."""
        discovered = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    if lightweight:
                        entry = self.add_raster_lightweight(file_path, validate=validate)
                    else:
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
            id=str(raster_id),  # Convert int to str
            name=file_path.stem,
            path=file_path,
            dataset_type=dataset_type,
            resolution_degrees=full_metadata['spatial_info']['pixel_size']['x'],
            bounds=(
                full_metadata['spatial_info']['extent']['west'],   # minx
                full_metadata['spatial_info']['extent']['south'],  # miny  
                full_metadata['spatial_info']['extent']['east'],   # maxx
                full_metadata['spatial_info']['extent']['north']   # maxy
            ),
            data_type=full_metadata['data_info']['bands'][0]['data_type'],
            nodata_value=full_metadata['data_info']['bands'][0]['nodata_value'],
            file_size_mb=full_metadata['file_info']['size_mb'],
            last_validated=datetime.now() if validate else None,
            is_active=True,
            metadata={**full_metadata, 'validation': validation_results}
        )
        
        return entry
    
    def add_raster_lightweight(self, file_path: Path, 
                              dataset_type: Optional[str] = None,
                              validate: bool = False,
                              progress_callback: Optional[callable] = None) -> RasterEntry:
        """Add a raster to the catalog using lightweight metadata extraction."""
        logger.info(f"Adding raster with lightweight extraction: {file_path.name}")
        
        # Use lightweight metadata extraction with timeout protection
        raster_id, lightweight_metadata = extract_metadata_with_progress(
            file_path, 
            self.db,
            timeout=self.config.get('raster_processing.gdal_timeout', 30),
            progress_callback=progress_callback
        )
        
        # Detect dataset type if not provided
        if dataset_type is None:
            dataset_type = self._detect_dataset_type(file_path)
        
        # Create catalog entry from lightweight metadata
        spatial_info = lightweight_metadata.get('spatial_info', {})
        data_info = lightweight_metadata.get('data_info', {})
        file_info = lightweight_metadata.get('file_info', {})
        
        # Handle missing spatial info gracefully
        bounds = spatial_info.get('bounds')
        if bounds is None:
            logger.warning(f"No spatial bounds available for {file_path.name}, using placeholder")
            bounds = [0, 0, 1, 1]  # Placeholder bounds
        
        entry = RasterEntry(
            id=str(raster_id),
            name=file_path.stem,
            path=file_path,
            dataset_type=dataset_type,
            resolution_degrees=spatial_info.get('resolution_degrees', 0.05),  # Default fallback
            bounds=tuple(bounds),
            data_type=data_info.get('data_type', 'unknown'),
            nodata_value=data_info.get('nodata_value'),
            file_size_mb=file_info.get('size_mb', 0),
            last_validated=datetime.now() if validate else None,
            is_active=True,
            metadata={
                **lightweight_metadata, 
                'extraction_mode': 'lightweight',
                'full_metadata_available': False
            }
        )
        
        logger.info(f"✅ Added raster with lightweight metadata: {file_path.name}")
        return entry
    
    def extract_full_metadata_async(self, raster_id: str, file_path: Path) -> bool:
        """Extract full metadata for a raster that was registered with lightweight extraction."""
        try:
            logger.info(f"Extracting full metadata for {file_path.name}")
            
            # Use the original full metadata extractor
            full_metadata = self.metadata_extractor.extract_full_metadata(file_path)
            
            # Update the database record with full metadata
            with self.db.get_cursor() as cursor:
                # Update raster_sources with full metadata
                cursor.execute("""
                    UPDATE raster_sources 
                    SET metadata = %s, processing_status = %s, updated_at = %s
                    WHERE id = %s
                """, (
                    json.dumps({**full_metadata, 'extraction_mode': 'full'}),
                    'registered_full',
                    datetime.now(),
                    raster_id
                ))
            
            logger.info(f"✅ Updated {file_path.name} with full metadata")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to extract full metadata for {file_path.name}: {e}")
            return False
    
    def get_raster(self, name: str) -> Optional[RasterEntry]:
        """Get raster entry by name."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, name, file_path, pixel_size_degrees, data_type, 
                       nodata_value, file_size_mb, active, metadata,
                       ST_XMin(spatial_extent), ST_YMin(spatial_extent), 
                       ST_XMax(spatial_extent), ST_YMax(spatial_extent)
                FROM raster_sources
                WHERE name = %s AND active = true
            """, (name,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            return RasterEntry(
                id=row[0],
                name=row[1],
                path=Path(row[2]),
                dataset_type=self._detect_dataset_type(Path(row[2])),
                resolution_degrees=row[3],
                bounds=(row[9], row[10], row[11], row[12]),
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
            
            query = """
                SELECT id, name, file_path, pixel_size_degrees, data_type, 
                       nodata_value, file_size_mb, active, metadata,
                       ST_XMin(spatial_extent), ST_YMin(spatial_extent), ST_XMax(spatial_extent), ST_YMax(spatial_extent)
                FROM raster_sources
                WHERE 1=1
            """
            params = []
            
            if active_only:
                query += " AND active = true"
            
            if dataset_type:
                query += " AND metadata->>'dataset_type' = %s"
                params.append(dataset_type)
            
            query += " ORDER BY name"
            
            cur.execute(query, params)
            
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
        results: Dict[str, Any] = {
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
    
    def deactivate_raster(self, raster_id: str) -> None:  # Changed from int to str
        """Deactivate a raster in the catalog."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE raster_sources SET active = false WHERE id = %s",
                (raster_id,)
            )
            conn.commit()
    
    def _get_loader(self, file_path: Path) -> BaseRasterLoader:
        """Get appropriate loader for file."""
        # For now, assume GeoTIFF
        # In future, check file and return appropriate loader
        if 'geotiff' not in self._loader_cache:
            self._loader_cache['geotiff'] = GeoTIFFLoader(self.config)
        
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
    
    def get_rasters_by_type(self, dataset_types: List[str]) -> Dict[str, RasterEntry]:
        """Get multiple rasters by dataset type."""
        results = {}
        for dtype in dataset_types:
            entries = self.list_rasters(dataset_type=dtype, active_only=True)
            if entries:
                # Return first active entry for each type
                results[dtype] = entries[0]
        return results

    def generate_report(self, output_path: Path) -> None:
        """Generate catalog report."""
        entries = self.list_rasters()
        
        report: Dict[str, Any] = {
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