# src/pipelines/stages/export_stage.py
"""Export stage for pipeline results."""

from typing import List, Tuple
import logging
from datetime import datetime

from .base_stage import PipelineStage, StageResult
from src.processors.exporters.csv_exporter import CSVExporter, ExportConfig

logger = logging.getLogger(__name__)


class ExportStage(PipelineStage):
    """Stage for exporting merged data to CSV."""
    
    @property
    def name(self) -> str:
        return "export"
    
    @property
    def dependencies(self) -> List[str]:
        # Export happens after merge, BEFORE analysis
        return ["merge"]
    
    @property
    def memory_requirements(self) -> float:
        return 2.0  # GB - streaming export, low memory
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate export configuration."""
        return True, []
    
    def execute(self, context) -> StageResult:
        """Export merged dataset to CSV."""
        logger.info("Starting export stage")
        
        try:
            # Get merged dataset from context
            merged_dataset = context.get('merged_dataset')
            if merged_dataset is None:
                raise RuntimeError("No merged dataset found in context")
            
            # Determine output path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"merged_data_{context.experiment_id}_{timestamp}.csv"
            output_path = context.output_dir / output_filename
            
            # Configure export
            compression = 'gzip' if context.config.get('export.compress', False) else None
            if compression:
                output_path = output_path.with_suffix('.csv.gz')
            
            # Export xarray dataset directly to CSV
            # First, convert to pandas DataFrame
            df = merged_dataset.to_dataframe()
            
            # Reset index to get coordinate columns
            df = df.reset_index()
            
            # Write to CSV with optional compression
            chunk_size = context.config.get('export.chunk_size', 10000)
            rows_exported = 0
            
            if compression == 'gzip':
                import gzip
                with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                    # Write header
                    df.head(0).to_csv(f, index=False)
                    # Write in chunks
                    for i in range(0, len(df), chunk_size):
                        chunk = df.iloc[i:i+chunk_size]
                        chunk.to_csv(f, index=False, header=False)
                        rows_exported += len(chunk)
                        logger.info(f"Export progress: {rows_exported:,} rows")
            else:
                # Write header
                df.head(0).to_csv(output_path, index=False)
                # Write in chunks
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk.to_csv(output_path, index=False, header=False, mode='a')
                    rows_exported += len(chunk)
                    logger.info(f"Export progress: {rows_exported:,} rows")
            
            # Validate export
            if not output_path.exists():
                raise RuntimeError("Export failed - output file not created")
            
            file_size = output_path.stat().st_size
            if file_size == 0:
                raise RuntimeError("Export failed - output file is empty")
            
            # Store exported file path in context for analysis stage
            context.set('exported_csv_path', str(output_path))
            
            # Create metadata if requested
            if context.config.get('export.include_metadata', True):
                metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'output_file': str(output_path),
                    'rows_exported': rows_exported,
                    'compression': compression,
                    'experiment_id': context.experiment_id,
                    'shape': list(merged_dataset.dims.values()),
                    'variables': list(merged_dataset.data_vars)
                }
                
                metadata_file = output_path.with_suffix('.meta.json')
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadata saved to: {metadata_file}")
            
            metrics = {
                'rows_exported': rows_exported,
                'chunks_processed': (rows_exported + chunk_size - 1) // chunk_size,
                'duration_seconds': 0,  # Could track this if needed
                'file_size_mb': file_size / (1024**2),
                'output_path': str(output_path)
            }
            
            return StageResult(
                success=True,
                data={'exported_file': str(output_path)},
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Export stage failed: {e}")
            raise