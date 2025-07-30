# src/pipelines/stages/export_stage.py
"""Export stage for pipeline results."""

from typing import List, Tuple
from datetime import datetime
import time

from .base_stage import PipelineStage, StageResult
from src.processors.exporters.csv_exporter import CSVExporter, ExportConfig
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_stage

logger = get_logger(__name__)


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
    
    @log_stage("export")
    def execute(self, context) -> StageResult:
        """Export merged dataset to configured formats (CSV and/or Parquet)."""
        logger.info("Starting export stage")
        
        try:
            # Get merged dataset from context
            merged_dataset = context.get('merged_dataset')
            if merged_dataset is None:
                raise RuntimeError("No merged dataset found in context")
            
            # Determine export formats from config
            export_formats = context.config.get('export.formats', ['csv'])
            if isinstance(export_formats, str):
                export_formats = [export_formats]
            
            logger.info(f"Exporting to formats: {export_formats}")
            
            # Track exported files
            exported_files = {}
            total_rows_exported = 0
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export to each requested format
            for format in export_formats:
                if format == 'csv':
                    # CSV export (existing logic)
                    output_filename = f"merged_data_{context.experiment_id}_{timestamp}.csv"
                    output_path = context.output_dir / output_filename
                    
                    # Configure compression
                    compression = 'gzip' if context.config.get('export.compress', False) else None
                    if compression:
                        output_path = output_path.with_suffix('.csv.gz')
                    
                    # Track export time
                    csv_start_time = time.time()
            
                    # Export xarray dataset to CSV
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
                                logger.info(f"CSV export progress: {rows_exported:,} rows")
                    else:
                        # Write header
                        df.head(0).to_csv(output_path, index=False)
                        # Write in chunks
                        for i in range(0, len(df), chunk_size):
                            chunk = df.iloc[i:i+chunk_size]
                            chunk.to_csv(output_path, index=False, header=False, mode='a')
                            rows_exported += len(chunk)
                            logger.info(f"CSV export progress: {rows_exported:,} rows")
                    
                    # Validate export
                    if not output_path.exists():
                        raise RuntimeError("CSV export failed - output file not created")
                    
                    file_size = output_path.stat().st_size
                    if file_size == 0:
                        raise RuntimeError("CSV export failed - output file is empty")
                    
                    exported_files['csv'] = str(output_path)
                    total_rows_exported = rows_exported
                    
                    # Store CSV path in context for analysis stage
                    context.set('exported_csv_path', str(output_path))
                    
                    # Log performance
                    csv_duration = time.time() - csv_start_time
                    logger.log_performance(
                        "csv_export",
                        csv_duration,
                        rows=rows_exported,
                        size_mb=file_size / (1024**2),
                        format='csv'
                    )
                    logger.info(f"CSV export complete: {output_path}")
                
                elif format == 'parquet':
                    # Parquet export
                    output_filename = f"merged_data_{context.experiment_id}_{timestamp}.parquet"
                    output_path = context.output_dir / output_filename
                    
                    # Track export time
                    parquet_start_time = time.time()
                    
                    # Convert xarray to dataframe for parquet
                    df = merged_dataset.to_dataframe()
                    df = df.reset_index()
                    
                    # Write to parquet
                    df.to_parquet(output_path, index=False, compression='snappy')
                    
                    # Validate export
                    if not output_path.exists():
                        raise RuntimeError("Parquet export failed - output file not created")
                    
                    file_size = output_path.stat().st_size
                    if file_size == 0:
                        raise RuntimeError("Parquet export failed - output file is empty")
                    
                    exported_files['parquet'] = str(output_path)
                    total_rows_exported = len(df)
                    
                    # Store parquet path in context for analysis stage
                    context.set('ml_ready_path', str(output_path))
                    
                    # Log performance
                    parquet_duration = time.time() - parquet_start_time
                    logger.log_performance(
                        "parquet_export",
                        parquet_duration,
                        rows=total_rows_exported,
                        size_mb=file_size / (1024**2),
                        format='parquet'
                    )
                    logger.info(f"Parquet export complete: {output_path}")
                
                else:
                    logger.warning(f"Unknown export format: {format}, skipping")
            
            # Create metadata if requested
            if context.config.get('export.include_metadata', True):
                metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'exported_files': exported_files,
                    'rows_exported': total_rows_exported,
                    'experiment_id': context.experiment_id,
                    'shape': list(merged_dataset.dims.values()),
                    'variables': list(merged_dataset.data_vars),
                    'formats': export_formats
                }
                
                metadata_file = context.output_dir / f"export_metadata_{context.experiment_id}_{timestamp}.json"
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadata saved to: {metadata_file}")
            
            # Calculate total file sizes
            total_size_mb = 0
            for file_path in exported_files.values():
                total_size_mb += Path(file_path).stat().st_size / (1024**2)
            
            metrics = {
                'rows_exported': total_rows_exported,
                'formats_exported': len(exported_files),
                'total_file_size_mb': total_size_mb,
                'exported_files': exported_files
            }
            
            return StageResult(
                success=True,
                data={
                    'exported_files': exported_files,
                    'formats': list(exported_files.keys())
                },
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Export stage failed: {e}")
            raise