# src/pipelines/stages/export_stage.py (Updated)
"""Export stage for pipeline results."""

from typing import List, Tuple
import logging
from pathlib import Path

from .base_stage import PipelineStage, StageResult
from src.processors.exporters.csv_exporter import CSVExporter, ExportConfig

logger = logging.getLogger(__name__)


class ExportStage(PipelineStage):
    """Stage for exporting results to CSV."""
    
    @property
    def name(self) -> str:
        return "export"
    
    @property
    def dependencies(self) -> List[str]:
        return ["merge", "analysis"]  # After merge and analysis
    
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
            # Create exporter
            exporter = CSVExporter(context.db)
            
            # Determine output path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"resampled_data_{context.experiment_id}_{timestamp}.csv"
            output_path = context.output_dir / output_filename
            
            # Configure export
            config = ExportConfig(
                output_path=output_path,
                chunk_size=context.config.get('export.chunk_size', 10000),
                include_metadata=True,
                compression='gzip' if context.config.get('export.compress', False) else None
            )
            
            # Export data
            exported_file = exporter.export_merged_dataset(
                experiment_id=context.experiment_id,
                config=config,
                progress_callback=lambda p: logger.info(
                    f"Export progress: {p['rows_exported']:,} rows"
                )
            )
            
            # Validate export
            if not exporter.validate_export(exported_file):
                raise RuntimeError("Export validation failed")
            
            # Get export statistics
            stats = exporter.get_export_stats()
            
            metrics = {
                'rows_exported': stats['rows_exported'],
                'chunks_processed': stats['chunks_processed'],
                'duration_seconds': stats.get('duration_seconds', 0),
                'file_size_mb': exported_file.stat().st_size / (1024**2),
                'output_path': str(exported_file)
            }
            
            # Also export analysis results summary
            self._export_analysis_summary(context, exporter)
            
            return StageResult(
                success=True,
                data={'exported_file': str(exported_file)},
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Export stage failed: {e}")
            raise
    
    def _export_analysis_summary(self, context, exporter):
        """Export analysis results summary."""
        som_results = context.get('som_results')
        if not som_results:
            return
        
        # Create summary CSV with analysis results
        summary_path = context.output_dir / f"analysis_summary_{context.experiment_id}.csv"
        
        # This could be extended to export cluster assignments, etc.
        logger.info(f"Analysis summary would be exported to: {summary_path}")