# src/pipelines/monitors/quality_checker.py
"""Quality checking for pipeline outputs."""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
import numpy as np

from src.pipelines.stages.base_stage import StageResult, PipelineStage

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality check report."""
    stage_name: str
    checks_passed: int = 0
    checks_failed: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 1.0
    
    def has_critical_issues(self) -> bool:
        """Check if there are critical quality issues."""
        return len(self.errors) > 0 or self.overall_score < 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'stage': self.stage_name,
            'score': self.overall_score,
            'passed': self.checks_passed,
            'failed': self.checks_failed,
            'warnings': self.warnings,
            'errors': self.errors,
            'metrics': self.metrics
        }


class QualityChecker:
    """Check quality of pipeline outputs."""
    
    def __init__(self, config):
        self.config = config
        self.quality_thresholds = config.get('pipeline.quality_thresholds', {})
        
        # Default thresholds
        self.default_thresholds = {
            'min_completeness': 0.8,  # 80% data completeness
            'max_outlier_ratio': 0.05,  # 5% outliers
            'min_coverage': 0.9  # 90% spatial coverage
        }
    
    def check_stage_output(self, stage: PipelineStage, result: StageResult, 
                          context) -> QualityReport:
        """Check quality of stage output."""
        report = QualityReport(stage_name=stage.name)
        
        # Stage-specific checks
        if stage.name == "data_load":
            self._check_data_load_quality(result, report)
        elif stage.name == "resample":
            self._check_resample_quality(result, report, context)
        elif stage.name == "merge":
            self._check_merge_quality(result, report, context)
        elif stage.name == "analysis":
            self._check_analysis_quality(result, report, context)
        
        # Calculate overall score
        total_checks = report.checks_passed + report.checks_failed
        if total_checks > 0:
            report.overall_score = report.checks_passed / total_checks
        
        return report
    
    def check_stage_failure(self, stage: PipelineStage, error: Exception, context):
        """Analyze stage failure for quality issues."""
        # Log failure analysis
        logger.error(f"Analyzing failure for stage '{stage.name}': {error}")
        
        # Check for common quality-related failures
        error_str = str(error).lower()
        
        if "memory" in error_str:
            logger.warning("Failure appears to be memory-related")
        elif "corrupt" in error_str or "invalid" in error_str:
            logger.warning("Failure appears to be data quality related")
        elif "timeout" in error_str:
            logger.warning("Failure appears to be performance-related")
    
    def _check_data_load_quality(self, result: StageResult, report: QualityReport):
        """Check data loading quality."""
        datasets = result.data.get('datasets', [])
        
        # Check if any datasets were loaded
        if not datasets:
            report.errors.append("No datasets loaded")
            report.checks_failed += 1
        else:
            report.checks_passed += 1
        
        # Check dataset validity
        for dataset in datasets:
            if 'path' not in dataset:
                report.warnings.append(f"Dataset {dataset.get('name')} missing path")
            elif not Path(dataset['path']).exists():
                report.errors.append(f"Dataset file not found: {dataset['path']}")
                report.checks_failed += 1
            else:
                report.checks_passed += 1
        
        report.metrics['datasets_loaded'] = len(datasets)
    
    def _check_resample_quality(self, result: StageResult, report: QualityReport, context):
        """Check resampling quality."""
        resampled = result.data.get('resampled_datasets', [])
        
        if not resampled:
            report.errors.append("No datasets resampled")
            report.checks_failed += 1
            return
        
        # Check each resampled dataset
        for dataset_info in resampled:
            # Check resolution consistency
            target_res = context.config.get('resampling.target_resolution')
            if abs(dataset_info.target_resolution - target_res) > 0.0001:
                report.warnings.append(
                    f"Resolution mismatch for {dataset_info.name}: "
                    f"{dataset_info.target_resolution} vs {target_res}"
                )
                report.checks_failed += 1
            else:
                report.checks_passed += 1
            
            # Check data validity
            if dataset_info.shape[0] == 0 or dataset_info.shape[1] == 0:
                report.errors.append(f"Invalid shape for {dataset_info.name}: {dataset_info.shape}")
                report.checks_failed += 1
            else:
                report.checks_passed += 1
        
        report.metrics['datasets_resampled'] = len(resampled)
    
    def _check_merge_quality(self, result: StageResult, report: QualityReport, context):
        """Check merge quality."""
        merged_dataset = context.get('merged_dataset')
        
        if merged_dataset is None:
            report.errors.append("No merged dataset created")
            report.checks_failed += 1
            return
        
        # Check dataset structure
        if len(merged_dataset.data_vars) < 2:
            report.warnings.append("Merged dataset has less than 2 variables")
            report.checks_failed += 1
        else:
            report.checks_passed += 1
        
        # Check for NaN values
        for var_name in merged_dataset.data_vars:
            data = merged_dataset[var_name].values
            nan_ratio = np.isnan(data).sum() / data.size
            
            if nan_ratio > 0.1:  # More than 10% NaN
                report.warnings.append(f"High NaN ratio in {var_name}: {nan_ratio:.2%}")
                report.checks_failed += 1
            else:
                report.checks_passed += 1
            
            report.metrics[f'{var_name}_nan_ratio'] = nan_ratio
        
        # Check spatial coverage
        total_pixels = merged_dataset.sizes.get('x', 0) * merged_dataset.sizes.get('y', 0)
        report.metrics['total_pixels'] = total_pixels
    
    def _check_analysis_quality(self, result: StageResult, report: QualityReport, context):
        """Check analysis quality."""
        som_results = context.get('som_results')
        
        if som_results is None:
            report.errors.append("No analysis results")
            report.checks_failed += 1
            return
        
        # Check SOM quality metrics
        if hasattr(som_results, 'statistics'):
            stats = som_results.statistics
            
            # Check quantization error
            qe = stats.get('quantization_error', float('inf'))
            if qe > 1.0:
                report.warnings.append(f"High quantization error: {qe:.3f}")
                report.checks_failed += 1
            else:
                report.checks_passed += 1
            
            # Check topographic error
            te = stats.get('topographic_error', 1.0)
            if te > 0.1:
                report.warnings.append(f"High topographic error: {te:.3f}")
                report.checks_failed += 1
            else:
                report.checks_passed += 1
            
            report.metrics.update({
                'quantization_error': qe,
                'topographic_error': te
            })


# Import Path at module level
from pathlib import Path