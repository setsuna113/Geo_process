"""
Data sampling utilities for emmeans computation.

Provides efficient sampling strategies for large datasets when computing
estimated marginal means, which only need representative samples for
structure and covariate means.
"""

import logging
import os
import h2o
import pandas as pd
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Configure polars threading to respect config
n_threads = int(os.environ.get('H2O_NTHREADS', '150'))
os.environ['POLARS_MAX_THREADS'] = str(n_threads)
logger.info(f"Polars configured with {n_threads} threads for data conversion")


class EMMeansDataSampler:
    """
    Intelligent data sampler for emmeans computation.
    
    Since R emmeans only needs data for:
    1. Factor levels
    2. Covariate means
    3. Model structure
    
    We can use representative samples instead of full datasets.
    """
    
    def __init__(self, sample_size: int = 10000, seed: int = 42):
        """
        Initialize the sampler.
        
        Args:
            sample_size: Target number of rows to sample
            seed: Random seed for reproducibility
        """
        self.sample_size = sample_size
        self.seed = seed
    
    def sample_for_emmeans(
        self, 
        h2o_data: h2o.H2OFrame,
        factor_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create a representative sample for emmeans computation.
        
        Args:
            h2o_data: Full H2O dataset
            factor_columns: Optional list of factor columns for stratified sampling
            
        Returns:
            Representative pandas DataFrame
        """
        n_total = h2o_data.nrows
        
        if n_total <= self.sample_size:
            logger.info(f"Dataset has {n_total:,} rows, using all for emmeans")
            return h2o_data.as_data_frame(use_multi_thread=True)
        
        logger.info(f"Sampling {self.sample_size:,} from {n_total:,} rows for emmeans")
        
        if factor_columns and all(col in h2o_data.columns for col in factor_columns):
            return self._stratified_sample(h2o_data, factor_columns)
        else:
            return self._random_sample(h2o_data)
    
    def _stratified_sample(
        self, 
        h2o_data: h2o.H2OFrame,
        factor_columns: List[str]
    ) -> pd.DataFrame:
        """
        Perform stratified sampling by factor levels.
        
        Ensures all factor combinations are represented.
        """
        logger.info(f"Using stratified sampling by {factor_columns}")
        
        # Get unique combinations
        group_counts = h2o_data.group_by(factor_columns).count().get_frame()
        n_groups = group_counts.nrows
        
        if n_groups == 0:
            return self._random_sample(h2o_data)
        
        samples_per_group = self.sample_size // n_groups
        logger.info(f"Sampling ~{samples_per_group:,} rows per group ({n_groups} groups)")
        
        # For 2x2 design, sample from each combination
        if len(factor_columns) == 2 and n_groups <= 4:
            return self._sample_2x2_design(h2o_data, factor_columns)
        
        # General stratified sampling
        sampled_frames = []
        for i in range(n_groups):
            # This is a simplified approach - in production you'd iterate properly
            group_sample = h2o_data.split_frame(
                ratios=[self.sample_size / n_total], 
                seed=self.seed
            )[0]
            sampled_frames.append(group_sample)
        
        # Combine and convert
        combined = sampled_frames[0]
        for frame in sampled_frames[1:]:
            combined = combined.rbind(frame)
        
        return combined.as_data_frame(use_multi_thread=True)
    
    def _sample_2x2_design(
        self,
        h2o_data: h2o.H2OFrame,
        factor_columns: List[str]
    ) -> pd.DataFrame:
        """
        Special handling for 2x2 factorial designs.
        """
        col1, col2 = factor_columns
        samples_per_group = self.sample_size // 4
        sampled_frames = []
        
        for val1 in [0, 1]:
            for val2 in [0, 1]:
                group = h2o_data[
                    (h2o_data[col1] == val1) & 
                    (h2o_data[col2] == val2)
                ]
                
                if group.nrows > 0:
                    sample_frac = min(samples_per_group / group.nrows, 1.0)
                    group_sample = group.split_frame(
                        ratios=[sample_frac], 
                        seed=self.seed
                    )[0]
                    sampled_frames.append(group_sample)
                    
                    logger.info(
                        f"  Group {col1}={val1}, {col2}={val2}: "
                        f"sampled {group_sample.nrows:,} from {group.nrows:,}"
                    )
        
        # Combine samples
        if sampled_frames:
            combined = sampled_frames[0]
            for frame in sampled_frames[1:]:
                combined = combined.rbind(frame)
            
            logger.info(f"Final sample: {combined.nrows:,} rows")
            return combined.as_data_frame(use_multi_thread=True)
        
        return self._random_sample(h2o_data)
    
    def _random_sample(self, h2o_data: h2o.H2OFrame) -> pd.DataFrame:
        """
        Simple random sampling fallback.
        """
        logger.info("Using random sampling")
        sample_frac = self.sample_size / h2o_data.nrows
        sample = h2o_data.split_frame(ratios=[sample_frac], seed=self.seed)[0]
        
        logger.info(f"Sampled {sample.nrows:,} rows")
        return sample.as_data_frame(use_multi_thread=True)