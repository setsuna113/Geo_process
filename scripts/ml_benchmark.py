#!/usr/bin/env python
"""Performance benchmarking script for ML models."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
import time
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.machine_learning import (
    LinearRegressionAnalyzer,
    LightGBMAnalyzer,
    CompositeFeatureBuilder,
    SpatialKNNImputer,
    SpatialBlockCV
)
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLBenchmark:
    """Benchmark ML models for performance metrics."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def measure_memory_time(self, func, *args, **kwargs):
        """Measure memory usage and execution time of a function."""
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        return result, end_time - start_time, memory_used
    
    def benchmark_data_sizes(self, sizes: List[int], n_features: int = 20):
        """Benchmark models with different data sizes."""
        logger.info(f"Benchmarking with data sizes: {sizes}")
        
        for n_samples in sizes:
            logger.info(f"\nBenchmarking with {n_samples} samples...")
            
            # Generate data
            data = self.generate_synthetic_data(n_samples, n_features)
            
            # Prepare data
            X, y = self.prepare_data(data)
            
            # Benchmark each model
            models = {
                'linear_regression': LinearRegressionAnalyzer(config),
            }
            
            try:
                models['lightgbm'] = LightGBMAnalyzer(config)
            except ImportError:
                logger.warning("LightGBM not available")
            
            for model_name, model in models.items():
                logger.info(f"  Benchmarking {model_name}...")
                
                # Training time
                _, train_time, train_memory = self.measure_memory_time(
                    model.fit, X, y
                )
                
                # Prediction time (average over multiple runs)
                pred_times = []
                for _ in range(5):
                    _, pred_time, _ = self.measure_memory_time(
                        model.predict, X
                    )
                    pred_times.append(pred_time)
                
                avg_pred_time = np.mean(pred_times)
                
                # Model size
                import pickle
                model_size = len(pickle.dumps(model)) / 1024  # KB
                
                # Store results
                result = {
                    'model': model_name,
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'train_time': train_time,
                    'train_memory_mb': train_memory,
                    'predict_time': avg_pred_time,
                    'model_size_kb': model_size,
                    'throughput_samples_per_sec': n_samples / train_time
                }
                
                self.results.append(result)
                
                logger.info(f"    Train time: {train_time:.2f}s")
                logger.info(f"    Memory used: {train_memory:.2f}MB")
                logger.info(f"    Prediction time: {avg_pred_time*1000:.2f}ms")
                logger.info(f"    Model size: {model_size:.2f}KB")
    
    def benchmark_feature_dimensions(self, n_samples: int = 10000, 
                                    feature_counts: List[int] = None):
        """Benchmark models with different feature dimensions."""
        if feature_counts is None:
            feature_counts = [10, 50, 100, 500, 1000]
        
        logger.info(f"Benchmarking with feature counts: {feature_counts}")
        
        for n_features in feature_counts:
            logger.info(f"\nBenchmarking with {n_features} features...")
            
            # Generate data
            data = self.generate_synthetic_data(n_samples, n_features)
            X, y = self.prepare_data(data)
            
            # Benchmark linear regression (scales differently with features)
            model = LinearRegressionAnalyzer(config)
            
            _, train_time, train_memory = self.measure_memory_time(
                model.fit, X, y
            )
            
            result = {
                'model': 'linear_regression',
                'n_samples': n_samples,
                'n_features': n_features,
                'train_time': train_time,
                'train_memory_mb': train_memory,
                'features_per_second': n_features / train_time
            }
            
            self.results.append(result)
    
    def benchmark_cv_strategies(self, n_samples: int = 5000):
        """Benchmark different CV strategies."""
        logger.info("Benchmarking CV strategies...")
        
        # Generate spatial data
        data = self.generate_synthetic_data(n_samples, 20, spatial=True)
        X, y = self.prepare_data(data)
        
        model = LinearRegressionAnalyzer(config)
        
        cv_strategies = {
            'spatial_block_50': SpatialBlockCV(n_splits=5, block_size=50),
            'spatial_block_100': SpatialBlockCV(n_splits=5, block_size=100),
            'spatial_block_200': SpatialBlockCV(n_splits=5, block_size=200),
        }
        
        for cv_name, cv_strategy in cv_strategies.items():
            logger.info(f"\n  Benchmarking {cv_name}...")
            
            _, cv_time, cv_memory = self.measure_memory_time(
                model.cross_validate,
                X, y,
                cv_strategy=cv_strategy,
                metrics=['r2', 'rmse']
            )
            
            result = {
                'cv_strategy': cv_name,
                'n_samples': n_samples,
                'n_features': X.shape[1],
                'cv_time': cv_time,
                'cv_memory_mb': cv_memory,
                'time_per_fold': cv_time / 5
            }
            
            self.results.append(result)
    
    def benchmark_imputation(self, missing_rates: List[float] = None):
        """Benchmark imputation performance with different missing rates."""
        if missing_rates is None:
            missing_rates = [0.1, 0.2, 0.3, 0.5]
        
        logger.info(f"Benchmarking imputation with missing rates: {missing_rates}")
        
        n_samples = 5000
        base_data = self.generate_synthetic_data(n_samples, 10, spatial=True)
        
        for missing_rate in missing_rates:
            logger.info(f"\n  Missing rate: {missing_rate*100}%")
            
            # Add missing values
            data = base_data.copy()
            mask = np.random.random((n_samples, 8)) < missing_rate
            data.iloc[:, 2:10][mask] = np.nan
            
            imputer = SpatialKNNImputer(n_neighbors=10, spatial_weight=0.6)
            
            _, impute_time, impute_memory = self.measure_memory_time(
                imputer.fit_transform, data
            )
            
            result = {
                'operation': 'imputation',
                'missing_rate': missing_rate,
                'n_samples': n_samples,
                'impute_time': impute_time,
                'impute_memory_mb': impute_memory,
                'samples_per_second': n_samples / impute_time
            }
            
            self.results.append(result)
    
    def generate_synthetic_data(self, n_samples: int, n_features: int, 
                               spatial: bool = False) -> pd.DataFrame:
        """Generate synthetic data for benchmarking."""
        np.random.seed(42)
        
        if spatial:
            # Generate spatial data
            data = pd.DataFrame({
                'latitude': np.random.uniform(-60, 60, n_samples),
                'longitude': np.random.uniform(-180, 180, n_samples)
            })
            
            # Add feature columns
            for i in range(n_features - 2):
                data[f'feature_{i}'] = np.random.randn(n_samples)
        else:
            # Generate regular features
            data = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=[f'feature_{i}' for i in range(n_features)]
            )
        
        # Add richness columns for feature engineering
        data['plants_richness'] = np.random.poisson(50, n_samples)
        data['animals_richness'] = np.random.poisson(30, n_samples)
        
        return data
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for modeling."""
        # Simple target generation
        feature_cols = [col for col in data.columns if col.startswith('feature_')]
        if feature_cols:
            X = data[feature_cols]
            y = pd.Series(np.random.randn(len(data)))
        else:
            # Use all numeric columns except richness
            X = data.select_dtypes(include=[np.number])
            X = X.drop(columns=[col for col in X.columns if 'richness' in col], errors='ignore')
            y = pd.Series(np.random.randn(len(data)))
        
        return X, y
    
    def plot_results(self):
        """Create performance plots."""
        df = pd.DataFrame(self.results)
        
        # Plot 1: Training time vs data size
        if 'n_samples' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Training time
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                ax1.plot(model_df['n_samples'], model_df['train_time'], 
                        marker='o', label=model)
            
            ax1.set_xlabel('Number of Samples')
            ax1.set_ylabel('Training Time (seconds)')
            ax1.set_title('Training Time Scaling')
            ax1.legend()
            ax1.grid(True)
            
            # Memory usage
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                ax2.plot(model_df['n_samples'], model_df['train_memory_mb'], 
                        marker='o', label=model)
            
            ax2.set_xlabel('Number of Samples')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Memory Usage Scaling')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'scaling_performance.png', dpi=150)
            plt.close()
        
        # Plot 2: CV strategy comparison
        cv_df = df[df.get('cv_strategy') is not None]
        if not cv_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            cv_strategies = cv_df['cv_strategy'].values
            cv_times = cv_df['cv_time'].values
            
            bars = ax.bar(range(len(cv_strategies)), cv_times)
            ax.set_xticks(range(len(cv_strategies)))
            ax.set_xticklabels(cv_strategies, rotation=45)
            ax.set_ylabel('CV Time (seconds)')
            ax.set_title('Cross-Validation Strategy Performance')
            
            # Add value labels on bars
            for bar, time in zip(bars, cv_times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{time:.1f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'cv_performance.png', dpi=150)
            plt.close()
        
        # Plot 3: Imputation performance
        impute_df = df[df.get('operation') == 'imputation']
        if not impute_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(impute_df['missing_rate'] * 100, 
                   impute_df['impute_time'], 
                   marker='o', linewidth=2)
            
            ax.set_xlabel('Missing Data Rate (%)')
            ax.set_ylabel('Imputation Time (seconds)')
            ax.set_title('Imputation Performance vs Missing Data Rate')
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'imputation_performance.png', dpi=150)
            plt.close()
    
    def save_results(self):
        """Save benchmark results."""
        # Save raw results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.output_dir / 'benchmark_results.csv', index=False)
        
        # Save summary report
        report = self.generate_report()
        with open(self.output_dir / 'benchmark_report.md', 'w') as f:
            f.write(report)
        
        # Save JSON for programmatic access
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate benchmark report."""
        df = pd.DataFrame(self.results)
        
        report = []
        report.append("# ML Module Performance Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Model comparison
        if 'model' in df.columns:
            report.append("\n## Model Performance Comparison")
            
            model_summary = df.groupby('model').agg({
                'train_time': ['mean', 'std'],
                'train_memory_mb': ['mean', 'std'],
                'throughput_samples_per_sec': 'mean'
            }).round(2)
            
            report.append("\n### Training Performance")
            report.append(f"\n{model_summary.to_string()}")
        
        # Scaling analysis
        if 'n_samples' in df.columns:
            report.append("\n## Scaling Analysis")
            
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                if len(model_df) > 1:
                    # Simple linear regression for scaling
                    from scipy import stats
                    slope, intercept, r_value, _, _ = stats.linregress(
                        model_df['n_samples'], 
                        model_df['train_time']
                    )
                    
                    report.append(f"\n### {model}")
                    report.append(f"- Time complexity: ~O(n^{np.log(slope):.2f})")
                    report.append(f"- R²: {r_value**2:.3f}")
        
        # CV strategy comparison
        cv_df = df[df.get('cv_strategy') is not None]
        if not cv_df.empty:
            report.append("\n## Cross-Validation Performance")
            
            for _, row in cv_df.iterrows():
                report.append(f"\n### {row['cv_strategy']}")
                report.append(f"- Total time: {row['cv_time']:.2f}s")
                report.append(f"- Time per fold: {row['time_per_fold']:.2f}s")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        if 'model' in df.columns:
            fastest_model = df.groupby('model')['train_time'].mean().idxmin()
            report.append(f"\n1. **Fastest model**: {fastest_model}")
            
            most_memory_efficient = df.groupby('model')['train_memory_mb'].mean().idxmin()
            report.append(f"2. **Most memory efficient**: {most_memory_efficient}")
        
        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='Benchmark ML module performance')
    parser.add_argument('--output-dir', type=Path, default=Path('ml_benchmarks'),
                        help='Output directory for results')
    parser.add_argument('--data-sizes', nargs='+', type=int,
                        default=[1000, 5000, 10000, 50000],
                        help='Data sizes to benchmark')
    parser.add_argument('--feature-counts', nargs='+', type=int,
                        default=[10, 50, 100, 500],
                        help='Feature counts to benchmark')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark with smaller sizes')
    
    args = parser.parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.data_sizes = [1000, 5000, 10000]
        args.feature_counts = [10, 50, 100]
    
    # Create benchmark instance
    benchmark = MLBenchmark(args.output_dir)
    
    # Run benchmarks
    logger.info("Starting ML module performance benchmark...")
    
    # 1. Data size scaling
    benchmark.benchmark_data_sizes(args.data_sizes)
    
    # 2. Feature dimension scaling
    benchmark.benchmark_feature_dimensions(
        n_samples=10000,
        feature_counts=args.feature_counts
    )
    
    # 3. CV strategy comparison
    benchmark.benchmark_cv_strategies()
    
    # 4. Imputation performance
    benchmark.benchmark_imputation()
    
    # Generate plots
    if not args.skip_plots:
        logger.info("\nGenerating performance plots...")
        benchmark.plot_results()
    
    # Save results
    logger.info("\nSaving results...")
    benchmark.save_results()
    
    logger.info(f"\nBenchmark complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()