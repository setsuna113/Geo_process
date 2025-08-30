"""End-to-end integration tests for CP-WOPT pipeline."""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil
import time
from unittest.mock import Mock, patch

from src.biodiversity_analysis.methods.cpwopt.analyzer_adapter import CPWOPTAnalyzer
from src.biodiversity_analysis.methods.cpwopt.tests.test_data_generator import (
    SyntheticBiodiversityGenerator, SyntheticDataConfig, create_validation_dataset
)
from src.core_abstractions import AnalysisResult
from src.config.cpwopt.cpwopt_config_loader import CPWOPTConfigLoader


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete CP-WOPT analysis pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, 'outputs')
        os.makedirs(self.output_dir)
        
        # Generate test data
        config = SyntheticDataConfig(
            n_samples=2000,
            n_species=25,
            rank=4,
            sparsity=0.8,
            noise_level=0.15,
            seed=42
        )
        generator = SyntheticBiodiversityGenerator(config)
        self.test_df = generator.generate_dataframe()
        
        # Create minimal config
        self.config = {
            'rank': 4,
            'max_iterations': 50,
            'convergence_threshold': 1e-4,
            'initialization': 'random',
            'use_minibatch': True,
            'minibatch_size': 500,
            'preprocessing': {
                'lat_bins': 15,
                'lon_bins': 15,
                'aggregation': 'sum',
                'transform': 'log1p'
            },
            'output_dir': self.output_dir,
            'verbose': False
        }
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis from DataFrame to results."""
        # Create analyzer
        analyzer = CPWOPTAnalyzer(self.config)
        
        # Run analysis
        start_time = time.time()
        result = analyzer.analyze(self.test_df)
        analysis_time = time.time() - start_time
        
        # Check result structure
        self.assertIsInstance(result, AnalysisResult)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        
        # Check metadata
        self.assertIn('analysis_time', result.metadata)
        self.assertIn('config', result.metadata)
        self.assertIn('convergence_iterations', result.metadata)
        
        # Check data contains expected fields
        self.assertIn('factors', result.data)
        self.assertIn('metrics', result.data)
        self.assertIn('tensor_shape', result.data)
        
        # Check performance
        self.assertLess(analysis_time, 60)  # Should complete in under 1 minute
        
        # Check output files exist
        self.assertTrue(os.path.exists(self.output_dir))
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)
    
    def test_analysis_with_missing_data(self):
        """Test analysis with significant missing data."""
        # Add 30% missing values
        missing_df = self.test_df.copy()
        for col in missing_df.columns[2:]:  # Skip lat/lon
            mask = np.random.rand(len(missing_df)) < 0.3
            missing_df.loc[mask, col] = np.nan
        
        analyzer = CPWOPTAnalyzer(self.config)
        result = analyzer.analyze(missing_df)
        
        # Should still succeed
        self.assertTrue(result.success)
        
        # Check that weights reflect missing data
        weights = result.data['tensor_weights']
        self.assertLess(np.mean(weights), 1.0)  # Some weights should be < 1
    
    def test_analysis_with_extreme_sparsity(self):
        """Test with extremely sparse data (>95% zeros)."""
        # Create very sparse data
        sparse_config = SyntheticDataConfig(
            n_samples=1000,
            n_species=20,
            sparsity=0.96,
            seed=123
        )
        sparse_gen = SyntheticBiodiversityGenerator(sparse_config)
        sparse_df = sparse_gen.generate_dataframe()
        
        analyzer = CPWOPTAnalyzer(self.config)
        result = analyzer.analyze(sparse_df)
        
        # Should handle extreme sparsity
        self.assertTrue(result.success)
        
        # Check sparsity is recorded
        self.assertIn('sparsity', result.metadata)
        self.assertGreater(result.metadata['sparsity'], 0.95)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        # Configure with fixed seed
        config_seeded = self.config.copy()
        config_seeded['random_state'] = 42
        
        # Run analysis twice
        analyzer1 = CPWOPTAnalyzer(config_seeded)
        result1 = analyzer1.analyze(self.test_df)
        
        analyzer2 = CPWOPTAnalyzer(config_seeded)
        result2 = analyzer2.analyze(self.test_df)
        
        # Results should be very similar
        factors1 = result1.data['factors'].factors
        factors2 = result2.data['factors'].factors
        
        for f1, f2 in zip(factors1, factors2):
            # Account for sign ambiguity in factors
            similarity = max(
                np.corrcoef(f1.flatten(), f2.flatten())[0, 1],
                np.corrcoef(f1.flatten(), -f2.flatten())[0, 1]
            )
            self.assertGreater(abs(similarity), 0.99)
    
    def test_convergence_behavior(self):
        """Test convergence behavior and early stopping."""
        # Configure for convergence testing
        config_conv = self.config.copy()
        config_conv['max_iterations'] = 200
        config_conv['convergence_threshold'] = 1e-5
        
        analyzer = CPWOPTAnalyzer(config_conv)
        result = analyzer.analyze(self.test_df)
        
        # Should converge before max iterations
        self.assertTrue(result.success)
        iterations = result.metadata['convergence_iterations']
        self.assertLess(iterations, 200)
        
        # Check convergence history
        convergence_history = result.data['metrics'].convergence_history
        self.assertGreater(len(convergence_history), 10)
        
        # Error should decrease
        self.assertLess(convergence_history[-1], convergence_history[0])
    
    def test_rank_selection_impact(self):
        """Test impact of rank selection on results."""
        ranks = [2, 5, 10]
        fits = []
        
        for rank in ranks:
            config_rank = self.config.copy()
            config_rank['rank'] = rank
            
            analyzer = CPWOPTAnalyzer(config_rank)
            result = analyzer.analyze(self.test_df)
            
            self.assertTrue(result.success)
            fits.append(result.data['metrics'].fit)
        
        # Higher rank should give better fit
        self.assertGreater(fits[1], fits[0])  # rank 5 > rank 2
        self.assertGreater(fits[2], fits[1])  # rank 10 > rank 5
        
        # But improvement should diminish
        improvement_1 = fits[1] - fits[0]
        improvement_2 = fits[2] - fits[1]
        self.assertGreater(improvement_1, improvement_2)
    
    def test_preprocessing_options(self):
        """Test different preprocessing configurations."""
        preprocessing_configs = [
            {'transform': 'none', 'aggregation': 'sum'},
            {'transform': 'log1p', 'aggregation': 'mean'},
            {'transform': 'sqrt', 'aggregation': 'max'}
        ]
        
        for preproc in preprocessing_configs:
            with self.subTest(preprocessing=preproc):
                config_prep = self.config.copy()
                config_prep['preprocessing'].update(preproc)
                
                analyzer = CPWOPTAnalyzer(config_prep)
                result = analyzer.analyze(self.test_df)
                
                self.assertTrue(result.success)
                self.assertIn('preprocessing', result.metadata)
    
    def test_memory_efficiency(self):
        """Test memory-efficient processing of large data."""
        # Create larger dataset
        large_config = SyntheticDataConfig(
            n_samples=10000,
            n_species=50,
            sparsity=0.9,
            seed=456
        )
        large_gen = SyntheticBiodiversityGenerator(large_config)
        large_df = large_gen.generate_dataframe()
        
        # Configure for memory efficiency
        config_mem = self.config.copy()
        config_mem['use_minibatch'] = True
        config_mem['minibatch_size'] = 100
        config_mem['preprocessing']['lat_bins'] = 30
        config_mem['preprocessing']['lon_bins'] = 30
        
        analyzer = CPWOPTAnalyzer(config_mem)
        
        # Should complete without memory errors
        result = analyzer.analyze(large_df)
        self.assertTrue(result.success)
    
    def test_validation_against_ground_truth(self):
        """Test recovery of known ground truth structure."""
        # Create data with known structure
        tensor_data, true_factors = create_validation_dataset()
        
        # Convert to DataFrame format
        df_data = []
        for i in range(tensor_data.shape[0]):
            for j in range(tensor_data.shape[1]):
                if np.any(tensor_data.tensor[i, j, :] > 0):
                    row = {
                        'latitude': -45 + i * 90 / tensor_data.shape[0],
                        'longitude': -180 + j * 360 / tensor_data.shape[1]
                    }
                    for k, species in enumerate(tensor_data.feature_names):
                        if tensor_data.weights[i, j, k] > 0:
                            row[species] = tensor_data.tensor[i, j, k]
                        else:
                            row[species] = np.nan
                    df_data.append(row)
        
        validation_df = pd.DataFrame(df_data)
        
        # Configure analyzer
        config_val = self.config.copy()
        config_val['rank'] = true_factors.rank
        config_val['preprocessing']['lat_bins'] = tensor_data.shape[0]
        config_val['preprocessing']['lon_bins'] = tensor_data.shape[1]
        
        analyzer = CPWOPTAnalyzer(config_val)
        result = analyzer.analyze(validation_df)
        
        self.assertTrue(result.success)
        
        # Check recovery quality
        recovered_factors = result.data['factors'].factors
        
        # Compare factor subspaces (accounting for permutation and sign)
        from src.biodiversity_analysis.shared.tensor_utils import calculate_tucker_congruence
        
        for mode in range(3):
            congruence = calculate_tucker_congruence(
                true_factors.factors[mode],
                recovered_factors[mode]
            )
            # At least one component should match well
            max_congruence = np.max(np.abs(congruence))
            self.assertGreater(max_congruence, 0.8)
    
    def test_error_handling(self):
        """Test graceful error handling."""
        # Test with invalid data
        invalid_cases = [
            # Empty DataFrame
            pd.DataFrame(),
            # No species columns
            pd.DataFrame({'latitude': [1, 2], 'longitude': [3, 4]}),
            # All NaN species data
            pd.DataFrame({
                'latitude': [1, 2, 3],
                'longitude': [4, 5, 6],
                'species': [np.nan, np.nan, np.nan]
            }),
            # Invalid coordinates
            pd.DataFrame({
                'latitude': [200, -200],  # Out of bounds
                'longitude': [0, 0],
                'species': [1, 2]
            })
        ]
        
        analyzer = CPWOPTAnalyzer(self.config)
        
        for i, invalid_df in enumerate(invalid_cases):
            with self.subTest(case=i):
                result = analyzer.analyze(invalid_df)
                
                # Should return failure with error message
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)
                self.assertIn('error_type', result.metadata)
    
    def test_output_generation(self):
        """Test that all expected outputs are generated."""
        # Configure with all outputs enabled
        config_output = self.config.copy()
        config_output['generate_plots'] = True
        config_output['save_factors'] = True
        config_output['save_reconstructed'] = True
        
        analyzer = CPWOPTAnalyzer(config_output)
        result = analyzer.analyze(self.test_df)
        
        self.assertTrue(result.success)
        
        # Check for expected output files
        output_files = []
        for root, dirs, files in os.walk(self.output_dir):
            output_files.extend(files)
        
        # Should have various plots and data files
        self.assertGreater(len(output_files), 5)
        
        # Check for specific file types
        has_png = any(f.endswith('.png') for f in output_files)
        has_txt = any(f.endswith('.txt') for f in output_files)
        has_npz = any(f.endswith('.npz') for f in output_files)
        
        self.assertTrue(has_png, "No PNG plots generated")
        self.assertTrue(has_txt, "No text reports generated")
        self.assertTrue(has_npz, "No data files saved")
    
    def test_integration_with_config_loader(self):
        """Test integration with configuration loading system."""
        # Create temporary config file
        config_file = os.path.join(self.test_dir, 'test_config.yml')
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump({
                'cpwopt': {
                    'rank': 5,
                    'max_iterations': 100,
                    'preprocessing': {
                        'transform': 'asinh'
                    }
                }
            }, f)
        
        # Load config
        loader = CPWOPTConfigLoader()
        loaded_config = loader.load_config(config_file)
        
        # Create analyzer with loaded config
        analyzer = CPWOPTAnalyzer(loaded_config)
        result = analyzer.analyze(self.test_df)
        
        self.assertTrue(result.success)
        self.assertEqual(result.metadata['config']['rank'], 5)
        self.assertEqual(
            result.metadata['config']['preprocessing']['transform'], 
            'asinh'
        )


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for CP-WOPT."""
    
    def test_scaling_with_data_size(self):
        """Test performance scaling with data size."""
        sizes = [1000, 5000, 10000]
        times = []
        
        config = {
            'rank': 3,
            'max_iterations': 20,
            'preprocessing': {
                'lat_bins': 20,
                'lon_bins': 20
            },
            'verbose': False
        }
        
        for n_samples in sizes:
            # Generate data
            data_config = SyntheticDataConfig(
                n_samples=n_samples,
                n_species=20,
                sparsity=0.8,
                seed=42
            )
            generator = SyntheticBiodiversityGenerator(data_config)
            df = generator.generate_dataframe()
            
            # Time analysis
            analyzer = CPWOPTAnalyzer(config)
            start_time = time.time()
            result = analyzer.analyze(df)
            elapsed = time.time() - start_time
            
            self.assertTrue(result.success)
            times.append(elapsed)
        
        # Check scaling is reasonable (not quadratic)
        # Time should increase less than linearly with data size
        time_ratio_1 = times[1] / times[0]  # 5000 vs 1000
        time_ratio_2 = times[2] / times[1]  # 10000 vs 5000
        size_ratio = 5.0 / 1.0
        
        # Time increase should be sublinear
        self.assertLess(time_ratio_1, size_ratio)
        self.assertLess(time_ratio_2, size_ratio / 5.0 * 2.0)
    
    def test_scaling_with_tensor_size(self):
        """Test performance scaling with tensor dimensions."""
        dimensions = [(10, 10), (20, 20), (30, 30)]
        times = []
        
        for lat_bins, lon_bins in dimensions:
            config = {
                'rank': 3,
                'max_iterations': 20,
                'preprocessing': {
                    'lat_bins': lat_bins,
                    'lon_bins': lon_bins
                },
                'verbose': False
            }
            
            # Use fixed data size
            data_config = SyntheticDataConfig(
                n_samples=5000,
                n_species=20,
                seed=42
            )
            generator = SyntheticBiodiversityGenerator(data_config)
            df = generator.generate_dataframe()
            
            analyzer = CPWOPTAnalyzer(config)
            start_time = time.time()
            result = analyzer.analyze(df)
            elapsed = time.time() - start_time
            
            self.assertTrue(result.success)
            times.append(elapsed)
        
        # Time should scale roughly with tensor size
        # But minibatching should keep it reasonable
        self.assertLess(times[2], times[0] * 10)  # Not 9x slower for 3x3 size


if __name__ == '__main__':
    unittest.main()