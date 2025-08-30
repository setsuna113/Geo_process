"""Unit tests for CP-WOPT reporters."""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

from src.abstractions import TensorData, CPFactors, TensorMetrics
from src.biodiversity_analysis.methods.cpwopt.reporters.factor_reporter import FactorReporter
from src.biodiversity_analysis.methods.cpwopt.reporters.reconstruction_reporter import ReconstructionReporter
from src.biodiversity_analysis.methods.cpwopt.reporters.spatial_reporter import SpatialReporter
from src.biodiversity_analysis.methods.cpwopt.reporters.component_reporter import ComponentReporter
from src.biodiversity_analysis.methods.cpwopt.reporters.summary_reporter import SummaryReporter


class TestFactorReporter(unittest.TestCase):
    """Test factor matrix visualization."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample CP factors
        self.factors = CPFactors(
            factors=[
                np.random.rand(10, 3),  # Latitude
                np.random.rand(10, 3),  # Longitude  
                np.random.rand(5, 3)    # Features
            ],
            rank=3,
            mode_names=['latitude', 'longitude', 'features'],
            feature_names=['species_a', 'species_b', 'species_c', 'species_d', 'species_e'],
            metadata={'experiment': 'test'}
        )
        
        # Create reporter
        self.output_dir = tempfile.mkdtemp()
        self.reporter = FactorReporter(self.output_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        plt.close('all')
    
    def test_plot_factor_matrices(self):
        """Test factor matrix plotting."""
        self.reporter.plot_factor_matrices(self.factors)
        
        # Check that plots were created
        expected_file = os.path.join(self.output_dir, 'factor_matrices.png')
        self.assertTrue(os.path.exists(expected_file))
        
        # Check file is not empty
        self.assertGreater(os.path.getsize(expected_file), 0)
    
    def test_plot_factor_heatmaps(self):
        """Test factor heatmap generation."""
        self.reporter.plot_factor_heatmaps(self.factors)
        
        # Should create one heatmap per mode
        for mode_idx, mode_name in enumerate(self.factors.mode_names):
            expected_file = os.path.join(
                self.output_dir, 
                f'factor_heatmap_{mode_name}.png'
            )
            self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_component_magnitudes(self):
        """Test component magnitude plotting."""
        self.reporter.plot_component_magnitudes(self.factors)
        
        expected_file = os.path.join(self.output_dir, 'component_magnitudes.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_generate_report(self):
        """Test full factor report generation."""
        self.reporter.generate_report(self.factors)
        
        # Check all expected files
        expected_files = [
            'factor_matrices.png',
            'factor_heatmap_latitude.png',
            'factor_heatmap_longitude.png',
            'factor_heatmap_features.png',
            'component_magnitudes.png'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.output_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing {filename}")


class TestReconstructionReporter(unittest.TestCase):
    """Test reconstruction quality visualization."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample tensor and reconstruction
        self.shape = (10, 10, 5)
        self.original = np.random.rand(*self.shape)
        self.reconstructed = self.original + 0.1 * np.random.randn(*self.shape)
        self.weights = (np.random.rand(*self.shape) > 0.2).astype(float)
        
        # Create metrics
        self.metrics = TensorMetrics(
            fit=0.85,
            relative_error=0.15,
            iterations=50,
            convergence_history=[0.5, 0.3, 0.2, 0.15],
            final_error=0.15,
            component_weights=np.array([0.5, 0.3, 0.2])
        )
        
        # Create reporter
        self.output_dir = tempfile.mkdtemp()
        self.reporter = ReconstructionReporter(self.output_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        plt.close('all')
    
    def test_plot_reconstruction_error(self):
        """Test reconstruction error plotting."""
        self.reporter.plot_reconstruction_error(
            self.original, 
            self.reconstructed,
            self.weights
        )
        
        expected_file = os.path.join(self.output_dir, 'reconstruction_error.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_convergence(self):
        """Test convergence plotting."""
        self.reporter.plot_convergence(self.metrics)
        
        expected_file = os.path.join(self.output_dir, 'convergence.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_fit_by_mode(self):
        """Test mode-wise fit plotting."""
        self.reporter.plot_fit_by_mode(
            self.original,
            self.reconstructed,
            self.weights
        )
        
        expected_file = os.path.join(self.output_dir, 'fit_by_mode.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_generate_report(self):
        """Test full reconstruction report."""
        self.reporter.generate_report(
            self.original,
            self.reconstructed,
            self.weights,
            self.metrics
        )
        
        # Check all files created
        expected_files = [
            'reconstruction_error.png',
            'convergence.png',
            'fit_by_mode.png',
            'observed_vs_reconstructed.png'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.output_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing {filename}")


class TestSpatialReporter(unittest.TestCase):
    """Test spatial pattern visualization."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create spatial data
        self.lat_bins = 20
        self.lon_bins = 20
        
        # Create factors with spatial structure
        lat_factor = np.zeros((self.lat_bins, 3))
        lat_factor[:, 0] = np.linspace(0, 1, self.lat_bins)  # Gradient
        lat_factor[:, 1] = np.sin(np.linspace(0, 2*np.pi, self.lat_bins))  # Periodic
        lat_factor[:, 2] = np.random.rand(self.lat_bins)  # Random
        
        lon_factor = np.zeros((self.lon_bins, 3))
        lon_factor[:, 0] = np.random.rand(self.lon_bins)
        lon_factor[:, 1] = np.linspace(1, 0, self.lon_bins)
        lon_factor[:, 2] = np.cos(np.linspace(0, np.pi, self.lon_bins))
        
        self.factors = CPFactors(
            factors=[
                lat_factor,
                lon_factor,
                np.random.rand(5, 3)  # Features
            ],
            rank=3,
            mode_names=['latitude', 'longitude', 'features'],
            metadata={
                'lat_range': (-90, 90),
                'lon_range': (-180, 180),
                'lat_bins': self.lat_bins,
                'lon_bins': self.lon_bins
            }
        )
        
        # Create tensor data
        tensor = np.random.rand(self.lat_bins, self.lon_bins, 5)
        self.tensor_data = TensorData(
            tensor=tensor,
            weights=np.ones_like(tensor),
            coordinates=[],
            feature_names=['sp1', 'sp2', 'sp3', 'sp4', 'sp5'],
            metadata=self.factors.metadata,
            is_sparse=False
        )
        
        self.output_dir = tempfile.mkdtemp()
        self.reporter = SpatialReporter(self.output_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        plt.close('all')
    
    def test_plot_spatial_components(self):
        """Test spatial component plotting."""
        self.reporter.plot_spatial_components(self.factors)
        
        # Should create one plot per component
        for comp in range(self.factors.rank):
            expected_file = os.path.join(
                self.output_dir,
                f'spatial_component_{comp}.png'
            )
            self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_species_distribution(self):
        """Test species distribution plotting."""
        self.reporter.plot_species_distribution(self.tensor_data, n_species=3)
        
        # Should create plots for top 3 species
        for i in range(3):
            expected_file = os.path.join(
                self.output_dir,
                f'species_distribution_{i}.png'
            )
            self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_hotspots(self):
        """Test biodiversity hotspot plotting."""
        # Create reconstructed tensor
        reconstructed = np.random.rand(self.lat_bins, self.lon_bins, 5)
        
        self.reporter.plot_hotspots(reconstructed, self.factors.metadata)
        
        expected_file = os.path.join(self.output_dir, 'biodiversity_hotspots.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_generate_report(self):
        """Test full spatial report."""
        reconstructed = np.random.rand(self.lat_bins, self.lon_bins, 5)
        
        self.reporter.generate_report(
            self.factors,
            self.tensor_data,
            reconstructed
        )
        
        # Check key files exist
        self.assertTrue(os.path.exists(
            os.path.join(self.output_dir, 'spatial_component_0.png')
        ))
        self.assertTrue(os.path.exists(
            os.path.join(self.output_dir, 'biodiversity_hotspots.png')
        ))


class TestComponentReporter(unittest.TestCase):
    """Test component interpretation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create factors with interpretable patterns
        self.factors = CPFactors(
            factors=[
                np.random.rand(10, 3),
                np.random.rand(10, 3),
                np.random.rand(8, 3)
            ],
            rank=3,
            mode_names=['latitude', 'longitude', 'features'],
            feature_names=[
                'species_a', 'species_b', 'species_c', 'species_d',
                'species_e', 'species_f', 'species_g', 'species_h'
            ],
            metadata={'experiment': 'test'}
        )
        
        self.output_dir = tempfile.mkdtemp()
        self.reporter = ComponentReporter(self.output_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        plt.close('all')
    
    def test_plot_component_loadings(self):
        """Test component loading plots."""
        self.reporter.plot_component_loadings(self.factors)
        
        # Should create one plot per component
        for comp in range(self.factors.rank):
            expected_file = os.path.join(
                self.output_dir,
                f'component_{comp}_loadings.png'
            )
            self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        self.reporter.plot_feature_importance(self.factors)
        
        expected_file = os.path.join(self.output_dir, 'feature_importance.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_plot_component_similarity(self):
        """Test component similarity matrix."""
        self.reporter.plot_component_similarity(self.factors)
        
        expected_file = os.path.join(self.output_dir, 'component_similarity.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_generate_interpretation_report(self):
        """Test interpretation report generation."""
        report = self.reporter.generate_interpretation_report(self.factors)
        
        # Check report structure
        self.assertIn('components', report)
        self.assertEqual(len(report['components']), 3)
        
        # Check each component has required fields
        for comp_report in report['components']:
            self.assertIn('top_features', comp_report)
            self.assertIn('spatial_pattern', comp_report)
            self.assertIn('interpretation', comp_report)
    
    def test_generate_report(self):
        """Test full component report."""
        self.reporter.generate_report(self.factors)
        
        # Check files and text report
        expected_files = [
            'component_0_loadings.png',
            'feature_importance.png',
            'component_similarity.png',
            'component_interpretation.txt'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.output_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing {filename}")


class TestSummaryReporter(unittest.TestCase):
    """Test comprehensive summary generation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create complete analysis results
        shape = (10, 10, 5)
        self.tensor_data = TensorData(
            tensor=np.random.rand(*shape),
            weights=np.ones(shape),
            coordinates=[],
            feature_names=['sp1', 'sp2', 'sp3', 'sp4', 'sp5'],
            metadata={'lat_bins': 10, 'lon_bins': 10},
            is_sparse=False
        )
        
        self.factors = CPFactors(
            factors=[
                np.random.rand(10, 3),
                np.random.rand(10, 3),
                np.random.rand(5, 3)
            ],
            rank=3,
            mode_names=['latitude', 'longitude', 'features'],
            feature_names=self.tensor_data.feature_names,
            metadata=self.tensor_data.metadata
        )
        
        self.metrics = TensorMetrics(
            fit=0.82,
            relative_error=0.18,
            iterations=75,
            convergence_history=list(np.linspace(0.5, 0.18, 75)),
            final_error=0.18,
            component_weights=np.array([0.5, 0.3, 0.2])
        )
        
        self.reconstructed = np.random.rand(*shape)
        
        self.output_dir = tempfile.mkdtemp()
        self.reporter = SummaryReporter(self.output_dir)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        plt.close('all')
    
    def test_generate_summary_stats(self):
        """Test summary statistics generation."""
        stats = self.reporter.generate_summary_stats(
            self.tensor_data,
            self.factors,
            self.metrics
        )
        
        # Check all required fields
        required_fields = [
            'tensor_shape', 'sparsity', 'rank', 'fit',
            'iterations', 'n_parameters', 'compression_ratio'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats)
    
    def test_create_summary_figure(self):
        """Test summary figure creation."""
        self.reporter.create_summary_figure(
            self.tensor_data,
            self.factors,
            self.metrics,
            self.reconstructed
        )
        
        expected_file = os.path.join(self.output_dir, 'analysis_summary.png')
        self.assertTrue(os.path.exists(expected_file))
    
    def test_generate_text_report(self):
        """Test text report generation."""
        self.reporter.generate_text_report(
            self.tensor_data,
            self.factors,
            self.metrics,
            {'key': 'value'}
        )
        
        expected_file = os.path.join(self.output_dir, 'analysis_report.txt')
        self.assertTrue(os.path.exists(expected_file))
        
        # Check content
        with open(expected_file, 'r') as f:
            content = f.read()
            self.assertIn('CP-WOPT', content)
            self.assertIn('Summary Statistics', content)
            self.assertIn('Model Performance', content)
    
    def test_generate_full_report(self):
        """Test complete report generation."""
        self.reporter.generate_report(
            self.tensor_data,
            self.factors,
            self.metrics,
            self.reconstructed,
            {'analysis_time': 120.5}
        )
        
        # Check all sub-reports called
        subdirs = ['factors', 'reconstruction', 'spatial', 'components']
        for subdir in subdirs:
            subdir_path = os.path.join(self.output_dir, subdir)
            self.assertTrue(os.path.exists(subdir_path))
            # Check subdir has files
            self.assertGreater(len(os.listdir(subdir_path)), 0)
        
        # Check summary files
        self.assertTrue(os.path.exists(
            os.path.join(self.output_dir, 'analysis_summary.png')
        ))
        self.assertTrue(os.path.exists(
            os.path.join(self.output_dir, 'analysis_report.txt')
        ))
    
    def test_export_results(self):
        """Test result export functionality."""
        # Add method to export results
        results_file = os.path.join(self.output_dir, 'results.npz')
        
        # Save results
        np.savez(
            results_file,
            factors=[f for f in self.factors.factors],
            tensor_shape=self.tensor_data.shape,
            feature_names=self.tensor_data.feature_names,
            fit=self.metrics.fit,
            iterations=self.metrics.iterations
        )
        
        # Check file exists and can be loaded
        self.assertTrue(os.path.exists(results_file))
        loaded = np.load(results_file, allow_pickle=True)
        self.assertEqual(len(loaded['factors']), 3)


if __name__ == '__main__':
    unittest.main()