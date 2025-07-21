# tests/test_spatial_analysis/test_integration.py
"""End-to-end integration tests for spatial analysis workflows."""

import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path

from src.config.config import Config
from src.spatial_analysis.som.som_trainer import SOMAnalyzer
from src.spatial_analysis.maxp_regions.region_optimizer import MaxPAnalyzer
from src.spatial_analysis.gwpca.gwpca_analyzer import GWPCAAnalyzer
# from src.spatial_analysis.report_generator import UnifiedReportGenerator


@pytest.fixture
def test_data():
    """Create realistic test dataset."""
    np.random.seed(42)
    
    # Create spatial gradients to simulate realistic patterns
    y, x = np.mgrid[0:10, 0:10]
    
    # P: Productivity gradient from north to south
    P = 0.3 + 0.05 * y + 0.1 * np.random.rand(10, 10)
    
    # A: Activity gradient from west to east  
    A = 0.2 + 0.05 * x + 0.1 * np.random.rand(10, 10)
    
    # F: Fragmentation with some spatial structure
    F = 0.5 + 0.2 * np.sin(x/2) * np.cos(y/2) + 0.1 * np.random.rand(10, 10)
    
    # Normalize to [0, 1]
    P = (P - P.min()) / (P.max() - P.min())
    A = (A - A.min()) / (A.max() - A.min())
    F = (F - F.min()) / (F.max() - F.min())
    
    return xr.Dataset({
        'P': xr.DataArray(P, dims=['lat', 'lon'], 
                         coords={'lat': np.linspace(40, 50, 10),
                                'lon': np.linspace(-120, -110, 10)}),
        'A': xr.DataArray(A, dims=['lat', 'lon']),
        'F': xr.DataArray(F, dims=['lat', 'lon'])
    })


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.config = {
        'spatial_analysis': {
            'normalize_data': True,
            'save_results': True,
            'output_dir': 'test_output',
            'som': {
                'grid_size': [3, 3],
                'iterations': 50,
                'sigma': 1.0,
                'learning_rate': 0.5
            },
            'maxp': {
                'pixel_size_km': 10,  # Larger pixels for test
                'min_area_km2': 300,  # 3 pixels minimum
                'ecological_scale': 'landscape',
                'iterations': 5
            },
            'gwpca': {
                'pixel_size_km': 10,
                'block_size_km': 20,  # 2x2 pixel blocks
                'use_block_aggregation': True,
                'n_components': 2,
                'adaptive': True
            }
        },
        'processors': {
            'data_preparation': {
                'normalization': {
                    'method': 'min_max'
                }
            }
        }
    }
    return config


class TestSOMWorkflow:
    """Test complete SOM workflow."""
    
    def test_som_analysis_pipeline(self, config, test_data):
        """Test SOM from data loading to visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.config['spatial_analysis']['output_dir'] = tmpdir
            
            # Initialize analyzer
            analyzer = SOMAnalyzer(config)
            
            # Run analysis
            result = analyzer.analyze(
                test_data,
                grid_size=[3, 3],
                iterations=20  # Reduced for speed
            )
            
            # Verify basic results
            assert result.metadata.analysis_type == 'SOM'
            assert result.labels.shape == (100,)  # 10x10 grid
            assert 0 <= result.labels.max() < 9  # 3x3 SOM grid
            assert result.statistics['n_clusters'] <= 9
            
            # Check saved files
            saved_files = list(Path(tmpdir).rglob('*'))
            assert any('labels.npy' in str(f) for f in saved_files)
            assert any('metadata.json' in str(f) for f in saved_files)
            
            # Test visualization workflow
            from src.spatial_analysis.som.som_visualizer import SOMVisualizer
            visualizer = SOMVisualizer()
            
            # Create all visualizations
            cluster_fig = visualizer.plot_cluster_map(result)
            assert cluster_fig is not None
            plt.close(cluster_fig)
            
            umatrix_fig = visualizer.plot_umatrix(result)
            assert umatrix_fig is not None
            plt.close(umatrix_fig)
            
            # Test reporting workflow
            from src.spatial_analysis.som.som_reporter import SOMReporter
            reporter = SOMReporter(output_dir=Path(tmpdir))
            
            report = reporter.generate_full_report(result)
            assert 'quality_metrics' in report
            assert 'cluster_summary' in report
            assert 'interpretation' in report
    
    def test_som_parameter_sensitivity(self, config, test_data):
        """Test SOM with different parameters."""
        analyzer = SOMAnalyzer(config)
        
        # Test different grid sizes
        for grid_size in [[2, 2], [4, 4]]:
            result = analyzer.analyze(
                test_data,
                grid_size=grid_size,
                iterations=10
            )
            
            max_clusters = grid_size[0] * grid_size[1]
            assert result.statistics['n_clusters'] <= max_clusters
            assert result.statistics['empty_neurons'] >= 0


class TestMaxPWorkflow:
    """Test complete Max-p workflow."""
    
    def test_maxp_analysis_pipeline(self, config, test_data):
        """Test Max-p from data loading to reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.config['spatial_analysis']['output_dir'] = tmpdir
            
            # Initialize analyzer
            analyzer = MaxPAnalyzer(config)
            
            # Run analysis
            result = analyzer.analyze(
                test_data.to_array(dim='band'),  # Convert to DataArray
                min_area_km2=300,  # 3 pixels minimum
                ecological_scale='landscape',
                run_perturbation=True,
                perturbation_range=[0.5, 1.0, 1.5]
            )
            
            # Verify results
            assert result.metadata.analysis_type == 'MaxP'
            assert result.labels.shape == (100,)
            
            # Check region sizes
            region_areas = result.additional_outputs['region_sizes_km2']
            for area in region_areas.values():
                assert area >= 300  # Meets threshold
            
            # Check perturbation analysis
            assert 'perturbation_results' in result.additional_outputs
            pert = result.additional_outputs['perturbation_results']
            assert len(pert['tested_areas_km2']) == 3
            assert 'stability_assessment' in pert
            
            # Test visualization
            from src.spatial_analysis.maxp_regions.region_reporter import RegionReporter
            reporter = RegionReporter()
            
            char_fig = reporter.plot_region_characteristics(result)
            assert char_fig is not None
            plt.close(char_fig)
            
            pert_fig = reporter.plot_perturbation_analysis(result)
            assert pert_fig is not None
            plt.close(pert_fig)
    
    def test_maxp_ecological_scales(self, config, test_data):
        """Test Max-p with different ecological scales."""
        analyzer = MaxPAnalyzer(config)
        
        scales = {
            'landscape': (200, 400),  # Expected region count range
            'ecoregion': (100, 1000)
        }
        
        for scale, (min_area, max_area) in scales.items():
            # Use appropriate area for scale
            area = (min_area + max_area) / 2
            
            result = analyzer.analyze(
                test_data.to_array(dim='band'),
                min_area_km2=area,
                ecological_scale=scale,
                run_perturbation=False
            )
            
            assert result.statistics['ecological_scale'] == scale
            assert result.statistics['smallest_region_km2'] >= area * 0.9  # Allow small deviation


class TestGWPCAWorkflow:
    """Test complete GWPCA workflow."""
    
    def test_gwpca_analysis_pipeline(self, config, test_data):
        """Test GWPCA with block aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.config['spatial_analysis']['output_dir'] = tmpdir
            
            # Initialize analyzer
            analyzer = GWPCAAnalyzer(config)
            
            # Run analysis
            result = analyzer.analyze(
                test_data,
                use_block_aggregation=True,
                block_size_km=20,  # 2x2 pixel blocks
                bandwidth=3  # Fixed for testing
            )
            
            # Verify results
            assert result.metadata.analysis_type == 'GWPCA'
            assert result.labels.shape == (25,)  # 5x5 blocks from 10x10 pixels
            assert 'n_blocks_analyzed' in result.statistics
            assert result.statistics['n_blocks_analyzed'] == 25
            
            # Check R² values
            assert 0 <= result.statistics['local_r2_mean'] <= 1
            assert result.statistics['local_r2_min'] >= 0
            assert result.statistics['local_r2_max'] <= 1
            
            # Test visualization
            from src.spatial_analysis.gwpca.local_stats_mapper import LocalStatsMapper
            mapper = LocalStatsMapper()
            
            r2_fig = mapper.plot_local_r2_map(result)
            assert r2_fig is not None
            plt.close(r2_fig)
            
            summary_fig = mapper.create_summary_figure(result)
            assert summary_fig is not None
            plt.close(summary_fig)
    
    def test_gwpca_bandwidth_selection(self, config, test_data):
        """Test GWPCA with automatic bandwidth selection."""
        analyzer = GWPCAAnalyzer(config)
        
        # Small data for fast bandwidth selection
        small_data = test_data.isel(lat=slice(0, 6), lon=slice(0, 6))
        
        result = analyzer.analyze(
            small_data,
            use_block_aggregation=False,  # Pixel-level for bandwidth test
            bandwidth=None  # Auto-select
        )
        
        assert 'bandwidth' in result.statistics
        assert result.statistics['bandwidth'] > 0


class TestUnifiedWorkflow:
    """Test unified report generation across methods."""
    
    def test_comparative_analysis(self, config, test_data):
        """Run all three methods and compare."""
        results = {}
        
        # Run SOM
        som_analyzer = SOMAnalyzer(config)
        results['som'] = som_analyzer.analyze(test_data, iterations=20)
        
        # Run Max-p
        maxp_analyzer = MaxPAnalyzer(config)
        results['maxp'] = maxp_analyzer.analyze(
            test_data.to_array(dim='band'),
            min_area_km2=500,  # 5 pixels with 10km pixel size (100 km² per pixel)
            run_perturbation=False
        )
        
        # Run GWPCA
        gwpca_analyzer = GWPCAAnalyzer(config)
        results['gwpca'] = gwpca_analyzer.analyze(
            test_data,
            block_size_km=20,
            bandwidth=3
        )
        
        # Verify all completed
        assert all(r.metadata.analysis_type in ['SOM', 'MaxP', 'GWPCA'] 
                  for r in results.values())
        
        # Compare spatial patterns
        som_clusters = len(np.unique(results['som'].labels))
        maxp_regions = results['maxp'].statistics['n_regions']
        
        # Different methods should find different patterns
        assert som_clusters != maxp_regions
        
        # GWPCA should show spatial variation
        gwpca_heterogeneity = results['gwpca'].statistics['spatial_heterogeneity']
        assert gwpca_heterogeneity > 0


# Import matplotlib for closing figures
import matplotlib.pyplot as plts