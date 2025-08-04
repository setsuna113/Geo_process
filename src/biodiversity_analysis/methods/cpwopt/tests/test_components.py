"""Unit tests for CP-WOPT components."""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from src.biodiversity_analysis.methods.cpwopt.minibatch_processor import MiniBatchProcessor
from src.biodiversity_analysis.methods.cpwopt.factor_updater import FactorUpdater
from src.biodiversity_analysis.methods.cpwopt.sparse_handler import SparseHandler
from src.biodiversity_analysis.methods.cpwopt.tensor_metrics import TensorMetricsCalculator
from src.abstractions.types.tensor_types import TensorData, CPFactors


class TestMiniBatchProcessor(unittest.TestCase):
    """Test mini-batch processing functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.shape = (20, 20, 10)
        self.tensor = np.random.rand(*self.shape)
        self.weights = (np.random.rand(*self.shape) > 0.3).astype(float)
        
        self.tensor_data = TensorData(
            tensor=self.tensor,
            weights=self.weights,
            coordinates=[],
            feature_names=[f'f{i}' for i in range(10)],
            metadata={},
            is_sparse=False
        )
        
        self.factors = [
            np.random.rand(20, 3),
            np.random.rand(20, 3),
            np.random.rand(10, 3)
        ]
    
    def test_batch_creation(self):
        """Test mini-batch creation."""
        processor = MiniBatchProcessor(batch_size=100)
        
        batches = list(processor.create_batches(
            self.tensor_data, 
            mode=0
        ))
        
        # Check we get multiple batches
        self.assertGreater(len(batches), 1)
        
        # Check batch properties
        for batch_tensor, batch_weights, indices in batches:
            # Shape should match except for mode dimension
            expected_shape = list(self.shape)
            expected_shape[0] = len(indices)
            self.assertEqual(batch_tensor.shape, tuple(expected_shape))
            self.assertEqual(batch_weights.shape, tuple(expected_shape))
            
            # Indices should be valid
            self.assertTrue(all(0 <= idx < self.shape[0] for idx in indices))
    
    def test_factor_update_accumulation(self):
        """Test accumulating factor updates from batches."""
        processor = MiniBatchProcessor(batch_size=50)
        
        # Initialize accumulator
        mode = 0
        accumulated = np.zeros((self.shape[mode], 3))
        total_weight = 0
        
        # Process batches
        for batch_tensor, batch_weights, indices in processor.create_batches(
            self.tensor_data, mode
        ):
            # Simulate factor update (simplified)
            batch_update = np.random.rand(len(indices), 3)
            
            # Accumulate
            accumulated[indices] += batch_update
            total_weight += len(indices)
        
        # Check all indices were covered
        self.assertEqual(total_weight, self.shape[mode])
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        processor = MiniBatchProcessor(batch_size=100)
        
        memory_estimate = processor.estimate_memory_usage(
            self.tensor_data,
            rank=3
        )
        
        # Should return reasonable estimate
        self.assertGreater(memory_estimate, 0)
        self.assertLess(memory_estimate, 1e9)  # Less than 1GB for small tensor
    
    def test_adaptive_batch_size(self):
        """Test adaptive batch size adjustment."""
        processor = MiniBatchProcessor(
            batch_size=1000,  # Large initial size
            adaptive=True,
            max_memory_mb=10  # Small memory limit
        )
        
        # Should adapt batch size based on tensor
        adapted_size = processor.get_adapted_batch_size(self.tensor_data)
        
        # Should be smaller than initial
        self.assertLess(adapted_size, 1000)
        self.assertGreater(adapted_size, 0)
    
    def test_sparse_tensor_batching(self):
        """Test batching with sparse tensors."""
        # Create very sparse tensor
        sparse_tensor = self.tensor.copy()
        sparse_mask = np.random.rand(*self.shape) > 0.95
        sparse_tensor[~sparse_mask] = 0
        
        sparse_data = TensorData(
            tensor=sparse_tensor,
            weights=sparse_mask.astype(float),
            coordinates=[],
            feature_names=[f'f{i}' for i in range(10)],
            metadata={},
            is_sparse=True
        )
        
        processor = MiniBatchProcessor(batch_size=100)
        batches = list(processor.create_batches(sparse_data, mode=0))
        
        # Should handle sparse data
        self.assertGreater(len(batches), 0)
        
        # Check sparsity is preserved
        for batch_tensor, _, _ in batches:
            batch_sparsity = (batch_tensor == 0).sum() / batch_tensor.size
            self.assertGreater(batch_sparsity, 0.9)


class TestFactorUpdater(unittest.TestCase):
    """Test factor update algorithms."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.tensor = np.random.rand(10, 8, 5)
        self.weights = np.ones_like(self.tensor)
        self.factors = [
            np.random.rand(10, 3),
            np.random.rand(8, 3),
            np.random.rand(5, 3)
        ]
    
    def test_als_update(self):
        """Test ALS factor update."""
        updater = FactorUpdater(method='als')
        
        # Update first mode
        updated_factor = updater.update_factor(
            self.tensor,
            self.weights,
            self.factors,
            mode=0
        )
        
        # Check shape
        self.assertEqual(updated_factor.shape, (10, 3))
        
        # Check no NaN or inf
        self.assertFalse(np.any(np.isnan(updated_factor)))
        self.assertFalse(np.any(np.isinf(updated_factor)))
        
        # Should be different from original
        self.assertFalse(np.allclose(updated_factor, self.factors[0]))
    
    def test_weighted_update(self):
        """Test updates with weight mask."""
        updater = FactorUpdater(method='als')
        
        # Create partial weights
        partial_weights = self.weights.copy()
        partial_weights[:5, :, :] = 0  # Half the data missing
        
        # Update should still work
        updated_factor = updater.update_factor(
            self.tensor,
            partial_weights,
            self.factors,
            mode=0
        )
        
        # Check valid update
        self.assertEqual(updated_factor.shape, (10, 3))
        self.assertFalse(np.any(np.isnan(updated_factor)))
    
    def test_regularized_update(self):
        """Test regularized factor updates."""
        updater = FactorUpdater(
            method='als',
            regularization=0.01
        )
        
        # Update with regularization
        updated_regular = updater.update_factor(
            self.tensor,
            self.weights,
            self.factors,
            mode=0
        )
        
        # Compare with non-regularized
        updater_no_reg = FactorUpdater(method='als', regularization=0)
        updated_no_reg = updater_no_reg.update_factor(
            self.tensor,
            self.weights,
            self.factors,
            mode=0
        )
        
        # Should be different
        self.assertFalse(np.allclose(updated_regular, updated_no_reg))
        
        # Regularized should have smaller norm
        self.assertLess(
            np.linalg.norm(updated_regular),
            np.linalg.norm(updated_no_reg) * 1.1  # Allow some tolerance
        )
    
    def test_normalization(self):
        """Test factor normalization."""
        updater = FactorUpdater(method='als')
        
        # Get normalized factors
        normalized = updater.normalize_factors(self.factors)
        
        # Check all factors have unit norm columns
        for factor in normalized:
            for col in range(factor.shape[1]):
                col_norm = np.linalg.norm(factor[:, col])
                self.assertAlmostEqual(col_norm, 1.0, places=6)
    
    def test_gradient_update(self):
        """Test gradient-based updates."""
        updater = FactorUpdater(
            method='gradient',
            learning_rate=0.01
        )
        
        # Should be implemented if gradient method is supported
        try:
            updated = updater.update_factor(
                self.tensor,
                self.weights,
                self.factors,
                mode=0
            )
            
            # If implemented, check properties
            self.assertEqual(updated.shape, (10, 3))
            self.assertFalse(np.any(np.isnan(updated)))
        except NotImplementedError:
            # Gradient method might not be implemented yet
            pass


class TestSparseHandler(unittest.TestCase):
    """Test sparse tensor handling."""
    
    def setUp(self):
        """Set up sparse test data."""
        np.random.seed(42)
        self.shape = (50, 50, 20)
        
        # Create very sparse tensor
        self.dense = np.random.rand(*self.shape)
        self.mask = np.random.rand(*self.shape) > 0.95  # 95% sparse
        self.sparse_tensor = self.dense * self.mask
    
    def test_sparsity_detection(self):
        """Test sparsity level detection."""
        handler = SparseHandler()
        
        sparsity = handler.compute_sparsity(self.sparse_tensor)
        
        # Should be close to 0.95
        self.assertGreater(sparsity, 0.94)
        self.assertLess(sparsity, 0.96)
    
    def test_sparse_format_conversion(self):
        """Test conversion to/from sparse formats."""
        handler = SparseHandler()
        
        # Convert to sparse format
        sparse_data = handler.to_sparse_format(
            self.sparse_tensor,
            threshold=0.9
        )
        
        # Should have indices and values
        self.assertIn('indices', sparse_data)
        self.assertIn('values', sparse_data)
        self.assertIn('shape', sparse_data)
        
        # Number of stored values should match non-zeros
        n_nonzero = np.count_nonzero(self.sparse_tensor)
        self.assertEqual(len(sparse_data['values']), n_nonzero)
        
        # Convert back
        reconstructed = handler.from_sparse_format(sparse_data)
        
        # Should match original
        np.testing.assert_array_almost_equal(
            reconstructed, 
            self.sparse_tensor
        )
    
    def test_sparse_operations(self):
        """Test operations on sparse tensors."""
        handler = SparseHandler()
        
        # Test sparse norm
        sparse_norm = handler.sparse_norm(self.sparse_tensor)
        dense_norm = np.linalg.norm(self.sparse_tensor)
        
        self.assertAlmostEqual(sparse_norm, dense_norm, places=6)
        
        # Test sparse inner product
        other = np.random.rand(*self.shape) * self.mask
        
        sparse_inner = handler.sparse_inner_product(
            self.sparse_tensor, 
            other,
            self.mask
        )
        dense_inner = np.sum(self.sparse_tensor * other)
        
        self.assertAlmostEqual(sparse_inner, dense_inner, places=6)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of sparse operations."""
        handler = SparseHandler()
        
        # Estimate memory usage
        dense_memory = self.sparse_tensor.nbytes
        sparse_memory = handler.estimate_sparse_memory(
            self.sparse_tensor,
            threshold=0.9
        )
        
        # Sparse should use much less memory
        compression_ratio = dense_memory / sparse_memory
        self.assertGreater(compression_ratio, 10)  # At least 10x compression


class TestTensorMetricsCalculator(unittest.TestCase):
    """Test tensor metrics computation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.shape = (10, 10, 5)
        self.original = np.random.rand(*self.shape)
        self.weights = np.ones(self.shape)
        
        # Create approximate reconstruction
        self.reconstructed = self.original + 0.1 * np.random.randn(*self.shape)
        
        self.calculator = TensorMetricsCalculator()
    
    def test_fit_calculation(self):
        """Test fit metric calculation."""
        fit = self.calculator.calculate_fit(
            self.original,
            self.reconstructed,
            self.weights
        )
        
        # Should be between 0 and 1
        self.assertGreaterEqual(fit, 0)
        self.assertLessEqual(fit, 1)
        
        # Should be high for good reconstruction
        self.assertGreater(fit, 0.8)
        
        # Perfect reconstruction
        perfect_fit = self.calculator.calculate_fit(
            self.original,
            self.original,
            self.weights
        )
        self.assertAlmostEqual(perfect_fit, 1.0, places=6)
    
    def test_relative_error(self):
        """Test relative error calculation."""
        rel_error = self.calculator.calculate_relative_error(
            self.original,
            self.reconstructed,
            self.weights
        )
        
        # Should be positive
        self.assertGreater(rel_error, 0)
        
        # Should be small for good reconstruction
        self.assertLess(rel_error, 0.2)
        
        # Perfect reconstruction
        perfect_error = self.calculator.calculate_relative_error(
            self.original,
            self.original,
            self.weights
        )
        self.assertAlmostEqual(perfect_error, 0.0, places=6)
    
    def test_mode_wise_fit(self):
        """Test fit calculation by mode."""
        mode_fits = self.calculator.calculate_mode_wise_fit(
            self.original,
            self.reconstructed,
            self.weights
        )
        
        # Should have fit for each mode
        self.assertEqual(len(mode_fits), 3)
        
        # All should be between 0 and 1
        for mode, fit in mode_fits.items():
            self.assertGreaterEqual(fit, 0)
            self.assertLessEqual(fit, 1)
    
    def test_component_importance(self):
        """Test component importance calculation."""
        # Create factors
        factors = [
            np.random.rand(10, 4),
            np.random.rand(10, 4),
            np.random.rand(5, 4)
        ]
        
        # Make first component stronger
        for f in factors:
            f[:, 0] *= 5
        
        importance = self.calculator.calculate_component_importance(
            CPFactors(
                factors=factors,
                rank=4,
                mode_names=['mode0', 'mode1', 'mode2']
            )
        )
        
        # Should have 4 values
        self.assertEqual(len(importance), 4)
        
        # First should be most important
        self.assertEqual(np.argmax(importance), 0)
        
        # Should sum to 1
        self.assertAlmostEqual(np.sum(importance), 1.0, places=6)
    
    def test_core_consistency(self):
        """Test core consistency diagnostic."""
        factors = [
            np.random.rand(10, 3),
            np.random.rand(10, 3),
            np.random.rand(5, 3)
        ]
        
        # For perfect CP model, core consistency should be high
        # Create perfect CP tensor
        perfect_cp = np.zeros(self.shape)
        for r in range(3):
            component = np.outer(
                np.outer(factors[0][:, r], factors[1][:, r]).ravel(),
                factors[2][:, r]
            ).reshape(self.shape)
            perfect_cp += component
        
        consistency = self.calculator.calculate_core_consistency(
            perfect_cp,
            CPFactors(
                factors=factors,
                rank=3,
                mode_names=['mode0', 'mode1', 'mode2']
            )
        )
        
        # Should be close to 100% for perfect CP
        self.assertGreater(consistency, 0.95)
    
    def test_explained_variance(self):
        """Test explained variance calculation."""
        factors = [
            np.random.rand(10, 3),
            np.random.rand(10, 3),
            np.random.rand(5, 3)
        ]
        
        explained_var = self.calculator.calculate_explained_variance(
            self.original,
            CPFactors(
                factors=factors,
                rank=3,
                mode_names=['mode0', 'mode1', 'mode2']
            ),
            self.weights
        )
        
        # Should have variance for each component
        self.assertEqual(len(explained_var), 3)
        
        # Should be non-negative
        self.assertTrue(all(var >= 0 for var in explained_var))
        
        # Should be decreasing (components ordered by importance)
        self.assertTrue(all(
            explained_var[i] >= explained_var[i+1] 
            for i in range(len(explained_var)-1)
        ))


if __name__ == '__main__':
    unittest.main()