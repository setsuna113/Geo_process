"""Unit tests for tensor utilities."""

import unittest
import numpy as np

from src.biodiversity_analysis.shared.tensor_utils import (
    unfold, fold, khatri_rao_product, compute_mttkrp,
    cp_reconstruct, weighted_norm, calculate_fit,
    check_convergence, initialize_factors, validate_cp_factors,
    compute_gram_matrices
)
from src.core_abstractions import CPFactors


class TestTensorOperations(unittest.TestCase):
    """Test basic tensor operations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.tensor_3d = np.random.rand(4, 5, 3)
        self.tensor_4d = np.random.rand(3, 4, 5, 2)
    
    def test_unfold_fold_3d(self):
        """Test unfolding and folding for 3D tensor."""
        for mode in range(3):
            # Unfold
            unfolded = unfold(self.tensor_3d, mode)
            
            # Check shape
            mode_size = self.tensor_3d.shape[mode]
            other_size = self.tensor_3d.size // mode_size
            self.assertEqual(unfolded.shape, (mode_size, other_size))
            
            # Fold back
            refolded = fold(unfolded, mode, self.tensor_3d.shape)
            
            # Should be identical
            np.testing.assert_array_almost_equal(refolded, self.tensor_3d)
    
    def test_unfold_fold_4d(self):
        """Test unfolding and folding for 4D tensor."""
        for mode in range(4):
            unfolded = unfold(self.tensor_4d, mode)
            refolded = fold(unfolded, mode, self.tensor_4d.shape)
            np.testing.assert_array_almost_equal(refolded, self.tensor_4d)
    
    def test_khatri_rao_product(self):
        """Test Khatri-Rao product."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        B = np.array([[7, 8], [9, 10]])
        
        kr = khatri_rao_product(A, B)
        
        # Check shape
        self.assertEqual(kr.shape, (6, 2))  # (3*2, 2)
        
        # Check values
        expected_col_0 = [7, 9, 21, 27, 35, 45]
        np.testing.assert_array_equal(kr[:, 0], expected_col_0)
    
    def test_khatri_rao_product_multiple(self):
        """Test Khatri-Rao product with multiple matrices."""
        matrices = [
            np.random.rand(3, 2),
            np.random.rand(4, 2),
            np.random.rand(5, 2)
        ]
        
        kr = khatri_rao_product(*matrices)
        
        # Check shape
        self.assertEqual(kr.shape, (60, 2))  # (3*4*5, 2)
    
    def test_cp_reconstruct(self):
        """Test CP tensor reconstruction."""
        # Create factor matrices
        factors = [
            np.random.rand(4, 2),
            np.random.rand(5, 2),
            np.random.rand(3, 2)
        ]
        
        # Reconstruct
        reconstructed = cp_reconstruct(factors)
        
        # Check shape
        self.assertEqual(reconstructed.shape, (4, 5, 3))
        
        # Manual reconstruction for verification
        manual = np.zeros((4, 5, 3))
        for r in range(2):
            for i in range(4):
                for j in range(5):
                    for k in range(3):
                        manual[i, j, k] += factors[0][i, r] * factors[1][j, r] * factors[2][k, r]
        
        np.testing.assert_array_almost_equal(reconstructed, manual)
    
    def test_mttkrp(self):
        """Test Matricized Tensor Times Khatri-Rao Product."""
        # Create small tensor and factors
        tensor = np.random.rand(3, 4, 2)
        factors = [
            np.random.rand(3, 2),
            np.random.rand(4, 2),
            np.random.rand(2, 2)
        ]
        
        # Test MTTKRP for each mode
        for mode in range(3):
            result = compute_mttkrp(tensor, factors, mode)
            
            # Check shape
            expected_shape = (tensor.shape[mode], 2)
            self.assertEqual(result.shape, expected_shape)
            
            # Verify against manual computation
            # MTTKRP(X, factors, n) = X_(n) * KR(factors without n)
            X_unfold = unfold(tensor, mode)
            other_factors = [f for i, f in enumerate(factors) if i != mode]
            kr = khatri_rao_product(*other_factors[::-1])  # Reverse order
            manual_result = X_unfold @ kr
            
            np.testing.assert_array_almost_equal(result, manual_result)


class TestNormCalculations(unittest.TestCase):
    """Test norm and distance calculations."""
    
    def test_weighted_norm(self):
        """Test weighted Frobenius norm."""
        tensor = np.array([[1, 2], [3, 4]])
        weights = np.array([[1, 0], [1, 1]])
        
        # Weighted norm
        w_norm = weighted_norm(tensor, weights)
        
        # Manual calculation: sqrt(1^2 + 3^2 + 4^2) = sqrt(26)
        expected = np.sqrt(26)
        self.assertAlmostEqual(w_norm, expected)
    
    def test_weighted_norm_zero_weights(self):
        """Test weighted norm with all zero weights."""
        tensor = np.random.rand(5, 5)
        weights = np.zeros((5, 5))
        
        w_norm = weighted_norm(tensor, weights)
        self.assertEqual(w_norm, 0.0)
    
    def test_calculate_fit(self):
        """Test fit calculation."""
        tensor = np.random.rand(10, 10)
        weights = np.ones((10, 10))
        
        # Perfect reconstruction
        reconstructed = tensor.copy()
        fit = calculate_fit(tensor, reconstructed, weights)
        self.assertAlmostEqual(fit, 1.0)
        
        # Zero reconstruction
        reconstructed = np.zeros_like(tensor)
        fit = calculate_fit(tensor, reconstructed, weights)
        self.assertAlmostEqual(fit, 0.0)
        
        # Partial fit
        reconstructed = 0.5 * tensor
        fit = calculate_fit(tensor, reconstructed, weights)
        self.assertGreater(fit, 0.0)
        self.assertLess(fit, 1.0)


class TestInitialization(unittest.TestCase):
    """Test factor initialization methods."""
    
    def test_random_initialization(self):
        """Test random factor initialization."""
        shape = (10, 8, 5)
        rank = 3
        
        factors = initialize_factors(shape, rank, method='random', random_state=42)
        
        # Check number of factors
        self.assertEqual(len(factors), 3)
        
        # Check shapes
        self.assertEqual(factors[0].shape, (10, 3))
        self.assertEqual(factors[1].shape, (8, 3))
        self.assertEqual(factors[2].shape, (5, 3))
        
        # Check values are in [0, 1]
        for factor in factors:
            self.assertTrue(np.all(factor >= 0))
            self.assertTrue(np.all(factor <= 1))
    
    def test_svd_initialization(self):
        """Test SVD-based initialization."""
        shape = (6, 7, 4)
        rank = 2
        
        factors = initialize_factors(shape, rank, method='svd', random_state=42)
        
        # Check shapes
        self.assertEqual(len(factors), 3)
        for i, factor in enumerate(factors):
            self.assertEqual(factor.shape, (shape[i], rank))
    
    def test_custom_initialization(self):
        """Test custom initialization function."""
        shape = (5, 5, 5)
        rank = 2
        
        def custom_init(size, rank):
            return np.ones((size, rank)) * 0.5
        
        factors = initialize_factors(shape, rank, method=custom_init)
        
        # Check that custom function was used
        for factor in factors:
            np.testing.assert_array_equal(factor, np.ones(factor.shape) * 0.5)


class TestConvergenceChecking(unittest.TestCase):
    """Test convergence checking functionality."""
    
    def test_check_convergence_true(self):
        """Test convergence detection when converged."""
        errors = [1.0, 0.5, 0.25, 0.125, 0.124, 0.123]
        
        # Should converge (small relative change)
        converged = check_convergence(errors, threshold=0.01)
        self.assertTrue(converged)
    
    def test_check_convergence_false(self):
        """Test convergence detection when not converged."""
        errors = [1.0, 0.5, 0.25, 0.125]
        
        # Should not converge (large relative changes)
        converged = check_convergence(errors, threshold=0.01)
        self.assertFalse(converged)
    
    def test_check_convergence_min_iterations(self):
        """Test minimum iterations requirement."""
        errors = [1.0, 0.999]  # Very small change
        
        # Should not converge (too few iterations)
        converged = check_convergence(errors, threshold=0.01, min_iterations=5)
        self.assertFalse(converged)


class TestCPFactorsValidation(unittest.TestCase):
    """Test CP factors validation."""
    
    def test_validate_cp_factors_valid(self):
        """Test validation of valid CP factors."""
        factors = [
            np.random.rand(10, 3),
            np.random.rand(8, 3),
            np.random.rand(5, 3)
        ]
        
        cp_factors = CPFactors(
            factors=factors,
            rank=3,
            mode_names=['mode0', 'mode1', 'mode2']
        )
        
        # Should validate without error
        is_valid = validate_cp_factors(cp_factors)
        self.assertTrue(is_valid)
    
    def test_validate_cp_factors_inconsistent_rank(self):
        """Test validation with inconsistent ranks."""
        factors = [
            np.random.rand(10, 3),
            np.random.rand(8, 2),  # Wrong rank!
            np.random.rand(5, 3)
        ]
        
        cp_factors = CPFactors(
            factors=factors,
            rank=3,
            mode_names=['mode0', 'mode1', 'mode2']
        )
        
        with self.assertRaises(ValueError):
            validate_cp_factors(cp_factors)
    
    def test_validate_cp_factors_wrong_number_modes(self):
        """Test validation with wrong number of mode names."""
        factors = [
            np.random.rand(10, 3),
            np.random.rand(8, 3)
        ]
        
        cp_factors = CPFactors(
            factors=factors,
            rank=3,
            mode_names=['mode0', 'mode1', 'mode2']  # Too many names!
        )
        
        with self.assertRaises(ValueError):
            validate_cp_factors(cp_factors)


class TestGramMatrices(unittest.TestCase):
    """Test Gram matrix computations."""
    
    def test_compute_gram_matrices(self):
        """Test Gram matrix computation."""
        factors = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8], [9, 10]]),
            np.array([[11, 12], [13, 14]])
        ]
        
        # Compute all Gram matrices
        grams = compute_gram_matrices(factors)
        
        # Check number of matrices
        self.assertEqual(len(grams), 3)
        
        # Check shapes (all should be rank x rank)
        for gram in grams:
            self.assertEqual(gram.shape, (2, 2))
        
        # Verify first Gram matrix
        expected_gram_0 = factors[0].T @ factors[0]
        np.testing.assert_array_almost_equal(grams[0], expected_gram_0)
    
    def test_compute_gram_matrices_skip_mode(self):
        """Test Gram matrix computation with skipped mode."""
        factors = [
            np.random.rand(5, 2),
            np.random.rand(6, 2),
            np.random.rand(4, 2)
        ]
        
        # Skip mode 1
        grams = compute_gram_matrices(factors, skip_mode=1)
        
        # Should have 2 matrices (skipped mode 1)
        self.assertEqual(len(grams), 2)
        
        # Verify they correspond to modes 0 and 2
        np.testing.assert_array_almost_equal(grams[0], factors[0].T @ factors[0])
        np.testing.assert_array_almost_equal(grams[1], factors[2].T @ factors[2])


if __name__ == '__main__':
    unittest.main()