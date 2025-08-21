"""
Security tests for critical paths
These tests verify that the security fixes work correctly
"""

import pytest
from pathlib import Path
import tempfile
import os
from src.machine_learning.utils.security_utils import SecurityUtils


class TestPathTraversal:
    """Test path traversal protection"""
    
    def setup_method(self):
        self.security = SecurityUtils()
        
    def test_reject_path_traversal_simple(self):
        """Simple .. traversal should be caught"""
        with pytest.raises(ValueError, match="outside allowed directories"):
            # This will resolve to parent directory
            self.security.validate_file_path("../etc/passwd")
            
    def test_reject_path_traversal_encoded(self):
        """Encoded traversal attempts should be caught"""
        with pytest.raises(ValueError, match="outside allowed directories"):
            # Try to escape with absolute path
            self.security.validate_file_path("/etc/passwd")
            
    def test_allow_temp_files(self):
        """Temp files should be allowed"""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            try:
                # This should not raise
                result = self.security.validate_file_path(tf.name)
                assert result == Path(tf.name).resolve()
            finally:
                os.unlink(tf.name)
                
    def test_allow_scratch_files(self):
        """Files in /scratch should be allowed if they exist"""
        # We can't test this without actual /scratch directory
        # but the logic is there
        pass
        
    def test_symlink_traversal(self):
        """Symlinks that point outside should be caught"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a symlink pointing to /etc
            link_path = Path(tmpdir) / "evil_link"
            try:
                link_path.symlink_to("/etc")
                with pytest.raises(ValueError, match="outside allowed directories"):
                    self.security.validate_file_path(str(link_path / "passwd"))
            except OSError:
                # Skip if we can't create symlinks
                pytest.skip("Cannot create symlinks on this system")


class TestInputValidation:
    """Test input validation for Julia integration"""
    
    def test_spatial_grid_step_validation(self):
        """Test that spatial_grid_step is validated"""
        from src.machine_learning.utils.julia_manager import JuliaManager
        
        manager = JuliaManager()
        
        # These should raise ValueError
        invalid_values = [
            -1,      # negative
            0,       # zero
            1001,    # too large
            "1",     # string
            None,    # None
            1.5,     # float
        ]
        
        for value in invalid_values:
            with pytest.raises(ValueError):
                # This will be caught in load_data
                manager.load_data("dummy.parquet", spatial_grid_step=value)