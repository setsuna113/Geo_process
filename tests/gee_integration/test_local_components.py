"""Local tests for GEE integration components (no API calls)"""

import pytest
from pathlib import Path
import tempfile
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.gee_integration.config import GEEConfig, DatasetConfig, ResolutionConfig
from src.gee_integration.workflow.state_manager import WorkflowState, WorkflowStep
from src.gee_integration.upload.validators import TIFValidator
from src.gee_integration.download.converter import ParquetConverter


class TestConfig:
    """Test configuration loading and validation"""
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML"""
        config_content = """
gee:
  auth:
    method: "service_account"
    service_account_key: "/path/to/key.json"
  target_resolutions:
    fine:
      name: "1_8km"
      resolution: 0.016667
      description: "Fine resolution"
  inputs:
    biodiversity:
      - path: "test.tif"
        name: "test"
        band_name: "test_band"
    climate:
      variables: "all"
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(config_content)
        
        config = GEEConfig.from_yaml(str(config_file))
        
        assert config.auth_method == "service_account"
        assert len(config.resolutions) == 1
        assert "fine" in config.resolutions
        assert len(config.biodiversity_datasets) == 1
        assert len(config.climate_variables) == 19  # all variables
    
    def test_config_validation(self):
        """Test config validation"""
        config = GEEConfig()
        config.auth_method = "service_account"
        config.service_account_key = None  # Missing
        
        issues = config.validate()
        assert len(issues) > 0
        assert any("service account key" in issue.lower() for issue in issues)
    
    def test_resolution_config(self):
        """Test resolution configuration"""
        res = ResolutionConfig(
            name="test",
            resolution=0.1,
            description="Test resolution"
        )
        assert res.name == "test"
        assert res.resolution == 0.1


class TestWorkflowState:
    """Test workflow state management"""
    
    def test_state_initialization(self, tmp_path):
        """Test state initialization"""
        state_file = tmp_path / "state.json"
        state = WorkflowState(state_file)
        
        assert state.current_step == WorkflowStep.NOT_STARTED
        assert len(state.state['completed_steps']) == 0
    
    def test_state_persistence(self, tmp_path):
        """Test state save/load"""
        state_file = tmp_path / "state.json"
        
        # Create and update state
        state1 = WorkflowState(state_file)
        state1.update_step(WorkflowStep.AUTHENTICATED)
        state1.set_metadata("test_key", "test_value")
        
        # Load in new instance
        state2 = WorkflowState(state_file)
        assert state2.current_step == WorkflowStep.AUTHENTICATED
        assert state2.get_metadata("test_key") == "test_value"
    
    def test_step_progression(self, tmp_path):
        """Test workflow step progression"""
        state = WorkflowState(tmp_path / "state.json")
        
        # Progress through steps
        steps = [
            WorkflowStep.PREREQUISITES_VALIDATED,
            WorkflowStep.AUTHENTICATED,
            WorkflowStep.ASSETS_UPLOADED
        ]
        
        for step in steps:
            state.update_step(step)
            assert state.current_step == step
            assert state.is_step_complete(step)


class TestValidators:
    """Test TIF validators"""
    
    def test_tif_validator_missing_file(self):
        """Test validation of missing file"""
        validator = TIFValidator()
        result = validator.validate_file("/nonexistent/file.tif")
        
        assert not result['valid']
        assert len(result['issues']) > 0
        assert "not found" in result['issues'][0]
    
    @patch('rasterio.open')
    def test_tif_validator_valid_file(self, mock_rasterio):
        """Test validation of valid file"""
        # Mock rasterio
        mock_src = MagicMock()
        mock_src.width = 100
        mock_src.height = 100
        mock_src.count = 1
        mock_src.dtypes = ['float32']
        mock_src.crs = MagicMock(is_geographic=True)
        mock_src.bounds = MagicMock(left=0, right=10, bottom=0, top=10)
        mock_src.nodata = -9999
        mock_src.read.return_value = np.random.rand(100, 100)
        
        mock_rasterio.return_value.__enter__.return_value = mock_src
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
            validator = TIFValidator()
            result = validator.validate_file(tmp.name)
            
            assert result['valid']
            assert len(result['issues']) == 0
            assert result['metadata']['width'] == 100


class TestParquetConverter:
    """Test CSV to Parquet conversion"""
    
    def test_csv_conversion(self, tmp_path):
        """Test converting CSV to parquet"""
        # Create test CSV
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            'x': np.linspace(-180, 180, 100),
            'y': np.linspace(-90, 90, 100),
            'bio01': np.random.rand(100),
            'bio12': np.random.rand(100),
            'am_fungi': np.random.rand(100)
        })
        df.to_csv(csv_file, index=False)
        
        # Convert
        converter = ParquetConverter(tmp_path / "output")
        output = converter.convert_csv_files([csv_file], "test_res")
        
        assert output.exists()
        assert output.suffix == '.parquet'
        
        # Verify content
        result_df = pd.read_parquet(output)
        assert len(result_df) == 100
        assert all(col in result_df.columns for col in ['x', 'y', 'bio01'])
    
    def test_dataframe_cleaning(self, tmp_path):
        """Test dataframe cleaning"""
        converter = ParquetConverter(tmp_path)
        
        # Create dirty dataframe
        df = pd.DataFrame({
            'x': [1, 2, None, 4],
            'y': [1, 2, 3, None],
            'system:index': [1, 2, 3, 4],  # Should be removed
            '.geo': ['a', 'b', 'c', 'd'],  # Should be removed
            'data': [1, 2, 3, 4]
        })
        
        cleaned = converter._clean_dataframe(df)
        
        assert 'system:index' not in cleaned.columns
        assert '.geo' not in cleaned.columns
        assert len(cleaned) == 2  # Rows with null coordinates removed


class TestMockedGEE:
    """Test with mocked GEE components"""
    
    @patch('ee.Initialize')
    @patch('ee.Image')
    def test_gee_authentication_mock(self, mock_image, mock_init):
        """Test GEE authentication with mocks"""
        from src.gee_integration.auth import GEEAuthenticator
        
        # Mock successful auth
        mock_init.return_value = None
        mock_image.return_value.getInfo.return_value = {'bands': [{'id': 'bio01'}]}
        
        auth = GEEAuthenticator()
        # Would test auth here but need to mock ee module import
    
    def test_workflow_dry_run(self, tmp_path):
        """Test workflow without actual GEE calls"""
        # This would test the workflow orchestrator
        # with all GEE calls mocked
        pass


class TestIntegration:
    """Integration tests for local components"""
    
    def test_config_to_workflow(self, tmp_path):
        """Test config flows through workflow"""
        # Create minimal config
        config_content = """
gee:
  target_resolutions:
    test:
      name: "test"
      resolution: 0.1
      description: "Test"
"""
        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)
        
        # Load config
        config = GEEConfig.from_yaml(str(config_file))
        
        # Create workflow state
        state = WorkflowState(tmp_path / "state.json")
        
        # Verify config can be used in workflow
        assert len(config.resolutions) == 1
        assert state.current_step == WorkflowStep.NOT_STARTED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])