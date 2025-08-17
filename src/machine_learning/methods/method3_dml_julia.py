"""Method 3: Double Machine Learning (Julia-based simplified version).

Tests causal effects using DML methodology with Julia results.
This is a simplified version that works with Julia GLM results.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from scipy import stats

from ..config_loader import MLRegressionConfig
from ..utils import JuliaManager

logger = logging.getLogger(__name__)


class Method3DML:
    """Method 3 - Simplified DML using Julia results.
    
    Tests H0 vs H1 (biodiversity begets biodiversity) using DML approach.
    This simplified version uses the Julia DML results directly.
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 3 DML.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        self.julia = JuliaManager.get_or_init()
        self.model = None
        self.causal_effects = None
    
    def fit_model(self, data_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fit DML model (or load from Julia results).
        
        Args:
            data_metadata: Metadata about data in Julia
            
        Returns:
            Model fitting results
        """
        logger.info("Method 3 DML: Using Julia results")
        
        # In the simplified version, we just use the Julia results
        # Real DML would require complex cross-fitting
        return {
            'method': 'DML',
            'status': 'simplified_version',
            'note': 'Using Julia GLM results as proxy for DML'
        }
    
    def estimate_causal_effects(self) -> Dict[str, Any]:
        """Estimate causal effects from model.
        
        Returns:
            Causal effect estimates
        """
        if self.model is None:
            return {'error': 'Model not fitted'}
        
        # Extract biodiversity coefficients as causal effects
        effects = {}
        
        coef_names = self.model.get('coef_names', [])
        coefficients = self.model.get('coefficients', [])
        std_errors = self.model.get('std_errors', [])
        
        # Find biodiversity variables
        biodiv_vars = {
            'plants_richness_log': 'Plants richness effect',
            'vertebrates_richness_log': 'Vertebrates richness effect'
        }
        
        for var, description in biodiv_vars.items():
            if var in coef_names:
                idx = coef_names.index(var)
                effects[var] = {
                    'description': description,
                    'estimate': coefficients[idx],
                    'std_error': std_errors[idx],
                    'z_statistic': coefficients[idx] / std_errors[idx],
                    'p_value': 2 * (1 - stats.norm.cdf(abs(coefficients[idx] / std_errors[idx])))
                }
        
        self.causal_effects = effects
        return effects
    
    def test_h0_h1_causal(self) -> Dict[str, Any]:
        """Test H0 vs H1 using causal framework.
        
        Returns:
            Test results
        """
        if self.causal_effects is None:
            self.causal_effects = self.estimate_causal_effects()
        
        results = {
            'hypothesis': 'H0: No causal effect of biodiversity on fungi richness',
            'alternative': 'H1: Biodiversity causes increased fungi richness',
            'method': 'Simplified DML (using Julia GLM)',
            'effects': self.causal_effects
        }
        
        # Overall test - any significant effect?
        any_significant = any(
            effect.get('p_value', 1) < 0.05 
            for effect in self.causal_effects.values()
        )
        
        results['conclusion'] = (
            'Reject H0: Evidence for causal biodiversity effect' 
            if any_significant 
            else 'Fail to reject H0: No clear causal evidence'
        )
        
        return results
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate report for Method 3.
        
        Args:
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with report components
        """
        report = {
            'method': 'Method 3: Simplified DML Analysis',
            'description': 'Causal inference using Julia GLM as proxy',
            'causal_effects': self.causal_effects,
            'hypothesis_test': self.test_h0_h1_causal() if self.causal_effects else None
        }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            report_file = output_dir / 'method3_dml_report.json'
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Method 3 report saved to {report_file}")
        
        return report
    
    def predict(self, **kwargs) -> np.ndarray:
        """Make predictions (placeholder for compatibility).
        
        Returns:
            Coefficients as placeholder
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        return self.model.get('coefficients', np.array([]))