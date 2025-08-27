"""Unified Model Comparison - ONE source of truth for model evaluation.

No duplication. Just metrics and comparison logic.
Visualization delegated to visualization module.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from scipy import stats
import logging
logger = logging.getLogger(__name__)


class ModelComparison:
    """Model comparison using information criteria and hypothesis tests.
    
    This is the ONLY ModelComparison class. No duplicates!
    Handles AIC, BIC, LRT, and model weights calculation.
    """
    
    @staticmethod
    def compare_models(models: Dict[str, Any]) -> Dict[str, Any]:
        """Compare models using information criteria.
        
        Args:
            models: Dictionary of models with their metrics
                   Each model should have: aic, bic, deviance, n_params, n_obs
                   
        Returns:
            Dictionary with comparison metrics and rankings
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        model_names = list(models.keys())
        results = {
            'models': {},
            'weights': {},
            'best_model': None
        }
        
        # Extract metrics from each model
        aics = []
        bics = []
        deviances = []
        null_deviances = []
        logliks = []
        null_logliks = []
        n_params_list = []
        
        # Track n_observations from first model for threshold calculation
        n_observations = None
        
        for name, model in models.items():
            # Handle different model formats - STANDARDIZED NAMING
            if isinstance(model, dict):
                aic = model.get('aic')
                bic = model.get('bic')
                deviance = model.get('deviance')
                null_deviance = model.get('null_deviance')
                loglik = model.get('loglik')
                null_loglik = model.get('null_loglik')
                # Standardize to n_parameters
                n_params = model.get('n_parameters', model.get('n_params'))
                # Standardize to n_observations
                n_obs = model.get('n_observations', model.get('n_obs'))
            else:
                # Model object
                aic = getattr(model, 'aic', None)
                bic = getattr(model, 'bic', None)
                deviance = getattr(model, 'deviance', None)
                null_deviance = getattr(model, 'null_deviance', None)
                loglik = getattr(model, 'loglik', None)
                null_loglik = getattr(model, 'null_loglik', None)
                n_params = getattr(model, 'n_parameters', getattr(model, 'n_params', None))
                n_obs = getattr(model, 'n_observations', getattr(model, 'n_obs', None))
                
                # Call if method
                if callable(aic): aic = aic()
                if callable(bic): bic = bic()
                if callable(deviance): deviance = deviance()
                if callable(null_deviance): null_deviance = null_deviance()
                if callable(loglik): loglik = loglik()
                if callable(null_loglik): null_loglik = null_loglik()
            
            # Store n_observations from first model that has it
            if n_observations is None and n_obs is not None:
                n_observations = n_obs
            
            aics.append(aic)
            bics.append(bic)
            deviances.append(deviance)
            null_deviances.append(null_deviance)
            logliks.append(loglik)
            null_logliks.append(null_loglik)
            n_params_list.append(n_params)
            
            # Calculate R-squared values if we have the necessary data
            mcfadden_r2 = None
            deviance_r2 = None
            nagelkerke_r2 = None
            
            if loglik is not None and null_loglik is not None:
                # McFadden's R² = 1 - (log likelihood_model / log likelihood_null)
                mcfadden_r2 = ModelComparison.calculate_mcfadden_r2(loglik, null_loglik)
                
                # Nagelkerke's R² - scaled version that can reach 1
                if n_obs is not None:
                    nagelkerke_r2 = ModelComparison.calculate_nagelkerke_r2(loglik, null_loglik, n_obs)
            
            if deviance is not None and null_deviance is not None:
                # Deviance-based R² = 1 - (residual deviance / null deviance)
                deviance_r2 = ModelComparison.calculate_deviance_r2(deviance, null_deviance)
            
            results['models'][name] = {
                'aic': aic,
                'bic': bic,
                'deviance': deviance,
                'null_deviance': null_deviance,
                'loglik': loglik,
                'null_loglik': null_loglik,
                'n_parameters': n_params,
                'n_observations': n_obs,
                'mcfadden_r2': mcfadden_r2,
                'deviance_r2': deviance_r2,
                'nagelkerke_r2': nagelkerke_r2
            }
        
        # Calculate delta values (difference from best) with error handling
        valid_aics = [a for a in aics if a is not None]
        valid_bics = [b for b in bics if b is not None]
        
        if not valid_aics:
            raise ValueError("No valid AIC values found in any model")
        if not valid_bics:
            raise ValueError("No valid BIC values found in any model")
            
        min_aic = min(valid_aics)
        min_bic = min(valid_bics)
        
        for i, name in enumerate(model_names):
            if aics[i] is not None:
                delta_aic = aics[i] - min_aic
                results['models'][name]['delta_aic'] = delta_aic
            
            if bics[i] is not None:
                delta_bic = bics[i] - min_bic
                results['models'][name]['delta_bic'] = delta_bic
        
        # Calculate AIC weights (Akaike weights)
        if all(a is not None for a in aics):
            delta_aics = [a - min_aic for a in aics]
            exp_terms = [np.exp(-0.5 * d) for d in delta_aics]
            sum_exp = sum(exp_terms)
            aic_weights = [e / sum_exp for e in exp_terms]
            
            for name, weight in zip(model_names, aic_weights):
                results['weights'][name] = {'aic_weight': weight}
        
        # Determine best model
        best_by_aic = model_names[aics.index(min_aic)] if min_aic else None
        best_by_bic = model_names[bics.index(min_bic)] if min_bic else None
        
        results['best_model'] = {
            'by_aic': best_by_aic,
            'by_bic': best_by_bic,
            'consensus': best_by_aic if best_by_aic == best_by_bic else None
        }
        
        # Calculate evidence ratios for all models
        evidence_ratios = {}
        for name in model_names:
            delta_aic = results['models'][name].get('delta_aic', 0)
            evidence_ratios[name] = ModelComparison.calculate_evidence_ratio(delta_aic)
        results['evidence_ratios'] = evidence_ratios
        
        # Add interpretations with sample-size adjusted thresholds
        if n_observations:
            threshold = ModelComparison.get_aic_threshold(n_observations)
            results['threshold'] = threshold
            results['n_observations'] = n_observations  # Store for visualization
            
            # Check if differences are meaningful
            for name in model_names:
                if name != best_by_aic:
                    delta = results['models'][name].get('delta_aic', 0)
                    results['models'][name]['meaningful_difference'] = delta > threshold
        
        # Calculate improvement metrics if we have null and full models
        if 'null' in results['models'] and 'full' in results['models']:
            null_model = results['models']['null']
            full_model = results['models']['full']
            results['improvement'] = {}
            
            # Calculate percent improvement in R² metrics
            if null_model.get('mcfadden_r2') is not None and full_model.get('mcfadden_r2') is not None:
                mcfadden_improvement = full_model['mcfadden_r2'] - null_model['mcfadden_r2']
                results['improvement']['mcfadden_r2'] = mcfadden_improvement
                results['improvement']['mcfadden_r2_pct'] = mcfadden_improvement * 100
            
            if null_model.get('deviance_r2') is not None and full_model.get('deviance_r2') is not None:
                deviance_improvement = full_model['deviance_r2'] - null_model['deviance_r2']
                results['improvement']['deviance_r2'] = deviance_improvement
                results['improvement']['deviance_r2_pct'] = deviance_improvement * 100
            
            if null_model.get('nagelkerke_r2') is not None and full_model.get('nagelkerke_r2') is not None:
                nagelkerke_improvement = full_model['nagelkerke_r2'] - null_model['nagelkerke_r2']
                results['improvement']['nagelkerke_r2'] = nagelkerke_improvement
                results['improvement']['nagelkerke_r2_pct'] = nagelkerke_improvement * 100
        
        # Create summary table DataFrame
        results['summary_table'] = ModelComparison.create_comparison_table(results)
        
        return results
    
    @staticmethod
    def get_aic_threshold(n_obs: int) -> float:
        """Get meaningful AIC difference threshold based on sample size.
        
        Replaces the entire aic_definitions.py file!
        
        Args:
            n_obs: Number of observations
            
        Returns:
            Threshold for meaningful AIC difference
        """
        if n_obs < 1000:
            return 2.0
        elif n_obs < 10000:
            return 10.0
        elif n_obs < 100000:
            return 50.0
        elif n_obs < 1000000:
            return 100.0
        else:  # > 1 million
            return 100 * np.log10(n_obs / 1000000 + 1)
    
    @staticmethod
    def calculate_evidence_ratio(delta_ic: float) -> float:
        """Calculate evidence ratio from delta IC.
        
        Evidence ratio = exp(delta_IC / 2)
        
        Args:
            delta_ic: Difference in information criterion
            
        Returns:
            Evidence ratio
        """
        return np.exp(delta_ic / 2)
    
    @staticmethod
    def calculate_mcfadden_r2(loglik: float, null_loglik: float) -> float:
        """Calculate McFadden's R² (log-likelihood based pseudo R²).
        
        McFadden's R² = 1 - (log likelihood_model / log likelihood_null)
        
        This is the original and most commonly used pseudo R² for GLMs.
        Values range from 0 to 1, with higher values indicating better fit.
        Values of 0.2-0.4 are considered excellent for GLMs.
        
        Args:
            loglik: Model log-likelihood
            null_loglik: Null model log-likelihood
            
        Returns:
            McFadden's R² value
        """
        if null_loglik == 0:
            return np.nan
        return 1 - (loglik / null_loglik)
    
    @staticmethod
    def calculate_deviance_r2(deviance: float, null_deviance: float) -> float:
        """Calculate deviance-based R².
        
        Deviance R² = 1 - (residual deviance / null deviance)
        
        Alternative pseudo R² formulation using deviances directly.
        Often equivalent to McFadden's R² but calculated differently.
        
        Args:
            deviance: Model residual deviance
            null_deviance: Null model deviance
            
        Returns:
            Deviance-based R² value
        """
        if null_deviance == 0:
            return np.nan
        return 1 - (deviance / null_deviance)
    
    @staticmethod
    def calculate_nagelkerke_r2(loglik: float, null_loglik: float, n_obs: int) -> float:
        """Calculate Nagelkerke's R² (scaled pseudo R²).
        
        Nagelkerke's R² = (1 - (L₀/L₁)^(2/n)) / (1 - L₀^(2/n))
        
        Where L₀ and L₁ are likelihood ratios (not log-likelihoods).
        This is a scaled version of Cox & Snell R² that can reach 1.
        
        Args:
            loglik: Model log-likelihood
            null_loglik: Null model log-likelihood  
            n_obs: Number of observations
            
        Returns:
            Nagelkerke's R² value
        """
        if n_obs == 0:
            return np.nan
            
        # Calculate Cox & Snell R²
        # R²_CS = 1 - exp((2/n) * (LL_null - LL_model))
        cox_snell = 1 - np.exp((2 / n_obs) * (null_loglik - loglik))
        
        # Calculate maximum possible Cox & Snell R²
        # R²_max = 1 - exp((2/n) * LL_null) = 1 - L₀^(2/n)
        max_cox_snell = 1 - np.exp((2 / n_obs) * null_loglik)
        
        if max_cox_snell == 0:
            return np.nan
            
        # Nagelkerke's R² = Cox & Snell R² / maximum possible Cox & Snell R²
        return cox_snell / max_cox_snell
    
    @staticmethod
    def create_comparison_table(results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary table from comparison results.
        
        Args:
            results: Results from compare_models()
            
        Returns:
            DataFrame with model comparison metrics
        """
        rows = []
        for name, metrics in results['models'].items():
            row = {
                'Model': name,
                'AIC': metrics.get('aic', np.nan),
                'ΔAIC': metrics.get('delta_aic', np.nan),
                'AIC Weight': results['weights'].get(name, {}).get('aic_weight', np.nan),
                'BIC': metrics.get('bic', np.nan),
                'ΔBIC': metrics.get('delta_bic', np.nan),
                'McFadden R²': metrics.get('mcfadden_r2', np.nan),
                'Deviance R²': metrics.get('deviance_r2', np.nan),
                'Nagelkerke R²': metrics.get('nagelkerke_r2', np.nan),
                'Parameters': metrics.get('n_parameters', np.nan)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        # Sort by AIC
        df = df.sort_values('AIC', na_position='last')
        return df