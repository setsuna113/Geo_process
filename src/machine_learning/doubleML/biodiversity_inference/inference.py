#!/usr/bin/env python3
"""
Biodiversity-specific inference and interpretation for DML results

This module provides domain-specific interpretation of DML results
for the biodiversity → fungi richness hypothesis, using EconML's
built-in inference correctly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
from scipy import stats as scipy_stats


class BiodiversityInference:
    """
    Domain-specific inference for biodiversity effects on fungi richness
    
    H0: Biodiversity does not provide additional information beyond abiotic features
    H1: Biodiversity provides additional information for fungi richness
    """
    
    def __init__(self, estimator=None, results_path: Optional[str] = None):
        """
        Initialize with either fitted EconML estimator or saved results
        
        Parameters:
        -----------
        estimator : LinearDML
            Fitted EconML estimator with inference
        results_path : str
            Path to saved results JSON
        """
        self.estimator = estimator
        self.results = None
        
        if results_path and Path(results_path).exists():
            with open(results_path, 'r') as f:
                self.results = json.load(f)
        
        self.treatment_names = ['plants_richness', 'vertebrates_richness', 'plants_x_vertebrates']
    
    def extract_inference_properly(self) -> Dict:
        """
        Extract inference from EconML estimator using CORRECT API
        """
        if not self.estimator:
            return self._extract_from_saved_results()
        
        # CORRECT: Use double underscore for parameter inference
        inference = self.estimator.intercept__inference()
        coef = self.estimator.intercept_
        
        # These ALWAYS exist if inference was run
        stderr = inference.stderr
        pvalues = inference.pvalue(value=0)  # Test H0: effect = 0
        conf_intervals = inference.conf_int(alpha=0.05)
        z_stats = inference.zstat(value=0)
        
        results = {}
        for i, name in enumerate(self.treatment_names):
            # Handle different confidence interval formats
            if isinstance(conf_intervals, tuple):
                # Tuple format: (lower_bounds, upper_bounds)
                ci_low = float(conf_intervals[0][i])
                ci_high = float(conf_intervals[1][i])
            elif hasattr(conf_intervals, 'ndim'):
                if conf_intervals.ndim == 2:
                    ci_low = float(conf_intervals[i, 0])
                    ci_high = float(conf_intervals[i, 1])
                else:
                    # Single treatment case
                    ci_low = float(conf_intervals[0])
                    ci_high = float(conf_intervals[1])
            else:
                # Fallback: assume it's iterable
                ci_low = float(conf_intervals[i][0])
                ci_high = float(conf_intervals[i][1])
            
            results[name] = {
                'coefficient': float(coef[i]),
                'std_error': float(stderr[i]),
                'z_statistic': float(z_stats[i]),
                'p_value': float(pvalues[i]),
                'ci_95': [ci_low, ci_high],
                'significant': bool(pvalues[i] < 0.05)
            }
        
        # Get summary table if available
        try:
            summary = self.estimator.summary()
            results['summary_table'] = str(summary)
        except:
            pass
        
        return results
    
    def _extract_from_saved_results(self) -> Dict:
        """Extract from saved JSON results"""
        if not self.results:
            raise ValueError("No estimator or saved results available")
        
        treatments = self.results.get('treatments', {})
        # Ensure 'significant' key exists
        for name, stats in treatments.items():
            if 'significant' not in stats and 'statistically_significant' in stats:
                stats['significant'] = stats['statistically_significant']
        
        return treatments
    
    def test_biodiversity_hypothesis(self) -> Dict:
        """
        Test the biodiversity hypothesis with proper interpretation
        """
        # Get treatment effects
        treatments = self.extract_inference_properly()
        
        print("\n" + "="*70)
        print("BIODIVERSITY → FUNGI RICHNESS HYPOTHESIS TEST")
        print("="*70)
        
        # 1. Individual Effects
        print("\n1. INDIVIDUAL BIODIVERSITY EFFECTS:")
        print("-"*50)
        
        significant_count = 0
        for name in self.treatment_names:
            if name not in treatments:
                continue
                
            stats = treatments[name]
            print(f"\n{self._get_readable_name(name)}:")
            print(f"  Effect size: {stats['coefficient']:.6f}")
            print(f"  Std error: {stats['std_error']:.6f}")
            print(f"  Z-statistic: {stats['z_statistic']:.4f}")
            print(f"  P-value: {stats['p_value']:.6f}")
            print(f"  95% CI: [{stats['ci_95'][0]:.6f}, {stats['ci_95'][1]:.6f}]")
            
            # Domain-specific interpretation
            interpretation = self._interpret_biodiversity_effect(name, stats)
            print(f"  Interpretation: {interpretation}")
            
            if stats['significant']:
                significant_count += 1
                print("  ✓ SIGNIFICANT at 5% level")
        
        # 2. Joint Significance Test
        print("\n2. JOINT BIODIVERSITY SIGNIFICANCE:")
        print("-"*50)
        
        # Always use proper covariance-based test when estimator is available
        joint_test_method = "not_available"
        wald_stat = None
        p_joint = None
        df = len(treatments)
        
        if self.estimator is not None:
            try:
                # Use EconML's proper covariance-based test
                inference = self.estimator.intercept__inference()
                coef = self.estimator.intercept_
                
                # Check for covariance matrix availability
                if hasattr(inference, 'var'):
                    # Compute proper Wald statistic: coef' * inv(Var) * coef
                    # This accounts for correlation between estimates
                    var_matrix = inference.var
                    
                    # Ensure var_matrix is properly shaped
                    if var_matrix.ndim == 1:
                        # Diagonal variance only
                        var_matrix = np.diag(var_matrix)
                    
                    try:
                        # Use more stable inverse computation
                        inv_var = np.linalg.inv(var_matrix + 1e-10 * np.eye(len(var_matrix)))
                        wald_stat = float(coef @ inv_var @ coef)
                        df = len(coef)  # Update df to actual number of coefficients
                        p_joint = 1 - scipy_stats.chi2.cdf(wald_stat, df)
                        joint_test_method = "econml_covariance_based"
                    except np.linalg.LinAlgError:
                        # Matrix is singular, use pseudo-inverse
                        inv_var = np.linalg.pinv(var_matrix)
                        wald_stat = float(coef @ inv_var @ coef)
                        df = len(coef)  # Update df to actual number of coefficients
                        p_joint = 1 - scipy_stats.chi2.cdf(wald_stat, df)
                        joint_test_method = "econml_pseudoinverse"
                elif hasattr(inference, 'stderr'):
                    # No full covariance, but we have standard errors
                    # This assumes independence (diagonal covariance)
                    stderr = inference.stderr
                    z_stats = coef / stderr
                    wald_stat = float(np.sum(z_stats ** 2))
                    p_joint = 1 - scipy_stats.chi2.cdf(wald_stat, df)
                    joint_test_method = "econml_diagonal_covariance"
                    print("Note: Using diagonal covariance (assumes independent treatment effects)")
                else:
                    raise AttributeError("No variance information available from EconML")
                    
            except Exception as e:
                print(f"ERROR: Cannot compute joint test with EconML: {e}")
                print("EconML estimator must be fitted with inference enabled.")
                # No fallback - require proper inference
                return None
        else:
            # For saved results, we don't have access to covariance
            # Must state this limitation clearly
            print("WARNING: Using saved results without access to covariance matrix.")
            print("Joint test assumes independence between treatment effects (may be inaccurate).")
            wald_stat = sum(t['z_statistic']**2 for t in treatments.values())
            p_joint = 1 - scipy_stats.chi2.cdf(wald_stat, df)
            joint_test_method = "simple_wald_independence_assumed"
        
        df = len(treatments)
        print(f"Wald χ² = {wald_stat:.2f} (df={df})")
        print(f"P-value = {p_joint:.6f}")
        print(f"Method: {joint_test_method}")
        
        if p_joint < 0.001:
            print("✓ Biodiversity variables are JOINTLY HIGHLY SIGNIFICANT")
        elif p_joint < 0.05:
            print("✓ Biodiversity variables are jointly significant")
        else:
            print("✗ Biodiversity variables are not jointly significant")
        
        # 3. Biodiversity Contribution Assessment
        print("\n3. BIODIVERSITY CONTRIBUTION TO FUNGI PREDICTION:")
        print("-"*50)
        
        # Try to get Y standard deviation if available
        y_std = None
        if self.estimator is not None and hasattr(self.estimator, '_cached_values'):
            if hasattr(self.estimator._cached_values, 'y') and self.estimator._cached_values.y is not None:
                y_std = np.std(self.estimator._cached_values.y)
        elif self.results:
            # Check for outcome_stats (new format) or outcome_std (legacy)
            if 'outcome_stats' in self.results and self.results['outcome_stats']:
                y_std = self.results['outcome_stats'].get('std')
            elif 'outcome_std' in self.results:
                y_std = self.results['outcome_std']
        
        contribution = self._assess_biodiversity_contribution(treatments, y_std=y_std)
        
        print(f"Statistical Significance (Mean Z-statistic): {contribution['mean_z_statistic']:.2f}")
        print(f"  → {contribution['statistical_assessment']}")
        
        if contribution['mean_cohen_d'] is not None:
            print(f"\nPractical Effect Size (Mean Cohen's d): {contribution['mean_cohen_d']:.4f}")
            print(f"  → {contribution['practical_assessment']}")
        else:
            print("\nPractical Effect Size: Not calculated (outcome std dev not available)")
        
        print(f"\nOverall Assessment: {contribution['overall_assessment']}")
        
        # 4. Final Decision
        print("\n" + "="*70)
        print("DECISION ON BIODIVERSITY HYPOTHESIS")
        print("="*70)
        
        decision = self._make_decision(significant_count, p_joint)
        print(f"\nDecision: {decision['decision']}")
        print(f"Evidence: {decision['evidence']}")
        print(f"Recommendation: {decision['recommendation']}")
        
        return {
            'treatments': treatments,
            'significant_count': significant_count,
            'wald_statistic': wald_stat,
            'p_value_joint': p_joint,
            'decision': decision,
            'contribution': contribution
        }
    
    def _get_readable_name(self, name: str) -> str:
        """Get human-readable treatment name"""
        mapping = {
            'plants_richness': 'Plant Species Richness',
            'vertebrates_richness': 'Vertebrate Species Richness',
            'plants_x_vertebrates': 'Plants × Vertebrates Interaction'
        }
        return mapping.get(name, name)
    
    def _interpret_biodiversity_effect(self, name: str, stats: Dict) -> str:
        """
        Provide biodiversity-specific interpretation of effects
        """
        coef = stats['coefficient']
        sig = stats['significant']
        
        if name == 'plants_richness':
            if not sig:
                return "Plant diversity has no detectable effect on fungi"
            elif coef > 0:
                return f"Each additional plant species → +{coef:.4f} fungi richness"
            else:
                return f"Plant diversity shows unexpected negative association"
        
        elif name == 'vertebrates_richness':
            if not sig:
                return "Vertebrate diversity has no detectable effect on fungi"
            elif coef > 0:
                return f"Each additional vertebrate species → +{coef:.4f} fungi richness"
            else:
                return f"Vertebrate diversity shows unexpected negative association"
        
        elif name == 'plants_x_vertebrates':
            if not sig:
                return "No interaction between plant and vertebrate diversity"
            elif coef < 0:
                return "Competitive interaction: biodiversity components substitute for each other"
            else:
                return "Synergistic interaction: biodiversity components reinforce each other"
        
        return "Unknown effect"
    
    def _assess_biodiversity_contribution(self, treatments: Dict, y_std: Optional[float] = None) -> Dict:
        """
        Assess the overall contribution of biodiversity to fungi prediction
        
        Parameters:
        -----------
        treatments : Dict
            Treatment effects dictionary
        y_std : float, optional
            Standard deviation of outcome Y for Cohen's d calculation
        """
        # Calculate Z-statistics (statistical significance)
        z_stats = []
        cohen_d_values = []
        
        for name in self.treatment_names:
            if name in treatments:
                # Z-statistic: effect / stderr (measures statistical significance)
                z = abs(treatments[name]['coefficient'] / treatments[name]['std_error'])
                z_stats.append(z)
                
                # Cohen's d: effect / Y_std (measures practical effect size)
                # Only calculate if Y standard deviation is provided
                if y_std and y_std > 0:
                    cohen_d = abs(treatments[name]['coefficient'] / y_std)
                    cohen_d_values.append(cohen_d)
        
        mean_z = np.mean(z_stats) if z_stats else 0
        mean_cohen_d = np.mean(cohen_d_values) if cohen_d_values else None
        
        # Assessment based on Z-statistics (statistical significance)
        if mean_z > 10:
            statistical_strength = "OVERWHELMING"
            stat_assessment = "Extremely strong statistical evidence for biodiversity effects"
        elif mean_z > 5:
            statistical_strength = "VERY STRONG"
            stat_assessment = "Strong statistical evidence for biodiversity effects"
        elif mean_z > 3:
            statistical_strength = "STRONG"
            stat_assessment = "Clear statistical evidence for biodiversity effects"
        elif mean_z > 2:
            statistical_strength = "MODERATE"
            stat_assessment = "Moderate statistical evidence for biodiversity effects"
        else:
            statistical_strength = "WEAK"
            stat_assessment = "Limited statistical evidence for biodiversity effects"
        
        # Assessment based on Cohen's d (practical effect size) if available
        practical_assessment = None
        if mean_cohen_d is not None:
            if mean_cohen_d > 0.8:
                practical_magnitude = "LARGE"
                practical_assessment = "Large practical effect of biodiversity on fungi"
            elif mean_cohen_d > 0.5:
                practical_magnitude = "MEDIUM"
                practical_assessment = "Medium practical effect of biodiversity on fungi"
            elif mean_cohen_d > 0.2:
                practical_magnitude = "SMALL"
                practical_assessment = "Small practical effect of biodiversity on fungi"
            else:
                practical_magnitude = "NEGLIGIBLE"
                practical_assessment = "Negligible practical effect despite statistical significance"
        
        # Combined assessment
        if mean_cohen_d is not None:
            # We have both statistical and practical significance
            if statistical_strength in ["OVERWHELMING", "VERY STRONG"] and practical_magnitude in ["LARGE", "MEDIUM"]:
                overall = "Biodiversity has both strong statistical and practical importance"
            elif statistical_strength in ["OVERWHELMING", "VERY STRONG"] and practical_magnitude in ["SMALL", "NEGLIGIBLE"]:
                overall = "Statistically significant but practically small biodiversity effects (large sample artifact)"
            else:
                overall = stat_assessment
        else:
            overall = stat_assessment
        
        return {
            'mean_z_statistic': mean_z,
            'mean_cohen_d': mean_cohen_d,
            'statistical_strength': statistical_strength,
            'statistical_assessment': stat_assessment,
            'practical_magnitude': practical_magnitude if mean_cohen_d else "Not calculated (Y std not provided)",
            'practical_assessment': practical_assessment if practical_assessment else "Not calculated",
            'overall_assessment': overall
        }
    
    def _make_decision(self, significant_count: int, p_joint: float) -> Dict:
        """
        Make final decision on biodiversity hypothesis
        """
        n_total = len(self.treatment_names)
        
        if significant_count == n_total and p_joint < 0.001:
            return {
                'decision': 'STRONGLY REJECT H0',
                'evidence': 'Overwhelming evidence for biodiversity effects',
                'recommendation': 'Essential to include biodiversity in fungi models'
            }
        elif significant_count >= 2 and p_joint < 0.01:
            return {
                'decision': 'REJECT H0',
                'evidence': 'Strong evidence for biodiversity effects',
                'recommendation': 'Include biodiversity variables in models'
            }
        elif significant_count >= 1 and p_joint < 0.05:
            return {
                'decision': 'LIKELY REJECT H0',
                'evidence': 'Moderate evidence for biodiversity effects',
                'recommendation': 'Consider including biodiversity variables'
            }
        elif significant_count >= 1:
            return {
                'decision': 'WEAK EVIDENCE AGAINST H0',
                'evidence': 'Limited evidence for biodiversity effects',
                'recommendation': 'Biodiversity may add marginal value'
            }
        else:
            return {
                'decision': 'FAIL TO REJECT H0',
                'evidence': 'No convincing evidence for biodiversity effects',
                'recommendation': 'Abiotic factors alone appear sufficient'
            }
    


def main():
    """Demonstrate proper usage"""
    
    # Use saved results
    results_path = "/scratch/yl998/econml_results/econml_results_latest.json"
    
    if Path(results_path).exists():
        print("Testing biodiversity hypothesis from saved results...")
        
        # Create inference object
        bio_inference = BiodiversityInference(results_path=results_path)
        
        # Test hypothesis
        results = bio_inference.test_biodiversity_hypothesis()
        
        # Note: DML does not provide traditional R² or variance explained metrics
        # It focuses on unbiased causal effect estimation instead
        print("\n" + "="*70)
        print("NOTE ON DML INTERPRETATION")
        print("="*70)
        print("Double Machine Learning (DML) provides:")
        print("  ✓ Unbiased causal effect estimates (θ₀)")
        print("  ✓ Valid confidence intervals and p-values")
        print("  ✓ Nuisance model performance (R² for first stage)")
        print("\nDML does NOT provide:")
        print("  ✗ Overall model R² or variance explained")
        print("  ✗ Partial R² for biodiversity contribution")
        print("  ✗ Traditional goodness-of-fit metrics")
        print("\nThis is by design - DML trades predictive metrics for causal validity.")
        
        return results
    else:
        print("No saved results found. Run econml_pipeline.py first.")
        return None


if __name__ == "__main__":
    results = main()