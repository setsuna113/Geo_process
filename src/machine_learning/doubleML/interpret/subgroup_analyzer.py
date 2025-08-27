"""
Subgroup identification for heterogeneous treatment effects
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree


class SubgroupAnalyzer:
    """
    Identify subgroups with different treatment responses using decision trees
    
    Simple, interpretable approach to finding who responds differently
    """
    
    def __init__(self, estimator, effect_modifiers: list, treatment_names: list):
        """
        Args:
            estimator: Fitted EconML estimator
            effect_modifiers: Features that modify effects
            treatment_names: Treatment variable names
        """
        self.estimator = estimator
        self.effect_modifiers = effect_modifiers
        self.treatment_names = treatment_names
        self.trees = {}
        self.rules = {}
    
    def identify_subgroups(self, X: np.ndarray, 
                          max_depth: int = 2,
                          min_samples_leaf: int = 100,
                          min_effect_size: float = 0.1) -> Dict[str, Any]:
        """
        Find subgroups with distinct treatment effects
        
        Args:
            X: Feature matrix for effect modifiers
            max_depth: Maximum tree depth (keep shallow for interpretability)
            min_samples_leaf: Minimum samples per leaf
            min_effect_size: Minimum effect difference to consider meaningful
            
        Returns:
            Dictionary with subgroups for each treatment
        """
        # Get treatment effects for all samples
        effects = self.estimator.const_marginal_effect(X)
        
        results = {}
        
        for i, treatment in enumerate(self.treatment_names):
            # Fit tree to predict heterogeneous effects
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            treatment_effects = effects[:, i]
            tree.fit(X, treatment_effects)
            
            # Store tree
            self.trees[treatment] = tree
            
            # Extract subgroups
            subgroups = self._extract_subgroups(tree, X, treatment_effects, min_effect_size)
            
            # Get feature importance
            importance = dict(zip(self.effect_modifiers, tree.feature_importances_))
            
            results[treatment] = {
                'n_subgroups': len(subgroups),
                'subgroups': subgroups,
                'feature_importance': importance,
                'tree_depth': tree.get_depth(),
                'effect_range': (float(np.min(treatment_effects)), 
                               float(np.max(treatment_effects)))
            }
            
            # Print summary
            print(f"\n{treatment}:")
            print(f"  Found {len(subgroups)} distinct subgroups")
            print(f"  Effect range: [{results[treatment]['effect_range'][0]:.4f}, "
                  f"{results[treatment]['effect_range'][1]:.4f}]")
            
            # Print top splitting features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_features[0][1] > 0:
                print(f"  Key splitter: {top_features[0][0]} (importance: {top_features[0][1]:.3f})")
        
        return results
    
    def _extract_subgroups(self, tree, X: np.ndarray, effects: np.ndarray, 
                          min_effect_size: float) -> List[Dict[str, Any]]:
        """
        Extract interpretable subgroups from decision tree
        """
        # Get leaf assignments
        leaf_ids = tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        subgroups = []
        
        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            n_samples = np.sum(mask)
            
            if n_samples < 10:  # Skip tiny groups
                continue
            
            leaf_effects = effects[mask]
            mean_effect = np.mean(leaf_effects)
            
            # Skip if effect is too small
            if np.abs(mean_effect) < min_effect_size:
                continue
            
            # Get the path to this leaf
            path = self._get_leaf_path(tree, leaf_id)
            
            # Characterize the subgroup
            X_subgroup = X[mask]
            characteristics = {}
            for j, feature in enumerate(self.effect_modifiers):
                characteristics[feature] = {
                    'mean': float(np.mean(X_subgroup[:, j])),
                    'std': float(np.std(X_subgroup[:, j])),
                    'min': float(np.min(X_subgroup[:, j])),
                    'max': float(np.max(X_subgroup[:, j]))
                }
            
            subgroups.append({
                'leaf_id': int(leaf_id),
                'n_samples': int(n_samples),
                'mean_effect': float(mean_effect),
                'effect_std': float(np.std(leaf_effects)),
                'rules': path,
                'characteristics': characteristics,
                'response_type': self._classify_response(mean_effect)
            })
        
        # Sort by effect size
        subgroups.sort(key=lambda x: abs(x['mean_effect']), reverse=True)
        
        return subgroups
    
    def _get_leaf_path(self, tree, leaf_id: int) -> str:
        """
        Extract human-readable rules for reaching a leaf
        """
        # Simplified version - in practice would traverse tree structure
        # This returns a placeholder
        return f"Leaf {leaf_id}"
    
    def _classify_response(self, effect: float) -> str:
        """
        Classify response type based on effect size
        """
        if effect > 0.5:
            return "strong_positive"
        elif effect > 0.1:
            return "moderate_positive"
        elif effect > -0.1:
            return "neutral"
        elif effect > -0.5:
            return "moderate_negative"
        else:
            return "strong_negative"
    
    def get_subgroup_assignment(self, X: np.ndarray, treatment: str) -> np.ndarray:
        """
        Assign samples to identified subgroups
        
        Args:
            X: Feature matrix
            treatment: Which treatment's subgroups to use
            
        Returns:
            Array of subgroup IDs
        """
        if treatment not in self.trees:
            raise ValueError(f"No subgroups identified for {treatment}. Run identify_subgroups first.")
        
        return self.trees[treatment].apply(X)
    
    def compare_subgroups(self, X: np.ndarray, group1_mask: np.ndarray, 
                         group2_mask: np.ndarray) -> Dict[str, Any]:
        """
        Compare treatment effects between two groups
        
        Args:
            X: Feature matrix
            group1_mask: Boolean mask for group 1
            group2_mask: Boolean mask for group 2
            
        Returns:
            Comparison statistics
        """
        effects_g1 = self.estimator.const_marginal_effect(X[group1_mask])
        effects_g2 = self.estimator.const_marginal_effect(X[group2_mask])
        
        comparisons = {}
        
        for i, treatment in enumerate(self.treatment_names):
            mean_g1 = np.mean(effects_g1[:, i])
            mean_g2 = np.mean(effects_g2[:, i])
            
            # Simple t-test
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(effects_g1[:, i], effects_g2[:, i])
            
            comparisons[treatment] = {
                'group1_mean': float(mean_g1),
                'group2_mean': float(mean_g2),
                'difference': float(mean_g1 - mean_g2),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significantly_different': bool(p_value < 0.05)
            }
        
        return comparisons
    
    def plot_tree(self, treatment: str, figsize: tuple = (15, 10)) -> plt.Figure:
        """
        Visualize the decision tree for a treatment
        
        Args:
            treatment: Which treatment's tree to plot
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if treatment not in self.trees:
            raise ValueError(f"No tree for {treatment}. Run identify_subgroups first.")
        
        tree = self.trees[treatment]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_tree(tree, 
                 feature_names=self.effect_modifiers,
                 filled=True,
                 rounded=True,
                 ax=ax,
                 fontsize=10,
                 proportion=True)
        
        ax.set_title(f'Treatment Effect Subgroups - {treatment}', fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_forest(self, X: np.ndarray, treatments: Optional[List[str]] = None,
                   figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Create forest plot showing treatment effects across subgroups
        
        Args:
            X: Feature matrix
            treatments: Which treatments to plot (None = all)
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if treatments is None:
            treatments = self.treatment_names
        
        # Get effects for all samples
        effects = self.estimator.const_marginal_effect(X)
        
        fig, axes = plt.subplots(len(treatments), 1, figsize=figsize)
        
        if len(treatments) == 1:
            axes = [axes]
        
        for idx, treatment in enumerate(treatments):
            ax = axes[idx]
            treat_idx = self.treatment_names.index(treatment)
            
            if treatment not in self.trees:
                continue
            
            # Get subgroup assignments
            tree = self.trees[treatment]
            leaf_ids = tree.apply(X)
            unique_leaves = np.unique(leaf_ids)
            
            # Calculate statistics for each subgroup
            subgroup_stats = []
            for leaf_id in unique_leaves:
                mask = leaf_ids == leaf_id
                if np.sum(mask) < 10:  # Skip small groups
                    continue
                
                leaf_effects = effects[mask, treat_idx]
                mean_effect = np.mean(leaf_effects)
                se = np.std(leaf_effects) / np.sqrt(len(leaf_effects))
                ci_lower = mean_effect - 1.96 * se
                ci_upper = mean_effect + 1.96 * se
                
                subgroup_stats.append({
                    'leaf_id': leaf_id,
                    'n': np.sum(mask),
                    'mean': mean_effect,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
            
            # Sort by effect size
            subgroup_stats.sort(key=lambda x: x['mean'])
            
            # Plot forest plot
            y_pos = np.arange(len(subgroup_stats))
            
            for i, stats in enumerate(subgroup_stats):
                # Error bar
                ax.errorbar(stats['mean'], i, 
                          xerr=[[stats['mean'] - stats['ci_lower']], 
                                [stats['ci_upper'] - stats['mean']]],
                          fmt='o', color='steelblue', markersize=8,
                          capsize=5, capthick=2)
                
                # Add sample size
                ax.text(stats['ci_upper'] + 0.001, i, f"n={stats['n']}", 
                       va='center', fontsize=8, alpha=0.7)
            
            # Formatting
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"Subgroup {s['leaf_id']}" for s in subgroup_stats])
            ax.set_xlabel('Treatment Effect')
            ax.set_title(f'{treatment}')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Treatment Effects by Subgroup (95% CI)', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    
    def plot_subgroup_characteristics(self, X: np.ndarray, treatment: str,
                                     figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot distributions of features within each subgroup
        
        Args:
            X: Feature matrix
            treatment: Which treatment
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if treatment not in self.trees:
            raise ValueError(f"No subgroups for {treatment}. Run identify_subgroups first.")
        
        tree = self.trees[treatment]
        leaf_ids = tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        # Filter to meaningful subgroups
        valid_leaves = [leaf for leaf in unique_leaves 
                       if np.sum(leaf_ids == leaf) >= 10]
        
        n_features = len(self.effect_modifiers)
        n_subgroups = len(valid_leaves)
        
        fig, axes = plt.subplots(n_features, n_subgroups, figsize=figsize)
        
        if n_features == 1:
            axes = axes.reshape(1, -1)
        if n_subgroups == 1:
            axes = axes.reshape(-1, 1)
        
        # Get treatment effects
        effects = self.estimator.const_marginal_effect(X)
        treat_idx = self.treatment_names.index(treatment)
        
        for col, leaf_id in enumerate(valid_leaves):
            mask = leaf_ids == leaf_id
            n_samples = np.sum(mask)
            mean_effect = np.mean(effects[mask, treat_idx])
            
            for row, feature in enumerate(self.effect_modifiers):
                ax = axes[row, col]
                
                feature_vals = X[mask, row]
                
                # Violin plot or histogram
                if n_samples > 30:
                    ax.violinplot([feature_vals], positions=[0], showmeans=True)
                else:
                    ax.hist(feature_vals, bins=10, alpha=0.7, color='steelblue')
                
                # Add reference line for overall median
                overall_median = np.median(X[:, row])
                ax.axhline(y=overall_median, color='red', linestyle='--', 
                         alpha=0.5, label='Overall median')
                
                # Labels
                if row == 0:
                    ax.set_title(f'Subgroup {leaf_id}\nn={n_samples}\nEffect={mean_effect:.4f}')
                if col == 0:
                    ax.set_ylabel(feature)
                if row == n_features - 1:
                    ax.set_xlabel('Value')
        
        plt.suptitle(f'Feature Distributions by Subgroup - {treatment}', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_toc_curve(self, X: np.ndarray, treatment: str,
                      figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot Targeting Operator Characteristic (TOC) curve
        Shows cumulative treatment effect when treating top X% by predicted effect
        
        Args:
            X: Feature matrix
            treatment: Which treatment
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        treat_idx = self.treatment_names.index(treatment)
        
        # Get predicted effects
        effects = self.estimator.const_marginal_effect(X)[:, treat_idx]
        
        # Sort by predicted effect (descending)
        sorted_idx = np.argsort(effects)[::-1]
        sorted_effects = effects[sorted_idx]
        
        # Calculate cumulative average effect
        n_samples = len(effects)
        percentiles = np.arange(1, n_samples + 1) / n_samples * 100
        cumulative_effects = np.cumsum(sorted_effects) / np.arange(1, n_samples + 1)
        
        # Random policy baseline
        random_effect = np.mean(effects)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot TOC curve
        ax.plot(percentiles, cumulative_effects, 'b-', linewidth=2, 
               label='Targeted Policy')
        ax.axhline(y=random_effect, color='red', linestyle='--', 
                  label=f'Random Policy (ATE={random_effect:.4f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Fill area between targeted and random
        ax.fill_between(percentiles, cumulative_effects, random_effect,
                       where=(cumulative_effects > random_effect),
                       alpha=0.3, color='green', label='Gain from targeting')
        
        ax.set_xlabel('Percentage of Population Treated (%)')
        ax.set_ylabel('Average Treatment Effect')
        ax.set_title(f'TOC Curve - {treatment}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotation for optimal treatment percentage
        max_idx = np.argmax(cumulative_effects)
        ax.plot(percentiles[max_idx], cumulative_effects[max_idx], 'ro', markersize=8)
        ax.annotate(f'Optimal: {percentiles[max_idx]:.1f}%\nEffect: {cumulative_effects[max_idx]:.4f}',
                   xy=(percentiles[max_idx], cumulative_effects[max_idx]),
                   xytext=(percentiles[max_idx] + 10, cumulative_effects[max_idx]),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        plt.tight_layout()
        return fig


def create_subgroup_report(analyzer: SubgroupAnalyzer, X: np.ndarray,
                          save_path: Optional[str] = None) -> str:
    """
    Create a text report of subgroup analysis
    
    Args:
        analyzer: Fitted SubgroupAnalyzer
        X: Feature matrix
        save_path: Optional path to save report
        
    Returns:
        Report as string
    """
    results = analyzer.identify_subgroups(X)
    
    report = []
    report.append("="*60)
    report.append("SUBGROUP ANALYSIS REPORT")
    report.append("="*60)
    
    for treatment, data in results.items():
        report.append(f"\n{treatment.upper()}")
        report.append("-"*40)
        report.append(f"Number of subgroups: {data['n_subgroups']}")
        report.append(f"Effect range: {data['effect_range']}")
        
        # Feature importance
        report.append("\nKey drivers of heterogeneity:")
        for feat, imp in sorted(data['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True):
            if imp > 0.01:
                report.append(f"  - {feat}: {imp:.3f}")
        
        # Subgroups
        report.append("\nIdentified subgroups:")
        for sg in data['subgroups']:
            report.append(f"\n  Subgroup {sg['leaf_id']} ({sg['response_type']})")
            report.append(f"    Size: {sg['n_samples']} samples")
            report.append(f"    Mean effect: {sg['mean_effect']:.4f} ± {sg['effect_std']:.4f}")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {save_path}")
    
    return report_text