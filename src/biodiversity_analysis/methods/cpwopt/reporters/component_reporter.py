from src.abstractions import get_logger
"""Reporter for interpreting CP-WOPT components.

This module creates interpretable summaries of what each component represents,
including dominant features, spatial patterns, and ecological interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
import json
from src.abstractions import save_json

from src.abstractions import CPFactors, TensorData

logger = get_logger(__name__)


class ComponentReporter:
    """Creates interpretable summaries of tensor components."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize component reporter.
        
        Args:
            output_dir: Directory to save outputs
            config: Reporting configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.plot_format = config.get('plot_format', 'png')
        self.plot_dpi = config.get('plot_dpi', 150)
        self.n_top_features = config.get('top_n_features', 10)
        self.significance_threshold = config.get('significance_threshold', 0.1)
        
    def report(self, factors: CPFactors,
               feature_names: List[str],
               component_weights: Optional[np.ndarray] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate component interpretation report.
        
        Args:
            factors: CP factors
            feature_names: Names of features
            component_weights: Importance weights for components
            metadata: Additional metadata
            
        Returns:
            Dictionary with interpretation results and plot paths
        """
        results = {
            'plots': {},
            'interpretations': {},
            'statistics': {}
        }
        
        # Generate component summaries
        results['interpretations'] = self._generate_component_summaries(
            factors, feature_names, component_weights
        )
        
        # Plot component feature profiles
        results['plots']['feature_profiles'] = self._plot_feature_profiles(
            factors, feature_names
        )
        
        # Plot component similarities
        results['plots']['component_similarities'] = self._plot_component_similarities(
            factors
        )
        
        # Generate word clouds for top components
        if len(feature_names) > 5:  # Need enough features for word cloud
            results['plots']['word_clouds'] = self._generate_word_clouds(
                factors, feature_names, component_weights
            )
        
        # Plot component stability if we have multiple runs
        # (This would require multiple factor sets from stability analysis)
        
        # Calculate component statistics
        results['statistics'] = self._calculate_component_statistics(
            factors, component_weights
        )
        
        # Save interpretation to JSON
        self._save_interpretations(results['interpretations'])
        
        return results
    
    def _generate_component_summaries(self, factors: CPFactors,
                                    feature_names: List[str],
                                    component_weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate text summaries for each component.
        
        Args:
            factors: CP factors
            feature_names: Feature names
            component_weights: Component importance weights
            
        Returns:
            Dictionary of component interpretations
        """
        interpretations = {}
        
        # Get feature factor (usually last mode)
        feature_factor = factors.factors[-1]
        
        # Determine number of components to interpret
        n_interpret = min(10, factors.rank)  # Limit to top 10
        
        # Sort by importance if weights provided
        if component_weights is not None:
            component_order = np.argsort(component_weights)[::-1][:n_interpret]
        else:
            component_order = range(n_interpret)
        
        for idx, comp in enumerate(component_order):
            comp_dict = {
                'component_index': int(comp),
                'rank': idx + 1,
                'weight': float(component_weights[comp]) if component_weights is not None else None
            }
            
            # Get top positive and negative features
            loadings = feature_factor[:, comp]
            
            # Top positive features
            pos_idx = np.where(loadings > self.significance_threshold)[0]
            if len(pos_idx) > 0:
                pos_order = np.argsort(loadings[pos_idx])[::-1][:self.n_top_features]
                comp_dict['top_positive_features'] = [
                    {
                        'name': feature_names[pos_idx[i]],
                        'loading': float(loadings[pos_idx[i]])
                    }
                    for i in pos_order
                ]
            else:
                comp_dict['top_positive_features'] = []
            
            # Top negative features
            neg_idx = np.where(loadings < -self.significance_threshold)[0]
            if len(neg_idx) > 0:
                neg_order = np.argsort(loadings[neg_idx])[:self.n_top_features]
                comp_dict['top_negative_features'] = [
                    {
                        'name': feature_names[neg_idx[i]],
                        'loading': float(loadings[neg_idx[i]])
                    }
                    for i in neg_order
                ]
            else:
                comp_dict['top_negative_features'] = []
            
            # Generate interpretation text
            comp_dict['interpretation'] = self._interpret_component(
                comp_dict, comp, factors
            )
            
            # Add spatial characterization if available
            if factors.n_modes >= 2:
                comp_dict['spatial_pattern'] = self._characterize_spatial_pattern(
                    factors, comp
                )
            
            interpretations[f'component_{comp}'] = comp_dict
        
        return interpretations
    
    def _interpret_component(self, comp_dict: Dict[str, Any],
                           comp_idx: int,
                           factors: CPFactors) -> str:
        """Generate text interpretation for a component.
        
        Args:
            comp_dict: Component dictionary
            comp_idx: Component index
            factors: CP factors
            
        Returns:
            Interpretation text
        """
        interpretation = f"Component {comp_idx + 1}"
        
        if comp_dict['weight'] is not None:
            interpretation += f" (Weight: {comp_dict['weight']:.3f})"
        
        interpretation += " captures patterns characterized by:\n"
        
        # Positive features
        if comp_dict['top_positive_features']:
            pos_features = [f['name'] for f in comp_dict['top_positive_features'][:3]]
            interpretation += f"- High values of: {', '.join(pos_features)}\n"
        
        # Negative features
        if comp_dict['top_negative_features']:
            neg_features = [f['name'] for f in comp_dict['top_negative_features'][:3]]
            interpretation += f"- Low values of: {', '.join(neg_features)}\n"
        
        # Overall characterization
        n_pos = len(comp_dict['top_positive_features'])
        n_neg = len(comp_dict['top_negative_features'])
        
        if n_pos > 0 and n_neg > 0:
            interpretation += "- This represents a trade-off pattern\n"
        elif n_pos > 0:
            interpretation += "- This represents a co-occurrence pattern\n"
        elif n_neg > 0:
            interpretation += "- This represents an absence/avoidance pattern\n"
        
        return interpretation
    
    def _characterize_spatial_pattern(self, factors: CPFactors,
                                    comp: int) -> Dict[str, str]:
        """Characterize spatial pattern of a component.
        
        Args:
            factors: CP factors
            comp: Component index
            
        Returns:
            Dictionary with spatial characterization
        """
        lat_factor = factors.factors[0][:, comp]
        lon_factor = factors.factors[1][:, comp]
        
        characterization = {}
        
        # Latitude pattern
        lat_peak = np.argmax(np.abs(lat_factor))
        lat_range = np.where(np.abs(lat_factor) > 0.5 * np.max(np.abs(lat_factor)))[0]
        
        if len(lat_range) > 0:
            if lat_range[-1] - lat_range[0] < len(lat_factor) * 0.2:
                characterization['latitude'] = 'localized'
            else:
                characterization['latitude'] = 'widespread'
        else:
            characterization['latitude'] = 'uniform'
        
        # Longitude pattern
        lon_peak = np.argmax(np.abs(lon_factor))
        lon_range = np.where(np.abs(lon_factor) > 0.5 * np.max(np.abs(lon_factor)))[0]
        
        if len(lon_range) > 0:
            if lon_range[-1] - lon_range[0] < len(lon_factor) * 0.2:
                characterization['longitude'] = 'localized'
            else:
                characterization['longitude'] = 'widespread'
        else:
            characterization['longitude'] = 'uniform'
        
        # Overall pattern
        if characterization['latitude'] == 'localized' and characterization['longitude'] == 'localized':
            characterization['overall'] = 'hotspot'
        elif characterization['latitude'] == 'localized':
            characterization['overall'] = 'latitudinal band'
        elif characterization['longitude'] == 'localized':
            characterization['overall'] = 'longitudinal band'
        else:
            characterization['overall'] = 'broad scale'
        
        return characterization
    
    def _plot_feature_profiles(self, factors: CPFactors,
                             feature_names: List[str]) -> Path:
        """Plot feature loading profiles for components.
        
        Args:
            factors: CP factors
            feature_names: Feature names
            
        Returns:
            Path to saved plot
        """
        feature_factor = factors.factors[-1]
        n_comps = min(6, factors.rank)  # Show top 6 components
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx in range(n_comps):
            ax = axes[idx]
            
            # Get loadings
            loadings = feature_factor[:, idx]
            
            # Sort by absolute value
            sorted_idx = np.argsort(np.abs(loadings))[::-1][:self.n_top_features]
            
            # Create bar plot
            y_pos = np.arange(len(sorted_idx))
            values = loadings[sorted_idx]
            labels = [feature_names[i] for i in sorted_idx]
            
            # Color by sign
            colors = ['darkred' if v < 0 else 'darkblue' for v in values]
            
            bars = ax.barh(y_pos, values)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Loading')
            ax.set_title(f'Component {idx + 1}')
            ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            
        # Remove extra subplots
        for idx in range(n_comps, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'feature_profiles.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature profiles to: {plot_path}")
        return plot_path
    
    def _plot_component_similarities(self, factors: CPFactors) -> Path:
        """Plot similarity matrix between components.
        
        Args:
            factors: CP factors
            
        Returns:
            Path to saved plot
        """
        # Calculate cosine similarity between components
        n_modes = factors.n_modes
        rank = factors.rank
        
        # Average similarity across modes
        similarity_matrix = np.zeros((rank, rank))
        
        for mode in range(n_modes):
            factor = factors.factors[mode]
            # Normalize
            factor_norm = factor / (np.linalg.norm(factor, axis=0) + 1e-10)
            # Compute similarity
            mode_sim = np.abs(factor_norm.T @ factor_norm)
            similarity_matrix += mode_sim / n_modes
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use hierarchical clustering to reorder
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Cluster
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        order = dendro['leaves']
        
        # Reorder similarity matrix
        similarity_ordered = similarity_matrix[order][:, order]
        
        # Plot heatmap
        im = ax.imshow(similarity_ordered, cmap='RdBu_r', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Similarity')
        
        # Labels
        ax.set_xticks(range(rank))
        ax.set_yticks(range(rank))
        ax.set_xticklabels([f'C{i+1}' for i in order], rotation=45)
        ax.set_yticklabels([f'C{i+1}' for i in order])
        ax.set_title('Component Similarity Matrix (Reordered by Clustering)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'component_similarities.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved component similarities to: {plot_path}")
        return plot_path
    
    def _generate_word_clouds(self, factors: CPFactors,
                            feature_names: List[str],
                            component_weights: Optional[np.ndarray]) -> Path:
        """Generate word clouds for top components.
        
        Args:
            factors: CP factors
            feature_names: Feature names
            component_weights: Component weights
            
        Returns:
            Path to saved plot
        """
        feature_factor = factors.factors[-1]
        
        # Select top 4 components
        n_clouds = min(4, factors.rank)
        if component_weights is not None:
            top_comps = np.argsort(component_weights)[::-1][:n_clouds]
        else:
            top_comps = range(n_clouds)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, comp in enumerate(top_comps):
            ax = axes[idx]
            
            # Get loadings
            loadings = np.abs(feature_factor[:, comp])
            
            # Create word frequency dict
            word_freq = {}
            for i, loading in enumerate(loadings):
                if loading > self.significance_threshold:
                    # Clean feature name
                    clean_name = feature_names[i].replace('_', ' ').title()
                    word_freq[clean_name] = loading
            
            if len(word_freq) > 0 and HAS_WORDCLOUD:
                # Generate word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='viridis',
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate_from_frequencies(word_freq)
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Component {comp + 1}')
            elif len(word_freq) > 0:
                # Fallback: show text list when wordcloud not available
                sorted_features = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                text = '\n'.join([f'{name}: {score:.3f}' for name, score in sorted_features[:10]])
                ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top', fontsize=8)
                ax.set_title(f'Component {comp + 1} (Top Features)')
            else:
                ax.text(0.5, 0.5, 'No significant features',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Component {comp + 1}')
            
            ax.axis('off')
        
        # Remove extra subplots
        for idx in range(n_clouds, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'component_wordclouds.{self.plot_format}'
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved component word clouds to: {plot_path}")
        return plot_path
    
    def _calculate_component_statistics(self, factors: CPFactors,
                                      component_weights: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate statistics about components.
        
        Args:
            factors: CP factors
            component_weights: Component weights
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Component weight statistics
        if component_weights is not None:
            stats['weight_distribution'] = {
                'mean': float(np.mean(component_weights)),
                'std': float(np.std(component_weights)),
                'min': float(np.min(component_weights)),
                'max': float(np.max(component_weights)),
                'gini': float(self._gini_coefficient(component_weights))
            }
            
            # Cumulative variance explained
            sorted_weights = np.sort(component_weights)[::-1]
            cumsum = np.cumsum(sorted_weights)
            total = np.sum(sorted_weights)
            
            stats['variance_explained'] = {
                'top_1': float(cumsum[0] / total),
                'top_5': float(cumsum[min(4, len(cumsum)-1)] / total),
                'top_10': float(cumsum[min(9, len(cumsum)-1)] / total),
                'n_components_90_percent': int(np.argmax(cumsum >= 0.9 * total) + 1)
            }
        
        # Factor sparsity
        stats['factor_sparsity'] = {}
        for mode, (name, factor) in enumerate(zip(factors.mode_names, factors.factors)):
            sparsity = np.mean(np.abs(factor) < self.significance_threshold)
            stats['factor_sparsity'][name] = float(sparsity)
        
        # Component orthogonality
        stats['orthogonality'] = {}
        for mode, (name, factor) in enumerate(zip(factors.mode_names, factors.factors)):
            # Compute average absolute correlation between components
            factor_norm = factor / (np.linalg.norm(factor, axis=0) + 1e-10)
            corr_matrix = np.abs(factor_norm.T @ factor_norm)
            np.fill_diagonal(corr_matrix, 0)
            avg_corr = np.mean(corr_matrix)
            stats['orthogonality'][name] = float(1 - avg_corr)  # Higher is more orthogonal
        
        return stats
    
    def _gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement.
        
        Args:
            values: Array of values
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    def _save_interpretations(self, interpretations: Dict[str, Any]) -> None:
        """Save interpretations to JSON file.
        
        Args:
            interpretations: Component interpretations
        """
        output_path = self.output_dir / 'component_interpretations.json'
        
        with open(output_path, 'w') as f:
            json.dump(interpretations, f, indent=2)
        
        logger.info(f"Saved component interpretations to: {output_path}")