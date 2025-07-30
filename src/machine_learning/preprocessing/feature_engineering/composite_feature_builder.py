"""Composite feature builder that combines multiple feature builders."""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging

from ....base.feature_builder import BaseCompositeFeatureBuilder
from ....core.registry import feature_builder, component_registry

logger = logging.getLogger(__name__)


@feature_builder(
    "composite",
    required_columns=set(),  # Will be determined by component builders
    description="Combines multiple feature builders into a pipeline"
)
class CompositeFeatureBuilder(BaseCompositeFeatureBuilder):
    """
    Composite feature builder that combines multiple feature builders.
    
    Can automatically discover and use registered feature builders or
    be configured with specific builders.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize composite feature builder."""
        super().__init__(config)
        
        # Get configuration
        self.auto_discover = config.get('auto_discover', True) if config else True
        self.feature_categories = config.get('feature_categories', ['richness', 'spatial', 'ecological']) if config else ['richness', 'spatial', 'ecological']
        
        # Auto-discover and add feature builders if enabled
        if self.auto_discover:
            self._auto_discover_builders()
    
    def _auto_discover_builders(self) -> None:
        """Automatically discover and add registered feature builders."""
        logger.info("Auto-discovering registered feature builders")
        
        for category in self.feature_categories:
            # Find builders for this category
            builders = component_registry.find_feature_builders_by_category(category)
            
            for builder_metadata in builders:
                try:
                    # Get builder class
                    builder_class = builder_metadata.component_class
                    
                    # Skip composite builders to avoid recursion
                    if builder_class.__name__ == 'CompositeFeatureBuilder':
                        continue
                    
                    # Create instance
                    builder_instance = builder_class(self.config)
                    
                    # Add with category prefix to avoid conflicts
                    self.add_builder(builder_instance, prefix=category)
                    
                    logger.info(f"Added {builder_class.__name__} for category '{category}'")
                    
                except Exception as e:
                    logger.warning(f"Failed to add builder {builder_metadata.name}: {e}")
    
    def configure_builders(self, builder_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure specific builders with custom settings.
        
        Args:
            builder_config: Dictionary mapping builder names to their configurations
        """
        for builder_name, config in builder_config.items():
            if builder_name in self.builders:
                # Update builder configuration
                if hasattr(self.builders[builder_name], 'config'):
                    self.builders[builder_name].config.update(config)
                logger.info(f"Updated configuration for {builder_name}")
            else:
                logger.warning(f"Builder {builder_name} not found in composite")
    
    def add_builder_by_name(self, builder_name: str, prefix: Optional[str] = None) -> None:
        """
        Add a feature builder by its registered name.
        
        Args:
            builder_name: Name of the registered builder
            prefix: Optional prefix for feature names
        """
        try:
            # Get builder from registry
            builder_class = component_registry.feature_builders.get(builder_name)
            
            # Create instance
            builder_instance = builder_class(self.config)
            
            # Add to composite
            self.add_builder(builder_instance, prefix=prefix)
            
            logger.info(f"Added {builder_name} to composite")
            
        except Exception as e:
            logger.error(f"Failed to add builder {builder_name}: {e}")
            raise
    
    def remove_builder_by_category(self, category: str) -> None:
        """
        Remove all builders of a specific category.
        
        Args:
            category: Category of builders to remove
        """
        builders_to_remove = []
        
        for builder_name, builder in self.builders.items():
            # Check if builder belongs to category
            if hasattr(builder, 'feature_config'):
                builder_categories = getattr(builder, 'feature_categories', [])
                if category in builder_categories:
                    builders_to_remove.append(builder_name)
            # Also check by prefix
            elif self.builder_prefixes.get(builder_name) == category:
                builders_to_remove.append(builder_name)
        
        for builder_name in builders_to_remove:
            self.remove_builder(builder_name)
            logger.info(f"Removed {builder_name} from composite")
    
    def get_active_categories(self) -> List[str]:
        """Get list of active feature categories."""
        categories = set()
        
        for builder_name in self.builders:
            prefix = self.builder_prefixes.get(builder_name)
            if prefix:
                categories.add(prefix)
        
        return list(categories)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of all features that will be created.
        
        Returns:
            Dictionary with feature counts and details by category
        """
        summary = {
            'total_features': len(self.get_feature_names()),
            'categories': {},
            'builders': {}
        }
        
        # Count features by category
        for category in self.get_active_categories():
            category_features = [
                f for f in self.get_feature_names() 
                if f.startswith(f"{category}_")
            ]
            summary['categories'][category] = {
                'count': len(category_features),
                'features': category_features
            }
        
        # Summary by builder
        for builder_name, builder in self.builders.items():
            summary['builders'][builder_name] = {
                'class': builder.__class__.__name__,
                'prefix': self.builder_prefixes.get(builder_name),
                'feature_count': len(builder.get_feature_names()),
                'required_columns': list(builder.get_required_columns())
            }
        
        return summary
    
    def validate_data_compatibility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check which builders can work with the provided data.
        
        Args:
            data: Input DataFrame to check
            
        Returns:
            Dictionary with compatibility information
        """
        compatibility = {
            'compatible_builders': [],
            'incompatible_builders': [],
            'missing_columns': {}
        }
        
        for builder_name, builder in self.builders.items():
            required = builder.get_required_columns()
            missing = required - set(data.columns)
            
            if not missing:
                compatibility['compatible_builders'].append(builder_name)
            else:
                compatibility['incompatible_builders'].append(builder_name)
                compatibility['missing_columns'][builder_name] = list(missing)
        
        return compatibility
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'CompositeFeatureBuilder':
        """
        Fit all compatible builders.
        
        Only fits builders that have their required columns in the data.
        """
        # Check compatibility
        compatibility = self.validate_data_compatibility(data)
        
        if compatibility['incompatible_builders']:
            logger.warning(
                f"Some builders are incompatible with the data: "
                f"{compatibility['incompatible_builders']}"
            )
        
        # Fit only compatible builders
        for builder_name in compatibility['compatible_builders']:
            builder = self.builders[builder_name]
            self._update_progress(
                self.builders_fitted, 
                len(compatibility['compatible_builders']), 
                f"Fitting {builder_name}"
            )
            
            try:
                builder.fit(data, **kwargs)
                self.builders_fitted += 1
            except Exception as e:
                logger.error(f"Failed to fit {builder_name}: {e}")
                # Continue with other builders
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data using all compatible builders.
        
        Only uses builders that have their required columns in the data.
        """
        # Check compatibility
        compatibility = self.validate_data_compatibility(data)
        
        all_features = []
        
        for i, builder_name in enumerate(compatibility['compatible_builders']):
            builder = self.builders[builder_name]
            prefix = self.builder_prefixes.get(builder_name)
            
            self._update_progress(
                i, 
                len(compatibility['compatible_builders']), 
                f"Transforming with {builder_name}"
            )
            
            try:
                # Transform with this builder
                features = builder.transform(data, **kwargs)
                
                # Apply prefix if specified
                if prefix:
                    features = features.add_prefix(f"{prefix}_")
                
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Failed to transform with {builder_name}: {e}")
                # Continue with other builders
        
        # Combine all features
        if all_features:
            combined = pd.concat(all_features, axis=1)
            
            # Handle duplicate columns
            if combined.columns.duplicated().any():
                logger.warning("Duplicate feature names found, making unique")
                combined = combined.loc[:, ~combined.columns.duplicated()]
            
            return combined
        else:
            logger.warning("No features could be created")
            return pd.DataFrame(index=data.index)
    
    # Private attribute for tracking fitting progress
    builders_fitted = 0