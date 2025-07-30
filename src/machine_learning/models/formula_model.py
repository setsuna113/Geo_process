"""Formula-based model specification for research-oriented ML."""

import re
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FormulaSpec:
    """Specification for a formula-based model."""
    target: str
    terms: List[Union[str, Tuple[str, str]]]  # Single terms or interaction pairs
    formula_string: str
    
    @property
    def feature_names(self) -> List[str]:
        """Get all unique feature names used in the formula."""
        features = set()
        for term in self.terms:
            if isinstance(term, str):
                features.add(term)
            else:  # Interaction term
                features.update(term)
        return sorted(features)
    
    @property
    def has_interactions(self) -> bool:
        """Check if formula contains interaction terms."""
        return any(isinstance(term, tuple) for term in self.terms)


class FormulaParser:
    """Parse R-style formula strings into usable specifications."""
    
    @staticmethod
    def parse(formula: str) -> FormulaSpec:
        """
        Parse formula string like 'Y ~ X1 + X2 + X1:X2' into components.
        
        Supports:
        - Main effects: X1, X2
        - Interactions: X1:X2 or X1*X2
        - Nested formulas: Y ~ X1 + X2 + (X1 + X2):Z
        """
        # Split target and predictors
        if '~' not in formula:
            raise ValueError(f"Invalid formula: {formula}. Must contain '~'")
        
        target, predictors = formula.split('~', 1)
        target = target.strip()
        
        # Parse predictor terms
        terms = []
        predictor_parts = FormulaParser._split_terms(predictors)
        
        for part in predictor_parts:
            part = part.strip()
            if ':' in part:
                # Interaction term
                left, right = part.split(':', 1)
                terms.append((left.strip(), right.strip()))
            elif '*' in part:
                # Full interaction (main effects + interaction)
                left, right = part.split('*', 1)
                left, right = left.strip(), right.strip()
                terms.extend([left, right, (left, right)])
            else:
                # Main effect
                terms.append(part)
        
        return FormulaSpec(
            target=target,
            terms=terms,
            formula_string=formula
        )
    
    @staticmethod
    def _split_terms(predictors: str) -> List[str]:
        """Split predictor string by + while respecting parentheses."""
        terms = []
        current_term = ""
        paren_depth = 0
        
        for char in predictors:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '+' and paren_depth == 0:
                if current_term.strip():
                    terms.append(current_term.strip())
                current_term = ""
                continue
            current_term += char
        
        if current_term.strip():
            terms.append(current_term.strip())
        
        return terms


class FormulaTransformer:
    """Transform data according to formula specification."""
    
    def __init__(self, formula_spec: FormulaSpec):
        self.formula_spec = formula_spec
        
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform dataframe according to formula.
        
        Returns:
            X: Feature matrix with interaction terms
            y: Target series
        """
        # Extract target
        if self.formula_spec.target not in df.columns:
            raise ValueError(f"Target '{self.formula_spec.target}' not found in data")
        
        y = df[self.formula_spec.target]
        
        # Build feature matrix
        X_parts = []
        feature_names = []
        
        for term in self.formula_spec.terms:
            if isinstance(term, str):
                # Main effect
                if term not in df.columns:
                    raise ValueError(f"Feature '{term}' not found in data")
                X_parts.append(df[[term]])
                feature_names.append(term)
            else:
                # Interaction term
                feat1, feat2 = term
                if feat1 not in df.columns or feat2 not in df.columns:
                    raise ValueError(f"Features for interaction '{feat1}:{feat2}' not found")
                
                interaction = df[feat1] * df[feat2]
                interaction_df = pd.DataFrame(
                    {f"{feat1}:{feat2}": interaction},
                    index=df.index
                )
                X_parts.append(interaction_df)
                feature_names.append(f"{feat1}:{feat2}")
        
        X = pd.concat(X_parts, axis=1)
        
        logger.info(f"Formula '{self.formula_spec.formula_string}' created {len(X.columns)} features")
        
        return X, y


class NestedFormulaModels:
    """Handle a sequence of nested formula-based models."""
    
    def __init__(self, formulas: List[str]):
        """
        Initialize with list of formula strings.
        
        Args:
            formulas: List of formula strings in nested order
                     e.g., ['Y ~ X1', 'Y ~ X1 + X2', 'Y ~ X1 + X2 + X1:X2']
        """
        self.formulas = formulas
        self.formula_specs = [FormulaParser.parse(f) for f in formulas]
        self.transformers = [FormulaTransformer(spec) for spec in self.formula_specs]
        self._validate_nesting()
        
    def _validate_nesting(self):
        """Validate that models are properly nested."""
        # Check all have same target
        targets = [spec.target for spec in self.formula_specs]
        if len(set(targets)) > 1:
            raise ValueError(f"All formulas must have same target. Found: {set(targets)}")
        
        # Check each model adds features
        prev_features = set()
        for i, spec in enumerate(self.formula_specs):
            curr_features = set(spec.feature_names)
            if i > 0 and not prev_features.issubset(curr_features):
                raise ValueError(
                    f"Model {i+1} is not properly nested. "
                    f"Missing features from previous model: {prev_features - curr_features}"
                )
            prev_features = curr_features
    
    def transform_all(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """Transform data for all nested models."""
        return [transformer.transform(df) for transformer in self.transformers]
    
    def get_model_comparison_info(self) -> List[Dict[str, Any]]:
        """Get information about model comparisons."""
        comparisons = []
        
        for i in range(1, len(self.formula_specs)):
            prev_spec = self.formula_specs[i-1]
            curr_spec = self.formula_specs[i]
            
            # Find new terms
            prev_terms_set = set(map(str, prev_spec.terms))
            curr_terms_set = set(map(str, curr_spec.terms))
            new_terms = curr_terms_set - prev_terms_set
            
            comparisons.append({
                'comparison': f"Model {i+1} vs Model {i}",
                'model_1': prev_spec.formula_string,
                'model_2': curr_spec.formula_string,
                'new_terms': list(new_terms),
                'df_difference': len(curr_spec.terms) - len(prev_spec.terms)
            })
        
        return comparisons