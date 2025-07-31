# src/domain/resampling/strategies/__init__.py
"""Resampling strategies for different data types."""

from .sum_aggregation import SumAggregationStrategy
from .average_aggregation import AverageAggregationStrategy
from .area_weighted import AreaWeightedStrategy
from .block_sum_aggregation import BlockSumAggregationStrategy
from .majority_vote import MajorityVoteStrategy

__all__ = [
    'SumAggregationStrategy',
    'AverageAggregationStrategy', 
    'AreaWeightedStrategy',
    'BlockSumAggregationStrategy',
    'MajorityVoteStrategy'
]