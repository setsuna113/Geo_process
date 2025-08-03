"""Google Earth Engine Integration Module

Unified module for biodiversity and climate data integration using GEE.
Replaces the old climate_gee module with a more comprehensive workflow.
"""

from .config import GEEConfig

# Lazy import to avoid requiring ee module at import time
def get_workflow_orchestrator():
    """Get workflow orchestrator (lazy import)"""
    from .workflow.orchestrator import GEEWorkflowOrchestrator
    return GEEWorkflowOrchestrator

__all__ = ['get_workflow_orchestrator', 'GEEConfig']

__version__ = '1.0.0'