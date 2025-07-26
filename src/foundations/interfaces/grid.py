from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GridCell:
    """Pure data class - no behavior."""
    id: str
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    level: int
    parent_id: Optional[str] = None

class IGrid(ABC):
    """Pure grid interface."""
    
    @abstractmethod
    def create_cells(self) -> List[GridCell]:
        """Create grid cells."""
        pass
    
    @abstractmethod
    def get_resolution(self) -> float:
        """Get grid resolution in meters."""
        pass