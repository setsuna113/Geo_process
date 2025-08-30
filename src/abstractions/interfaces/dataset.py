"""Dataset interface - defines what every dataset must implement."""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, List, Optional, Union
from pathlib import Path
from ..types import DatasetInfo, DataType


class IDataset(ABC):
    """Interface for all dataset implementations."""
    
    @abstractmethod
    def load_info(self) -> DatasetInfo:
        """Load dataset metadata without reading all data."""
        pass
    
    @abstractmethod
    def read_records(self) -> Iterator[Dict[str, Any]]:
        """Read records from dataset."""
        pass
    
    @abstractmethod
    def read_chunks(self) -> Iterator[List[Dict[str, Any]]]:
        """Read dataset in chunks for memory efficiency."""
        pass
    
    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Get the type of data this dataset contains."""
        pass
    
    @abstractmethod
    def estimate_size(self) -> float:
        """Estimate dataset size in MB."""
        pass