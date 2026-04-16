from abc import ABC, abstractmethod
from typing import Any, List, Optional

class NeuralHealer(ABC):
    """
    Abstract base class for Hardware-Specific Self-Healing Kernels.
    """
    
    @abstractmethod
    def generate_mask(self, seq_len: int, strikes: List[dict], num_heads: int) -> Any:
        """
        Generates the 4D Gaussian Mutation Mask for the specific hardware.
        """
        pass

    @abstractmethod
    def eval_arrays(self, *arrays: Any) -> None:
        """
        Synchronizes/evaluates arrays on the specific hardware.
        """
        pass

    @abstractmethod
    def concat_arrays(self, arrays: List[Any], axis: int) -> Any:
        """
        Hardware-agnostic concatenation.
        """
        pass

    @abstractmethod
    def take_arrays(self, array: Any, indices: Any, axis: int) -> Any:
        """
        Hardware-agnostic indexing.
        """
        pass

    @abstractmethod
    def page_to_disk(self, array: Any, path: str) -> None:
        """
        Offloads a tensor to NVMe storage.
        """
        pass

    @abstractmethod
    def page_from_disk(self, path: str, shape: tuple, dtype: Any) -> Any:
        """
        Retrieves a tensor from NVMe storage.
        """
        pass
