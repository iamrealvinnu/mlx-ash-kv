import platform
import sys
from .base_backend import NeuralHealer

class SiliconFactory:
    """
    Hardware Factory for Detecting Silicon and Selecting the Right Healer.
    """

    @staticmethod
    def get_healer() -> NeuralHealer:
        os_name = platform.system()
        
        if os_name == "Darwin":
            try:
                from .mlx_backend import MLXHealer
                return MLXHealer()
            except ImportError:
                pass
        
        # Fallback to CUDA for Linux/Windows or if MLX is unavailable
        try:
            from .cuda_backend import CudaHealer
            return CudaHealer()
        except ImportError:
            raise RuntimeError("No compatible NeuralHealer backend found for this platform.")
