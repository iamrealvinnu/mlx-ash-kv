import numpy as np
from typing import Any

class UniversalTensorCritic:
    """
    Academic-Grade Neural Uncertainty Detector.
    Implements Varentropy-Proxy monitoring by analyzing the variance of 
    Key tensors within the Attention Manifold.
    """
    def __init__(self, healthy_baseline: float = 0.1):
        self.healthy_baseline = healthy_baseline

    def calculate_varentropy_proxy(self, cache: Any) -> float:
        """
        Calculates a normalized uncertainty index (0.0 - 1.0) based on 
        the variance of the final layer's KV-cache.
        """
        try:
            with cache._lock:
                last_layer = cache.layer_keys[-1]
                if last_layer is None:
                    return 0.0
                
                backend_name = cache.healer.__class__.__name__
                
                if "MLX" in backend_name:
                    import mlx.core as mx
                    var = mx.var(last_layer).item()
                else:
                    import torch
                    if isinstance(last_layer, torch.Tensor):
                        var = torch.var(last_layer).item()
                    else:
                        var = np.var(last_layer)

            # Varentropy-Proxy Logic:
            # We map the deviation from the manifold baseline to an uncertainty index.
            uncertainty_index = min(1.0, abs(self.healthy_baseline - var) * 10)
            
            # Filter low-level noise
            if uncertainty_index < 0.1:
                uncertainty_index = 0.0
                
            return uncertainty_index

        except Exception:
            return 0.0
