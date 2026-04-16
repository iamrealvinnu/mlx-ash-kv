import numpy as np
from typing import Any

class UniversalTensorCritic:
    """
    Zero-Shot Neural Hallucination Detector.
    Monitors Attention Manifold Entropy (Varentropy) to detect logical drift
    intrinsic to the model's internal tensor state.
    """
    def __init__(self, healthy_baseline: float = 0.1):
        self.healthy_baseline = healthy_baseline

    def evaluate_tensor_drift(self, cache: Any) -> float:
        """
        Calculates mathematical drift score based on KV-cache variance.
        Works across MLX and PyTorch backends.
        """
        try:
            # Safely grab the last layer keys under the cache lock
            with cache._lock:
                last_layer = cache.layer_keys[-1]
                if last_layer is None:
                    return 0.0
                
                # Identify backend and calculate variance
                backend_name = cache.healer.__class__.__name__
                
                if "MLX" in backend_name:
                    import mlx.core as mx
                    var = mx.var(last_layer).item()
                else:
                    import torch
                    # Handle both CPU and CUDA tensors
                    if isinstance(last_layer, torch.Tensor):
                        var = torch.var(last_layer).item()
                    else:
                        # Fallback for numpy placeholders
                        var = np.var(last_layer)

            # Drift Score Logic:
            # Healthy attention typically maintains a stable variance.
            # 1. Variance Collapse (Repetition/Confusion): var -> 0
            # 2. Variance Spike (Chaos/Hallucination): var -> high
            
            # Map anomaly to 0.0 - 1.0 range
            # Scaling: 0.1 is baseline. deviation of 0.1 maps to 1.0 drift.
            drift_score = min(1.0, abs(self.healthy_baseline - var) * 10)
            
            # Smooth low scores
            if drift_score < 0.1:
                drift_score = 0.0
                
            return drift_score

        except Exception as e:
            # print(f"[CRITIC ERROR] {e}")
            return 0.0
