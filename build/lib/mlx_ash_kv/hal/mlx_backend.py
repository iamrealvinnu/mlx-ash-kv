import mlx.core as mx
import numpy as np
from typing import Any, List, Optional
from .base_backend import NeuralHealer

class MLXHealer(NeuralHealer):
    """
    Apple Silicon Backend (Metal-Optimized).
    """

    @staticmethod
    @mx.compile
    def _generate_head_specific_mask_compiled(seq_len: int, strike_indices: mx.array, strike_sigmas: mx.array, head_bitmasks: mx.array, num_heads: int) -> mx.array:
        """
        Fused Metal Kernel for 4D Head-Specific Gaussian Masking.
        """
        t = mx.arange(seq_len, dtype=mx.float16)
        mu = strike_indices[:, None]
        sigma = strike_sigmas[:, None]
        
        dist_sq = mx.square(t[None, :] - mu)
        penalty = -10000.0 * mx.exp(-dist_sq / (2 * mx.square(sigma) + 1e-6))
        
        valid = mx.logical_and(t[None, :] >= mu, t[None, :] > 0)
        penalty = mx.where(valid, penalty, 0.0)
        
        strike_masks = penalty[:, None, :] * head_bitmasks[:, :, None]
        mask = mx.min(strike_masks, axis=0)
        return mask.reshape(1, -1, 1, seq_len)

    def generate_mask(self, seq_len: int, strikes: List[dict], num_heads: int) -> mx.array:
        if not strikes:
            return mx.zeros((1, num_heads, 1, seq_len), dtype=mx.float16)
            
        indices = mx.array([s["index"] for s in strikes], dtype=mx.float16)
        sigmas = mx.array([s["sigma"] for s in strikes], dtype=mx.float16)
        
        h_masks = []
        for s in strikes:
            h_row = np.zeros(num_heads)
            h_row[s["heads"]] = 1.0
            h_masks.append(h_row)
        head_bitmasks = mx.array(np.array(h_masks), dtype=mx.float16)
        
        return self._generate_head_specific_mask_compiled(seq_len, indices, sigmas, head_bitmasks, num_heads)

    def eval_arrays(self, *arrays: Any) -> None:
        mx.eval(*arrays)

    def concat_arrays(self, arrays: List[mx.array], axis: int) -> mx.array:
        return mx.concatenate(arrays, axis=axis)

    def take_arrays(self, array: mx.array, indices: mx.array, axis: int) -> mx.array:
        return mx.take(array, indices, axis=axis)

    def page_to_disk(self, array: mx.array, path: str) -> None:
        """Saves MLX array to disk (using safetensors format via mx.save if possible)."""
        mx.save(path, array)

    def page_from_disk(self, path: str, shape: tuple, dtype: Any) -> mx.array:
        """Loads MLX array from disk."""
        return mx.load(path)
