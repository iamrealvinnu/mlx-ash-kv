"""
@file cache.py
@brief Core implementation of the Asynchronous Self-Healing Cache (ASH-KV).
"""

import mlx.core as mx
import threading
import math
import numpy as np
import coremltools as ct
import os
from typing import Tuple, List, Optional, Dict

class ASHCache:
    """
    Asynchronous Self-Healing Cache (ASH-KV) v8.0.2 (Silicon-Native).
    
    Implements vectorized Head-Specific Causal Pruning and EDCC.
    """
    def __init__(self, critic_model_path: str = None, num_layers: int = 32, num_heads: int = 32):
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.layer_keys: List[Optional[mx.array]] = [None] * num_layers
        self.layer_values: List[Optional[mx.array]] = [None] * num_layers
        
        self.strikes: List[Dict[str, any]] = []
        self.active_mask: Optional[mx.array] = None
        self._lock = threading.Lock()
        
        self.critic_model = None
        if critic_model_path and os.path.exists(critic_model_path):
            self.critic_model = ct.models.MLModel(
                critic_model_path, 
                compute_units=ct.ComputeUnit.CPU_AND_NE
            )

    @property
    def seq_len(self) -> int:
        return self.layer_keys[0].shape[2] if self.layer_keys[0] is not None else 0

    @staticmethod
    @mx.compile
    def _generate_head_specific_mask_compiled(seq_len: int, strike_indices: mx.array, strike_sigmas: mx.array, head_bitmasks: mx.array) -> mx.array:
        """
        Fused Metal Kernel for 4D Head-Specific Gaussian Masking.
        math: mask = min_{strikes}( -10000 * exp(-dist^2 / 2sigma^2) * head_filter )
        """
        # (S)
        t = mx.arange(seq_len, dtype=mx.float16)
        
        # (N, 1)
        mu = strike_indices[:, None]
        sigma = strike_sigmas[:, None]
        
        # (N, S)
        dist_sq = mx.square(t[None, :] - mu)
        penalty = -10000.0 * mx.exp(-dist_sq / (2 * mx.square(sigma) + 1e-6))
        
        valid = mx.logical_and(t[None, :] >= mu, t[None, :] > 0)
        penalty = mx.where(valid, penalty, 0.0)
        
        # head_bitmasks: (N, H) -> (N, H, 1)
        # Broadcast strike penalties across targeted heads
        strike_masks = penalty[:, None, :] * head_bitmasks[:, :, None] # (N, H, S)
        
        # Aggregate: (H, S) -> (1, H, 1, S)
        mask = mx.min(strike_masks, axis=0)
        return mask.reshape(1, -1, 1, seq_len)

    def get_mask(self) -> mx.array:
        """Atomic retrieve/generate with memoization."""
        with self._lock:
            seq_len = self.seq_len
            if self.active_mask is None or self.active_mask.shape[3] != seq_len:
                if not self.strikes:
                    self.active_mask = mx.zeros((1, self.num_heads, 1, seq_len), dtype=mx.float16)
                else:
                    indices = mx.array([s["index"] for s in self.strikes], dtype=mx.float16)
                    sigmas = mx.array([s["sigma"] for s in self.strikes], dtype=mx.float16)
                    
                    # Build head filter matrix (N, H)
                    h_masks = []
                    for s in self.strikes:
                        h_row = np.zeros(self.num_heads)
                        h_row[s["heads"]] = 1.0
                        h_masks.append(h_row)
                    head_bitmasks = mx.array(np.array(h_masks), dtype=mx.float16)
                    
                    self.active_mask = self._generate_head_specific_mask_compiled(
                        seq_len, indices, sigmas, head_bitmasks
                    )
            return self.active_mask

    def update_layer(self, layer_idx: int, new_k: mx.array, new_v: mx.array) -> Tuple[mx.array, mx.array]:
        with self._lock:
            if self.layer_keys[layer_idx] is None:
                self.layer_keys[layer_idx] = new_k
                self.layer_values[layer_idx] = new_v
            else:
                self.layer_keys[layer_idx] = mx.concatenate([self.layer_keys[layer_idx], new_k], axis=2)
                self.layer_values[layer_idx] = mx.concatenate([self.layer_values[layer_idx], new_v], axis=2)
            return self.layer_keys[layer_idx], self.layer_values[layer_idx]

    def sync_eval(self, *arrays):
        with self._lock:
            mx.eval(*arrays)

    def flag_logical_drift(self, index: int, severity_score: float = 0.5, target_heads: Optional[List[int]] = None) -> None:
        with self._lock:
            sigma = 1.0 + (severity_score * 19.0)
            heads = target_heads if target_heads is not None else list(range(self.num_heads // 2))
            if not any(s["index"] == index for s in self.strikes):
                self.strikes.append({"index": float(index), "sigma": sigma, "heads": heads})
                self.active_mask = None 

    def analyze_manifold_chunk(self, start_idx: int, chunk_size: int = 128) -> Optional[float]:
        if not self.critic_model or self.layer_keys[0] is None:
            return None
        with self._lock:
            seq_len = self.seq_len
            if start_idx + chunk_size > seq_len: return None
            chunk = self.layer_keys[0][0, 0, start_idx:start_idx+chunk_size, :].astype(mx.float32)
            mx.eval(chunk)
            np_chunk = np.array(chunk).reshape(1, chunk_size, -1)
            if np_chunk.shape[2] != 128:
                np_chunk = np.pad(np_chunk, ((0,0), (0,0), (0, 128 - np_chunk.shape[2])))[:, :, :128]
        prediction = self.critic_model.predict({"hidden_states": np_chunk})
        return float(prediction[list(prediction.keys())[0]][0])

    def compact_manifold(self, threshold: float = -9000.0) -> int:
        if not self.strikes: return 0
        with self._lock:
            if self.layer_keys[0] is None: return 0
            mask = self.get_mask()
            # Truncate only if severely penalized in logic heads
            reasoning_heads_mask = mask[0, :self.num_heads // 2, 0, :]
            max_penalty = mx.max(reasoning_heads_mask, axis=0)
            mx.eval(max_penalty)
            keep_indices = mx.nonzero(max_penalty > threshold)[0]
            mx.eval(keep_indices)
            new_seq_len = keep_indices.size
            tokens_freed = self.seq_len - new_seq_len
            if tokens_freed <= 0:
                self.strikes.clear()
                return 0
            for l in range(self.num_layers):
                self.layer_keys[l] = mx.take(self.layer_keys[l], keep_indices, axis=2)
                self.layer_values[l] = mx.take(self.layer_values[l], keep_indices, axis=2)
            self.strikes.clear()
            self.active_mask = None
            mx.eval(*self.layer_keys, *self.layer_values)
            return tokens_freed
