"""
@file cache.py
@brief Core implementation of the Asynchronous Self-Healing Cache (ASH-KV).
"""

import mlx.core as mx
import threading
import math
import numpy as np
import coremltools as ct
from typing import Tuple, List, Optional, Dict

class ASHCache:
    """
    Asynchronous Self-Healing Cache (ASH-KV) v4.1.1 (Metal-Safe Edition).
    
    Implements 'Temporal Rollback' with hardware-isolated ANE verification.
    Uses strict command-stream serialization to prevent Metal buffer collisions.
    """
    def __init__(self, critic_model_path: str = None):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.strikes: List[Dict[str, float]] = []
        self.active_mask: Optional[mx.array] = None
        self._lock = threading.Lock()
        
        self.critic_model = None
        if critic_model_path:
            self.critic_model = ct.models.MLModel(
                critic_model_path, 
                compute_units=ct.ComputeUnit.CPU_AND_NE
            )

    @property
    def seq_len(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    @staticmethod
    @mx.compile
    def _generate_immune_mask_compiled(seq_len: int, strike_indices: mx.array, strike_sigmas: mx.array) -> mx.array:
        mask = mx.zeros((1, 1, 1, seq_len), dtype=mx.float16)
        if strike_indices.size == 0:
            return mask

        t = mx.arange(seq_len, dtype=mx.float16)
        t_expanded = mx.broadcast_to(t[None, :], (strike_indices.size, seq_len))
        mu = strike_indices[:, None]
        sigma = strike_sigmas[:, None]

        dist_sq = mx.square(t_expanded - mu)
        penalty_val = -10000.0
        exponent = -dist_sq / (2 * mx.square(sigma) + 1e-6)
        
        strike_masks = mx.exp(exponent) * penalty_val
        causal_mask = (t_expanded >= mu)
        sink_mask = (t_expanded > 0)
        valid_strikes = mx.logical_and(causal_mask, sink_mask)
        
        strike_masks = mx.where(valid_strikes, strike_masks, 0.0)
        mask = mx.min(strike_masks, axis=0, keepdims=True)
        return mask.reshape(1, 1, 1, seq_len)

    def update(self, new_k: mx.array, new_v: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        with self._lock:
            if self.keys is None:
                self.keys = new_k
                self.values = new_v
            else:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

            seq_len = self.keys.shape[2]

            if self.active_mask is None or self.active_mask.shape[3] != seq_len:
                if not self.strikes:
                    self.active_mask = mx.zeros((1, 1, 1, seq_len), dtype=mx.float16)
                else:
                    indices = mx.array([s["index"] for s in self.strikes], dtype=mx.float16)
                    sigmas = mx.array([s["sigma"] for s in self.strikes], dtype=mx.float16)
                    self.active_mask = self._generate_immune_mask_compiled(seq_len, indices, sigmas)

            return self.keys, self.values, self.active_mask

    def sync_eval(self, *arrays):
        """Thread-safe evaluation to prevent Metal command buffer collisions."""
        with self._lock:
            mx.eval(*arrays)

    def flag_hallucination(self, index: int, severity_score: float = 0.5) -> None:
        with self._lock:
            sigma = 1.0 + (severity_score * 19.0)
            if not any(s["index"] == index for s in self.strikes):
                self.strikes.append({"index": float(index), "sigma": sigma})
                self.active_mask = None

    def analyze_manifold_chunk(self, start_idx: int, chunk_size: int = 128) -> Optional[float]:
        if not self.critic_model or self.keys is None:
            return None
            
        seq_len = self.keys.shape[2]
        if start_idx + chunk_size > seq_len:
            return None
            
        with self._lock:
            # Atomic extraction and evaluation to lock Metal encoder
            chunk = self.keys[0, 0, start_idx:start_idx+chunk_size, :].astype(mx.float32)
            mx.eval(chunk) # Force encoding and commit before releasing lock
            np_chunk = np.array(chunk).reshape(1, chunk_size, -1)
            
            if np_chunk.shape[2] != 128:
                np_chunk = np.pad(np_chunk, ((0,0), (0,0), (0, 128 - np_chunk.shape[2])))[:, :, :128]

        # Prediction happens outside lock (ANE is independent silicon)
        prediction = self.critic_model.predict({"hidden_states": np_chunk})
        output_key = list(prediction.keys())[0] 
        severity = float(prediction[output_key][0])
        return severity
