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
    Asynchronous Self-Healing Cache (ASH-KV) v6.0.0 (Multi-Head Pruning Edition).
    
    Implements Head-Specific Causal Pruning and Entropy-Driven Context Compaction (EDCC)
    with hardware-isolated verification via the Asynchronous Verification Daemon (AVD).
    """
    def __init__(self, critic_model_path: str = None, num_heads: int = 32):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.num_heads = num_heads
        self.strikes: List[Dict[str, any]] = []
        self.active_mask: Optional[mx.array] = None
        self._lock = threading.Lock()
        
        self.critic_model = None
        if critic_model_path and os.path.exists(critic_model_path):
            print(f"[SYSTEM] Loading ANE Verification Daemon from {critic_model_path}...")
            self.critic_model = ct.models.MLModel(
                critic_model_path, 
                compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            print("[SYSTEM] Asynchronous Verification Daemon (AVD) armed.")

    @property
    def seq_len(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    def _generate_head_specific_mask(self, seq_len: int) -> mx.array:
        """Generates a 4D mask applying Gaussian penalties strictly to targeted attention heads."""
        # Base mask: (1, num_heads, 1, seq_len)
        mask = np.zeros((1, self.num_heads, 1, seq_len), dtype=np.float16)
        if not self.strikes:
            return mx.array(mask)

        t = np.arange(seq_len)
        for strike in self.strikes:
            mu = strike["index"]
            sigma = strike["sigma"]
            target_heads = strike["heads"]

            # Causal Gaussian Decay
            dist_sq = (t - mu)**2
            penalty = -10000.0 * np.exp(-dist_sq / (2 * sigma**2 + 1e-6))
            
            # Apply constraints (Causal & Sink Preservation)
            valid = (t >= mu) & (t > 0)
            penalty = np.where(valid, penalty, 0.0)

            # Apply ONLY to targeted heads
            for h in target_heads:
                mask[0, h, 0, :] = np.minimum(mask[0, h, 0, :], penalty)

        return mx.array(mask)

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
                self.active_mask = self._generate_head_specific_mask(seq_len)

            return self.keys, self.values, self.active_mask

    def sync_eval(self, *arrays):
        """Thread-safe evaluation to prevent Metal command buffer collisions."""
        with self._lock:
            mx.eval(*arrays)

    def flag_logical_drift(self, index: int, severity_score: float = 0.5, target_heads: Optional[List[int]] = None) -> None:
        """Asynchronous API for the AVD to flag logical drift on specific heads."""
        with self._lock:
            sigma = 1.0 + (severity_score * 19.0)
            # Default to penalizing the first 16 heads (Logical/Reasoning heads) if none specified
            heads = target_heads if target_heads is not None else list(range(self.num_heads // 2))
            
            if not any(s["index"] == index for s in self.strikes):
                self.strikes.append({"index": float(index), "sigma": sigma, "heads": heads})
                self.active_mask = None # Trigger mask recreation

    def analyze_manifold_chunk(self, start_idx: int, chunk_size: int = 128) -> Optional[float]:
        if not self.critic_model or self.keys is None:
            return None
            
        seq_len = self.keys.shape[2]
        if start_idx + chunk_size > seq_len:
            return None
            
        with self._lock:
            chunk = self.keys[0, 0, start_idx:start_idx+chunk_size, :].astype(mx.float32)
            mx.eval(chunk)
            np_chunk = np.array(chunk).reshape(1, chunk_size, -1)
            
            if np_chunk.shape[2] != 128:
                np_chunk = np.pad(np_chunk, ((0,0), (0,0), (0, 128 - np_chunk.shape[2])))[:, :, :128]

        prediction = self.critic_model.predict({"hidden_states": np_chunk})
        output_key = list(prediction.keys())[0]
        return float(prediction[output_key][0])

    def compact_manifold(self, threshold: float = -9000.0) -> int:
        """Entropy-Driven Context Compaction (EDCC). Physically deallocates dead nodes."""
        if not self.strikes:
            return 0
            
        with self._lock:
            if self.keys is None or self.active_mask is None:
                return 0
                
            seq_len = self.keys.shape[2]
            # We compact tokens that are severely penalized across ALL reasoning heads
            reasoning_heads_mask = self.active_mask[0, :self.num_heads // 2, 0, :]
            max_penalty_across_heads = mx.max(reasoning_heads_mask, axis=0)
            
            mx.eval(max_penalty_across_heads)
            keep_condition = max_penalty_across_heads > threshold
            keep_indices = mx.nonzero(keep_condition)[0]
            mx.eval(keep_indices)
            
            new_seq_len = keep_indices.size
            tokens_freed = seq_len - new_seq_len
            
            if tokens_freed <= 0:
                self.strikes.clear()
                return 0
                
            self.keys = mx.take(self.keys, keep_indices, axis=2)
            self.values = mx.take(self.values, keep_indices, axis=2)
            
            self.strikes.clear()
            self.active_mask = None
            mx.eval(self.keys, self.values)
            return tokens_freed
