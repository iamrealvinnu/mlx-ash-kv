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
    Asynchronous Self-Healing Cache (ASH-KV) v8.0.0 (Native Override Edition).
    
    Implements a Multi-Layer Memory Hypervisor for live LLM integration.
    Manages head-specific pruning and EDCC across all model layers.
    """
    def __init__(self, critic_model_path: str = None, num_layers: int = 32, num_heads: int = 32):
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Multi-Layer Manifold
        self.layer_keys: List[Optional[mx.array]] = [None] * num_layers
        self.layer_values: List[Optional[mx.array]] = [None] * num_layers
        
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
        return self.layer_keys[0].shape[2] if self.layer_keys[0] is not None else 0

    def _generate_head_specific_mask(self, seq_len: int) -> mx.array:
        """Generates a 4D mask applying Gaussian penalties strictly to targeted attention heads."""
        mask = np.zeros((1, self.num_heads, 1, seq_len), dtype=np.float16)
        if not self.strikes:
            return mx.array(mask)

        t = np.arange(seq_len)
        for strike in self.strikes:
            mu = strike["index"]
            sigma = strike["sigma"]
            target_heads = strike["heads"]

            dist_sq = (t - mu)**2
            penalty = -10000.0 * np.exp(-dist_sq / (2 * sigma**2 + 1e-6))
            valid = (t >= mu) & (t > 0)
            penalty = np.where(valid, penalty, 0.0)

            for h in target_heads:
                mask[0, h, 0, :] = np.minimum(mask[0, h, 0, :], penalty)

        return mx.array(mask)

    def update_layer(self, layer_idx: int, new_k: mx.array, new_v: mx.array) -> Tuple[mx.array, mx.array]:
        """Layer-specific manifold ingestion."""
        with self._lock:
            if self.layer_keys[layer_idx] is None:
                self.layer_keys[layer_idx] = new_k
                self.layer_values[layer_idx] = new_v
            else:
                self.layer_keys[layer_idx] = mx.concatenate([self.layer_keys[layer_idx], new_k], axis=2)
                self.layer_values[layer_idx] = mx.concatenate([self.layer_values[layer_idx], new_v], axis=2)
            
            return self.layer_keys[layer_idx], self.layer_values[layer_idx]

    def get_mask(self) -> mx.array:
        """Retrieves or regenerates the active immune mask."""
        with self._lock:
            seq_len = self.seq_len
            if self.active_mask is None or self.active_mask.shape[3] != seq_len:
                self.active_mask = self._generate_head_specific_mask(seq_len)
            return self.active_mask

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
            if start_idx + chunk_size > seq_len:
                return None
            
            # Analyze primary layer logic heads
            chunk = self.layer_keys[0][0, 0, start_idx:start_idx+chunk_size, :].astype(mx.float32)
            mx.eval(chunk)
            np_chunk = np.array(chunk).reshape(1, chunk_size, -1)
            if np_chunk.shape[2] != 128:
                np_chunk = np.pad(np_chunk, ((0,0), (0,0), (0, 128 - np_chunk.shape[2])))[:, :, :128]

        prediction = self.critic_model.predict({"hidden_states": np_chunk})
        output_key = list(prediction.keys())[0]
        return float(prediction[output_key][0])

    def compact_manifold(self, threshold: float = -9000.0) -> int:
        """Entropy-Driven Context Compaction (EDCC) across all layers."""
        if not self.strikes:
            return 0
            
        with self._lock:
            mask = self.get_mask()
            reasoning_heads_mask = mask[0, :self.num_heads // 2, 0, :]
            max_penalty_across_heads = mx.max(reasoning_heads_mask, axis=0)
            mx.eval(max_penalty_across_heads)
            
            keep_indices = mx.nonzero(max_penalty_across_heads > threshold)[0]
            mx.eval(keep_indices)
            
            new_seq_len = keep_indices.size
            tokens_freed = self.seq_len - new_seq_len
            
            if tokens_freed <= 0:
                self.strikes.clear()
                return 0
                
            # Compact all layers atomically
            for l in range(self.num_layers):
                self.layer_keys[l] = mx.take(self.layer_keys[l], keep_indices, axis=2)
                self.layer_values[l] = mx.take(self.layer_values[l], keep_indices, axis=2)
            
            self.strikes.clear()
            self.active_mask = None
            mx.eval(*self.layer_keys, *self.layer_values)
            return tokens_freed
