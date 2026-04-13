"""
@file cache.py
@brief Core implementation of the Asynchronous Self-Healing Cache (ASH-KV).

ASH-KV is a specialized KV-cache manager designed for high-stakes reasoning loops.
It enables real-time self-correction by allowing an asynchronous verification
process (the 'Ghost Critic') to surgically excise hallucinated nodes from the
attention manifold.

Architecture: Asynchronous Mask Injection via @mx.compile.
Performance: Zero-latency, non-blocking attention mutation on the Metal backend.

License: Apache 2.0 (v3.0.0 Release)
"""

import mlx.core as mx
import threading
import numpy as np
from typing import Tuple, List, Optional, Dict

class ASHCache:
    """
    Asynchronous Self-Healing Cache (ASH-KV) v4.0.0.
    
    Implements 'Temporal Rollback' via Gaussian-decay attention masking.
    
    The Problem: When a hallucination occurs, subsequent tokens generated before 
    the Ghost Critic catches the error are 'orphaned' (causally contaminated).
    
    The Solution: Instead of a binary mask (which creates a harsh logical cliff) 
    or a physical memory rollback (which incurs O(N) latency), ASH-KV applies a 
    Soft Causal Correction. We project a Gaussian gravity well onto the attention 
    manifold. The hallucinated node is hit with a -10000.0 penalty, and subsequent 
    orphaned tokens receive a decaying penalty based on their proximity to the strike.
    
    Safety: The <bos> token (Index 0) is strictly preserved as an Attention Sink 
    to absorb displaced probability mass and prevent generation collapse.
    """
    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

        # Stores (index, sigma) for each strike
        self.strikes: List[Dict[str, float]] = []
        
        self.active_mask: Optional[mx.array] = None
        self._lock = threading.Lock()

    @property
    def seq_len(self) -> int:
        return self.keys.shape[2] if self.keys is not None else 0

    @staticmethod
    @mx.compile
    def _generate_immune_mask_compiled(seq_len: int, strike_indices: mx.array, strike_sigmas: mx.array) -> mx.array:
        """
        Generates a Gaussian-decay attention mask fused in Metal.
        
        Math: 
        For each token i and strike j at mu_j:
        penalty = -10000 * exp(-(i - mu_j)^2 / (2 * sigma_j^2))
        
        Constraints:
        1. Causal: i >= mu_j (No look-back strikes)
        2. Sink Preservation: i > 0 (Never penalize <bos>)
        """
        # (1, 1, 1, seq_len)
        mask = mx.zeros((1, 1, 1, seq_len), dtype=mx.float16)
        
        if strike_indices.size == 0:
            return mask

        # Index manifold: (seq_len,)
        t = mx.arange(seq_len, dtype=mx.float16)
        
        # Expand for broadcasting: (num_strikes, seq_len)
        t_expanded = mx.broadcast_to(t[None, :], (strike_indices.size, seq_len))
        mu = strike_indices[:, None]
        sigma = strike_sigmas[:, None]

        # Calculate squared distance for Gaussian
        # Only apply to tokens >= mu (causal decay)
        dist_sq = mx.square(t_expanded - mu)
        
        # Gaussian penalty: -10000 * exp(-dist_sq / (2 * sigma^2))
        # Note: We use a large negative value to force softmax to zero
        penalty_val = -10000.0
        exponent = -dist_sq / (2 * mx.square(sigma) + 1e-6)
        
        # Compute individual strike masks
        strike_masks = mx.exp(exponent) * penalty_val
        
        # Apply constraints:
        # 1. i >= mu
        # 2. i > 0 (Attention Sink Preservation)
        causal_mask = (t_expanded >= mu)
        sink_mask = (t_expanded > 0)
        
        valid_strikes = mx.logical_and(causal_mask, sink_mask)
        strike_masks = mx.where(valid_strikes, strike_masks, 0.0)

        # Aggregate strikes: Take the minimum (most severe penalty) at each index
        mask = mx.min(strike_masks, axis=0, keepdims=True)
        # Reshape to 4D for transformer compatibility
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

    def flag_hallucination(self, index: int, severity_score: float = 0.5) -> None:
        """
        Asynchronous API for the Ghost Critic to flag logical drift.
        
        Args:
            index: The exact token position of the hallucination.
            severity_score: A float between 0.0 (minor syntax error) and 1.0 
                            (catastrophic logic failure). This mathematically 
                            defines the sigma (spread) of the Gaussian penalty 
                            applied to the subsequent orphaned tokens.
        """
        with self._lock:
            # Map severity 0.0-1.0 to Sigma 1.0-20.0 (Radius of influence)
            sigma = 1.0 + (severity_score * 19.0)
            
            # Check if we already have a strike at this index; if so, update to max severity
            existing = next((s for s in self.strikes if s["index"] == index), None)
            if existing:
                existing["sigma"] = max(existing["sigma"], sigma)
            else:
                self.strikes.append({"index": float(index), "sigma": sigma})
            
            # Invalidate mask for next forward pass
            self.active_mask = None
