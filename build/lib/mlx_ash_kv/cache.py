"""
@file cache.py
@brief Core implementation of the Asynchronous Self-Healing Cache (ASH-KV).
"""

import threading
import math
import numpy as np
import coremltools as ct
import os
import time
import shutil
from typing import Tuple, List, Optional, Dict, Any

from .hal.factory import SiliconFactory

class PerformanceMonitor:
    """Measures execution time of Fused Metal Mutations."""
    def __init__(self):
        self.timings = []
    
    def record(self, duration_ns: int):
        self.timings.append(duration_ns)
        # Keep only last 100 timings
        if len(self.timings) > 100:
            self.timings.pop(0)
        
    @property
    def average_ms(self) -> float:
        if not self.timings: return 0.0
        return (sum(self.timings) / len(self.timings)) / 1e6

    @property
    def last_ms(self) -> float:
        if not self.timings: return 0.0
        return self.timings[-1] / 1e6

class MemoryGovernor:
    """
    Manages VRAM pressure by offloading LRU chunks to NVMe.
    """
    def __init__(self, cache_dir: str = ".ash_kv_paging", vram_limit_gb: float = 12.0):
        self.cache_dir = cache_dir
        self.vram_limit_gb = vram_limit_gb
        self.paged_layers: Dict[int, List[str]] = {}
        
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

    def should_page(self, current_vram_gb: float) -> bool:
        return current_vram_gb > self.vram_limit_gb

    def get_page_path(self, layer_idx: int, chunk_idx: int) -> str:
        return os.path.join(self.cache_dir, f"layer_{layer_idx}_chunk_{chunk_idx}.bin")

class ASHCache:
    """
    Asynchronous Self-Healing Cache (ASH-KV) v8.2.0 (Infinite Horizon).
    
    Implements vectorized Head-Specific Causal Pruning and NVMe Paging.
    """
    def __init__(self, critic_model_path: str = None, num_layers: int = 32, num_heads: int = 32, paging_enabled: bool = True):
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.layer_keys: List[Optional[Any]] = [None] * num_layers
        self.layer_values: List[Optional[Any]] = [None] * num_layers
        
        self.strikes: List[Dict[str, any]] = []
        self.active_mask: Optional[Any] = None
        self._lock = threading.Lock()
        
        self.perf_monitor = PerformanceMonitor()
        self.healer = SiliconFactory.get_healer()
        
        self.paging_enabled = paging_enabled
        self.governor = MemoryGovernor() if paging_enabled else None
        self.page_map: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_layers)}
        self.total_paged_tokens = 0
        
        self.critic_model = None
        if critic_model_path and os.path.exists(critic_model_path):
            self.critic_model = ct.models.MLModel(
                critic_model_path, 
                compute_units=ct.ComputeUnit.CPU_AND_NE
            )

    @property
    def seq_len(self) -> int:
        # Returns the current "hot" sequence length in VRAM
        if self.layer_keys[0] is not None:
            return self.layer_keys[0].shape[2]
        return 0

    @property
    def total_seq_len(self) -> int:
        # Returns total length (VRAM + NVMe)
        paged_len = 0
        if self.page_map[0]:
            paged_len = sum(p["len"] for p in self.page_map[0])
        return paged_len + self.seq_len

    def get_mask(self) -> Any:
        """Atomic retrieve/generate with memoization."""
        with self._lock:
            seq_len = self.seq_len
            if self.active_mask is None or self.active_mask.shape[3] != seq_len:
                start = time.perf_counter_ns()
                
                self.active_mask = self.healer.generate_mask(
                    seq_len, self.strikes, self.num_heads
                )
                
                self.healer.eval_arrays(self.active_mask)
                
                end = time.perf_counter_ns()
                if self.strikes:
                    self.perf_monitor.record(end - start)
                    
            return self.active_mask

    def update_layer(self, layer_idx: int, new_k: Any, new_v: Any) -> Tuple[Any, Any]:
        with self._lock:
            if self.layer_keys[layer_idx] is None:
                self.layer_keys[layer_idx] = new_k
                self.layer_values[layer_idx] = new_v
            else:
                self.layer_keys[layer_idx] = self.healer.concat_arrays(
                    [self.layer_keys[layer_idx], new_k], axis=2
                )
                self.layer_values[layer_idx] = self.healer.concat_arrays(
                    [self.layer_values[layer_idx], new_v], axis=2
                )
            
            # Paging trigger: if "hot" window exceeds 5000 tokens
            if self.paging_enabled and self.seq_len > 5000:
                self._page_layer_lru(layer_idx)
                
            return self.layer_keys[layer_idx], self.layer_values[layer_idx]

    def _page_layer_lru(self, layer_idx: int):
        """Offloads the first half (cold) of the layer to disk."""
        current_k = self.layer_keys[layer_idx]
        current_v = self.layer_values[layer_idx]
        
        mid = current_k.shape[2] // 2
        chunk_idx = len(self.page_map[layer_idx])
        
        # Paths
        k_path = self.governor.get_page_path(layer_idx, f"k_{chunk_idx}")
        v_path = self.governor.get_page_path(layer_idx, f"v_{chunk_idx}")
        
        # Slicing (Hardware Agnostic via Healer-compliant slicing)
        # We assume standard slicing works on backend tensors or use healer helper
        # For simplicity in this implementation, we use direct slicing
        cold_k = current_k[:, :, :mid, :]
        cold_v = current_v[:, :, :mid, :]
        hot_k = current_k[:, :, mid:, :]
        hot_v = current_v[:, :, mid:, :]
        
        # Offload
        self.healer.page_to_disk(cold_k, k_path)
        self.healer.page_to_disk(cold_v, v_path)
        
        # Update VRAM
        self.layer_keys[layer_idx] = hot_k
        self.layer_values[layer_idx] = hot_v
        
        # Track
        self.page_map[layer_idx].append({
            "k_path": k_path,
            "v_path": v_path,
            "len": mid,
            "shape": cold_k.shape,
            "dtype": cold_k.dtype
        })
        
        if layer_idx == 0:
            self.total_paged_tokens += mid
        
        self.active_mask = None # Invalidate mask for new hot length

    def get_context_chunk(self, layer_idx: int, start: int, end: int) -> Any:
        """Retrieves a chunk, pulling from disk if necessary."""
        # Simplified: for Ghost Critic analysis
        # If the index is in the paged region, load the chunk
        # This is a high-level abstraction for the demo
        with self._lock:
            # Check if start is in paged region
            paged_offset = 0
            for page in self.page_map[layer_idx]:
                if start >= paged_offset and start < paged_offset + page["len"]:
                    # Pull from disk
                    return self.healer.page_from_disk(page["k_path"], page["shape"], page["dtype"])
                paged_offset += page["len"]
            
            # If not paged, return from VRAM
            vram_start = start - paged_offset
            vram_end = end - paged_offset
            return self.layer_keys[layer_idx][:, :, vram_start:vram_end, :]

    def sync_eval(self, *arrays):
        with self._lock:
            self.healer.eval_arrays(*arrays)

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
            
            chunk = self.layer_keys[0][0, 0, start_idx:start_idx+chunk_size, :]
            self.healer.eval_arrays(chunk)
            
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
            
            reasoning_heads_mask = mask[0, :self.num_heads // 2, 0, :]
            
            # Using healer for take but still need mx for indexing logic on Mac
            import mlx.core as mx
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
                self.layer_keys[l] = self.healer.take_arrays(self.layer_keys[l], keep_indices, axis=2)
                self.layer_values[l] = self.healer.take_arrays(self.layer_values[l], keep_indices, axis=2)
            
            self.strikes.clear()
            self.active_mask = None
            self.sync_eval(*self.layer_keys, *self.layer_values)
            return tokens_freed
