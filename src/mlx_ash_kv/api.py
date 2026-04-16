import threading
import time
import os
from typing import Any, Optional, List, Tuple, Generator

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .cache import ASHCache
from .critic import UniversalTensorCritic

# --- 🧠 THE HIVE MIND PROXY ---
class ASHCacheProxy:
    def __init__(self, hypervisor: ASHCache, layer_idx: int):
        self.hypervisor = hypervisor
        self.layer_idx = layer_idx
        self.offset = 0

    def update_and_fetch(self, keys: Any, values: Any) -> Tuple[Any, Any]:
        k, v = self.hypervisor.update_layer(self.layer_idx, keys, values)
        # Sequence length indexing (compatible with MLX and PyTorch)
        self.offset = k.shape[2]
        return k, v

# --- 🧪 THE BRAIN TRANSPLANT (SDPA PATCH) ---
_patched = False
def patch_mlx_lm(ash_cache: ASHCache):
    global _patched
    if _patched or not HAS_MLX: return
    
    import mlx_lm.models.base as base_models
    from mlx_lm.models.base import create_causal_mask

    original_sdpa = base_models.scaled_dot_product_attention

    def patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        if ash_cache.strikes:
            seq_len = keys.shape[2]
            custom_mask = ash_cache.get_mask()
            custom_mask = mx.array(custom_mask, dtype=queries.dtype)
            
            if isinstance(mask, str) and mask == "causal":
                L = queries.shape[2]
                offset = seq_len - L
                mask = create_causal_mask(L, offset)
                mask = mx.array(mask, dtype=queries.dtype)
            
            if mask is not None:
                mask = mask + custom_mask
            else:
                mask = custom_mask
                
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    base_models.scaled_dot_product_attention = patched_sdpa
    _patched = True

class AdaptiveSensitivity:
    """
    Autonomous Calibration Agent for ASH-KV.
    Adjusts the 'sensitivity' threshold based on historical AVD scores
    to balance model performance and integrity.
    """
    def __init__(self, initial_sensitivity: float = 0.85):
        self.sensitivity = initial_sensitivity
        self.history: List[float] = []
        self._lock = threading.Lock()
        
    def record_score(self, score: float):
        with self._lock:
            self.history.append(score)
            if len(self.history) > 50:
                self.history.pop(0)
            
            if len(self.history) >= 10:
                recent = self.history[-5:]
                avg_recent = sum(recent) / len(recent)
                variance = sum((x - avg_recent)**2 for x in recent) / len(recent)
                
                if avg_recent > 0.75 and variance < 0.01:
                    self.sensitivity = min(0.95, self.sensitivity + 0.01)
                elif avg_recent > 0.9: 
                    self.sensitivity = max(0.5, self.sensitivity - 0.05)

    @property
    def current_threshold(self) -> float:
        with self._lock:
            return self.sensitivity

def protect(model: Any, sensitivity: float = 0.85, critic_model_path: Optional[str] = None):
    """
    Wraps a model with ASH-KV protection.
    """
    num_layers = getattr(model, "num_layers", 32)
    num_heads = getattr(model, "n_heads", 32)
    
    cache = ASHCache(critic_model_path=critic_model_path, num_layers=num_layers, num_heads=num_heads)
    adapter = AdaptiveSensitivity(initial_sensitivity=sensitivity)
    
    patch_mlx_lm(cache)
    proxies = [ASHCacheProxy(cache, i) for i in range(num_layers)]
    
    return model, cache, adapter, proxies

def generate_stream(
    model: Any, 
    tokenizer: Any, 
    cache: ASHCache, 
    proxies: List[ASHCacheProxy],
    prompt: str,
    max_tokens: int = 512,
    adapter: Optional[AdaptiveSensitivity] = None
) -> Generator[Tuple[str, float], None, None]:
    """
    Yields tokens and real-time health scores using the UniversalTensorCritic.
    """
    critic = UniversalTensorCritic()
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    y = mx.array(tokenizer.encode(formatted_prompt))
    
    for i in range(max_tokens):
        logits = model(y[None], cache=proxies)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        y = next_token
        
        token_id = next_token.item()
        text_chunk = tokenizer.decode(token_id)
        
        # Real-time mathematical evaluation of the tensor manifold
        drift_score = critic.evaluate_tensor_drift(cache)
        
        if drift_score > 0:
            if adapter: adapter.record_score(drift_score)
            threshold = adapter.current_threshold if adapter else 0.85
            
            if drift_score > threshold:
                # Trigger the Fused Metal Kernel mutation based on uncertainty
                cache.flag_logical_drift(index=cache.seq_len - 1, severity_score=drift_score)
        
        # Health score is inverse of drift
        health_score = 1.0 - drift_score
        
        yield text_chunk, health_score
        
        if token_id == tokenizer.eos_token_id:
            break
