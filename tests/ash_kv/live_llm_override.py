import mlx.core as mx
from mlx_lm import load
import time
import os
import sys
from typing import Any, Optional, Tuple

# Ensure src is in path for ASHCache
sys.path.append(os.path.join(os.getcwd(), "src"))
from mlx_ash_kv.cache import ASHCache

# --- 🧠 THE HIVE MIND PROXY ---
class ASHCacheProxy:
    def __init__(self, hypervisor: ASHCache, layer_idx: int):
        self.hypervisor = hypervisor
        self.layer_idx = layer_idx
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        k, v = self.hypervisor.update_layer(self.layer_idx, keys, values)
        self.offset = k.shape[2]
        return k, v

# --- 🧪 THE BRAIN TRANSPLANT (SDPA PATCH) ---
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

# --- 🚀 EXECUTION ---

model_path = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
model, tokenizer = load(model_path)

num_layers = getattr(model, "num_layers", 32)
num_heads = getattr(model, "n_heads", 32)

ash_cache = ASHCache(
    critic_model_path="models/mock_critic.mlpackage", 
    num_layers=num_layers, 
    num_heads=num_heads
)

cache_proxies = [ASHCacheProxy(ash_cache, i) for i in range(num_layers)]

def generate_interactive(max_tokens: int = 1000):
    print("\n" + "="*60)
    print("🧠 ASH-KV v8.0.2 INTERACTIVE OVERRIDE (Llama-3-8B)")
    print("Type 'exit' to quit. Press Ctrl+C to stop generation.")
    print("="*60 + "\n")

    while True:
        try:
            prompt = input("\n[USER]: ")
            if prompt.lower() in ["exit", "quit"]:
                break
            
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            y = mx.array(tokenizer.encode(formatted_prompt))
            
            print(f"\n[LLAMA-3 (ASH-KV POWERED)]:", end=" ", flush=True)
            
            start_time = time.perf_counter()
            tokens_generated = 0
            
            for i in range(max_tokens):
                logits = model(y[None], cache=cache_proxies)
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
                y = next_token
                
                text_chunk = tokenizer.decode(next_token.item())
                print(text_chunk, end="", flush=True)
                tokens_generated += 1
                
                # 🛡️ THE IMMUNE SYSTEM (AVD)
                if i > 0 and i % 40 == 0:
                    ash_cache.sync_eval(logits)
                    severity = ash_cache.analyze_manifold_chunk(start_idx=max(0, ash_cache.seq_len - 128))
                    
                    if severity and severity > 0.5:
                        target_idx = max(0, ash_cache.seq_len - 5)
                        ash_cache.flag_logical_drift(index=target_idx, severity_score=severity)
                        print(f"\n\n[AVD] 🛡️ LOGICAL DRIFT ({severity:.2f}) -> PRUNING @ {target_idx}\n", end="", flush=True)

                # ♻️ THE GARBAGE COLLECTOR (EDCC)
                if any(mark in text_chunk for mark in [".", "\n", "!", "?"]) and ash_cache.seq_len > 400:
                    freed = ash_cache.compact_manifold(threshold=-9000.0)
                    if freed > 0:
                        print(f"\n\n[SYSTEM] ♻️ EDCC COMPACTED: {freed} tokens pruned.\n", end="", flush=True)

                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            elapsed = time.perf_counter() - start_time
            print(f"\n\n📊 [TELEMETRY] {tokens_generated} tokens @ {tokens_generated/elapsed:.2f} tok/s")
            
        except KeyboardInterrupt:
            print("\n[SYSTEM] Generation interrupted.")
            continue

if __name__ == "__main__":
    generate_interactive()
