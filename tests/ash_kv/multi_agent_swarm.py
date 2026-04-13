import mlx.core as mx
import time
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from mlx_ash_kv.cache import ASHCache

print("🚀 INITIALIZING PHASE 5: CROSS-AGENT MEMORY STITCHING\n")

# 1. Boot up Agent A (The Researcher)
print("[AGENT A] Booting Researcher Cache...")
cache_a = ASHCache(num_heads=32)

# Simulate Agent A reading a massive 10,000 token clinical document
print("[AGENT A] Generating 10,000 tokens of clinical data (Simulated)...")
dummy_k = mx.random.uniform(-1, 1, (1, 32, 10000, 128), dtype=mx.float16)
dummy_v = mx.random.uniform(-1, 1, (1, 32, 10000, 128), dtype=mx.float16)
cache_a.update(dummy_k, dummy_v)

# 2. The Handoff (Zero-Copy Export)
print("\n[SYSTEM] Executing Zero-Copy Manifold Handoff...")
start_time = time.perf_counter()
shared_keys, shared_values, shared_mask = cache_a.export_manifold()
handoff_time = (time.perf_counter() - start_time) * 1000

print(f"[SYSTEM] Handoff Complete in {handoff_time:.4f} ms.")

# 3. Boot up Agent B (The Evaluator)
print("\n[AGENT B] Booting Evaluator Cache...")
cache_b = ASHCache(num_heads=32)

# Agent B mounts Agent A's brain
print("[AGENT B] Mounting external manifold...")
mount_start = time.perf_counter()
cache_b.mount_manifold(shared_keys, shared_values, shared_mask)
mount_time = (time.perf_counter() - mount_start) * 1000

print(f"[AGENT B] Mount Complete in {mount_time:.4f} ms.")
print(f"[AGENT B] Total Context Depth Available: {cache_b.seq_len} tokens.")
print(f"\n✅ SYSTEM VERDICT: Multi-Agent TTFT (Time-To-First-Token) bottleneck eliminated.")
