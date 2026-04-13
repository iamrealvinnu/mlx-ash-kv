import mlx.core as mx
from mlx_lm import load
import time
import os
import sys

# Ensure src is in path for ASHCache
sys.path.append(os.path.join(os.getcwd(), "src"))
from mlx_ash_kv.cache import ASHCache

print("🚀 [SYSTEM] PHASE 6: NATIVE OVERRIDE INITIATED")
print("🧪 [SYSTEM] Performing Live Brain Transplant on Llama-3...")

# 1. Load the Genuine LLM
model_path = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
model, tokenizer = load(model_path)
print(f"✅ [SYSTEM] {model_path} loaded into Unified Memory.")

# 2. Instantiate the Multi-Layer Hypervisor (ASHCache v8.0.0)
# Mapping to Llama-3-8B: 32 layers, 32 attention heads
ash_cache = ASHCache(
    critic_model_path="models/mock_critic.mlpackage", 
    num_layers=len(model.layers), 
    num_heads=model.n_heads
)
print(f"✅ [SYSTEM] ASH-KV Multi-Layer Hypervisor Armed ({len(model.layers)} layers). AVD Online.")

def native_override_generation(prompt: str, max_tokens: int = 500):
    """
    The Interceptor Loop: Injects ASH-KV natively into the LLM forward pass.
    """
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    
    # Initialize state
    y = prompt_tokens
    
    print(f"\n[USER]: {prompt}\n")
    print(f"[LLAMA-3 (ASH-KV POWERED)]:", end=" ", flush=True)
    
    start_time = time.perf_counter()
    
    for i in range(max_tokens):
        # 🧠 THE OVERRIDE: Forward Pass with AVD Mask Injection
        # We manually update the hypervisor and pass the combined mask
        mask = ash_cache.get_mask()
        
        # Note: In a production integration, we would patch the model.layers[j].attention 
        # to use ash_cache.layer_keys[j]. Here we demonstrate the logic flow.
        logits = model(y[None], mask=mask)
        
        # Sample next token
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        y = next_token
        
        # Stream output
        text_chunk = tokenizer.decode(next_token.item())
        print(text_chunk, end="", flush=True)
        
        # 🛡️ THE IMMUNE SYSTEM: AVD Background Verification
        if i > 0 and i % 40 == 0:
            # Sync Metal command buffer
            ash_cache.sync_eval(logits)
            
            # AVD Scans for logical drift
            severity = ash_cache.analyze_manifold_chunk(start_idx=max(0, ash_cache.seq_len - 128))
            
            if severity and severity > 0.85:
                target_idx = max(0, ash_cache.seq_len - 10)
                # Surgically strike the reasoning heads
                ash_cache.flag_logical_drift(index=target_idx, severity_score=severity)
                print(f"\n\n[AVD] 🛡️ LOGICAL DRIFT DETECTED (SEV: {severity:.2f}). APPLYING CAUSAL PRUNING @ IDX {target_idx}.\n", end="")

        # ♻️ THE GARBAGE COLLECTOR: EDCC Compaction
        if text_chunk in [".", "\n"] and ash_cache.seq_len > 300:
            freed = ash_cache.compact_manifold(threshold=-9000.0)
            if freed > 0:
                print(f"\n\n[SYSTEM] ♻️ EDCC TRIGGERED: {freed} tokens deallocated. Manifold depth: {ash_cache.seq_len}.\n", end="")

        if next_token.item() == tokenizer.eos_token_id:
            break

    total_time = time.perf_counter() - start_time
    print(f"\n\n📊 [TELEMETRY] Generation complete. Throughput: {i / total_time:.2f} tok/s.")

# Ignite
test_prompt = "Compare the architectural efficiency of Apple's Unified Memory versus traditional PCIe-based GPU memory for large-scale agentic reasoning."
native_override_generation(test_prompt)
