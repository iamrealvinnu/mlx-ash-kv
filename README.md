---
title: MLX-ASH-KV
emoji: ⚡
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: apache-2.0
---

# ⚡️ MLX-ASH-KV: Asynchronous Self-Healing Cache

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Dashboard-emerald)](https://huggingface.co/spaces/guptavinay/mlx-ash-kv)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-slate)](https://github.com/iamrealvinnu/mlx-ash-kv)

**ASH-KV v4.0.0** is a high-performance memory infrastructure for Large Language Models on Apple Silicon. It introduces **"Temporal Rollback"**—a zero-latency mechanism for excising hallucinations and their causal downstream contamination.

---

## 🔬 Breakthrough: Temporal Rollback (Phase 1)

Traditional masking is binary: a token is either active or dead. In v4.0.0, we solve the **"Orphaned State"** problem (tokens generated *after* a hallucination but *before* its detection) using soft causal correction.

### 🧠 Gaussian-Decay Manifold
When a hallucination is flagged, ASH-KV projects a **Gaussian gravity well** onto the attention manifold:
1.  **Surgical Strike:** The flagged node receives a `-10000.0` penalty (Softmax zero).
2.  **Causal Decay:** Subsequent orphaned tokens receive a decaying penalty based on their proximity to the strike, mathematically diluting their influence on future generation.
3.  **Sink Preservation:** The `<bos>` token (Index 0) is strictly protected as an **Attention Sink**, ensuring the Softmax probability mass has a stable anchor to prevent catastrophic collapse.

### 📊 The Receipts (Architectural Guarantees)
- **Critic Overhead:** Asynchronous / Non-blocking (Bypasses primary generation thread).
- **Mutation Latency:** Native Metal Speed (Fused via `@mx.compile`).
- **Throughput:** Zero O(N) memory reallocation penalties during healing events.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/iamrealvinnu/mlx-ash-kv.git
cd mlx-ash-kv
pip install .
```

### 🧩 MLX-LM Integration
```python
from mlx_lm import load
from mlx_ash_kv import ASHCache

model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
cache = ASHCache()

# Inside your loop:
for token in generation_loop:
    keys, values, immune_mask = cache.update(new_k, new_v)
    
    # Asynchronous strike from a Ghost Critic:
    # severity_score (0.0 to 1.0) defines the Gaussian spread (sigma)
    cache.flag_hallucination(index=102, severity_score=0.7)
    
    logits = model(token, cache=(keys, values), mask=immune_mask)
```

---

## 📊 Diagnostics & Visualization

### 1. Terminal Monitor (TUI)
Visualize Gaussian gradients and sink preservation in real-time:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 tests/ash_kv/benchmark.py
```

### 2. Neural Triage Dashboard (Web)
Launch the interactive Plotly dashboard to visualize the spectral attention manifold and simulated Ghost Critic strikes:
```bash
pip install gradio plotly numpy
python3 app.py
```

---

## 🛡️ License
Apache 2.0. Built for the future of agentic reasoning on Apple Silicon.
