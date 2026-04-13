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

**ASH-KV v4.1.1** is a high-performance memory infrastructure for Large Language Models on Apple Silicon. It introduces **"ANE-Daemon"**—true hardware-level parallelism that offloads hallucination verification to the Apple Neural Engine.

---

## 🔬 Breakthrough: ANE-Daemon Parallelism (Phase 2)

Traditional inference blocks the GPU or CPU to verify logic. ASH-KV v4.1.1 decouples these processes entirely by exploiting the **Apple Neural Engine (ANE)**.

### 🧠 Triple-Engine Architecture
1.  **Generation (GPU):** The primary LLM loop runs at maximum Metal throughput.
2.  **Verification (ANE):** A hardware-isolated Ghost Critic (Core ML) monitors the Unified Memory manifold asynchronously.
3.  **Healing (GPU/Metal):** When the ANE flags drift, a surgical Metal kernel applies **Gaussian Temporal Rollback** to excise the bad context.

### 📊 Hardware Receipts
- **Verification Engine:** Apple Neural Engine (ANE).
- **GPU Overhead:** `0.00%` during verification cycles.
- **Memory Protocol:** Zero-copy Unified Memory handoff (MLX to Core ML).
- **Stability:** Strict Command-Buffer Serialization (Metal-Safe).

---

## 🚀 Quick Start

### 1. Build the ANE Critic
```bash
python3 scripts/build_ane_critic.py
```

### 2. Integration
```python
from mlx_ash_kv import ASHCache

# Load with the ANE-compiled model
cache = ASHCache(critic_model_path="models/mock_critic.mlpackage")

# Inside generation loop
keys, values, mask = cache.update(new_k, new_v)

# Periodically analyze the manifold on the ANE (Non-blocking)
severity = cache.analyze_manifold_chunk(start_idx=0)
if severity > 0.8:
    cache.flag_hallucination(index=102, severity_score=severity)
```

---

## 📊 Diagnostics & Visualization

### Terminal Hardware Monitor (TUI)
Watch the ANE firing and the manifold healing in real-time:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 tests/ash_kv/benchmark.py
```

---

## 🛡️ License
Apache 2.0. Built for the future of agentic reasoning on Apple Silicon.
