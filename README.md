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

**ASH-KV v5.0.2** is a high-performance memory infrastructure for Large Language Models on Apple Silicon. It introduces the **"Immortal Manifold"**—a self-cleaning cache architecture that enables effectively infinite local context loops via High-Entropy Semantic Compaction.

---

## 🔬 Breakthrough: Entropy-Driven Semantic Compaction (Phase 3)

Traditional KV caches suffer from linear memory bloat. Even with masking, "dead" tokens physically occupy RAM. ASH-KV v5.0.2 solves this by implementing **Biological Memory Consolidation** with a High-Entropy stability layer.

### 🧠 The Splicing Mechanism
1.  **Hardware Monitor (ANE):** The ANE-Daemon continuously identifies hallucinated or low-entropy nodes in Unified Memory.
2.  **User-Pause Trigger:** During idle cycles (simulated or natural), the engine triggers a physical memory compaction.
3.  **The Surgical Splice:** A zero-latency Metal kernel physically slices out the low-entropy tokens and stitches the healthy context back together using `mx.take`.
4.  **Stability (v5.0.2):** Implements **High-Entropy Skip logic** and synchronized GPU-to-CPU readbacks to prevent thread-stalls and Metal command-buffer collisions.

### 📊 Hardware Receipts
- **Memory Strategy:** Physical In-Place Compaction (O(1) indexing).
- **Context Limit:** Theoretically Infinite (constrained only by healthy information density).
- **Parallelism:** Triple-Engine (GPU Generation, ANE Verification, Metal Splicing).

---

## 🚀 Quick Start

### 1. Build the ANE Critic
```bash
python3 scripts/build_ane_critic.py
```

### 2. Integration
```python
from mlx_ash_kv import ASHCache

# Load the Immortal Manifold engine
cache = ASHCache(critic_model_path="models/mock_critic.mlpackage")

# Periodically trigger compaction during idle time
# v5.0.2 automatically skips if the manifold is already optimal
freed = cache.compact_manifold(threshold=-9000.0)
```

---

## 📊 Diagnostics & Visualization

### Terminal Hardware Monitor (TUI)
Watch the manifold physically shrink and memory being reclaimed in real-time:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 tests/ash_kv/benchmark.py
```

---

## 🛡️ License
Apache 2.0. Built for the future of agentic reasoning on Apple Silicon.
