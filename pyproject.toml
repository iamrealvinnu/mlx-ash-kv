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

# ⚡️ MLX-ASH-KV: Multi-Layer Memory Hypervisor

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Dashboard-emerald)](https://huggingface.co/spaces/guptavinay/mlx-ash-kv)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-slate)](https://github.com/iamrealvinnu/mlx-ash-kv)

**ASH-KV v8.0.2** is an enterprise-grade memory infrastructure for Large Language Models on Apple Silicon. It intercepts the `mlx_lm` inference pipeline to provide real-time causal correction and infinite context loops via the **Asynchronous Verification Daemon (AVD)** and **Entropy-Driven Context Compaction (EDCC)**.

---

## 🛠️ Implementation Tiers

### 1. The Production Proof: Live LLM Override
Perform a live "brain transplant" on a production Llama-3-8B model. Watch the AVD surgically prune logical drift and the EDCC vaporize dead RAM tokens *while the model is actively typing*.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 tests/ash_kv/live_llm_override.py
```

### 2. The Systems Diagnostic: Hardware TUI
A high-density Terminal UI for systems engineers. Visualize the 32-layer tensor manifold mutation and monitor ANE/GPU hardware parallelism using high-speed simulated tensors.
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 tests/ash_kv/benchmark.py
```

### 3. The Research Showcase: Web Dashboard
An interactive Plotly dashboard hosted on Hugging Face. Visualizes the 2D head-specific attention manifold and demonstrates the mathematical Gaussian decay and compaction protocols.

---

## 🔬 Core Architecture
*   **Asynchronous Verification Daemon (AVD):** Offloads hallucination detection to the Apple Neural Engine (ANE) to ensure zero GPU generation overhead.
*   **Head-Specific Causal Pruning:** Surgical 4D attention masking that corrects logic heads while preserving linguistic/stylistic heads.
*   **Entropy-Driven Context Compaction (EDCC):** Zero-latency physical RAM reclamation that enables effectively infinite context.
*   **Silicon-Native Implementation:** Fused Metal kernels (`@mx.compile`) eliminate Python bottlenecks.

---

## 🚀 Quick Start
```bash
pip install mlx-lm coremltools torch rich
python3 scripts/build_ane_critic.py
```

## 🛡️ License
Apache 2.0. Built for the future of agentic reasoning on Apple Silicon.
