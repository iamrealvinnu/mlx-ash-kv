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

# ASH-KV: The Self-Healing Middleware for LLMs

[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%26%20NVIDIA-blue)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)
[![Version](https://img.shields.io/badge/Version-8.2.1-emerald)](#)
[![Company](https://img.shields.io/badge/Developed%20by-GDI%20Nexus-black)](https://gdinexus.com)

**ASH-KV** is a high-performance, hardware-aware middleware layer designed for **High-Assurance Inference**. Developed by **GDI Nexus**, it surgically intercepts and corrects the KV cache at the silicon level, preventing logical drift and clinical hallucinations with zero detectable latency.

---

## 🏛️ Core Value Pillars

### ⚡ Zero-Latency Integrity
Surgical KV cache mutation at **Metal (Apple Silicon)** and **CUDA (NVIDIA)** speeds. Our Fused Kernels ensure that the "Immune System" adds virtually 0% overhead to inference throughput.

### 🔌 Hardware Agnostic (Universal HAL)
The **Hardware Abstraction Layer (HAL)** automatically detects your silicon and hot-swaps between **MLX** and **PyTorch** backends. The same code runs on an M4 MacBook or an NVIDIA H100 server.

### 🛡️ Adaptive Shielding & Real-Time Healing
Autonomous sensitivity scaling via the **AdaptiveSensitivity Agent**. Integrated with a **Deterministic Clinical Rules Engine (DCRE)**, ASH-KV monitors token generation in real-time and prunes attention heads the microsecond a contraindication is detected.

### ♾️ Infinite Horizon (NVMe Paging)
Break the VRAM ceiling. ASH-KV dynamically offloads "Cold" context chunks to NVMe storage, allowing for 100k+ token windows on consumer-grade hardware without OOM crashes.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install mlx-ash-kv
```

### 2. Corporate Integration (3 Lines of Code)
Integrate ASH-KV into any production pipeline to add an immediate safety layer.

```python
from mlx_ash_kv.api import protect

# Wrap your existing model with the ASH-KV shield
protected_model, cache, shield, proxies = protect(model, sensitivity=0.85)

# Inference continues normally, but with real-time surgical healing
```

---

## 🛠️ Command Center (CLI)
ASH-KV comes with a professional CLI for systems verification and benchmarking.

*   `ash-kv install`: Verify hardware drivers, silicon backend, and **NVMe Paging Stress Test**.
*   `ash-kv benchmark`: Run the 100-case "Hard Truth" evaluation suite.
*   `ash-kv monitor`: Launch the Live Diagnostic TUI to see layer-wise health and [HOT/WARM] memory distribution.
*   `ash-kv demo`: Launch the Gradio B2B Reliability Playground.

---

## 🔬 About GDI Nexus
**GDI Nexus** is a premier AI infrastructure firm. We are the architects of the AI-first era, blending deep data science with elite cloud orchestration. Our mission is to empower global enterprises with autonomous, reliable, and structurally resilient AI ecosystems.

### Locations
*   **USA (HQ)**: Woodbridge, VA 22191
*   **India**: Fingerpost Kandal, Udagamandalam, Tamil Nadu 643001

**Contact**: [contactus@gdinexus.com](mailto:contactus@gdinexus.com) | [www.gdinexus.com](https://gdinexus.com)

---

### ⚠️ DISCLAIMER
**ASH-KV is a hardware-level reliability layer designed to assist professionals. It is NOT a substitute for professional medical or legal judgment. All AI-generated outputs, even those "healed" by ASH-KV, must be verified by qualified human professionals before making clinical or legal decisions.**

---
© 2026 GDI Nexus Software Solutions LLP. All rights reserved.
