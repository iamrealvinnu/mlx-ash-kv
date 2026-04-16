

# ASH-KV: Dynamic Attention Steering & KV-Cache Integrity Middleware

[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%26%20NVIDIA-blue)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)
[![Version](https://img.shields.io/badge/Version-8.3.0--beta-emerald)](#)
[![Company](https://img.shields.io/badge/Developed%20by-GDI%20Nexus-black)](https://gdinexus.com)

**ASH-KV** (Asynchronous Self-Healing KV Cache) is a high-performance middleware layer designed for **Runtime Manifold Integrity Enforcement**. It leverages silicon-native kernels to monitor the mathematical uncertainty (**Varentropy**) of the Attention Manifold and surgically prunes logical drift at the hardware level.

---

## Technical Methodology

### Varentropy-Proxy Monitoring
ASH-KV implements a deterministic uncertainty detector by analyzing the mathematical variance across the KV-Cache. This real-time analysis identifies **Manifold Collapse**—the mathematical state where a model's transition probability distribution becomes unstable—allowing for intervention before semantic errors materialize.

### Real-Time Attention Steering
When uncertainty exceeds the threshold, ASH-KV executes a **Gaussian Manifold Mutation** directly within the compute graph.
*   **Apple Silicon**: Leverages `@mx.compile` Fused Metal kernels for sub-millisecond mutation.
*   **NVIDIA**: Implements synchronized PyTorch/CUDA tensor operations via the Hardware Abstraction Layer (HAL).
*   **Latency**: Verified at **< 0.9ms** on Apple M4 hardware (negligible inference overhead).

### Infinite Horizon (NVMe Paging)
To bypass physical VRAM constraints, ASH-KV utilizes an LRU-based paging protocol. Inactive context chunks are offloaded to NVMe storage using zero-copy memory mapping, enabling 100k+ token windows on consumer-grade hardware.

---

## API Reference

### `protect(model, sensitivity=0.85, critic_model_path=None)`
Initializes the ASH-KV Hypervisor for a given neural model.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | Required | An MLX or PyTorch model instance. |
| `sensitivity` | `float` | `0.85` | The Varentropy threshold (0.0 to 1.0). |
| `critic_model_path` | `str` | `None` | Optional path for ANE-accelerated manifold critics. |

---

## Research & Reproducibility
Our benchmarks use `time.perf_counter_ns()` to track the exact overhead of the Fused Metal Mutations.
```bash
ash-kv install    # Platform driver verification
ash-kv benchmark  # Unified Latency & Integrity suite
```

---

## Hardware Abstraction Layer (HAL)
*   **`MLXHealer`**: Fused Metal backends for macOS.
*   **`CudaHealer`**: Synchronized tensor backends for NVIDIA/Linux.
*   **`UniversalTensorCritic`**: Pure mathematical manifold evaluation.

---

### DISCLAIMER
ASH-KV is a probabilistic reliability layer. It is NOT a substitute for professional clinical or legal judgment. All AI-generated outputs must be verified by qualified human professionals.

---
© 2026 GDI Nexus Software Solutions LLP. All rights reserved.
