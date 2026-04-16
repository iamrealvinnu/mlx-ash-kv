# ASH-KV: The Self-Healing Middleware for LLMs

[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20%26%20NVIDIA-blue)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](#)

**ASH-KV** is a high-performance, hardware-aware middleware layer designed to provide **Runtime Integrity** for Large Language Models. By surgically intercepting and correcting the KV cache at the silicon level, it prevents logical drift and hallucinations with zero detectable latency.

---

## 🏛️ Core Value Pillars

### ⚡ Zero-Latency Integrity
Surgical KV cache mutation at **Metal (Apple Silicon)** and **CUDA (NVIDIA)** speeds. Our Fused Kernels ensure that the "Immune System" adds virtually 0% overhead to inference throughput.

### 🔌 Hardware Agnostic (Universal HAL)
The **Hardware Abstraction Layer (HAL)** automatically detects your silicon and hot-swaps between **MLX** and **PyTorch** backends. The same code runs on an M4 MacBook or an NVIDIA H100 server.

### 🛡️ Adaptive Shielding
Autonomous sensitivity scaling via the **AdaptiveSensitivity Agent**. The system learns from historical AVD (Asynchronous Verification Daemon) scores to balance strict integrity with model creativity.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install .
```

### 2. Corporate Integration (3 Lines of Code)
Integrate ASH-KV into any production pipeline to add an immediate safety layer.

```python
from mlx_ash_kv.api import protect

# Wrap your existing model with the ASH-KV shield
protected_model, cache, shield = protect(model, sensitivity=0.85)

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

## 🔬 Scientific Foundation
ASH-KV implements **Asynchronous Self-Healing** protocols that offload hallucination detection to secondary silicon (like the ANE or secondary GPU cores), ensuring the main generation loop remains unobstructed.

---

### ⚠️ DISCLAIMER
**ASH-KV is a hardware-level reliability layer designed to assist professionals. It is NOT a substitute for professional medical or legal judgment. All AI-generated outputs, even those "healed" by ASH-KV, must be verified by qualified human professionals before making clinical or legal decisions.**

---
Built for the future of mission-critical Agentic Reasoning.
      run_benchmark()
    elif args.command == "monitor":
        run_monitor()
    elif args.command == "install":
        check_install()
    elif args.command == "demo":
        run_demo()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
