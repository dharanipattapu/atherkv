# atherkv
# ⚡ AetherKV: High-Performance LLM KV-Cache Compression & Scheduling Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AetherKV** is a software-based, software-only middleware layer designed to accelerate Long-Context LLM Inference. By leveraging dynamic **Shannon Entropy profiling**, **block-wise INT4 quantization**, and **asynchronous PCIe host streaming**, AetherKV drastically reduces Key-Value (KV) cache memory footprint without requiring hardware modifications.

---

## 🎯 Key Metrics & Highlights

* **~69.7% VRAM Reduction:** Drastically compresses low-impact context tokens.
* **3.3x Memory Compression:** Expands effective GPU sequence capacity and continuous batching concurrency.
* **100% Retrieval Retention:** Retains complete accuracy in Needle-In-A-Haystack (NIAH) benchmark evaluations.
* **CPU Vectorized Fallback:** Fully operational across local CPU dev environments and high-performance NVIDIA CUDA GPUs.

---

## 🏛️ System Architecture

```text
aetherkv/
├── csrc/                       # [C++ / PyBind11 Engine]
│   ├── include/
│   │   └── entropy_profiler.h  # LibTorch profiling header
│   └── entropy_profiler.cpp    # Shannon Entropy calculation & tier classification
├── aetherkv/                   # [Python Core Engine]
│   ├── device.py               # Hardware Dispatcher (GPU / CPU Vectorized)
│   ├── core/
│   │   ├── profiler.py         # Module 1: EntropyProfiler (C++ & PyTorch Fallback)
│   │   ├── quant_kernel.py     # Module 2: AdaptiveCacheAllocator (INT4 Quantization)
│   │   ├── offloader.py        # Module 3: AsyncPCIeOffloader (CUDA Streams)
│   │   └── scheduler.py        # Module 4: EntropyAwareScheduler (VRAM Admission)
│   └── utils/
│       └── metrics.py          # VRAM, F1-Score, and Retrieval Evaluation Utilities
├── benchmarks/                 # [Performance Suite]
│   ├── benchmark_suite.py      # End-to-End Latency, Concurrency & VRAM Benchmark
│   └── niah_benchmark.py       # Needle-In-A-Haystack Factual Retrieval Test
├── dashboard.py                # Streamlit Real-Time Visual Telemetry Engine
├── setup.py                    # PyBind11 C++ Extension Build Script
└── CMakeLists.txt              # CMake LibTorch Build System

AbstractThe deployment of Long-Context Large Language Models (LLMs) in production environments is strictly constrained by the linear expansion of Key-Value (KV) cache memory during autoregressive decoding. While state-of-the-art continuous batching frameworks utilize block memory management (e.g., PagedAttention) to eliminate fragmentation, they maintain cached tokens in uniform high-precision representations (BF16/FP16), causing severe GPU VRAM exhaustion and capping batch concurrency.To resolve this bottleneck, we present AetherKV, a software-only, layer-wise memory management and scheduling middleware designed to compress and offload KV caches dynamically during LLM forward passes. AetherKV introduces a multi-tiered memory architecture operating in four synchronized modules:Shannon Entropy Profiling (EntropyProfiler): Intercepts attention matrices to compute per-token Shannon Entropy ($H(X) = -\sum P \log_2 P$), dynamically categorizing tokens into critical Attention Sinks, Recent Locality windows, and Low-Impact Passive context.Adaptive Block Quantization (AdaptiveCacheAllocator): Protects Sink and Recent tokens in high-precision BF16 format while dynamically compressing Passive blocks to INT4 precision using block-wise dynamic scale factors.Asynchronous Host Offloading (AsyncPCIeOffloader): Offloads compressed INT4 Passive blocks to pinned CPU Host RAM over dedicated, non-blocking CUDA streams to eliminate GPU execution stalls.VRAM-Aware Admission Scheduling (EntropyAwareScheduler): Dynamically adjusts memory headroom based on real-time compression ratios to maximize continuous multi-user batch concurrency.Evaluated on Needle-In-A-Haystack (NIAH) retrieval benchmarks and end-to-end memory profiling, AetherKV achieves a ~69.7% reduction in KV-cache VRAM footprint and a 3.3× memory compression ratio, while maintaining 100% factual context retrieval accuracy. Designed with dual-path dispatching, AetherKV seamlessly executes on high-performance NVIDIA GPUs via PyBind11 C++/Triton kernels as well as standard CPU vectorized runtime environments.
