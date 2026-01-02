# Hybrid CPU+GPU LLM Inference on Windows with ONNX Runtime
*A PowerInfer/FlexGen/Fiddler/FlexInfer-inspired design using ONNX Runtime (ORT) + Execution Providers (EPs), aimed at running models far beyond GPU VRAM.*

> **Core idea:** ORT won’t “offload N layers to GPU” like llama.cpp by itself. But you can **build that behavior** by (1) **profiling** the model to identify “hot” compute/weights, (2) **partitioning** the ONNX graph into **submodels** (GPU + CPU partitions), and (3) orchestrating execution with a **scheduler** that overlaps compute and transfers and adapts to memory/latency constraints.

---

## 1) What ONNX Runtime Can and Can’t Do (Practical Clarification)

### What ORT does well
- **Fast execution** via EPs (CUDA, TensorRT-RTX, CPU, DirectML/OpenVINO in other cases).
- **Fallback**: unsupported ops for a higher-priority EP can execute on a lower-priority EP (typically CPU).

### What ORT does *not* do natively (today)
- **Explicit layer offload** (“put first N transformer layers on GPU, rest on CPU”).
- **Automatic multi-device sharding** across GPU+CPU or multiple GPUs for *one* inference request.
- **A built-in hybrid scheduler** that balances latency/throughput across heterogeneous devices.

### Therefore
To run **models that exceed VRAM**, you need **manual partitioning and orchestration** around ORT sessions.

---

## 2) Proposed Solution: “Hot-Partitioned Pipeline” (HPP)

### Goals
- Run **huge models** (e.g., 60GB) on **limited VRAM** (e.g., 24GB) by keeping the **most valuable/expensive compute** on GPU.
- Keep correctness intact (no approximations beyond chosen quantization).
- Minimize PCIe stalls via **asynchronous staging** and **overlap**.

### High-level approach
1. **Offline profiling** to compute “hotness” scores per layer/block.
2. **Partition** model into **GPU ONNX** and **CPU ONNX** submodels (and optionally more partitions).
3. **Run sessions in a pipeline**: GPU partition(s) + CPU partition(s), with explicit tensor handoff.
4. **Add scheduling**: prefetch, pinned buffers, stream overlap, cache policy, dynamic retuning.

---

## 3) Offline Profiling and Hotness Scoring

You want the GPU to host the parts that yield the best payoff under VRAM constraints.

### What to score
- **Compute cost** per layer/block (e.g., matmul FLOPs, attention cost).
- **Activation “importance”** proxies:
  - average |activation| or variance (proxy for “signal strength”),
  - sensitivity metrics (how perturbations change logits),
  - attention entropy / head importance (optional).
- **Memory footprint**:
  - weights,
  - KV cache growth contributions,
  - intermediate tensor sizes.
- **Observed latency**: per-block runtime on CPU vs GPU.

### Practical hotness metric
A robust, engineering-friendly score:

`hotness(block) = (cpu_time(block) - gpu_time(block)) / (gpu_weight_bytes(block) + kv_pressure(block))`

Interpretation: GPU gets blocks with the highest “latency savings per VRAM byte”.

### How to gather profiling data
- Run a representative prompt set (your actual usage distribution matters).
- Collect:
  - block-level time,
  - tensor sizes,
  - (optional) activation stats.
- If you can’t instrument inside a single ONNX easily, profile **candidate partitions** or build an **instrumented export** that exposes intermediate taps at layer boundaries.

---

## 4) Partitioning Strategy

### Why partition into submodels
ORT’s EP system won’t reliably assign “layer N to CPU” when everything is supported on GPU—plus VRAM OOM happens before you get a graceful split.

So you create **separate ONNX graphs**:
- `model_gpu.onnx`: hot blocks, optimized for CUDA/TensorRT
- `model_cpu.onnx`: cold blocks, optimized for CPU

Optionally:
- multiple GPU partitions (e.g., `gpu_0.onnx`, `gpu_1.onnx`) for tighter VRAM packing,
- multiple CPU partitions for threading and cache locality.

### Recommended cut points (transformer LLM)
- Partition at **layer boundaries** (best for correctness and stable interfaces).
- Ensure interfaces include:
  - hidden state tensor,
  - KV cache tensors for that layer range (or a global KV cache model; see below),
  - attention mask / positional encodings if needed.

### KV cache placement
KV cache is often the *real* memory hog during long chats.
- Keep **KV cache for GPU layers** on GPU memory.
- Keep KV cache for CPU layers on CPU memory.
- Consider “paged KV” designs: CPU holds full cache; GPU pulls windows (harder, but huge win).

---

## 5) Runtime Orchestration Principles (Borrowed from PowerInfer/FlexGen/FlexInfer)

### A) VRAM as a cache
Treat GPU VRAM as a **managed cache** of:
- weights (for hot blocks),
- (optionally) sub-KV caches,
- intermediate buffers.

### B) Avoid CPU↔GPU ping-pong
The biggest throughput killer is bouncing activations every layer.
- Partition in **coarse chunks** to reduce transfers.
- Overlap transfers with compute (async copies, pinned buffers).

### C) Pipelining and prefetch
While GPU is computing token *t* on `gpu_part`, CPU can:
- prepare next token inputs,
- run its own part for token *t-1* (if pipeline depth allows),
- prefetch weights for the next partition.

### D) Dynamic tuning
Adjust partition boundaries based on:
- VRAM headroom (changes with context length),
- observed latency (CPU load / thermal throttling),
- model mix (different models have different hotness curves).

---

## 6) “Better Ways” Than Static Layer Hotness (Yes)

Your “most frequently referenced layers” idea is directionally good, but there are often *better control knobs*:

### 1) Focus on **KV cache**, not just weights
For long contexts, KV cache dominates VRAM, not weights.
- Offloading certain layers’ KV to CPU can extend context or allow bigger models.

### 2) Use **weight streaming / paging**
Instead of permanently assigning layers, stream weights for a block into GPU just-in-time
(“GPU as a cache, CPU as backing store”). This is closer to FlexGen’s memory-centric view.

### 3) Quantize differently per partition
- GPU partition: FP16 or INT8/FP8/4-bit depending on EP support.
- CPU partition: INT8 (or even 4-bit if CPU kernels support) to reduce RAM bandwidth and improve cache.

### 4) Split **within a layer** only if forced
Most benefit comes from coarse partitioning; splitting inside attention/FFN increases handoffs.

### 5) Consider **MoE-aware partitioning**
For MoE models, only a subset of experts activate per token:
- Keep the “popular experts” on GPU, cold experts on CPU or even disk.
This can outperform static layer partitioning.

---

# Reference Architecture

## A) Component Diagram (Logical)

```
+---------------------------+        +---------------------------+
|        UI / CLI           |        |     Telemetry / Logs      |
|  - Chat history           |        | - tokens/sec              |
|  - Model selection        |        | - device memory           |
|  - Params (temp, top-p)   |        | - per-part latency        |
+-------------+-------------+        +-------------+-------------+
              |                                        ^
              v                                        |
+-------------+----------------------------------------+--------+
|                     Chat Orchestrator / Engine                |
|  - Prompt templating                                             |
|  - Tokenizer / detokenizer                                       |
|  - KV cache manager                                              |
|  - Scheduler (plans partitions, transfers, prefetch)             |
+-------------+----------------------------+----------------------+
              |                            |
              | Session calls              | Session calls
              v                            v
+-------------+-------------+    +---------+----------------------+
|   ORT Session: GPU Part   |    |     ORT Session: CPU Part     |
|  EP: TensorRT-RTX or CUDA |    |   EP: CPU (or OpenVINO, etc.) |
|  - hot layer blocks       |    |   - cold layer blocks          |
+-------------+-------------+    +---------+----------------------+
              |                            |
              | GPU tensors                | CPU tensors
              v                            v
+---------------------------+    +--------------------------------+
|     GPU Memory Manager    |    |       CPU Memory Manager       |
| - pinned staging buffers  |    | - RAM budget / paging          |
| - CUDA streams            |    | - (optional) disk backing      |
| - weight cache            |    | - weight + KV store            |
+---------------------------+    +--------------------------------+
```

## B) Dataflow per token (Simplified)

1. Tokenize user input -> `input_ids`
2. Prefill:
   - Run `gpu_part` prefill -> hidden + partial KV
   - Transfer hidden (if needed) -> CPU
   - Run `cpu_part` prefill -> logits + KV
3. Decode loop (for each token):
   - Prepare next step inputs (last token, masks, positions)
   - Run GPU partition decode -> hidden
   - Transfer hidden -> CPU partition
   - Run CPU partition decode -> logits
   - Sample next token, append to output
   - Stream text to UI

**Key optimization:** minimize transfers by keeping **coarse partitions** and using pinned buffers + async memcpy.

---

# Prototype Scheduler Design (Feed to Code Generation)

Below is a scheduler design you can hand to a codegen model. It’s written to be implementable in C# (or C++) with ONNX Runtime.

## 1) Scheduler Responsibilities
- Choose which partitions reside on GPU vs CPU given budgets (VRAM, RAM).
- Manage memory (weight residency, activation staging, KV cache placement).
- Overlap compute and transfers.
- Adapt partition plan over time (context grows, latency changes).

## 2) Core Types (Pseudo-Interfaces)

### Model metadata
```text
ModelSpec:
  - model_id
  - partitions: List[PartitionSpec]
  - tokenizer_path
  - prompt_template
  - quantization_info
  - kv_schema
```

### Partition metadata
```text
PartitionSpec:
  - name: "gpu_part_0" | "cpu_part_0" | ...
  - onnx_path
  - preferred_device: GPU | CPU
  - input_tensors: List[TensorSpec]
  - output_tensors: List[TensorSpec]
  - weight_bytes
  - temp_bytes_estimate
  - kv_bytes_per_token_estimate
  - supported_eps: [TensorRT, CUDA, CPU, ...]
  - profiling: PartitionProfile
```

### Profiling + cost model
```text
PartitionProfile:
  - gpu_ms_prefill
  - gpu_ms_decode
  - cpu_ms_prefill
  - cpu_ms_decode
  - transfer_bytes_in
  - transfer_bytes_out
  - transfer_ms_estimate  (PCIe)
  - hotness_score
```

### Runtime plan
```text
ExecutionPlan:
  - steps: List[PlanStep]
  - budgets: MemoryBudgets
  - device_assignments: {partition -> device}
  - staging_buffers: BufferPlan
  - streams: StreamPlan
```

### Plan step
```text
PlanStep:
  - partition: PartitionSpec
  - device: GPU|CPU
  - mode: PREFILL|DECODE
  - inputs: BindingPlan
  - outputs: BindingPlan
  - prefetch: PrefetchPlan (optional)
  - async_copy_before: CopyPlan (optional)
  - async_copy_after: CopyPlan (optional)
```

---

## 3) Planning Algorithm (Greedy + Safety Constraints)

Inputs:
- VRAM budget (e.g., 24GB minus safety margin)
- RAM budget
- desired max context length
- perf objective: maximize tokens/sec under memory constraints

Steps:
1. Compute `effective_gpu_budget = vram_total - reserve_for_kv(context_max) - reserve_for_workspace`
2. Sort partitions by `hotness_score` descending
3. Assign partitions to GPU until budget reached
4. Remaining partitions assigned to CPU
5. Validate:
   - required tensors can be transferred (interfaces match)
   - EP compatibility exists
6. Generate staging buffer plan:
   - pinned host buffer size = max(intermediate_tensor_bytes)
   - device buffers reused across steps
7. Generate streams:
   - one compute stream + one transfer stream (at minimum)
8. Save plan and begin execution

Optional refinement:
- Run a quick **micro-benchmark** (N tokens) to verify predicted tokens/sec; adjust cut points.

---

## 4) Runtime Loop and Overlap Strategy

### Prefill
- Execute GPU partition prefill
- Async copy hidden -> pinned host -> CPU
- Execute CPU partition prefill
- Produce initial logits and initialize KV caches

### Decode (per token)
- **Double-buffer** hidden activations:
  - buffer A used by GPU producing hidden_t
  - buffer B being transferred / consumed by CPU for hidden_{t-1}
- Overlap:
  - GPU compute for token t
  - CPU compute for token t-1 (if pipeline is structured)
  - async memcpy for token t hidden

If strict sequential dependency prevents full overlap, still overlap transfer with sampling and CPU prep.

---

## 5) KV Cache Policy

### Simple policy (works first)
- GPU partition KV on GPU
- CPU partition KV on CPU
- Hidden states transferred between partitions each step

### Advanced policy (later)
- Paged KV:
  - Keep a sliding window of KV on GPU, full KV on CPU
  - Prefetch needed pages for attention
This reduces VRAM pressure as context grows.

---

## 6) EP Selection Policy

For GPU partitions:
1. Prefer **TensorRT-RTX EP** if available (fastest on GeForce RTX for many ops)
2. Else **CUDA EP**
3. Else fallback (DirectML) only if required (may be slower than CUDA on NVIDIA)

For CPU partitions:
- CPU EP (baseline)
- Optional: OpenVINO EP on Intel, etc.

---

## 7) Observability & Auto-Tuning Hooks

Collect per token:
- gpu_part ms
- cpu_part ms
- transfer ms
- tokens/sec moving average
- VRAM usage (weights + KV + workspace)
- CPU RAM usage

Auto-tune triggers:
- if transfer dominates -> increase partition coarseness (fewer boundaries)
- if CPU dominates -> move additional hot blocks to GPU (if VRAM allows) or quantize CPU partitions more
- if VRAM OOM risk as context grows -> spill KV pages to CPU or shrink GPU partition set

---

# Suggested Implementation Milestones

1. **Milestone 1: Static 2-part partition**
   - one `gpu.onnx`, one `cpu.onnx`
   - sequential execution, correctness first

2. **Milestone 2: Pinned buffers + async copies**
   - reduce transfer overhead, add streaming output

3. **Milestone 3: Hotness-driven repartition**
   - implement profiler + greedy planner

4. **Milestone 4: KV paging**
   - scale context length without VRAM explosion

5. **Milestone 5: MoE-aware expert placement (if applicable)**
   - GPU stores popular experts, CPU holds the rest

---

# Notes for Codegen Models

When generating code:
- Use **separate ORT sessions** per partition with explicit EP selection.
- Use **I/O binding** if possible to avoid redundant copies.
- Use **pinned host memory** for staging transfers (C++ easiest; C# requires careful interop or supported APIs).
- Keep the runtime loop minimal and measurable first; add overlap later.
- Ensure prompts/tokenization match the model.

---

## Appendix: Why This Beats “Just Use EP Fallback”
EP fallback won’t help if the model can’t allocate weights/KV on GPU. You need **explicit placement** and **spillover policies**, which is exactly what partitioning + scheduling provides.


---

# References: Prior Art and Research Projects

This design is inspired by and synthesizes ideas from the following research systems and projects. These are **essential reading** for anyone implementing hybrid CPU/GPU inference beyond GPU VRAM limits.

## PowerInfer
**Title:** *PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU*  
**Key idea:** Exploits activation sparsity and power-law neuron usage to keep only “hot” neurons on GPU, while executing the rest on CPU.  
**Relevance:** Demonstrates that intelligent CPU/GPU orchestration can outperform naive GPU-only or static offload approaches when VRAM is constrained.

- Paper (arXiv): https://arxiv.org/abs/2312.12456  
- Project page: https://adsl-rg.github.io/PowerInfer/  
- Source code (GitHub): https://github.com/SJTU-IPADS/PowerInfer  

---

## FlexGen
**Title:** *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU*  
**Key idea:** Treats GPU, CPU, and disk as a unified memory hierarchy; uses optimization to decide where weights, activations, and KV cache reside.  
**Relevance:** Introduced the idea that **memory placement**, not raw FLOPs, is the bottleneck for large-model inference.

- Paper (MLSys 2023): https://arxiv.org/abs/2303.06865  
- Proceedings: https://proceedings.mlr.press/v202/sheng23a.html  
- Source code (GitHub): https://github.com/FMInference/FlexGen  

---

## Fiddler
**Title:** *Fiddler: CPU-GPU Co-Execution for Large Transformer Models*  
**Key idea:** Fine-grained co-execution of transformer layers across CPU and GPU, focusing on efficient scheduling and minimizing data transfer overhead.  
**Relevance:** Explores partitioning and execution strategies closer to operator- and block-level scheduling rather than coarse full-layer offload.

- Paper (arXiv): https://arxiv.org/abs/2402.07033  

(Note: Fiddler is primarily a research prototype; public source code availability may be limited.)

---

## FlexInfer
**Title:** *FlexInfer: Efficient Generative Inference of Large Language Models with Flexible Resource Management*  
**Key idea:** Asynchronous execution, weight prefetching, and flexible preservation of tensors to overlap CPU/GPU computation and data movement.  
**Relevance:** Shows how overlapping transfers with compute is critical to hiding PCIe latency in hybrid inference systems.

- Paper (EuroMLSys 2025): https://euromlsys.eu/pdf/euromlsys25-38.pdf  

---

## Additional Contextual References

### ONNX Runtime
- Execution Providers overview: https://onnxruntime.ai/docs/execution-providers/  
- High-level design: https://onnxruntime.ai/docs/reference/high-level-design.html  

### ONNX Runtime GenAI
- Repository: https://github.com/microsoft/onnxruntime-genai  
- Documentation: https://onnxruntime.ai/docs/genai/  

### Windows ML / TensorRT-RTX
- Windows ML announcement: https://devblogs.microsoft.com/windowsml/windows-machine-learning-is-generally-available/  
- NVIDIA TensorRT-RTX blog: https://developer.nvidia.com/blog/deploy-high-performance-ai-models-in-windows-apps-on-rtx-pcs/  

---

## Positioning Summary

- **PowerInfer** shows *what* to keep on GPU (hot neurons / high-impact compute).
- **FlexGen** shows *how* to think about memory as a hierarchy, not a device.
- **Fiddler** explores *where* to split execution for CPU/GPU co-execution.
- **FlexInfer** shows *when* and *how* to overlap computation and transfers.

The architecture in this document adapts these ideas into a form that is **practical with ONNX Runtime today**, without requiring a custom compiler or kernel stack.
