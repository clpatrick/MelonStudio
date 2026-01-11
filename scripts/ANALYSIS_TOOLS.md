# Model Analysis Tools Available

## Current Analysis Capabilities

### 1. **Basic Model Analysis** (`analyze_safetensors.py` / `analyze_onnx_size.py`)
- ✅ Layer count extraction
- ✅ Size calculation (total, base, per-layer)
- ✅ Precision detection (FP32, FP16, BF16, INT8, INT4)
- ✅ Architecture detection
- ✅ Basic dimension extraction (hidden size, heads, etc.)

### 2. **ONNX Model Inspection** (`split_model.py` / `analyze_onnx_size.py`)
- ✅ ONNX graph structure analysis
- ✅ Layer boundary detection
- ✅ Node counting and topology
- ✅ Input/output tensor inspection

### 3. **HuggingFace Model Analysis** (`split_model_v2.py analyze`)
- ✅ Architecture-specific config parsing
- ✅ Layer count from config
- ✅ KV cache estimation
- ✅ Split point recommendations
- ✅ VRAM budget calculations

---

## Available Tools in Olive Framework

### Olive Model Handlers
- **`HfModelHandler`**: Loads HuggingFace models (supports safetensors, PyTorch)
  - Can load config.json
  - Supports custom code execution (`trust_remote_code=True`)
  - Validates model structure

### Olive Analysis Capabilities
1. **Model Structure Inspection**
   - Load model without full weights (config-only)
   - Extract architecture information
   - Validate model compatibility

2. **Quantization Analysis**
   - Identify quantizable layers
   - Estimate quantization impact
   - Precision compatibility checking

3. **Optimization Passes**
   - Graph optimization analysis
   - Operator fusion opportunities
   - Memory layout optimization

---

## ONNX Runtime Analysis Tools

### 1. **ONNX Model Inspector** (via `onnx` Python package)
```python
import onnx
model = onnx.load("model.onnx")
# Inspect:
# - model.graph.node (all operations)
# - model.graph.input/output (tensor shapes)
# - model.graph.initializer (weights)
```

### 2. **ONNX Runtime Profiler**
- Can profile execution time per operator
- Memory usage per operator
- EP (Execution Provider) utilization

### 3. **ONNX Shape Inference**
```python
from onnx import shape_inference
inferred_model = shape_inference.infer_shapes(model)
# Get tensor shapes throughout the graph
```

### 4. **ONNX Runtime GenAI Builder**
- Can analyze model structure before export
- Identifies supported operations
- Estimates memory requirements

---

## Recommended Additional Analysis

### For Complex Architectures (MoE, Mamba-2, GQA)

1. **Layer Type Analysis**
   - Identify MoE layers (23 layers)
   - Identify Mamba-2 layers (23 layers)
   - Identify GQA layers (6 layers)
   - Map layer types to indices

2. **Expert Analysis** (for MoE)
   - Count experts per MoE layer (128 routed + 1 shared = 129 total)
   - Identify which experts are activated per token (6 active)
   - Estimate MoE layer memory requirements
   - Calculate expert routing overhead

3. **Mamba-2 Analysis**
   - State space model parameters
   - Selective scan operations
   - Memory-efficient attention patterns

4. **GQA Analysis**
   - Grouped query attention structure
   - KV head groups (2 groups mentioned)
   - Attention pattern differences from standard MHA

5. **Memory Footprint Analysis**
   - Per-layer weight size
   - Activation memory per layer type
   - KV cache requirements per layer type
   - Expert weights memory (MoE)

6. **Conversion Planning Analysis**
   - Identify conversion challenges:
     - MoE routing logic (needs custom handling)
     - Mamba-2 state space ops (may need custom ONNX ops)
     - GQA attention (standard but with grouped KV)
   - Estimate conversion time
   - Identify quantization opportunities

7. **Hybrid Partition Recommendations**
   - Which layers benefit most from GPU (MoE? Mamba-2? GQA?)
   - Optimal split points considering:
     - Layer type distribution
     - Memory requirements
     - Compute intensity
   - VRAM budget planning

---

## Implementation Suggestions

### Enhanced Analysis Script
Create `analyze_model_detailed.py` that:
1. Loads config.json with `transformers.AutoConfig`
2. Parses architecture-specific details:
   - MoE configuration (experts, routing)
   - Mamba-2 configuration
   - GQA configuration
3. Analyzes weight files to:
   - Map weights to layer types
   - Calculate per-layer-type sizes
   - Identify expert weights
4. Outputs structured JSON with:
   - Layer type map (index -> type)
   - Per-type statistics
   - Conversion recommendations
   - Memory breakdown

### Olive Integration
- Use `HfModelHandler` to load model structure
- Leverage Olive's quantization analyzers
- Use Olive's optimization pass analysis

### ONNX Runtime Tools
- Use `onnx.shape_inference` for tensor shape analysis
- Profile with ONNX Runtime to identify hot paths
- Use ONNX Runtime GenAI builder's analysis capabilities

---

## Next Steps

1. **Fix current analysis** (precision, size calculation) ✅
2. **Create detailed architecture parser** for MoE/Mamba-2/GQA
3. **Add layer type mapping** to analysis output
4. **Integrate Olive analysis** for conversion planning
5. **Add conversion compatibility checker** for complex ops
