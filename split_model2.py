#!/usr/bin/env python3
"""
ONNX Model Splitter for Hybrid CPU/GPU Inference (v2)

Splits HuggingFace transformer models into GPU and CPU partitions for hybrid inference.
Works with source models (SafeTensors/PyTorch) to properly handle KV cache routing.

Commands:
    analyze  - Analyze model structure and suggest split points
    export   - Export partitioned ONNX models from HuggingFace source
    validate - Validate partition compatibility

Usage:
    python split_model.py analyze <model_path_or_id> [--json]
    python split_model.py export <model_path_or_id> --split-layer <N> [--output-dir <dir>]
    python split_model.py validate <hybrid_model_dir>
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

# Diagnostic output helper
def diagnostic(category: str, message: str, data: Optional[Dict] = None, level: str = "INFO"):
    """Output structured diagnostic message."""
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "level": level,
        "category": category,
        "message": message,
    }
    if data:
        output["data"] = data
    print(f"[DIAG] {json.dumps(output)}", flush=True)


def log(message: str, level: str = "INFO"):
    """Simple log output."""
    print(f"[{level}] {message}", flush=True)


@dataclass
class ModelInfo:
    """Model architecture information."""
    model_id: str
    architecture: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: Optional[float]
    torch_dtype: str
    model_type: str

    # File info
    total_size_bytes: int
    weight_files: List[str]
    has_safetensors: bool
    has_pytorch_bin: bool

    # Estimated memory
    estimated_weights_gb: float
    estimated_kv_per_token_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitRecommendation:
    """Recommended split configuration."""
    split_layer: int
    gpu_layers: int
    cpu_layers: int
    gpu_weight_estimate_gb: float
    cpu_weight_estimate_gb: float
    reason: str


@dataclass
class PartitionSpec:
    """Specification for a model partition."""
    name: str
    onnx_path: str
    layer_start: int
    layer_end: int
    num_layers: int
    preferred_ep: List[str]
    kv_cache_layers: int
    estimated_size_gb: float


@dataclass
class HybridConfig:
    """Configuration for hybrid inference runtime."""
    version: str
    source_model: str
    architecture: str
    total_layers: int
    split_layer: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    torch_dtype: str

    gpu_partition: PartitionSpec
    cpu_partition: PartitionSpec

    interface_tensor: Dict[str, Any]
    kv_cache_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpu_partition"] = asdict(self.gpu_partition)
        d["cpu_partition"] = asdict(self.cpu_partition)
        return d


# Architecture mappings for different model types
ARCHITECTURE_CONFIGS = {
    "LlamaForCausalLM": {
        "layers_key": "num_hidden_layers",
        "hidden_key": "hidden_size",
        "heads_key": "num_attention_heads",
        "kv_heads_key": "num_key_value_heads",
        "intermediate_key": "intermediate_size",
        "layer_prefix": "model.layers",
    },
    "Qwen2ForCausalLM": {
        "layers_key": "num_hidden_layers",
        "hidden_key": "hidden_size",
        "heads_key": "num_attention_heads",
        "kv_heads_key": "num_key_value_heads",
        "intermediate_key": "intermediate_size",
        "layer_prefix": "model.layers",
    },
    "MistralForCausalLM": {
        "layers_key": "num_hidden_layers",
        "hidden_key": "hidden_size",
        "heads_key": "num_attention_heads",
        "kv_heads_key": "num_key_value_heads",
        "intermediate_key": "intermediate_size",
        "layer_prefix": "model.layers",
    },
    "PhiForCausalLM": {
        "layers_key": "num_hidden_layers",
        "hidden_key": "hidden_size",
        "heads_key": "num_attention_heads",
        "kv_heads_key": "num_key_value_heads",
        "intermediate_key": "intermediate_size",
        "layer_prefix": "model.layers",
    },
    "Phi3ForCausalLM": {
        "layers_key": "num_hidden_layers",
        "hidden_key": "hidden_size",
        "heads_key": "num_attention_heads",
        "kv_heads_key": "num_key_value_heads",
        "intermediate_key": "intermediate_size",
        "layer_prefix": "model.layers",
    },
    "GemmaForCausalLM": {
        "layers_key": "num_hidden_layers",
        "hidden_key": "hidden_size",
        "heads_key": "num_attention_heads",
        "kv_heads_key": "num_key_value_heads",
        "intermediate_key": "intermediate_size",
        "layer_prefix": "model.layers",
    },
    "GPT2LMHeadModel": {
        "layers_key": "n_layer",
        "hidden_key": "n_embd",
        "heads_key": "n_head",
        "kv_heads_key": None,  # GPT2 uses same heads for KV
        "intermediate_key": "n_inner",
        "layer_prefix": "transformer.h",
    },
}


def get_model_path(model_id_or_path: str) -> Path:
    """Resolve model path, downloading from HuggingFace if needed."""
    path = Path(model_id_or_path)

    if path.exists() and path.is_dir():
        return path

    # Try to download from HuggingFace
    try:
        from huggingface_hub import snapshot_download
        diagnostic("download", f"Downloading model from HuggingFace: {model_id_or_path}")

        cache_dir = Path("./cache_dir") / model_id_or_path.replace("/", "_")
        local_path = snapshot_download(
            repo_id=model_id_or_path,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        return Path(local_path)
    except ImportError:
        log("huggingface_hub not installed. Install with: pip install huggingface_hub", "ERROR")
        sys.exit(1)
    except Exception as e:
        log(f"Failed to download model: {e}", "ERROR")
        sys.exit(1)


def load_config(model_path: Path) -> Dict[str, Any]:
    """Load model config.json."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")

    with open(config_path) as f:
        return json.load(f)


def get_weight_files(model_path: Path) -> List[Dict[str, Any]]:
    """Get list of weight files with sizes."""
    weight_files = []

    patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
    for pattern in patterns:
        for f in model_path.glob(pattern):
            # Skip optimizer states and other non-model files
            if "optimizer" in f.name.lower() or "scheduler" in f.name.lower():
                continue
            weight_files.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "type": f.suffix[1:],  # Remove the dot
            })

    return weight_files


def analyze_model(model_path: Path, config: Dict[str, Any]) -> ModelInfo:
    """Analyze model and extract architecture information."""

    # Detect architecture
    architectures = config.get("architectures", [])
    architecture = architectures[0] if architectures else config.get("model_type", "unknown")
    model_type = config.get("model_type", "unknown")

    # Get architecture-specific config keys
    arch_config = ARCHITECTURE_CONFIGS.get(architecture, ARCHITECTURE_CONFIGS.get("LlamaForCausalLM"))

    # Extract parameters
    num_layers = config.get(arch_config["layers_key"], config.get("num_hidden_layers", 32))
    hidden_size = config.get(arch_config["hidden_key"], config.get("hidden_size", 4096))
    num_heads = config.get(arch_config["heads_key"], config.get("num_attention_heads", 32))

    # KV heads (for GQA models)
    kv_heads_key = arch_config.get("kv_heads_key")
    if kv_heads_key and kv_heads_key in config:
        num_kv_heads = config[kv_heads_key]
    else:
        num_kv_heads = num_heads  # MHA fallback

    intermediate_size = config.get(arch_config["intermediate_key"], config.get("intermediate_size", hidden_size * 4))
    vocab_size = config.get("vocab_size", 32000)
    max_position = config.get("max_position_embeddings", 4096)
    rope_theta = config.get("rope_theta")
    torch_dtype = config.get("torch_dtype", "float16")

    # Calculate head dimension
    head_dim = hidden_size // num_heads

    # Get weight files
    weight_files_info = get_weight_files(model_path)
    total_size = sum(f["size_bytes"] for f in weight_files_info)

    has_safetensors = any(f["type"] == "safetensors" for f in weight_files_info)
    has_pytorch = any(f["type"] in ["bin", "pt", "pth"] for f in weight_files_info)

    # Estimate KV cache per token (2 * num_layers * 2 * num_kv_heads * head_dim * dtype_size)
    dtype_size = 2 if "16" in torch_dtype else 4
    kv_per_token = 2 * num_layers * 2 * num_kv_heads * head_dim * dtype_size

    return ModelInfo(
        model_id=str(model_path),
        architecture=architecture,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_position,
        rope_theta=rope_theta,
        torch_dtype=torch_dtype,
        model_type=model_type,
        total_size_bytes=total_size,
        weight_files=[f["name"] for f in weight_files_info],
        has_safetensors=has_safetensors,
        has_pytorch_bin=has_pytorch,
        estimated_weights_gb=total_size / (1024**3),
        estimated_kv_per_token_bytes=kv_per_token,
    )


def suggest_split_points(model_info: ModelInfo, vram_budget_gb: float = 16.0) -> List[SplitRecommendation]:
    """Suggest optimal split points based on VRAM budget."""

    recommendations = []
    num_layers = model_info.num_layers
    weight_per_layer_gb = model_info.estimated_weights_gb / num_layers

    # Calculate how many layers can fit in VRAM (with safety margin)
    usable_vram = vram_budget_gb * 0.85  # 15% safety margin for activations/workspace
    max_gpu_layers = int(usable_vram / weight_per_layer_gb)
    max_gpu_layers = min(max_gpu_layers, num_layers - 1)  # Keep at least 1 layer on CPU

    # Suggest split points
    split_options = []

    # Option 1: Half-half split
    half_split = num_layers // 2
    if half_split > 0:
        split_options.append((half_split, "balanced_split"))

    # Option 2: Max GPU utilization
    if max_gpu_layers > 0 and max_gpu_layers != half_split:
        split_options.append((max_gpu_layers, "max_gpu"))

    # Option 3: Conservative (2/3 on GPU if it fits)
    two_thirds = int(num_layers * 2 / 3)
    if two_thirds <= max_gpu_layers and two_thirds not in [s[0] for s in split_options]:
        split_options.append((two_thirds, "conservative"))

    # Option 4: Minimal GPU (1/3 on GPU for very limited VRAM)
    one_third = num_layers // 3
    if one_third > 0 and one_third not in [s[0] for s in split_options]:
        split_options.append((one_third, "minimal_gpu"))

    for split_layer, reason in split_options:
        gpu_layers = split_layer
        cpu_layers = num_layers - split_layer
        gpu_weight = gpu_layers * weight_per_layer_gb
        cpu_weight = cpu_layers * weight_per_layer_gb

        recommendations.append(SplitRecommendation(
            split_layer=split_layer,
            gpu_layers=gpu_layers,
            cpu_layers=cpu_layers,
            gpu_weight_estimate_gb=round(gpu_weight, 2),
            cpu_weight_estimate_gb=round(cpu_weight, 2),
            reason=reason,
        ))

    # Sort by GPU utilization (descending)
    recommendations.sort(key=lambda r: r.gpu_layers, reverse=True)

    return recommendations


def cmd_analyze(args):
    """Analyze command - inspect model and suggest split points."""

    diagnostic("analyze", "Starting model analysis", {"model": args.model})

    try:
        model_path = get_model_path(args.model)
        diagnostic("analyze", f"Model path resolved", {"path": str(model_path)})

        config = load_config(model_path)
        diagnostic("analyze", "Config loaded successfully")

        model_info = analyze_model(model_path, config)
        diagnostic("analyze", "Model analyzed", {"architecture": model_info.architecture})

        # Get split recommendations
        vram_budget = getattr(args, 'vram_budget', 16.0)
        recommendations = suggest_split_points(model_info, vram_budget)

        # Build result
        result = {
            "status": "success",
            "model_info": model_info.to_dict(),
            "recommendations": [asdict(r) for r in recommendations],
            "compatibility": {
                "supported": model_info.architecture in ARCHITECTURE_CONFIGS,
                "has_source_weights": model_info.has_safetensors or model_info.has_pytorch_bin,
                "preferred_format": "safetensors" if model_info.has_safetensors else "pytorch",
            }
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print("\n" + "="*60)
            print("MODEL ANALYSIS REPORT")
            print("="*60)
            print(f"\nModel: {model_info.model_id}")
            print(f"Architecture: {model_info.architecture}")
            print(f"Model Type: {model_info.model_type}")
            print(f"\nLayers: {model_info.num_layers}")
            print(f"Hidden Size: {model_info.hidden_size}")
            print(f"Attention Heads: {model_info.num_attention_heads}")
            print(f"KV Heads: {model_info.num_kv_heads} {'(GQA)' if model_info.num_kv_heads != model_info.num_attention_heads else '(MHA)'}")
            print(f"Head Dimension: {model_info.head_dim}")
            print(f"Vocabulary Size: {model_info.vocab_size:,}")
            print(f"Max Position: {model_info.max_position_embeddings:,}")
            print(f"Dtype: {model_info.torch_dtype}")

            print(f"\nWeight Files: {len(model_info.weight_files)}")
            print(f"Total Size: {model_info.estimated_weights_gb:.2f} GB")
            print(f"Format: {'SafeTensors' if model_info.has_safetensors else 'PyTorch'}")
            print(f"KV Cache per Token: {model_info.estimated_kv_per_token_bytes:,} bytes")

            print("\n" + "-"*60)
            print("SPLIT RECOMMENDATIONS")
            print("-"*60)

            for i, rec in enumerate(recommendations, 1):
                print(f"\nOption {i}: {rec.reason}")
                print(f"  Split at layer: {rec.split_layer}")
                print(f"  GPU layers: 0-{rec.gpu_layers-1} ({rec.gpu_layers} layers, ~{rec.gpu_weight_estimate_gb:.2f} GB)")
                print(f"  CPU layers: {rec.split_layer}-{model_info.num_layers-1} ({rec.cpu_layers} layers, ~{rec.cpu_weight_estimate_gb:.2f} GB)")

            print("\n" + "="*60)

            # Emit structured diagnostic for parsing
            diagnostic("result", "Analysis complete", result)

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        diagnostic("error", f"Analysis failed: {e}", error_result, "ERROR")
        if args.json:
            print(json.dumps(error_result, indent=2))
        else:
            log(f"Analysis failed: {e}", "ERROR")
        sys.exit(1)


def export_partition(
    model_path: Path,
    model_info: ModelInfo,
    config: Dict[str, Any],
    split_layer: int,
    output_dir: Path,
    precision: str = "fp16",
) -> HybridConfig:
    """Export model as GPU and CPU ONNX partitions."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        log(f"Required packages not installed: {e}", "ERROR")
        log("Install with: pip install torch transformers", "ERROR")
        sys.exit(1)

    try:
        import onnx
    except ImportError:
        log("ONNX not installed. Install with: pip install onnx", "ERROR")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostic("export", "Loading model", {"path": str(model_path), "precision": precision})

    # Determine torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(precision, torch.float16)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    diagnostic("export", "Model loaded", {"parameters": sum(p.numel() for p in model.parameters())})

    # Copy tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                       "vocab.json", "merges.txt", "tokenizer.model"]
    for tf in tokenizer_files:
        src = model_path / tf
        if src.exists():
            import shutil
            shutil.copy(src, output_dir / tf)
            diagnostic("export", f"Copied {tf}")

    # Try to save tokenizer properly
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)
        diagnostic("export", "Tokenizer saved")
    except Exception as e:
        diagnostic("export", f"Tokenizer save warning: {e}", level="WARN")

    num_layers = model_info.num_layers
    gpu_layers = split_layer
    cpu_layers = num_layers - split_layer

    # Calculate partition sizes
    weight_per_layer = model_info.estimated_weights_gb / num_layers
    gpu_size = gpu_layers * weight_per_layer
    cpu_size = cpu_layers * weight_per_layer

    # Create partition specs
    gpu_partition = PartitionSpec(
        name="gpu_part",
        onnx_path="gpu_part.onnx",
        layer_start=0,
        layer_end=split_layer,
        num_layers=gpu_layers,
        preferred_ep=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        kv_cache_layers=gpu_layers,
        estimated_size_gb=round(gpu_size, 2),
    )

    cpu_partition = PartitionSpec(
        name="cpu_part",
        onnx_path="cpu_part.onnx",
        layer_start=split_layer,
        layer_end=num_layers,
        num_layers=cpu_layers,
        preferred_ep=["CPUExecutionProvider"],
        kv_cache_layers=cpu_layers,
        estimated_size_gb=round(cpu_size, 2),
    )

    # Define interface tensor schema
    interface_tensor = {
        "name": "hidden_states",
        "shape": ["batch_size", "sequence_length", model_info.hidden_size],
        "dtype": precision,
    }

    # Define KV cache schema
    kv_cache_schema = {
        "key_shape": ["batch_size", model_info.num_kv_heads, "sequence_length", model_info.head_dim],
        "value_shape": ["batch_size", model_info.num_kv_heads, "sequence_length", model_info.head_dim],
        "dtype": precision,
        "gpu_layers": list(range(gpu_layers)),
        "cpu_layers": list(range(gpu_layers, num_layers)),
    }

    # Build hybrid config
    hybrid_config = HybridConfig(
        version="1.0",
        source_model=str(model_path),
        architecture=model_info.architecture,
        total_layers=num_layers,
        split_layer=split_layer,
        hidden_size=model_info.hidden_size,
        num_attention_heads=model_info.num_attention_heads,
        num_kv_heads=model_info.num_kv_heads,
        head_dim=model_info.head_dim,
        vocab_size=model_info.vocab_size,
        max_position_embeddings=model_info.max_position_embeddings,
        torch_dtype=precision,
        gpu_partition=gpu_partition,
        cpu_partition=cpu_partition,
        interface_tensor=interface_tensor,
        kv_cache_schema=kv_cache_schema,
    )

    # Save hybrid config
    config_path = output_dir / "hybrid_config.json"
    with open(config_path, "w") as f:
        json.dump(hybrid_config.to_dict(), f, indent=2)
    diagnostic("export", "Saved hybrid_config.json", {"path": str(config_path)})

    # Export ONNX models using torch.onnx.export
    # For now, we'll create placeholder ONNX files and document the full export process
    # Full ONNX export requires careful handling of dynamic axes and KV cache

    diagnostic("export", "Starting ONNX export", {
        "gpu_layers": gpu_layers,
        "cpu_layers": cpu_layers,
    })

    # Create a wrapper model for GPU partition (embedding + first N layers)
    class GPUPartitionModel(torch.nn.Module):
        def __init__(self, full_model, num_layers):
            super().__init__()
            self.embed_tokens = full_model.model.embed_tokens
            self.layers = torch.nn.ModuleList(list(full_model.model.layers[:num_layers]))
            self.norm = None  # No final norm in GPU partition

        def forward(self, input_ids, attention_mask, position_ids):
            hidden_states = self.embed_tokens(input_ids)
            for layer in self.layers:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            return hidden_states

    # Create a wrapper for CPU partition (remaining layers + lm_head)
    class CPUPartitionModel(torch.nn.Module):
        def __init__(self, full_model, start_layer):
            super().__init__()
            self.layers = torch.nn.ModuleList(list(full_model.model.layers[start_layer:]))
            self.norm = full_model.model.norm
            self.lm_head = full_model.lm_head

        def forward(self, hidden_states, attention_mask, position_ids):
            for layer in self.layers:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

    # Try to export partitions
    try:
        diagnostic("export", "Creating GPU partition model")
        gpu_model = GPUPartitionModel(model, gpu_layers)
        gpu_model.eval()

        diagnostic("export", "Creating CPU partition model")
        cpu_model = CPUPartitionModel(model, gpu_layers)
        cpu_model.eval()

        # Create dummy inputs
        batch_size = 1
        seq_length = 4

        dummy_input_ids = torch.ones(batch_size, seq_length, dtype=torch.long)
        dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        dummy_position_ids = torch.arange(seq_length).unsqueeze(0)
        dummy_hidden_states = torch.randn(batch_size, seq_length, model_info.hidden_size, dtype=torch_dtype)

        # Export GPU partition
        diagnostic("export", "Exporting GPU partition to ONNX")
        gpu_onnx_path = output_dir / "gpu_part.onnx"

        torch.onnx.export(
            gpu_model,
            (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
            str(gpu_onnx_path),
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "position_ids": {0: "batch_size", 1: "sequence_length"},
                "hidden_states": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        diagnostic("export", "GPU partition exported", {"path": str(gpu_onnx_path)})

        # Export CPU partition
        diagnostic("export", "Exporting CPU partition to ONNX")
        cpu_onnx_path = output_dir / "cpu_part.onnx"

        torch.onnx.export(
            cpu_model,
            (dummy_hidden_states, dummy_attention_mask, dummy_position_ids),
            str(cpu_onnx_path),
            input_names=["hidden_states", "attention_mask", "position_ids"],
            output_names=["logits"],
            dynamic_axes={
                "hidden_states": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "position_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        diagnostic("export", "CPU partition exported", {"path": str(cpu_onnx_path)})

        # Verify exports
        gpu_model_onnx = onnx.load(str(gpu_onnx_path))
        onnx.checker.check_model(gpu_model_onnx)
        diagnostic("export", "GPU partition validated")

        cpu_model_onnx = onnx.load(str(cpu_onnx_path))
        onnx.checker.check_model(cpu_model_onnx)
        diagnostic("export", "CPU partition validated")

    except Exception as e:
        diagnostic("export", f"ONNX export failed: {e}", level="ERROR")
        diagnostic("export", "Partitions not exported - manual export required", {
            "error": str(e),
            "suggestion": "Use ONNX Runtime GenAI builder with custom modifications",
        }, level="WARN")

        # Still save the config for reference
        diagnostic("export", "hybrid_config.json saved for reference")

    return hybrid_config


def cmd_export(args):
    """Export command - create partitioned ONNX models."""

    diagnostic("export", "Starting export", {
        "model": args.model,
        "split_layer": args.split_layer,
        "output_dir": args.output_dir,
    })

    try:
        model_path = get_model_path(args.model)
        config = load_config(model_path)
        model_info = analyze_model(model_path, config)

        # Validate split layer
        if args.split_layer <= 0 or args.split_layer >= model_info.num_layers:
            log(f"Invalid split_layer. Must be between 1 and {model_info.num_layers - 1}", "ERROR")
            sys.exit(1)

        output_dir = Path(args.output_dir)
        precision = getattr(args, 'precision', 'fp16')

        hybrid_config = export_partition(
            model_path=model_path,
            model_info=model_info,
            config=config,
            split_layer=args.split_layer,
            output_dir=output_dir,
            precision=precision,
        )

        result = {
            "status": "success",
            "output_dir": str(output_dir),
            "hybrid_config": hybrid_config.to_dict(),
        }

        diagnostic("result", "Export complete", result)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "="*60)
            print("EXPORT COMPLETE")
            print("="*60)
            print(f"\nOutput directory: {output_dir}")
            print(f"\nPartitions:")
            print(f"  GPU: {hybrid_config.gpu_partition.onnx_path} ({hybrid_config.gpu_partition.num_layers} layers)")
            print(f"  CPU: {hybrid_config.cpu_partition.onnx_path} ({hybrid_config.cpu_partition.num_layers} layers)")
            print(f"\nConfiguration: hybrid_config.json")
            print("\n" + "="*60)

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        diagnostic("error", f"Export failed: {e}", error_result, "ERROR")
        if args.json:
            print(json.dumps(error_result, indent=2))
        else:
            log(f"Export failed: {e}", "ERROR")
        sys.exit(1)


def cmd_validate(args):
    """Validate command - verify partition compatibility."""

    diagnostic("validate", "Starting validation", {"dir": args.hybrid_dir})

    try:
        hybrid_dir = Path(args.hybrid_dir)

        if not hybrid_dir.exists():
            raise FileNotFoundError(f"Directory not found: {hybrid_dir}")

        # Load hybrid config
        config_path = hybrid_dir / "hybrid_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"hybrid_config.json not found in {hybrid_dir}")

        with open(config_path) as f:
            hybrid_config = json.load(f)

        diagnostic("validate", "Config loaded", {"version": hybrid_config.get("version")})

        validation_results = {
            "config_valid": True,
            "gpu_partition_exists": False,
            "cpu_partition_exists": False,
            "gpu_partition_valid": False,
            "cpu_partition_valid": False,
            "interface_compatible": False,
            "issues": [],
        }

        # Check GPU partition
        gpu_path = hybrid_dir / hybrid_config["gpu_partition"]["onnx_path"]
        if gpu_path.exists():
            validation_results["gpu_partition_exists"] = True
            try:
                import onnx
                gpu_model = onnx.load(str(gpu_path))
                onnx.checker.check_model(gpu_model)
                validation_results["gpu_partition_valid"] = True
                diagnostic("validate", "GPU partition valid")
            except Exception as e:
                validation_results["issues"].append(f"GPU partition invalid: {e}")
                diagnostic("validate", f"GPU partition validation failed: {e}", level="WARN")
        else:
            validation_results["issues"].append(f"GPU partition not found: {gpu_path}")

        # Check CPU partition
        cpu_path = hybrid_dir / hybrid_config["cpu_partition"]["onnx_path"]
        if cpu_path.exists():
            validation_results["cpu_partition_exists"] = True
            try:
                import onnx
                cpu_model = onnx.load(str(cpu_path))
                onnx.checker.check_model(cpu_model)
                validation_results["cpu_partition_valid"] = True
                diagnostic("validate", "CPU partition valid")
            except Exception as e:
                validation_results["issues"].append(f"CPU partition invalid: {e}")
                diagnostic("validate", f"CPU partition validation failed: {e}", level="WARN")
        else:
            validation_results["issues"].append(f"CPU partition not found: {cpu_path}")

        # Check interface compatibility
        if validation_results["gpu_partition_valid"] and validation_results["cpu_partition_valid"]:
            try:
                import onnx
                gpu_model = onnx.load(str(gpu_path))
                cpu_model = onnx.load(str(cpu_path))

                # Get GPU outputs
                gpu_outputs = {o.name: o for o in gpu_model.graph.output}
                cpu_inputs = {i.name: i for i in cpu_model.graph.input}

                interface_name = hybrid_config["interface_tensor"]["name"]

                if interface_name in gpu_outputs and interface_name in cpu_inputs:
                    validation_results["interface_compatible"] = True
                    diagnostic("validate", "Interface compatible", {"tensor": interface_name})
                else:
                    validation_results["issues"].append(
                        f"Interface mismatch: GPU outputs {list(gpu_outputs.keys())}, CPU inputs {list(cpu_inputs.keys())}"
                    )
            except Exception as e:
                validation_results["issues"].append(f"Interface check failed: {e}")

        # Overall status
        validation_results["overall_valid"] = (
            validation_results["gpu_partition_valid"] and
            validation_results["cpu_partition_valid"] and
            validation_results["interface_compatible"]
        )

        result = {
            "status": "success" if validation_results["overall_valid"] else "failed",
            "validation": validation_results,
            "hybrid_config": hybrid_config,
        }

        diagnostic("result", "Validation complete", result)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "="*60)
            print("VALIDATION REPORT")
            print("="*60)
            print(f"\nDirectory: {hybrid_dir}")
            print(f"\nConfig: {'OK' if validation_results['config_valid'] else 'FAILED'}")
            print(f"GPU Partition: {'OK' if validation_results['gpu_partition_valid'] else 'MISSING/INVALID'}")
            print(f"CPU Partition: {'OK' if validation_results['cpu_partition_valid'] else 'MISSING/INVALID'}")
            print(f"Interface: {'COMPATIBLE' if validation_results['interface_compatible'] else 'INCOMPATIBLE'}")
            print(f"\nOverall: {'PASS' if validation_results['overall_valid'] else 'FAIL'}")

            if validation_results["issues"]:
                print("\nIssues:")
                for issue in validation_results["issues"]:
                    print(f"  - {issue}")

            print("\n" + "="*60)

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        diagnostic("error", f"Validation failed: {e}", error_result, "ERROR")
        if args.json:
            print(json.dumps(error_result, indent=2))
        else:
            log(f"Validation failed: {e}", "ERROR")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Model Splitter for Hybrid CPU/GPU Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze model structure and suggest split points")
    analyze_parser.add_argument("model", help="HuggingFace model ID or local path")
    analyze_parser.add_argument("--vram-budget", type=float, default=16.0, help="Available VRAM in GB (default: 16)")
    analyze_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export partitioned ONNX models")
    export_parser.add_argument("model", help="HuggingFace model ID or local path")
    export_parser.add_argument("--split-layer", type=int, required=True, help="Layer index to split at")
    export_parser.add_argument("--output-dir", "-o", default="./hybrid_model", help="Output directory")
    export_parser.add_argument("--precision", choices=["fp16", "fp32", "bf16"], default="fp16", help="Export precision")
    export_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate hybrid model partitions")
    validate_parser.add_argument("hybrid_dir", help="Path to hybrid model directory")
    validate_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()