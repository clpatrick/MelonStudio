#!/usr/bin/env python3
"""
Analyze safetensors/PyTorch model structure for conversion planning.

Extracts layer count, dimensions, precision, and size information from
HuggingFace model directories (safetensors or PyTorch format).
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from transformers import AutoConfig
except ImportError:
    print("ERROR: transformers library not found. Install with: pip install transformers", file=sys.stderr)
    sys.exit(1)


def analyze_safetensors_model(model_path):
    """Analyze a safetensors/PyTorch model directory."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load config.json
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found in {model_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get architecture info
    architectures = getattr(config, "architectures", [])
    architecture = architectures[0] if architectures else getattr(config, "model_type", "unknown")
    model_type = getattr(config, "model_type", "unknown")
    
    # Extract layer count (architecture-specific)
    num_layers = getattr(config, "num_hidden_layers", None) or \
                 getattr(config, "num_layers", None) or \
                 getattr(config, "n_layer", None) or \
                 getattr(config, "num_decoder_layers", None) or 0
    
    hidden_size = getattr(config, "hidden_size", None) or \
                  getattr(config, "n_embd", None) or \
                  getattr(config, "d_model", None) or 0
    
    num_heads = getattr(config, "num_attention_heads", None) or \
                getattr(config, "n_head", None) or \
                getattr(config, "num_heads", None) or 0
    
    intermediate_size = getattr(config, "intermediate_size", None) or \
                        getattr(config, "n_inner", None) or \
                        getattr(config, "ffn_dim", None) or (hidden_size * 4 if hidden_size > 0 else 0)
    
    vocab_size = getattr(config, "vocab_size", 32000)
    
    # Get torch_dtype from config or detect from files
    torch_dtype = getattr(config, "torch_dtype", None)
    precision_str = "FP16"  # Default
    
    if isinstance(torch_dtype, str):
        dtype_lower = torch_dtype.lower()
        if "bfloat16" in dtype_lower or "bf16" in dtype_lower:
            precision_str = "BF16"
        elif "float16" in dtype_lower or "fp16" in dtype_lower:
            precision_str = "FP16"
        elif "float32" in dtype_lower or "fp32" in dtype_lower:
            precision_str = "FP32"
        else:
            precision_str = torch_dtype.upper()
    elif torch_dtype is not None:
        # Handle torch dtype objects
        dtype_str = str(torch_dtype).lower()
        if "bfloat16" in dtype_str or "bf16" in dtype_str:
            precision_str = "BF16"
        elif "float16" in dtype_str or "fp16" in dtype_str:
            precision_str = "FP16"
        elif "float32" in dtype_str or "fp32" in dtype_str:
            precision_str = "FP32"
    
    # Also check model name/path for precision hints
    model_path_str = str(model_path).lower()
    if "bf16" in model_path_str or "bfloat16" in model_path_str:
        precision_str = "BF16"
    elif "fp16" in model_path_str or "float16" in model_path_str:
        precision_str = "FP16"
    
    # Calculate total size from weight files
    # Use index.json if available (more accurate for sharded models)
    # Otherwise, sum all safetensors/bin files directly
    total_size_bytes = 0
    safetensors_files = []
    pytorch_files = []
    processed_files = set()  # Track files to avoid double-counting
    
    # Check for index file first (preferred method for sharded models)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                # Get unique filenames from index
                unique_files = set(weight_map.values())
                for filename in unique_files:
                    file_path = model_path / filename
                    if file_path.exists():
                        size = file_path.stat().st_size
                        total_size_bytes += size
                        processed_files.add(filename)
                        if filename.endswith(".safetensors"):
                            safetensors_files.append(filename)
                        elif filename.endswith(".bin"):
                            pytorch_files.append(filename)
        except Exception as e:
            if os.getenv("VERBOSE"):
                print(f"WARNING: Could not read index file: {e}", file=sys.stderr)
    
    # Also count any safetensors/bin files not in the index (fallback)
    for file in model_path.glob("*.safetensors"):
        if file.name not in processed_files:
            size = file.stat().st_size
            total_size_bytes += size
            safetensors_files.append(str(file.name))
            processed_files.add(file.name)
    
    for file in model_path.glob("*.bin"):
        if file.name not in processed_files:
            size = file.stat().st_size
            total_size_bytes += size
            pytorch_files.append(str(file.name))
            processed_files.add(file.name)
    
    total_size_gb = total_size_bytes / (1024 ** 3)
    
    # Estimate base size (embeddings, norm layers, etc.) vs layer size
    # Rough estimate: base is ~10-15% of total, rest is layers
    base_size_gb = total_size_gb * 0.12  # ~12% for base
    avg_layer_size_gb = (total_size_gb - base_size_gb) / num_layers if num_layers > 0 else 0
    
    # Output in same format as analyze_onnx_size.py for compatibility
    print(f"Total Calculated Params Size: {total_size_gb:.4f} GB")
    print(f"Base Size: {base_size_gb:.4f} GB")
    print(f"Average Layer Size: {avg_layer_size_gb:.4f} GB")
    print(f"Layers Found: {num_layers}")
    print(f"Dominant Precision: {precision_str}")
    
    # Additional info for debugging
    if os.getenv("VERBOSE"):
        print(f"Architecture: {architecture}")
        print(f"Model Type: {model_type}")
        print(f"Hidden Size: {hidden_size}")
        print(f"Num Heads: {num_heads}")
        print(f"Intermediate Size: {intermediate_size}")
        print(f"Vocab Size: {vocab_size}")
        print(f"SafeTensors Files: {len(safetensors_files)}")
        print(f"PyTorch Files: {len(pytorch_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze safetensors/PyTorch model structure")
    parser.add_argument("model_path", help="Path to model directory (containing config.json)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    if args.verbose:
        os.environ["VERBOSE"] = "1"
    
    analyze_safetensors_model(args.model_path)
