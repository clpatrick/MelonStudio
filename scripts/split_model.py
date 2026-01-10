#!/usr/bin/env python3
"""
ONNX Model Splitter for Hybrid CPU/GPU Inference

Takes an already-exported ONNX model (from onnxruntime-genai builder) and splits it
into GPU and CPU partitions at a specified layer boundary.

Usage:
    python split_onnx_model.py <model_path> --split-layer 16 --output-dir <output>
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import onnx
from onnx import helper, TensorProto


def analyze_model(model_path: str) -> dict:
    """Analyze the ONNX model structure to find layer boundaries."""
    print(f"Loading model from {model_path}...")
    # User confirmed high RAM, load fully into memory for reliability
    model = onnx.load(model_path, load_external_data=True)
    
    nodes = list(model.graph.node)
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    
    # Find all layer indices
    layer_indices = set()
    for node in nodes:
        if '/model/layers.' in node.name:
            # Extract layer number
            parts = node.name.split('/model/layers.')[1]
            layer_num = int(parts.split('/')[0])
            layer_indices.add(layer_num)
    
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    
    print(f"Model has {num_layers} layers, {len(nodes)} nodes")
    print(f"Inputs: {inputs[:3]}...")  # Show first 3
    print(f"Outputs: {outputs[:3]}...")  # Show first 3
    
    return {
        "num_layers": num_layers,
        "num_nodes": len(nodes),
        "inputs": inputs,
        "outputs": outputs,
    }


def find_split_tensors(model_path: str, split_layer: int) -> dict:
    """
    Find the tensor names at the split boundary.
    
    The split happens AFTER split_layer - 1 and BEFORE split_layer.
    So if split_layer=16, GPU runs layers 0-15, CPU runs layers 16-31.
    """
    # Load fully into memory
    model = onnx.load(model_path, load_external_data=True)
    nodes = list(model.graph.node)
    
    # Find the SkipLayerNorm node at the start of split_layer
    # This node's inputs are what we need to pass between partitions
    target_node_name = f"/model/layers.{split_layer}/input_layernorm/SkipLayerNorm"
    
    split_node = None
    for node in nodes:
        if node.name == target_node_name:
            split_node = node
            break
    
    if split_node is None:
        raise ValueError(f"Could not find node: {target_node_name}")
    
    # The SkipLayerNorm inputs are:
    # [0]: residual from post_attention_layernorm of previous layer
    # [1]: MLP output from previous layer
    # [2]: layernorm weight
    split_inputs = list(split_node.input)
    print(f"Split point inputs: {split_inputs[:2]}")  # First 2 are the data tensors
    
    return {
        "residual_tensor": split_inputs[0],  # From attention block
        "mlp_output_tensor": split_inputs[1],  # From MLP
        "split_node": target_node_name,
    }


def extract_gpu_partition(model_path: str, output_path: str, split_layer: int):
    """
    Extract the GPU partition: embedding + layers 0 to split_layer-1.
    
    Outputs the hidden state tensors that feed into split_layer.
    """
    print(f"\nExtracting GPU partition (layers 0-{split_layer-1})...")
    
    # Load fully into memory
    model = onnx.load(model_path, load_external_data=True)
    
    # Find the output tensors for the GPU partition
    split_info = find_split_tensors(model_path, split_layer)
    
    # For GPU partition, we want:
    # - All original inputs (input_ids, attention_mask, past_key_values.0-15.key/value)
    # - Outputs: the two tensors feeding into layer 16's SkipLayerNorm
    #            + present.0-15.key/value (KV cache for GPU layers)
    
    # Filter inputs: only include KV cache for layers 0 to split_layer-1
    gpu_inputs = []
    for inp in model.graph.input:
        name = inp.name
        if 'past_key_values' in name:
            # Extract layer number
            layer_num = int(name.split('.')[1])
            if layer_num < split_layer:
                gpu_inputs.append(name)
        else:
            gpu_inputs.append(name)
    
    # GPU outputs: hidden states + KV cache for GPU layers
    gpu_outputs = [
        split_info["residual_tensor"],
        split_info["mlp_output_tensor"],
    ]
    
    # Add KV cache outputs for GPU layers
    for outp in model.graph.output:
        name = outp.name
        if 'present.' in name:
            layer_num = int(name.split('.')[1])
            if layer_num < split_layer:
                gpu_outputs.append(name)
    
    print(f"GPU inputs: {len(gpu_inputs)} tensors")
    print(f"GPU outputs: {len(gpu_outputs)} tensors")
    
    # Use onnx.utils.extract_model
    onnx.utils.extract_model(
        model_path,
        output_path,
        input_names=gpu_inputs,
        output_names=gpu_outputs,
        check_model=False,
    )
    
    print(f"GPU partition saved to {output_path}")
    return {
        "inputs": gpu_inputs,
        "outputs": gpu_outputs,
    }


def extract_cpu_partition(model_path: str, output_path: str, split_layer: int, gpu_outputs: list):
    """
    Extract the CPU partition: layers split_layer to end + lm_head.
    
    Inputs: hidden states from GPU + KV cache for CPU layers
    Outputs: logits + KV cache for CPU layers
    """
    print(f"\nExtracting CPU partition (layers {split_layer}-end)...")
    
    # Load fully into memory
    model = onnx.load(model_path, load_external_data=True)
    split_info = find_split_tensors(model_path, split_layer)
    
    # CPU inputs: hidden states from GPU + KV cache for layers >= split_layer
    cpu_inputs = [
        split_info["residual_tensor"],
        split_info["mlp_output_tensor"],
    ]
    
    # Also need attention_mask and position info that flows through
    for inp in model.graph.input:
        name = inp.name
        if 'past_key_values' in name:
            layer_num = int(name.split('.')[1])
            if layer_num >= split_layer:
                cpu_inputs.append(name)
        elif name == 'attention_mask':
            cpu_inputs.append(name)
    
    # CPU outputs: logits + KV cache for CPU layers
    cpu_outputs = ['logits']
    for outp in model.graph.output:
        name = outp.name
        if 'present.' in name:
            layer_num = int(name.split('.')[1])
            if layer_num >= split_layer:
                cpu_outputs.append(name)
    
    print(f"CPU inputs: {len(cpu_inputs)} tensors")
    print(f"CPU outputs: {len(cpu_outputs)} tensors")
    
    # Use onnx.utils.extract_model
    onnx.utils.extract_model(
        model_path,
        output_path,
        input_names=cpu_inputs,
        output_names=cpu_outputs,
        check_model=False,
    )
    
    print(f"CPU partition saved to {output_path}")
    return {
        "inputs": cpu_inputs,
        "outputs": cpu_outputs,
    }


def create_hybrid_config(
    output_dir: Path,
    source_model: str,
    split_layer: int,
    num_layers: int,
    gpu_info: dict,
    cpu_info: dict,
):
    """Create the hybrid_config.json for the runtime."""
    config = {
        "version": "2.0",
        "source_model": source_model,
        "split_method": "onnx_extract",
        "total_layers": num_layers,
        "split_layer": split_layer,
        "gpu_partition": {
            "name": "gpu_part",
            "onnx_path": "gpu_part.onnx",
            "layer_start": 0,
            "layer_end": split_layer,
            "num_layers": split_layer,
            "preferred_ep": ["CUDAExecutionProvider", "TensorrtExecutionProvider"],
            "inputs": gpu_info["inputs"],
            "outputs": gpu_info["outputs"],
        },
        "cpu_partition": {
            "name": "cpu_part",
            "onnx_path": "cpu_part.onnx",
            "layer_start": split_layer,
            "layer_end": num_layers,
            "num_layers": num_layers - split_layer,
            "preferred_ep": ["CPUExecutionProvider"],
            "inputs": cpu_info["inputs"],
            "outputs": cpu_info["outputs"],
        },
        "interface_tensors": {
            "residual": gpu_info["outputs"][0],
            "mlp_output": gpu_info["outputs"][1],
        },
    }
    
    config_path = output_dir / "hybrid_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nHybrid config saved to {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Split an ONNX model into GPU and CPU partitions"
    )
    parser.add_argument("model_path", help="Path to the ONNX model file or directory")
    parser.add_argument("--split-layer", type=int, required=True,
                        help="Layer index to split at (GPU runs 0 to split-1, CPU runs split to end)")
    parser.add_argument("--output-dir", "-o", default="./hybrid_model",
                        help="Output directory for partitioned models")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze the model, don't split")
    
    args = parser.parse_args()
    
    # Resolve model path
    model_path = Path(args.model_path)
    if model_path.is_dir():
        model_path = model_path / "model.onnx"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1
    
    # Analyze model
    info = analyze_model(str(model_path))
    
    if args.analyze_only:
        print("\nAnalysis complete.")
        return 0
    
    # Validate split layer
    if args.split_layer <= 0 or args.split_layer >= info["num_layers"]:
        print(f"Error: split_layer must be between 1 and {info['num_layers'] - 1}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract partitions
    gpu_path = output_dir / "gpu_part.onnx"
    cpu_path = output_dir / "cpu_part.onnx"
    
    gpu_info = extract_gpu_partition(str(model_path), str(gpu_path), args.split_layer)
    cpu_info = extract_cpu_partition(str(model_path), str(cpu_path), args.split_layer, gpu_info["outputs"])
    
    # Create hybrid config
    create_hybrid_config(
        output_dir,
        str(model_path),
        args.split_layer,
        info["num_layers"],
        gpu_info,
        cpu_info,
    )
    
    # Copy tokenizer files from source
    source_dir = model_path.parent
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "genai_config.json", "chat_template.jinja"]
    for tf in tokenizer_files:
        src = source_dir / tf
        if src.exists():
            shutil.copy(src, output_dir / tf)
            print(f"Copied {tf}")
    
    print(f"\n{'='*60}")
    print("SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"GPU partition: layers 0-{args.split_layer - 1} ({args.split_layer} layers)")
    print(f"CPU partition: layers {args.split_layer}-{info['num_layers'] - 1} ({info['num_layers'] - args.split_layer} layers)")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
