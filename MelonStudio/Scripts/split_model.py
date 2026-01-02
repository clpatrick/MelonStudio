#!/usr/bin/env python3
"""
ONNX Model Splitter for Hybrid CPU/GPU Inference

Splits a transformer model at a specified layer boundary, creating:
- gpu_part.onnx: Layers 0 to split_layer-1 (runs on GPU)
- cpu_part.onnx: Layers split_layer to N (runs on CPU)

Usage:
    python split_model.py <model_path> <split_layer> [--output-dir <dir>]
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Error: onnx package required. Install with: pip install onnx")
    sys.exit(1)


def find_layer_boundaries(model: onnx.ModelProto) -> list[tuple[str, str, int]]:
    """
    Find layer boundaries in a transformer model.
    Returns list of (layer_name, output_tensor_name, node_index)
    """
    boundaries = []
    graph = model.graph
    
    # Common patterns for layer boundaries in transformer models
    layer_patterns = [
        "layer_norm",
        "layernorm", 
        "ln_",
        "post_attention",
        "block_output",
        "hidden_states",
        "decoder.layers.",
        "model.layers.",
        "transformer.h.",
    ]
    
    for i, node in enumerate(graph.node):
        node_name = node.name.lower() if node.name else ""
        output_name = node.output[0].lower() if node.output else ""
        
        # Check if this looks like a layer boundary
        for pattern in layer_patterns:
            if pattern in node_name or pattern in output_name:
                # Get the output tensor
                if node.output:
                    boundaries.append((node.name, node.output[0], i))
                break
    
    return boundaries


def analyze_model(model_path: str) -> dict:
    """
    Analyze ONNX model structure and find splittable layers.
    """
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    
    info = {
        "model_path": model_path,
        "inputs": [(inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim]) 
                   for inp in model.graph.input],
        "outputs": [(out.name, [d.dim_value for d in out.type.tensor_type.shape.dim]) 
                    for out in model.graph.output],
        "total_nodes": len(model.graph.node),
        "total_initializers": len(model.graph.initializer),
    }
    
    # Find layer boundaries
    boundaries = find_layer_boundaries(model)
    info["layer_boundaries"] = [(name, tensor, idx) for name, tensor, idx in boundaries]
    info["num_layers"] = len(boundaries)
    
    # Estimate model size
    total_bytes = sum(init.ByteSize() for init in model.graph.initializer)
    info["model_size_gb"] = total_bytes / (1024**3)
    
    return info


def split_model(model_path: str, split_layer: int, output_dir: str):
    """
    Split model at specified layer boundary.
    
    Args:
        model_path: Path to ONNX model
        split_layer: Layer index to split at (0-indexed, layers before this go to GPU)
        output_dir: Output directory for split models
    """
    print(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    
    # Find layer boundaries
    boundaries = find_layer_boundaries(model)
    
    if split_layer < 0 or split_layer >= len(boundaries):
        print(f"Error: split_layer must be between 0 and {len(boundaries)-1}")
        sys.exit(1)
    
    # Get the tensor name at the split point
    split_name, split_tensor, split_node_idx = boundaries[split_layer]
    print(f"Splitting at layer {split_layer}: {split_name}")
    print(f"Split tensor: {split_tensor}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU partition: from inputs to split point
    gpu_output_path = os.path.join(output_dir, "gpu_part.onnx")
    print(f"\nExtracting GPU partition (layers 0-{split_layer-1})...")
    
    try:
        # Get input names
        input_names = [inp.name for inp in model.graph.input]
        
        # Extract GPU subgraph
        onnx.utils.extract_model(
            model_path,
            gpu_output_path,
            input_names=input_names,
            output_names=[split_tensor]
        )
        print(f"Saved GPU partition: {gpu_output_path}")
        
        # Verify GPU partition
        gpu_model = onnx.load(gpu_output_path)
        onnx.checker.check_model(gpu_model)
        print(f"  - Nodes: {len(gpu_model.graph.node)}")
        
    except Exception as e:
        print(f"Warning: GPU extraction failed: {e}")
        print("Attempting alternative extraction method...")
        # Fallback: manual node extraction
        extract_nodes_range(model, 0, split_node_idx, gpu_output_path)
    
    # CPU partition: from split point to outputs
    cpu_output_path = os.path.join(output_dir, "cpu_part.onnx")
    print(f"\nExtracting CPU partition (layers {split_layer}-{len(boundaries)-1})...")
    
    try:
        # Get output names
        output_names = [out.name for out in model.graph.output]
        
        # Extract CPU subgraph
        onnx.utils.extract_model(
            model_path,
            cpu_output_path,
            input_names=[split_tensor],
            output_names=output_names
        )
        print(f"Saved CPU partition: {cpu_output_path}")
        
        # Verify CPU partition
        cpu_model = onnx.load(cpu_output_path)
        onnx.checker.check_model(cpu_model)
        print(f"  - Nodes: {len(cpu_model.graph.node)}")
        
    except Exception as e:
        print(f"Warning: CPU extraction failed: {e}")
        print("Attempting alternative extraction method...")
        extract_nodes_range(model, split_node_idx, len(model.graph.node), cpu_output_path)
    
    # Save partition metadata
    metadata = {
        "original_model": model_path,
        "split_layer": split_layer,
        "split_tensor": split_tensor,
        "gpu_partition": "gpu_part.onnx",
        "cpu_partition": "cpu_part.onnx",
        "total_layers": len(boundaries),
        "gpu_layers": split_layer,
        "cpu_layers": len(boundaries) - split_layer,
    }
    
    metadata_path = os.path.join(output_dir, "partition_info.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved partition metadata: {metadata_path}")
    
    print("\n✓ Model split complete!")
    print(f"  GPU partition: {gpu_output_path}")
    print(f"  CPU partition: {cpu_output_path}")
    
    return metadata


def extract_nodes_range(model: onnx.ModelProto, start_idx: int, end_idx: int, output_path: str):
    """
    Fallback: manually extract a range of nodes.
    This is a simplified implementation that may not handle all cases.
    """
    graph = model.graph
    
    # Get nodes in range
    nodes = list(graph.node)[start_idx:end_idx]
    
    # Find required initializers
    initializer_names = set()
    for node in nodes:
        for inp in node.input:
            initializer_names.add(inp)
    
    initializers = [init for init in graph.initializer if init.name in initializer_names]
    
    # Find inputs (tensors not produced by nodes in range)
    produced = set()
    for node in nodes:
        produced.update(node.output)
    
    input_names = set()
    for node in nodes:
        for inp in node.input:
            if inp not in produced and inp not in {init.name for init in initializers}:
                input_names.add(inp)
    
    # Create input value infos (simplified - may need type info)
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in input_names]
    
    # Get outputs
    output_names = list(nodes[-1].output) if nodes else []
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in output_names]
    
    # Create new graph
    new_graph = helper.make_graph(
        nodes=nodes,
        name="subgraph",
        inputs=inputs,
        outputs=outputs,
        initializer=initializers
    )
    
    # Create new model
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    
    onnx.save(new_model, output_path)
    print(f"Saved (fallback method): {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Split ONNX transformer model for hybrid inference")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze model structure")
    analyze_parser.add_argument("model_path", help="Path to ONNX model")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split model at layer boundary")
    split_parser.add_argument("model_path", help="Path to ONNX model")
    split_parser.add_argument("split_layer", type=int, help="Layer index to split at")
    split_parser.add_argument("--output-dir", "-o", default="./hybrid_model", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        info = analyze_model(args.model_path)
        print("\n=== Model Analysis ===")
        print(f"Model: {info['model_path']}")
        print(f"Size: {info['model_size_gb']:.2f} GB")
        print(f"Nodes: {info['total_nodes']}")
        print(f"Detected layers: {info['num_layers']}")
        print("\nInputs:")
        for name, shape in info['inputs']:
            print(f"  {name}: {shape}")
        print("\nOutputs:")
        for name, shape in info['outputs']:
            print(f"  {name}: {shape}")
        print("\nLayer boundaries (first 10):")
        for name, tensor, idx in info['layer_boundaries'][:10]:
            print(f"  [{idx}] {name} -> {tensor}")
        if info['num_layers'] > 10:
            print(f"  ... and {info['num_layers'] - 10} more")
            
    elif args.command == "split":
        split_model(args.model_path, args.split_layer, args.output_dir)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
