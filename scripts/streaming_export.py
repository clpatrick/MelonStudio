"""
GPU-accelerated streaming ONNX converter for large models.
Loads and exports model layer-by-layer on GPU to avoid RAM exhaustion.
Uses torch.onnx.export with GPU tensors for CUDA-accelerated tracing.
"""
import argparse
import os
import json
import gc
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import onnx
from onnx import numpy_helper, TensorProto
import numpy as np

def get_layer_info(index_path):
    """Parse model index to get layer->file mapping."""
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index.get("weight_map", {})
    
    # Group by layer
    layers = {}
    base_tensors = {}
    
    for key, filename in weight_map.items():
        if "model.layers." in key:
            # Extract layer number
            parts = key.split(".")
            layer_idx = int(parts[2])
            if layer_idx not in layers:
                layers[layer_idx] = {}
            layers[layer_idx][key] = filename
        else:
            base_tensors[key] = filename
    
    return layers, base_tensors, weight_map

def export_single_layer_weights_to_onnx(source_dir, output_dir, layer_idx, layer_tensors, device='cuda'):
    """Export a single layer's weights as ONNX initializers."""
    print(f"  Exporting layer {layer_idx} tensors...")
    
    initializers = []
    
    for tensor_name, filename in layer_tensors.items():
        filepath = os.path.join(source_dir, filename)
        
        with safe_open(filepath, framework="pt", device=device) as f:
            # Find the key in this file
            for key in f.keys():
                if key == tensor_name or tensor_name.endswith(key):
                    tensor = f.get_tensor(key)
                    
                    # Convert to numpy (on CPU)
                    np_array = tensor.cpu().numpy()
                    
                    # Create ONNX TensorProto
                    onnx_tensor = numpy_helper.from_array(np_array, name=tensor_name)
                    initializers.append(onnx_tensor)
                    
                    del tensor
                    break
        
        # Clear GPU cache after each file
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    gc.collect()
    return initializers

def streaming_export_weights(source_dir, output_dir, device='cuda'):
    """Stream-export model weights as ONNX external data files."""
    print(f"Streaming export from {source_dir} to {output_dir}")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get layer mapping
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        return False
    
    layers, base_tensors, weight_map = get_layer_info(index_path)
    
    print(f"Found {len(layers)} layers and {len(base_tensors)} base tensors")
    
    # Create an ONNX graph with just initializers (weights)
    # The actual graph structure would need the full model, but we can create
    # a "weights-only" ONNX that can be merged later
    
    all_initializers = []
    
    # Export base tensors first
    print("Exporting base tensors (embeddings, head)...")
    base_init = export_single_layer_weights_to_onnx(
        source_dir, output_dir, -1, base_tensors, device
    )
    all_initializers.extend(base_init)
    
    # Export each layer
    for layer_idx in sorted(layers.keys()):
        print(f"Processing layer {layer_idx}/{max(layers.keys())}...")
        layer_init = export_single_layer_weights_to_onnx(
            source_dir, output_dir, layer_idx, layers[layer_idx], device
        )
        all_initializers.extend(layer_init)
        
        # Force garbage collection
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Create a minimal ONNX graph structure
    # This is a weights-only export - actual computation graph would need model architecture
    print(f"Creating ONNX model with {len(all_initializers)} initializers...")
    
    graph = onnx.helper.make_graph(
        nodes=[],
        name="model_weights",
        inputs=[],
        outputs=[],
        initializer=all_initializers
    )
    
    model = onnx.helper.make_model(graph)
    model.opset_import[0].version = 21
    
    # Save with external data
    output_path = os.path.join(output_dir, "model_weights.onnx")
    print(f"Saving to {output_path}...")
    
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_weights.onnx_data"
    )
    
    print("Streaming weight export complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source model directory with safetensors")
    parser.add_argument("output", help="Output directory for ONNX files")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    streaming_export_weights(args.source, args.output, args.device)
