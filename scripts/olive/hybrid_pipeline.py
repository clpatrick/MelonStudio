import argparse
import os
import json
import shutil
import glob
import onnx
import re
from onnx import helper, shape_inference
import olive.workflows

def find_split_tensor(model, split_layer):
    """
    Finds the tensor name that connects layer[split_layer-1] to layer[split_layer].
    """
    print(f"Searching for split point at layer {split_layer}...")
    
    # Regex to capture layer index
    # Standard Llama: /model/layers/0/... or model.layers.0...
    # Optimized might be: /model/layers_0/...
    
    # We look for inputs to the first node of layer {split_layer}
    # But finding "First Node" is hard in a graph.
    
    # Alternative: Look for the output of layer {split_layer-1}.
    # The output of a layer usually feeds into the next layer.
    
    # Let's map nodes to layers
    layer_pattern = re.compile(r"(?:layers|layer|block|h)[./_](\d+)[./_]")
    
    layer_nodes = {} # layer_idx -> list of nodes
    
    for node in model.graph.node:
        match = layer_pattern.search(node.name)
        if match:
            idx = int(match.group(1))
            if idx not in layer_nodes:
                layer_nodes[idx] = []
            layer_nodes[idx].append(node)
            
    if split_layer not in layer_nodes:
        raise ValueError(f"Could not find nodes for layer {split_layer}. Max layer found: {max(layer_nodes.keys()) if layer_nodes else 'None'}")
        
    # Get all inputs of split_layer
    split_layer_inputs = set()
    for node in layer_nodes[split_layer]:
        for inp in node.input:
            split_layer_inputs.add(inp)
            
    # Get all outputs of split_layer-1 (or previous layers)
    # Actually, we want the tensor that flows FROM prev layers INTO split_layer.
    # Usually strictly FROM split_layer-1.
    
    prev_layer = split_layer - 1
    if prev_layer < 0:
        # Split at 0?
        # Then input is embeddings?
        # We assume split_layer > 0 for now.
        raise ValueError("Split layer must be > 0")
        
    prev_layer_outputs = set()
    if prev_layer in layer_nodes:
        for node in layer_nodes[prev_layer]:
            for out in node.output:
                prev_layer_outputs.add(out)
                
    # Intersection: Output of Prev that is Input of Current
    # This is the "Edge" connecting them.
    # Note: Residual connections might skip?
    # In Transformer, the "State" is passed.
    
    candidates = prev_layer_outputs.intersection(split_layer_inputs)
    
    if not candidates:
        print("Warning: No direct connection found between layer N-1 and N. Checking for residual connection skipping...")
        # Search for inputs to N that come from ANY node < N?
        # Simpler: Just print candidates if any
        pass
        
    # In optimized graph, names might be messy.
    # Fallback to Topological Sort?
    # No, let's assume one main tensor.
    
    if len(candidates) == 1:
        return list(candidates)[0]
    
    print(f"Candidates for split tensor: {candidates}")
    # Heuristic: The tensor that is also an output of a LayerNorm or Add node?
    # Or just return the first one?
    # In Llama, it's usually the residual stream.
    return list(candidates)[0] 

def split_onnx_model(onnx_path, output_dir, split_layer):
    print(f"Splitting {onnx_path} at layer {split_layer}...")
    model = onnx.load(onnx_path, load_external_data=False) # Metadata only first
    
    # 1. Identify Split Tensor
    # We need to load properly to traverse? 
    # load_external_data=False is fine for Graph inspection.
    split_tensor = find_split_tensor(model, split_layer)
    print(f"Identified split tensor: {split_tensor}")
    
    # 2. Extract GPU Part (Inputs -> Split Tensor)
    # Inputs: Model Inputs
    # Outputs: Split Tensor
    gpu_path = os.path.join(output_dir, "gpu.onnx")
    print("Extracting GPU partition...")
    onnx.utils.extract_model(onnx_path, gpu_path, input_names=[i.name for i in model.graph.input], output_names=[split_tensor])
    
    # 3. Extract CPU Part (Split Tensor -> Outputs)
    # Inputs: Split Tensor
    # Outputs: Model Outputs
    cpu_path = os.path.join(output_dir, "cpu.onnx")
    print("Extracting CPU partition...")
    onnx.utils.extract_model(onnx_path, cpu_path, input_names=[split_tensor], output_names=[o.name for o in model.graph.output])
    
    return gpu_path, cpu_path

def run(source_path, output_dir, split_layer):
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp_olive")
    
    print("--- Step 1: Olive Optimization (Whole Model) ---")
    
    # Simple Config for Conversion + INT4
    # Note: This is a simplified config. For production Llama 4, more specific tuning is needed.
    conf = {
        "input_model": {
            "type": "PyTorchModel",
            "config": {
                "model_path": source_path,
                "io_config": {
                    "input_names": ["input_ids", "attention_mask", "position_ids"],
                    "output_names": ["logits"],
                    "dynamic_axes": {
                        "input_ids": {0: "batch", 1: "seq"},
                        "attention_mask": {0: "batch", 1: "seq"},
                        "position_ids": {0: "batch", 1: "seq"},
                        "logits": {0: "batch", 1: "seq"}
                    }
                }
            }
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": ["gpu"]
            }
        },
        "passes": {
            "conversion": {
                "type": "OnnxConversion",
                "config": {
                    "target_opset": 16,
                    "save_as_external_data": True,
                    "all_tensors_to_one_file": True, # Keep it single file for now
                }
            },
            # "transformer_optimization": { ... } # Optional fusion
            "quantization": {
                "type": "OnnxQuantization",
                "config": {
                    "weight_type": "QInt4",
                    "activation_type": "fp16", # Keep activations fp16
                    "save_as_external_data": True,
                    "all_tensors_to_one_file": True
                }
            }
        },
        "engine": {
            "host": "local_system",
            "target": "local_system",
            "start_pass": "conversion",
            "output_dir": temp_dir
        }
    }
    
    # Run Olive
    # Allow failure to demonstrate flow if dependencies missing in this context
    try:
        res = olive.workflows.run(conf)
        # Find output
        # Olive output structure varies. Assuming implicit known path or parsing 'res'.
        # For simplicity, we look in temp_dir for the .onnx file
        optimized_model = glob.glob(os.path.join(temp_dir, "**", "*.onnx"), recursive=True)[0]
        print(f"Optimized model found: {optimized_model}")
    except Exception as e:
        print(f"Olive run failed: {e}")
        return

    print("--- Step 2: Splitting ---")
    gpu_onnx, cpu_onnx = split_onnx_model(optimized_model, output_dir, split_layer)
    
    print("--- Step 3: Config Generation ---")
    config = {
        "hybrid_config": {
            "version": "1.0",
            "split_layer": split_layer,
            "total_layers": 100, # Approximation or parse from model
            "gpu_partition": {
                "onnx_path": "gpu.onnx",
                "provider": "cuda"
            },
            "cpu_partition": {
                "onnx_path": "cpu.onnx",
                "provider": "cpu"
            }
        }
    }
    
    with open(os.path.join(output_dir, "hybrid_config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Done. Output in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split_layer", type=int, default=20)
    args = parser.parse_args()
    
    run(args.source_path, args.output_path, args.split_layer)
