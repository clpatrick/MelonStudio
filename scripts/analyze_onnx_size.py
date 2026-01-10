
import argparse
import onnx
import os
import numpy as np
import re
from onnx import numpy_helper

def analyze_size(model_path):
    print(f"Analyzing sizes for {model_path}...")
    
    # Load model structure (no large data if possible, but for initializers we need info)
    # onnx.load won't load external data by default if it's in separate files, 
    # but we need to know the size. 
    # For a consolidated .onnx file (usually from optimum export without splitting?), local tensors are in .graph.initializer
    
    model = onnx.load(model_path, load_external_data=False)
    
    base_size = 0
    layer_sizes = {}
    
    # Regex for layers (supports layers.N., layers/N/, layer.N., block.N., h.N.)
    # Matches: layers.0., layers/0/, layer.0., etc.
    layer_pattern = re.compile(r"(?:layers|layer|block|h)[./](\d+)[./]")
    
    print(f"Counting initializers: {len(model.graph.initializer)}")
    
    for tensor in model.graph.initializer:
        name = tensor.name
        
        # Calculate size in bytes
        # If external, use external_data info?
        size = 0
        if tensor.HasField("raw_data"):
            size = len(tensor.raw_data)
        else:
            # Fallback for non-raw (float_data etc)
            # This is rare for quantized/large models
            dtype = tensor.data_type
            # simple estimate
            elem_count = np.prod(tensor.dims)
            if dtype == 1: # Float
                size = elem_count * 4
            elif dtype == 16 or dtype == 10: # Float16 / BFloat16
                size = elem_count * 2
            elif dtype == 2 or dtype == 3: # UInt8 / Int8
                size = elem_count * 1
                
        # Check if external data
        if size == 0:
             # Look at external_data_location? 
             # OR optimum 'save_model_to_file' from quantizer usually embeds? 
             # Exported model usually uses external data.
             pass

        # If size is still 0, try to infer? 
        if size == 0:
             # Assume filtered out by load_external_data=False?
             # We should probably use load_external_data=True if we want size?
             # No, that loads memory.
             # We can check specific attributes.
             # But let's assume raw_data IS present but empty? 
             # Let's just try loading tensor info.
             pass

    # Better approach: Use os.path.getsize of the external data files?
    # Or rely on tensor metadata.
    
    # Let's iterate the graph inputs/initializers properly.
    # Actually, for "sizing", simpler is:
    # 1. Total file size of the folder (assuming 2-layer ONNX is mostly weights).
    # 2. Assume Overhead.
    
    # But I want Base vs Layer.
    # I'll rely on the NAMING to sort the tensors, then I need their size.
    # `tensor.dims` is always available. `tensor.data_type` is available.
    
    total_calculated = 0
    dtype_counts = {}
    
    for tensor in model.graph.initializer:
        dims = tensor.dims
        count = np.prod(dims) if dims else 1
        dtype = tensor.data_type
        
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
        
        # Bit width
        # Bit width
        bits = 32 # default
        if dtype in [1]: bits = 32
        elif dtype in [16, 10]: bits = 16
        elif dtype in [2, 3]: bits = 8 # uint8/int8
        elif dtype == 4: bits = 16 # uint16
        # INT4 is usually packed in uint8 (type 2), so bits=8 for packed elements.
        # But wait, packed means 2 elements per byte. count is "packed count" or "unpacked"?
        # In MatMulNBits, B is usually packed. Shape of B is [N, K/2].
        # So count reflects packed shape.
        # Thus bytes = count * 1 (for uint8).
        
        bytes_val = int(count * (bits / 8) if bits >= 8 else count) # simplistic
        
        # Refine for packed types?
        # If I see uint8, I assume 1 byte per element of shape.
        
        total_calculated += bytes_val
        
        match = layer_pattern.search(tensor.name)
        if match and "final_norm" not in tensor.name:
            idx = int(match.group(1))
            layer_sizes[idx] = layer_sizes.get(idx, 0) + bytes_val
        else:
            base_size += bytes_val
            
    print(f"Total Calculated Params Size: {total_calculated / 1024**3:.4f} GB")
    print(f"Base Size: {base_size / 1024**3:.4f} GB")
    
    # Determine dominant precision
    # Types: 1=Float, 2=UInt8, 3=Int8, 10=Float16, 16=BFloat16
    if dtype_counts:
        dom_type = max(dtype_counts, key=dtype_counts.get)
        prec_str = "Unknown"
        if dom_type == 1: prec_str = "FP32"
        elif dom_type == 10: prec_str = "FP16"
        elif dom_type == 16: prec_str = "BF16"
        elif dom_type == 2 or dom_type == 3:
            # Check for block quantization (INT4 often packed in uint8)
            # If simplistic, just call it INT8/4
            prec_str = "INT8/INT4" 
            
        print(f"Dominant Precision: {prec_str}")
    else:
        print("Dominant Precision: Unknown")
    
    if len(layer_sizes) > 0:
        avg_layer = sum(layer_sizes.values()) / len(layer_sizes)
        print(f"Average Layer Size: {avg_layer / 1024**3:.4f} GB")
        min_layer = min(layer_sizes.keys())
        max_layer = max(layer_sizes.keys())
        print(f"Layers Found: {len(layer_sizes)} (indices {min_layer}-{max_layer})")
    else:
        print("No layers found (Regex mismatch?)")
        avg_layer = 0
        
    return base_size, avg_layer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to ONNX model file")
    args = parser.parse_args()
    
    analyze_size(args.model)
