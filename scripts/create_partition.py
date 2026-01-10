"""
Script to create GPU or CPU partition from the full model.
GPU partition: embeddings + layers 0 to N-1
CPU partition: layers N to end + lm_head
"""
import os
import json
import shutil
import argparse
import re

def create_partition(source_dir, dest_dir, start_layer, end_layer, include_embeddings=True, include_head=True):
    print(f"Creating partition with layers {start_layer}-{end_layer}...")
    print(f"  Include embeddings: {include_embeddings}")
    print(f"  Include head: {include_head}")
    print(f"Source: {source_dir}")
    print(f"Dest: {dest_dir}")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Load Index
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        return False

    with open(index_path, 'r') as f:
        index_data = json.load(f)
        
    weight_map = index_data.get("weight_map", {})
    
    # 2. Filter Keys
    new_weight_map = {}
    required_files = set()
    
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    
    for key, filename in weight_map.items():
        match = layer_pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            if start_layer <= layer_idx <= end_layer:
                new_weight_map[key] = filename
                required_files.add(filename)
        else:
            # Non-layer keys: embed_tokens, norm, lm_head
            is_embedding = "embed_tokens" in key or "embed" in key.lower()
            is_head = "lm_head" in key or "head" in key.lower()
            is_norm = "norm" in key.lower() and "layer" not in key.lower()
            
            if is_embedding and include_embeddings:
                new_weight_map[key] = filename
                required_files.add(filename)
            elif is_head and include_head:
                new_weight_map[key] = filename
                required_files.add(filename)
            elif is_norm:
                # Final norm usually needed with head
                if include_head:
                    new_weight_map[key] = filename
                    required_files.add(filename)
            
    print(f"Filtered weight map: {len(new_weight_map)} keys, {len(required_files)} files required.")
    
    # 3. Copy/Link Files
    for filename in required_files:
        src_file = os.path.join(source_dir, filename)
        dst_file = os.path.join(dest_dir, filename)
        
        if not os.path.exists(src_file):
            print(f"Error: Required file missing: {src_file}")
            return False
            
        if not os.path.exists(dst_file):
            print(f"Linking {filename}...")
            try:
                os.link(src_file, dst_file)
            except OSError:
                print(f"Hardlink failed, copying...")
                shutil.copy2(src_file, dst_file)
    
    # 4. Write new index
    new_index = index_data.copy()
    new_index["weight_map"] = new_weight_map
    
    with open(os.path.join(dest_dir, "model.safetensors.index.json"), 'w') as f:
        json.dump(new_index, f, indent=2)
        
    # 5. Config Patching
    config_path = os.path.join(source_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_layers = end_layer - start_layer + 1
    
    # Slice layer_types if present
    if "layer_types" in config and isinstance(config["layer_types"], list):
        config["layer_types"] = config["layer_types"][start_layer:end_layer+1]
        print(f"Sliced layer_types to {len(config['layer_types'])} entries")

    config["num_hidden_layers"] = num_layers
    config["_partition_start_layer"] = start_layer
    config["_partition_end_layer"] = end_layer
    
    with open(os.path.join(dest_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # 6. Copy Tokenizer files
    for f in os.listdir(source_dir):
        if f.startswith("tokenizer") or f.endswith(".model"):
            shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_dir, f))
                 
    print("Partition created successfully.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source model directory")
    parser.add_argument("dest", help="Destination directory")
    parser.add_argument("--start", type=int, default=0, help="Start layer (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End layer (inclusive)")
    parser.add_argument("--no-embeddings", action="store_true", help="Exclude embeddings")
    parser.add_argument("--no-head", action="store_true", help="Exclude lm_head")
    args = parser.parse_args()
    
    create_partition(
        args.source, 
        args.dest, 
        args.start, 
        args.end,
        include_embeddings=not args.no_embeddings,
        include_head=not args.no_head
    )
