
import os
import json
import shutil
import argparse
import re

def create_proxy_model(source_dir, dest_dir, layers=20):
    print(f"Creating proxy model with {layers} layers...")
    print(f"Source: {source_dir}")
    print(f"Dest: {dest_dir}")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Load Index
    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        return

    with open(index_path, 'r') as f:
        index_data = json.load(f)
        
    weight_map = index_data.get("weight_map", {})
    
    # 2. Filter Keys
    new_weight_map = {}
    required_files = set()
    
    # regex for layers: model.layers.{i}.
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.")
    
    for key, filename in weight_map.items():
        # Keep non-layer keys (embeddings, norm, head)
        match = layer_pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx < layers:
                new_weight_map[key] = filename
                required_files.add(filename)
        else:
            # Embeddings, final norm, lm_head usually needed
            new_weight_map[key] = filename
            required_files.add(filename)
            
    print(f"Filtered weight map: {len(new_weight_map)} keys, {len(required_files)} files required.")
    
    # 3. Copy/Link Files
    for filename in required_files:
        src_file = os.path.join(source_dir, filename)
        dst_file = os.path.join(dest_dir, filename)
        
        if not os.path.exists(src_file):
            print(f"Error: Required file missing: {src_file}")
            return
            
        if not os.path.exists(dst_file):
            print(f"Linking {filename}...")
            try:
                # Try Hardlink first (fast, saves space)
                os.link(src_file, dst_file)
            except OSError:
                print(f"Hardlink failed, trying copy...")
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
        
    # Validation fix for GPT-OSS: Slice layer_types if present
    if "layer_types" in config and isinstance(config["layer_types"], list):
        print(f"Slicing layer_types from {len(config['layer_types'])} to {layers}")
        config["layer_types"] = config["layer_types"][:layers]

    config["num_hidden_layers"] = layers
    
    with open(os.path.join(dest_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # 6. Copy Tokenizer files
    for f in os.listdir(source_dir):
        if f.startswith("tokenizer") or f.endswith(".json") or f.endswith(".model"):
            if f not in ["config.json", "model.safetensors.index.json"]:
                 shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_dir, f))
                 
    print("Proxy model created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Source model directory")
    parser.add_argument("dest", help="Destination directory")
    parser.add_argument("--layers", type=int, default=20)
    args = parser.parse_args()
    
    create_proxy_model(args.source, args.dest, args.layers)
