import argparse
import os
import sys
import json
from huggingface_hub import snapshot_download, HfApi
from olive.model import HfModelHandler

def download_and_copy(model_id, output_dir, token=None, subfolder=None):
    print(f"Initializing download for {model_id}...")
    
    try:
        # 1. Fetch Metadata for Progress UI
        print("Fetching model metadata...")
        try:
            api = HfApi(token=token)
            model_info = api.model_info(repo_id=model_id, files_metadata=True)
            
            files_data = []
            total_bytes = 0
            
            for sibling in model_info.siblings:
                if sibling.rfilename:
                    size = getattr(sibling, 'size', 0)
                    if size is None: size = 0
                    total_bytes += size
                    files_data.append({
                        "name": sibling.rfilename,
                        "sizeBytes": str(size)
                    })
            
            metadata = {
                "total_bytes": total_bytes,
                "file_count": len(files_data),
                "files": files_data
            }
            # Print specifically formatted JSON for C# parsing
            print(f"METADATA: {json.dumps(metadata)}")
            
        except Exception as meta_ex:
            print(f"[WARNING] Failed to fetch metadata: {meta_ex}")

        allow_patterns = None
        if subfolder:
            # If subfolder is "cuda/cuda-int4", we want "cuda/cuda-int4/*" AND the root files potentially?
            # Actually, standard is to just download that folder.
            print(f"Downloading subset: {subfolder}")
            allow_patterns = [f"{subfolder}/*", "config.json", "*.py", "*.md"] # include basics just in case
            
        saved_path = snapshot_download(
            repo_id=model_id,
            token=token,
            local_dir=output_dir,
            local_dir_use_symlinks=False, # CRITICAL FIX for Windows
            resume_download=True,
            allow_patterns=allow_patterns
        )
            
        # Replace unicode characters to avoid encoding errors on Windows console
        print(f"[OK] Model successfully saved to {saved_path}")
        
        # 3. Validation
        try:
            print("Validating with Olive HfModelHandler...")
            # We point to the LOCAL copy
            handler = HfModelHandler(model_path=saved_path)
            print("[OK] Olive acceptance check passed")
        except Exception as olive_ex:
            print(f"[WARNING] Olive validation warning (download succeeded though): {olive_ex}")
        
    except Exception as e:
        print(f"Error executing download: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace Model ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Final output directory")
    parser.add_argument("--token", type=str, help="HuggingFace Token (optional)")
    parser.add_argument("--subfolder", type=str, help="Optional subfolder to download (e.g. cuda/cuda-int4)")
    
    args = parser.parse_args()
    
    if args.token:
        os.environ["HF_TOKEN"] = args.token
    
    download_and_copy(args.model_id, args.output_dir, args.token, args.subfolder)
