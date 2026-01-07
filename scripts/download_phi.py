
import time
import sys
from huggingface_hub import snapshot_download

def download_model():
    repo_id = "microsoft/Phi-3.5-mini-instruct"
    local_dir = "cache_dir/microsoft_Phi-3.5-mini-instruct"
    
    print(f"Starting download of {repo_id} to {local_dir}...")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            resume_download=True,
            ignore_patterns=["*.gguf", "*.awq", "*.gptq"],
            max_workers=8  # Parallel download
        )
        print(f"SUCCESS: Model downloaded to {path}")
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
