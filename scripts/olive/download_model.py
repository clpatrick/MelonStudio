import argparse
import os
from huggingface_hub import snapshot_download

def download_model(model_id, output_dir):
    print(f"Downloading {model_id} to {output_dir}...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Download
    # allow_patterns can be used to filter files if we only want safetensors
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--output_path", type=str, default=r"C:\Models\Llama-4-Scout-Source", help="Local download path")
    args = parser.parse_args()
    
    download_model(args.model_id, args.output_path)
