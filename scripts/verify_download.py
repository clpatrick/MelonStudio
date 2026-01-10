
import os
import argparse
from huggingface_hub import list_repo_files, get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError
import requests

def verify_download(repo_id, local_dir):
    print(f"Verifying {repo_id} against {local_dir}...")
    
    # Get remote file list
    try:
        remote_files = list_repo_files(repo_id=repo_id)
        # Filter for relevant files (safetensors, json, model)
        remote_files = [f for f in remote_files if f.endswith('.safetensors') or f.endswith('.json') or f.endswith('.model')]
    except Exception as e:
        print(f"Error fetching remote list: {e}")
        return

    print(f"Found {len(remote_files)} relevant files in remote repo.")
    
    missing_files = []
    size_mismatches = []
    valid_files = 0
    
    for filename in remote_files:
        local_path = os.path.join(local_dir, filename)
        
        if not os.path.exists(local_path):
            print(f"MISSING: {filename}")
            missing_files.append(filename)
            continue
            
        # Get remote metadata (size)
        try:
            # We can use hf_hub_url to get the url, then HEAD request for size
            url = hf_hub_url(repo_id, filename)
            response = requests.head(url, allow_redirects=True)
            if response.status_code == 401:
                 # Check if we need token? The repo was ungated earlier.
                 pass
            
            remote_size = int(response.headers.get('Content-Length', 0))
            
            # Fallback if content-length missing (LFS)
            if remote_size == 0:
                 # Try get_hf_file_metadata??
                 pass
                 
        except Exception as e:
            print(f"Warning: Could not fetch metadata for {filename}: {e}")
            remote_size = -1

        local_size = os.path.getsize(local_path)
        
        if remote_size != -1 and local_size != remote_size:
            print(f"MISMATCH: {filename} (Local: {local_size}, Remote: {remote_size})")
            # Check if it's a pointer file (LFS pointer is small)
            if local_size < 2000 and remote_size > 1000000:
                print(f" -> Likely an LFS pointer (Download failed for this file)")
            size_mismatches.append(filename)
        else:
            # print(f"OK: {filename}")
            valid_files += 1
            if valid_files % 10 == 0:
                print(f"Verified {valid_files} files...")

    print("-" * 30)
    print(f"Verification Complete.")
    print(f"Valid: {valid_files}")
    print(f"Missing: {len(missing_files)}")
    print(f"Mismatches: {len(size_mismatches)}")
    
    if len(missing_files) > 0 or len(size_mismatches) > 0:
        print("VERIFICATION FAILED")
        exit(1)
    else:
        print("VERIFICATION SUCCESSFUL")
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Hugging Face Repo ID")
    parser.add_argument("local_dir", help="Local directory containing model files")
    args = parser.parse_args()
    
    verify_download(args.repo_id, args.local_dir)
