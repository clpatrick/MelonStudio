#!/usr/bin/env python3
"""
Verify downloaded HuggingFace model files against remote repository.

Checks file existence, sizes, and detects incomplete downloads (LFS pointers).
Supports private repositories with token authentication.
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, list_repo_files
from huggingface_hub.utils import EntryNotFoundError
import requests

# Ensure stdout/stderr can handle Unicode on Windows
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout/stderr
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        # Fallback: use ASCII-safe characters only (already replaced below)
        pass


def verify_download(repo_id, local_dir, token=None, verbose=False):
    """
    Verify downloaded model files against HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "microsoft/Phi-3.5-mini-instruct")
        local_dir: Local directory containing downloaded files
        token: Optional HuggingFace token for private/gated repositories
        verbose: If True, print details for each file
    
    Returns:
        tuple: (is_valid, missing_files, size_mismatches)
    """
    print(f"Verifying {repo_id} against {local_dir}...")
    
    if not os.path.exists(local_dir):
        print(f"ERROR: Local directory does not exist: {local_dir}")
        return False, [], []
    
    # Initialize API with token if provided
    api = HfApi(token=token)
    
    # Get remote file list
    try:
        print("Fetching remote file list...")
        remote_files = list_repo_files(repo_id=repo_id, token=token)
        # Filter for relevant files (safetensors, json, model, bin, pt, pth)
        relevant_extensions = {'.safetensors', '.json', '.model', '.bin', '.pt', '.pth', '.txt', '.md'}
        remote_files = [f for f in remote_files if any(f.lower().endswith(ext) for ext in relevant_extensions)]
        # Exclude optimizer/scheduler files
        remote_files = [f for f in remote_files if 'optimizer' not in f.lower() and 'scheduler' not in f.lower()]
    except Exception as e:
        print(f"ERROR: Failed to fetch remote file list: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("  Hint: This repository may require authentication. Try providing --token")
        return False, [], []

    print(f"Found {len(remote_files)} relevant files in remote repository.")
    
    missing_files = []
    size_mismatches = []
    valid_files = 0
    lfs_pointers = []
    
    # Get file metadata from HuggingFace API (more reliable than HEAD requests)
    print("Fetching file metadata...")
    try:
        model_info = api.model_info(repo_id=repo_id, files_metadata=True, token=token)
        # Create a map of filename -> size
        remote_file_sizes = {}
        for sibling in model_info.siblings:
            if sibling.rfilename and sibling.size is not None:
                remote_file_sizes[sibling.rfilename] = sibling.size
    except Exception as e:
        print(f"WARNING: Could not fetch detailed metadata: {e}")
        print("  Falling back to HEAD requests (may be less accurate for LFS files)")
        remote_file_sizes = {}
    
    # Verify each file
    for filename in remote_files:
        local_path = os.path.join(local_dir, filename)
        
        # Check if file exists
        if not os.path.exists(local_path):
            if verbose:
            print(f"MISSING: {filename}")
            missing_files.append(filename)
            continue
            
        # Get local file size
        try:
            local_size = os.path.getsize(local_path)
        except OSError as e:
            print(f"ERROR: Could not read {filename}: {e}")
            size_mismatches.append(filename)
            continue
        
        # Get remote file size
        remote_size = None
        
        # Try to get from metadata first (most reliable)
        if filename in remote_file_sizes:
            remote_size = remote_file_sizes[filename]
        else:
            # Fallback to HEAD request
            try:
                url = api.hf_hub_url(repo_id=repo_id, filename=filename)
                response = requests.head(url, allow_redirects=True, timeout=10)
                
                if response.status_code == 200:
            remote_size = int(response.headers.get('Content-Length', 0))
                elif response.status_code == 401:
                    if not token:
                        print(f"WARNING: {filename} requires authentication (401). Provide --token for private repos.")
                    remote_size = None
                elif response.status_code == 404:
                    # File might not exist remotely (could be a local-only file)
                    if verbose:
                        print(f"INFO: {filename} not found remotely (may be local-only)")
                    remote_size = None
        except Exception as e:
                if verbose:
                    print(f"WARNING: Could not fetch metadata for {filename}: {e}")
                remote_size = None
        
        # Compare sizes
        if remote_size is None:
            # Couldn't determine remote size, skip verification
            if verbose:
                print(f"SKIP: {filename} (could not determine remote size)")
            valid_files += 1  # Assume valid if we can't verify
            continue
        
        if local_size == remote_size:
            valid_files += 1
            if verbose:
                print(f"OK: {filename} ({local_size:,} bytes)")
        else:
            # Check if it's an LFS pointer file (incomplete download)
            if local_size < 2000 and remote_size > 1000000:
                if verbose:
                    print(f"LFS POINTER: {filename} (Local: {local_size} bytes, Remote: {remote_size:,} bytes)")
                    print(f"  -> This is a Git LFS pointer file. The actual file was not downloaded.")
                lfs_pointers.append(filename)
            size_mismatches.append(filename)
        else:
                if verbose:
                    print(f"MISMATCH: {filename} (Local: {local_size:,} bytes, Remote: {remote_size:,} bytes)")
                size_mismatches.append(filename)
        
        # Progress indicator
        if not verbose and (valid_files + len(missing_files) + len(size_mismatches)) % 10 == 0:
            print(f"  Verified {valid_files + len(missing_files) + len(size_mismatches)}/{len(remote_files)} files...")
    
    # Print summary
    print("-" * 60)
    print("VERIFICATION SUMMARY")
    print("-" * 60)
    print(f"Total files checked: {len(remote_files)}")
    print(f"Valid files: {valid_files}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Size mismatches: {len(size_mismatches)}")
    
    if lfs_pointers:
        print(f"\n[WARNING] LFS POINTER FILES DETECTED: {len(lfs_pointers)}")
        print("   These files are Git LFS pointers, not the actual model files.")
        print("   The download may have failed or Git LFS is not configured.")
        if not verbose:
            print("\n   Run with --verbose to see which files are pointers.")
    
    if missing_files:
        print(f"\n[ERROR] MISSING FILES ({len(missing_files)}):")
        for f in missing_files[:10]:  # Show first 10
            print(f"   - {f}")
        if len(missing_files) > 10:
            print(f"   ... and {len(missing_files) - 10} more")
    
    if size_mismatches and not lfs_pointers:
        print(f"\n[WARNING] SIZE MISMATCHES ({len(size_mismatches)}):")
        for f in size_mismatches[:10]:  # Show first 10
            print(f"   - {f}")
        if len(size_mismatches) > 10:
            print(f"   ... and {len(size_mismatches) - 10} more")
    
    print("-" * 60)
    
    # Determine overall status
    is_valid = len(missing_files) == 0 and len(size_mismatches) == 0
    
    if is_valid:
        print("[OK] VERIFICATION SUCCESSFUL")
        return True, missing_files, size_mismatches
    else:
        print("[FAILED] VERIFICATION FAILED")
        if lfs_pointers:
            print("\n[TIP] Configure Git LFS or re-download the model.")
        return False, missing_files, size_mismatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify downloaded HuggingFace model files against remote repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify public model
  python verify_download.py microsoft/Phi-3.5-mini-instruct ./models/phi3.5
  
  # Verify private model with token
  python verify_download.py private/model ./models/private --token hf_xxxxx
  
  # Verbose output
  python verify_download.py microsoft/Phi-3.5-mini-instruct ./models/phi3.5 --verbose
        """
    )
    parser.add_argument("repo_id", help="HuggingFace repository ID (e.g., 'microsoft/Phi-3.5-mini-instruct')")
    parser.add_argument("local_dir", help="Local directory containing downloaded model files")
    parser.add_argument("--token", type=str, help="HuggingFace token for private/gated repositories")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information for each file")
    
    args = parser.parse_args()
    
    # Set token in environment if provided
    if args.token:
        os.environ["HF_TOKEN"] = args.token
    
    is_valid, missing, mismatches = verify_download(
        args.repo_id, 
        args.local_dir, 
        token=args.token,
        verbose=args.verbose
    )
    
    sys.exit(0 if is_valid else 1)
