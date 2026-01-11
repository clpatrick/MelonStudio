# Installing mamba-ssm for Mamba Architecture Models

Some models (like Nemotron) use the Mamba architecture and require the `mamba-ssm` package. This package needs to be built from source and requires CUDA-enabled PyTorch.

## Prerequisites

1. **CUDA Toolkit** - Already installed for MelonStudio
2. **Visual C++ Build Tools** - Required for compiling Python packages
3. **PyTorch with CUDA** - Must be installed before mamba-ssm

## Installation Steps

1. **Activate the Olive environment:**
   ```powershell
   cd C:\Repos\MelonStudio
   .\scripts\olive\.olive-env\Scripts\Activate.ps1
   ```

2. **Uninstall CPU-only PyTorch (if present):**
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   ```

3. **Install PyTorch with CUDA support:**
   
   **Note:** If you have CUDA 13.1, PyTorch may not have wheels for it yet. Use CUDA 12.4 wheels (backward compatible):
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   
   **Alternative CUDA versions:**
   - For CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
   - For CUDA 12.6: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
   - For CUDA 12.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   - For CUDA 11.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
   
   **Note:** `torchaudio` is optional and may not be available for all CUDA versions. You can skip it.

5. **Verify PyTorch CUDA support:**
   ```powershell
   python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
   ```
   You should see `CUDA available: True` and a CUDA version number.

6. **Install mamba-ssm:**
   
   **Option 1: Try installing with build isolation disabled (may fix setup.py bug):**
   ```powershell
   pip install mamba-ssm --no-build-isolation
   ```
   
   **Option 2: If Option 1 fails, try installing from git (may have fixes):**
   ```powershell
   pip install git+https://github.com/state-spaces/mamba.git
   ```
   
   **Option 3: Standard installation (may fail with setup.py bug):**
   ```powershell
   pip install mamba-ssm
   ```
   
   **Note:** If you get "nvcc was not found" error, ensure CUDA Toolkit bin directory is in your PATH:
   ```powershell
   # Check if nvcc is accessible
   nvcc --version
   
   # If not found, add CUDA bin to PATH for this session:
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
   # Then retry installation
   ```

## Troubleshooting

### Error: "nvcc was not found"
- Ensure CUDA Toolkit is installed and `nvcc` is in your PATH
- Check: `nvcc --version`

### Error: "Microsoft Visual C++ 14.0 or greater is required"
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Select "Desktop development with C++" workload during installation

### Error: "NameError: name 'bare_metal_version' is not defined"
- This is a known bug in mamba-ssm setup.py
- Try installing with `--no-build-isolation`: `pip install mamba-ssm --no-build-isolation`
- Or install from git: `pip install git+https://github.com/state-spaces/mamba.git`
- Or try a different version: `pip install mamba-ssm==2.2.6`

### Error: "nvcc was not found" during mamba-ssm installation
- The build environment may not have CUDA in PATH
- Add CUDA bin to PATH for this session:
  ```powershell
  $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
  ```
- Then retry: `pip install mamba-ssm --no-build-isolation`

### PyTorch still shows CPU version
- Make sure you uninstalled the CPU version first
- Verify the CUDA index URL is correct for your CUDA version
- For CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- For CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`

## Alternative: Use Pre-built Wheels (if available)

If building from source fails, check if pre-built wheels are available:
```powershell
pip install mamba-ssm --only-binary :all:
```

If that fails, you may need to build from source with the proper build tools installed.
