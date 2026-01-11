# Installing TensorRT for Olive GPU Acceleration

TensorRT is required when using GPU-accelerated quantization algorithms (AWQ, GPTQ) with Olive. This guide will help you install TensorRT and its dependencies.

## Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit 12.x** (already required for MelonStudio)
3. **Python 3.10+** (used by Olive virtual environment)

## Installation Steps

### 1. Download TensorRT

1. Visit [NVIDIA TensorRT Download Page](https://developer.nvidia.com/nvidia-tensorrt-download)
2. Log in with your NVIDIA Developer account (free registration)
3. Select TensorRT version compatible with your CUDA version (typically TensorRT 10.x for CUDA 12.x)
4. Download the Windows zip package

### 2. Extract and Install TensorRT

1. Extract the downloaded zip to a directory (e.g., `C:\TensorRT-10.x.x.x`)
2. This directory will be referred to as `<installpath>`

### 3. Add TensorRT to System PATH

1. Press `Win + X` and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System variables", select "Path" and click "Edit"
5. Click "New" and add: `<installpath>\bin`
6. Click "OK" to save all changes
7. **Restart your computer** for PATH changes to take effect

### 4. Install TensorRT Python Bindings

1. Open PowerShell or Command Prompt
2. Navigate to the MelonStudio repository root:
   ```powershell
   cd C:\Repos\MelonStudio
   ```
3. Activate the Olive virtual environment:
   ```powershell
   .\scripts\olive\.olive-env\Scripts\Activate.ps1
   ```
   Or if you're already in the `scripts\olive` directory:
   ```powershell
   .\.olive-env\Scripts\Activate.ps1
   ```
4. Navigate to TensorRT Python directory:
   ```powershell
   cd <installpath>\python
   ```
   (Replace `<installpath>` with your TensorRT installation path, e.g., `C:\TensorRT-10.x.x.x`)
5. Install the appropriate wheel file for your Python version:
   ```powershell
   python -m pip install tensorrt-*-cp310-none-win_amd64.whl
   ```
   (Replace `cp310` with your Python version: `cp310` for Python 3.10, `cp311` for Python 3.11, etc.)

### 5. Verify Installation

1. Make sure you're still in the activated Olive environment (from step 4.3)
2. Test TensorRT import:
   ```powershell
   python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"
   ```

3. Verify `nvinfer_10.dll` is accessible:
   ```powershell
   python -c "import os; print('TensorRT bin in PATH:', 'C:\\Program Files\\NVIDIA\\TensorRT-10.14.1.48\\bin' in os.environ.get('PATH', ''))"
   ```
   (Replace `<installpath>` with your TensorRT installation path)

## Troubleshooting

### Error: "nvinfer_10.dll is missing"

- **Solution**: Ensure TensorRT `bin` directory is in your system PATH and you've restarted your computer
- Check that `nvinfer_10.dll` exists in `<installpath>\bin`

### Error: "TensorRT Python module not found"

- **Solution**: Install the TensorRT Python wheel in the Olive virtual environment
- Make sure you're using the correct Python version (check with `python --version`)

### Error: "CUDA version mismatch"

- **Solution**: Ensure TensorRT version matches your CUDA version
- TensorRT 10.x requires CUDA 12.x
- Check CUDA version: `nvcc --version`

## Additional Dependencies

Some models may require additional Python packages. If you encounter `ModuleNotFoundError` during conversion:

1. **Mamba models** (e.g., Nemotron): Require `mamba-ssm`
   ```powershell
   .\scripts\olive\.olive-env\Scripts\Activate.ps1
   # First ensure PyTorch has CUDA support (required for mamba-ssm)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   # Then install mamba-ssm
   pip install mamba-ssm
   ```
   
   **Note**: If installation fails, `mamba-ssm` may need to be built from source. Ensure:
   - CUDA Toolkit is installed and `nvcc` is in PATH
   - PyTorch with CUDA support is installed (not CPU-only version)

2. **Other model-specific dependencies**: Check the model's HuggingFace page for required packages

## Notes

- TensorRT is only needed for GPU-accelerated quantization (AWQ, GPTQ)
- CPU-based quantization (Quarot, Spinquant, etc.) does not require TensorRT
- The TensorRT provider will be automatically used by ONNX Runtime when available
- Model-specific dependencies (like `mamba-ssm`) must be installed in the Olive environment
