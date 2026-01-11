# Create virtual environment if it doesn't exist
if (!(Test-Path .olive-env)) {
    Write-Host "Creating virtual environment .olive-env..."
    python -m venv .olive-env
}

# Activate environment
Write-Host "Activating environment..."
. .\.olive-env\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
# Install olive-ai with GPU support, ONNX Runtime GenAI for CUDA, and transformers
# Note: Install PyTorch with CUDA support first (required for mamba-ssm and GPU quantization)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install olive-ai[gpu] onnxruntime-genai-cuda transformers huggingface_hub
# Install datasets library (required for GPTQ quantization calibration)
pip install datasets
# Install accelerate (required for loading quantized models in auto-opt)
pip install accelerate
# Install optimum with ONNX support (required for ONNX conversion dummy inputs)
pip install "optimum[onnxruntime]"
# Install autoawq (required for AWQ quantization algorithm)
pip install autoawq
# Install mamba-ssm for Mamba architecture models (e.g., Nemotron)
# Note: mamba-ssm requires CUDA-enabled PyTorch and may need to be built from source
pip install mamba-ssm

# Verify installation
Write-Host "Verifying installation..."
python -c "import olive; print('Olive version:', olive.__version__)"
python -c "import onnxruntime_genai; print('ONNX Runtime GenAI installed')"

Write-Host "Setup complete."
