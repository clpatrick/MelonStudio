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
# Note: Specific torch version might be needed for CUDA 12, letting pip resolve for now.
pip install olive-ai[gpu] onnxruntime-genai-cuda transformers torch huggingface_hub

# Verify installation
Write-Host "Verifying installation..."
python -c "import olive; print('Olive version:', olive.__version__)"
python -c "import onnxruntime_genai; print('ONNX Runtime GenAI installed')"

Write-Host "Setup complete."
