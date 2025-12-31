# MelonStudio - Windows 11 Native AI

A native Windows 11 application (WPF) that runs local LLMs (Phi-3, Llama-3) on NVIDIA RTX 4090 using ONNX Runtime GenAI with CUDA/TensorRT acceleration.

## Prerequisites

1. **Hardware**: NVIDIA RTX GPU (4090 recommended for best performance)
2. **Drivers**: Latest NVIDIA Game Ready or Studio Driver
3. **Software**:
   - **.NET 8 SDK**: [Download Here](https://dotnet.microsoft.com/download/dotnet/8.0)
   - **CUDA Toolkit 12.x**: [Download Here](https://developer.nvidia.com/cuda-downloads)
   - **cuDNN**: Compatible version for your CUDA installation

## Model Setup

This app does **not** download models automatically. You must provide the path to an **ONNX-optimized** model folder.

1. Go to HuggingFace and search for ONNX models:
   - Recommended: [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)
   - Choose the `cuda` variant for RTX GPUs

2. Extract the model to a folder, e.g., `C:\AI\Models\Phi-3-mini-4k-instruct-onnx`

3. Update `Services/SettingsService.cs` with this path

## Building

```bash
# Clone the repository
git clone https://github.com/clpatrick/MelonStudio.git
cd MelonStudio

# Restore and build
dotnet restore
dotnet build --configuration Release
```

## Running

```bash
# Run the GUI app
dotnet run --project MelonStudio

# Run benchmarks (requires model)
dotnet run --project MelonStudio.Benchmark -- "C:\AI\Models\Phi-3" "benchmarks/results.json"
```

## Project Structure

```
MelonStudio/
├── MelonStudio/           # WPF GUI application
├── MelonStudio.Benchmark/ # Console benchmark tool
├── benchmarks/            # Benchmark results (committed)
└── MelonStudio.sln        # Solution file
```

## Benchmarking

The benchmark tool measures tokens per second across standard prompts. Results are saved to `benchmarks/results.json` and can be committed to share performance data.

See [benchmarks/README.md](benchmarks/README.md) for details.

## Troubleshooting

- **`DllNotFoundException`**: CUDA or cuDNN not in PATH
- **Model Load Error**: Point to the *folder* containing `config.json`, not the file
- **Build Errors**: Ensure .NET 8 SDK and x64 platform
