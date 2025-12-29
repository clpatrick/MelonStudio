# LocalLLMchat - Windows 11 Native AI

A native Windows 11 application (WinUI 3) that runs local LLMs (Phi-3, Llama-3) on NVIDIA RTX 4090 using standard ONNX Runtime GenAI with TensorRT acceleration.

## Prerequisites

1.  **Hardware**: NVIDIA RTX GPU (4090 recommended for best performance).
2.  **Drivers**: Latest NVIDIA Game Ready or Studio Driver.
3.  **Software**:
    *   **Visual Studio 2022** (with ".NET Desktop Development" and "Windows App SDK" workloads).
    *   **CUDA Toolkit 13.1**: [Download Here](https://developer.nvidia.com/cuda-downloads).
    *   **TensorRT for RTX 1.3.0.35**: Ensure this version is installed and accessible.
    *   **cuDNN**: Compatible version for CUDA 13.1.

## Model Setup

This app does **not** download models automatically. You must providing the path to an **ONNX-optimized** model folder.

1.  Go to HuggingFace and search for ONNX models. 
    *   Recommended: [microsoft/Phi-3-mini-4k-instruct-onnx](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)
    *   Make sure to download the `main` branch or the specific nested folder containing `.onnx` files and `config.json`.
    *   **IMPORTANT**: For RTX 4090, choose the `cuda` or `tensorrt` quantized version if available (e.g., `int4-cuda`), otherwise `cpu_and_mobile` versions might run slower or on CPU. The `direct-ml` version is also a good versatile backup.

2.  Extract the model to a folder, e.g., `C:\AI\Models\Phi-3`.
3.  Update `Services/SettingsService.cs` with this path, or just run the app and let it error (future feature: settings UI).

## Building

1.  Open `LocalLLMchat.sln` in Visual Studio.
2.  Restore NuGet packages (this may take a few minutes as `Microsoft.ML.OnnxRuntimeGenAI.Cuda` downloads large binaries).
3.  Set platform to `x64`.
4.  Build and Run.

## Troubleshooting

- **`DllNotFoundException`**: Usually means CUDA, cuDNN, or TensorRT is not in your PATH.
- **Model Load Error**: Ensure you pointed to the *folder* containing `config.json`, not the file itself.
