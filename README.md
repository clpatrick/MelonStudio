# MelonStudio - Local LLM Chat for Windows

A native Windows 11 application (WPF/.NET 8) for running local LLMs on NVIDIA RTX GPUs using ONNX Runtime GenAI with CUDA acceleration.

![Build Status](https://github.com/clpatrick/MelonStudio/actions/workflows/build.yml/badge.svg)

## Features

### ✅ Implemented

- **Chat Interface** - Multi-turn conversation with streaming responses
  - Stop generation mid-stream
  - Configurable max tokens, temperature, top-p
  - System prompt customization
  - Auto-scroll with message history

- **Model Discovery** - Search HuggingFace for compatible models
  - Filter by ONNX-ready, CUDA, INT4, FP16
  - Sort by downloads, likes, recency
  - Compatibility indicators (✓/⚠/✗) with detailed tooltips
  - Smart filtering of incompatible formats (MLX, quantized GGUF, EXL2)

- **Model Conversion** - Convert source models to ONNX format
  - Dedicated Convert view with full UI
  - Support for HuggingFace ID, local path, or existing models
  - Precision options: INT4, FP16, FP32
  - Provider options: CUDA, CPU, DML
  - Multi-variant support (create GPU + CPU versions)
  - Enhanced error diagnostics with categorized failures

- **Local Model Management** - Track downloaded/converted models
  - Status indicators: 📥 Downloaded, ⏳ Converting, ✓ Converted, ⚠ Failed
  - Format and precision display
  - Click-to-convert for unconverted models
  - Scan temp folder for failed conversions

- **Settings** - Persistent configuration
  - Default output folder
  - HuggingFace token storage
  - Generation parameters

### 🚧 Planned (Priority Order)

1. **Multi-Device Inference** - Split model across GPU + CPU
   - Layer-level offloading
   - Support for multiple GPUs
   - Optimized sharding for hybrid execution

2. **Olive Integration** - Advanced optimization
   - Dynamic quantization
   - Layer fusion
   - Custom optimization passes

3. **Batch Conversion** - Queue multiple models
   - Sequential processing
   - Resume failed conversions

4. **Conversation Management** - Save/load chat history
   - Export to markdown
   - Session persistence

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Windows 10/11 (x64) |
| **GPU** | NVIDIA RTX (4090 recommended) |
| **Drivers** | Latest NVIDIA Game Ready or Studio Driver |
| **.NET** | [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) |
| **CUDA** | [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) |
| **Python** | Python 3.10+ (for model conversion) |
| **onnxruntime-genai** | `pip install onnxruntime-genai-cuda` |

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/clpatrick/MelonStudio.git
cd MelonStudio
dotnet restore
dotnet build --configuration Release

# Run
dotnet run --project MelonStudio
```

---

## Navigation

| Icon | View | Description |
|------|------|-------------|
| 💬 | Chat | Conversation interface |
| 📁 | My Models | Local model management |
| 🔍 | Discover | Search HuggingFace |
| 🔄 | Convert | Model conversion tools |
| ⚙️ | Settings | App configuration |

---

## Project Structure

```
MelonStudio/
├── MelonStudio/                    # WPF GUI application
│   ├── ViewModels/                 # MVVM view models
│   │   ├── ChatViewModel.cs        # Chat conversation logic
│   │   ├── ModelManagerViewModel.cs # HuggingFace search/download
│   │   └── ConvertViewModel.cs     # Conversion UI logic
│   ├── Services/                   # Business logic
│   │   ├── LLMService.cs           # ONNX Runtime GenAI wrapper
│   │   ├── HuggingFaceService.cs   # HF API integration
│   │   ├── ModelBuilderService.cs  # Python builder wrapper
│   │   └── LocalModelService.cs    # Local model scanning
│   ├── Models/                     # Data models
│   │   └── LocalModelInfo.cs       # Local model status tracking
│   ├── Converters/                 # WPF value converters
│   ├── MainWindow.xaml             # Main app window
│   ├── ModelManagerControl.xaml    # Discover view
│   └── ConvertControl.xaml         # Convert view
├── MelonStudio.Benchmark/          # Console benchmark tool
├── benchmarks/                     # Benchmark results
└── MelonStudio.sln                 # Solution file
```

---

## Conversion Workflow

### From HuggingFace ID
1. Go to 🔄 Convert
2. Enter model ID (e.g., `microsoft/Phi-4`)
3. Select precision (INT4 recommended) and provider (CUDA)
4. Click "Start Conversion"

### From Local Path
1. Go to 🔄 Convert
2. Select "Local Path" and browse to model folder
3. Configure options and convert

### Re-quantize Existing Model
1. Go to 🔄 Convert
2. Select "Select from My Models"
3. Choose model and change precision
4. Optionally check "Also create CPU variant"

---

## Error Diagnostics

Failed conversions show categorized diagnostics:

| Category | Example | Suggested Action |
|----------|---------|------------------|
| **UpstreamBug** | Assertion error in rotary embedding | Report to onnxruntime-genai |
| **MissingDependency** | ModuleNotFoundError | `pip install <module>` |
| **ModelIncompatible** | Unsupported architecture | Try a different model |
| **AuthenticationError** | 401 Unauthorized | Check HuggingFace token |
| **NetworkError** | Connection failed | Check internet |
| **DiskSpaceError** | No space left | Free disk space |

---

## Supported Model Architectures

ONNX Runtime GenAI supports:
- Phi (1, 2, 3, 4)
- Llama (2, 3, 3.2)
- Mistral / Mixtral
- Qwen (2, 2.5, 3)
- Gemma (1, 2, 3n)
- DeepSeek V2/V3
- SmolLM3
- And more...

See [ONNX Runtime GenAI docs](https://onnxruntime.ai/docs/genai/) for full list.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `DllNotFoundException` | Add CUDA/cuDNN to PATH |
| Model won't load | Point to folder with `genai_config.json` |
| Conversion fails | Check Python output for specific error |
| Build errors | Ensure .NET 8 SDK and x64 platform |

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Ensure CI build passes
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.
