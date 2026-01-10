using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MelonStudio.Models;
using MelonStudio.Services;

namespace MelonStudio.ViewModels
{
    public enum ModelSourceType
    {
        HuggingFace,
        LocalPath,
        MyModels
    }

    public partial class ConvertViewModel : ObservableObject
    {
        private readonly ModelBuilderService _modelBuilderService;
        private readonly LocalModelService _localModelService;
        private readonly AppSettings _settings;

        // Model Source
        [ObservableProperty]
        private ModelSourceType _selectedSourceType = ModelSourceType.HuggingFace;

        [ObservableProperty]
        private string _huggingFaceModelId = "";

        [ObservableProperty]
        private string _localModelPath = "";

        [ObservableProperty]
        private LocalModelInfo? _selectedLocalModel;

        // Converter
        [ObservableProperty]
        private string _selectedConverter = "ONNX Runtime GenAI";

        // Target Configuration
        [ObservableProperty]
        private string _selectedPrecision = "int4";

        [ObservableProperty]
        private string _selectedProvider = "cuda";

        [ObservableProperty]
        private bool _createCpuVariant = false;

        // Advanced Options
        [ObservableProperty]
        private string _outputFolder = "";

        [ObservableProperty]
        private bool _enableCudaGraph = false;

        [ObservableProperty]
        private string _huggingFaceToken = "";

        // State
        [ObservableProperty]
        private bool _isConverting;

        [ObservableProperty]
        private string _statusMessage = "Ready to convert";

        [ObservableProperty]
        private string _conversionLog = "";

        [ObservableProperty]
        private ConversionDiagnostic? _lastDiagnostic;

        // Analysis State
        [ObservableProperty]
        [NotifyCanExecuteChangedFor(nameof(StartConversionCommand))]
        private bool _isModelAnalyzed;

        [ObservableProperty]
        private bool _isAnalyzing;

        // Hybrid Mode
        [ObservableProperty]
        private bool _enableHybridMode = false;

        [ObservableProperty]
        private int _gpuLayerCount = 16;

        [ObservableProperty]
        private int _maxGpuLayers = 32;

        [ObservableProperty]
        private double _estimatedVramGb = 4.0;

        [ObservableProperty]
        private double _estimatedCpuGb = 4.0;

        [ObservableProperty]
        private double _totalModelSizeGb = 8.0;

        // Verification Results
        [ObservableProperty]
        private ModelAnalysisResult? _gpuVerificationResult;

        [ObservableProperty]
        private ModelAnalysisResult? _cpuVerificationResult;

        [ObservableProperty]
        private bool _hasVerificationResults;

        private string? _lastHybridOutputPath;

        // Collections
        public ObservableCollection<LocalModelInfo> LocalModels { get; } = new();
        public string[] PrecisionOptions { get; } = new[] { "int4", "fp16", "fp32" };
        public string[] ProviderOptions { get; } = new[] { "cuda", "cpu", "dml" };
        public string[] ConverterOptions { get; } = new[] { "ONNX Runtime GenAI", "Olive (coming soon)" };

        // Callback to request model unload from main window
        public Action? RequestModelUnload { get; set; }

        public ConvertViewModel()
        {
            _settings = AppSettings.Load();
            _modelBuilderService = new ModelBuilderService();
            _localModelService = new LocalModelService(_settings.DefaultOutputFolder);
            _outputFolder = _settings.DefaultOutputFolder;
            _huggingFaceToken = _settings.HuggingFaceToken ?? "";

            // Wire up events
            _modelBuilderService.OnOutputReceived += line => 
            {
                App.Current.Dispatcher.Invoke(() => ConversionLog += line + "\n");
            };
            _modelBuilderService.OnErrorReceived += line => 
            {
                App.Current.Dispatcher.Invoke(() => ConversionLog += line + "\n");
            };
            _modelBuilderService.OnCompleted += async success =>
            {
                await App.Current.Dispatcher.InvokeAsync(async () =>
                {
                    StatusMessage = success ? "✓ Conversion completed!" : "✗ Conversion failed";
                    
                    if (success && EnableHybridMode && !string.IsNullOrEmpty(_lastHybridOutputPath))
                    {
                         StatusMessage = "Verifying split models...";
                         try
                         {
                             var (gpu, cpu) = await _modelBuilderService.VerifySplitModelsAsync(_lastHybridOutputPath);
                             GpuVerificationResult = gpu;
                             CpuVerificationResult = cpu;
                             StatusMessage = "✓ Verified!";
                         }
                         catch (Exception ex)
                         {
                             StatusMessage = "⚠ Verification failed";
                             ConversionLog += $"\nVerification Error: {ex.Message}\n";
                         }
                    }

                    IsConverting = false;
                    
                    // Save conversion log to file
                    SaveConversionLog(success);
                });
            };
            _modelBuilderService.OnDiagnosticGenerated += diagnostic =>
            {
                App.Current.Dispatcher.Invoke(() => LastDiagnostic = diagnostic);
            };
        }

        /// <summary>
        /// Saves the conversion log to a file in the logs folder.
        /// </summary>
        private void SaveConversionLog(bool success)
        {
            try
            {
                // Hardcoded logs folder as requested
                var logsFolder = @"C:\Repos\MelonStudio\logs";
                if (!Directory.Exists(logsFolder))
                {
                    Directory.CreateDirectory(logsFolder);
                }

                var timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
                var status = success ? "success" : "failed";
                var logFileName = $"conversion_{timestamp}_{status}.log";
                var logPath = Path.Combine(logsFolder, logFileName);

                File.WriteAllText(logPath, ConversionLog);
                ConversionLog += $"\n📄 Log saved to: {logPath}\n";
            }
            catch (Exception ex)
            {
                ConversionLog += $"\n⚠ Failed to save log: {ex.Message}\n";
            }
        }

        public async Task LoadLocalModelsAsync()
        {
            LocalModels.Clear();
            var models = await _localModelService.ScanModelsFolderAsync();
            foreach (var model in models)
            {
                LocalModels.Add(model);
            }
        }



        [RelayCommand]
        private async Task AnalyzeModelAsync()
        {
            if (IsAnalyzing) return;

            string pathData = SelectedSourceType switch
            {
                ModelSourceType.LocalPath => LocalModelPath,
                ModelSourceType.MyModels => SelectedLocalModel?.Path ?? "",
                _ => ""
            };

            if (string.IsNullOrEmpty(pathData))
            {
                StatusMessage = "Please select a model first.";
                return;
            }

            IsAnalyzing = true;
            StatusMessage = "Analyzing model structure...";


            try 
            {
                var result = await _modelBuilderService.InspectModelAsync(pathData);
                
                if (result.Success)
                {
                    UpdateModelInfo(result.LayerCount, result.TotalSizeGb);
                    IsModelAnalyzed = true;
                    AnalyzedPrecision = result.Precision;
                    StatusMessage = $"Analysis complete: {result.LayerCount} layers, {result.Precision}, {result.TotalSizeGb:F1} GB";
                    
                    // Auto-enable hybrid if layers found, but respect user choice if they change it back?
                    // For now, just suggest it.
                    if (result.LayerCount > 0)
                    {
                        EnableHybridMode = true;
                    }

                    // Auto-select precision
                    if (!string.IsNullOrEmpty(result.Precision))
                    {
                        var prec = result.Precision.ToLower();
                        if (prec.Contains("int4") || prec.Contains("int8")) SelectedPrecision = "int4";
                        else if (prec.Contains("fp16")) SelectedPrecision = "fp16";
                        else if (prec.Contains("fp32")) SelectedPrecision = "fp32";
                    }
                }
                else
                {
                    // If analysis failed, show the specific error
                    IsModelAnalyzed = true; // Still reveal options so they can try standard conversion
                    StatusMessage = $"Analysis incomplete: {result.ErrorMessage}";
                }
            }
            finally
            {
                IsAnalyzing = false;
                StartConversionCommand.NotifyCanExecuteChanged();
            }
        }

        public async Task LoadTempModelsAsync()
        {
            // Also scan temp folder for downloaded but unconverted models
            var tempFolder = Path.Combine(_settings.DefaultOutputFolder, "temp");
            if (Directory.Exists(tempFolder))
            {
                var tempService = new LocalModelService(tempFolder);
                var tempModels = await tempService.ScanModelsFolderAsync();
                foreach (var model in tempModels)
                {
                    model.Name = $"(temp) {model.Name}";
                    LocalModels.Add(model);
                }
            }
        }

        [RelayCommand]
        private void BrowseLocalPath()
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "Select Model File (select any file to pick folder, or .onnx for direct use)",
                Filter = "Model Files|*.onnx;*.safetensors;*.bin;*.gguf;*.json|All Files|*.*",
                CheckFileExists = true,
                CheckPathExists = true,
                InitialDirectory = !string.IsNullOrEmpty(_settings.LastConversionBrowseFolder) && Directory.Exists(_settings.LastConversionBrowseFolder) 
                    ? _settings.LastConversionBrowseFolder 
                    : _settings.DefaultOutputFolder
            };

            if (dialog.ShowDialog() == true)
            {
                var filePath = dialog.FileName;
                var directory = Path.GetDirectoryName(filePath);
                
                // Save the directory for next time
                if (!string.IsNullOrEmpty(directory))
                {
                    _settings.LastConversionBrowseFolder = directory;
                    _settings.Save();
                }

                // Logic:
                // If it's an ONNX file, we might want to use the FILE path (for splitting or direct conversion if supported).
                // If it's another file (safetensors, bin, json), the user implies "This Folder".
                
                if (Path.GetExtension(filePath).Equals(".onnx", StringComparison.OrdinalIgnoreCase))
                {
                    // For ONNX, use the specific file
                    LocalModelPath = filePath;
                }
                else
                {
                    // For others, point to the directory
                    LocalModelPath = directory ?? filePath;
                }
                
                SelectedSourceType = ModelSourceType.LocalPath;
            }
        }

        [RelayCommand]
        private void BrowseOutputFolder()
        {
            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "Select Output Folder"
            };
            if (dialog.ShowDialog() == true)
            {
                OutputFolder = dialog.FolderName;
            }
        }

        [RelayCommand(CanExecute = nameof(CanStartConversion))]
        private async Task StartConversionAsync()
        {
            if (IsConverting) return;

            // Request unload of any active chat models to avoid file locks
            RequestModelUnload?.Invoke();

            IsConverting = true;
            ConversionLog = "";
            LastDiagnostic = null;

            // Determine model source
            string modelSource = SelectedSourceType switch
            {
                ModelSourceType.HuggingFace => HuggingFaceModelId,
                ModelSourceType.LocalPath => LocalModelPath,
                ModelSourceType.MyModels => SelectedLocalModel?.Path ?? "",
                _ => ""
            };

            if (string.IsNullOrWhiteSpace(modelSource))
            {
                StatusMessage = "Please select a model source";
                IsConverting = false;
                return;
            }

            // Generate output folder name
            var modelName = Path.GetFileName(modelSource.TrimEnd('/', '\\'));
            if (modelSource.Contains('/')) // HuggingFace ID
            {
                modelName = modelSource.Replace("/", "_");
            }
            if (modelSource.Contains('/')) // HuggingFace ID
            {
                modelName = modelSource.Replace("/", "_");
            }

            // Ensure unique output folder name
            var baseFolderName = $"{modelName}_{SelectedPrecision}_{SelectedProvider}";
            var finalFolderName = baseFolderName;
            var counter = 1;

            while (Directory.Exists(Path.Combine(OutputFolder ?? string.Empty, finalFolderName)))
            {
                counter++;
                finalFolderName = $"{baseFolderName}_{counter}";
            }

            var outputPath = Path.Combine(OutputFolder ?? string.Empty, finalFolderName);

            StatusMessage = $"Converting to {outputPath}...";
            ConversionLog = $"Model: {modelSource}\n";
            ConversionLog += $"Output: {outputPath}\n";
            ConversionLog += $"Precision: {SelectedPrecision}, Provider: {SelectedProvider}\n";
            ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

            // Cache directory
            var cacheDir = Path.Combine(OutputFolder ?? string.Empty, "temp", modelName);

            try
            {
                // Check if source is already an ONNX model
                bool isOnnxSource = modelSource.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);
                
                // Determine if we need to run conversion (quantization)
                // Run if:
                // 1. Source is NOT ONNX (need to build from safetensors/bin)
                // 2. Source IS ONNX, but requested precision differs from analyzed precision (need to re-quantize)
                bool precisionMatches = isOnnxSource && !string.IsNullOrEmpty(AnalyzedPrecision) && 
                                      AnalyzedPrecision.IndexOf(SelectedPrecision, StringComparison.OrdinalIgnoreCase) >= 0;
                
                bool pendingConversion = !isOnnxSource || (isOnnxSource && !precisionMatches);

                if (pendingConversion)
                {
                    // If converting an existing ONNX file, usually we pass the folder to the builder
                    string convertSource = modelSource;
                    if (isOnnxSource && File.Exists(modelSource))
                    {
                         convertSource = Path.GetDirectoryName(modelSource) ?? modelSource;
                    }

                    if (isOnnxSource)
                    {
                         ConversionLog += $"Re-converting/Quantizing ONNX model (Source: {AnalyzedPrecision} -> Target: {SelectedPrecision})...\n";
                    }

                    var conversionSuccess = await _modelBuilderService.ConvertModelAsync(
                        convertSource,
                        outputPath,
                        SelectedPrecision,
                        SelectedProvider,
                        EnableCudaGraph,
                        string.IsNullOrEmpty(HuggingFaceToken) ? null : HuggingFaceToken,
                        cacheDir
                    );

                    if (!conversionSuccess)
                    {
                        StatusMessage = "✗ Conversion failed";
                        IsConverting = false;
                        SaveConversionLog(false);
                        return;
                    }
                }
                else
                {
                    ConversionLog += "Skipping conversion (Source is already ONNX and matches target precision)\n";
                    // For split-only, use the source path as the input for splitting
                    outputPath = Path.GetDirectoryName(modelSource) ?? OutputFolder ?? string.Empty; 
                }

                // Create hybrid partitions if enabled
                if (EnableHybridMode)
                {
                    // Use the source model's parent folder as the base location
                    var sourceParent = isOnnxSource 
                        ? Path.GetDirectoryName(Path.GetDirectoryName(modelSource)) ?? OutputFolder 
                        : Path.GetDirectoryName(modelSource) ?? OutputFolder;
                    sourceParent = sourceParent ?? OutputFolder ?? string.Empty;

                    // Ensure unique hybrid output folder name
                    var baseHybridName = $"{modelName}_hybrid_{GpuLayerCount}gpu";
                    var finalHybridName = baseHybridName;
                    var hybridCounter = 0;

                    while (Directory.Exists(Path.Combine(sourceParent, finalHybridName)))
                    {
                        hybridCounter++;
                        finalHybridName = $"{baseHybridName}_{hybridCounter}";
                    }

                    var hybridOutputPath = Path.Combine(sourceParent, finalHybridName);
                    _lastHybridOutputPath = hybridOutputPath;
                    ConversionLog += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                    ConversionLog += $"Creating hybrid partitions (GPU: {GpuLayerCount} layers)...\n";
                    ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

                    // Determine input for splitting:
                    // If we ran conversion, use the output; otherwise use original source
                    string splitInputPath;
                    if (pendingConversion)
                    {
                        // We converted/quantized, so use the new model
                        splitInputPath = Path.Combine(outputPath, "model.onnx"); 
                    }
                    else
                    {
                        // Skipped conversion, use original source directly
                        splitInputPath = modelSource;
                    }

                    await _modelBuilderService.ExportHybridPartitionsAsync(
                        splitInputPath,
                        hybridOutputPath,
                        GpuLayerCount, // Layer index to split at
                        string.IsNullOrEmpty(HuggingFaceToken) ? null : HuggingFaceToken,
                        cacheDir
                    );

                    // Status will be updated by OnCompleted callback
                }
                // Create CPU variant if requested (only when not in hybrid mode)
                else if (CreateCpuVariant && SelectedProvider != "cpu")
                {
                    var cpuOutputPath = Path.Combine(OutputFolder ?? string.Empty, $"{modelName}_{SelectedPrecision}_cpu");
                    ConversionLog += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                    ConversionLog += "Creating CPU variant...\n";
                    ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

                    await _modelBuilderService.ConvertModelAsync(
                        modelSource,
                        cpuOutputPath,
                        SelectedPrecision,
                        "cpu",
                        false, // No CUDA graph for CPU
                        string.IsNullOrEmpty(HuggingFaceToken) ? null : HuggingFaceToken,
                        cacheDir
                    );
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error: {ex.Message}";
                ConversionLog += $"\nException: {ex.Message}\n";
            }
            finally
            {
                IsConverting = false;
            }
        }

        private bool CanStartConversion()
        {
            return !IsConverting && IsModelAnalyzed && SelectedSourceType switch
            {
                ModelSourceType.HuggingFace => !string.IsNullOrWhiteSpace(HuggingFaceModelId),
                ModelSourceType.LocalPath => !string.IsNullOrWhiteSpace(LocalModelPath),
                ModelSourceType.MyModels => SelectedLocalModel != null,
                _ => false
            };
        }

        [RelayCommand]
        private void CancelConversion()
        {
            _modelBuilderService.Cancel();
            StatusMessage = "Cancelled";
        }

        partial void OnSelectedSourceTypeChanged(ModelSourceType value)
        {
            StartConversionCommand.NotifyCanExecuteChanged();
        }

        partial void OnHuggingFaceModelIdChanged(string value)
        {
            StartConversionCommand.NotifyCanExecuteChanged();
        }

        partial void OnLocalModelPathChanged(string value)
        {
            IsModelAnalyzed = false;
            StartConversionCommand.NotifyCanExecuteChanged();
        }

        // Removed auto-trigger async method
        // private async Task AnalyzeModelAsync(string path) ...

        partial void OnSelectedLocalModelChanged(LocalModelInfo? value)
        {
            StartConversionCommand.NotifyCanExecuteChanged();
        }

        // Hybrid mode property change handlers
        partial void OnEnableHybridModeChanged(bool value)
        {
            if (value)
            {
                // When enabling hybrid mode, update estimates based on current model
                UpdateVramEstimates();
            }
        }

        partial void OnGpuLayerCountChanged(int value)
        {
            UpdateVramEstimates();
        }



        [ObservableProperty]
        private string _analyzedPrecision = "";

        /// <summary>
        /// Updates VRAM estimates based on current GPU/CPU layer split.
        /// This uses a simplified calculation - actual values come from model analysis.
        /// </summary>
        private void UpdateVramEstimates()
        {
            if (MaxGpuLayers <= 0) return;

            // Estimate based on layer distribution
            double gpuRatio = (double)GpuLayerCount / MaxGpuLayers;
            double cpuRatio = 1.0 - gpuRatio;

            EstimatedVramGb = Math.Round(TotalModelSizeGb * gpuRatio, 2);
            EstimatedCpuGb = Math.Round(TotalModelSizeGb * cpuRatio, 2);
        }

        /// <summary>
        /// Updates model info from analysis results.
        /// Called when model is analyzed or selected.
        /// </summary>
        public void UpdateModelInfo(int totalLayers, double totalSizeGb)
        {
            // Hybrid mode requires at least 1 layer on CPU implies max GPU layers = Total - 1
            // Ensure we have at least 1 layer
            MaxGpuLayers = totalLayers > 1 ? totalLayers - 1 : 1;
            TotalModelSizeGb = totalSizeGb;
            
            // Default to half on GPU, clamped to max
            var target = totalLayers / 2;
            if (target > MaxGpuLayers) target = MaxGpuLayers;
            GpuLayerCount = target;
            
            UpdateVramEstimates();
        }
    }
}
