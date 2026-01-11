using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
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

        // Olive-specific properties
        [ObservableProperty]
        private string _selectedDevice = "gpu";

        [ObservableProperty]
        private string _selectedQuantizationAlgorithm = "awq";

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
        public ObservableCollection<string> PrecisionOptions { get; } = new();
        public ObservableCollection<string> ProviderOptions { get; } = new();
        public ObservableCollection<string> QuantizationAlgorithmOptions { get; } = new();
        public string[] DeviceOptions { get; } = new[] { "cpu", "gpu" };
        public string[] ConverterOptions { get; } = new[] { "ONNX Runtime GenAI", "Olive Optimization Tool" };

        // Callback to request model unload from main window
        public Action? RequestModelUnload { get; set; }

        public ConvertViewModel()
        {
            _settings = AppSettings.Load();
            _modelBuilderService = new ModelBuilderService();
            _localModelService = new LocalModelService(_settings.DefaultOutputFolder);
            _outputFolder = _settings.DefaultOutputFolder;
            _huggingFaceToken = _settings.HuggingFaceToken ?? "";

            // Initialize options
            UpdateConverterOptions();

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
            ConversionLog = ""; // Clear previous log

            try 
            {
                // Run comprehensive analysis with all tools
                var results = await _modelBuilderService.RunComprehensiveAnalysisAsync(pathData);
                
                if (results.Count == 0)
                {
                    StatusMessage = "No analysis tools available";
                    ConversionLog += "ERROR: No analysis tools found.\n";
                    return;
                }
                
                // Find the primary result (basic analysis) for UI updates
                var primaryResult = results.FirstOrDefault(r => 
                    r.ToolName == "SafeTensors Analyzer" || r.ToolName == "ONNX Analyzer");
                
                // Display each tool's results
                foreach (var result in results)
                {
                    ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                    ConversionLog += $"MODEL ANALYSIS: {result.ToolName}\n";
                    ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
                    
                    if (!string.IsNullOrEmpty(result.RawOutput))
                    {
                        ConversionLog += result.RawOutput;
                        ConversionLog += "\n";
                    }
                    
                    // For primary analysis, show parsed results
                    if (result == primaryResult)
                    {
                        ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                        ConversionLog += "PARSED RESULTS\n";
                        ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                        
                        if (result.Success)
                        {
                            ConversionLog += $"✓ Success: Analysis completed successfully\n";
                            ConversionLog += $"  Layers: {result.LayerCount}\n";
                            ConversionLog += $"  Total Size: {result.TotalSizeGb:F2} GB\n";
                            ConversionLog += $"  Base Size: {result.BaseSizeGb:F2} GB\n";
                            ConversionLog += $"  Avg Layer Size: {result.AvgLayerSizeGb:F2} GB\n";
                            ConversionLog += $"  Precision: {result.Precision}\n";
                        }
                        else
                        {
                            ConversionLog += $"✗ Failed: {result.ErrorMessage}\n";
                        }
                        ConversionLog += "\n";
                    }
                    else
                    {
                        // For other tools, just show status
                        if (result.Success)
                        {
                            ConversionLog += "✓ Analysis completed successfully\n\n";
                        }
                        else if (!string.IsNullOrEmpty(result.ErrorMessage))
                        {
                            ConversionLog += $"⚠ {result.ErrorMessage}\n\n";
                        }
                    }
                }
                
                // Update UI with primary result
                if (primaryResult != null && primaryResult.Success)
                {
                    UpdateModelInfo(primaryResult.LayerCount, primaryResult.TotalSizeGb);
                    IsModelAnalyzed = true;
                    AnalyzedPrecision = primaryResult.Precision;
                    StatusMessage = $"Analysis complete: {primaryResult.LayerCount} layers, {primaryResult.Precision}, {primaryResult.TotalSizeGb:F1} GB";
                    
                    // Auto-enable hybrid if layers found (only for ORT GenAI)
                    if (primaryResult.LayerCount > 0 && IsOrtGenAiSelected)
                    {
                        EnableHybridMode = true;
                    }

                    // Auto-select precision (only for ORT GenAI)
                    if (IsOrtGenAiSelected && !string.IsNullOrEmpty(primaryResult.Precision))
                    {
                        var prec = primaryResult.Precision.ToLower();
                        if (prec.Contains("int4") || prec.Contains("int8")) SelectedPrecision = "int4";
                        else if (prec.Contains("fp16") || prec.Contains("bf16")) SelectedPrecision = "fp16";
                        else if (prec.Contains("fp32")) SelectedPrecision = "fp32";
                    }

                    // Update quantization algorithms for Olive
                    if (IsOliveSelected)
                    {
                        UpdateQuantizationAlgorithms();
                    }
                }
                else if (primaryResult != null)
                {
                    IsModelAnalyzed = true; // Still reveal options so they can try standard conversion
                    StatusMessage = $"Analysis incomplete: {primaryResult.ErrorMessage}";
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
            string baseFolderName;
            if (IsOliveSelected)
            {
                // Olive: {modelName}_{algorithm}_{provider} or {modelName}_{algorithm}_cpu
                if (SelectedDevice == "cpu")
                {
                    baseFolderName = $"{modelName}_{SelectedQuantizationAlgorithm}_cpu";
                }
                else
                {
                    baseFolderName = $"{modelName}_{SelectedQuantizationAlgorithm}_{SelectedProvider}";
                }
            }
            else
            {
                // ORT GenAI: {modelName}_{precision}_{provider}
                baseFolderName = $"{modelName}_{SelectedPrecision}_{SelectedProvider}";
            }

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
            if (IsOliveSelected)
            {
                ConversionLog += $"Algorithm: {SelectedQuantizationAlgorithm}, Device: {SelectedDevice}";
                if (SelectedDevice == "gpu")
                    ConversionLog += $", Provider: {SelectedProvider}";
                ConversionLog += "\n";
            }
            else
            {
                ConversionLog += $"Precision: {SelectedPrecision}, Provider: {SelectedProvider}\n";
            }
            ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

            // Cache directory
            var cacheDir = Path.Combine(OutputFolder ?? string.Empty, "temp", modelName);

            try
            {
                bool isOnnxSource = false;
                bool pendingConversion = false;

                if (IsOliveSelected)
                {
                    // Olive workflow: 2-step process (quantize + auto-opt)
                    ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                    ConversionLog += "Step 1/2: Quantizing model...\n";
                    ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
                    StatusMessage = "Step 1/2: Quantizing model...";

                    // Determine provider for auto-opt step
                    string executionProvider = SelectedDevice == "cpu" ? "cpu" : SelectedProvider;

                    // Determine precision for quantization (only for GPTQ and BitsAndBytes)
                    string? quantizationPrecision = null;
                    if (RequiresPrecisionSelection)
                    {
                        quantizationPrecision = SelectedPrecision;
                    }

                    var conversionSuccess = await _modelBuilderService.ConvertModelWithOliveAsync(
                        modelSource,
                        outputPath,
                        SelectedQuantizationAlgorithm,
                        quantizationPrecision,
                        SelectedDevice,
                        executionProvider,
                        true, // useOrtGenai
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

                    // For Olive, we always do conversion, so pendingConversion is true
                    pendingConversion = true;
                }
                else
                {
                    // ORT GenAI workflow: single-step conversion
                    // Check if source is already an ONNX model
                    isOnnxSource = modelSource.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);
                    
                    // Determine if we need to run conversion (quantization)
                    // Run if:
                    // 1. Source is NOT ONNX (need to build from safetensors/bin)
                    // 2. Source IS ONNX, but requested precision differs from analyzed precision (need to re-quantize)
                    bool precisionMatches = isOnnxSource && !string.IsNullOrEmpty(AnalyzedPrecision) && 
                                          AnalyzedPrecision.IndexOf(SelectedPrecision, StringComparison.OrdinalIgnoreCase) >= 0;
                    
                    pendingConversion = !isOnnxSource || (isOnnxSource && !precisionMatches);

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
                }

                // Create hybrid partitions if enabled (only for ORT GenAI)
                if (EnableHybridMode && IsOrtGenAiSelected)
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

        partial void OnSelectedConverterChanged(string value)
        {
            UpdateConverterOptions();
            if (IsOliveSelected)
            {
                UpdateQuantizationAlgorithms();
            }
            // Notify computed properties
            OnPropertyChanged(nameof(IsOliveSelected));
            OnPropertyChanged(nameof(IsOrtGenAiSelected));
            OnPropertyChanged(nameof(IsProviderEnabled));
            OnPropertyChanged(nameof(IsHybridModeEnabled));
            OnPropertyChanged(nameof(ConverterInfoText));
            OnPropertyChanged(nameof(RequiresPrecisionSelection));
        }

        partial void OnSelectedQuantizationAlgorithmChanged(string value)
        {
            UpdatePrecisionOptionsForAlgorithm();
            OnPropertyChanged(nameof(RequiresPrecisionSelection));
            
            // Auto-select first precision if algorithm requires it and no precision is selected
            if (RequiresPrecisionSelection && PrecisionOptions.Count > 0 && !PrecisionOptions.Contains(SelectedPrecision))
            {
                SelectedPrecision = PrecisionOptions[0];
            }
        }

        partial void OnSelectedDeviceChanged(string value)
        {
            if (IsOliveSelected)
            {
                UpdateProviderOptionsForOlive();
            }
            // Notify computed properties
            OnPropertyChanged(nameof(IsProviderEnabled));
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

        // Computed properties
        public bool IsOliveSelected => SelectedConverter == "Olive Optimization Tool";
        public bool IsOrtGenAiSelected => SelectedConverter == "ONNX Runtime GenAI";
        public bool IsProviderEnabled => IsOrtGenAiSelected || (IsOliveSelected && SelectedDevice == "gpu");
        public bool IsHybridModeEnabled => IsOrtGenAiSelected;
        public string ConverterInfoText => IsOrtGenAiSelected 
            ? "ⓘ Uses python -m onnxruntime_genai.models.builder" 
            : "ⓘ Uses olive quantize + auto-opt (2-step process)";

        /// <summary>
        /// Updates converter-specific options when converter selection changes.
        /// </summary>
        private void UpdateConverterOptions()
        {
            if (IsOrtGenAiSelected)
            {
                // ORT GenAI: Show precision options
                PrecisionOptions.Clear();
                foreach (var opt in new[] { "int4", "fp16", "fp32" })
                    PrecisionOptions.Add(opt);
                
                // Provider options for ORT GenAI
                ProviderOptions.Clear();
                foreach (var opt in new[] { "cuda", "cpu", "dml" })
                    ProviderOptions.Add(opt);
            }
            else if (IsOliveSelected)
            {
                // Olive: Hide precision (replaced by quantization algorithm)
                PrecisionOptions.Clear();
                
                // Provider options depend on device (will be updated by UpdateProviderOptionsForOlive)
                UpdateProviderOptionsForOlive();
            }
        }

        /// <summary>
        /// Updates provider options for Olive based on device selection.
        /// </summary>
        private void UpdateProviderOptionsForOlive()
        {
            if (!IsOliveSelected) return;

            ProviderOptions.Clear();
            if (SelectedDevice == "gpu")
            {
                foreach (var opt in new[] { "cuda", "dml", "tensorrt" })
                    ProviderOptions.Add(opt);
                
                // Auto-select first option if current selection is not available
                if (!ProviderOptions.Contains(SelectedProvider) && ProviderOptions.Count > 0)
                    SelectedProvider = ProviderOptions[0];
            }
            // If CPU, provider options remain empty (combobox will be disabled)
        }

        /// <summary>
        /// Updates quantization algorithm options based on model format.
        /// </summary>
        private void UpdateQuantizationAlgorithms()
        {
            if (!IsOliveSelected) return;

            QuantizationAlgorithmOptions.Clear();

            // Detect model format
            string pathData = SelectedSourceType switch
            {
                ModelSourceType.LocalPath => LocalModelPath,
                ModelSourceType.MyModels => SelectedLocalModel?.Path ?? "",
                _ => ""
            };

            bool isOnnxModel = !string.IsNullOrEmpty(pathData) && 
                              (pathData.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ||
                               Directory.Exists(pathData) && Directory.GetFiles(pathData, "*.onnx", SearchOption.TopDirectoryOnly).Length > 0);

            // Check GPU availability
            bool hasGpu = _modelBuilderService.CheckGpuAvailable();

            if (isOnnxModel)
            {
                // ONNX models: BitsAndBytes, NVMO, Olive (4-bit only)
                var onnxAlgorithms = new List<string> { "bitsandbytes", "nvmo", "olive" };
                if (hasGpu)
                {
                    // AWQ and GPTQ also support ONNX
                    onnxAlgorithms.Insert(0, "awq");
                    onnxAlgorithms.Insert(1, "gptq");
                }
                foreach (var alg in onnxAlgorithms)
                    QuantizationAlgorithmOptions.Add(alg);
            }
            else
            {
                // PyTorch/safetensors models: Quarot, Spinquant
                var pytorchAlgorithms = new List<string> { "quarot", "spinquant" };
                if (hasGpu)
                {
                    // AWQ and GPTQ support PyTorch
                    pytorchAlgorithms.Insert(0, "awq");
                    pytorchAlgorithms.Insert(1, "gptq");
                }
                foreach (var alg in pytorchAlgorithms)
                    QuantizationAlgorithmOptions.Add(alg);
            }

            // Auto-select first option if current selection is not available
            if (!QuantizationAlgorithmOptions.Contains(SelectedQuantizationAlgorithm) && QuantizationAlgorithmOptions.Count > 0)
                SelectedQuantizationAlgorithm = QuantizationAlgorithmOptions[0];
            
            // Update precision options based on selected algorithm
            UpdatePrecisionOptionsForAlgorithm();
        }

        /// <summary>
        /// Updates precision options based on the selected quantization algorithm.
        /// Only shows precision options for algorithms that support multiple precisions.
        /// </summary>
        private void UpdatePrecisionOptionsForAlgorithm()
        {
            if (!IsOliveSelected) return;

            PrecisionOptions.Clear();

            var algorithm = SelectedQuantizationAlgorithm?.ToLowerInvariant() ?? "";
            
            if (algorithm == "gptq")
            {
                // GPTQ supports 8, 4, 3, or 2 bits
                foreach (var prec in new[] { "int8", "int4", "int3", "int2" })
                    PrecisionOptions.Add(prec);
            }
            else if (algorithm == "bitsandbytes")
            {
                // BitsAndBytes supports 2, 3, 4, 5, 6, 7 bits
                foreach (var prec in new[] { "int7", "int6", "int5", "int4", "int3", "int2" })
                    PrecisionOptions.Add(prec);
            }
            // For other algorithms (AWQ, Quarot, Olive, Spinquant, NVMO), precision is fixed
            // No precision options shown - precision will be set automatically
        }

        /// <summary>
        /// Returns true if the selected algorithm requires precision selection.
        /// </summary>
        public bool RequiresPrecisionSelection => IsOliveSelected && 
            (SelectedQuantizationAlgorithm?.ToLowerInvariant() == "gptq" || 
             SelectedQuantizationAlgorithm?.ToLowerInvariant() == "bitsandbytes");

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
