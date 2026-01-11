using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MelonStudio.Services;

namespace MelonStudio.ViewModels
{
    public partial class ModelManagerViewModel : ObservableObject
    {
        private readonly HuggingFaceService _huggingFaceService;
        private readonly ModelBuilderService _modelBuilderService;
        private readonly AppSettings _settings;

        [ObservableProperty]
        private string _searchQuery = "";

        [ObservableProperty]
        private string _selectedModelId = "";

        [ObservableProperty]
        private string _outputFolder = @"C:\Repos\MelonStudio\models";

        [ObservableProperty]
        private string _selectedPrecision = "int4";

        [ObservableProperty]
        private string _selectedProvider = "cuda";

        [ObservableProperty]
        private bool _enableCudaGraph = false;

        [ObservableProperty]
        private string _huggingFaceToken = "";

        [ObservableProperty]
        private bool _isSearching;

        [ObservableProperty]
        private bool _isConverting;

        [ObservableProperty]
        private string _statusMessage = "Ready";

        [ObservableProperty]
        private double _downloadProgress;

        [ObservableProperty]
        private bool _isDownloadProgressIndeterminate = true;

        [ObservableProperty]
        private long _totalDownloadBytes;

        [ObservableProperty]
        private string _conversionLog = "";

        [ObservableProperty]
        private bool _pythonAvailable;

        [ObservableProperty]
        private bool _onnxGenAiAvailable;

        // Filter properties
        [ObservableProperty]
        private bool _filterOnnx = true;

        [ObservableProperty]
        private bool _filterCuda = false;

        [ObservableProperty]
        private bool _filterInt4 = false;

        [ObservableProperty]
        private bool _filterFp16 = false;

        [ObservableProperty]
        private bool _filterSourceModels = false;

        // Sort property
        [ObservableProperty]
        private string _selectedSort = "downloads";

        partial void OnSelectedSortChanged(string value)
        {
            if (SearchResults.Count > 0)
            {
                _ = SearchModelsAsync();
            }
        }

        // Model details
        [ObservableProperty]
        private HuggingFaceModelDetails? _selectedModelDetails;

        [ObservableProperty]
        private bool _isLoadingDetails;

        [ObservableProperty]
        private int _resultCount;

        public ObservableCollection<HuggingFaceModel> SearchResults { get; } = new();

        [ObservableProperty]
        private ObservableCollection<string> _precisionOptions = new(new[] { "fp32", "fp16", "int4" });
        
        [ObservableProperty]
        private ObservableCollection<string> _providerOptions = new(new[] { "cuda", "dml", "cpu" });
        public string[] SortOptions { get; } = new[] { "downloads", "likes", "lastModified" };
        public Dictionary<string, string> SortDisplayNames { get; } = new()
        {
            { "downloads", "Most Downloads" },
            { "likes", "Most Likes" },
            { "lastModified", "Recently Updated" }
        };

        public ModelManagerViewModel()
        {
            _huggingFaceService = new HuggingFaceService();
            _modelBuilderService = new ModelBuilderService();
            
            // Load saved settings
            _settings = AppSettings.Load();
            _huggingFaceToken = _settings.HuggingFaceToken;
            _outputFolder = _settings.DefaultOutputFolder;
            _selectedPrecision = _settings.DefaultPrecision;
            _selectedProvider = _settings.DefaultProvider;
            _enableCudaGraph = _settings.EnableCudaGraph;

            _modelBuilderService.OnOutputReceived += line => 
                App.Current.Dispatcher.Invoke(() => ConversionLog += line + "\n");
            _modelBuilderService.OnErrorReceived += line => 
                App.Current.Dispatcher.Invoke(() => ConversionLog += "[ERROR] " + line + "\n");
            _modelBuilderService.OnCompleted += success =>
                App.Current.Dispatcher.Invoke(() =>
                {
                    IsConverting = false;
                    StatusMessage = success ? "Conversion completed!" : "Conversion failed";
                    SaveConversionLog(success);
                    SaveSettings(); // Save settings after conversion
                });
        }

        /// <summary>
        /// Saves the conversion log to a file in the logs folder.
        /// </summary>
        private void SaveConversionLog(bool success)
        {
            try
            {
                var logsFolder = Path.Combine(OutputFolder, "logs");
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

        public void SaveSettings()
        {
            _settings.HuggingFaceToken = HuggingFaceToken;
            _settings.DefaultOutputFolder = OutputFolder;
            _settings.DefaultPrecision = SelectedPrecision;
            _settings.DefaultProvider = SelectedProvider;
            _settings.EnableCudaGraph = EnableCudaGraph;
            _settings.Save();
        }

        public async Task InitializeAsync()
        {
            PythonAvailable = await _modelBuilderService.CheckPythonAvailableAsync();
            if (PythonAvailable)
            {
                OnnxGenAiAvailable = await _modelBuilderService.CheckOnnxRuntimeGenAIAvailableAsync();
            }

            if (!PythonAvailable)
            {
                StatusMessage = "Python not found. Please install Python 3.10+";
            }
            else if (!OnnxGenAiAvailable)
            {
                StatusMessage = "Run: pip install onnxruntime-genai";
            }
            else
            {
                StatusMessage = "Ready to search and convert models";
                await LoadRecommendedModelsAsync();
            }
        }

        [RelayCommand]
        private async Task SearchModelsAsync()
        {
            IsSearching = true;
            StatusMessage = "Searching...";
            SearchResults.Clear();
            SelectedModelDetails = null;

            try
            {
                var results = await _huggingFaceService.SearchModelsAsync(
                    SearchQuery,
                    SelectedSort,
                    FilterOnnx,
                    FilterCuda,
                    FilterInt4,
                    FilterFp16,
                    FilterSourceModels,
                    limit: 50);

                foreach (var model in results)
                {
                    SearchResults.Add(model);
                }
                
                ResultCount = SearchResults.Count;
                StatusMessage = $"Found {ResultCount} models";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Search failed: {ex.Message}";
            }
            finally
            {
                IsSearching = false;
            }
        }

        [RelayCommand]
        private async Task LoadRecommendedModelsAsync()
        {
            IsSearching = true;
            StatusMessage = "Loading popular models...";
            SearchResults.Clear();
            SelectedModelDetails = null;

            try
            {
                var results = await _huggingFaceService.GetRecommendedModelsAsync();
                foreach (var model in results)
                {
                    SearchResults.Add(model);
                }
                ResultCount = SearchResults.Count;
                StatusMessage = $"Loaded {ResultCount} popular models";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Failed to load: {ex.Message}";
            }
            finally
            {
                IsSearching = false;
            }
        }

        [ObservableProperty]
        private ObservableCollection<string> _variants = new();

        [ObservableProperty]
        private string? _selectedVariant;

        [ObservableProperty]
        private bool _hasVariants;

        [RelayCommand]
        private async Task SelectModelAsync(HuggingFaceModel? model)
        {
            if (model == null)
        {
                ClearModelSelection();
                return;
            }

            SelectedModelId = model.Id;
            IsLoadingDetails = true;
            Variants.Clear();
            SelectedVariant = null;
            HasVariants = false;
            ShowConvertButton = false;
            DownloadedModelPath = null;
            
            try
            {
                SelectedModelDetails = await _huggingFaceService.GetModelDetailsAsync(model.Id);
                
                // Update precision and provider options based on model
                UpdatePrecisionAndProviderOptions();
                
                if (SelectedModelDetails != null && SelectedModelDetails.AvailableVariants.Count > 0)
                {
                    foreach (var v in SelectedModelDetails.AvailableVariants)
                    {
                        Variants.Add(v);
                    }
                    
                    HasVariants = true;
                    
                    // Smart default selection
                    var preferred = Variants.FirstOrDefault(v => v.Contains("cuda") && v.Contains("int4")) 
                                 ?? Variants.FirstOrDefault(v => v.Contains("cuda"))
                                 ?? Variants.FirstOrDefault();
                    
                    SelectedVariant = preferred;
                }
            }
            catch
            {
                SelectedModelDetails = null;
                // Reset to defaults if model load fails
                PrecisionOptions.Clear();
                foreach (var opt in new[] { "bf16", "fp32", "fp16", "int4" })
                    PrecisionOptions.Add(opt);
                ProviderOptions.Clear();
                foreach (var opt in new[] { "cuda", "dml", "cpu" })
                    ProviderOptions.Add(opt);
            }
            finally
            {
                IsLoadingDetails = false;
            }
        }

        private void ClearModelSelection()
        {
            SelectedModelDetails = null;
            SelectedModelId = "";
            ShowConvertButton = false;
            DownloadedModelPath = null;
            Variants.Clear();
            SelectedVariant = null;
            HasVariants = false;
            
            // Reset to defaults
            PrecisionOptions.Clear();
            foreach (var opt in new[] { "bf16", "fp32", "fp16", "int4" })
                PrecisionOptions.Add(opt);
            ProviderOptions.Clear();
            foreach (var opt in new[] { "cuda", "dml", "cpu" })
                ProviderOptions.Add(opt);
        }

        private void UpdatePrecisionAndProviderOptions()
        {
            if (SelectedModelDetails == null)
            {
                // Reset to defaults
                PrecisionOptions.Clear();
                foreach (var opt in new[] { "fp32", "fp16", "bf16", "int4" })
                    PrecisionOptions.Add(opt);
                ProviderOptions.Clear();
                foreach (var opt in new[] { "cuda", "dml", "cpu" })
                    ProviderOptions.Add(opt);
                return;
            }

            // Extract precision options from model ID, variants, and file names
            var precisions = new HashSet<string>();
            var providers = new HashSet<string>();

            // Check model ID for precision indicators (e.g., "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
            var modelIdLower = SelectedModelDetails.Id.ToLowerInvariant();
            if (modelIdLower.Contains("bf16") || modelIdLower.Contains("bfloat16")) precisions.Add("bf16");
            if (modelIdLower.Contains("fp16") || modelIdLower.Contains("float16")) precisions.Add("fp16");
            if (modelIdLower.Contains("fp32") || modelIdLower.Contains("float32")) precisions.Add("fp32");
            if (modelIdLower.Contains("int4") || modelIdLower.Contains("i4")) precisions.Add("int4");
            if (modelIdLower.Contains("int8") || modelIdLower.Contains("i8")) precisions.Add("int4"); // Map int8 to int4

            // Check variants
            foreach (var variant in SelectedModelDetails.AvailableVariants)
            {
                var lower = variant.ToLowerInvariant();
                if (lower.Contains("bf16") || lower.Contains("bfloat16")) precisions.Add("bf16");
                if (lower.Contains("int4") || lower.Contains("i4")) precisions.Add("int4");
                if (lower.Contains("int8") || lower.Contains("i8")) precisions.Add("int4"); // Map int8 to int4
                if (lower.Contains("fp16") || lower.Contains("f16") || lower.Contains("float16")) precisions.Add("fp16");
                if (lower.Contains("fp32") || lower.Contains("f32") || lower.Contains("float32")) precisions.Add("fp32");
                
                if (lower.Contains("cuda")) providers.Add("cuda");
                if (lower.Contains("dml") || lower.Contains("directml")) providers.Add("dml");
                if (lower.Contains("cpu")) providers.Add("cpu");
            }

            // Check file names
            if (SelectedModelDetails.Siblings != null)
            {
                foreach (var file in SelectedModelDetails.Siblings)
                {
                    var filename = file.Filename?.ToLowerInvariant() ?? "";
                    if (filename.Contains("bf16") || filename.Contains("bfloat16")) precisions.Add("bf16");
                    if (filename.Contains("int4") || filename.Contains("i4")) precisions.Add("int4");
                    if (filename.Contains("int8") || filename.Contains("i8")) precisions.Add("int4");
                    if (filename.Contains("fp16") || filename.Contains("f16") || filename.Contains("float16")) precisions.Add("fp16");
                    if (filename.Contains("fp32") || filename.Contains("f32") || filename.Contains("float32")) precisions.Add("fp32");
                    
                    if (filename.Contains("cuda")) providers.Add("cuda");
                    if (filename.Contains("dml") || filename.Contains("directml")) providers.Add("dml");
                    if (filename.Contains("cpu")) providers.Add("cpu");
                }
            }

            // For source models (safetensors), check if we can infer precision from model card or tags
            // If it's a source model and no precision detected, check tags/description
            if (precisions.Count == 0 && !SelectedModelDetails.IsOnnxModel && SelectedModelDetails.Tags != null)
            {
                var tagsLower = string.Join(" ", SelectedModelDetails.Tags).ToLowerInvariant();
                if (tagsLower.Contains("bf16") || tagsLower.Contains("bfloat16")) precisions.Add("bf16");
                if (tagsLower.Contains("fp16") || tagsLower.Contains("float16")) precisions.Add("fp16");
                if (tagsLower.Contains("fp32") || tagsLower.Contains("float32")) precisions.Add("fp32");
            }

            // Update collections
            PrecisionOptions.Clear();
            if (precisions.Count > 0)
            {
                // Add in preferred order (higher precision first for source models)
                var order = new[] { "bf16", "fp32", "fp16", "int4" };
                foreach (var p in order)
                {
                    if (precisions.Contains(p))
                        PrecisionOptions.Add(p);
                }
            }
            else
            {
                // Default options if none detected (for source models, prefer higher precision)
                foreach (var opt in new[] { "bf16", "fp32", "fp16", "int4" })
                    PrecisionOptions.Add(opt);
            }

            ProviderOptions.Clear();
            if (providers.Count > 0)
            {
                // Add in preferred order
                var order = new[] { "cuda", "dml", "cpu" };
                foreach (var p in order)
                {
                    if (providers.Contains(p))
                        ProviderOptions.Add(p);
                }
            }
            else
            {
                // Default options if none detected
                foreach (var opt in new[] { "cuda", "dml", "cpu" })
                    ProviderOptions.Add(opt);
            }

            // Auto-select first option if current selection is not available
            if (!PrecisionOptions.Contains(SelectedPrecision) && PrecisionOptions.Count > 0)
                SelectedPrecision = PrecisionOptions[0];
            if (!ProviderOptions.Contains(SelectedProvider) && ProviderOptions.Count > 0)
                SelectedProvider = ProviderOptions[0];
        }

        [RelayCommand]
        private async Task DownloadSourceModelAsync()
        {
            if (string.IsNullOrWhiteSpace(SelectedModelId))
            {
                StatusMessage = "Please select or enter a model";
                return;
            }
            
            if (SelectedModelDetails != null && SelectedModelDetails.IsOnnxModel)
            {
                StatusMessage = "This is an ONNX model, use Download button instead";
                return;
            }

            // Set UI state immediately on UI thread
            IsConverting = true;
            ShowConvertButton = false;
            ConversionLog = "";
            StatusMessage = "Preparing download...";

            var modelName = SelectedModelId.Replace("/", "_").Replace("\\", "_");
            var modelOutputFolder = Path.Combine(OutputFolder, modelName);

            try
            {
                // Do file system operations off UI thread
                var folderError = await Task.Run(() =>
                {
            if (!Directory.Exists(OutputFolder))
            {
                try
                {
                    Directory.CreateDirectory(OutputFolder);
                }
                catch (Exception ex)
                {
                            return ex.Message;
                        }
                    }
                    return null;
                }).ConfigureAwait(false);

                if (folderError != null)
                {
                    App.Current.Dispatcher.Invoke(() => 
                        StatusMessage = $"Cannot create output folder: {folderError}");
                    App.Current.Dispatcher.Invoke(() => IsConverting = false);
                    return;
                }

                // Check if files already exist and verify them
                var filesAlreadyValid = await CheckAndVerifyExistingFiles(modelOutputFolder, SelectedModelId);
                if (filesAlreadyValid)
                {
                    // Files exist and are valid - skip download and proceed to success workflow
                    App.Current.Dispatcher.Invoke(() =>
                    {
                        StatusMessage = $"✓ Model already exists and verified: {modelOutputFolder}";
                        ConversionLog += "\n✓ All files verified. Ready for conversion.\n";
                        DownloadedModelPath = modelOutputFolder;
                        ShowConvertButton = true;
                        DownloadProgress = 100;
                        IsDownloadProgressIndeterminate = false;
                        IsConverting = false;
                    });
                    App.Current.Dispatcher.Invoke(() => ModelDownloaded?.Invoke(modelOutputFolder));
                    return;
                }

                // Files don't exist or verification failed - proceed with download
                App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = "Starting download...";
                    ConversionLog += "Downloading model files...\n";
                });

                // Use Olive's download script to download source model (safetensors/pytorch)
                var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "olive", "olive_download.py");
                
                // Locate the .olive-env python
                var pythonPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "olive", ".olive-env", "Scripts", "python.exe");
                if (!File.Exists(pythonPath))
                {
                    pythonPath = "python";
                }

                var tokenArg = string.IsNullOrWhiteSpace(HuggingFaceToken) ? "" : $"--token {HuggingFaceToken}";
                var subfolderArg = (!string.IsNullOrEmpty(SelectedVariant) && SelectedVariant != "Root") 
                    ? $"--subfolder \"{SelectedVariant}\"" 
                    : "";
                
                var arguments = $"\"{scriptPath}\" --model_id {SelectedModelId} --output_dir \"{modelOutputFolder}\" {tokenArg} {subfolderArg}";

                // Update UI on dispatcher thread
                App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = $"Downloading {SelectedModelDetails?.DisplayName ?? SelectedModelId}...";
                    ConversionLog += $"Downloading model to: {modelOutputFolder}\n";
                    ConversionLog += $"Executing: {pythonPath} {arguments}\n";
                    DownloadProgress = 0;
                    IsDownloadProgressIndeterminate = true;
                });

                var process = new System.Diagnostics.Process
                {
                    StartInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = pythonPath,
                        Arguments = arguments,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                process.OutputDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        App.Current.Dispatcher.BeginInvoke(() => ConversionLog += e.Data + "\n");
                    }
                };
                process.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        var line = e.Data;
                        // Parse tqdm progress output from stderr
                        // Format examples:
                        // "Downloading: 45%|████▌     | 2.1G/4.7G [00:30<00:35, 73.2MB/s]"
                        // "100%|████████████████████| 4.7G/4.7G [01:05<00:00, 72.1MB/s]"
                        // "model.safetensors: 67%|██████▋   | 1.2G/1.8G [00:15<00:07, 85.3MB/s]"
                        var progressMatch = System.Text.RegularExpressions.Regex.Match(line, @"(\d+)%");
                        if (progressMatch.Success)
                        {
                            var percentage = int.Parse(progressMatch.Groups[1].Value);
                            App.Current.Dispatcher.BeginInvoke(() =>
                            {
                                DownloadProgress = percentage;
                                IsDownloadProgressIndeterminate = false;
                            });
                        }
                        
                        // Also try to parse bytes downloaded from tqdm format
                        // Pattern: "2.1G/4.7G" or "500M/1.2G" or "1234567/9876543"
                        var bytesMatch = System.Text.RegularExpressions.Regex.Match(line, @"([\d.]+)([GMK]?)\s*/\s*([\d.]+)([GMK]?)");
                        if (bytesMatch.Success && TotalDownloadBytes == 0)
                        {
                            try
                            {
                                var downloadedStr = bytesMatch.Groups[1].Value;
                                var downloadedUnit = bytesMatch.Groups[2].Value;
                                var totalStr = bytesMatch.Groups[3].Value;
                                var totalUnit = bytesMatch.Groups[4].Value;
                                
                                var downloadedBytes = ParseSize(downloadedStr, downloadedUnit);
                                var totalBytes = ParseSize(totalStr, totalUnit);
                                
                                if (totalBytes > 0)
            {
                                    App.Current.Dispatcher.BeginInvoke(() =>
                                    {
                                        TotalDownloadBytes = totalBytes;
                                        DownloadProgress = (downloadedBytes * 100.0) / totalBytes;
                                        IsDownloadProgressIndeterminate = false;
                                    });
                                }
                            }
                            catch { /* Ignore parse errors */ }
                        }
                        
                        App.Current.Dispatcher.BeginInvoke(() => ConversionLog += "[STDERR] " + line + "\n");
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await Task.Run(() => process.WaitForExit()).ConfigureAwait(false);

                if (process.ExitCode == 0)
                {
                    // Verify the download
                    App.Current.Dispatcher.Invoke(() =>
                    {
                        StatusMessage = "Verifying download...";
                        ConversionLog += "\nVerifying downloaded files...\n";
                    });
                    
                    var verifyScriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "verify_download.py");
                    var verifyTokenArg = string.IsNullOrWhiteSpace(HuggingFaceToken) ? "" : $"--token {HuggingFaceToken}";
                    var verifyArguments = $"\"{verifyScriptPath}\" {SelectedModelId} \"{modelOutputFolder}\" {verifyTokenArg}";

                    var verifyProcess = new System.Diagnostics.Process
                    {
                        StartInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = pythonPath,
                            Arguments = verifyArguments,
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        }
                    };

                    verifyProcess.OutputDataReceived += (s, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                        {
                            App.Current.Dispatcher.BeginInvoke(() => ConversionLog += e.Data + "\n");
                        }
                    };
                    verifyProcess.ErrorDataReceived += (s, e) =>
                    {
                        if (!string.IsNullOrEmpty(e.Data))
                            App.Current.Dispatcher.BeginInvoke(() => ConversionLog += "[STDERR] " + e.Data + "\n");
                    };

                    verifyProcess.Start();
                    verifyProcess.BeginOutputReadLine();
                    verifyProcess.BeginErrorReadLine();

                    await Task.Run(() => verifyProcess.WaitForExit()).ConfigureAwait(false);

                    if (verifyProcess.ExitCode == 0)
                    {
                        App.Current.Dispatcher.Invoke(() =>
                        {
                            StatusMessage = $"✓ Download verified: {modelOutputFolder}";
                            ConversionLog += "\n✓ Download and verification complete!\n";
                            DownloadedModelPath = modelOutputFolder;
                            ShowConvertButton = true;
                        });
                        
                        // Notify MainWindow to refresh models list (on UI thread)
                        App.Current.Dispatcher.Invoke(() => ModelDownloaded?.Invoke(modelOutputFolder));
                    }
                    else
                    {
                        App.Current.Dispatcher.Invoke(() =>
                        {
                            StatusMessage = "⚠ Download completed but verification failed";
                            ConversionLog += "\n⚠ Verification failed. Model may be incomplete.\n";
                        });
                    }
                }
                else
                {
                    App.Current.Dispatcher.Invoke(() =>
                    {
                        StatusMessage = "✗ Download failed";
                        ConversionLog += "\n✗ Download failed. See log for details.\n";
                    });
                }
            }
            catch (Exception ex)
            {
                App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = $"Download error: {ex.Message}";
                    ConversionLog += $"\nError: {ex.Message}\n";
                });
            }
            finally
            {
                App.Current.Dispatcher.Invoke(() =>
                {
                    IsConverting = false;
                    DownloadProgress = 0;
                    IsDownloadProgressIndeterminate = false;
                    TotalDownloadBytes = 0;
                });
            }
        }

        // ... skipping CancelConversion, BrowseOutputFolder, OpenModelPage ...

        [RelayCommand]
        private async Task DownloadOnnxModelAsync()
        {
            if (SelectedModelDetails == null || string.IsNullOrWhiteSpace(SelectedModelId))
            {
                StatusMessage = "Please select a model first";
                return;
            }

            if (!SelectedModelDetails.IsOnnxModel)
            {
                StatusMessage = "This model needs conversion, use Convert instead";
                return;
            }

            if (!Directory.Exists(OutputFolder))
            {
                try
                {
                    Directory.CreateDirectory(OutputFolder);
                }
                catch (Exception ex)
                {
                    StatusMessage = $"Cannot create output folder: {ex.Message}";
                    return;
                }
            }

            IsConverting = true;
            ConversionLog = "";
            DownloadProgress = 0;
            IsDownloadProgressIndeterminate = true;
            TotalDownloadBytes = 0;
            var modelName = SelectedModelId.Replace("/", "_").Replace("\\", "_");
            var modelOutputFolder = Path.Combine(OutputFolder, modelName);

            StatusMessage = $"Downloading {SelectedModelDetails.DisplayName}...";
            ConversionLog += $"Downloading ONNX model to: {modelOutputFolder}\n";
            
            if (!string.IsNullOrEmpty(SelectedVariant) && SelectedVariant != "Root")
            {
                 StatusMessage += $" (Variant: {SelectedVariant})";
                 ConversionLog += $"Selected Variant: {SelectedVariant}\n";
            }

            try
            {
                // Check if files already exist and verify them
                var filesAlreadyValid = await CheckAndVerifyExistingFiles(modelOutputFolder, SelectedModelId);
                if (filesAlreadyValid)
                {
                    // Files exist and are valid - skip download and proceed to success workflow
                    App.Current.Dispatcher.Invoke(() =>
                    {
                        StatusMessage = $"✓ Model already exists and verified: {modelOutputFolder}";
                        ConversionLog += "\n✓ All files verified. Ready to use.\n";
                        DownloadProgress = 100;
                        IsDownloadProgressIndeterminate = false;
                        IsConverting = false;
                    });
                    return;
                }

                // Files don't exist or verification failed - proceed with download
                App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = "Starting download...";
                    ConversionLog += "Downloading model files...\n";
                });

                // Use Olive's HfModelHandler via our python script
                var finalOutputFolder = modelOutputFolder;
                var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "olive", "olive_download.py");
                
                // Locate the .olive-env python
                var pythonPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "olive", ".olive-env", "Scripts", "python.exe");
                if (!File.Exists(pythonPath))
                {
                    // Fallback to searching in PATH if venv not found (though venv is preferred)
                    pythonPath = "python";
                }

                var tokenArg = string.IsNullOrWhiteSpace(HuggingFaceToken) ? "" : $"--token {HuggingFaceToken}";
                var subfolderArg = (!string.IsNullOrEmpty(SelectedVariant) && SelectedVariant != "Root") 
                    ? $"--subfolder \"{SelectedVariant}\"" 
                    : "";
                
                var arguments = $"\"{scriptPath}\" --model_id {SelectedModelId} --output_dir \"{finalOutputFolder}\" {tokenArg} {subfolderArg}";

                var process = new System.Diagnostics.Process
                {
                    StartInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = pythonPath,
                        Arguments = arguments,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                
                StatusMessage = $"Downloading {SelectedModelDetails.DisplayName} with Olive...";
                ConversionLog += $"Executing: {pythonPath} {arguments}\n";

                process.OutputDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        var line = e.Data;
                        // Parse metadata
                        if (line.StartsWith("METADATA:"))
                        {
                            try
                            {
                                var json = line.Substring(9).Trim();
                                // Simple manual parse to avoid dependency, or assume simple JSON structure
                                // Extract total_bytes and file_count using basic string ops for robustness
                                var totalBytesMatch = System.Text.RegularExpressions.Regex.Match(json, "\"total_bytes\":\\s*(\\d+)");
                                var fileCountMatch = System.Text.RegularExpressions.Regex.Match(json, "\"file_count\":\\s*(\\d+)");
                                
                                if (totalBytesMatch.Success && fileCountMatch.Success)
                                {
                                    long bytes = long.Parse(totalBytesMatch.Groups[1].Value);
                                    int count = int.Parse(fileCountMatch.Groups[1].Value);
                                    
                                    string sizeStr;
                                    if (bytes > 1024 * 1024 * 1024) sizeStr = $"{(bytes / 1024.0 / 1024.0 / 1024.0):F2} GB";
                                    else sizeStr = $"{(bytes / 1024.0 / 1024.0):F1} MB";
                                    
                                    App.Current.Dispatcher.BeginInvoke(() => 
                                    {
                                        StatusMessage = $"Downloading {count} files ({sizeStr})...";
                                        ConversionLog += $"[METADATA] Total size: {sizeStr}, Files: {count}\n";
                                        TotalDownloadBytes = bytes;
                                        IsDownloadProgressIndeterminate = false;
                                    });
                                }
                            }
                            catch { /* Ignore parse errors, just fallback to logging */ }
                        }
                        
                        App.Current.Dispatcher.BeginInvoke(() => ConversionLog += line + "\n");
                    }
                };
                process.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        var line = e.Data;
                        // Parse tqdm progress output from stderr
                        // Format examples:
                        // "Downloading: 45%|████▌     | 2.1G/4.7G [00:30<00:35, 73.2MB/s]"
                        // "100%|████████████████████| 4.7G/4.7G [01:05<00:00, 72.1MB/s]"
                        // "model.safetensors: 67%|██████▋   | 1.2G/1.8G [00:15<00:07, 85.3MB/s]"
                        var progressMatch = System.Text.RegularExpressions.Regex.Match(line, @"(\d+)%");
                        if (progressMatch.Success)
                        {
                            var percentage = int.Parse(progressMatch.Groups[1].Value);
                            App.Current.Dispatcher.BeginInvoke(() =>
                            {
                                DownloadProgress = percentage;
                                IsDownloadProgressIndeterminate = false;
                            });
                        }
                        
                        // Also try to parse bytes downloaded from tqdm format
                        // Pattern: "2.1G/4.7G" or "500M/1.2G" or "1234567/9876543"
                        var bytesMatch = System.Text.RegularExpressions.Regex.Match(line, @"([\d.]+)([GMK]?)\s*/\s*([\d.]+)([GMK]?)");
                        if (bytesMatch.Success && TotalDownloadBytes == 0)
                        {
                            try
                            {
                                var downloadedStr = bytesMatch.Groups[1].Value;
                                var downloadedUnit = bytesMatch.Groups[2].Value;
                                var totalStr = bytesMatch.Groups[3].Value;
                                var totalUnit = bytesMatch.Groups[4].Value;
                                
                                var downloadedBytes = ParseSize(downloadedStr, downloadedUnit);
                                var totalBytes = ParseSize(totalStr, totalUnit);
                                
                                if (totalBytes > 0)
                                {
                                    App.Current.Dispatcher.BeginInvoke(() =>
                                    {
                                        TotalDownloadBytes = totalBytes;
                                        DownloadProgress = (downloadedBytes * 100.0) / totalBytes;
                                        IsDownloadProgressIndeterminate = false;
                                    });
                                }
                            }
                            catch { /* Ignore parse errors */ }
                        }
                        
                        App.Current.Dispatcher.BeginInvoke(() => ConversionLog += "[STDERR] " + line + "\n");
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await Task.Run(() => process.WaitForExit()).ConfigureAwait(false);

                if (process.ExitCode == 0)
                {
                    App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = $"✓ Downloaded to {modelOutputFolder}";
                    ConversionLog += "\n✓ Download complete!\n";
                        DownloadProgress = 100; // Set to 100 on success
                    });
                }
                else
                {
                    App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = "✗ Download failed";
                    ConversionLog += "\n✗ Download failed. See log for details.\n";
                    });
                }
            }
            catch (Exception ex)
            {
                App.Current.Dispatcher.Invoke(() =>
            {
                StatusMessage = $"Download error: {ex.Message}";
                ConversionLog += $"\nError: {ex.Message}\n";
                });
            }
            finally
            {
                App.Current.Dispatcher.Invoke(() =>
            {
                IsConverting = false;
                    DownloadProgress = 0;
                    IsDownloadProgressIndeterminate = true;
                });
            }
        }

        // Local model conversion support
        [ObservableProperty]
        private string? _localModelPath;

        [ObservableProperty]
        private bool _isLocalConversion;

        [ObservableProperty]
        private bool _showConvertButton;

        [ObservableProperty]
        private string? _downloadedModelPath;

        /// <summary>
        /// Event raised when a model is successfully downloaded and verified.
        /// MainWindow can subscribe to this to refresh the models list.
        /// </summary>
        public event Action<string>? ModelDownloaded;

        /// <summary>
        /// Sets up the view for converting a local model that was previously downloaded.
        /// </summary>
        public void SetLocalModelForConversion(string modelPath, string modelName)
        {
            LocalModelPath = modelPath;
            IsLocalConversion = true;
            SelectedModelId = modelName;
            StatusMessage = $"Ready to convert local model: {modelName}";
            ConversionLog = $"Local model selected: {modelPath}\n";
            ConversionLog += "Select precision and provider, then click Convert.\n";
        }

        [RelayCommand]
        private void OpenConvertTab()
        {
            if (!string.IsNullOrEmpty(DownloadedModelPath) && Directory.Exists(DownloadedModelPath))
            {
                // Raise event to notify MainWindow to switch to Convert tab
                ModelDownloaded?.Invoke(DownloadedModelPath);
            }
        }

        /// <summary>
        /// Parses size strings like "2.1G", "500M", "1234K" to bytes.
        /// </summary>
        private long ParseSize(string value, string unit)
        {
            if (!double.TryParse(value, out double size))
                return 0;

            return unit.ToUpperInvariant() switch
            {
                "G" => (long)(size * 1024 * 1024 * 1024),
                "M" => (long)(size * 1024 * 1024),
                "K" => (long)(size * 1024),
                _ => (long)size
            };
        }

        /// <summary>
        /// Checks if model files already exist and verifies them.
        /// Returns true if files exist and verification passes, false otherwise.
        /// </summary>
        private async Task<bool> CheckAndVerifyExistingFiles(string modelOutputFolder, string modelId)
        {
            // Check if output folder exists and has files
            if (!Directory.Exists(modelOutputFolder))
                return false;

            var files = Directory.GetFiles(modelOutputFolder, "*", SearchOption.AllDirectories);
            if (files.Length == 0)
                return false;

            // Check for at least one model file (safetensors, bin, onnx, etc.)
            var modelExtensions = new[] { ".safetensors", ".bin", ".onnx", ".pt", ".pth" };
            var hasModelFiles = files.Any(f => modelExtensions.Any(ext => f.EndsWith(ext, StringComparison.OrdinalIgnoreCase)));
            if (!hasModelFiles)
                return false;

            // Run verification script
            App.Current.Dispatcher.Invoke(() =>
            {
                StatusMessage = "Checking existing files...";
                ConversionLog += $"Found existing files in {modelOutputFolder}\n";
                ConversionLog += "Verifying existing files...\n";
            });

            var verifyScriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "verify_download.py");
            var pythonPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "olive", ".olive-env", "Scripts", "python.exe");
            if (!File.Exists(pythonPath))
            {
                pythonPath = "python";
            }

            var tokenArg = string.IsNullOrWhiteSpace(HuggingFaceToken) ? "" : $"--token {HuggingFaceToken}";
            var verifyArguments = $"\"{verifyScriptPath}\" \"{modelId}\" \"{modelOutputFolder}\" {tokenArg}";

            var verifyProcess = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = verifyArguments,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            var verificationOutput = new System.Text.StringBuilder();
            var verificationError = new System.Text.StringBuilder();

            verifyProcess.OutputDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    verificationOutput.AppendLine(e.Data);
                    App.Current.Dispatcher.BeginInvoke(() => ConversionLog += e.Data + "\n");
                }
            };

            verifyProcess.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    verificationError.AppendLine(e.Data);
                    App.Current.Dispatcher.BeginInvoke(() => ConversionLog += "[STDERR] " + e.Data + "\n");
                }
            };

            verifyProcess.Start();
            verifyProcess.BeginOutputReadLine();
            verifyProcess.BeginErrorReadLine();

            await Task.Run(() => verifyProcess.WaitForExit()).ConfigureAwait(false);

            // Exit code 0 means verification succeeded
            if (verifyProcess.ExitCode == 0)
            {
                App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = $"✓ Existing files verified: {modelOutputFolder}";
                    ConversionLog += "\n✓ All existing files are valid. Skipping download.\n";
                });
                return true;
            }
            else
            {
                App.Current.Dispatcher.Invoke(() =>
                {
                    StatusMessage = "Existing files failed verification. Will download/update...";
                    ConversionLog += "\n⚠ Verification failed. Proceeding with download...\n";
                });
                return false;
            }
        }
    }
}
