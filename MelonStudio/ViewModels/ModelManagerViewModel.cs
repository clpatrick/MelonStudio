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
        private string _outputFolder = @"C:\models";

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

        public string[] PrecisionOptions { get; } = new[] { "fp32", "fp16", "int4" };
        public string[] ProviderOptions { get; } = new[] { "cuda", "dml", "cpu" };
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

        [RelayCommand]
        private async Task SelectModelAsync(HuggingFaceModel model)
        {
            SelectedModelId = model.Id;
            IsLoadingDetails = true;
            
            try
            {
                SelectedModelDetails = await _huggingFaceService.GetModelDetailsAsync(model.Id);
            }
            catch
            {
                SelectedModelDetails = null;
            }
            finally
            {
                IsLoadingDetails = false;
            }
        }

        [RelayCommand]
        private async Task ConvertModelAsync()
        {
            if (string.IsNullOrWhiteSpace(SelectedModelId))
            {
                StatusMessage = "Please select or enter a model";
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
            StatusMessage = "Converting model...";

            var modelName = SelectedModelId.Replace("/", "_").Replace("\\", "_");
            var modelOutputFolder = Path.Combine(OutputFolder, modelName);
            
            // Use temp folder for HuggingFace cache downloads
            var cacheFolder = Path.Combine(OutputFolder, "temp", modelName);
            
            // Ensure cache folder exists
            if (!Directory.Exists(cacheFolder))
            {
                Directory.CreateDirectory(cacheFolder);
            }

            await _modelBuilderService.ConvertModelAsync(
                SelectedModelId,
                modelOutputFolder,
                SelectedPrecision,
                SelectedProvider,
                EnableCudaGraph,
                string.IsNullOrWhiteSpace(HuggingFaceToken) ? null : HuggingFaceToken,
                cacheFolder
            );
        }

        [RelayCommand]
        private void CancelConversion()
        {
            _modelBuilderService.Cancel();
            StatusMessage = "Cancelling...";
        }

        [RelayCommand]
        private void BrowseOutputFolder()
        {
            // WPF folder dialog would go here
            // For now, just use the default
        }

        [RelayCommand]
        private void OpenModelPage()
        {
            if (SelectedModelDetails != null)
            {
                System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
                {
                    FileName = SelectedModelDetails.HuggingFaceUrl,
                    UseShellExecute = true
                });
            }
        }

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
            var modelName = SelectedModelId.Replace("/", "_").Replace("\\", "_");
            var modelOutputFolder = Path.Combine(OutputFolder, modelName);

            StatusMessage = $"Downloading {SelectedModelDetails.DisplayName}...";
            ConversionLog += $"Downloading ONNX model to: {modelOutputFolder}\n";

            try
            {
                // Use huggingface-cli to download the model
                var tokenArg = string.IsNullOrWhiteSpace(HuggingFaceToken) 
                    ? "" 
                    : $"--token {HuggingFaceToken}";

                var process = new System.Diagnostics.Process
                {
                    StartInfo = new System.Diagnostics.ProcessStartInfo
                    {
                        FileName = "huggingface-cli",
                        Arguments = $"download {SelectedModelId} --local-dir \"{modelOutputFolder}\" {tokenArg}",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                process.OutputDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        App.Current.Dispatcher.Invoke(() => ConversionLog += e.Data + "\n");
                };
                process.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        App.Current.Dispatcher.Invoke(() => ConversionLog += e.Data + "\n");
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                await Task.Run(() => process.WaitForExit());

                if (process.ExitCode == 0)
                {
                    StatusMessage = $"✓ Downloaded to {modelOutputFolder}";
                    ConversionLog += "\n✓ Download complete!\n";
                }
                else
                {
                    StatusMessage = "✗ Download failed";
                    ConversionLog += "\n✗ Download failed\n";
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Download error: {ex.Message}";
                ConversionLog += $"\nError: {ex.Message}\n";
                ConversionLog += "Make sure huggingface-cli is installed: pip install huggingface_hub\n";
            }
            finally
            {
                IsConverting = false;
            }
        }

        // Local model conversion support
        [ObservableProperty]
        private string? _localModelPath;

        [ObservableProperty]
        private bool _isLocalConversion;

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
    }
}
