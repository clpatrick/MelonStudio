using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
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

        public ObservableCollection<HuggingFaceModel> SearchResults { get; } = new();

        public string[] PrecisionOptions { get; } = new[] { "fp32", "fp16", "int4" };
        public string[] ProviderOptions { get; } = new[] { "cuda", "dml", "cpu" };

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
                    SaveSettings(); // Save settings after conversion
                });
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
                StatusMessage = "Ready to convert models";
                await LoadRecommendedModelsAsync();
            }
        }

        [RelayCommand]
        private async Task SearchModelsAsync()
        {
            if (string.IsNullOrWhiteSpace(SearchQuery)) return;

            IsSearching = true;
            StatusMessage = "Searching...";
            SearchResults.Clear();

            try
            {
                var results = await _huggingFaceService.SearchModelsAsync(SearchQuery);
                foreach (var model in results)
                {
                    if (_huggingFaceService.IsModelArchitectureSupported(model))
                    {
                        SearchResults.Add(model);
                    }
                }
                StatusMessage = $"Found {SearchResults.Count} compatible models";
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
            StatusMessage = "Loading recommended models...";
            SearchResults.Clear();

            try
            {
                var results = await _huggingFaceService.GetRecommendedModelsAsync();
                foreach (var model in results)
                {
                    SearchResults.Add(model);
                }
                StatusMessage = $"Loaded {SearchResults.Count} recommended models";
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
        private void SelectModel(HuggingFaceModel model)
        {
            SelectedModelId = model.Id;
            StatusMessage = $"Selected: {model.Id}";
        }

        [RelayCommand]
        private void BrowseOutputFolder()
        {
            // WPF folder dialog would go here
            // For now, just use the default
        }
    }
}
