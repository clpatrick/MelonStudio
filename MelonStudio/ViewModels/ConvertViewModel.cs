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

        // Collections
        public ObservableCollection<LocalModelInfo> LocalModels { get; } = new();
        public string[] PrecisionOptions { get; } = new[] { "int4", "fp16", "fp32" };
        public string[] ProviderOptions { get; } = new[] { "cuda", "cpu", "dml" };
        public string[] ConverterOptions { get; } = new[] { "ONNX Runtime GenAI", "Olive (coming soon)" };

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
            _modelBuilderService.OnCompleted += success =>
            {
                App.Current.Dispatcher.Invoke(() =>
                {
                    IsConverting = false;
                    StatusMessage = success ? "✓ Conversion completed!" : "✗ Conversion failed";
                });
            };
            _modelBuilderService.OnDiagnosticGenerated += diagnostic =>
            {
                App.Current.Dispatcher.Invoke(() => LastDiagnostic = diagnostic);
            };
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
            var dialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "Select Model Folder"
            };
            if (dialog.ShowDialog() == true)
            {
                LocalModelPath = dialog.FolderName;
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
            var outputPath = Path.Combine(OutputFolder, $"{modelName}_{SelectedPrecision}_{SelectedProvider}");

            StatusMessage = $"Converting to {outputPath}...";
            ConversionLog = $"Model: {modelSource}\n";
            ConversionLog += $"Output: {outputPath}\n";
            ConversionLog += $"Precision: {SelectedPrecision}, Provider: {SelectedProvider}\n";
            ConversionLog += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

            // Cache directory
            var cacheDir = Path.Combine(OutputFolder, "temp", modelName);

            try
            {
                await _modelBuilderService.ConvertModelAsync(
                    modelSource,
                    outputPath,
                    SelectedPrecision,
                    SelectedProvider,
                    EnableCudaGraph,
                    string.IsNullOrEmpty(HuggingFaceToken) ? null : HuggingFaceToken,
                    cacheDir
                );

                // Create CPU variant if requested
                if (CreateCpuVariant && SelectedProvider != "cpu")
                {
                    var cpuOutputPath = Path.Combine(OutputFolder, $"{modelName}_{SelectedPrecision}_cpu");
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
            return !IsConverting && SelectedSourceType switch
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
            StartConversionCommand.NotifyCanExecuteChanged();
        }

        partial void OnSelectedLocalModelChanged(LocalModelInfo? value)
        {
            StartConversionCommand.NotifyCanExecuteChanged();
        }
    }
}
