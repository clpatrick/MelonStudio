using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MelonStudio.Models;
using MelonStudio.Services;

namespace MelonStudio.ViewModels
{
    public partial class ChatViewModel : ObservableObject
    {
        private readonly LLMService _llmService;
        private readonly HybridLLMService _hybridService;
        private AppSettings _settings;
        private CancellationTokenSource? _generationCts;
        private bool _isUsingHybrid;

        [ObservableProperty]
        private string _inputMessage = string.Empty;

        [ObservableProperty]
        private bool _isGenerating;

        [ObservableProperty]
        private bool _isLoading;

        [ObservableProperty]
        private string _statusMessage = "Ready to load model.";

        [ObservableProperty]
        private string _loadedModelName = "No model loaded";

        [ObservableProperty]
        private int _modelContextLength = 0;

        [ObservableProperty]
        private string _hybridStatus = "";

        [ObservableProperty]
        private bool _isHybridMode = false;

        public ObservableCollection<ChatMessage> Messages { get; } = new();

        public ChatViewModel()
        {
            _llmService = new LLMService();
            _hybridService = new HybridLLMService();
            _settings = AppSettings.Load();

            // Wire up hybrid service events for diagnostics
            _hybridService.OnStatusChanged += status => 
                System.Windows.Application.Current?.Dispatcher.Invoke(() => StatusMessage = status);
            _hybridService.OnDiagnostic += diag => 
                System.Diagnostics.Debug.WriteLine($"[Hybrid] {diag}");
        }

        public void UpdateSettings(AppSettings settings)
        {
            _settings = settings;
            _llmService.UpdateSettings(settings.MaxLength, settings.Temperature, settings.TopP);
            _hybridService.UpdateSettings(settings.MaxLength, (float)settings.Temperature, (float)settings.TopP);
        }

        [RelayCommand]
        private async Task LoadModelAsync()
        {
            await LoadModelFromPathAsync(_settings.LastModelPath);
        }

        public async Task LoadModelFromPathAsync(string modelPath, string? onnxFileName = null)
        {
            try
            {
                var modelName = Path.GetFileName(modelPath);
                LoadedModelName = $"Loading {modelName}...";
                StatusMessage = "Loading model...";
                IsLoading = true;

                // Check if this is a hybrid model
                if (HybridLLMService.IsHybridModelDirectory(modelPath))
                {
                    // Load as hybrid
                    await _hybridService.InitializeHybridAsync(modelPath);
                    
                    _isUsingHybrid = true;
                    IsHybridMode = true;
                    LoadedModelName = $"{modelName} (Hybrid)";
                    HybridStatus = _hybridService.Summary;
                    ModelContextLength = 0; // Not available for hybrid yet
                    
                    StatusMessage = $"Hybrid: {_hybridService.GpuLayers} GPU + {_hybridService.CpuLayers} CPU layers";
                }
                else
                {
                    // Load as standard
                    var contextLength = await _llmService.InitializeAsync(modelPath, onnxFileName);
                    
                    _isUsingHybrid = false;
                    IsHybridMode = false;
                    LoadedModelName = modelName;
                    HybridStatus = "";
                    ModelContextLength = contextLength;
                    
                    StatusMessage = contextLength > 0 
                        ? $"Ready (Context: {contextLength:N0} tokens)"
                        : "Ready";
                }
            }
            catch (Exception ex)
            {
                LoadedModelName = "Load failed";
                StatusMessage = $"Error: {ex.Message}";
                Messages.Add(new ChatMessage(ChatRole.System, $"Failed to load model: {ex.Message}"));
            }
            finally
            {
                IsLoading = false;
            }
        }

        [RelayCommand]
        private async Task SendMessageAsync()
        {
            if (string.IsNullOrWhiteSpace(InputMessage) || IsGenerating) return;

            var userText = InputMessage;
            InputMessage = string.Empty;

            Messages.Add(new ChatMessage(ChatRole.User, userText));
            
            var modelLabel = IsHybridMode 
                ? $"{LoadedModelName} ({HybridStatus})" 
                : LoadedModelName;
            var assistantMessage = new ChatMessage(ChatRole.Assistant, "", modelLabel);
            Messages.Add(assistantMessage);

            IsGenerating = true;
            StatusMessage = IsHybridMode ? "Generating (hybrid)..." : "Generating...";

            // Create new cancellation token for this generation
            _generationCts?.Dispose();
            _generationCts = new CancellationTokenSource();

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            int tokenCount = 0;

            try
            {
                IAsyncEnumerable<string> tokenStream;
                
                if (_isUsingHybrid)
                {
                    tokenStream = _hybridService.GenerateResponseAsync(
                        userText,
                        _settings.SystemPrompt,
                        _generationCts.Token);
                }
                else
                {
                    tokenStream = _llmService.GenerateResponseAsync(
                        userText,
                        _settings.SystemPrompt,
                        _generationCts.Token);
                }

                await foreach (var token in tokenStream)
                {
                    assistantMessage.Content += token;
                    tokenCount++;
                }
                
                stopwatch.Stop();
                assistantMessage.SetPerformanceMetrics(tokenCount, stopwatch.Elapsed.TotalSeconds);
            }
            catch (OperationCanceledException)
            {
                stopwatch.Stop();
                assistantMessage.SetPerformanceMetrics(tokenCount, stopwatch.Elapsed.TotalSeconds);
                StatusMessage = "Generation stopped.";
            }
            catch (Exception ex)
            {
                Messages.Add(new ChatMessage(ChatRole.System, $"Error generating: {ex.Message}"));
            }
            finally
            {
                IsGenerating = false;
                if (StatusMessage.StartsWith("Generating") || StatusMessage.StartsWith("Stopping"))
                    StatusMessage = IsHybridMode ? $"Ready ({HybridStatus})" : "Ready.";
            }
        }

        public void UnloadModel()
        {
            if (_isUsingHybrid)
            {
                _hybridService.Cleanup();
            }
            else
            {
                _llmService.Dispose(); 
                // Re-initialize to be safe for next load
                // _llmService = new LLMService(); // Ideally we'd just Dispose the session inside LLMService
            }

            LoadedModelName = "No model loaded";
            StatusMessage = "Model unloaded.";
            _isUsingHybrid = false;
            IsHybridMode = false;
            ModelContextLength = 0;
            HybridStatus = "";
        }

        [RelayCommand]
        private void StopGeneration()
        {
            _generationCts?.Cancel();
            StatusMessage = "Stopping...";
        }
    }
}
