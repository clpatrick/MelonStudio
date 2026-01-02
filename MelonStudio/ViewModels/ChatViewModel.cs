using System;
using System.Collections.ObjectModel;
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
        private AppSettings _settings;
        private CancellationTokenSource? _generationCts;

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

        public ObservableCollection<ChatMessage> Messages { get; } = new();

        public ChatViewModel()
        {
            _llmService = new LLMService();
            _settings = AppSettings.Load();
        }

        public void UpdateSettings(AppSettings settings)
        {
            _settings = settings;
            _llmService.UpdateSettings(settings.MaxLength, settings.Temperature, settings.TopP);
        }

        [RelayCommand]
        private async Task LoadModelAsync()
        {
            try
            {
                StatusMessage = "Loading model...";
                IsLoading = true;
                
                await _llmService.InitializeAsync(_settings.LastModelPath);
                
                StatusMessage = "Model loaded via CUDA/TensorRT.";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error: {ex.Message}";
                Messages.Add(new ChatMessage(ChatRole.System, $"Failed to load model: {ex.Message}"));
            }
            finally
            {
                IsLoading = false;
            }
        }

        public async Task LoadModelFromPathAsync(string modelPath)
        {
            try
            {
                var modelName = System.IO.Path.GetFileName(modelPath);
                LoadedModelName = $"Loading {modelName}...";
                StatusMessage = "Loading model...";
                IsLoading = true;
                
                var contextLength = await _llmService.InitializeAsync(modelPath);
                
                LoadedModelName = modelName;
                ModelContextLength = contextLength;
                
                if (contextLength > 0)
                {
                    StatusMessage = $"Ready (Context: {contextLength:N0} tokens)";
                }
                else
                {
                    StatusMessage = "Ready";
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
            
            var assistantMessage = new ChatMessage(ChatRole.Assistant, "", LoadedModelName);
            Messages.Add(assistantMessage);

            IsGenerating = true;
            StatusMessage = "Generating...";

            // Create new cancellation token for this generation
            _generationCts?.Dispose();
            _generationCts = new CancellationTokenSource();

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            int tokenCount = 0;

            try
            {
                await foreach (var token in _llmService.GenerateResponseAsync(
                    userText, 
                    _settings.SystemPrompt, 
                    _generationCts.Token))
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
                if (StatusMessage == "Generating...")
                    StatusMessage = "Ready.";
            }
        }

        [RelayCommand]
        private void StopGeneration()
        {
            _generationCts?.Cancel();
            StatusMessage = "Stopping...";
        }
    }
}
