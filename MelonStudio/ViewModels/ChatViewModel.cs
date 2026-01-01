using System;
using System.Collections.ObjectModel;
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

        [ObservableProperty]
        private string _inputMessage = string.Empty;

        [ObservableProperty]
        private bool _isGenerating;

        [ObservableProperty]
        private string _statusMessage = "Ready to load model.";

        [ObservableProperty]
        private string _loadedModelName = "No model loaded";

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
                StatusMessage = "Loading Model... (this may take a moment)";
                IsGenerating = true;
                
                await _llmService.InitializeAsync(_settingsService.ModelPath);
                
                StatusMessage = "Model Loaded via CUDA/TensorRT.";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error: {ex.Message}";
                Messages.Add(new ChatMessage(ChatRole.System, $"Failed to load model: {ex.Message}"));
            }
            finally
            {
                IsGenerating = false;
            }
        }

        public async Task LoadModelFromPathAsync(string modelPath)
        {
            try
            {
                var modelName = System.IO.Path.GetFileName(modelPath);
                StatusMessage = $"Loading {modelName}...";
                IsGenerating = true;
                
                await _llmService.InitializeAsync(modelPath);
                
                LoadedModelName = modelName;
                StatusMessage = $"Ready";
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error: {ex.Message}";
                Messages.Add(new ChatMessage(ChatRole.System, $"Failed to load model: {ex.Message}"));
            }
            finally
            {
                IsGenerating = false;
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

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            int tokenCount = 0;

            try
            {
                await foreach (var token in _llmService.GenerateResponseAsync(userText, _settings.SystemPrompt))
                {
                    assistantMessage.Content += token;
                    tokenCount++;
                }
                
                stopwatch.Stop();
                assistantMessage.SetPerformanceMetrics(tokenCount, stopwatch.Elapsed.TotalSeconds);
            }
            catch (Exception ex)
            {
                Messages.Add(new ChatMessage(ChatRole.System, $"Error generating: {ex.Message}"));
            }
            finally
            {
                IsGenerating = false;
                StatusMessage = "Ready.";
            }
        }
    }
}
