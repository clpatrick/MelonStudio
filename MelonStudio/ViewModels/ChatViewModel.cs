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
        private readonly SettingsService _settingsService;

        [ObservableProperty]
        private string _inputMessage = string.Empty;

        [ObservableProperty]
        private bool _isGenerating;

        [ObservableProperty]
        private string _statusMessage = "Ready to load model.";

        public ObservableCollection<ChatMessage> Messages { get; } = new();

        public ChatViewModel()
        {
            _llmService = new LLMService();
            _settingsService = new SettingsService();
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

        [RelayCommand]
        private async Task SendMessageAsync()
        {
            if (string.IsNullOrWhiteSpace(InputMessage) || IsGenerating) return;

            var userText = InputMessage;
            InputMessage = string.Empty;

            Messages.Add(new ChatMessage(ChatRole.User, userText));
            
            var assistantMessage = new ChatMessage(ChatRole.Assistant, "");
            Messages.Add(assistantMessage);

            IsGenerating = true;
            StatusMessage = "Generating...";

            try
            {
                await foreach (var token in _llmService.GenerateResponseAsync(userText, _settingsService.SystemPrompt))
                {
                    // ChatMessage is now observable, so Content updates will notify UI
                    assistantMessage.Content += token;
                }
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
