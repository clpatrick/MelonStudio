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
        private string _inputMessage;

        [ObservableProperty]
        private bool _isGenerating;

        [ObservableProperty]
        private string _statusMessage;

        public ObservableCollection<ChatMessage> Messages { get; } = new ObservableCollection<ChatMessage>();

        public ChatViewModel()
        {
            _llmService = new LLMService();
            _settingsService = new SettingsService();
            StatusMessage = "Ready to load model.";
        }

        [RelayCommand]
        private async Task LoadModel()
        {
            try
            {
                StatusMessage = "Loading Model... (this may take a moment)";
                IsGenerating = true; // Block input while loading
                
                // Use default path from settings
                await _llmService.InitializeAsync(_settingsService.ModelPath);
                
                StatusMessage = "Model Loaded via TensorRT/DirectML.";
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
        private async Task SendMessage()
        {
            if (string.IsNullOrWhiteSpace(InputMessage) || IsGenerating) return;

            var userText = InputMessage;
            InputMessage = string.Empty;

            Messages.Add(new ChatMessage(ChatRole.User, userText));
            
            // Prepare assistant message placeholder
            // In a real binding scenario, we might want a streaming-friendly object.
            // For simplicity, we'll append to a new message object and try to update UI specific way,
            // or just use valid property notification if we made "Content" observable.
            // Since ChatMessage is POCO, let's replace it or update the collection.
            // Better: update the string and re-notify or make ChatMessage observable.
            // Let's assume for this sample we just add the finalized message or stream update if UI supports it.
            // To support streaming in WinUI ListView, the property inside the item must be observable or we re-insert.
            // We'll create a ViewModel wrapper for messages or just hack the observable update.
            // Hack: Replace the last item.
            
            var assistantMessage = new ChatMessage(ChatRole.Assistant, "");
            Messages.Add(assistantMessage);
            int assistantIndex = Messages.Count - 1;

            IsGenerating = true;
            StatusMessage = "Generating...";

            try
            {
                await foreach (var token in _llmService.GenerateResponseAsync(userText, _settingsService.SystemPrompt))
                {
                    assistantMessage.Content += token;
                    // Force refresh. In standard WinUI binding, string updates needing INPC.
                    // Since ChatMessage isn't observable, we replace the item to trigger UI update (inefficient but simple)
                    // messages[i] = newMessage;
                    
                    // Actually, let's just make ChatMessage observable implementation here for correctness
                    // Wait, I already defined ChatMessage as POCO. 
                    // Let's just update the content and trigger a collection changed event or similar? 
                    // Standard workaround:
                    Messages[assistantIndex] = new ChatMessage(ChatRole.Assistant, assistantMessage.Content);
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
