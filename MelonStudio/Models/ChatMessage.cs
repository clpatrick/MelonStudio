using CommunityToolkit.Mvvm.ComponentModel;

namespace MelonStudio.Models
{
    public enum ChatRole
    {
        User,
        Assistant,
        System
    }

    public partial class ChatMessage : ObservableObject
    {
        [ObservableProperty]
        private ChatRole _role;
        
        [ObservableProperty]
        private string _content = string.Empty;

        [ObservableProperty]
        private string _modelName = string.Empty;

        [ObservableProperty]
        private double _tokensPerSecond;

        [ObservableProperty]
        private int _tokenCount;

        [ObservableProperty]
        private double _generationTimeSeconds;

        public string Metadata
        {
            get
            {
                if (Role != ChatRole.Assistant || string.IsNullOrEmpty(ModelName))
                    return string.Empty;
                
                if (TokensPerSecond > 0)
                    return $"🧠 {ModelName} • {TokenCount} tokens • {TokensPerSecond:F1} tok/s • {GenerationTimeSeconds:F1}s";
                
                return $"🧠 {ModelName}";
            }
        }

        public bool HasMetadata => Role == ChatRole.Assistant && !string.IsNullOrEmpty(ModelName);

        public ChatMessage(ChatRole role, string content)
        {
            Role = role;
            Content = content;
        }

        public ChatMessage(ChatRole role, string content, string modelName) : this(role, content)
        {
            ModelName = modelName;
        }

        public void SetPerformanceMetrics(int tokenCount, double generationTimeSeconds)
        {
            TokenCount = tokenCount;
            GenerationTimeSeconds = generationTimeSeconds;
            TokensPerSecond = generationTimeSeconds > 0 ? tokenCount / generationTimeSeconds : 0;
            OnPropertyChanged(nameof(Metadata));
            OnPropertyChanged(nameof(HasMetadata));
        }
    }
}
