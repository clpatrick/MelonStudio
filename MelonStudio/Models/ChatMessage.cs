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

        public ChatMessage(ChatRole role, string content)
        {
            Role = role;
            Content = content;
        }
    }
}
