using Microsoft.UI.Xaml;
using LocalLLMchat.ViewModels;

namespace LocalLLMchat
{
    public sealed partial class MainWindow : Window
    {
        public ChatViewModel ViewModel { get; }

        public MainWindow()
        {
            this.InitializeComponent();
            ViewModel = new ChatViewModel();
            this.Title = "Local LLM Chat (RTX 4090 Native)";
        }
    }
}
