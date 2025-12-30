using Microsoft.UI.Xaml;
using MelonStudio.ViewModels;

namespace MelonStudio
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
